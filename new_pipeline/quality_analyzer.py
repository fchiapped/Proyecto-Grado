"""proyecto.new_pipeline.quality_analyzer
数据质量评估模块，实现：
1. 数据完整性、一致性、有效范围和预期变异性的检查
2. Drift 检测（支持 decay/golden/seasonal 策略，使用 KS/Mann-Whitney/PSI/Wasserstein 等统计方法）
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# 添加 Analisis 目录到 Python 路径以导入 drift 相关函数
ANALISIS_DIR = Path(__file__).resolve().parent.parent / "Analisis"
if ANALISIS_DIR.exists():
    sys.path.append(str(ANALISIS_DIR))

try:
    import Funciones_Drift as drift_funcs
except ImportError:
    print("警告: 未找到 Funciones_Drift.py，将使用内部实现的 drift 检测函数")
    drift_funcs = None

@dataclass
class DriftConfig:
    """Drift 检测配置"""
    current_window: str = "3D"  # 当前窗口大小
    resample: Optional[str] = None  # 重采样频率，如 "5min"
    resample_agg: str = "mean"  # 重采样聚合方法
    exclude_columns: List[str] = None  # 要排除的列
    num_method: str = "auto"  # 统计方法
    num_threshold: Optional[float] = None  # drift 阈值
    metrics: List[str] = None  # 要使用的指标
    strategies: List[str] = None  # 要使用的策略

    def __post_init__(self):
        """设置默认值"""
        if self.exclude_columns is None:
            self.exclude_columns = []
        if self.metrics is None:
            self.metrics = ["ks", "mannwhitney", "psi", "wasserstein"]
        if self.strategies is None:
            self.strategies = ["decay", "golden", "seasonal"]

class QualityAnalyzer:
    def __init__(self, config):
        # 原有的质量指标配置
        self.config = config.get('quality_metrics', {})
        
        # drift 检测配置
        drift_config = config.get('drift', {})
        self.drift_config = DriftConfig(
            current_window=drift_config.get('current_window', '3D'),
            resample=drift_config.get('resample'),
            resample_agg=drift_config.get('resample_agg', 'mean'),
            exclude_columns=drift_config.get('exclude_columns', []),
            num_method=drift_config.get('num_method', 'auto'),
            num_threshold=drift_config.get('num_threshold'),
            metrics=drift_config.get('metrics'),
            strategies=drift_config.get('strategies'),
        )
        
        self.drift_funcs = drift_funcs
        
    def check_completeness(self, df, date_col):
        """检查数据完整性"""
        if not self.config['completeness']['enabled']:
            return None
            
        threshold = self.config['completeness']['threshold']
        completeness = df.count() / len(df)
        flags = completeness < threshold
        
        return {
            'metric': 'completeness',
            'flags': flags,
            'scores': completeness
        }
        
    def check_valid_range(self, df):
        """检查数据是否在有效范围内"""
        if not self.config['valid_range']['enabled']:
            return None
            
        flags = pd.DataFrame(index=df.index)
        parameters = self.config['valid_range']['parameters']
        
        for param, limits in parameters.items():
            if param in df.columns:
                flags[f'{param}_range'] = ~df[param].between(limits['min'], limits['max'])
                
        return {
            'metric': 'valid_range',
            'flags': flags
        }
        
    def check_variability(self, df, date_col):
        """检查数据变异性"""
        if not self.config['variability']['enabled']:
            return None
            
        window = self.config['variability']['window_size']
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        flags = pd.DataFrame(index=df.index)
        for col in num_cols:
            rolling_std = df[col].rolling(window).std()
            rolling_mean = df[col].rolling(window).mean()
            cv = rolling_std / rolling_mean
            flags[f'{col}_variability'] = cv > cv.quantile(0.95)
            
        return {
            'metric': 'variability',
            'flags': flags
        }

    def _compute_drift_metric(
        self,
        reference: pd.Series,
        current: pd.Series,
        metric: str = "ks"
    ) -> Tuple[float, bool]:
        """计算单个 drift 指标
        
        Args:
            reference: 参考数据
            current: 当前数据
            metric: 使用的指标，可选 "ks"/"mannwhitney"/"psi"/"wasserstein"
            
        Returns:
            (score, is_drift): 得分和是否检测到 drift
        """
        ref = reference.dropna()
        cur = current.dropna()
        
        if len(ref) < 2 or len(cur) < 2:
            return 0.0, False

        if metric == "ks":
            stat, p_value = stats.ks_2samp(ref, cur)
            return p_value, p_value < 0.05
        
        elif metric == "mannwhitney":
            stat, p_value = stats.mannwhitneyu(
                ref, cur, alternative="two-sided", use_continuity=True
            )
            return p_value, p_value < 0.05
        
        elif metric == "psi":
            # Population Stability Index
            bins = np.histogram_bin_edges(
                np.concatenate([ref, cur]), bins="auto"
            )
            ref_hist = np.histogram(ref, bins=bins)[0] / len(ref)
            cur_hist = np.histogram(cur, bins=bins)[0] / len(cur)
            
            # 避免除零
            ref_hist = np.where(ref_hist == 0, 1e-6, ref_hist)
            cur_hist = np.where(cur_hist == 0, 1e-6, cur_hist)
            
            psi = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))
            threshold = 0.2  # 一般认为 PSI > 0.2 表示显著 drift
            return psi, psi > threshold
        
        elif metric == "wasserstein":
            # Wasserstein distance (Earth Mover's Distance)
            distance = stats.wasserstein_distance(ref, cur)
            # 归一化
            scale = np.std(np.concatenate([ref, cur])) + 1e-6
            normed_dist = distance / scale
            threshold = 0.3  # 可配置
            return normed_dist, normed_dist > threshold
            
        else:
            raise ValueError(f"不支持的度量方法: {metric}")

    def detect_drift(
        self,
        data: pd.DataFrame,
        reference_data: Optional[pd.DataFrame] = None,
        date_col: str = "date_time",
    ) -> Dict:
        """检测数据 drift
        
        如果存在外部 drift_funcs 模块，优先使用其实现；
        否则使用内部实现的基本 drift 检测。

        Args:
            data: 要分析的数据
            reference_data: 可选的参考数据，如果未提供则从 data 中划分
            date_col: 时间列名
            
        Returns:
            包含检测结果的字典
        """
        if not self.config.get('drift', {}).get('enabled', True):
            return None

        if self.drift_funcs is not None:
            # 优先使用项目现有的 drift 检测实现
            try:
                output_dir = Path("../output/drift")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                results = self.drift_funcs.run_drift_aggregate(
                    plant_names=["current"],
                    strategies=self.drift_config.strategies,
                    plant_files={"current": data},
                    flag_files={},  # TODO: 支持传入 flags
                    output_root=output_dir,
                    metrics=self.drift_config.metrics,
                    CURRENT_WINDOW=self.drift_config.current_window,
                    RESAMPLE=self.drift_config.resample,
                    RESAMPLE_AGG=self.drift_config.resample_agg,
                    EXCLUDE_COLUMNS=self.drift_config.exclude_columns,
                    NUM_METHOD=self.drift_config.num_method,
                    NUM_THRESHOLD=self.drift_config.num_threshold,
                )
                return {
                    'metric': 'drift',
                    'results': results
                }
            except Exception as e:
                print(f"警告：使用 Funciones_Drift.py 检测失败: {e}")
                print("将使用内部实现的基本 drift 检测...")

        # 回退到基本的 drift 检测实现
        results = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        cols_to_check = [c for c in numeric_cols if c not in self.drift_config.exclude_columns]
        
        if reference_data is None:
            # 如果没有提供参考数据，使用前一个窗口作为参考
            window = pd.Timedelta(self.drift_config.current_window)
            split_time = data[date_col].max() - window
            reference_data = data[data[date_col] <= split_time]
            current_data = data[data[date_col] > split_time]
        else:
            current_data = data

        for col in cols_to_check:
            col_results = {}
            for metric in self.drift_config.metrics:
                score, is_drift = self._compute_drift_metric(
                    reference_data[col], current_data[col], metric
                )
                col_results[metric] = {
                    "score": score,
                    "drift_detected": is_drift
                }
            results[col] = col_results

        return {
            'metric': 'drift',
            'results': {
                "column_results": results,
                "overall_drift": any(
                    any(m["drift_detected"] for m in col.values())
                    for col in results.values()
                )
            }
        }

    def run_all_checks(self, df, date_col):
        """运行所有数据质量检查，包括完整性、有效范围、变异性和 drift 检测"""
        results = {}
        
        results['completeness'] = self.check_completeness(df, date_col)
        results['valid_range'] = self.check_valid_range(df)
        results['variability'] = self.check_variability(df, date_col)
        results['drift'] = self.detect_drift(df, date_col=date_col)
        
        return results