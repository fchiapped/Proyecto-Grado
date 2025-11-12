import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Union, Dict

from data_loader import _find_preproc_file
import importlib.util
import sys

# 导入分析函数
def import_analysis_functions():
    """动态导入分析函数模块"""
    analysis_path = Path(__file__).parent.parent / 'Analisis' / 'funciones_analisis.py'
    if not analysis_path.exists():
        raise FileNotFoundError(f"无法找到分析函数文件: {analysis_path}")
    
    spec = importlib.util.spec_from_file_location("analysis_functions", analysis_path)
    analysis_module = importlib.util.module_from_spec(spec)
    sys.modules["analysis_functions"] = analysis_module
    spec.loader.exec_module(analysis_module)
    return analysis_module

class PipelineManager:
    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        """
        初始化pipeline管理器
        
        Args:
            data_path: 数据文件或目录的路径
        """
        self.data_path = Path(data_path) if data_path else None
        self.df = None
        self.analysis_funcs = import_analysis_functions()
        
    def load_data(self, file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        加载数据文件
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            处理后的DataFrame
        """
        if file_path:
            self.data_path = Path(file_path)
        if not self.data_path or not self.data_path.exists():
            raise FileNotFoundError("请提供有效的数据文件路径")
            
        self.df = pd.read_csv(self.data_path)
        return self.df
    
    def detect_outliers(self, column: str, method: str = 'robust', 
                       threshold: float = 3.5, plot: bool = True,
                       return_details: bool = False):
        """
        检测异常值
        
        算法说明：
        - robust: 稳健Z分数（基于MAD）。计算 robust_z = 0.6745 * |x - median| / MAD，
                  当 robust_z > threshold 判定为异常；与 Analisis/outliers.ipynb 中
                  plot_outliers 使用的度量一致。若列名包含“pH”，额外应用物理范围 [0, 14] 约束。
        - zscore: 标准Z-score方法，计算 z = (x - mean) / std，当 |z| > threshold 时判定为异常
        - iqr: 四分位距方法，使用 Q1-1.5*IQR 和 Q3+1.5*IQR 作为边界
        - rolling: 滑动窗口方法，基于局部统计量检测异常
        
        注意：为与 Analisis/outliers.ipynb 完全一致，默认 method='robust'，并对 pH 列应用
             物理范围校验（<0 或 >14 视为异常）。
        
        Args:
            column: 要分析的列名
            method: 检测方法 ('robust', 'zscore', 'iqr', 'rolling')
            threshold: 阈值（zscore默认3.5，iqr默认1.5）
            plot: 是否绘制图表
            
        Returns:
            异常值的布尔掩码（True表示异常值）
        """
        if self.df is None:
            raise ValueError("请先加载数据")
            
        # 为安全起见，先将列强制转换为数值（非数值变为 NaN）
        series = pd.to_numeric(self.df[column], errors='coerce')

        # 是否为 pH 列（与 notebook 中 plot_outliers 的 ph 参数保持一致）
        is_ph = ('pH' in column) or ('ph' in column.lower())

        if method == 'robust':
            # 与 Analisis/funciones_analisis.plot_outliers 完全一致的“robust”实现：
            # 使用 mean 与 mean absolute deviation（非传统 MAD），并缩放为 3 * mad。
            if is_ph:
                base = series[(series >= 0) & (series <= 14)]
            else:
                base = series.copy()
            base = base.dropna()

            if len(base) >= 2:
                mean_val = base.mean()
                mad_mean = (base - mean_val).abs().mean()  # mean absolute deviation
                if pd.isna(mad_mean) or mad_mean == 0:
                    robust_mask = pd.Series(False, index=series.index)
                    robust_z = pd.Series(0.0, index=series.index)
                else:
                    robust_z = (series - mean_val) / (3 * mad_mean)
                    robust_mask = robust_z.abs() >= threshold
            else:
                robust_mask = pd.Series(False, index=series.index)
                robust_z = pd.Series(0.0, index=series.index)

            # pH 物理范围约束（0-14 之外判为异常）
            if is_ph:
                ph_mask = (series < 0) | (series > 14)
                mask = (robust_mask | ph_mask).fillna(False)
            else:
                ph_mask = pd.Series(False, index=series.index)
                mask = robust_mask.fillna(False)

        elif method == 'zscore':
            mask = self.analysis_funcs.outliers_zscore(series, threshold)
            robust_mask = mask.copy()
            ph_mask = pd.Series(False, index=series.index)
            robust_z = None
        elif method == 'iqr':
            mask = self.analysis_funcs.outliers_iqr(series)
            robust_mask = mask.copy()
            ph_mask = pd.Series(False, index=series.index)
            robust_z = None
        elif method == 'rolling':
            mask = self.analysis_funcs.outliers_rolling(series)
            robust_mask = mask.copy()
            ph_mask = pd.Series(False, index=series.index)
            robust_z = None
        else:
            raise ValueError(f"不支持的方法: {method}")
            
        if plot:
            # 为了绘图使用数值化后的列，避免字符串引起的问题
            tmp_df = self.df.copy()
            tmp_df[column] = series
            # 向 plot_outliers 传入 ph 标志，保证与 notebook 图例/标注一致
            try:
                self.analysis_funcs.plot_outliers(tmp_df, column, ph=is_ph)
            except TypeError:
                # 向后兼容：如果老版本函数没有 ph 参数，则退化为原始调用
                self.analysis_funcs.plot_outliers(tmp_df, column)

        if return_details:
            return {
                'mask': mask.fillna(False),
                'robust_mask': robust_mask.fillna(False) if isinstance(robust_mask, pd.Series) else mask.fillna(False),
                'ph_mask': ph_mask.fillna(False) if isinstance(ph_mask, pd.Series) else pd.Series(False, index=series.index),
                'robust_z': robust_z if robust_z is not None else None,
            }
        return mask
    
    def analyze_missing_dates(self, dt_col: str = "date_time", 
                            min_rows: int = 1) -> Dict:
        """
        分析缺失数据的日期
        
        Args:
            dt_col: 日期时间列名
            min_rows: 每天最小行数
            
        Returns:
            包含有数据和无数据日期的字典
        """
        if self.df is None:
            raise ValueError("请先加载数据")
            
        return self.analysis_funcs.fechas_con_y_sin_datos(
            self.df, dt_col=dt_col, min_rows=min_rows
        )

    def reload_analysis_functions(self):
        """在同一进程中重新加载 Analisis/funciones_analisis.py。

        使用场景：当你在编辑 `Analisis/funciones_analisis.py`（或同事更新该文件）
        并希望在同一 Python 进程中（例如长期运行的分析器）获取最新修改时，
        调用此方法可以重新载入模块并更新 self.analysis_funcs。
        """
        analysis_path = Path(__file__).parent.parent / 'Analisis' / 'funciones_analisis.py'
        if not analysis_path.exists():
            raise FileNotFoundError(f"无法找到分析函数文件: {analysis_path}")

        spec = importlib.util.spec_from_file_location("analysis_functions", analysis_path)
        analysis_module = importlib.util.module_from_spec(spec)
        # 将新模块替换到 sys.modules 中，确保后续 importlib.reload 或直接引用都使用新模块
        sys.modules["analysis_functions"] = analysis_module
        spec.loader.exec_module(analysis_module)
        self.analysis_funcs = analysis_module
        return self.analysis_funcs
        
    def run_drift_analysis(self, reference_data: Optional[pd.DataFrame] = None, 
                          method: str = 'auto', **kwargs):
        """
        运行数据漂移分析
        
        接口说明：
        此方法会调用 Analisis/funciones_analisis.py 中的 drift 分析函数。
        当你的同事完成 drift 分析代码后，只需在 funciones_analisis.py 中添加对应函数，
        然后在此处调用即可，无需修改 pipeline 其他部分。
        
        Args:
            reference_data: 参考数据集（可选，用于对比分析）
            method: drift 检测方法，默认 'auto'
                   可选: 'statistical', 'model_based', 'distance_based' 等
            **kwargs: 其他参数传递给 drift 分析函数
            
        Returns:
            drift 分析结果字典，包含：
            - drift_detected: bool, 是否检测到漂移
            - drift_score: float, 漂移分数
            - drift_features: list, 发生漂移的特征列表
            - report: dict, 详细报告
            
        示例用法：
            pm = PipelineManager()
            pm.load_data("data.csv")
            result = pm.run_drift_analysis()
            if result['drift_detected']:
                print(f"检测到漂移，分数: {result['drift_score']}")
        """
        if self.df is None:
            raise ValueError("请先加载数据")
        
        # TODO: 等待同事实现 drift 分析函数
        # 预期在 funciones_analisis.py 中有类似函数：
        # def analizar_drift(df, reference_df=None, method='auto', **kwargs):
        #     ...
        #     return {
        #         'drift_detected': bool,
        #         'drift_score': float,
        #         'drift_features': [...],
        #         'report': {...}
        #     }
        
        # 当函数准备好后，取消下面的注释：
        # return self.analysis_funcs.analizar_drift(
        #     self.df, 
        #     reference_data=reference_data,
        #     method=method,
        #     **kwargs
        # )
        
        # 临时返回占位结果
        print("⚠️  Drift 分析功能尚未实现，等待 funciones_analisis.py 中添加相关函数")
        return {
            'drift_detected': False,
            'drift_score': 0.0,
            'drift_features': [],
            'report': {},
            'status': 'not_implemented'
        }
    
    def generate_basic_info(self, df: pd.DataFrame) -> dict:
        """
        生成基本数据信息
        
        Returns:
            包含数据形状、类型、缺失值等信息的字典
        """
        return {
            'shape': df.shape,
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> dict:
        """
        生成数据质量报告
        
        Returns:
            包含时间覆盖、数据完整性、Laguna分析等信息的字典
        """
        missing_dates = self.analyze_missing_dates()
        
        return {
            'time_range': {
                'start': df['date_time'].min(),
                'end': df['date_time'].max()
            },
            'completeness': {
                'dates_with_data': missing_dates.get('total_con', 0),
                'dates_without_data': missing_dates.get('total_sin', 0),
                'percentage_with_data': missing_dates.get('porcentaje_con', 0),
                'percentage_without_data': missing_dates.get('porcentaje_sin', 0)
            },
            'lagunas': missing_dates.get('sin_datos', [])
        }
    
    def analyze_outliers_batch(self, df: pd.DataFrame, numeric_cols: list = None, 
                               method: str = 'robust', threshold: float = 3.5) -> dict:
        """
        批量分析多个列的异常值
        
        Args:
            df: 数据框
            numeric_cols: 要分析的数值列列表，默认自动检测
            method: 检测方法
            threshold: 阈值
            
        Returns:
            包含所有异常值信息的字典
        """
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            numeric_cols = [col for col in numeric_cols if col != 'date_time']
        
        results = {
            'summary': {},
            'details': {},
            'all_outliers': []
        }
        
        for col in numeric_cols:
            # 检测异常值
            details = self.detect_outliers(
                col, method=method, threshold=threshold, 
                plot=False, return_details=True
            )
            
            outliers_mask = details['mask']
            robust_mask = details['robust_mask']
            ph_mask = details['ph_mask']
            robust_z = details.get('robust_z')
            
            results['summary'][col] = int(outliers_mask.sum())
            results['details'][col] = details
            
            # 收集异常值数据
            if outliers_mask.sum() > 0:
                series_col = pd.to_numeric(df[col], errors='coerce')
                
                # pH 边界异常
                if ph_mask.any():
                    ph_rows = df[ph_mask].copy()
                    ph_rows['variable'] = col
                    ph_rows['value'] = series_col[ph_mask]
                    ph_rows['detection_method'] = 'ph_bounds'
                    results['all_outliers'].append(ph_rows[['date_time', 'variable', 'value', 'detection_method']])
                
                # Robust 异常
                robust_only_mask = robust_mask & (~ph_mask)
                if robust_only_mask.any():
                    rb_rows = df[robust_only_mask].copy()
                    rb_rows['variable'] = col
                    rb_rows['value'] = series_col[robust_only_mask]
                    rb_rows['detection_method'] = 'robust_mad3_3.5'
                    if robust_z is not None:
                        rb_rows['robust_z'] = robust_z[robust_only_mask]
                        results['all_outliers'].append(rb_rows[['date_time', 'variable', 'value', 'detection_method', 'robust_z']])
                    else:
                        results['all_outliers'].append(rb_rows[['date_time', 'variable', 'value', 'detection_method']])
        
        return results
    
    def run_ar_analysis(self) -> dict:
        """
        AR模型分析（预留接口）
        
        Returns:
            AR分析结果字典
        """
        if self.df is None:
            raise ValueError("请先加载数据")
        
        print("⚠️  AR 模型分析功能尚未实现")
        return {
            'status': 'not_implemented',
            'message': 'AR 模型分析等待实现'
        }
    
    def run_diff_analysis(self) -> dict:
        """
        差分分析（预留接口）
        
        Returns:
            差分分析结果字典
        """
        if self.df is None:
            raise ValueError("请先加载数据")
        
        print("⚠️  差分分析功能尚未实现")
        return {
            'status': 'not_implemented',
            'message': '差分分析等待实现'
        }