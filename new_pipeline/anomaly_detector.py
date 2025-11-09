"""
异常检测模块
集成outliers、drift和数据间隔检测
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('../Analisis')
import funciones_analisis as fa
import drift_funcs as df

class AnomalyDetector:
    def __init__(self, config):
        self.config = config['anomaly_detection']
        
    def detect_outliers(self, data):
        """检测异常值"""
        if not self.config['outliers']['enabled']:
            return None
            
        threshold = self.config['outliers']['threshold']
        flags = pd.DataFrame(index=data.index)
        
        num_cols = data.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            # 使用之前实现的outlier检测函数
            ph = 'pH' in col
            flags[f'{col}_outlier'] = fa.outliers_zscore(data[col], threshold)
            
        return flags
        
    def detect_drift(self, data, date_col):
        """检测数据漂移"""
        if not self.config['drift']['enabled']:
            return None
            
        window = self.config['drift']['window']
        strategies = self.config['drift']['strategies']
        
        drift_results = {}
        for strategy in strategies:
            # 使用之前实现的drift检测函数
            try:
                results = df.make_report_for_plant(
                    data,
                    Path('../output/drift_reports'),
                    strategy=strategy,
                    CURRENT_WINDOW=window,
                    plant_name='plant',
                    SAVE_HTML=True
                )
                drift_results[strategy] = results
            except Exception as e:
                print(f"Drift detection failed for strategy {strategy}: {str(e)}")
                
        return drift_results
        
    def detect_gaps(self, data, date_col):
        """检测数据间隔"""
        if not self.config['gaps']['enabled']:
            return None
            
        max_gap = pd.Timedelta(self.config['gaps']['max_gap'])
        data = data.sort_values(date_col)
        
        time_diff = data[date_col].diff()
        gaps = time_diff > max_gap
        
        return pd.DataFrame({
            'gap_detected': gaps,
            'gap_duration': time_diff
        }, index=data.index)
        
    def run_all_detections(self, data, date_col):
        """运行所有异常检测"""
        results = {}
        
        results['outliers'] = self.detect_outliers(data)
        results['drift'] = self.detect_drift(data, date_col)
        results['gaps'] = self.detect_gaps(data, date_col)
        
        return results