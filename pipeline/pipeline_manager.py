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
    
    def detect_outliers(self, column: str, method: str = 'zscore', 
                       threshold: float = 3.5, plot: bool = True) -> pd.Series:
        """
        检测异常值
        
        Args:
            column: 要分析的列名
            method: 检测方法 ('zscore', 'iqr', 'rolling')
            threshold: 阈值
            plot: 是否绘制图表
            
        Returns:
            异常值的布尔掩码
        """
        if self.df is None:
            raise ValueError("请先加载数据")
            
        if method == 'zscore':
            mask = self.analysis_funcs.outliers_zscore(self.df[column], threshold)
        elif method == 'iqr':
            mask = self.analysis_funcs.outliers_iqr(self.df[column])
        elif method == 'rolling':
            mask = self.analysis_funcs.outliers_rolling(self.df[column])
        else:
            raise ValueError(f"不支持的方法: {method}")
            
        if plot:
            self.analysis_funcs.plot_outliers(self.df, column)
            
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
        
    def run_drift_analysis(self):
        """
        运行数据漂移分析（待实现）
        """
        pass  # 待实现drift分析部分