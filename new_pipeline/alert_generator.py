"""
警报和标志生成模块
生成和管理数据质量警报
"""
import pandas as pd
from pathlib import Path
import logging
import yaml
from datetime import datetime

class AlertGenerator:
    def __init__(self, config):
        self.config = config['output']
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志记录"""
        log_file = Path(self.config['alert_log'])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            filename=str(log_file),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def generate_flags(self, quality_results, anomaly_results, plant_name):
        """生成质量标志"""
        flags_dir = Path(self.config['flags_dir'])
        flags_dir.mkdir(parents=True, exist_ok=True)
        
        # 合并所有标志
        all_flags = pd.DataFrame()
        
        # 添加质量指标标志
        if quality_results.get('completeness'):
            all_flags = pd.concat([all_flags, quality_results['completeness']['flags']], axis=1)
        if quality_results.get('valid_range'):
            all_flags = pd.concat([all_flags, quality_results['valid_range']['flags']], axis=1)
        if quality_results.get('variability'):
            all_flags = pd.concat([all_flags, quality_results['variability']['flags']], axis=1)
            
        # 添加异常检测标志
        if anomaly_results.get('outliers') is not None:
            all_flags = pd.concat([all_flags, anomaly_results['outliers']], axis=1)
        if anomaly_results.get('gaps') is not None:
            all_flags = pd.concat([all_flags, anomaly_results['gaps']], axis=1)
            
        # 保存标志文件
        output_file = flags_dir / f"flags_{plant_name}_{datetime.now().strftime('%Y%m%d')}.csv"
        all_flags.to_csv(output_file)
        return output_file
        
    def generate_alerts(self, quality_results, anomaly_results, plant_name):
        """生成警报"""
        # 记录质量问题
        if quality_results.get('completeness'):
            scores = quality_results['completeness']['scores']
            for col, score in scores.items():
                if score < 0.8:  # 完整性阈值
                    logging.warning(f"{plant_name} - Low completeness in {col}: {score:.2f}")
                    
        # 记录异常检测结果
        if anomaly_results.get('outliers') is not None:
            n_outliers = anomaly_results['outliers'].sum().sum()
            if n_outliers > 0:
                logging.warning(f"{plant_name} - Detected {n_outliers} outliers")
                
        if anomaly_results.get('gaps') is not None:
            n_gaps = anomaly_results['gaps']['gap_detected'].sum()
            if n_gaps > 0:
                logging.warning(f"{plant_name} - Detected {n_gaps} data gaps")