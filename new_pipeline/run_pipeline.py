"""
主运行脚本
协调整个pipeline的运行
"""
import yaml
from pathlib import Path
from data_loader import DataLoader
from quality_analyzer import QualityAnalyzer
from anomaly_detector import AnomalyDetector
from alert_generator import AlertGenerator

def main():
    # 加载配置
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # 初始化组件
    loader = DataLoader(config_path)
    quality_analyzer = QualityAnalyzer(config)
    anomaly_detector = AnomalyDetector(config)
    alert_gen = AlertGenerator(config)
    
    # 处理每个数据文件
    for file_name in loader.get_files_list():
        print(f"Processing {file_name}...")
        
        # 1. 加载数据
        df = loader.load_data(file_name)
        date_col = config['input']['date_column']
        
        # 2. 运行数据质量检查
        quality_results = quality_analyzer.run_all_checks(df, date_col)
        
        # 3. 运行异常检测
        anomaly_results = anomaly_detector.run_all_detections(df, date_col)
        
        # 4. 生成警报和标志
        plant_name = file_name.replace('.csv', '').replace('df_', '')
        flags_file = alert_gen.generate_flags(quality_results, anomaly_results, plant_name)
        alert_gen.generate_alerts(quality_results, anomaly_results, plant_name)
        
        print(f"Completed processing {file_name}")
        print(f"Flags saved to: {flags_file}")
        
if __name__ == "__main__":
    main()