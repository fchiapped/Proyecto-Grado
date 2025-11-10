import os
from pathlib import Path
from pipeline_manager import PipelineManager
import matplotlib.pyplot as plt

def test_pipeline():
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    
    # 设置数据文件路径
    data_path = project_root / "df_procesados" / "df_planta_1.csv"
    
    # 初始化pipeline
    pipeline = PipelineManager(data_path)
    
    # 1. 测试数据加载
    print("1. 测试数据加载...")
    df = pipeline.load_data()
    print(f"数据形状: {df.shape}")
    print("\n数据前5行:")
    print(df.head())
    
    # 2. 查看数据列
    print("\n2. 数据列名:")
    print(df.columns.tolist())
    
    # 3. 测试异常值检测
    print("\n3. 测试异常值检测...")
    # 选择可用的数据列进行异常值检测
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns[:3]  # 先测试前3个数值列
    columns_to_test = numeric_columns.tolist()
    
    plt.figure(figsize=(15, 5 * len(columns_to_test)))
    for i, col in enumerate(columns_to_test, 1):
        print(f"\n分析 {col} 的异常值...")
        outliers = pipeline.detect_outliers(col, method='zscore', threshold=3.5, plot=False)
        print(f"发现 {outliers.sum()} 个异常值")
        
        # 创建子图
        plt.subplot(len(columns_to_test), 1, i)
        pipeline.analysis_funcs.plot_outliers(df, col)
    
    plt.tight_layout()
    plt.show()
    
    # 3. 测试缺失数据分析
    print("\n3. 测试缺失数据分析...")
    missing_dates = pipeline.analyze_missing_dates()
    print("\n数据缺失情况:")
    print(f"有数据的日期数: {len(missing_dates.get('fechas_con', []))}")
    print(f"无数据的日期数: {len(missing_dates.get('fechas_sin', []))}")
    
    # 展示一些具体的缺失日期
    if 'fechas_sin' in missing_dates and missing_dates['fechas_sin']:
        print("\n部分缺失数据的日期示例:")
        print(list(missing_dates['fechas_sin'])[:5])

if __name__ == "__main__":
    test_pipeline()