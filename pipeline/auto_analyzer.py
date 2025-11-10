# 设置无界面后端，避免GUI锁死
import matplotlib
matplotlib.use("Agg")  # 必须在import pyplot之前设置

# 设置matplotlib支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from pipeline_manager import PipelineManager
import seaborn as sns
from datetime import datetime
import gc  # 用于垃圾回收

class AutoAnalyzer:
    def __init__(self, data_path: str = None):
        """
        初始化自动分析器
        
        Args:
            data_path: 数据文件或目录的路径
        """
        self.project_root = Path(__file__).parent.parent
        self.pipeline = PipelineManager()
        self.results_dir = self.project_root / "resultados_analisis"
        self.results_dir.mkdir(exist_ok=True)
        
    def create_report_dir(self, plant_name: str) -> Path:
        """创建报告目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.results_dir / f"{plant_name}_{timestamp}"
        report_dir.mkdir(exist_ok=True)
        return report_dir

    def analyze_plant_data(self, data_file: str, do_time_series: bool = False):
        """
        分析单个工厂的数据
        
        Args:
            data_file: 数据文件路径
            do_time_series: 是否执行时间序列分析
        """
        from time import time
        start_time = time()

        # 在分析每个数据文件前重新加载分析函数，确保使用最新代码（同一进程生效）
        try:
            self.pipeline.reload_analysis_functions()
            print("已重新加载 Analisis/funciones_analisis.py（运行时热重载）")
        except Exception as e:
            print(f"警告：重新加载分析函数失败，将使用已有加载模块。错误: {e}")
        
        # 提取工厂名称
        plant_name = Path(data_file).stem.split('_')[1]  # 假设文件名格式为 df_planta_X.csv
        print(f"\n{'='*50}")
        print(f"开始分析 {plant_name} 的数据...")
        
        # 创建报告目录
        report_dir = self.create_report_dir(plant_name)
        
        # 1. 数据加载
        df = self.pipeline.load_data(data_file)
        self._save_basic_info(df, report_dir)
        
        # 2. 数据质量分析
        self._analyze_data_quality(df, report_dir)
        
        # 3. 异常值分析
        self._analyze_outliers(df, report_dir)
        
        # 4. 时间序列分析（可选）
        if do_time_series:
            self._analyze_time_series(df, report_dir)
        
        end_time = time()
        print(f"\n分析完成！用时: {end_time - start_time:.2f}秒")
        print(f"报告保存在: {report_dir.absolute()}")
        
    def _save_basic_info(self, df: pd.DataFrame, report_dir: Path):
        """保存基本数据信息"""
        with open(report_dir / "1_基本信息.txt", "w", encoding='utf-8') as f:
            f.write("数据基本信息\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"数据形状: {df.shape}\n\n")
            f.write("数据类型:\n")
            f.write(df.dtypes.to_string())
            f.write("\n\n缺失值统计:\n")
            f.write(df.isnull().sum().to_string())
            
    def _analyze_data_quality(self, df: pd.DataFrame, report_dir: Path):
        """分析数据质量"""
        # 分析缺失数据
        missing_dates = self.pipeline.analyze_missing_dates()
        
        with open(report_dir / "2_数据质量报告.txt", "w", encoding='utf-8') as f:
            f.write("数据质量报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("时间覆盖范围:\n")
            f.write(f"起始时间: {df['date_time'].min()}\n")
            f.write(f"结束时间: {df['date_time'].max()}\n\n")
            
            f.write("数据完整性:\n")
            f.write(f"有数据的日期数: {missing_dates.get('total_con', 0)}\n")
            f.write(f"无数据的日期数: {missing_dates.get('total_sin', 0)}\n")
            f.write(f"有数据占比: {missing_dates.get('porcentaje_con', 0)}%\n")
            f.write(f"无数据占比: {missing_dates.get('porcentaje_sin', 0)}%\n\n")
            
            # 显示缺失数据的日期范围（lagunas）
            if 'sin_datos' in missing_dates and missing_dates['sin_datos']:
                f.write("缺失数据的时间段（Lagunas）:\n")
                for inicio, fin in missing_dates['sin_datos'][:10]:  # 显示前10个时间段
                    f.write(f"  从 {inicio} 到 {fin}\n")
                if len(missing_dates['sin_datos']) > 10:
                    f.write(f"  ... 还有 {len(missing_dates['sin_datos']) - 10} 个时间段\n")
                    
    def _analyze_outliers(self, df: pd.DataFrame, report_dir: Path):
        """分析异常值"""
        from time import time
        
        # 选择数值列进行分析
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'date_time']
        outlier_summary = {}
        
        # 用于收集所有异常值数据
        all_outliers_data = []
        
        print(f"\n开始异常值分析，共 {len(numeric_cols)} 个变量需要处理...")
        print("-" * 50)
        
        for i, col in enumerate(numeric_cols, 1):
            start_time = time()
            print(f"[{i}/{len(numeric_cols)}] 分析 {col} 的异常值...", end=' ')
            
            # 检测异常值并保存统计
            outliers_mask = self.pipeline.detect_outliers(col, method='zscore', threshold=3.5, plot=False)
            outlier_summary[col] = int(outliers_mask.sum())
            
            # 为每个变量创建新的图表
            fig, ax = plt.subplots(figsize=(15, 5))
            
            # 准备用于绘图的DataFrame（确保date_time是datetime类型）
            plot_df = df.copy()
            if 'date_time' in plot_df.columns:
                plot_df['date_time'] = pd.to_datetime(plot_df['date_time'], errors='coerce')
            
            # 判断是否为pH列，使用特殊处理
            is_ph = 'pH' in col or 'ph' in col.lower()
            self.pipeline.analysis_funcs.plot_outliers(plot_df, col, ax=ax, ph=is_ph)
            
            # 保存异常值数据到列表
            if outliers_mask.sum() > 0:
                outlier_rows = df[outliers_mask].copy()
                outlier_rows['variable'] = col
                outlier_rows['value'] = pd.to_numeric(df[col], errors='coerce')[outliers_mask]
                outlier_rows['detection_method'] = 'zscore_3.5'
                all_outliers_data.append(outlier_rows[['date_time', 'variable', 'value', 'detection_method']])
            
            # 保存并清理
            fig.tight_layout()
            fig.savefig(report_dir / f"3_异常值分析_{col}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            gc.collect()  # 强制垃圾回收
            
            end_time = time()
            print(f"完成! ({end_time - start_time:.2f}秒, 发现 {outlier_summary[col]} 个异常值)")  # 显示每个变量的处理时间
        
        # 导出所有异常值到CSV
        if all_outliers_data:
            all_outliers_df = pd.concat(all_outliers_data, ignore_index=True)
            all_outliers_df.to_csv(report_dir / "3_异常值数据.csv", index=False, encoding='utf-8-sig')
            print(f"\n异常值数据已导出到: 3_异常值数据.csv (共 {len(all_outliers_df)} 行)")
        else:
            print("\n未检测到异常值")
        
        # 保存异常值统计
        with open(report_dir / "3_异常值统计.txt", "w", encoding='utf-8') as f:
            f.write("异常值统计\n")
            f.write("=" * 50 + "\n\n")
            for col, count in outlier_summary.items():
                f.write(f"{col}: {count} 个异常值\n")
                
    def _analyze_time_series(self, df: pd.DataFrame, report_dir: Path):
        """时间序列分析"""
        from time import time
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'date_time']
        
        print(f"\n开始时间序列分析，共 {len(numeric_cols)} 个变量...")
        print("-" * 50)
        
        # 预先计算小时列，避免重复计算
        df['hour'] = pd.to_datetime(df['date_time']).dt.hour
        
        for i, col in enumerate(numeric_cols, 1):
            start_time = time()
            print(f"[{i}/{len(numeric_cols)}] 分析时间序列 {col}...", end=' ')
            
            # 为每个变量创建新的图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
            
            # 时间序列图
            ax1.plot(df['date_time'], df[col])
            ax1.set_title(f"{col} 时间序列")
            ax1.set_xlabel('时间')
            ax1.set_ylabel(col)
            
            # 箱线图（按小时）
            sns.boxplot(x='hour', y=col, data=df, ax=ax2)
            ax2.set_title(f"{col} 每小时分布")
            ax2.set_xlabel('小时')
            ax2.set_ylabel(col)
            
            # 保存并清理
            fig.tight_layout()
            fig.savefig(report_dir / f"4_时间序列_{col}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            gc.collect()  # 强制垃圾回收
            
            end_time = time()
            print(f"完成! ({end_time - start_time:.2f}秒)")
            
        # 删除临时列
        del df['hour']
        gc.collect()

def main():
    """主函数"""
    analyzer = AutoAnalyzer()
    ################################
    # 指定要分析的工厂数据文件
    data_dir = analyzer.project_root / "df_procesados"
    plant_files = [
        data_dir / "df_planta_1.csv"


    ]
    ################################
    # 分析指定的工厂数据
    for file in plant_files:
        if not file.exists():
            print(f"文件不存在: {file}")
            continue
            
        print(f"\n{'='*50}")
        print(f"开始分析: {file.name}")
        print(f"{'='*50}")
        
        try:
            # 只执行到异常值分析
            analyzer.analyze_plant_data(str(file), do_time_series=False)
        except Exception as e:
            print(f"分析 {file.name} 时出错: {str(e)}")
            import traceback
            print(traceback.format_exc())
        finally:
            # 强制清理内存
            plt.close('all')
            gc.collect()
            
    print("\n所有分析完成！")

if __name__ == "__main__":
    main()