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

    def analyze_plant_data(self, data_file: str, do_outliers: bool = True,
                          do_lagunas: bool = True, do_ar: bool = False, 
                          do_diff: bool = False, do_drift: bool = False):
        """
        分析单个工厂的数据
        
        Args:
            data_file: 数据文件路径
            do_outliers: 是否执行异常值分析（包含异常值图像，不可分离）
            do_lagunas: 是否执行数据空档(Laguna)分析
            do_ar: 是否执行AR模型分析
            do_diff: 是否执行差分分析
            do_drift: 是否执行漂移分析
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
        
        # 2. 数据质量分析（Lagunas - 可选）
        if do_lagunas:
            self._analyze_data_quality(df, report_dir)
        
        # 3. 异常值分析（可选，包含图像生成）
        if do_outliers:
            self._analyze_outliers(df, report_dir)
        
        # 4. AR 模型分析（可选）
        if do_ar:
            self._analyze_ar(df, report_dir)
        
        # 5. 差分分析（可选）
        if do_diff:
            self._analyze_diff(df, report_dir)
        
        # 6. Drift 分析（可选）
        if do_drift:
            self._analyze_drift(df, report_dir)
        
        end_time = time()
        print(f"\n分析完成！用时: {end_time - start_time:.2f}秒")
        print(f"报告保存在: {report_dir.absolute()}")
        
    def _save_basic_info(self, df: pd.DataFrame, report_dir: Path):
        """保存基本数据信息"""
        info = self.pipeline.generate_basic_info(df)
        
        with open(report_dir / "1_基本信息.txt", "w", encoding='utf-8') as f:
            f.write("数据基本信息\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"数据形状: {info['shape']}\n\n")
            f.write("数据类型:\n")
            for col, dtype in info['dtypes'].items():
                f.write(f"  {col}: {dtype}\n")
            f.write("\n缺失值统计:\n")
            for col, missing in info['missing_values'].items():
                if missing > 0:
                    f.write(f"  {col}: {missing}\n")
            
    def _analyze_data_quality(self, df: pd.DataFrame, report_dir: Path):
        """分析数据质量"""
        quality_report = self.pipeline.generate_data_quality_report(df)
        
        with open(report_dir / "2_数据质量报告.txt", "w", encoding='utf-8') as f:
            f.write("数据质量报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("时间覆盖范围:\n")
            f.write(f"起始时间: {quality_report['time_range']['start']}\n")
            f.write(f"结束时间: {quality_report['time_range']['end']}\n\n")
            
            f.write("数据完整性:\n")
            comp = quality_report['completeness']
            f.write(f"有数据的日期数: {comp['dates_with_data']}\n")
            f.write(f"无数据的日期数: {comp['dates_without_data']}\n")
            f.write(f"有数据占比: {comp['percentage_with_data']}%\n")
            f.write(f"无数据占比: {comp['percentage_without_data']}%\n\n")
            
            # 显示缺失数据的日期范围（lagunas）
            lagunas = quality_report['lagunas']
            if lagunas:
                f.write("缺失数据的时间段（Lagunas）:\n")
                for inicio, fin in lagunas[:10]:
                    f.write(f"  从 {inicio} 到 {fin}\n")
                if len(lagunas) > 10:
                    f.write(f"  ... 还有 {len(lagunas) - 10} 个时间段\n")
                    
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
            
            # 检测异常值并保存统计（与 notebook 一致：使用 mean+mean abs deviation/3 的稳健度量）
            details = self.pipeline.detect_outliers(
                col, method='robust', threshold=3.5, plot=False, return_details=True
            )
            outliers_mask = details['mask']
            robust_mask = details['robust_mask']
            ph_mask = details['ph_mask']
            robust_z = details.get('robust_z')
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
            
            # 保存异常值数据到列表（区分 ph 范围异常与“robust”异常，保持与图像颜色一致）
            if outliers_mask.sum() > 0:
                series_col = pd.to_numeric(df[col], errors='coerce')

                # 1) ph 边界异常（红色）
                if ph_mask.any():
                    ph_rows = df[ph_mask].copy()
                    ph_rows['variable'] = col
                    ph_rows['value'] = series_col[ph_mask]
                    ph_rows['detection_method'] = 'ph_bounds'
                    all_outliers_data.append(ph_rows[['date_time', 'variable', 'value', 'detection_method']])

                # 2) robust-only 异常（橙色）
                robust_only_mask = robust_mask & (~ph_mask)
                if robust_only_mask.any():
                    rb_rows = df[robust_only_mask].copy()
                    rb_rows['variable'] = col
                    rb_rows['value'] = series_col[robust_only_mask]
                    rb_rows['detection_method'] = 'robust_mad3_3.5'
                    # 若可用，附加 robust_z 便于事后筛选
                    if robust_z is not None:
                        rb_rows['robust_z'] = robust_z[robust_only_mask]
                        all_outliers_data.append(rb_rows[['date_time', 'variable', 'value', 'detection_method', 'robust_z']])
                    else:
                        all_outliers_data.append(rb_rows[['date_time', 'variable', 'value', 'detection_method']])
            
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
                
    def _analyze_visualization(self, df: pd.DataFrame, report_dir: Path):
        """时间序列可视化（原 _analyze_time_series，重命名以明确功能）"""
        from time import time
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'date_time']
        
        print(f"\n开始时间序列可视化，共 {len(numeric_cols)} 个变量...")
        print("-" * 50)
        
        # 预先计算小时列，避免重复计算
        df['hour'] = pd.to_datetime(df['date_time']).dt.hour
        
        for i, col in enumerate(numeric_cols, 1):
            start_time = time()
            print(f"[{i}/{len(numeric_cols)}] 可视化 {col}...", end=' ')
            
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
    
    def _analyze_drift(self, df: pd.DataFrame, report_dir: Path):
        """
        数据漂移分析
        
        接口说明：
        当你的同事在 funciones_analisis.py 中实现 drift 分析函数后，
        此方法会自动调用并生成报告。
        
        预期 funciones_analisis.py 中的函数格式：
        - analizar_drift(df, **kwargs) -> dict
        - detectar_drift_temporal(df, window=30) -> dict
        - 或其他 drift 相关函数
        """
        from time import time
        start_time = time()
        
        print(f"\n开始数据漂移分析...")
        print("-" * 50)
        
        # 调用 PipelineManager 的 drift 分析接口
        drift_result = self.pipeline.run_drift_analysis()
        
        # 保存 drift 分析结果
        with open(report_dir / "5_Drift分析报告.txt", "w", encoding='utf-8') as f:
            f.write("数据漂移分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            if drift_result.get('status') == 'not_implemented':
                f.write("⚠️  Drift 分析功能尚未实现\n")
                f.write("等待在 Analisis/funciones_analisis.py 中添加相关函数\n\n")
                f.write("预期函数示例：\n")
                f.write("def analizar_drift(df, reference_df=None, method='auto', **kwargs):\n")
                f.write("    # 实现 drift 检测逻辑\n")
                f.write("    return {'drift_detected': bool, 'drift_score': float, ...}\n")
            else:
                f.write(f"Drift 检测结果: {'检测到漂移' if drift_result['drift_detected'] else '未检测到漂移'}\n")
                f.write(f"Drift 分数: {drift_result['drift_score']:.4f}\n\n")
                
                if drift_result['drift_features']:
                    f.write("发生漂移的特征:\n")
                    for feature in drift_result['drift_features']:
                        f.write(f"  - {feature}\n")
                
                if drift_result.get('report'):
                    f.write("\n详细报告:\n")
                    f.write(str(drift_result['report']))
        
        end_time = time()
        print(f"Drift 分析完成! ({end_time - start_time:.2f}秒)")
        print(f"状态: {drift_result.get('status', 'completed')}")
    
    def _analyze_ar(self, df: pd.DataFrame, report_dir: Path):
        """AR模型分析（预留接口）"""
        from time import time
        start_time = time()
        
        print(f"\n开始 AR 模型分析...")
        print("-" * 50)
        
        with open(report_dir / "6_AR模型分析.txt", "w", encoding='utf-8') as f:
            f.write("AR 模型分析报告\n")
            f.write("=" * 50 + "\n\n")
            f.write("⚠️  AR 模型功能尚未实现\n")
            f.write("计划集成自 Modelos/AR_model.ipynb\n\n")
            f.write("预期功能：\n")
            f.write("- 自回归模型训练与拟合\n")
            f.write("- 未来N步预测\n")
            f.write("- 残差分析与诊断\n")
        
        end_time = time()
        print(f"AR 模型分析占位完成! ({end_time - start_time:.2f}秒)")
        print(f"状态: not_implemented")
    
    def _analyze_diff(self, df: pd.DataFrame, report_dir: Path):
        """差分分析（预留接口）"""
        from time import time
        start_time = time()
        
        print(f"\n开始差分分析...")
        print("-" * 50)
        
        with open(report_dir / "7_差分分析.txt", "w", encoding='utf-8') as f:
            f.write("差分分析报告\n")
            f.write("=" * 50 + "\n\n")
            f.write("⚠️  差分分析功能尚未实现\n")
            f.write("计划集成自 Modelos/DIFF_model.ipynb\n\n")
            f.write("预期功能：\n")
            f.write("- 一阶/二阶差分处理\n")
            f.write("- 平稳性检验（ADF测试）\n")
            f.write("- 趋势与季节性分离\n")
        
        end_time = time()
        print(f"差分分析占位完成! ({end_time - start_time:.2f}秒)")
        print(f"状态: not_implemented")

def main():
    """主函数 - 双层菜单：主菜单（分析文件/分析设置）+ 设置子菜单"""
    analyzer = AutoAnalyzer()
    data_dir = analyzer.project_root / "df_procesados"

    # 默认分析配置
    config = {
        'outliers': True,
        'lagunas': True,
        'ar': False,
        'diff': False,
        'drift': False
    }

    while True:
        # ===== 主菜单 =====
        print("\n" + "="*70)
        print("  🔬 水处理厂数据自动分析系统")
        print("  Water Treatment Plant Data Analysis System")
        print("="*70)
        print("\n主菜单：")
        print("  [1] 分析文件")
        print("  [2] 分析设置")
        print("  [0] 退出")
        print("-"*70)
        
        main_choice = input("请选择操作: ").strip()
        
        if main_choice == '0':
            print("\n👋 再见！")
            return
        
        elif main_choice == '1':
            # ===== 进入文件选择 =====
            if not data_dir.exists():
                print(f"❌ 数据目录不存在: {data_dir}")
                continue

            csv_files = sorted([p for p in data_dir.glob('*.csv') if p.is_file()])
            if not csv_files:
                print(f"❌ 未在 {data_dir} 找到任何 CSV 文件")
                continue

            print("\n" + "="*70)
            print("📁 选择要分析的文件（输入编号并回车）：")
            print("="*70)
            for idx, f in enumerate(csv_files, 1):
                try:
                    size_mb = f.stat().st_size / (1024*1024)
                    print(f"  [{idx:>2}] {f.name:<50} ({size_mb:.2f} MB)")
                except Exception:
                    print(f"  [{idx:>2}] {f.name}")
            print("  [ 0] 返回主菜单")

            chosen_path = None
            while True:
                choice = input("\n请输入编号: ").strip()
                if choice == '0':
                    break
                if not choice.isdigit():
                    print("请输入有效数字编号。")
                    continue
                idx = int(choice)
                if 1 <= idx <= len(csv_files):
                    chosen_path = csv_files[idx-1]
                    break
                else:
                    print("编号超出范围，请重试。")
            
            if not chosen_path:
                continue  # 返回主菜单

            # 开始分析所选文件
            print(f"\n{'='*70}")
            print(f"开始分析: {chosen_path.name}")
            print(f"当前配置: 异常值={config['outliers']}, 空档={config['lagunas']}, "
                  f"AR={config['ar']}, DIFF={config['diff']}, Drift={config['drift']}")
            print(f"{'='*70}")

            try:
                analyzer.analyze_plant_data(
                    str(chosen_path),
                    do_outliers=config['outliers'],
                    do_lagunas=config['lagunas'],
                    do_ar=config['ar'],
                    do_diff=config['diff'],
                    do_drift=config['drift']
                )
            except Exception as e:
                print(f"分析 {chosen_path.name} 时出错: {str(e)}")
                import traceback
                print(traceback.format_exc())
            finally:
                plt.close('all')
                gc.collect()
            
            print("\n分析完成！按 Enter 返回主菜单...")
            input()
        
        elif main_choice == '2':
            # ===== 进入分析设置子菜单 =====
            while True:
                print("\n" + "="*70)
                print("⚙️  分析设置菜单")
                print("="*70)
                print(f"  [1] 异常值分析     : {'✓ 开启' if config['outliers'] else '✗ 关闭'} (含图像)")
                print(f"  [2] 空档分析       : {'✓ 开启' if config['lagunas'] else '✗ 关闭'} (Lagunas)")
                print(f"  [3] AR模型分析     : {'✓ 开启' if config['ar'] else '✗ 关闭'} (预留)")
                print(f"  [4] 差分分析       : {'✓ 开启' if config['diff'] else '✗ 关闭'} (预留)")
                print(f"  [5] Drift分析      : {'✓ 开启' if config['drift'] else '✗ 关闭'} (预留)")
                print("  [0] 返回主菜单")
                print("-"*70)
                
                setting_choice = input("请选择要切换的选项（输入编号）: ").strip()
                
                if setting_choice == '0':
                    break
                elif setting_choice == '1':
                    config['outliers'] = not config['outliers']
                    print(f"✓ 异常值分析已{'开启' if config['outliers'] else '关闭'}")
                elif setting_choice == '2':
                    config['lagunas'] = not config['lagunas']
                    print(f"✓ 空档分析已{'开启' if config['lagunas'] else '关闭'}")
                elif setting_choice == '3':
                    config['ar'] = not config['ar']
                    print(f"✓ AR模型分析已{'开启' if config['ar'] else '关闭'}")
                elif setting_choice == '4':
                    config['diff'] = not config['diff']
                    print(f"✓ 差分分析已{'开启' if config['diff'] else '关闭'}")
                elif setting_choice == '5':
                    config['drift'] = not config['drift']
                    print(f"✓ Drift分析已{'开启' if config['drift'] else '关闭'}")
                else:
                    print("❌ 无效选项，请重试。")
        
        else:
            print("❌ 无效选项，请输入 0、1 或 2。")

if __name__ == "__main__":
    main()