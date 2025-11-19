# 设置无界面后端，避免GUI锁死
import matplotlib
matplotlib.use("Agg")

from auto_analyzer import AutoAnalyzer
from pathlib import Path
import gc
import matplotlib.pyplot as plt

analyzer = AutoAnalyzer()
data_dir = analyzer.project_root / "df_procesados"
test_file = data_dir / "df_planta_1.csv"

if test_file.exists():
    print(f"测试文件: {test_file.name}")
    try:
        analyzer.analyze_plant_data(str(test_file), do_time_series=False)
        print("\n测试完成！")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        plt.close('all')
        gc.collect()
else:
    print(f"文件不存在: {test_file}")
