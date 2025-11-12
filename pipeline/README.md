# Pipeline 自动化分析器使用说明

## 概述
此 pipeline 提供了自动化的数据质量分析和异常值检测功能，直接引用 `Analisis/funciones_analisis.py` 中的函数。

## 异常值检测算法

### 当前使用的算法
Pipeline 使用 **标准 Z-score 方法** 进行异常值检测：
- 计算公式：`z = (x - mean) / std`
- 判定标准：当 `|z| > 3.5` 时判定为异常值
- 阈值可调：默认 3.5，可通过参数修改

### 与 outliers.ipynb 的关系
- **检测算法一致**：都使用 z-score 方法（阈值 3.5）
- **可视化差异**：
  - `funciones_analisis.plot_outliers` 使用 **robust z-score**（基于 MAD - Mean Absolute Deviation）进行标注
  - 这是一种更稳健的方法，对极端值不太敏感
  - 两种方法都是有效的异常检测手段

### 其他可用算法
Pipeline 还支持以下方法（可在 `detect_outliers` 中指定）：
- `iqr`: 四分位距方法（默认 k=1.5）
- `rolling`: 滑动窗口方法（默认窗口=30）

## 输出文件说明

### 1. 基本信息 (`1_基本信息.txt`)
- 数据形状
- 数据类型
- 缺失值统计

### 2. 数据质量报告 (`2_数据质量报告.txt`)
- 时间覆盖范围
- 数据完整性统计
- **Lagunas（数据空档）时间段**

### 3. 异常值分析
- **`3_异常值统计.txt`**: 每个变量的异常值数量汇总
- **`3_异常值数据.csv`**: 所有异常值的详细数据，包含：
  - `date_time`: 时间戳
  - `variable`: 变量名
  - `value`: 异常值
  - `detection_method`: 检测方法（例如 zscore_3.5）
- **`3_异常值分析_[变量名].png`**: 每个变量的可视化图表
  - 蓝色点：正常值
  - 橙色点：robust z-score 检测的异常值
  - 红色点：pH 物理界限异常值（仅 pH 变量）

## 使用方法

### 快速测试单个文件
```powershell
cd "d:\学习使我快乐\2025年下学期\Proyecto-Grado\pipeline"
D:/学习使我快乐/2025年下学期/.venv/Scripts/python.exe quick_test_single.py
```

### 分析所有工厂数据
```powershell
D:/学习使我快乐/2025年下学期/.venv/Scripts/python.exe auto_analyzer.py
```

### 在代码中使用
```python
from pipeline_manager import PipelineManager

# 初始化
pm = PipelineManager()
pm.load_data("path/to/data.csv")

# 检测异常值
outliers = pm.detect_outliers("pH", method='zscore', threshold=3.5)
print(f"发现 {outliers.sum()} 个异常值")

# 分析缺失数据
missing = pm.analyze_missing_dates()
print(f"缺失 {missing['total_sin']} 天数据")
```

## 热重载功能
Pipeline 会在每次分析前自动重新加载 `Analisis/funciones_analisis.py`，无需重启 Python 进程即可看到最新修改。

## 注意事项
- 图表使用无界面后端（Agg），不会弹出窗口
- 每个变量独立生成图表并立即释放内存，避免内存泄漏
- pH 变量会自动启用物理界限检测（0-14 范围）
- 所有报告保存在 `resultados_analisis/[工厂名]_[时间戳]/` 目录

## 扩展功能（预留接口）

### 时间序列分析
**状态**: ✅ 已实现（默认关闭）

启用方法：
```python
analyzer.analyze_plant_data(file, do_time_series=True)
```

输出：每个变量的时间序列图和小时分布箱线图

### Drift 分析
**状态**: ⏳ 待实现

**已预留接口**：
- `PipelineManager.run_drift_analysis()`
- `AutoAnalyzer._analyze_drift()`

**集成步骤**（详见 `集成指南.md`）：
1. 在 `Analisis/funciones_analisis.py` 中实现 `analizar_drift()` 函数
2. 在 `pipeline_manager.py` 中取消注释调用代码
3. 运行测试：`python test_drift_integration.py`
4. 启用分析：`analyzer.analyze_plant_data(file, do_drift=True)`

**测试当前接口**：
```powershell
python test_drift_integration.py
```
