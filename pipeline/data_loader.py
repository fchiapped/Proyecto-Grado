"""proyecto.new_pipeline.data_loader
提供统一的数据加载与准备接口：

- 自动读取 CSV(s)
- 自动发现并导入 `Pre-Procesamiento/funciones_pre_procesamiento.py` 中的预处理函数
- 自动对齐时间索引（重采样/填充选项）和选择变量
- 可选地生成并附加缺失/空洞（huecos）标志，便于后续 drift/outlier 分析

设计契约（简要）：
- 输入: CSV 文件路径或 pandas.DataFrame
- 输出: 标准化的 pandas.DataFrame（包含 `date_time` 列或索引为 DatetimeIndex）

实现要点：为避免包名/目录中包含连字符等不可导入字符，模块以文件路径动态加载 `funciones_pre_procesamiento.py`。
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd


def _find_preproc_file(start_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
	"""尝试在 repo 中查找 `funciones_pre_procesamiento.py` 的文件路径。

	搜索策略：从 start_path（或当前文件所在目录）向上最多 4 层，
	在每一层查找常见目录名（'Pre-Procesamiento', 'Pre_Procesamiento', 'PreProcesamiento', 'Pre-Procesamiento')
	下的文件。
	"""
	start = Path(start_path) if start_path else Path(__file__).resolve().parent
	candidates = [
		"Pre-Procesamiento",
		"Pre_Procesamiento",
		"PreProcesamiento",
		"Pre-Procesamiento",
		"Preprocesamiento",
	]

	for level in range(5):
		for cand in candidates:
			p = start / cand / "funciones_pre_procesamiento.py"
			if p.exists():
				return p.resolve()
		start = start.parent
	return None


def _load_preproc_module(path: Optional[Union[str, Path]] = None):
	"""动态导入预处理脚本并返回 module 对象。"""
	p = Path(path) if path else _find_preproc_file()
	if not p or not p.exists():
		raise FileNotFoundError("找不到 'funciones_pre_procesamiento.py'，请检查项目中的 Pre-Procesamiento 目录。")

	spec = importlib.util.spec_from_file_location("project_preproc", str(p))
	module = importlib.util.module_from_spec(spec)
	loader = spec.loader
	assert loader is not None
	loader.exec_module(module)
	return module


def load_csvs(paths: Union[str, Path, Iterable[Union[str, Path]]], dt_col: str = "date_time") -> pd.DataFrame:
	"""读取单个或多个 CSV 文件并合并为一个 DataFrame。

	自动尝试将 dt_col 转换为 datetime 并排序。
	"""
	if isinstance(paths, (str, Path)):
		paths = [paths]

	dfs = []
	for p in paths:
		df = pd.read_csv(p)
		if dt_col in df.columns:
			df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
		dfs.append(df)
	out = pd.concat(dfs, ignore_index=True)
	if dt_col in out.columns:
		out = out.sort_values(dt_col).reset_index(drop=True)
	return out


def prepare_dataframe(
	df: Union[pd.DataFrame, str, Path, Iterable[Union[str, Path]]],
	columns: Optional[List[str]] = None,
	dt_col: str = "date_time",
	freq: str = "T",
	resample_method: str = "mean",  # or 'first','median'
	preproc_path: Optional[Union[str, Path]] = None,
	attach_huecos: bool = True,
	huecos_params: Optional[dict] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
	"""主入口：加载并准备数据，返回 (df_prepared, flags_df)

	- df 可以是 DataFrame 或 CSV 路径 或 路径列表
	- columns: 可选，用于选择要保留的变量
	- freq: 目标时间频率，例如 'T'（按分钟）
	- attach_huecos: 是否调用预处理模块中的 `generar_flags_huecos` 并返回 flags
	"""
	# 如果给的是路径，先加载
	if isinstance(df, (str, Path)) or (hasattr(df, "__iter__") and not isinstance(df, pd.DataFrame)):
		df = load_csvs(df, dt_col=dt_col)

	if dt_col not in df.columns:
		# 尝试一些常见替代名
		for alt in ["timestamp", "Fecha", "fecha", "date"]:
			if alt in df.columns:
				df = df.rename(columns={alt: dt_col})
				break
	if dt_col not in df.columns:
		raise ValueError(f"DataFrame 必须包含时间列（{dt_col} 或 常见别名），当前列: {df.columns.tolist()}")

	df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
	df = df.dropna(subset=[dt_col]).copy()

	# 选择变量
	if columns:
		keep = [dt_col] + [c for c in columns if c in df.columns]
		df = df[keep].copy()

	# 将时间设为索引并重采样到目标频率
	df = df.sort_values(dt_col).set_index(dt_col)

	# 选择重采样策略
	if resample_method == "mean":
		df_res = df.resample(freq).mean()
	elif resample_method == "median":
		df_res = df.resample(freq).median()
	elif resample_method == "first":
		df_res = df.resample(freq).first()
	else:
		# 允许传入自定义函数名或可调用
		if callable(resample_method):
			df_res = df.resample(freq).apply(resample_method)
		else:
			raise ValueError("不支持的 resample_method")

	# 载入预处理模块（按需）
	flags_df = None
	if attach_huecos:
		try:
			pre = _load_preproc_module(preproc_path)
		except FileNotFoundError:
			pre = None

		if pre and hasattr(pre, "generar_flags_huecos"):
			params = huecos_params or {}
			# generar_flags_huecos 期待一个有列 (date_time + columnas) 的 DataFrame
			# 我们传入重采样之前的原始（分钟级别）或重采样后的（取决于 freq）
			try:
				cols = list(df_res.columns)
				if not cols:
					cols = [c for c in df.columns if c != dt_col]
				flags_df = pre.generar_flags_huecos(
					df=df.reset_index().rename(columns={dt_col: "date_time"}),
					columnas=cols,
					dt_col="date_time",
					freq=freq,
					attach_to_df=False,
					**params,
				)
			except Exception as e:
				# 不阻塞主流程：记录并继续
				print(f"警告：生成 huecos 标志失败: {e}")

	# 将索引恢复为列，返回
	df_out = df_res.reset_index().rename(columns={"index": dt_col})
	# 确保保持 date_time 列名
	if df_out.index.name == dt_col:
		df_out = df_out.reset_index()

	return df_out, flags_df


if __name__ == "__main__":
	# 简单示例（仅在作为脚本执行时运行）
	here = Path(__file__).resolve().parent
	sample = here.parent / "df_procesados" / "df_planta_1_timeseries.csv"
	if sample.exists():
		df, flags = prepare_dataframe(sample, freq="T", attach_huecos=True)
		print(df.head())
		if flags is not None:
			print(flags.head())
