实时监控 pipeline（方案 A - 文件追加/轮询）

运行说明（在项目根目录下）：

1. 安装依赖（建议在虚拟环境中）：

```powershell
pip install -r pipeline/requirements_pipeline.txt
```

2. 启动监控：

```powershell
python pipeline/runner.py
```

3. 配置：编辑 `pipeline/config.yaml`，修改 `plants` 中的 csv_path、datetime_col、columns 和 flags_path。

说明：
- runner 会每隔 `poll_interval_seconds` 轮询 CSV 文件，读取比上次时间戳晚的新增行，运行检测器并将检测到的异常追加到 `flags_path`。
- 初版使用文件轮询来实现最低风险的实时输入方案，后续可以切换到 `watchdog` 监听或消息队列。