时尚商品数据资产工作台。现有目录、标签汇总、沙箱试用和治理账本支持本地资产分析。

环境：Linux，Python 3.11.2 或更新的兼容版本。所有依赖来自 Python 标准库，无需安装第三方包。

测试：python3 -m unittest discover -s tests/data_asset_trials -p 'test_*.py'
构建：python3 -m compileall -q fashion_data_foundry tests/data_asset_trials
验收：python3 -m fashion_data_foundry.workbench.cli fixtures/demo.json --database /tmp/fashion_data_foundry-acceptance.sqlite3
所有输入均为本地合成数据，运行不需要网络。

目录：fashion_data_foundry 保存业务处理与存储模块；tests/data_asset_trials 保存业务回归；fixtures/demo.json 为原有业务样例，fixtures/analysis 为经营分析资料及字段说明。

原有验收命令每次使用新的 SQLite 路径，避免复用上次状态。新增输入完整性检查：
python3 -m unittest discover -s tests/analysis_inputs -v

容器运行可使用已经预装的 ccusr/prepared:3f007a95fcee-4441968e5ff6 镜像；将当前仓库只读挂载至 /source，在容器内复制到 /tmp/product 后运行上述命令。依赖准备和正式运行分离，运行过程中不需要网络访问。
