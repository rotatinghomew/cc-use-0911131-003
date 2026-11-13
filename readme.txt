时尚商品数据资产工作台。现有目录、标签汇总、沙箱试用和治理账本支持本地资产分析。

环境：Linux，Python 3.11.2 或更新的兼容版本。所有依赖来自 Python 标准库，无需安装第三方包。

测试：python3 -m unittest discover -s tests/data_asset_trials -p 'test_*.py'
构建：python3 -m compileall -q fashion_data_foundry tests/data_asset_trials
验收：python3 -m fashion_data_foundry.workbench.cli fixtures/demo.json --database /tmp/fashion_data_foundry-acceptance.sqlite3
所有输入均为本地合成数据，运行不需要网络。

目录：fashion_data_foundry 保存业务处理与存储模块；tests/data_asset_trials 保存业务回归；fixtures/demo.json 为原有业务样例，fixtures/analysis 为经营分析资料及字段说明。

原有验收命令每次使用新的 SQLite 路径，避免复用上次状态。新增输入完整性检查：
python3 -m unittest discover -s tests/analysis_inputs -v

风格标签模型选型与人工复核队列（fashion_data_foundry/model_review）：
- 回归：python3 -m unittest discover -s tests/model_review_trials -p 'test_*.py'
- 正常发布：python3 -m fashion_data_foundry.model_review --database /tmp/model-review.sqlite3 --input fixtures/analysis/input.json --output /tmp/model-review-report.json
- 数据不足状态：将 --input 换为 fixtures/analysis/insufficient_input.json，输出 status=insufficient_data 并带 blockers，与正常发布区分。
- 隔离模式（python3 -I）下需显式设置仓库根目录，详见 tests/model_review_trials/test_cli.py。
划分以 style_family 为单位保证 train/validation/test 家族不相交；final_cutoff 之后的标注与撤回资产不参与选型；候选含朴素先验基线、加权逻辑回归、高斯朴素贝叶斯，预测可拒识并报告分类别指标与覆盖率；review_queue 不超过 review_budget，且不重复推送已确认或已撤回条目。发布报告绑定 asset_revision、label_version、split_manifest 与 feature_manifest，冻结 model_bundle 支持不依赖当前标签的历史复现。

容器运行可使用已经预装的 ccusr/prepared:3f007a95fcee-4441968e5ff6 镜像；将当前仓库只读挂载至 /source，在容器内复制到 /tmp/product 后运行上述命令。依赖准备和正式运行分离，运行过程中不需要网络访问。
