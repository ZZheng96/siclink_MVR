# SiClink 新对话交接

更新日期：2026-11-26。工作区：`D:/WORK/siclink`，Windows / PowerShell 5，Python：`.venv/Scripts/python.exe`。

## 1. 当前目标与硬边界

- 按 `ref/siclink_research_and_solution.md` 持续改进三任务视觉重建，生产代码维护在 `src/`，评估脚本和编号实验放在 `test/`。
- 长期目标是提高真实 Kaggle 成绩；**不得把本地 OCF 当成 Kaggle 分数，也不得宣称已达到榜首**。
- 只使用公开 train 数据及合法字体/图像先验。禁止隐藏测试 GT、下标对应、人工 ID 对齐等作弊路线。
- 新实验必须先冻结协议、训练/验证隔离、选择集和通过门槛；tune 锁定后才可评 heldout。已复用的 heldout 不是 pristine 独立确认。
- final 与 oracle、recall、assignment 前后指标必须分开。oracle 使用公开 GT，只作诊断，不能进入提交。
- 多鼠、多折稳定增益才允许接入生产。单次平均 OCF 提高不能覆盖某只鼠的明确退化。
- 长任务使用子 agent，主线审计其 diff、测试和结果；及时提交 Git，并通过 SSH 推送 `origin/main`。

## 2. 当前可提交版本与真实成绩

**生产及下一次提交基线仍是 v5：`mlp_encoding_task2_rerank_task3_soft_v5`。**

| 项目 | 当前事实 |
|---|---|
| v5 Kaggle Public OCF | **21.80563**，用户于 2026-11-25 回报，未经助手在线复核 |
| v3 Kaggle Public OCF | 21.09590 |
| v5 相对 v3 | **+0.70973** |
| 本地榜首快照 | 54.01486；不等于 Private 榜 |
| v5 距榜首快照 | **32.20923** |
| v5 提交包 | `test/90_submission/artifacts/mlp_encoding_task2_rerank_task3_soft_v5/kaggle_submission.zip` |
| 压缩包 SHA256 | `2311a23d3423fc77ab6814d121a06df280dfc480e59b695a47d4d05e406e03c9` |

真实分数和用户回报记录位于 `test/90_submission/kaggle_scores.json`。v5 同时改变 Task 2 reranker 和 Task 3 soft 融合，**不能把 +0.70973 单独归因于某任务**。不要用两个提交点拟合本地→Kaggle 的线性映射或固定比例。

## 3. 代码与评估入口

- `src/final.py`：完整训练、公开评估、全量重训和 Kaggle 打包；公共 `train_task` / `predict_task` 供评估复用。
- `src/final_config.json`：v5 配置。Task 2 为 MLP+encoding+OOF CNN reranker；Task 3 为 MLP+encoding+softmax DCT continuous 等权 row-z 融合。
- `test/evaluate.py`：评估已有 `predictions.npz`，使用官方 OCF。
- `test/evaluate_final.py`：固定公开 split 下端到端训练、预测和 final/oracle OCF，显式记录 reranker 来源和 SHA256。
- `test/collect_results.py`：按任务、协议、鼠/日分组汇总 final 与 oracle，区分未评估 heldout 和历史结果；不要把跨协议最佳值拼成虚拟提交分。
- `test/results_summary.json`：当前机器可读汇总，含 `comparable_to_hidden_leaderboard=false` 和协议警告。
- `test/.cache/evaluate_final_task23_v1/`：本地公开评估模型缓存，通常不入 Git；缓存不应被误当作 v5 全量测试模型。

常用命令：

```powershell
.\.venv\Scripts\python.exe -B -m py_compile src\final.py test\evaluate_final.py
.\.venv\Scripts\python.exe -B test\collect_results.py
.\.venv\Scripts\python.exe -B -u src\final.py --device cuda --reranker-run-name mlp_encoding_task2_rerank_task3_soft_v5
```

完整评估与打包命令参数先读各自 `argparse`，不要凭记忆运行。生产改动后需回归合成/公开评估，并重新打包、校验条目和哈希；不得直接覆盖历史已提交包。

## 4. v5 固定公开评估锚点

以下为本地 OCF，不是 Kaggle 分数；heldout 已被历史实验复用。

| 任务与协议 | Tune final OCF | Heldout final OCF | Heldout oracle@32 |
|---|---:|---:|---:|
| Task 1：留字、20 repeats 聚合 | 16.81193 | 18.43706 | 33.01849 |
| Task 2：留 session4、4 repeats | 25.007633 | 23.824141 | 61.217716 |
| Task 3：留同字其他 trial、单 trial | 17.555726 | 16.759832 | 40.493480 |

v5 固定公开 heldout 三任务等权平均为 **19.67368**，只用于同协议回归。Task 2 的 tune 同样来自留 session4，不能解释为真正 session5 盲测。

通用 split：公开 `make_split(..., seed=42)`；Task 2/3 图片子集常用 `default_rng(123).permutation(2748)`，tune `[:256]`、heldout `[256:768]`。使用前以具体实验 `preregistration.json` 为准，不要跨任务假设相同语义。

## 5. 已知瓶颈与证据

1. **Task 3 是优先瓶颈**：单 trial heldout final OCF 16.759832，身份准确率很低，oracle@32 为 40.493480。候选召回和候选内排序都未解决，且 oracle 不是可直接获得的成绩。
2. **Task 2 跨日不稳定**：实验 34 的嵌套留日中，reranker heldout OCF 从 19.291846 降至 18.779709，覆盖后的 top1 条件准确率也下降。不要只凭 session4 成功宣称跨日有效。
3. **Task 1 是神经解码与字库/字体联合问题**：留字 final 18.43706、oracle@32 33.01849；尚不能把差距全归于字体或神经端。
4. **mice2 是反复出现的薄弱层**：所有候选必须按鼠报告；不能仅由 78 units 相对 mice1 的 134 units 推断因果，也不能事后为 mice2 搜权重。
5. **生产打包不是当前主要损失来源**：现有 OCF 使用官方 PNG 路径，分数差距主要应从信号、字形结构和排序解释。

## 6. 已否定或未通过的近期路线

- **实验 39**：Task 3 噪声加权只在单一 holdout 小幅提升，之后所有候选必须跨折复核。
- **实验 40**：日条件 CNN 读出四折平均 final OCF 比原 CNN 低 0.807139，recall 下降；不集成。
- **实验 41**：三种子 CNN 相对单种子 CNN 均值提高，但不是 v5 对照，不能据此接生产。
- **实验 42**：全库 CNN 融合提高两鼠 OCF，却使 mice2 recall@32 从 182/512 降至 174/512；不集成。
- **实验 43**：保留原 top32 的 CNN reranker 使 mice1 OCF 提高，但 mice2 下降；不集成。
- **实验 44**：Task 3 双候选像素混合，锁定 `k=2, temperature=0.5, threshold=0.5`。heldout 总 OCF 16.759832→16.819239，两鼠都微升，但 8 个鼠×日组仅 4 组正，未达至少 5 组门槛。结果在 `test/44_task3_score_mixtures/results.json`。
- **实验 45**：固定计数 dropout（每行 0.5 概率原样，否则 `Binomial(x,0.5)`）。tune 中 mice1 改善，但 mice2 recall@32 188/1024→183/1024、OCF 下降；按预注册停止，heldout 未评估。结果在 `test/45_task3_repeat_dropout/results.json`。**不要再在同一协议中事后搜索 dropout p。**

实验 22 的 repeat 均值增广已经失败，不要重复。实验 40–45 均未修改 v5 生产提交，也没有产生新的 Kaggle 成绩。

## 7. 建议的下一项可证伪工作

优先顺序不是必须机械执行，但不能把失败路线换参数重复包装：

1. **Task 1 无训练“真字图→渲染字库”上界**：用公开合法真字图与字体库分离字库/字体损失、检索上限和神经解码损失；真字图上界低则先改字库匹配/连续重建，上界高才训练神经结构学生。
2. **Task 3 新增正确候选的表征学习**：需超出固定 dropout 和双候选像素平均；优先结构/时序表征或严格 paired 的新候选生成模型，同时检查 recall、oracle 和最终 OCF，不能只提升候选覆盖却降低分配后成绩。
3. **Task 2 简化跨日校准**：比较冻结 MLP+ridge、线性/逐候选校准与 CNN reranker，采用完整 nested 留日，不继续增加日条件读出。
4. 只有通过预注册多鼠、多折门槛的**单个冻结改动**才接入 v5，完整重训、打包并记录与 v5 的本地 paired delta；真实提交后记录 Kaggle paired delta，失败可回退已保存的 v5 包。

不要用同一历史 heldout 反复搜索参数来制造“确认增益”；若必须探索，明确标为 exploratory，并设计新的、合法的重复折或确认协议。

## 8. Git 与当前运行状态

- 最近结果提交：**`9853af3 test: record score-mixture reconstruction gate failure`**，已通过 SSH 推送 `github.com:ZZheng96/siclink_MVR.git` 的 `main`。
- 此前实验 45 提交：`64afe19`；评估复盘文档提交：`705403a`；v5 分数记录：`8e5694a`。
- 本交接文档创建前工作树干净，实验 44、45 的后台 agent 均已结束。没有待处理的模型训练或自动 Kaggle 提交。
- 新对话开始后先运行 `git status --short`、`git log -3 --oneline`，并检查是否有新增后台任务；以磁盘实际状态为准。
- 大模型 checkpoint、预测 NPZ、缓存通常被 `.gitignore` 排除；轻量 preregistration、selection、results、manifest、training_summary 等按白名单入库。不要为提交而打包大型缓存。
