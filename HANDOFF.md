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

- **实验 46**：共享 NPZ 预处理缓存 `test/.cache/shared_preprocessing_v1`（8 个 train 块、25ms uint16 计数、只读）。
- **实验 47**：train 集 87,936 条单 trial 的 target-ID 拟合，两鼠各 43,968/43,968 = 100%，无泄露。
- **实验 48**：高容量无正则对照，未通过；其 `production_recipe` 开发结果（Task2 epoch5、Task3 epoch12）作为后续实验的固定比较器。
- **实验 49**：特征分位传输，未通过。
- **实验 50**：固定 DCT16 全字库 cosine 分类头。Task2 18.371881→17.860802（Δ−0.511），Task3 15.695588→15.771780（Δ+0.076），未通过。
- **实验 51**：分层 DCT-Ridge 图像先验 + 经验 Poisson + 无标签边际校准。Task2 Δ−0.667、Task3 Δ−1.582，双鼠与所有折为负，未通过。
- **实验 52**：低秩相关噪声高斯模型（log1p、类模板、C=EᵀE/(N−G)、秩 Task2=4/Task3=1、Woodbury 全字库评分、有限 repeat 预测协方差）。Task2 18.371881→16.697472（Δ−1.674），Task3 15.695588→13.368808（Δ−2.327），双鼠与所有折为负，未通过。结果在 `test/52_lowrank_gaussian/selection.json`。

**重要结论**：纯生成式/原型式全字库分类（51、52）在开发协议上系统性低于 v5 的判别式 MLP 配方约 1.5–2.3 分，且 Task3 单 trial 的 raw recall@32 只有 4.3%（v5 为 13.0%）。继续在“类模板 + 噪声模型”族内换参数不会带来大幅提升；下一步应转向判别式表征或候选生成的结构性改变，而不是再包装同一族模型。

- **实验 53**：三分辨率（10/25/50ms）判别式集成，逐行 z-score 等权融合。Task2 18.371881→19.909159（Δ+1.537278，recall@32 25.65%→34.05%），Task3 15.695588→16.439417（Δ+0.743829，recall@32 12.96%→18.63%）。双鼠与所有折均非负（Task2 +0.97/+2.10，3/3 折；Task3 +0.87/+0.61，2/2 折），25ms 视图逐位复现 48（最大差 0.0）。但 macro 未达 +3，advance=false。10ms 单视图明显更差（16.14/14.49）。
- **实验 54（探索性，未预注册）**：Task3 推理策略诊断（50ms、epoch9）。最优 `zlogits+pS` 仅 +1.119，且 mice1 −0.123、mice2 +1.691，不稳健；top32 内两两 directional 相似度均值 58.29，正确答案进入 top32 时 medoid 命中仅 4.81%。确认 argmax 错误时 OCF 均值 14.65、P90 22.60。
- **目标可行性（关键）**：Task2/3 的预测直接取自 GT 字图库，分类正确即 OCF=100。由 Task3 final 16.44 与 top1 1.76% 得 `final≈100p+14.65(1−p)`；要 40 分需 p≈30%，即 top1 再提升 17 倍。4 repeat 平均的 Task2 top1 仅 4.1%，按 SNR 平方根规律，单 trial 达 30% 需约 290 repeat，物理不可达。用户已确认继续 Task2/3 表征侧而非转向 Task1。
- **实验 55**：Sinkhorn 平衡软伪标签转导（50ms、base epoch Task2=7/Task3=9、2 轮、温度 0.05、20 次迭代、软权重 0.5、微调 2 epochs lr=3e-5）。Task2 20.346（相对 48 production +1.975，双鼠 +1.66/+2.29，3/3 折正；相对 50ms base +1.11）；Task3 16.117（+0.421，双鼠 +0.81/+0.04，仅 1/2 折正；相对 50ms base +0.26）。advance=false。
- **协议事实（重要）**：共享 cache 的 train 块 `group_ptr` 全为 1（10992 samples = 10992 trials），**没有 repeat 分组**；Task2 的 4-repeat 平均只能用 targets 分组，因此任何"无标签按 sample 分组"的转导在开发协议中无法复现 test 的 group_ptr 结构。50ms 特征维度为 mice1 670 / mice2 390（1340/780 属 25ms）。
- **Task2/3 累计最优**：Task2 20.346（+1.975，实验 55）；Task3 16.439（+0.744，实验 53 融合）。两者均未达 +3，Task3 差距最大。
- **方向决定**：用户确认 Task2/3 表征侧已穷尽后转向 **Task 1 上界诊断**（实验 56-60）。

## 6.1 Task 1 专项诊断（2026-11-26，只读，无新生产改动）

- **实验 56 字库上界**：NotoSansSC-VF size52 Regular 渲染 6763 字，全库完美检索 OCF 上界 **79.42**（51 个 heldout 行，P10 71.4 / P90 88.7）；用 GT 字图查询 + directional top1 = **80.10**（211 行），且 top1 在 211/211 行上就是 top32 内 OCF 最高的候选。**字库与检索都不是瓶颈。**
- **实验 57 连续重建**：v5 Task1 的 DCT32 回归系数 R²≈0.47（MSE 0.388 / 方差 0.733），预测系数 row-norm 仅为 GT 的 68%（18.58 vs 27.37），重建图对比度 57%（std 0.217 vs 0.378），前景 9.9% vs 20.0%。连续图 OCF 2.36、二值 OCF 11.10（heldout）；**训练字二值仅 15.19**，说明不是过拟合而是信号信息上限。尺度校准 gain=1.2 仅把二值 OCF 提到 13.15（+2.4）。
- **实验 58 结构上界**：对 211 个真实字图排除自身后，按 directional_multi 选最相似候选，**top1 错误候选 OCF = 42.67**，top500 内最优 48.31；全局平均字仅 4.20。**41.67×R²≈20，与实测 18.44 吻合**——40 分不需要 30% 分类准确率，只需要把连续重建 R² 提到约 0.9。
- **实验 59 时间分辨率**：Task1 pooled ridge 检索 OCF 10ms 16.03 < 25ms 16.79 < 50ms 16.93；**更细时间 bin 更差**，binned counts 的时间信息已饱和。
- **实验 60 多任务**：回归 + 0.1×2473 类分类共享 trunk，mice1 +0.23、mice2 −0.32（平均 −0.04），**无改善**。
- **Task1 结论**：字库、检索、时间分辨率、多任务均非杠杆；唯一杠杆是连续重建 R²，而它在 25/50ms counts + MLP/ridge 上稳定在 0.47–0.5。历史 07/17 DDPM、21 方向损失、24/26 CNN/attention、33 时序候选均未突破。

## 6.2 补充排除（实验 61-62）

- **实验 61 repeat 缩放**：Task1 检索 OCF 1→4 repeat 为 16.15→17.33→17.79→18.07，收益递减；按平方根外推到 test 的 20 repeat 仅约 +0.5。因此真实 Kaggle Task1 不会显著高于本地 18.44。
- **实验 62 Task3 连续重建**：50ms pooled ridge 预测 DCT32 后 directional 检索，macro OCF **15.43**，低于分类单视图 15.86；DCT **R²≈0.00**（−0.0008~0.0046）。单 trial 重建无法解释任何字形方差，分类已是最优利用方式。
- **实验 63 v6 固定协议配对评分**：v6 = v5 Task1 + 实验 53 三视图（10/25/50ms）等权融合 Task2/3，冻结轮次 7/5/7 与 7/12/9，整块唯一分配，无任何搜索；v5 三个 heldout 预测在同一脚本内重算并复现记录值。结果：Task1 **18.437062**（不变）、Task2 **20.964633**（v5 23.824141，Δ**−2.859508**）、Task3 **15.966224**（v5 16.759832，Δ**−0.793607**）；三任务等权平均 **18.455973**，比 v5 的 19.673678 低 **1.217705**。Task2 两鼠（23.363 / 18.566）与 Task3 八个鼠×日组全部低于 v5。**v6 不优于 v5，不接入生产、不提交 Kaggle。** 主因是 v5 的 Task2 还含三种子、编码分支与 reranker，Task3 含原型连续分支，单种子三视图融合无法补偿。结果与脚本在 `test/63_v6_fixed/`。
- **合并结论**：三任务共 15 个以上结构性方向全部排除（生成式、低秩噪声、集成、转导、结构重排、时间分辨率、多任务、连续检索、DDPM 历史、CNN/attention、对比学习、方向损失、噪声加权、dropout）。唯一稳定的正增益是集成与转导的部分叠加（Task2 +1.97、Task3 +0.74）。

## 6.3 字图视觉族与脑电共性（实验 64，探索性）

- **问题**：同族字图的脑电是否更相似，族级信息是否可用于粗到细解码。脚本与结果在 `test/64_glyph_cluster_commonality/`。
- **聚类**：2748 张公开字图用 `directional_multi` 方向场嵌入做 k-means；k=8/16/32/64/128 时两次随机初始化的 ARI 为 0.51/0.47/0.40/0.30/0.27，与 DCT16 聚类的 ARI 为 0.35/0.35/0.30/0.25/0.24。**视觉空间是连续的，不存在稳定的离散族**。
- **脑电共性（8 个 train session 全部为正）**：族内平均余弦相似度减族间为 +0.008~+0.025（k 越小效应越小），200 次标签置换下 z=+26.7~+84.2；图像 RDM 与神经 RDM 的 Spearman（RSA）为 +0.040~+0.076。跨 session 神经 RDM 相关仅 mice1 0.172 / mice2 0.145。**共性存在且统计稳健，但效应量小，且跨日几何不稳定。**
- **族级可解码性**（用 `test/63_v6_fixed` 已训练的固定 split 50ms 模型，公开 heldout 已被复用，仅探索）：Task2 k=8 族 top1/top3 = 32.3%/68.5%，k=32 = 17.8%/38.3%；Task3 k=8 = 25.3%/58.9%，k=32 = 11.3%/26.9%；两鼠几乎相同。对应 identity top1 仅 4.5%/1.2%。
- **族与 identity 候选并不重合**：top32 候选里真族占比 Task2 k=8 20.2%、k=32 7.9%，Task3 17.6%/6.1%（k=32 随机占比约 3.6%）。族先验会实际改变排序，不是重复现有分数。
- **含义**：粗族信号约为 identity 的 4~20 倍强，且 top32 尚未被真族占满，所以“族后验 + 族内 identity”是尚未测试的结构性杠杆；但聚类不稳、跨日几何相关仅约 0.15，预期增益有限，且实验 54 的结构重排只 +1.1。下一步须用开发折做配对 OCF 检验，不得用本节的固定 heldout 选择配置。
- **实验 65 两阶段配对测试**（`test/65_hierarchical_cluster_decoding/`）：开发折为 Task2 留 session1/2/3、Task3 种子 20261126/20261127；同一个 50ms 单种子模型，只改最终评分：base=logits，soft=`z(logits)+λ·z(log 族后验)`（k∈{8,16,32}、λ∈{0.25,0.5,1.0}），hard=只在 top-K 族内分配。官方 OCF（整块匈牙利后取 256 张开发图）结果：**Task2 全部 11 个臂 macro 为负**，最好 `soft_k8_λ0.25` −0.066（mouse_min −1.21）；**Task3 最好 `soft_k32_λ0.25` +0.238**（两折 +0.314/+0.163，但 mouse_min −0.271），`soft_k16_λ0.25` +0.235。hard 变体 Task2 一律 −0.78~−1.09。**advance=false：族后验无法转化为 OCF 增益。** 与实验 64 的“族可解码”不矛盾——族后验不足以在 identity 层重排，强制族约束还会丢掉正确的跨族候选。

## 7. 建议的下一项可证伪工作

**已完成的系统性排除（2026-11-26）**：实验 48-60 与历史 21/26/35/39/40-45 覆盖了高容量、正则、分位传输、DCT 全字库头、分层 Poisson、低秩相关噪声、多分辨率集成、转导、结构推理、时间分辨率、多任务、方向损失、CNN/attention、对比学习。三任务的最终瓶颈一致指向**神经信号单次呈现的信息量**：Task3 单 trial top1 1.76%，Task1 连续重建 R²≈0.47。综合 40 分在当前数据与合法方法下不可达。

**唯一仍有真实收益的行动**：把已通过双鼠多折稳定性检验的部分增益接入生产，用真实 Kaggle 提交验证：
1. Task2：实验 55（50ms + Sinkhorn 转导，相对 48 production +1.975，双鼠 +1.66/+2.29，3/3 折正）。
2. Task3：实验 53（三视图融合，+0.744，双鼠 +0.87/+0.61，2/2 折正）。
3. Task1：保持 v5（或仅加固定 gain=1.2 的系数校准，二值 OCF +2.4，但检索后增益未验证）。
接入前必须在固定公开 heldout（session4 / seed42）与 v5 完整配方做 paired 对比，再全量重训、打包、提交；失败可回退 v5 包（SHA256 见第 2 节）。

**修正（实验 63）**：上述 2 的三视图融合已按该要求做过固定协议 paired 对比，Task2/3 均低于完整 v5（−2.86 / −0.79），所以“直接接入生产”不再成立。1 尚未做过同样的 paired 对比；其 +1.975 也是相对实验 48 开发配方，而不是相对含编码分支与 reranker 的 v5。

## 8. Git 与当前运行状态

- 最近结果提交：**`1918cee test: score v6 multiresolution Task2/3 on the fixed protocol (worse than v5)`**，已通过 SSH 推送 `github.com:ZZheng96/siclink_MVR.git` 的 `main`。
- 实验 63：`1918cee`；实验 62：`b77b11a`；实验 61：`7f61925`；实验 60：`2f5676a`；实验 59：`ecae7fb`；实验 58：`a85b915`；实验 57：`4be8f17`；实验 56：`a57df61`；实验 55：`bb65d8d` / `dbd6c11`；实验 54：`7a1e4bf`；实验 53：`f1baae7` / `a591bcc`。
- 本交接文档创建前工作树干净，实验 44、45 的后台 agent 均已结束。没有待处理的模型训练或自动 Kaggle 提交。
- 新对话开始后先运行 `git status --short`、`git log -3 --oneline`，并检查是否有新增后台任务；以磁盘实际状态为准。
- 大模型 checkpoint、预测 NPZ、缓存通常被 `.gitignore` 排除；轻量 preregistration、selection、results、manifest、training_summary 等按白名单入库。不要为提交而打包大型缓存。
