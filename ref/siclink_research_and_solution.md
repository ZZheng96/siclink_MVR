# SiClink 小鼠视觉重建赛题：科学机理、相关工作、复现与可行方案

- 调研与复现日期：2026-11-24
- 赛题：[SiClink Mouse Visual Reconstruction BCI Competition](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge)
- 本地复现环境：`D:\WORK\siclink\.venv\Scripts\python.exe`
- 证据标记：**[小鼠-spike]**、**[其他物种/脑区-spike]**、**[钙成像]**、**[人-fMRI]**、**[方法迁移]**、**[本项目假设]**。

> 不同物种、脑区、记录模态、刺激与指标之间不能直接比较数值。本文将论文已证实的结论与针对本赛题的设计推论分开；没有把 fMRI、钙成像或视网膜结果当作小鼠 V1 spike 汉字重建的直接证据。

## 截至 2026-11-25 的复盘：用两次提交校准评价，而非校准一个映射公式

**当前可提交的最佳版本是 v5（`mlp_encoding_task2_rerank_task3_soft_v5`），用户回报的 Kaggle Public OCF 为 21.80563；v3 为 21.09590，真实增量为 +0.70973。** 两个提交与压缩包哈希见[榜单记录](../test/90_submission/kaggle_scores.json)。榜首本地快照为 **54.01486**，v5 距其 **32.20923**；快照与 Private 榜不能混用。v5 的结果支持继续采用其完整训练/预测链路，但不证明实验 42、43 的 CNN 融合有效：它们均未通过双鼠门槛，没有进入 v5。

### 2.1 两次实际提交能校准什么

| 版本 | Kaggle Public OCF | 本地证据与可解释范围 |
|---|---:|---|
| v3 `mlp_encoding_font_directional_v3` | 21.09590 | 早期 Task 1 字库检索、Task 2 与 Task 3 基线；旧本地子集与后续生产评估不完全相同。 |
| v5 `mlp_encoding_task2_rerank_task3_soft_v5` | **21.80563** | 加入 Task 2 OOF reranker 与 Task 3 soft 融合。由于同时改变两个任务，**不能把 +0.70973 归因于其中某一个任务**。 |

两次总分是不同模型上的两个观测，不是同一任务的配对成绩，也没有公开榜逐任务分数。因此只确立 **v5 > v3** 这一版本选择，拒绝以下做法：

1. 用两个点拟合 `Kaggle = a + b × local`，或用 `21.80563 / 19.67368` 缩放未来的本地增益。
2. 把留 session4 的本地 Task 2 OCF 当成真正未见 session5 的泛化估计；它只是公开数据中的留日代理。
3. 把不同历史实验的 tune/heldout 或不同分配实现的最佳值拼成一个“虚拟提交分”。v5 的统一公开 heldout 等权平均为 `(18.43706 + 23.824141 + 16.759832) / 3 = 19.67368`，只作为该固定协议下的回归锚点，不是 Kaggle 预测。

**此后采用配对校准，而不是分数外推。** 每个候选固定代码、配置、完整数据训练和提交构建方式，与 v5 在同一公开折上比较；只有所有相关鼠和预注册折的方向一致，才消耗一次提交。提交后记录 `(local paired Δ, Kaggle paired Δ)`，按改动类型积累证据。至少要有多个独立改动的结果，才可讨论局部相关性；仍不把其当成独立同分布的统计标定。每天的提交机会用于检验**冻结版本**，不用于搜索隐藏测试标签或调参。

### 2.2 统一评估协议：先消除评价口径造成的假瓶颈

[evaluate_final.py](../test/evaluate_final.py) 已复用生产函数完成公开训练、预测和官方 OCF 评估；reranker 的模型来源与 SHA256 显式记录。其固定公开评估如下，heldout 已被历史实验复用，**不是 pristine 盲测**。

| 任务与留法 | tune final OCF | heldout final OCF | heldout 样本/组数 | heldout oracle@32 OCF |
|---|---:|---:|---:|---:|
| Task 1：留字，20 repeats 聚合 | 16.81193（128 / 16） | 18.43706（1688 / 8） | 1688 / 8 鼠×日 | 33.01849 |
| Task 2：留 session4，源日训练，4 repeats | 25.007633（512 / 2） | 23.824141（1024 / 2） | 1024 / 2 鼠 | 61.217716 |
| Task 3：留同字其他 trial，单 trial | 17.555726（2048 / 8） | 16.759832（4096 / 8） | 4096 / 8 鼠×日 | 40.493480 |

这里的 `n` 是评分样本，不是独立字数：Task 3 的一个字在 8 个鼠×日组分别出现，相关记录不能当作独立重复。Task 1 必须保持留字；Task 2 是整日留出，训练日覆盖 2748 字，不应错误套用 Task 1 的留字数量。oracle 使用公开标签，只用于分解误差，绝不是可提交预测或最终候选成绩。

统一记录以下四类量，且比较必须在同一折、同一鼠、同一 full-gallery assignment 下配对：

- **coverage**：正确身份是否进入 top-k；`oracle@k` 只表示候选中结构最优者的上限，两者不可互相替代。
- **ranking**：候选已覆盖时的首位选择、身份准确率及错误候选的 OCF 分布。
- **assignment**：top-1 后唯一分配前、后 OCF；不能仅根据分配后的结果推断分类器学习了多少。
- **transfer/noise**：按鼠、完整留出日、重复数分层；模型选择和阈值只用 tune，冻结后才在声明复用历史的确认折运行一次。

当前 [results_summary.json](../test/results_summary.json) 已按任务、协议、鼠/日分组，并分离 final 与 oracle。它是可复现记录，不是一个跨协议可比较的排行榜；对尚无 heldout 的候选保留 `null`，不以 tune 结果补齐。

### 2.3 瓶颈定位：哪些已有证据支持，哪些仍只是假设

**结论 A：主要瓶颈不是 PNG 打包或阈值，而是神经证据对细粒度字形的分辨不足；其后才是排序与全局分配。** Task 3 单 trial 的公开 heldout final OCF 仅 16.759832，而 oracle@32 为 40.493480，说明前 32 名之外与之内都存在大量未实现的上限。此前相同评估族的 v5 对照中，recall@32 约 18%–20%，final 正确身份约 1.7%–2.0%；不同子集不可把这些比例当成同一个点估计。候选中错误字形仍能取得非零 OCF，因此“字形接近”不能替代“身份正确”。

**结论 B：Task 2 同样受候选生成与跨日泛化限制，不能只优化候选内排序。** 在生产 v5 的留 session4 heldout 上，final OCF 为 23.824141、oracle@32 为 61.217716；这巨大的差值是诊断上限差，不是可直接取得的分数。更关键的是，嵌套留日复核中，已选 reranker 的 heldout OCF 从 19.291846 降至 18.779709（−0.512137），正确身份的原 top32 覆盖不变，而覆盖后的 top1 条件准确率从 12.4655% 降至 10.5517%（[实验 34](../test/34_task2_nested_rerank/results.json)）。所以 reranker 的训练日/目标日匹配、outer-fold 样本量与 v5 全量 OOF 设置不同，不能把某一折的收益当成跨日稳定收益。

**结论 C：小型 CNN 融合的总体均值增益掩盖了 mice2 的退化，当前不支持继续堆叠分数。** 实验 41 的三种子正向 CNN 相对单种子在四折平均 final OCF 上提高 0.626817，且两鼠均值均正；但它是**单种子 CNN 对照**，不是生产 v5 对照。与 v5 直接配对时，实验 42 的固定全库融合虽然两鼠 OCF 都正，mice2 recall@32 却从 182/512 降至 174/512；实验 43 保留原 top32 集合后，mice2 final OCF 仍下降 0.513911。两者均未通过预注册门槛。由此只能否定这两个融合实现，不能证明“正向编码完全无效”，也不能通过在同一折搜索新权重来挽救结论。

**结论 D：把跨日变化写成每单元×时间的日读出，未得到支持。** 实验 40 的日条件 CNN 在四折 heldout 平均 final OCF 上比原 CNN 低 0.807139，仅 1/4 折为正，两鼠 recall@32 都下降。其完整 Poisson NLL 虽在宏平均略低于常数率，却只有 6/8 鼠×折胜过常数率；整体改善不等于辨字能力改善。因此不再为无辨别证据的 Poisson NLL 小增益增加模型复杂度。

**结论 E：Task 1 是“未见字解码 + 字库/字体误差”的联合问题，尚不能断言只是字体问题。** 当前 heldout final OCF 为 18.43706，oracle@32 为 33.01849，说明即使在字库候选中也存在排序损失；但 oracle 也受候选生成和重渲染字形限制。小样本候选诊断中，Task 1 的 oracle@32/128 约为 35.95/44.74，只能作为路线筛选，不是大样本泛化估计。文字身份或相近字在渲染字库中不对应测试字体时，检索即使身份正确也可能损失轮廓分；反之，没有公开真字图→字库的上界评估，就不能把差距全部归于神经端。

**结论 F：mice2 是稳健的薄弱层，但不能用 mice1 选择超参后宣称为 mice2 定制。** Task 3 的多个公开子集与实验 42、43 均显示 mice2 的召回/OCF 更弱。这支持按鼠报告、分层训练与方差控制；不支持仅凭一次留出结果对 mice2 选择不同权重。mice2 的 78 units 相对 mice1 的 134 units 是合理的容量/信噪比假设，仍需重复折叠和打乱输入对照验证，不能由单元数直接推出因果。

### 2.4 改进顺序：每项先回答一个可证伪问题

| 优先级 | 最小改动与要分离的原因 | 冻结评估与停止条件 |
|---|---|---|
| **1：Task 3 候选生成与噪声鲁棒性** | 在保持 v5 的 MLP/ridge/连续图像融合不变的条件下，训练一个仅使用训练 trial 的噪声条件化评分器；以 repeat dropout 或训练残差重采样匹配单 trial 不确定性，和等预算无增强对照比较。重点是新增正确候选，而非再加一层 reranker。 | 预先固定 2–3 个 repeat 条件折；按鼠报告 recall@32、oracle@32、最终 OCF 与赋值前后差。两鼠 recall@32 不升则停止；oracle 上升而 final 不升才进入排序诊断。不把均值 trial 当测试输入。 |
| **2：Task 2 跨日 reranker 的简化与样本效率** | 比较冻结 MLP+ridge、全局线性/逐候选校准与当前 CNN reranker；训练严格 nested 留日，检查只用 1 个内层源日是否导致 OOF 分布与全量 v5 不匹配。优先更稳的校准，不再增加日条件读出。 | 每个完整留出日都训练与评分；公开标签仅训练折可用。至少多数留日折与两鼠 final OCF 同向、recall@32 不降；只有单独 session4 成功不算跨日复核通过。 |
| **3：Task 1 无训练的“真字图→字库”上界** | 使用公开真字图与合法字体渲染库，先测同字不同字体的 OCF、top-k 覆盖与结构近邻错误。随后才训练 spike→部件/局部方向学生，或将连续解码与结构检索做留字配对。 | 真字图上界低，先修字库匹配/连续重建；上界高而 spike 学生低，才改神经解码。必须在完整留字折、两鼠上验证；不能由字体库下标推断测试身份。 |
| **4：提交前唯一的生产替换门** | 只把通过多鼠、多折的单个冻结改动接入 v5；完整重训、打包、校验哈希，再与当前 v5 提交作实际对照。 | 本地只保留 paired delta 与不确定范围，不用固定 offset 宣称将达到某个 Kaggle 分数；若新提交下降，回退到已保存的 v5 包，不改写历史结果。 |

这套顺序把预算首先放在 **Task 3 的低 recall 与 Task 2 的跨日不稳定**，而不是继续在已知失败的融合权重上搜索。50 或 54 分不是现有证据支持的可达性结论；当前目标应是获得下一个可复现、能经实际提交确认的相对增益。

## 历史复盘（截至第 22 组实验；数值与结论由上一节更新）

**已经完成的是训练、三个任务的本地验证和提交链路；尚未完成的是可在 Kaggle Public OCF 达到 50 的模型。** v3 唯一已知公开分为用户实测 **21.09590**，距 50 差 **28.90410**；v4 只改变 Task 2，尚无公开分。下表是固定本地子集，不是 v4 的公开成绩，也不能把本地增益加到 21.09590 上预测提交分。早期摘要与复现章节保留为历史记录，下一步以本节为准。

|任务与验证单位|本地最佳 OCF|决定下一步的实验事实|
|---|---:|---|
|Task 1：留字，211 字×8 鼠/会话块|18.62431|直接输出约 13.13，清晰字库检索约 18.39；换字体仅再提高约 0.14（相对原 directional2）。实验 21 换方向余弦训练，tune 从 16.81178 升至 17.00751，held-out 却降至 17.96979。不能把问题归结为“换损失”。|
|Task 2：整日留出，512 字×2 鼠|24.33936|实验 18 三种子比单种子加 1.01005，但 exact Top1 仍只有 8.203%。实验 19 改融合权重后 held-out 降至 24.12243；尚不能确定瓶颈在跨日漂移还是字形近邻本就不可分。|
|Task 3：每字留单 trial，512 字×8 块|16.57463|Top1 仅 1.5625%。跨会话 pooled 比逐会话分类更好。实验 22 加三次训练重复的均值，tune 从 17.01181 降至 16.80122；候选未评 held-out。不能把 16.57463 记成增广模型得分。|

证据：[统一结果](../test/results_summary.json) · [实验 21](../test/21_task1_directional_loss/results_task1_directional_loss.json) · [实验 22](../test/22_task3_mean_augmentation/results_task3_mean_augmentation.json)。等权本地综合 **19.84610**。浅层逐神经元时间卷积、自由 DDPM、对角 CORAL、结构 medoid、top-k 投票及重排均未提供所需跃升；这些负结果只约束被测实现，不否定跨神经元交互、生成式似然或笔画的组合泛化。

### 先判断 50 是否在候选集内，再选模型

对已见字的原图检索，令 p 为完全命中的比例、m 为错误样本的平均 OCF；在正确图得 100 的条件下，均分为 `100p+(1−p)m`。用 Task 2、3 的实测 OCF 和 Top1 反推，在 **m 不变** 时，单任务达到 50 分别需要约 **39.3%、41.0%** 的命中率。提高错字的结构得分同样可以提高 OCF，这不是必需的分类准确率。Task 1 的字体为重渲染，像素完全相同率为零不等于汉字身份识别率为零；不能用字体库下标直接对比图片 ID。等权总分 50 也不要求三任务各为 50。

**首个实验不训练。** 在训练内部另设诊断折，保存各模型前 1/8/32/128 个候选。分别算①图完整入库时的覆盖上限，②候选中与参考图官方 OCF 最大值的均值（oracle@k），③实际第一名 OCF，④已知身份时的 recall@k。按字和鼠/日成组报告分布与配对差值，不把同字八条记录当八个独立字符。若 oracle@32 仍低于目标，停止做该候选集的重排；若全库上限高而 oracle@32 低，改候选生成；若 oracle@32 高而实际低，才投入排序。**仅用公开训练标签做诊断；oracle 不作为可提交成绩，不用于测试样本选择。**

### 四条独立路线：各自的最小实验与否决条件

|优先级与假设|训练、推理及对照|什么结果值得继续|
|---|---|---|
|**1 · 正向编码 CNN：用字形预测放电**（三任务，尤其 Task 1/2；A03、A06、A15–A19）。如果不同字的局部笔画能可靠预测一组神经元，就不必从稀疏信号直接生成全部像素。|64×64 字图输入共享浅层多尺度 CNN；每鼠独立读出头输出神经元×时间窗的非负计数率，以训练 trial 的 Poisson 负对数似然拟合。候选字逐一预测响应，按观测 spike 的似然检索；重复 trial 累加对数似然而非复制均值。先只替换现有 DCT→ridge encoding，不动字库和分配。|同时检查留字/留日的预测似然、oracle@32 和实际 OCF。若似然优于常数模型却不能提高召回，说明学到的是平均放电而非辨字信息；停止增加 CNN 深度。|
|**2 · 时间 CNN＋小型 Transformer：学习跨单元证据**（Task 2/3；D 组、E 组）。已试卷积主要提取各单元自身的时间模式，未回答“哪些单元在什么时间共同区分形近字”。|每鼠保留单元身份，先时间卷积，再用少量潜在查询汇聚单元 token；先试仅卷积＋线性混合，再加两层注意力。使用源日单 trial 的遮挡计数预测预训练，微调时同字异重复为正例、形近异字为难负例，对齐固定字图 CNN 原型。无单元匹配证据时绝不假定跨鼠同序。|相同参数预算比较 MLP、混合 CNN、加注意力和打乱 spike 对照；报告已见日、留日及单 trial 的 recall 与 OCF。若注意力只改善训练日，保留简单 CNN；若打乱输入仍有同等收益，则否决“神经表征改善”。|
|**3 · 笔画和二维布局教师：拆开未见字泛化与字体误差**（Task 1；F02–F07）。结构 medoid 在既有候选中挑字，不会学会组合未见字。|只用规则允许的字表、字体及训练配对，建立字形 CNN 教师，预测四象限笔画方向、前景掩模和部件相对位置；神经学生从 spike 预测同一组局部特征，再在全字库检索。先做无外部部件标签的四象限版本；有合法可信映射才加部件，不把码位顺序当标签。|先测**真字图→教师→字库**的留字 oracle，用来隔离字体库缺失/错误与神经解码失败；再比较**spike→学生**。教师可辨而学生不行，优先改神经端；两者都不行就不训练生成器。边界与骨架度量只诊断断笔、粘连，不替代 OCF。|
|**4 · 单 trial 潜变量去噪与源日对齐**（Task 3/2；D18、E 组）。向训练集追加平均 trial 没有解决单次观测的不确定性。|用共享低维动态状态预测被遮住的神经元计数，每训练日独立读出；源日内同字不同重复对齐到原型，推理只输入一次 spike。Task 2 在源日模拟整日缺失，以无适配模型为对照；仅在规则允许时单列无标签新日适配，不用新日字标签、trial 顺序或伪造配对。|分别消融潜变量、跨日对齐和适配；报告单 trial 对数似然、按置信度分组的 OCF、每鼠每留出日 OCF。若自监督改善似然却降低形近字辨别，或适配只利于一个日，回退到不适配模型。|

**决策顺序。** 先做候选集诊断，再在训练内部轮换留字、留日和留重复折；模型选择只用 tune。已有 held-out 已被多次查看，不当作新的盲测：预先锁定模型和评价规则，用未参与决策的分组折作确认；若无新折，注明是复用验证而非独立复制。Task 1 另报公开数据 1/2/4 次重复曲线，不外推成测试 20 次的实测分。只有至少两鼠、多个划分上增益方向一致且结构错误确实下降，才讨论全量训练。**近期最值得投入的是路线 1**：可沿用现有训练字图、神经响应和检索流程，一次对照就能区分非线性视觉编码是否优于 ridge。没有证据把这四条路线中的任一条称为“已能得 50 分”。

[113 篇学术证据及检索覆盖说明](#academic-evidence-index)。其中 54 篇在 Scholar 结果页直接匹配题名，其余以出版方、会议或原稿补充。Google Scholar 后续触发验证码，未将补充来源算成 Scholar 命中。

## 早期摘要（复现阶段历史记录）

本赛题不是单一的“神经网络生成图片”问题，而是三个统计分布偏移任务等权组合：未见字形的组合泛化、跨记录日的神经迁移、以及单 trial 低信噪比重建。64×64 输出、OCF 轮廓评分和官方建议的二值化，使“模糊但像素均方误差较低”的输出不一定具有竞争力。

调研支持以下路线：

1. **先建立可证伪的轻量基线**：正则化线性/低秩解码、已见字原型分类、官方条件 DDPM。它们分别检验连续重建、已见字先验和生成模型是否真正带来收益。
2. **Task 1 必须以留字验证为核心**，采用多尺度连续解码，再谨慎加入汉字笔画/结构先验；不能把未见字任务退化为训练图库分类。
3. **Task 2/3 可利用“字图已见”这一条件**：神经分类或检索后输出训练图库中的清晰字图，是值得优先验证的强基线；Task 2 另需跨日归一化或潜空间对齐。
4. **保留时间维并做正则化**。文献支持 V1 单元间感受野、朝向选择性和响应时延存在显著差异，不支持把整个 250 ms 粗暴压成单一计数，也不支持统一删除前 100–110 ms。[cite:4][cite:6][cite:7]
5. **所有模型选择最终以任务对应的官方 OCF 验证分为准**，并在提交前二值化。官方讨论中，原始抗锯齿 GT 自评分约 70.8，而 `(gt >= 128) * 255` 为 100；这直接说明输出后处理不可忽视。[cite:3]

本地已完成官方条件 DDPM 的独立 demo 复现：四个 checkpoint 均训练成功，生成 6 张符合格式的 64×64 灰度 PNG 和对应 ZIP；验证脚本通过，OCF 相同图/空白图冒烟测试分别为 100/0。随后完成正式数据 Task 2、mice1、3 epochs pilot：39,568 个训练样本和 4,400 个验证样本上 validation loss 从 0.076901 降至 0.016299，生成并验证 2,748 张测试 PNG。测试集没有本地 GT，因此这些结果只证明正式训练与生成链路可运行，**不构成测试 OCF 或重建质量结论**。

## 1. 任务、数据与约束

### 1.1 三个任务

| 划分 | 样本数 | 每样本 trial 数 | 字形是否已见 | 记录日/session | 核心难点 |
|---|---:|---:|---|---|---|
| Train | 87,936 | 1 | 2,748 个已见字 | 两鼠 session1–4 | 每字每天 4 个公开训练 repeat |
| Task 1 | 2,720 | 20 | **未见** | 与训练同为 session1–4 | 组合泛化、重复聚合 |
| Task 2 | 5,496 | 4 | 已见 | 新日期 session5 | 跨日漂移、少量重复 |
| Task 3 | 21,984 | 1 | 已见 | 与训练同为 session1–4 | 单 trial 鲁棒性 |

- 两只小鼠分别为 134 和 78 个 units；同一小鼠的 unit 编号跨 session 一致，不能据此假设两只小鼠之间存在 unit 对应。
- 原始事件时间戳为 30 kHz 采样 tick，窗口为 0–250 ms；官方 loader 默认 1 ms 二值化脉冲矩阵。
- 官方 DDPM 使用 50 ms 分箱、对同一样本 repeat 的计数取平均、再做 `log1p`，因此两鼠输入维度分别为 670 和 390。
- 每个 `sample_id` 只输出一张图，而不是每个 trial 一张图。Task 1/2 可自由组合其 repeat；Task 3 禁止借助同刺激的其他测试 trial。[cite:2]

### 1.2 OCF 与最终分数

官方 OCF 在三个尺度、四个方向上比较轮廓能量，并允许有限旋转、平移和平滑形变。每个任务先求逐图 OCF 算术平均，再等权组合：[cite:1]

\[
S_{final}=\frac{S_1+S_2+S_3}{3},\qquad
S_j=\frac{1}{N_j}\sum_i s_{j,i}.
\]

因此：

- 不能把 30,200 个测试样本直接合并平均来替代任务等权。
- 三个任务都需要独立验证；只提高样本最多的 Task 3 并不等价于同幅度提高最终分。
- OCF 允许的小位移不是放松结构错误的理由；错误笔画、缺失局部结构仍是核心问题。
- 官方评分实现用于模型选择和阈值选择；训练初期可使用 BCE、Dice、多尺度边缘等可微辅助损失，但不能声称它们等价于 OCF 梯度。

### 1.3 输出与规则约束

- 输出为 64×64、8-bit、单通道 mode `L` PNG；官方讨论确认建议提交二值图，随机清晰图的 OCF 约 11.3，空白图为 0。[cite:3]
- 官方示例先生成含 `task1/task2/task3` 的 PNG ZIP，再由适配器转换成含 `sample_id,prediction` 的 `submission.csv` ZIP；不能直接把 PNG ZIP 上传到 Kaggle。[cite:1][cite:2]
- 每队每天（UTC）一次提交；最终截止为 2027-01-20 10:00 UTC。
- 外部数据、字体、结构标签、代码和预训练模型必须满足比赛的可获得性、成本与许可要求，并在提交材料中列明。GPL、无明确许可证和第三方数据许可需要分别处理，不能把“公开可下载”理解为“可随意再分发”。
- 未经明确许可，不上传比赛原始数据、预测文件或未公开方案到公共社区。本文和复现说明不公开原始竞赛数据或测试预测。

## 2. 科学机理及其工程含义

### 2.1 局部、重叠、多尺度感受野

Niell 与 Stryker 的麻醉小鼠 V1 单单元记录显示明确的局部感受野、朝向和空间频率选择性；感受野平均直径约 20.3°，空间频率偏好中位数约 0.04 cycles/degree。[cite:4] 更早的光学成像与电生理工作也支持小鼠 V1 存在有序 retinotopic map。[cite:5]

**边界**：这些实验主要使用光栅或稀疏噪声，`cycles/degree` 不能在缺少屏幕距离、视角和眼睛位置时直接换算为汉字像素。

**工程推论**：

- 多尺度局部基、DCT/小波、方向边缘和卷积解码器比“某 unit 对应某像素”的硬映射更合理。
- 78/134 个单元并不提供 4,096 个独立像素测量；输出分辨率高于神经群体维度时，必须依赖群体编码、共享刺激结构和图像先验。
- 图像先验可以提升轮廓，却也可能补出不存在的笔画；Task 1 必须用留字验证监控 hallucination。

### 2.2 时间信息不能简单丢弃

Gao 等在特定麻醉小鼠 V1 实验中报告中位起始时延约 110 ms、约 90% 单元位于 70–150 ms，且时延与空间/时间敏感性相关。[cite:6] 清醒小鼠 Neuropixels 大样本研究同样发现视觉区内时延分布很宽，并随功能层级变化。[cite:7]

**边界**：110 ms 是特定状态、刺激与估计方法下的数值，不是本赛题所有单元的固定潜伏期。

**工程推论**：

- 保留完整 0–250 ms，让模型学习单元特异的时间核。
- 优先比较 1、5、10、25、50 ms 分箱及低秩时间基，而不是只使用总计数或直接堆叠高维逐毫秒特征。
- 高时间分辨率必须与 ridge、低秩分解、时间平滑或 dropout 配套，避免特征数相对样本量过大。

### 2.3 单 trial 变异、状态增益与 repeat 平均

Stringer 等发现小鼠视觉皮层对相同自然图像的重复响应存在显著 trial-to-trial variability；主体数据是大规模钙成像，并有电生理子集支持相近的单细胞信噪比水平。[cite:9] 跑动等行为状态还可显著改变 V1 的视觉诱发增益。[cite:8]

在独立残差假设下，重复平均使噪声标准差按 \(1/\sqrt r\) 下降：20 次约 4.47 倍幅度 SNR，4 次约 2 倍。若残差跨 repeat 等相关为 \(\rho\)，则：

\[
\operatorname{Var}(\bar{x}_r)=\sigma^2\left(\rho+\frac{1-\rho}{r}\right).
\]

这只是统计模型，不是本数据收益保证。跨 repeat 相关与跨 unit 噪声相关也必须区分；后者对群体信息的影响取决于其与刺激信号方向的关系，不能简单概括为“相关越低越好”。[cite:10]

**工程推论**：

- Task 1/2 比较均值、trimmed mean、median 和学习式 repeat 聚合，并用 split-half 实测可靠性。
- 训练采用 repeat dropout，使模型不只适应高 SNR 平均输入。
- Task 3 使用真实单 trial 训练和验证；可加入基于训练残差的噪声增强，但不能凭空声称其分布与测试一致。

### 2.4 跨日变化不是单一现象

Deitch 等在小鼠视觉区观察到分钟到天尺度的表示漂移，同时群体活动模式之间的关系结构相对稳定。[cite:11] Marks 与 Goard 则发现漂移具有刺激依赖性：漂移光栅下较稳定，自然电影下更明显。[cite:12] 灵长类运动 BCI 的低维流形对齐为跨日固定解码器提供了方法启发，但不是小鼠 V1 视觉重建的直接证据。[cite:13]

**工程推论**：

- Task 2 必须按完整 session/day 留出验证，不能随机拆分 repeat 后声称具备跨日能力。
- 依次比较冻结模型、训练期统计归一化、无标签目标日统计校正、CORAL/低维对齐和轻量 adapter。
- 目标日无标签 spike 若用于统计适配，应在方案与提交材料中明确；不得使用隐藏标签。
- 跨日差异可能包含增益、基线、记录不稳定和真正编码变化，不能全部归因于 biological drift。

## 3. 相关论文与算法模型

### 3.1 直接 spike/神经脉冲重建

| 工作 | 方法与对象 | 可复现资源 | 对本赛题的价值与边界 |
|---|---|---|---|
| Chen et al., 2024 [cite:14] | 视觉皮层神经响应→Dense 64×64→CAE 256×256 | [官方代码](https://github.com/beizai/Decoding)，GPL-3.0；论文与示例数据 | 本项目主论文复现目标；输入维和数据管线需改造，旧 TF 环境不能直接套用到现有环境 |
| Stanley et al., 1999 [cite:15] | 猫 LGN 177 细胞 spikes 的线性时空重建 | 全文公开，无完整官方训练仓库 | 支持必须保留线性时空基线；对象和刺激不同 |
| Parthasarathy et al., 2017 [cite:16] | RGC spikes 的线性重建与卷积 autoencoder | 原代码地址已失效 | 可借模型思想，不能声称官方代码可复现 |
| RGC MAP Reconstruction, 2022 [cite:17] | GLM/LNP、spike history、细胞耦合与 dCNN 图像先验 | 官方代码和 Zenodo toy 资源；仓库未见明确许可证 | 神经统计建模更完整，但灵长类 RGC 到小鼠 V1 的适配成本高，是备选而非首选 |
| S2INet, 2023 [cite:18] | Gabor 卷积、autoencoder、感受野损失 | 代码仅见模型/损失，未见完整训练数据管线和许可证 | 可借 Gabor 归纳偏置，不宜作为端到端复现起点 |
| Robust Decoding, 2024 [cite:19] | RGC signal converter + U-Net，比较噪声与单/多 trial | 本次未找到完整官方代码 | 与 Task 3 鲁棒性研究相关，但结论不能直接外推到本数据 |

这些工作共同支持“线性基线—卷积解码—神经统计模型—图像先验”的递进比较，而不是一开始就假设更复杂的生成模型必然更好。

### 3.2 扩散模型与官方基线

官方基线是神经放电条件 DDPM：神经特征与扩散时间嵌入共同调制 64×64 卷积去噪网络，采用 64 个训练扩散步、v-prediction、Min-SNR 加权、10% 条件丢弃和 classifier-free guidance。

- DDPM 提供去噪生成框架。[cite:24]
- Classifier-free guidance 通过联合训练的条件/无条件估计进行引导，不需要额外分类器。[cite:25]
- Min-SNR 通过限制不同噪声步的权重冲突改善扩散训练。[cite:26]

**赛题边界**：生成质量、去噪 loss 和 OCF 不是同一目标。扩散模型可能产生更自然的笔画，也可能出现字形漂移；必须与确定性解码器在相同留字/留日/单 trial 验证上比较。

### 3.3 其他记录模态：只作迁移借鉴

- Miyawaki 等的人 fMRI 多尺度局部解码支持多尺度输出基，但 BOLD 的时空采样机制与 250 ms spike 完全不同。[cite:22]
- Brain-Diffuser 的低层布局与高层语义两阶段设计可借鉴，但其 NSD fMRI、CLIP 与大规模扩散资源不是本任务直接证据。[cite:23]
- SENSORIUM 小鼠视觉重建和动态电影重建工作可借鉴跨个体微调与视频先验；输入为钙成像，训练成本、采样率和信号来源均不同。[cite:27][cite:28]

### 3.4 汉字结构先验

- SDN 将汉字分解为笔画序列，支持 character/radical 级 zero-shot 识别。[cite:20]
- CCR-CLIP 使用 IDS 表意描述序列学习未见字的结构表征。[cite:21]

它们不是神经解码模型。可迁移的是**笔画、部件、IDS 等组合表征**，不是其识别精度。若使用外部字体、结构标签或预训练模型，必须验证许可与公平性，且不得引入隐藏测试 GT 或样本映射。对 Task 1，结构原型只能作为经留字验证的软约束，避免把神经证据不足的输出强行吸附到错误常见字。

## 4. 复现设计

### 4.1 主论文复现：Chen et al. 2024

选择理由：

1. 直接面向神经信号图像重建，模型中间输出恰为 64×64。
2. 官方代码、论文和示例数据可获取，GPL-3.0 许可明确。
3. Dense+CAE 结构简单，适合与官方 DDPM 做确定性/生成式对照。

复现目录：

```text
ref/
├── third_party/Decoding/           # 官方仓库，本地保留
├── papers/Chen_2024_Decoding.pdf  # 原论文
├── downloads/                     # 官方示例数据与结果
└── chen2024_reproduction/         # 兼容项目 .venv 的复现实现
```

原始实现固定 800 维输入，Dense 经 16,384、8,192、4,096 个单元生成 **64×64** 中间图，再由 CAE 六次 2 倍上采样至 **256×256** 最终输出；训练为 MSE、Adam、batch 16、400 epochs。原环境是 Python 3.6 与旧 TensorFlow/Keras 组合，不能将其旧 requirements 直接安装进当前项目环境。复现保留模型结构，使用当前 PyTorch，并明确标记移植与任何数值差异。

#### 已完成的工作

1. **资源下载与校验**：官方示例数据 5 个文件共 2,015,335,040 字节（约 1.88 GiB），全部按 Google Drive 元数据核对大小、逐文件 SHA256，并以只读内存映射完整扫描所有数组元素。
2. **模型移植**：逐层对应 `dense_decoder` 与 `CAE`，保留 sigmoid 位置、BN 无激活输出、nearest 上采样、BN `eps/momentum` 与 Dropout；`ncell=800` 时可训练参数 215,168,003。
3. **软件测试**：CPU 前向、反向和一次 Adam 更新通过，`dense_out` 为 `N×1×64×64` 且在 `[0,1]`，`cae_out` 为 `N×1×256×256`。
4. **论文指标复算**：用作者发布的预测与测试图，按 `test.py` 的 395 帧窗口复算，得到 **SSIM 0.8137、PSNR 25.75 dB、MSE 0.00439**（均值；中位数 SSIM 0.8367、PSNR 26.22 dB）。同一脚本可复现实验输出指标。
5. **缩短训练复现**：在 3,200 个训练帧、400 个测试帧上，用项目 `.venv` 从随机初始化实际训练 30 epochs（每轮约 116 秒），测试 SSIM 从随机初始化的 0.0003 提升到 **0.6355**，PSNR 从 5.02 dB 提升到 **20.14 dB**。

> 说明：原论文为 400 epochs、batch 16；本复现受 16 GiB 显存限制改用 batch 4，30 epochs 约相当于论文 30% 的优化步数，**未达到**作者结果的 SSIM 0.8137 / 25.75 dB。因此本复现验证的是“结构正确、可训练、指标同向提升”，不冒充论文数值的完整复刻。完整命令与逐轮曲线见 [REPRODUCTION.md](chen2024_reproduction/REPRODUCTION.md)。

#### 已校验的示例数据

| 文件 | shape | dtype | 值域 |
|---|---|---|---|
| `movie_03_train_pic_1999_VISp_800.npy` | (3200, 256, 256) | float64 | 0.0863–1.0 |
| `movie_03_train_spike_1999_VISp_800.npy` | (3200, 800) | float64 | 0–7.9 |
| `movie_03_test_pic_1999_VISp_800.npy` | (400, 256, 256) | float64 | 0.1020–1.0 |
| `movie_03_test_spike_1999_VISp_800.npy` | (400, 800) | float64 | 0–7.3 |
| `movie_03_iter_1999_VISp_800_400.npy` | (400, 256, 256, 1) | float32 | 0.1000–1.262 |

**与赛题的关键差异**：该示例是 **256×256 灰度自然电影**（背景约 0.10 而非 0 黑）、800 维输入、连续像素回归；本赛题是 **64×64 黑底白字二值轮廓**、78/134 units、OCF 结构评分。因此：

- 只应取 `dense_out` 的 64×64 分支，去掉 CAE 的 256×256 上采样。
- 像素 MSE 不能替代 OCF；示例中的 SSIM/PSNR 也不能与本赛题分数横向比较。
- 800 维输入需替换为 250 ms 时间特征；示例的自然电影预训练权重不能直接迁移到汉字任务。

### 4.2 官方条件 DDPM 复现

独立目录：`ref/official_ddpm_reproduction/`。原始参赛包不被修改，所有训练使用项目 `.venv`。

环境实测：

- CPython 3.10.22；PyTorch 2.5.1+cu121。
- RTX 3080 Laptop GPU 16 GiB，CUDA 与 BF16 可用。
- 补充安装 Pillow、diffusers 0.35.2 及评估/下载依赖。

demo 结果（带日志的独立复跑，`runs/demo_e1_verify`，日志 `logs/demo_e1_train.log`）：

| 分组 | epochs | train loss | validation loss |
|---|---:|---:|---:|
| mice1/session1 | 1 | 0.436826 | 0.393969 |
| mice1/across_sessions | 1 | 0.436826 | 0.393969 |
| mice2/session1 | 1 | 0.438975 | 0.395493 |
| mice2/across_sessions | 1 | 0.438975 | 0.395493 |

demo 仅 8 个训练 trial，且官方脚本的划分与初始化未固定随机种子，重复运行 loss 会不同；这些数值只用于确认链路正常，不用于比较。

验证结果：4 个 checkpoint、6 张 PNG、6 个 ZIP 条目全部通过；图像均为 64×64 mode `L`；官方 OCF 对相同二值图返回 100，对空白预测返回 0。

正式数据 pilot 采用 Task 2、`mice1/across_sessions`、3 epochs、训练 batch 32、预测 batch 16，未发生 CUDA OOM：

| Epoch | train loss | validation loss |
|---:|---:|---:|
| 1 | 0.049108 | 0.076901 |
| 2 | 0.018557 | 0.022562 |
| 3 | 0.015854 | 0.016299 |

- 数据划分：39,568 train、4,400 validation。
- checkpoint：`runs/task2_mice1_e3/mice1/across_sessions.pt`，`inputs=670`、`epoch=3`，严格加载成功。
- 输出：Task 2 mice1 的 2,748 张 PNG；ID 与 catalog 完全一致，无缺失或额外文件，全部为 64×64 mode `L`。
- 完整训练、预测和校验命令及日志见 [REPRODUCTION.md](official_ddpm_reproduction/REPRODUCTION.md)。

**解释限制**：demo 仅验证数据读取、模型分组、训练、采样、导出和打包链路；正式 pilot 进一步证明优化与正式规模预测可以运行，但 validation denoising loss 不是 OCF。测试集没有本地 GT，两个实验都不能给出真实测试 OCF 或泛化排名结论。

## 5. 推荐的整体技术方案

### 5.1 模型组合而非单模型包打天下

```text
共享特征与验证
  spike events → 多尺度时间特征 → 单元/会话归一化 → repeat 聚合与增强
                                      │
             ┌────────────────────────┼────────────────────────┐
             │                        │                        │
       Task 1 连续解码            Task 2 跨日适配           Task 3 单trial解码
       ridge/低秩 + CNN/DDPM       全局模型 + 对齐/adapter     原型分类 + 连续残差
             │                        │                        │
             └────────────────────────┼────────────────────────┘
                                      │
                         OCF验证选择阈值 → 二值化PNG → 格式适配器
```

两只小鼠使用独立输入层、归一化和正则化；可以共享图像基或 decoder，但共享是否有效必须由验证证明。

### 5.2 基线 A：正则化低秩连续解码

1. 比较总计数与多时间 bin 特征，标准化仅用训练折统计。
2. 将二值训练字图转到 DCT/小波/PCA 等低维基，先预测 64–256 个系数，而不是无约束预测全部 4,096 像素。
3. 使用 ridge 或 reduced-rank regression，分别拟合小鼠/会话；通过验证选择正则化与截断秩。
4. 反变换后裁剪到 `[0,1]`，用 OCF 选择二值化阈值。

作用：提供可解释、低成本、难被“复杂模型叙事”替代的下界。若深度模型不能超过它，就没有必要增加训练复杂度。[cite:15][cite:16][cite:22]

### 5.3 基线 B：已见字原型分类/检索

Task 2/3 的字图均在训练图库中，因此可先把神经响应映射到 2,748 类，再输出对应二值字图：

- 最近质心、余弦相似度、收缩协方差 Mahalanobis 距离作为首组对照。
- 类别均值只使用当前训练折的 repeat；验证或测试字图不得进入原型。
- Task 3 使用 session-specific 单 trial 分类；Task 2 使用跨 session 原型并聚合目标样本的 4 个 trial。
- 可将分类概率对应的多个字图做软混合，再与连续解码残差融合；最终仍按 OCF 二值化。

**[本项目假设]** 该路线可能显著优于自由生成，因为正确分类会直接给出清晰结构；其风险是单 trial 2,748 类分类很难，错误分类会产生完全错误的字。必须用实验而不是直觉判断。

### 5.4 Task 1：未见字的组合泛化

主分支采用连续低秩/CNN/扩散解码，训练与验证严格按字 identity 分组。可逐步加入：

1. **多时间尺度神经编码**：保留单元潜伏期差异，使用时间低秩基或轻量 temporal convolution。
2. **多尺度图像监督**： coarse layout 与 fine strokes 分层损失，避免只拟合平均模糊图。
3. **笔画/部件组合先验**：在许可允许时，以 SDN/IDS 等作为辅助表征或教师；不使用测试字映射。[cite:20][cite:21]
4. **软结构融合**：只有留字 OCF 证明有效时，才将自由重建与结构原型/字体先验融合。
5. **repeat 聚合**：比较平均、稳健平均和学习式加权；用 split-half 和不同 repeat 数的学习曲线选择。

### 5.5 Task 2：跨日已见字

- 用留一整天模拟 session5；公开训练日每天 4 个 repeat，恰好可模拟目标样本的 4-repeat 聚合。
- 比较冻结全局模型、按日 z-score/增益校正、CORAL/低维对齐、轻量仿射 adapter。
- 目标日适配仅使用允许的无标签 spike 统计；每字标签和 GT 图只能来自训练图库。
- 原型分类与连续解码分别做跨日验证，再决定是否融合。

### 5.6 Task 3：同 session 单 trial

- 从公开训练每天的 4 个 repeat 中留出 1 个做单 trial 验证，剩余 3 个训练，避免同 trial 泄漏。
- 以原型分类、低秩回归及其残差融合为主线；深度模型使用 unit dropout、repeat/残差增强和强正则化。
- 不使用测试端多 trial 平均，不把 Task 1 的 20-repeat 经验直接套到 Task 3。

## 6. 验证与实验矩阵

### 6.1 三种验证必须分开

| 目标 | 训练/验证划分 | 聚合方式 | 防止的泄漏 |
|---|---|---|---|
| 模拟 Task 1 | 按字 identity 留出，所有 session/repeat 同侧 | 公开数据最多 4 repeats；另做 1/2/4 学习曲线 | 同字不同 repeat 横跨训练与验证 |
| 模拟 Task 2 | 留完整一天，字在其他天已见 | 验证日每字 4 repeats | 随机 repeat 造成跨日虚高 |
| 模拟 Task 3 | 每个 session 留一个 repeat，字已见 | 单 trial | 用平均响应代替单 trial |

Task 1 的公开训练每字只有 4 个 repeat，不能声称完全复现了 20-repeat 测试信噪比；这是验证协议的明确限制。

### 6.2 实验顺序

| 优先级 | 对照 | 主要观察 | 继续条件 |
|---|---|---|---|
| P0 | 空白、随机清晰、图库均值、二值化阈值 | OCF 与格式链路 | 建立可靠本地评分与输出基线 |
| P0 | 最近质心/收缩原型 | Task 2/3 分类准确率和 OCF | 原型路线超过自由重建下界 |
| P0 | ridge/低秩连续解码 | 三任务各自 OCF、时间 bin/秩 | 作为所有复杂模型的共同对照 |
| P1 | Chen Dense/CAE 与官方 DDPM | 相同划分下 OCF、稳定性、推理成本 | 相对轻量基线有稳定收益 |
| P1 | repeat 聚合与跨日校正 | Task 1/2 分任务收益 | 只保留验证稳定的模块 |
| P2 | 汉字结构教师、深度时序模型、集成 | 留字泛化与 hallucination | 遵守外部资源规则且收益覆盖复杂度 |

所有实验报告任务分别的平均 OCF、二值化阈值、随机种子和训练配置；不能只报告 train/validation denoising loss。

## 7. 风险与停止条件

| 风险 | 识别方式 | 处理 |
|---|---|---|
| 留字或跨日泄漏 | 检查 split 的字、session、repeat 元数据 | 重建 split；废弃受污染分数 |
| 输出过平滑 | OCF 低而 MSE 看似合理；阈值曲线不稳定 | 二值化、多尺度结构损失、原型融合 |
| 生成错误但漂亮 | 留字集出现常见字吸附或 hallucinated strokes | 降低图像先验/结构吸附权重，保留连续分支 |
| 单 trial 过拟合 | 训练 loss 降、单 trial OCF 不升 | 强正则、unit/repeat dropout、退回轻量模型 |
| 跨日适配负迁移 | 留日验证低于冻结模型 | 停止适配或向全局参数收缩 |
| 旧依赖/许可问题 | 依赖冲突、仓库无 LICENSE、资源不可平等获取 | 隔离移植、记录修改；不使用许可不清资源 |
| 把 demo 当正式结论 | 小样本 loss/图片被解释为质量 | 只作软件测试；正式结论必须来自对应验证 |

## 8. 可执行落地方案

1. **统一数据与评估层**：缓存多时间 bin 特征，固化三类 split、OCF 调用、二值化与提交格式测试。
2. **先完成轻量基线**：低秩 ridge 与原型分类给出 Task 1/2/3 的可信下界。
3. **保留两个复现对照**：Chen Dense/CAE 代表确定性前馈重建，官方 DDPM 代表条件生成模型。
4. **分任务增加模块**：Task 1 加组合结构，Task 2 加跨日对齐，Task 3 加单 trial 鲁棒性；每个模块必须有独立消融。
5. **最后再集成和全量生成**：固定阈值与配置，生成 30,200 张图，执行 ID、尺寸、模式、Base64 与 ZIP 检查后再提交。

不建议在轻量基线尚未建立前直接训练十组大模型或引入重型 fMRI 视觉解码仓库；那会增加算力和许可成本，却不能回答三个任务各自的统计问题。

## 9. 本地产物与复现索引

所有路径相对 `D:\WORK\siclink`；统一解释器为 `D:\WORK\siclink\.venv\Scripts\python.exe`，依赖版本见 [requirements-repro.txt](requirements-repro.txt)。

| 内容 | 路径 | 说明 |
|---|---|---|
| 本报告 | `ref/siclink_research_and_solution.md` | 机理、文献、复现与方案 |
| Chen 2024 论文 | `ref/papers/Chen_2024_Decoding.pdf` | PLOS 可打印版 |
| Chen 官方仓库 | `ref/third_party/Decoding` | GPL-3.0，commit `631ebbcec0eedb907dd1ec24a908c4f99324c219` |
| Chen 示例数据与作者结果 | `ref/downloads/` | 5 个 NPY，含 SHA256 清单 `chen2024_manifest.json` |
| Chen PyTorch 移植与实验 | `ref/chen2024_reproduction/` | 模型、软件测试、指标复算、训练脚本与记录 |
| 官方 DDPM 独立复现 | `ref/official_ddpm_reproduction/` | demo 与正式 Task 2 pilot，含日志与验证脚本 |

常用复现命令（必须先切到对应目录，脚本用相对导入）：

```powershell
# Chen 2024：模型软件测试、论文示例指标复算
Set-Location 'D:\WORK\siclink\ref\chen2024_reproduction'
& 'D:\WORK\siclink\.venv\Scripts\python.exe' -B test_model.py
& 'D:\WORK\siclink\.venv\Scripts\python.exe' -B evaluate_paper_example.py

# 官方 DDPM：demo 产物校验
Set-Location 'D:\WORK\siclink\ref\official_ddpm_reproduction'
& 'D:\WORK\siclink\.venv\Scripts\python.exe' -B validate_demo.py
```

原始竞赛数据、第三方仓库、论文 PDF 与下载数组体积较大或有再分发限制，均已在 `.gitignore` 中排除，仅在本地保留。

## References

- [cite:1] [SiClink Competition — Overview, Evaluation and Submission](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/overview)
- [cite:2] [SiClink Competition — Data](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/data)
- [cite:3] [SiClink Discussion — What are typical OCF scores, and how can I sanity-check my decoder?](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge/discussion/671895)
- [cite:4] Niell & Stryker (2008), [Highly selective receptive fields in mouse visual cortex](https://doi.org/10.1523/JNEUROSCI.0623-08.2008).
- [cite:5] Schuett et al. (2002), [Mapping Retinotopic Structure in Mouse Visual Cortex with Optical Imaging](https://doi.org/10.1523/JNEUROSCI.22-15-06549.2002).
- [cite:6] Gao et al. (2010), [Parallel Input Channels to Mouse Primary Visual Cortex](https://doi.org/10.1523/JNEUROSCI.6456-09.2010).
- [cite:7] Siegle et al. (2021), [Survey of spiking in the mouse visual system reveals functional hierarchy](https://doi.org/10.1038/s41586-020-03171-x).
- [cite:8] Niell & Stryker (2010), [Modulation of visual responses by behavioral state in mouse visual cortex](https://doi.org/10.1016/j.neuron.2010.01.033).
- [cite:9] Stringer et al. (2019), [High-dimensional geometry of population responses in visual cortex](https://doi.org/10.1038/s41586-019-1346-5).
- [cite:10] Cohen & Kohn (2011), [Measuring and interpreting neuronal correlations](https://doi.org/10.1038/nn.2842).
- [cite:11] Deitch et al. (2021), [Representational drift in the mouse visual cortex](https://doi.org/10.1016/j.cub.2021.07.062).
- [cite:12] Marks & Goard (2021), [Stimulus-dependent representational drift in primary visual cortex](https://doi.org/10.1038/s41467-021-25436-3).
- [cite:13] Degenhart et al. (2020), [Stabilization of a brain-computer interface via the alignment of low-dimensional spaces of neural activity](https://doi.org/10.1038/s41551-020-0542-9).
- [cite:14] Chen et al. (2024), [Decoding dynamic visual scenes across the brain hierarchy](https://doi.org/10.1371/journal.pcbi.1012297); [official code](https://github.com/beizai/Decoding).
- [cite:15] Stanley et al. (1999), [Reconstruction of natural scenes from ensemble responses in the lateral geniculate nucleus](https://pmc.ncbi.nlm.nih.gov/articles/PMC6782475/).
- [cite:16] Parthasarathy et al. (2017), [Neural Networks for Visual Reconstruction](https://papers.neurips.cc/paper_files/paper/2017/file/b0169350cd35566c47ba83c6ec1d6f82-Paper.pdf).
- [cite:17] [RGC MAP Reconstruction — NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ae447e9dbfdd1189966e894b85bea062-Abstract-Conference.html); [official code](https://github.com/wueric/RGC_MAP_reconstruction).
- [cite:18] [S2INet — Frontiers 2023](https://doi.org/10.3389/fnins.2023.1130606); [code](https://github.com/WYCAS/S2INet).
- [cite:19] [Robust Decoding — IEEE TNNLS 2024](https://doi.org/10.1109/TNNLS.2024.3351120).
- [cite:20] [Stroke-level Decomposition Network — IJCAI 2021](https://www.ijcai.org/proceedings/2021/85); [code](https://github.com/FudanVI/FudanOCR/tree/main/stroke-level-decomposition).
- [cite:21] [CCR-CLIP — ICCV 2023](https://arxiv.org/abs/2309.01083); [code](https://github.com/FudanVI/FudanOCR/tree/main/image-ids-CTR).
- [cite:22] Miyawaki et al. (2008), [Visual image reconstruction from human brain activity using a combination of multiscale local image decoders](https://pubmed.ncbi.nlm.nih.gov/19081384/).
- [cite:23] [Brain-Diffuser — Scientific Reports 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10511448/); [code](https://github.com/ozcelikfu/brain-diffuser).
- [cite:24] Ho et al. (2020), [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).
- [cite:25] Ho & Salimans (2022), [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598v1).
- [cite:26] Hang et al. (2023), [Efficient Diffusion Training via Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556).
- [cite:27] [Sensorium-viz — Advanced Science 2026](https://doi.org/10.1002/advs.202520220); [code](https://github.com/GuanLab/sensorium-viz).
- [cite:28] Bauer et al. (2026), [Movie reconstruction from mouse visual cortical calcium activity](https://doi.org/10.7554/eLife.105081.3); [code](https://github.com/Joel-Bauer/movie_reconstruction_code).

<a id="academic-evidence-index"></a>

## 附录：学术证据索引

**收录 113 篇不同工作，其中 54 篇有直接 Google Scholar 题名匹配记录。** 三份已核验 JSON 分别提供视觉重建 43 条、神经时序 31 条、跨域与度量 34 条，共 108 条。比较时统一题名大小写、空格、连字符及标点，再对照 DOI、作者与版本：A07 的 arXiv 与期刊版合并，D18 的 LFADS 跨表重复合并，剩 106 篇；另列正文主论文和 6 篇字形相关工作。正文简称不另计，如 [cite:16]、[cite:17]、[cite:18]、[cite:23] 分别对应 A06、A09、D10、B11。赛题网页、教程、代码库不计。

**读法：** 题名链接指向出版方、会议论文页或作者原稿。`S`＝输入记录中至少一次在 Scholar 返回结果直接看见题名；`R`＝输入记录有原始来源，但没有直接 Scholar 题名匹配证据；`V`＝本次另查会议、期刊或 DOI 注册记录，未计 Scholar 命中。三表的 S 原始计数为 16、31、8；A07 两表均命中，只计一次，D18 一表命中、一表未确认，仍只计一次。未绕过验证码，不把查询链接当作命中。年份采用所列工作发表年或预印本年；链接至原稿不等于该版本已通过同行评审。

### A｜视觉神经编码与重建（19 篇）

A01–A09 是从群体放电恢复刺激的直接方法证据，但视网膜、外侧膝状体不等于小鼠 V1。A10–A14 涉及视觉皮层或钙信号；A15–A19 研究刺激如何驱动神经元，不是从一次放电还原真实刺激。

- A01 · 1997 · S · [Decoding Visual Information From a Population of Retinal Ganglion Cells](https://doi.org/10.1152/jn.1997.78.5.2336)
- A02 · 1999 · R · [Reconstruction of Natural Scenes from Ensemble Responses in the Lateral Geniculate Nucleus](https://doi.org/10.1523/JNEUROSCI.19-18-08036.1999)
- A03 · 2005 · R · [Prediction and Decoding of Retinal Ganglion Cell Responses with a Probabilistic Spiking Model](https://doi.org/10.1523/JNEUROSCI.3305-05.2005)
- A04 · 2001 · R · [Retinal ganglion cells act largely as independent encoders](https://doi.org/10.1038/35079612)
- A05 · 2008 · R · [Spatio-temporal correlations and visual signalling in a complete neuronal population](https://doi.org/10.1038/nature07140)
- A06 · 2017 · S · [Neural Networks for Efficient Bayesian Decoding of Natural Images from Retinal Neurons](https://proceedings.neurips.cc/paper/2017/hash/b0169350cd35566c47ba83c6ec1d6f82-Abstract.html)
- A07 · 2020 · S · [Reconstruction of natural visual scenes from neural spikes with deep neural networks](https://doi.org/10.1016/j.neunet.2020.01.033)（[同作预印本](https://arxiv.org/abs/1904.13007)）
- A08 · 2021 · S · [Nonlinear Decoding of Natural Images From Large-Scale Primate Retinal Ganglion Recordings](https://doi.org/10.1162/neco_a_01395)
- A09 · 2022 · S · [Maximum a posteriori natural scene reconstruction from retinal ganglion cells with deep denoiser priors](https://proceedings.neurips.cc/paper/2022/hash/ae447e9dbfdd1189966e894b85bea062-Abstract-Conference.html)
- A10 · 2019 · R · [High-dimensional geometry of population responses in visual cortex](https://doi.org/10.1038/s41586-019-1346-5)
- A11 · 2016 · S · [Benchmarking spike rate inference in population calcium imaging](https://doi.org/10.1016/j.neuron.2016.04.014)
- A12 · 2018 · S · [Community-based benchmarking improves spike rate inference from two-photon calcium imaging data](https://doi.org/10.1371/journal.pcbi.1006157)
- A13 · 2018 · S · [Robustness of Spike Deconvolution for Neuronal Calcium Imaging](https://doi.org/10.1523/JNEUROSCI.3339-17.2018)
- A14 · 2018 · R · [Visual reconstruction from 2-photon calcium imaging suggests linear readout properties of neurons in mouse primary visual cortex](https://doi.org/10.1101/300392)（bioRxiv）
- A15 · 2019 · R · [Deep convolutional models improve predictions of macaque V1 responses to natural images](https://doi.org/10.1371/journal.pcbi.1006897)
- A16 · 2019 · S · [Inception loops discover what excites neurons most using deep predictive models](https://doi.org/10.1038/s41593-019-0517-x)
- A17 · 2019 · S · [Evolving Images for Visual Neurons Using a Deep Generative Network Reveals Coding Principles and Neuronal Preferences](https://doi.org/10.1016/j.cell.2019.04.005)
- A18 · 2020 · S · [XDream: Finding preferred stimuli for visual neurons using generative networks and gradient-free optimization](https://doi.org/10.1371/journal.pcbi.1007973)
- A19 · 2023 · S · [Energy Guided Diffusion for Generating Neurally Exciting Images](https://proceedings.neurips.cc/paper_files/paper/2023/hash/67226725b09ca9363637f63f85ed4bba-Abstract-Conference.html)

### B｜人脑 fMRI 视觉重建（12 篇）

关联正文 §3.3：比较多尺度读出、贝叶斯图像先验与条件扩散。体素的空间汇聚和血氧响应时间尺度不同于 spike；这些是解码方法的参照，不是本赛题精度的证据。

- B01 · 2008 · R · [Visual Image Reconstruction from Human Brain Activity using a Combination of Multiscale Local Image Decoders](https://doi.org/10.1016/j.neuron.2008.11.004)
- B02 · 2009 · R · [Bayesian Reconstruction of Natural Images from Human Brain Activity](https://doi.org/10.1016/j.neuron.2009.06.017)
- B03 · 2011 · R · [Reconstructing Visual Experiences from Brain Activity Evoked by Natural Movies](https://doi.org/10.1016/j.cub.2011.08.031)
- B04 · 2019 · R · [Deep image reconstruction from human brain activity](https://doi.org/10.1371/journal.pcbi.1006633)
- B05 · 2019 · R · [From voxels to pixels and back: Self-supervision in natural-image reconstruction from fMRI](https://arxiv.org/abs/1907.02431)
- B06 · 2018 · R · [Generative adversarial networks for reconstructing natural images from brain activity](https://doi.org/10.1016/j.neuroimage.2018.07.043)
- B07 · 2022 · R · [Self-supervised natural image reconstruction and large-scale semantic classification from brain activity](https://doi.org/10.1016/j.neuroimage.2022.119121)
- B08 · 2023 · S · [High-resolution image reconstruction with latent diffusion models from human brain activity](https://doi.org/10.1109/CVPR52729.2023.01389)
- B09 · 2023 · R · [Seeing Beyond the Brain: Conditional Diffusion Model With Sparse Masked Modeling for Vision Decoding](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Seeing_Beyond_the_Brain_Conditional_Diffusion_Model_With_Sparse_Masked_CVPR_2023_paper.html)
- B10 · 2023 · S · [Reconstructing the Mind’s Eye: fMRI-to-Image with Contrastive Learning and Diffusion Priors](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4ddab70bf41ffe5d423840644d3357f4-Abstract-Conference.html)
- B11 · 2023 · R · [Natural scene reconstruction from fMRI signals using generative latent diffusion](https://doi.org/10.1038/s41598-023-42891-8)
- B12 · 2023 · S · [MindDiffuser: Controlled Image Reconstruction from Human Brain Activity with Semantic and Structural Diffusion](https://doi.org/10.1145/3581783.3613832)

### C｜图像读出、生成先验与相似度（12 篇）

关联正文 §3.2、§4.2：C01–C05 提供识别或图像映射模型，C06–C08 衡量重建差异，C09–C12 给出扩散建模与采样依据。图像相似度不是汉字身份正确率。

- C01 · 1998 · R · [Gradient-based learning applied to document recognition](https://doi.org/10.1109/5.726791)
- C02 · 2015 · R · [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- C03 · 2014 · R · [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- C04 · 2014 · R · [Generative Adversarial Nets](https://proceedings.neurips.cc/paper/2014/hash/f033ed80deb0234979a61f95710dbe25-Abstract.html)
- C05 · 2017 · R · [Image-to-Image Translation with Conditional Adversarial Networks](https://openaccess.thecvf.com/content_cvpr_2017/html/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.html)
- C06 · 2004 · R · [Image Quality Assessment: From Error Visibility to Structural Similarity](https://doi.org/10.1109/TIP.2003.819861)
- C07 · 2003 · R · [Multiscale structural similarity for image quality assessment](https://doi.org/10.1109/ACSSC.2003.1292216)
- C08 · 2017 · R · [Loss Functions for Image Restoration With Neural Networks](https://doi.org/10.1109/TCI.2016.2644865)
- C09 · 2020 · S · [Denoising Diffusion Probabilistic Models](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)
- C10 · 2022 · R · [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- C11 · 2021 · R · [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- C12 · 2015 · R · [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://proceedings.mlr.press/v37/sohl-dickstein15.html)

### D｜神经时序、潜动力学与自监督表征（30 篇）

关联正文 §2.2、§5.6：区分从视觉放电解码图像、预测下一时刻放电、以及从运动皮层预测行为。后两类可用于比较时间表征，但其预测目标不是字形。

- D01 · 2023 · S · [A Unified, Scalable Framework for Neural Population Decoding](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8ca113d122584f12a6727341aaf58887-Abstract-Conference.html)
- D02 · 2023 · S · [Neural Data Transformer 2: Multi-context Pretraining for Neural Spiking Activity](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fe51de4e7baf52e743b679e3bdba7905-Abstract-Conference.html)
- D03 · 2021 · S · [Representation Learning for Neural Population Activity with Neural Data Transformers](https://doi.org/10.1101/2021.01.16.426955)（bioRxiv 版本）
- D04 · 2022 · S · [STNDT: Modeling Neural Population Activity with Spatiotemporal Transformers](https://proceedings.neurips.cc/paper_files/paper/2022/hash/72163d1c3c1726f1c29157d06e9e93c1-Abstract-Conference.html)
- D05 · 2024 · S · [Towards a ‘Universal Translator’ for Neural Dynamics at Single-Cell, Single-Spike Resolution](https://proceedings.neurips.cc/paper_files/paper/2024/hash/934eb45b99eff8f16b5cb8e4d3cb5641-Abstract-Conference.html)
- D06 · 2025 · S · [Neural Encoding and Decoding at Scale](https://arxiv.org/abs/2504.08201)（预印本）
- D07 · 2025 · S · [Population Transformer: Learning Population-level Representations of Neural Activity](https://arxiv.org/abs/2406.03044)
- D08 · 2023 · S · [Capturing Cross-session Neural Population Variability through Self-supervised Identification of Consistent Neuron Ensembles](https://proceedings.mlr.press/v197/jude23a.html)
- D09 · 2025 · S · [Simple Temporal Attention Beats Complex Decoders for Neural-to-Visual Mapping from Primate Spiking Data](https://openreview.net/forum?id=zGZ7MjffUR)（研讨会论文）
- D10 · 2023 · S · [The Brain-inspired Decoder for Natural Visual Image Reconstruction](https://doi.org/10.3389/fnins.2023.1130606)
- D11 · 2022 · S · [Fusion of ANNs as Decoder of Retinal Spike Trains for Scene Reconstruction](https://doi.org/10.1007/s10489-022-03402-w)
- D12 · 2024 · S · [Robust Sensory Information Reconstruction and Classification With Augmented Spikes](https://ieeexplore.ieee.org/document/10547380/)
- D13 · 2025 · S · [Spatio-temporal Transformers for Decoding Neural Movement Control](https://doi.org/10.1088/1741-2552/adaef0)
- D14 · 2026 · S · [Leveraging Unlabelled Data for Generalizable Neural Population Decoding](https://arxiv.org/abs/2607.14086)（预印本）
- D15 · 2026 · S · [RPNT: Robust Pre-trained Neural Transformer — A Pathway for Generalized Motor Decoding](https://arxiv.org/abs/2601.17641)（预印本）
- D16 · 2022 · S · [A Large-scale Neural Network Training Framework for Generalized Estimation of Single-trial Population Dynamics](https://doi.org/10.1038/s41592-022-01675-0)
- D17 · 2022 · S · [A Deep Learning Framework for Inference of Single-trial Neural Population Dynamics from Calcium Imaging with Subframe Temporal Resolution](https://doi.org/10.1038/s41593-022-01189-0)
- D18 · 2018 · S · [Inferring Single-trial Neural Population Dynamics Using Sequential Auto-encoders](https://doi.org/10.1038/s41592-018-0109-9)（LFADS，亦关联 E 组）
- D19 · 2021 · S · [Neural Latents Benchmark ’21: Evaluating Latent Variable Models of Neural Population Activity](https://arxiv.org/abs/2109.04463)
- D20 · 2022 · S · [Decoding EEG With Spiking Neural Networks on Neuromorphic Hardware](https://openreview.net/forum?id=ZPBJPGX3Bz)
- D21 · 2010 · S · [A New Look at State-space Models for Neural Data](https://doi.org/10.1007/s10827-009-0179-x)
- D22 · 2014 · S · [Latent State-space Models for Neural Decoding](https://ieeexplore.ieee.org/document/6944262)
- D23 · 2019 · S · [End-to-end Hand Kinematic Decoding from LFPs Using Temporal Convolutional Network](https://ieeexplore.ieee.org/document/8919131/)
- D24 · 2026 · S · [Implicit Behavioral Decoding from Next-Step Spike Forecasts at Population Scale](https://arxiv.org/abs/2605.12999)（预印本）
- D25 · 2026 · S · [SpikeProphecy: A Large-Scale Benchmark for Autoregressive Neural Population Forecasting](https://arxiv.org/abs/2605.12992)（预印本）
- D26 · 2025 · S · [Generalizable, Real-time Neural Decoding with Hybrid State-space Models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4a40f444c3b0e6486586dabd86fbede4-Abstract-Conference.html)
- D27 · 2025 · S · [SPINT: Spatial Permutation-Invariant Neural Transformer for Consistent Intracortical Motor Decoding](https://proceedings.neurips.cc/paper_files/paper/2025/hash/24c8d2fe8520746f0084174b75b93ebe-Abstract-Conference.html)
- D28 · 2023 · S · [Towards Robust and Generalizable Representations of Extracellular Data Using Contrastive Learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/83c637c3bc0ca88eda6cf4f5f45bdced-Abstract-Conference.html)
- D29 · 2025 · S · [Temporal Spiking Generative Adversarial Networks for Heading Direction Decoding](https://www.sciencedirect.com/science/article/pii/S0893608024009043)
- D30 · 2025 · S · [Decoding Natural Visual Scenes via Learnable Representations of Neural Spiking Sequences](https://doi.org/10.1016/j.neunet.2025.107863)

### E｜跨日稳健性、原型度量与单试次不确定性（33 篇）

关联正文 §2.4、§5.5–5.6：E01–E07 讨论神经群体稳定性与子空间；E08–E21 区分域适配和域泛化；E22–E25 对应已见字检索；E26–E33 区分单试次噪声、预测不确定性与校准。无标注目标日适配不意味着可以用目标日标签选参。

- E01 · 2020 · S · [Stabilization of a brain–computer interface via the alignment of low-dimensional spaces of neural activity](https://doi.org/10.1038/s41551-020-0542-9)
- E02 · 2023 · S · [Using adversarial networks to extend brain computer interface decoding accuracy over time](https://elifesciences.org/articles/84296)
- E03 · 2023 · S · [Extracting Semantic-Dynamic Features for Long-Term Stable Brain Computer Interface](https://ojs.aaai.org/index.php/AAAI/article/view/25738)
- E04 · 2025 · S · [CRRL: Learning Channel-invariant Neural Representations for High-performance Cross-day Decoding](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7cd4afed8c577d879775f24259f8e261-Abstract-Conference.html)
- E05 · 2011 · R · [Long-term stability of neural prosthetic control signals from silicon cortical arrays in rhesus macaque motor cortex](https://doi.org/10.1088/1741-2560/8/4/045005)
- E06 · 2020 · R · [Long-term stability of cortical population dynamics underlying consistent behavior](https://doi.org/10.1038/s41593-019-0555-4)
- E07 · 2014 · S · [Cortical activity in the null space: permitting preparation without movement](https://doi.org/10.1038/nn.3643)
- E08 · 2016 · S · [Domain-Adversarial Training of Neural Networks](https://www.jmlr.org/papers/v17/15-239.html)
- E09 · 2016 · R · [Return of Frustratingly Easy Domain Adaptation](https://ojs.aaai.org/index.php/AAAI/article/view/10306)
- E10 · 2016 · R · [Deep CORAL: Correlation Alignment for Deep Domain Adaptation](https://doi.org/10.1007/978-3-319-49409-8_35)
- E11 · 2015 · R · [Learning Transferable Features with Deep Adaptation Networks](https://proceedings.mlr.press/v37/long15.html)
- E12 · 2018 · R · [Conditional Adversarial Domain Adaptation](https://proceedings.neurips.cc/paper/2018/hash/ab88b15733f543179858600245108dd8-Abstract.html)
- E13 · 2017 · R · [Revisiting Batch Normalization For Practical Domain Adaptation](https://arxiv.org/abs/1603.04779)（作者稿）
- E14 · 2020 · R · [Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation](https://proceedings.mlr.press/v119/liang20a.html)
- E15 · 2020 · R · [Universal Source-Free Domain Adaptation](https://openaccess.thecvf.com/content_CVPR_2020/html/Kundu_Universal_Source-Free_Domain_Adaptation_CVPR_2020_paper.html)
- E16 · 2021 · R · [Tent: Fully Test-time Adaptation by Entropy Minimization](https://arxiv.org/abs/2006.10726)（作者稿）
- E17 · 2020 · R · [Test-Time Training with Self-Supervision for Generalization under Distribution Shifts](https://proceedings.mlr.press/v119/sun20b.html)
- E18 · 2022 · R · [Continual Test-Time Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html)
- E19 · 2022 · R · [Efficient Test-Time Model Adaptation without Forgetting](https://proceedings.mlr.press/v162/niu22a.html)
- E20 · 2019 · R · [Domain Generalization by Solving Jigsaw Puzzles](https://openaccess.thecvf.com/content_CVPR_2019/html/Carlucci_Domain_Generalization_by_Solving_Jigsaw_Puzzles_CVPR_2019_paper.html)
- E21 · 2019 · R · [Domain Generalization via Model-Agnostic Learning of Semantic Features](https://arxiv.org/abs/1910.13580)（作者稿）
- E22 · 2017 · R · [Prototypical Networks for Few-shot Learning](https://proceedings.neurips.cc/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html)
- E23 · 2020 · R · [Supervised Contrastive Learning](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)
- E24 · 2020 · R · [A Simple Framework for Contrastive Learning of Visual Representations](https://proceedings.mlr.press/v119/chen20j.html)
- E25 · 2021 · R · [Prototypical Contrastive Learning of Unsupervised Representations](https://arxiv.org/abs/2005.04966)（作者稿）
- E26 · 2009 · R · [Gaussian-Process Factor Analysis for Low-Dimensional Single-Trial Analysis of Neural Population Activity](https://doi.org/10.1152/jn.90941.2008)
- E27 · 2022 · S · [Targeted dimensionality reduction enables reliable estimation of neural population coding accuracy from trial-limited data](https://doi.org/10.1371/journal.pone.0271136)
- E28 · 2018 · S · [Modeling correlated noise is necessary to decode uncertainty](https://arxiv.org/abs/1708.04860)（作者稿）
- E29 · 2017 · R · [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)
- E30 · 2017 · R · [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)
- E31 · 2019 · R · [Can you trust your model’s uncertainty? Evaluating predictive uncertainty under dataset shift](https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html)
- E32 · 2016 · R · [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://proceedings.mlr.press/v48/gal16.html)
- E33 · 2017 · R · [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://proceedings.neurips.cc/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html)

### F｜正文复现对象、汉字笔画与形状评价（7 篇）

F01 是正文 §4.1 的直接复现对象。F02–F04 区分字的笔画组成和字体风格；F05–F07 分别衡量连通性、轮廓偏差与形状对应，可解释两幅字图为何不同，不是官方 OCF 的替代定义。

- F01 · 2024 · V · [Decoding dynamic visual scenes across the brain hierarchy](https://doi.org/10.1371/journal.pcbi.1012297)（PLOS Computational Biology；正文 [cite:14]）
- F02 · 2021 · V · [Zero-Shot Chinese Character Recognition with Stroke-Level Decomposition](https://www.ijcai.org/proceedings/2021/85)（IJCAI 主会；正文 [cite:20]）
- F03 · 2021 · V · [DG-Font: Deformable Generative Networks for Unsupervised Font Generation](https://openaccess.thecvf.com/content/CVPR2021/html/Xie_DG-Font_Deformable_Generative_Networks_for_Unsupervised_Font_Generation_CVPR_2021_paper.html)（CVPR）
- F04 · 2021 · V · [Multiple Heads Are Better Than One: Few-Shot Font Generation With Multiple Localized Experts](https://openaccess.thecvf.com/content/ICCV2021/html/Park_Multiple_Heads_Are_Better_Than_One_Few-Shot_Font_Generation_With_ICCV_2021_paper.html)（ICCV）
- F05 · 2021 · V · [clDice - A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html)（CVPR）
- F06 · 2021 · V · [Boundary IoU: Improving Object-Centric Image Segmentation Evaluation](https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html)（CVPR）
- F07 · 2002 · V · [Shape matching and object recognition using shape contexts](https://doi.org/10.1109/34.993558)（IEEE TPAMI；Crossref 核对题名、卷期及页码 509–522）

<a id="academic-evidence-end"></a>
