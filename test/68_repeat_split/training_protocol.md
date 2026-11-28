# SiClink 第 4 次呈现留出协议（repeat-holdout v1）

**问题：已见过同一字、同一只鼠及同一 session 的前三次呈现后，能否仅凭第四次的单 trial 神经响应识别该字并重建轮廓？** 主终点是最终输出图的官方 OCF；身份准确率和像素误差解释分数来自哪里。本文件是预注册方案，不是运行结果。repeat 是每字在本 session 的呈现次序，不是四个独立采集日，也不保证不同字的时间窗彼此先后。这里的“独立”是第 4 次 trial 不参与本实验开发，不是未见字、跨鼠、跨日或 Kaggle 盲测。历史实验用过公开完整训练块，不能承诺这批 trial 在整个项目历史上从未被使用。

## 1. 数据边界与执行顺序

根据 `HANDOFF.md` §3.2、`info/local_dataset.md`、本目录 `build.py`；以下为按拆分设计推算的应有行数，不是读取 R4 后得到的统计。Rk 表示全部 2 鼠 × 4 session 的第 k 次呈现。每组每次 2,748 个字各 1 条，每个 Rk 共 21,984 条。每鼠单独训练一个解码器，合并该鼠四天，不跨鼠拼 unit，不平均同字 trial。

|阶段|拟合数据（两鼠合计）|只预测和评分的数据|用途|
|---|---:|---:|---|
|F1|R1：21,984|R2：21,984|一次历史呈现能否推广到下一次|
|F2|R1+R2：43,968|R3：21,984|最接近最终用法的开发折|
|冻结并重训|R1+R2+R3：65,952|无|所有训练条目入模，不再切随机验证集|
|封存测试|不拟合|R4：21,984|已锁定的对照与主模型同批一次推理、一次评分|

采用扩展窗口而非随机 K 折；F1/F2 分别从零训练，折的比较是敏感性检查，不是两个独立重复实验。训练量和呈现次序同时变化，折间差异不能单独归因为漂移。

**加载与权限**

1. 以 `data/derived/repeat_split_v1/mice{m}/session{s}_repeat{r}.npz` 显式枚举允许的 `(m,s,r)`；开发环境仅挂载 R1–R3，按折再限制 fit/score 输入。不递归扫描数据树，不挂载原完整 session、历史共享缓存、全量模型、伪标签或 R4 的摘要。在独立空目录从零生成特征；缓存键含拟合集文件哈希、折、鼠、特征版本。唯一共享先验是公开 `train/pictures.npz:images`（不是某个 trial 的答案），仅使用拟合标签已覆盖的字库，预期完整 2,748 类。
2. `np.load(...,allow_pickle=False)`。样本 i 的 trial 为 `group_ptr[i]:group_ptr[i+1]`，trial j 的事件为 `trial_ptr[j]:trial_ptr[j+1]`。检查指针从 0 起、单调、终点等于相应长度；每行恰好一个 trial；两事件数组等长。空事件是真实零向量，不删。每折拟合与验证各自按清单核对 repeat、行数、合法 target 与每字计数；身份键 `(mouse,session,source_trial_index)` 不重叠。**target 是图库索引，不是可做数值回归的连续变量。** `sample_ids`、源序号、repeat、文件行号和标签仅供路由/审计，绝不入模或解码。不同 repeat 有同字是设计目标，不按字去重。
3. 仅用 trial 内 `[0,7500)` 采样点（30 kHz），按 `b=t//750` 得 10 个 25 ms 窗，对 `(b,unit)` 计数，先时间后 unit 展平。鼠1/2 的 unit 数固定为 134/78。越界即报错，不从测试推断维度。`log1p(count)` 后逐列减拟合集均值、除以总体标准差（零方差取 1）；验证和推理只 transform。各折、最终拟合分别重算，绝不按验证/测试批次计算均值。
4. 封存前记录软件、数据及模型哈希。独立推理进程只接收盲化事件视图：由隔离适配层将 R4 中事件打乱为不透明 token 与单 trial 配对（保管端固定 RNG 901）；映射保管，不传顺序、ID、target 或同字配对。已知鼠仅用于路由。先固化全部预测及哈希，再由评分进程读标签、检查 trial 不重叠、逐行一对一回连；每模型每 trial 一张图、一条预测，不准丢失、重复、人工替换。若发现完整性错误，作废，不据错误样本修模型。禁止批量一一分配、测试时训练、测试归一化、伪标签、按测试频率校准、跨 trial 平均及看测试样例选参数。

## 2. 冻结的模型、训练与选择规则

本协议选用可从零复现的简化判别基线，借鉴 v5 的单 trial 全库识别思路；**不是复刻 v5**，不引用旧 OCF 作新协议对照。

- **B（主对照）：**输入 → Linear(输入维,512) → ReLU → Dropout(0.2) → Linear(512,2748)。类别为图库索引升序；交叉熵、无类权重/标签平滑。输出 logits 取 argmax 并返回对应原图库 uint8 图。这一路同时产生分类与重建结果。
- **C（唯一候选）：**同一主干与分类头，额外加 Linear(512,4096)+sigmoid 的回归头，目标为参考二值图 `T=(images[target]>=128)`。损失 `CE + mean((P−T)^2)`，系数固定 1；分类仍由 logits 产生，最终重建固定用回归头，不事后在图库图与回归图中择优。记录两路结果，避免把分类命中误当回归成功。
- **共同训练：**PyTorch Linear 默认初始化；AdamW(lr=0.001, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01，所有参数)，batch=256，FP32，无调度/增广/预训练/梯度裁剪。种子 42、202、777；每折每鼠每模型每种子重新设置 Python/NumPy/Torch RNG，按 session→repeat→文件原行顺序拼接后每 epoch 用种子生成器连续 shuffle，最后不足一批保留；workers=0。开启确定性算法，禁 TF32/cudnn benchmark；记录 CUDA、cuBLAS 配置、设备和依赖锁定版本。开发训练到 12 epochs，仅比较 e∈{4,8,12}。两鼠使用同一配方，每次拟合各自独立参数。
- **实际决策单位为三种子集成。** `eval()`、无梯度；分类取三组 softmax 概率算术均值（同分取小索引）；C 图取三个连续输出均值，clip 到 [0,1]，乘255、四舍五入为 uint8（np.rint），不调阈值。单种子结果仅作稳定性记录。按同样代码为 B/C 各自选使 F1/F2 宏 OCF 等权均值最高的 epoch，平手取较早。若 C 不达第4节开发晋级线，则锁定 B 为主模型，不对 R4 试 C；若达标，冻结 C 为主、B 为配对对照。锁定代码、模型、轮数、种子、指标、决策规则后，再从零训练全部 R1–R3 恰好选定轮数；不得在 R4 上早停、切换种子或更新模型。F1/F2 参与选型，其均值不是无偏最终成绩。

## 3. 指标字典与聚合

每条记录是一张 64×64 图。先算每 trial，再按鼠×session 取均值，4 天等权成每鼠，2 鼠等权成主宏平均。另报总体按样本加权成绩及分母；完整均衡时两者应相等。以下全部是 final、独立逐条推理口径，没有 assignment。分数差用 OCF 点，比例差用百分点，误差用 [0,1] 灰度单位。

|指标|计算与含义|
|---|---|
|**final OCF ↑（唯一主终点）**|走官方 PNG 评分路径，预测灰度除255、GT≥128二值化，经官方受限配准、多尺度方向轮廓能量交并比×100。范围0–100。衡量最终字形轮廓，不要求身份完全正确。以仓库 `test/evaluate.py` 所用官方实现为准；不以近似余弦、像素误差或 oracle 替代。|
|Top-1 / Recall@32 ↑|真图库 ID 排第一/前32的 trial 比例；分别表示身份命中与候选覆盖。固定按概率降序、并列按 ID 升序。闭集随机命中期望 1/2748、32/2748。另报 macro-F1：全库每类 `2TP/(2TP+FP+FN)` 的均值，零分母记0；暴露少数常猜字带来的偏差。|
|NLL ↓|各条 `−log(max(p_true,1e−12))` 均值；比 top-1 更早发现对错误类别越来越自信。均匀无信号参照为 ln(2748)。|
|MAE / RMSE ↓|对最终 PNG 解码后的 P 与二值 T，先算每图像素绝对误差均值 a、平方误差均值 q。组 MAE=mean(a)，RMSE=sqrt(mean(q))；宏 RMSE 另注明为组 RMSE 等权，pooled 先合并 q 再开根。描述位置/强度误差，黑背景多不能单独用于选型。|
|前景 IoU ↑|阈值 P≥0.5 后与 T 计算前景交并比，逐图取均值；双空记1并单列数量。一张全黑图可有低 MAE，但有笔画的参考图上 IoU=0。|
|稳定性|分别列 F1/F2、每鼠、8个鼠日的 OCF 和与 B 的配对差 Δ；报最差组、非负组数、折间绝对差。单种子同 epoch 的宏 OCF 报均值、样本标准差(ddof=1)、范围；不把种子当新测试样本。|
|过拟合|每个候选 epoch 在 eval 模式按相同推理规则计算训练与验证 OCF/NLL；记 `gap_OCF=train−val`、`gap_NLL=val−train`。分折分鼠画/列轨迹，区分拟合改善、泛化停滞与两者均差。测试只做冻结后描述。|

**配对不确定性：**开发折及最终测试分别用固定 seed=31415 对 2,748 个字 ID 有放回抽样 2,000 次；每个抽中字保留两鼠四日全部 trial、模型间配对不拆，重算宏 OCF 与 Δ，取 2.5%/97.5% 分位数。区间仅条件于这两鼠四日和该字库抽样口径，不是新动物置信区间；不对 21,984 个 trial 当作独立重复检验。若诊断 oracle@32，只在预测哈希冻结后由评分端从已预测的32张图库图取最高官方 OCF；单独列为「图库候选上界」，不是 C 回归头上界或可部署分数。

## 4. 异常判定和结论规则（预设工程阈值，非历史测量值）

|条件|处置|
|---|---|
|开发访问 R4 或读取不明来源模型/缓存；源 trial 跨集合；标签/ID/顺序入模；缺文件、重复、覆盖≠100%、非法取值、非有限预测/损失|本次运行无效。记录阶段、文件和原因，不用删行后的分母报正常成绩；若已解盲，不能以同一 R4 再充当新的独立确认。|
|同折同鼠连续两个检查点 gap_OCF>3 点且 gap_NLL>0，或连续两段训练 NLL 下降而验证 NLL 上升|标记过拟合，展示轨迹；不临时增减 epoch 或搜索新超参。|
|单种子宏 OCF 标准差>1 点，或某鼠日相对 B 退化>1 点|标记不稳定，列最差层与种子；不以总体均值隐去。|
|开发晋级|C−B 的两折等权宏平均≥0.5 OCF；两折宏差及每折两只鼠的差均≥0；每折至少6/8组非负，且没有上述不稳定报警。否则只保留 B。|
|R4 一次性结论|若 C 入围，只有宏 Δ≥0.5、配对95%区间下界>0、两鼠均非负且至少6/8组非负才称“在本留出上确认增益”；否则报告未确认，不换模型复测。仅 B 时报告绝对水平、区间和各层表现，不声称优于 v5。R4 与 F2 宏 OCF 相差>3 点记为外推偏差提示，不归因为单一因素。|

## 5. 结果记录模板（空值不是 0）

**运行台账，每次拟合一行：**

```text
protocol_id | run_id | git_commit/dirty_diff_hash | UTC | OS/python/numpy/torch/CUDA/GPU | deterministic_flags | fold | mouse | model | seed | optimizer_config | epoch | fit_files+sha256 | fit_rows | eval_rows | scaler_sha256 | library_sha256 | checkpoint_sha256 | elapsed | exception
```

**逐条预测与分层评分（两表分开，封存预测时不含真值）：**

```text
run_id | blind_token | model | seed_or_ensemble | predicted_ID | top32_IDs/scores | probabilities_artifact | final_png_sha256/path
run_id | phase/train/val/test | fold | model | seed_or_ensemble | epoch | mouse/session/macro/pooled | N_expected | N_scored | OCF[CI] | top1 | recall32 | macroF1 | NLL | MAE | RMSE | IoU | delta_OCF_vs_B[CI] | oracle32_diagnostic_only
```

**决策台账，每个协议一行：**

```text
F1_delta | F2_delta | selected_epoch_B/C | seed_SD_B/C | worst_mouse_day | nonnegative_groups_F1/F2 | gap_flags | selected_primary | lock_time/hash | prediction_seal_time/hash | first_unblind_time | R4_coverage | R4_macro_OCF[CI] | R4_delta[CI] | mouse1/2_delta | nonnegative_groups_R4 | anomalies | conclusion
```

未运行填 `NOT_RUN`，不适用填 `NA`，未解盲填 `SEALED`。先留存台账再作解释。本协议只交付设计；不调用原有全量训练/打包命令，不改变生产配置或提交包。

## 6. R4 作为后续反馈基准（2026-11-28 更新）

官方 test 无标签，用户决定后续候选先以第 4 次 repeat 做开发反馈。**R4 已解封，不再是一次性盲测**；下面的指标只用于方案筛选，不能当作 Kaggle 分数或最终确认。

**固定口径**

- 训练只用 repeat1-3（每鼠、每 session、每字 3 条）；评估只用 repeat4（每字 1 条）。严格非转导候选不得读 R4 做归一化、选轮、早停或伪标签。CSLS、列归一化等测试时方法只使用 R4 的无标签分数统计，必须单独标记为 **transductive**，不与普通模型同口径。
- 任务为单 trial 身份识别，对应 v5 Task3。Task1 的 20-repeat 聚合和 Task2 的 4-repeat 平均无法由 R4 单 trial 复现，不纳入该基准。
- 主指标：Top-1、Recall@32、NLL（统一用 MLP logits）、macro-F1；同时报 argmax 与整块匈牙利分配。按鼠、session 分层；主表为两鼠等权宏平均。无官方 OCF。
- 入口：`test/68_repeat_split/r4_benchmark.py --stage train|eval|all`（方案对比）与 `r4_iterations.py`（四轮迭代）；结果在 `data/derived/repeat_bench_v1/R4_BENCHMARK.{json,md}`、`R4_ITERATIONS.{json,md}`，仓库副本在 `test/68_repeat_split/`。
- 重复使用 R4 会逐渐过拟合该集合。新改动应先预注册，比较 character/session 配对差，并报告两鼠与 8 个鼠日；不能只报宏平均。

**首轮结果（宏平均；R1-3 训练、R4 评估）**

| 方案 | 分配 | Top-1 | Recall@32 | macro-F1 | NLL |
|---|---|---|---:|---:|---:|
| v5 分支三视图（10/25/50ms，3 seeds） | 匈牙利 | **2.09%** | 20.16% | 2.09% | **6.699** |
| 53 三视图（10/25/50ms，3 seeds） | 匈牙利 | 1.87% | **20.30%** | 1.87% | **6.699** |
| v5 Task3（50ms，3 seeds） | argmax | 1.94% | 19.77% | 1.30% | 6.796 |
| v5 Task3（50ms，3 seeds） | 匈牙利 | 1.82% | 19.77% | 1.82% | 6.796 |
| v5 Task3（50ms，1 seed） | 匈牙利 | 1.62% | 17.93% | 1.62% | 7.152 |
| 48 Task3（25ms，e12，1 seed） | 匈牙利 | 0.83% | 12.15% | 0.83% | 7.874 |
| 简单 25ms MLP（上一轮 B，e4，3 seeds） | argmax | 1.16% | 14.24% | 0.96% | 7.382 |

- v5 三分支（MLP + encoding + DCT continuous）和 10/25/50ms 三视图、三种子集成均显著优于简单 25ms MLP；48 的 25ms e12 单种子在本基准最弱。
- v5 三视图与 53 三视图的 Top-1 差仅 +0.22 个百分点，95% 区间重叠，**不能据此宣称前者明确更优**。
- 匈牙利分配通常提高 macro-F1，但可能降低 Top-1，尤其 v5 三种子；两个口径都要报。
- 本基准不含连续图 OCF；若候选输出连续图，另报 MAE/RMSE/IoU，但不以像素误差替代身份指标。

**四轮迭代后的当前最优（argmax；宏平均）**

| 轮次 | 方法 | Top-1 | Recall@32 | NLL |
|---|---|---|---:|---:|---:|
| B0 | v5 三视图三种子 | 2.1698% | 20.1556% | 6.6992 |
| R1 | + CSLS k=50 | 2.3426% | 21.1790% | 6.6992 |
| R2 | + 跨分支集成 + CSLS50 | 2.3790% | 22.8439% | 6.6992 |
| R3 | + 250ms 全局视图 + CSLS50 | 2.5564% | 23.1896% | 6.6992 |
| R4 | CSLS k=100 | **2.6019%** | **23.3397%** | 6.6992 |

- R2 的 Top-1 配对区间跨 0，按预设规则以 Recall@32（+1.67pp，区间 >0）通过；R3 的 Top-1 +0.18pp（区间 >0）；R4 的 Top-1 +0.05pp（区间跨 0）、Recall@32 +0.15pp（区间 >0），只能算边际增益。
- CSLS、列归一化是 **transductive**，只使用 R4 无标签分数统计；k=100 是在 R4 上选择的，不构成独立确认。
- 下一步：CSLS + 跨分支集成可直接进入 Task3 推理链做固定公开 heldout/OcF 复核；250ms 视图需要全量重训与打包后才能进生产。

