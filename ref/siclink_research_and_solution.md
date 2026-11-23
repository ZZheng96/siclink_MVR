# SiClink 小鼠视觉重建赛题：科学机理、相关工作、复现与可行方案

- 调研与复现日期：2026-11-24
- 赛题：[SiClink Mouse Visual Reconstruction BCI Competition](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge)
- 本地复现环境：`D:\WORK\siclink\.venv\Scripts\python.exe`
- 证据标记：**[小鼠-spike]**、**[其他物种/脑区-spike]**、**[钙成像]**、**[人-fMRI]**、**[方法迁移]**、**[本项目假设]**。

> 不同物种、脑区、记录模态、刺激与指标之间不能直接比较数值。本文将论文已证实的结论与针对本赛题的设计推论分开；没有把 fMRI、钙成像或视网膜结果当作小鼠 V1 spike 汉字重建的直接证据。

## 摘要

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
