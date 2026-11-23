# Chen et al. (2024) 复现记录

- 复现日期：2026-11-24
- 论文：Chen et al., *Decoding dynamic visual scenes across the brain hierarchy*, PLOS Computational Biology (2024)
- DOI：https://doi.org/10.1371/journal.pcbi.1012297
- 原仓库：https://github.com/beizai/Decoding，commit `631ebbcec0eedb907dd1ec24a908c4f99324c219`，GPL-3.0
- 解释器：`D:\WORK\siclink\.venv\Scripts\python.exe`（CPython 3.10.22，Windows x64）
- GPU：NVIDIA GeForce RTX 3080 Laptop GPU 16 GiB
- 本目录是独立复现，原始仓库保留在 `ref/third_party/Decoding`，未被修改

## 1. 复现范围

| 项目 | 状态 | 说明 |
|---|---|---|
| 官方资源下载与数组校验 | 完成 | 5 个 NPY，2,015,335,040 字节，SHA256 与逐元素扫描 |
| 模型结构移植（Keras→PyTorch） | 完成 | 逐层对应，参数量与形状实测 |
| CPU 前向/反向/优化器步 | 完成 | 合成张量，仅软件测试 |
| 论文示例指标复算 | 完成 | 用作者发布的预测文件复算 395 帧 |
| 从随机初始化训练 | 完成（缩短轮数） | 30 epochs，非论文的 400 epochs |
| 迁移到 SiClink 汉字赛题 | 未做 | 见第 5 节的改造要求 |

## 2. 下载与校验

来源为 README 中两个 Google Drive 目录，清单与哈希见 `ref/downloads/chen2024_manifest.json`。

| 文件 | shape | dtype | 值域 |
|---|---|---|---|
| `movie_03_train_pic_1999_VISp_800.npy` | (3200, 256, 256) | float64 | 0.0863–1.0 |
| `movie_03_train_spike_1999_VISp_800.npy` | (3200, 800) | float64 | 0–7.9 |
| `movie_03_test_pic_1999_VISp_800.npy` | (400, 256, 256) | float64 | 0.1020–1.0 |
| `movie_03_test_spike_1999_VISp_800.npy` | (400, 800) | float64 | 0–7.3 |
| `movie_03_iter_1999_VISp_800_400.npy`（作者结果） | (400, 256, 256, 1) | float32 | 0.1000–1.262 |

校验方式：`numpy.load(mmap_mode="r", allow_pickle=False)`，逐个读取全部元素检查有限性；文件字节数与 Drive 元数据一致。

## 3. 模型与软件测试

`model.py` 按原 `train.py` 逐层移植：

- `DenseDecoder`：`Linear(ncell,16384)` → BN → ReLU → Dropout(0.5) → `Linear(16384,8192)` → BN → ReLU → Dropout(0.5) → `Linear(8192,4096)` → Sigmoid → reshape 为 `N×1×64×64`。
- `CAE`：4 次 stride-2 下采样（7/5/3/3 卷积核）后接 6 次 `nearest` 2 倍上采样与卷积，末端为 **BN 且无激活**，输出 `N×1×256×256`。
- BN 取 Keras 默认 `epsilon=0.001`、`momentum=0.99`，对应 PyTorch `eps=1e-3`、`momentum=0.01`。

实测：

```text
PASS: CPU forward, backward, and one Adam optimizer step
dense_out shape: (2, 1, 64, 64)
cae_out shape: (2, 1, 256, 256)
dense_out range: [0.087924, 0.863529]
finite MSE loss: 1.327457
```

`ncell=800` 时参数量为 215,168,003（dense 180,957,184 + CAE 34,210,819）。

## 4. 指标复算与训练

### 4.1 作者发布结果的指标复算

```powershell
Set-Location 'D:\WORK\siclink\ref\chen2024_reproduction'
& 'D:\WORK\siclink\.venv\Scripts\python.exe' -B evaluate_paper_example.py
```

按原 `test.py` 的窗口（帧 5–399，共 395 帧）复算：

| 指标 | 均值 | 中位数 |
|---|---:|---:|
| SSIM | 0.81366 | 0.83674 |
| PSNR (dB) | 25.752 | 26.217 |
| MSE | 0.004388 | 0.002389 |

说明：原 `test.py` 使用 `multichannel=True`；对单通道数组与二维 SSIM 等价，本项目改用 `channel_axis` 语义下的二维计算，并显式设置 `data_range=1.0`。

### 4.2 从随机初始化训练（缩短轮数）

```powershell
& 'D:\WORK\siclink\.venv\Scripts\python.exe' -B train_example.py --epochs 30 --batch-size 4 --output runs\example_30ep
```

- 3200 训练帧、400 测试帧；MSE、Adam `lr=1e-3`；batch 4（原作者为 16，受 16 GiB 显存限制下调）。
- 每 epoch 约 116 秒，30 epochs 共 3489 秒（约 58 分钟）。
- 日志与逐 epoch 指标：`runs/example_30ep/history.json`。

| Epoch | train loss | SSIM | PSNR (dB) |
|---:|---:|---:|---:|
| 0（随机初始化） | — | 0.0003 | 5.02 |
| 1 | 0.4883 | 0.3652 | 10.21 |
| 2 | 0.0484 | 0.5611 | 16.78 |
| 5 | 0.0192 | 0.5782 | 17.77 |
| 10 | 0.0174 | 0.5848 | 18.15 |
| 15 | 0.0157 | 0.6028 | 18.88 |
| 20 | 0.0142 | 0.6178 | 19.47 |
| 25 | 0.0132 | 0.6274 | 19.85 |
| 30 | 0.0124 | 0.6355 | 20.14 |

**结论边界**：

- 移植模型可以正常训练，测试集指标从随机初始化的 SSIM 0.0003 提升到 0.6355。
- 30 epochs 的训练预算约为论文的 7.5%（按 epoch：30/400）或 30%（按优化步数：24,000/80,000），**未达到**作者 400 epochs 结果的 SSIM 0.8137 / PSNR 25.75 dB。
- 因此本复现验证的是“结构正确 + 可训练 + 指标同向提升”，**不是**论文数值的完整复刻。
- 指标为 SSIM/PSNR，与 SiClink 的 OCF 不可比。

## 5. 迁移到 SiClink 赛题所需的改造

| 论文设置 | SiClink 设置 | 改造 |
|---|---|---|
| 800 维放电特征 | 134/78 units、250 ms | 替换输入层，改用多时间 bin 特征 |
| 256×256 灰度自然电影 | 64×64 黑底白字 | 直接使用 `dense_out` 的 64×64 分支，去掉 CAE 上采样 |
| 连续像素 MSE | OCF 结构评分 | 二值化输出，用 OCF 选阈值；可加多尺度结构辅助损失 |
| 单一自然电影任务 | 三任务泛化 | 按字/按日/单 trial 分别划分验证 |

注意：示例数据的背景约为 0.10 而非 0，刺激为自然电影；其训练权重不适合直接迁移到二值汉字轮廓任务。
