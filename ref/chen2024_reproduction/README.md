# Chen et al. (2024) 官方模型的 PyTorch 兼容移植

**2026-11-24移植/修改自beizai/Decoding**，仅移植 `train.py` 中的 `dense_decoder(ncell)` 与 `CAE((64, 64, 1))`，未修改原仓库。

- 论文：Chen et al., *Decoding dynamic visual scenes across the brain hierarchy* (2024)
- DOI：https://doi.org/10.1371/journal.pcbi.1012297
- 原仓库：https://github.com/beizai/Decoding
- 对应 commit：[`631ebbcec0eedb907dd1ec24a908c4f99324c219`](https://github.com/beizai/Decoding/tree/631ebbcec0eedb907dd1ec24a908c4f99324c219)
- 原始模型定义：[`train.py`](https://github.com/beizai/Decoding/blob/631ebbcec0eedb907dd1ec24a908c4f99324c219/train.py)
- 许可证：GPL-3.0，完整条款见 `LICENSE`（从原仓库原样复制）。

## 使用

```python
from model import Chen2024Decoder

model = Chen2024Decoder(ncell=800)  # ncell 可配置，默认与原示例一致为 800
dense_out, cae_out = model(spikes)  # spikes: N x ncell
```

`dense_out` 的形状为 `N x 1 x 64 x 64`，`cae_out` 为 `N x 1 x 256 x 256`。本移植采用 PyTorch 的 NCHW 布局；原 Keras 模型采用 NHWC 布局。

## 逐层对应

### dense_decoder

| 顺序 | 原 Keras 层 | PyTorch 对应 |
|---|---|---|
| 1 | `Dense(16384)` | `Linear(ncell, 16384)` |
| 2 | `BatchNormalization()` + `Activation('relu')` | `BatchNorm1d(16384)` + `ReLU` |
| 3 | `Dropout(0.5)` | `Dropout(p=0.5)` |
| 4 | `Dense(8192)` | `Linear(16384, 8192)` |
| 5 | `BatchNormalization()` + `Activation('relu')` | `BatchNorm1d(8192)` + `ReLU` |
| 6 | `Dropout(0.5)` | `Dropout(p=0.5)` |
| 7 | `Dense(4096)` + `Activation('sigmoid')` | `Linear(8192, 4096)` + `Sigmoid` |
| 8 | `Reshape((64, 64, 1))` | reshape 为 `N x 1 x 64 x 64` |

### CAE

下表中的每个卷积均带 bias；BN 后是否接 ReLU/Dropout 严格按原实现排列。

| 阶段 | 卷积与后续层 | 上采样 | 输出空间尺寸 |
|---|---|---|---|
| 输入 | — | — | `1 x 64 x 64` |
| 1 | `Conv 1→256, k7, s2, p3` + BN + ReLU | — | `256 x 32 x 32` |
| 2 | `Conv 256→512, k5, s2, p2` + BN + ReLU + Dropout(0.25) | — | `512 x 16 x 16` |
| 3 | `Conv 512→1024, k3, s2, p1` + BN + ReLU + Dropout(0.25) | — | `1024 x 8 x 8` |
| 4 | `Conv 1024→1024, k3, s2, p1` + BN + ReLU + Dropout(0.25) | — | `1024 x 4 x 4` |
| 5 | `Conv 1024→1024, k3, s1, p1` + BN + ReLU + Dropout(0.25) | 前置 2 倍 nearest | `1024 x 8 x 8` |
| 6 | `Conv 1024→512, k3, s1, p1` + BN + ReLU + Dropout(0.25) | 前置 2 倍 nearest | `512 x 16 x 16` |
| 7 | `Conv 512→256, k3, s1, p1` + BN + ReLU + Dropout(0.25) | 前置 2 倍 nearest | `256 x 32 x 32` |
| 8 | `Conv 256→256, k3, s1, p1` + BN + ReLU + Dropout(0.25) | 前置 2 倍 nearest | `256 x 64 x 64` |
| 9 | `Conv 256→128, k5, s1, p2` + BN + ReLU + Dropout(0.25) | 前置 2 倍 nearest | `128 x 128 x 128` |
| 10 | `Conv 128→1, k7, s1, p1?` | 前置 2 倍 nearest | 见下方校正 |
| 输出 | `Conv 128→1, k7, s1, p3` + **BN（无激活）** | — | `1 x 256 x 256` |

所有 `same` padding 均转换为对应奇数卷积核的显式 padding。Keras BN 默认 `epsilon=0.001`、`momentum=0.99`；PyTorch 对应为 `eps=1e-3`、`momentum=0.01`，因为 PyTorch 的 momentum 表示新批次统计量的权重。

### 上采样与输出激活的关键区别

- 原 Keras `UpSampling2D((2, 2))` 默认 `interpolation='nearest'`，因此这里六次上采样全部使用 `nn.Upsample(scale_factor=2, mode='nearest')`，不是双线性插值，也不是转置卷积。
- `dense_out` 在 reshape **之前**经过 sigmoid，所以数值范围限定在 `[0, 1]`。
- `cae_out` 的最后一层是 **BatchNormalization**，之后没有 sigmoid 或其他激活；其输出不保证在 `[0, 1]`，也不保证非负。BN 不是 sigmoid，不能把 CAE 输出直接解释为已归一化图像概率。

## 参数量

默认 `ncell=800` 时，可训练参数共 **215,168,003**：

- `DenseDecoder`：180,957,184
- `CAE`：34,210,819

其他 `ncell` 的总参数量为 `215,168,003 + 16,384 * (ncell - 800)`；BN 的 running mean/variance 是缓冲区，不计入可训练参数。

## 最小 CPU 软件测试

使用项目现有虚拟环境，不安装依赖、不下载数据：

```powershell
D:\WORK\siclink\.venv\Scripts\python.exe D:\WORK\siclink\ref\chen2024_reproduction\test_model.py
```

测试使用 `ncell=8`、batch size 2，强制 CPU，检查两个输出形状、有限值、`dense_out` 的 sigmoid 范围，并对 CAE 的 MSE 损失执行一次完整反向传播和 Adam step，同时检查梯度有限且参数确实更新。

**测试中的张量均为随机合成数据，仅用于验证软件可运行性，不是实验数据，也不构成科研结果或论文指标复现。**

## 与原实现的差异与边界

- 仅数据布局从 NHWC 改为 NCHW；层数、顺序、卷积核、stride、padding、BN、Dropout、激活、nearest 上采样及输出空间尺寸均对应原实现。
- 本类的 `forward` 同时返回 `(dense_out, cae_out)`，对应原仓库的 multi-output 模型；原训练模型的损失只作用于 `cae_out`，最小测试也遵循这一点。
- 未读取或转换官方训练权重。Keras 与 PyTorch 的默认随机初始化及底层浮点算子不同，因此随机初始化后的逐元素输出不应期望一致；本测试也不是数值一致性或科研复现验证。
- 已在作者示例数据上完成 30 epochs 的缩短训练，并复算了作者发布结果的指标；完整命令、曲线与结论边界见 [REPRODUCTION.md](REPRODUCTION.md)。该训练未达到论文 400 epochs 的数值，也不能与 SiClink 的 OCF 分数比较。
