# 本地数据与示例盘点

- 盘点日期：2026-11-24
- 用途：记录本地比赛数据的实际路径、文件组织、NPZ 结构与官方示例行为，便于离线查阅。
- 说明：用户原给的路径 `d:/WORK/siclink/siclink-mouse-visual-reconstruction-bci-challenge` 不存在；实际位置是 `d:\WORK\siclink\data\siclink-mouse-visual-reconstruction-bci-challenge`。

## 1. 总量与格式

| 项目 | 值 |
|---|---:|
| 文件数 | 87 |
| 子目录数（不含根） | 54 |
| 总大小 | 249,659,601 字节（约 238.09 MiB） |

| 格式 | 文件数 | 字节数 |
|---|---:|---:|
| NPZ | 45 | 245,458,636 |
| PY | 27 | 112,194 |
| MD | 9 | 54,212 |
| JSON | 3 | 9,051 |
| CSV | 1 | 4,025,276 |
| TXT | 2 | 232 |

目录内没有 `.ipynb`、独立图片或视频；图像都存于 NPZ，`sample_submission.csv` 内含 Base64 占位 PNG。

## 2. 目录组织

```text
siclink-mouse-visual-reconstruction-bci-challenge/
├── siclink_chinese_character_decoding_dataset/   # 正式数据，28 个文件
├── participant_package_en/participant_package_en/ # 英文参赛包，28 个文件
├── _/参赛者包和案例/                               # 中文参赛包，28 个文件
└── kaggle_submission_adapter/kaggle_submission_adapter/  # 3 个文件
```

- 正式数据：27 个 NPZ + `catalog.json`（5,471 字节）。
- 两个参赛者包的 9 个 NPZ、`catalog.json` 及全部示例代码**逐字节相同**，仅说明文档语言不同。
- 中文目录名是 `_/参赛者包和案例`，注意通配符与编码。

## 3. 正式数据清单

| 相对路径 | 文件数 | 每文件样本数 | units |
|---|---:|---:|---:|
| `train/pictures.npz` | 1 | 2,748 张图 | — |
| `train/mice1/session1–4.npz` | 4 | 各 10,992 | 134 |
| `train/mice2/session1–4.npz` | 4 | 各 10,992 | 78 |
| `test/task1/mice1/session1–4.npz` | 4 | 各 340 | 134 |
| `test/task1/mice2/session1–4.npz` | 4 | 各 340 | 78 |
| `test/task2/mice1/session5.npz` | 1 | 2,748 | 134 |
| `test/task2/mice2/session5.npz` | 1 | 2,748 | 78 |
| `test/task3/mice1/session1–4.npz` | 4 | 各 2,748 | 134 |
| `test/task3/mice2/session1–4.npz` | 4 | 各 2,748 | 78 |

`catalog.json` 共 26 个 session block（train 8、task1 8、task2 2、task3 8），公共字段含 `window_ms=250`、`stimulus_ms=250`、`sample_rate_hz=30000`、`image_size=[64,64]`。

## 4. NPZ 内部结构

session 文件为两级 CSR 式指针存储：

| 键 | 形状 | dtype | 含义 |
|---|---|---|---|
| `sample_ids` | (S,) | `<U32` | 样本 ID |
| `group_ptr` | (S+1,) | int64 | 样本对应的 trial 边界 |
| `trial_ptr` | (T+1,) | int64 | trial 对应的事件边界 |
| `spike_samples` | (E,) | int32 | 相对刺激起始的 30 kHz 采样点，0–7,499 |
| `unit_index` | (E,) | int32 | 0 基 unit 编号 |
| `target` | (S,) | int64 | **仅训练集**，指向共享图库的索引 |

- 第 i 个样本的 trial 区间为 `group_ptr[i]:group_ptr[i+1]`；第 j 条 trial 的事件区间为 `trial_ptr[j]:trial_ptr[j+1]`。
- 250 ms × 30 kHz = 7,500 个采样点；默认 1 ms 分窗时每窗 30 个采样点。
- 训练标签是图库下标，不是汉字文本；本地没有汉字字符串标签，也没有测试集答案图。

`train/pictures.npz`：键 `images`，shape `(2748, 64, 64)`，dtype `uint8`，值域 0–255，无显式通道维。

## 5. 官方 loader 与示例

`src/dataloader.py` 的 `load_dataset(data_dir, split, task, bin_ms=1)`：

- `data_dir` 必须直接包含 `catalog.json`。
- 返回 `sample_id`、`mouse`、`session`、`n_units`、`trials`（uint8，`(n_trials, n_units, n_time_bins)`）与训练集的 `image`（`(64,64)` uint8）。
- `trials` 是 0/1 二值矩阵：同一 unit 在同一时间窗内放电一次或多次都记 1。
- 默认 `bin_ms=1` 时时间维为 250；mice1 为 134 维、mice2 为 78 维。

`src/spike_counts.py` 的 `count_spikes()` 则统计实际放电次数并对重复 trial 取平均，输出 float32 `(n_samples, n_units × n_time_bins)`；官方 DDPM 默认 `bin_ms=50`，故输入维度为 670（mice1）和 390（mice2），再取 `log1p`。

示例为 `example/` 下的 Python 模块（无 notebook）：神经条件 DDPM + 64 步反向扩散 + classifier-free guidance，每个样本输出一张 64×64 mode-L PNG，由 `src/submission.py` 校验并打包 `submission.zip`。

## 6. Kaggle 适配器

`kaggle_submission_adapter/kaggle_submission_adapter/`：

- `convert_to_kaggle.py`：把 30,200 张 PNG 的 ZIP 转为只含 `submission.csv` 的 ZIP。
- `submission.csv` 两列 `sample_id,prediction`，prediction 为 64×64 mode-L PNG 的标准 Base64。
- 校验 ID 全集、无重复/缺失、单帧 64×64 mode L、单文件 ≤65,536 字节、不接受加密条目；不解码重采样，不改像素。
- `sample_submission.csv`：30,200 行（task1 2,720 / task2 5,496 / task3 21,984），首行 `test_task1_pic0`，末行 `test_task3_pic21983`，全部为黑色占位图。

## 7. 核查边界

- 文件数、大小、目录归属、`catalog.json` 的 26 个 block、两个语言包的字节一致性：**全量核对**。
- NPZ 内部键、dtype、指针与计数：**对 8 个代表性文件实际直读**（正式 train/pictures、train/mice1/session1、train/mice2/session1、task1/mice1/session1、task2/mice1/session5、task3/mice1/session1，以及英文样例的 pictures 与 train/mice1/session1），其余文件依据 catalog 与同类结构推断，未逐文件直读。
- 未实际调用 `load_dataset()` 做端到端运行时验证；返回值维度依据源码与直读结果整理。
- 未运行训练/预测/评分，因此本文不含任何精度或 OCF 结论。
- 实际复现及其结果见 [ref/siclink_research_and_solution.md](../ref/siclink_research_and_solution.md)。
