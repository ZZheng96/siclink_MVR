# 官方条件 DDPM 复现记录

- 复现日期：2026-11-24
- 原始来源：SiClink Kaggle Data 中的 `participant_package_en`
- 工作目录：`D:\WORK\siclink\ref\official_ddpm_reproduction`
- Python：`D:\WORK\siclink\.venv\Scripts\python.exe`
- 设备：NVIDIA GeForce RTX 3080 Laptop GPU，16 GiB，CUDA/BF16 可用

本目录是独立副本；原始参赛包未修改。所有命令均应在本目录执行，并显式使用上述 `.venv` 解释器，不使用系统 Python。

## 1. 环境

实测核心版本：PyTorch 2.5.1+cu121、diffusers 0.35.2、NumPy 1.26.4、SciPy 1.13.1、Pillow 12.3.0。

环境缺失包通过以下方式补齐；镜像仅用于包下载，不改变代码来源：

```powershell
& 'D:\WORK\siclink\.venv\Scripts\python.exe' -m ensurepip --upgrade
& 'D:\WORK\siclink\.venv\Scripts\python.exe' -m pip install `
  --index-url 'https://mirrors.aliyun.com/pypi/simple/' `
  --trusted-host mirrors.aliyun.com Pillow 'diffusers==0.35.2'
```

## 2. Demo 冒烟复现

数据为包内 `data/`：8 个训练 trial；Task 1/2/3 各 2 个测试样本。

```powershell
& 'D:\WORK\siclink\.venv\Scripts\python.exe' -B -m example.read_data --data data
& 'D:\WORK\siclink\.venv\Scripts\python.exe' -B -m example.train --data data --output runs/demo_e1 --epochs 1 --device cuda
& 'D:\WORK\siclink\.venv\Scripts\python.exe' -B -m example.predict --data data --checkpoints runs/demo_e1 --output outputs/demo_e1 --device cuda --batch-size 4
& 'D:\WORK\siclink\.venv\Scripts\python.exe' -B validate_demo.py
```

训练结果（日志 `logs/demo_e1_train.log`，输出 `runs/demo_e1_verify`）：

| 分组 | epochs | train loss | validation loss |
|---|---:|---:|---:|
| mice1/session1 | 1 | 0.436826 | 0.393969 |
| mice1/across_sessions | 1 | 0.436826 | 0.393969 |
| mice2/session1 | 1 | 0.438975 | 0.395493 |
| mice2/across_sessions | 1 | 0.438975 | 0.395493 |

demo 仅 8 个训练 trial，官方脚本未固定划分与初始化随机种子，重复运行 loss 会变化；该表只用于确认链路可运行。

验证脚本实测输出：

```json
{
  "checkpoints": 4,
  "png_files": 6,
  "zip_entries": 6,
  "ocf_identical": 100.0,
  "ocf_blank": 0.0,
  "status": "PASS"
}
```

产物：`runs/demo_e1/`、`outputs/demo_e1/`、`outputs/demo_e1.zip`。预测脚本拒绝覆盖已有输出；重复实验应使用新的输出目录名。

## 3. 正式数据 Task 2 pilot

正式数据路径：

```text
D:\WORK\siclink\data\siclink-mouse-visual-reconstruction-bci-challenge\siclink_chinese_character_decoding_dataset
```

训练命令：

```powershell
& 'D:\WORK\siclink\.venv\Scripts\python.exe' -B -m example.train `
  --data 'D:\WORK\siclink\data\siclink-mouse-visual-reconstruction-bci-challenge\siclink_chinese_character_decoding_dataset' `
  --output runs/task2_mice1_e3 --task task2 --groups mice1/across_sessions `
  --epochs 3 --batch-size 32 --device cuda
```

预测命令：

```powershell
& 'D:\WORK\siclink\.venv\Scripts\python.exe' -B -m example.predict `
  --data 'D:\WORK\siclink\data\siclink-mouse-visual-reconstruction-bci-challenge\siclink_chinese_character_decoding_dataset' `
  --checkpoints runs/task2_mice1_e3 --output outputs/task2_mice1_e3 `
  --task task2 --mouse mice1 --batch-size 16 --device cuda
```

训练与预测均以退出码 0 完成，未发生 CUDA OOM，无需调整 batch。

| Epoch | train loss | validation loss |
|---:|---:|---:|
| 1 | 0.049108 | 0.076901 |
| 2 | 0.018557 | 0.022562 |
| 3 | 0.015854 | 0.016299 |

独立校验结果：

- 训练样本 39,568，验证样本 4,400。
- checkpoint 的 `inputs=670`、`epoch=3`，`strict=True` 加载成功。
- 对应 block：`test/task2/mice1/session5.npz`。
- 输出恰好 2,748 张 PNG；缺失 ID 0、额外 ID 0。
- 全部为 64×64、mode `L`，逐张 `verify()` 通过。

日志：

- `logs/task2_mice1_e3_train.log`
- `logs/task2_mice1_e3_predict.log`
- `logs/task2_mice1_e3_validation.log`

首次校验命令仅因 PowerShell 多行 `python -c` 引号问题产生 `SyntaxError`；改用标准输入执行相同校验后通过。该失败不涉及重新训练或预测，完整记录保留在校验日志中。

## 4. 结论边界

- demo 只有 8 个训练 trial，只用于验证软件链路。
- 正式 pilot 证明正式数据上的训练、采样和导出可以运行。
- denoising validation loss 不是 OCF；本地没有隐藏测试 GT，因此不报告或暗示测试 OCF、排行榜分数或重建质量。
- 下一阶段若要比较模型质量，必须在公开训练数据上建立留字、留日和单 trial 验证，并使用官方 OCF。
