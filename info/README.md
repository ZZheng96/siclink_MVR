# SiClink 比赛资料索引

- 比赛：[SiClink Mouse Visual Reconstruction BCI Competition](https://www.kaggle.com/competitions/siclink-mouse-visual-reconstruction-bci-challenge)
- 网页归档日期：2026-11-23（Asia/Shanghai）
- 用途：集中保存比赛网页、当前榜单和本地数据/示例的核查结果，便于后续离线查阅。
- 网页正文保留英文；动态人数、社区内容和榜单仅为抓取时快照。若与线上页面不一致，以最新官方页面和正式规则为准。

## 文档导航与覆盖范围

| 范围 | 本地文档 | 内容 |
|---|---|---|
| Overview | [overview.md](overview.md) | 任务背景、三任务协议、OCF、提交格式、奖金、社区、时间线、引用和参与人数快照 |
| Data | [data.md](data.md) | 官方数据说明、完整划分、目录、loader、示例训练/预测、局部评分与 Kaggle 转换步骤 |
| Rules：比赛专属及并入的一般条款 | [rules.md](rules.md) | A/B/C 完整条款；B 的 8 节和 C 的 18 节 |
| Rules：Kaggle Foundational Rules | [kaggle_foundational_rules.md](kaggle_foundational_rules.md) | 网页另列的基础规则完整 18 节，未与比赛专属条款混合或删减 |
| Code / Models / Discussion | [community.md](community.md) | 2 个公开 notebook、Models 空状态、1 个讨论主题及官方澄清评论 |
| Leaderboard | [leaderboard.md](leaderboard.md) | 用户提供的 CSV 中 42 支队伍全量转录 |
| 本地数据与示例 | [local_dataset.md](local_dataset.md) | 实际路径、完整文件组织、NPZ 结构、官方示例及提交适配器核查 |
| 机理研究与复现方案 | [../ref/siclink_research_and_solution.md](../ref/siclink_research_and_solution.md) | 文献与科学机理、模型对比、Chen 2024 与官方 DDPM 复现、可行方案 |

已检查比赛网页当时可见的 **Overview、Data、Code、Models、Discussion、Leaderboard、Rules** 七个标签页；Rules 页中分开显示的两套规则均已保存。

## 核心事实速览

目标是从小鼠视觉皮层的脉冲活动重建其看到的 **64 × 64 单通道灰度汉字图像**（黑底白字）。每次刺激呈现 250 ms，采样率 30 kHz；两只鼠分别为 134 和 78 个神经 units。

| 划分 | 样本数 | 每样本 trial 数 | 泛化要求 |
|---|---:|---:|---|
| Train | 87,936 | 1 | 两只鼠 session1–4，2,748 个已见汉字，每个保留 4 次重复 |
| Task 1 | 2,720 | 20 | 同 session、训练未见汉字 |
| Task 2 | 5,496 | 4 | 已见汉字、跨天 session5 |
| Task 3 | 21,984 | 1 | 同 session、已见汉字、单 trial |
| Test 合计 | 30,200 | — | 三个任务等权 |

OCF 越高越好，范围 0–100。先分别计算每个任务内所有图像的 OCF 算术平均值，再计算 `S_final = (S1 + S2 + S3) / 3`；不能直接对 30,200 个样本做一次样本加权平均来代替任务等权。

- 最终提交截止：**2027-01-20 10:00 UTC**（北京时间 18:00）。
- 每队每天（UTC）最多 **1 次**提交，最多选择 1 个 Final Submission；每队最多 **5 人**。
- 总奖金 **10,000 USD（税前）**，奖励符合资格的前 10 队。
- 比赛专属规则说明：全部 30,200 个测试样本用于 Public Leaderboard，没有另外的 private test set；最终获奖仍须审核和验证。

## 容易出错的事项

1. **先二值化再输出。** Discussion 中官方评论确认 GT 为 `{0,1}`，建议提交先二值化。原始抗锯齿 GT 自评分约 70.8，而 `(gt >= 128) * 255` 的自评分为 100；参见 [community.md](community.md)。
2. **Kaggle 上传的不是原始 PNG 包。** 官方示例先生成含 `task1/`、`task2/`、`task3/` 的 `submission.zip`，再用 `convert_to_kaggle.py` 转成 `kaggle_submission.zip`。后者包含 30,200 行、两列 `sample_id,prediction` 的 CSV，prediction 是 64 × 64 mode-L PNG 的标准 base64。
3. **每个 sample 输出一张图，不是每个 trial 一张图。** Task 1 和 Task 2 可使用对应 sample 内任意子集或组合的重复 trials。
4. **规则文本存在模板差异。** 比赛专属条款明确“无单独私有测试集、按最佳有效 Public Leaderboard 分数确定获奖排名”；另列的 Foundational Rules 仍含“Private Leaderboard 决定最终排名”的通用模板文字，并声明冲突时基础规则优先。两份原文均完整保留，不在本地文档中擅自消解；实际执行应以主办方最新正式说明为准。
5. **提交材料与许可义务较重。** 比赛规则要求参与者在主办方要求后 10 个工作日内交付足以复现提交的代码、权重、文档等材料，并赋予主办方规则所述的广泛非独占许可；数据仅限比赛及非商业/学术研究使用。外部数据和工具须满足 B(6) 的可获得性与 Reasonableness Standard，第三方组件及许可必须明确列出。原网页在 B(6)(b) 末尾冒号后直接进入 B(7)，未显示所述标准列表；在线复核确认并非归档漏抄。务必阅读 [rules.md](rules.md) 的 B(4)–B(7)。

## 本地资料说明

用户消息中的数据路径 `d:/WORK/siclink/siclink-mouse-visual-reconstruction-bci-challenge` 与实际磁盘布局不同；本地实际位置、文件清单和格式核查见 [local_dataset.md](local_dataset.md)。原始数据、榜单 CSV 和网页快捷方式仍保留在工作区，未复制进 `info/` 造成重复。
