# 实验23：固定候选集上限诊断

从仓库根目录运行 `D:\WORK\siclink\.venv\Scripts\python.exe -B -u test/23_candidate_oracle/run.py`。本次用 CUDA，按 Task1/2/3 顺序运行，无训练神经网络、无上传、不访问隐藏 GT。基线取自 `src/final_config.json`；实际加载实验05/04/18的对应折内权重，而非全量训练提交模型。

**抽样与隔离。** 沿用实验18的 `prepare_task1` / `prepare_seen`，`make_split(seed=42)`。Task1 按字留出 10%，训练不含这些字的任何试次；在留出字中按 seed 123 打乱后的前16字构成 tune。Task2 留出两鼠的 session4，重复试次按字取均值；留出日的 session 归一化使用该日无标签特征。Task3 每鼠日每字的四次中留一次、其余三次用于训练；预测使用留出的单试次。Task2/3 的 tune 都是对全图索引按 seed 123 打乱后的前256字。各任务对排好序的 tune 用 seed 23 重排取前4字，同任务各验证块用相同四字。预算事先固定，每块最多 4×(128+1) 次图对请求。样本数是字×鼠日，不是独立字数。tune 参与过既有模型选择或校准，结果不是独立盲测。

**候选。** Task1：25 ms、seed 42 回归 DCT32 系数，解码连续图；对 NotoSansSC Regular 52 的 6763 个 GB2312 字形比较 directional2 归一化嵌入内积。Task2/3：50 ms；Task2 平均 42/202/777 原始 logits，Task3 用 42；逐行标准化 logits 与标准化编码分数等权相加，2748 个候选为公开训练图。编码岭回归仅从对应折训练试次重建（alpha 分别为 1、0.01）。候选均按逐行分数降序，平分按较小索引优先，不用目标图像选候选。Task2/3 的实际输出在完整 2748 行块上做 Hungarian 一对一指派，不能用四行局部指派或逐行 argmax 代替。

**评分与结果字段。** `results.json` 中 `overall` 为按样本等权均值，`by_block` 为各鼠日，`rows` 保留候选排序、实际输出和每样本指标。对预测图和公开参考图走 `src.evaluation.score_predictions` 的 PNG 量化和官方 OCF；按两个二值图的有序像素内容跨块跨任务缓存，每对只评分一次。`oracle_ocf[k]` 是各行前 k 个候选对公开参考图的最高 OCF 的平均，不施加跨行唯一约束，不是可部署预测。`actual_top1_ocf` 是基线最终输出 OCF。`recall[k]` 为真正图像身份出现在该行前 k 的比例，`actual_accuracy` 是最终身份命中率；两者不以图像去重后的序号计算。Task1 缺少可核实的字库到真实身份映射，二者均为 null；Task1 的 actual_index 和 candidates 仅指字库位置。
