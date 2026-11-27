"""Build the self-contained SiClink visual report (Chinese) from public train data and local evaluation outputs.

Usage: .venv\\Scripts\\python.exe -B test\\91_visual_report\\build_report.py [--device cuda] [--skip-predictions]
Writes D:/WORK/siclink/siclink_visual_report.html. Hidden test targets are never read.

To update after a new experiment: append/edit its entry in test/91_visual_report/roadmap.json
(experiments / production / ceilings) and rerun. All roadmap charts in section E are derived from that file.
Section C/D read v5 config and heldout caches; --skip-predictions skips the GPU re-scoring of section D.
"""

import argparse
import base64
import csv
import html
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATASET = ROOT / "data" / "siclink-mouse-visual-reconstruction-bci-challenge" / "siclink_chinese_character_decoding_dataset"
OUTPUT = ROOT / "siclink_visual_report.html"
ROADMAP = Path(__file__).resolve().parent / "roadmap.json"
LEADERBOARD = ROOT / "siclink-mouse-visual-reconstruction-bci-challenge-publicleaderboard-2026-11-23T14_53_20.csv"
HELDOUT = {
    "task1": (ROOT / "test/.cache/evaluate_final_task1_v2/task1/heldout_final.npz",
              ROOT / "test/.cache/evaluate_final_task1_v2/results.json"),
    "task2": (ROOT / "test/.cache/evaluate_final_task23_v1/task2/heldout_final.npz",
              ROOT / "test/.cache/evaluate_final_task23_v1/resumed/results.json"),
    "task3": (ROOT / "test/.cache/evaluate_final_task23_v1/resumed/task3/heldout_final.npz",
              ROOT / "test/.cache/evaluate_final_task23_v1/resumed/results.json"),
}

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
C1, C2, C3, CG = "#2563eb", "#ea580c", "#16a34a", "#64748b"


def fig_to_img(fig, alt, dpi=90):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'


def esc(x):
    return html.escape(str(x))


def table(headers, rows, cls=""):
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def f(x, n=3):
    return f"{x:.{n}f}"


# ---------------------------------------------------------------- A. dataset
def section_dataset(pictures):
    catalog = json.loads((DATASET / "catalog.json").read_text(encoding="utf-8"))
    blocks = sorted(catalog["blocks"], key=lambda b: (b["split"] != "train", b["task"], b["mouse"], b["session"]))
    rows = [[esc(b["split"]), esc(b["task"]), esc(b["mouse"]), esc(b["session"]), b["n_units"], f'{b["n_samples"]:,}']
            for b in blocks]
    agg = {}
    for b in blocks:
        agg.setdefault((b["split"], b["task"]), [0, 0])
        agg[(b["split"], b["task"])][0] += 1
        agg[(b["split"], b["task"])][1] += b["n_samples"]
    agg_rows = [[esc(s), esc(t), n, f"{m:,}"] for (s, t), (n, m) in agg.items()]

    # glyph grid
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(pictures), 48, replace=False))
    fig, axes = plt.subplots(4, 12, figsize=(10, 3.6))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(pictures[i], cmap="gray_r", vmin=0, vmax=255)
        ax.set_title(str(i), fontsize=7)
        ax.axis("off")
    glyph_img = fig_to_img(fig, "字图示例")

    # spike data, mice1/session1 + mice2/session1
    stats = {}
    raster_img = psth_img = None
    for mouse in ("mice1", "mice2"):
        with np.load(DATASET / "train" / mouse / "session1.npz", allow_pickle=False) as z:
            group_ptr, trial_ptr = z["group_ptr"], z["trial_ptr"]
            spikes, units, target, sample_ids = z["spike_samples"], z["unit_index"], z["target"], z["sample_ids"]
        n_units = int(units.max()) + 1
        n_trials = len(trial_ptr) - 1
        unit_rate = np.bincount(units, minlength=n_units) / (n_trials * 0.25)
        hist, edges = np.histogram(spikes, bins=50, range=(0, 7500))
        psth = hist / (n_trials * n_units * (150 / 30000))
        counts = np.bincount(target, minlength=len(pictures))
        stats[mouse] = dict(n_units=n_units, n_trials=n_trials, n_samples=len(group_ptr) - 1, unit_rate=unit_rate,
                            psth=psth, edges=edges, counts=counts, n_events=len(spikes),
                            group_sizes=np.unique(np.diff(group_ptr)))
        if mouse == "mice1":
            s = 0
            t0, t1 = group_ptr[s], group_ptr[s + 1]
            e0, e1 = trial_ptr[t0], trial_ptr[t1]
            fig, (ax, axg) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [3.2, 1]})
            ax.scatter(spikes[e0:e1] / 30, units[e0:e1], s=4, c=C1, marker="|", linewidths=0.8)
            ax.set_xlim(0, 250)
            ax.set_ylim(-1, n_units)
            ax.set_xlabel("时间 (ms，刺激起始=0)")
            ax.set_ylabel("unit 编号")
            ax.set_title(f"mice1/session1 样本 {sample_ids[s]}：{e1 - e0} 个 spike 事件，{n_units} units")
            axg.imshow(pictures[target[s]], cmap="gray_r")
            axg.set_title(f"target = {target[s]}")
            axg.axis("off")
            raster_img = fig_to_img(fig, "spike raster")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for mouse, c in (("mice1", C1), ("mice2", C2)):
        st = stats[mouse]
        centers = (st["edges"][:-1] + st["edges"][1:]) / 2 / 30
        axes[0].plot(centers, st["psth"], color=c, label=f"{mouse} ({st['n_units']} units)")
        axes[1].hist(st["unit_rate"], bins=30, color=c, alpha=0.55, label=mouse)
    axes[0].set_xlabel("时间 (ms)")
    axes[0].set_ylabel("平均放电率 (Hz / unit)")
    axes[0].set_title("PSTH：session1 全部 10,992 trial 平均，5ms bin")
    axes[0].legend()
    axes[1].set_xlabel("unit 平均放电率 (Hz，0–250ms 窗)")
    axes[1].set_ylabel("unit 数")
    axes[1].set_title("每 unit 平均放电率分布 (session1)")
    axes[1].legend()
    cnt = stats["mice1"]["counts"]
    vals, freq = np.unique(cnt, return_counts=True)
    axes[2].bar([str(v) for v in vals], freq, color=C3)
    axes[2].set_xlabel("每个已见字在 train session1 中出现的 trial 数")
    axes[2].set_ylabel("字数")
    axes[2].set_title("重复结构：每字 4 次 (train)")
    psth_img = fig_to_img(fig, "PSTH 与放电率")

    srows = []
    for mouse in ("mice1", "mice2"):
        st = stats[mouse]
        srows.append([mouse, st["n_units"], f'{st["n_samples"]:,}', f'{st["n_trials"]:,}', f'{st["n_events"]:,}',
                      f'{st["n_events"] / st["n_trials"]:.1f}', f'{np.median(st["unit_rate"]):.2f}',
                      f'{st["unit_rate"].max():.1f}', esc(st["group_sizes"].tolist())])

    hierarchy = """
<div class="hier">
  <div class="hb">session<small>一只鼠一天的记录；train 每块 10,992 样本</small></div><div class="arr">→</div>
  <div class="hb">sample<small>sample_ids / target；group_ptr 指向 trial 范围</small></div><div class="arr">→</div>
  <div class="hb">trial<small>一次 250ms 呈现；trial_ptr 指向 event 范围</small></div><div class="arr">→</div>
  <div class="hb">event<small>spike_samples (0–7499 @30kHz) + unit_index</small></div>
</div>"""
    repeat = """
<div class="rep">
  <div><b>每个 session 内每个已见字共呈现 5 次</b>（2,748 已见字）</div>
  <div class="rrow"><span class="r tr">1</span><span class="r tr">2</span><span class="r tr">3</span><span class="r tr">4</span><span class="r t3">5</span></div>
  <div class="muted">前 4 次进入 train 块（group_ptr 步长 1，10,992 = 2,748×4）；第 5 次留给 Task3 测试（每场 2,748 × 1）。
  另有 340 个未见字 × 20 次构成 Task1 测试。每场合计 2,748×5 + 340×20 = 20,540 次。</div>
</div>"""
    return f"""
<section id="sec-a"><h2>A. 数据集</h2>
<p>数据来自两只小鼠（mice1 134 units，mice2 78 units）的视觉皮层 spike 记录。刺激是 64×64 二值汉字图，呈现 250ms，采样率 30kHz。</p>
<h3>A.1 层级结构</h3>{hierarchy}
<h3>A.2 catalog.json 分块统计</h3>
<div class="cols"><div>{table(["split", "task", "块数", "样本合计"], agg_rows)}</div>
<details><summary>展开全部 {len(blocks)} 个块</summary>{table(["split", "task", "mouse", "session", "units", "samples"], rows, "small")}</details></div>
<h3>A.3 字图示例（train/pictures.npz，{pictures.shape[0]}×64×64，随机 48 张）</h3>{glyph_img}
<h3>A.4 真实 spike raster（train/mice1/session1.npz 第 0 个样本）</h3>{raster_img}
<h3>A.5 放电统计与重复结构</h3>{psth_img}
{table(["mouse", "units", "samples", "trials", "spike 事件", "事件/trial", "unit 放电率中位数 Hz", "最大 Hz", "group 大小"], srows)}
{repeat}
</section>"""


# ---------------------------------------------------------------- B. tasks
def section_tasks():
    rows = [
        ["Task1", "未见字（340 个，不在 pictures.npz）", "session1–4", "20", "2,720", "跨字泛化：只能回归形状再检索/生成"],
        ["Task2", "已见字（2,748 个）", "session5（跨日，无训练数据）", "4", "5,496", "跨日漂移下的已见字识别"],
        ["Task3", "已见字（2,748 个）", "session1–4（同场第 5 次）", "1", "21,984", "单次呈现解码，SNR 最低"],
    ]
    return f"""
<section id="sec-b"><h2>B. 三个任务</h2>
{table(["任务", "字", "session", "每样本 trial", "测试样本数（两鼠合计）", "难点"], rows)}
<div class="card"><h3>OCF（Oriented Contour Fidelity）</h3>
<p>官方指标，逐张比较 64×64 预测灰度图与二值 GT 字图，范围 0–100。它比较的是方向化轮廓结构，像素级相似但笔画方向错误的图得分低。
预测在 Kaggle 端经 PNG uint8 往返；本地评估用 <code>src/evaluation.py</code> 复现 PNG 量化后调用官方 <code>src/ocf/score.py</code>。</p>
<p class="formula">总分 = (S1 + S2 + S3) / 3，S<sub>k</sub> 为 Task k 全部样本 OCF 的平均</p>
<p>Task2/3 的预测直接取自已知字图库，分类正确时 OCF=100；错误时取得的是"相似字"的部分分。Task1 的字不在库中，需要外部字库（NotoSansSC 6,763 字）检索。</p></div>
</section>"""


# ---------------------------------------------------------------- C. v5
def pipeline(title, steps, color):
    boxes = '<div class="arr">→</div>'.join(f'<div class="pb" style="border-color:{color}">{s}</div>' for s in steps)
    return f'<div class="pipe"><div class="pt" style="background:{color}">{title}</div><div class="pf">{boxes}</div></div>'


def section_v5(config, kaggle, task_results):
    t1, t2, t3 = config["task1"], config["task2"], config["task3"]
    rr = t2["reranker"]
    p1 = pipeline("Task1", [
        f"spike → {t1['bin_ms']}ms bin 计数<small>20 trial 聚合</small>",
        f"MLP 回归 {t1['basis'].upper()}<small>hidden {t1['hidden']}, dropout {t1['dropout']}, wd {t1['weight_decay']}<br>epochs m1 {t1['epochs']['mice1']} / m2 {t1['epochs']['mice2']}</small>",
        "DCT 逆变换 → 连续字形图",
        f"{t1['retrieval']} 检索<small>{t1['font_file']} size {t1['font_size']} {t1['font_variation']}<br>GB2312 6,763 字渲染库</small>",
        "输出 top1 二值字图",
    ], C1)
    p2 = pipeline("Task2", [
        f"{t2['bin_ms']}ms bin 计数<small>4 trial 聚合</small>",
        f"session_norm<small>目标场无标签统计量标准化</small>",
        f"MLP × seeds {t2['seeds']}<small>hidden {t2['hidden']}, dropout {t2['dropout']}<br>epochs m1 {t2['epochs']['mice1']} / m2 {t2['epochs']['mice2']}</small>",
        f"+ 编码模型（log1p 原型 MSE）<small>row-z 后 {t2['mlp_weight']}:{1 - t2['mlp_weight']} 融合</small>",
        f"OOF CNN listwise reranker<small>top{rr['top_n']}, k={rr['k']}<br>OOF 标签来自 {', '.join(rr['oof_sessions'])}</small>",
        "整块唯一分配 (assignment)<small>输出 2,748 库内字图</small>",
    ], C2)
    p3 = pipeline("Task3", [
        f"{t3['bin_ms']}ms bin 计数<small>单 trial</small>",
        "session_norm",
        f"MLP 分类<small>hidden {t3['hidden']}, epochs {t3['epochs']['mice1']}/{t3['epochs']['mice2']}</small>",
        f"+ 编码模型<small>alpha {t3['encoding_alpha']}</small>",
        "+ softmax DCT16 原型连续分支<small>temperature 1.0</small>",
        "三路 row-z 等权融合<small>equal_row_zscore</small>",
        "整块唯一分配",
    ], C3)

    # Kaggle bars + leaderboard
    lb = []
    with LEADERBOARD.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            lb.append((int(row["Rank"]), row["TeamName"], float(row["Score"])))
    scores = np.array([s for _, _, s in lb])
    top = scores.max()
    v3, v5 = kaggle[0]["score"], kaggle[1]["score"]
    rank_v5 = int((scores > v5).sum()) + 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 2.2]})
    ax = axes[0]
    bars = ax.bar(["v3", "v5", "榜首快照"], [v3, v5, top], color=[CG, C1, "#dc2626"])
    for b, v in zip(bars, [v3, v5, top]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.5f}", ha="center", fontsize=8)
    ax.set_ylim(0, top * 1.15)
    ax.set_title("Kaggle Public OCF（用户回报）")
    ax = axes[1]
    ax.bar(np.arange(1, len(scores) + 1), sorted(scores, reverse=True), color="#cbd5e1")
    ax.axhline(v5, color=C1, ls="--", lw=1)
    ax.text(len(scores) * 0.6, v5 + 1, f"v5 = {v5}（快照中约第 {rank_v5} 名）", color=C1)
    ax.set_xlabel("名次")
    ax.set_ylabel("Public Score")
    ax.set_title(f"公开榜快照 2026-11-23（{len(scores)} 支队伍）")
    lb_img = fig_to_img(fig, "排行榜")

    # heldout table
    anchor = {"task1": ("留字、20 repeats 聚合", "16.81193"), "task2": ("留 session4、4 repeats", "25.007633"),
              "task3": ("留同字其他 trial、单 trial", "17.555726")}
    hrows = []
    for t in ("task1", "task2", "task3"):
        h = task_results[t]["heldout"]
        hrows.append([t, anchor[t][0], anchor[t][1], f(h["final"]["ocf_mean"], 5), f(h["oracle"]["ocf_mean"], 5),
                      h["final"]["n"]])
    mean_final = np.mean([task_results[t]["heldout"]["final"]["ocf_mean"] for t in ("task1", "task2", "task3")])
    grows = []
    for t in ("task1", "task2", "task3"):
        h = task_results[t]["heldout"]
        for g, v in h["final"]["groups"].items():
            grows.append((t, g, v["n"], v["ocf_mean"], h["oracle"]["groups"][g]["ocf_mean"]))
    mouse_rows = []
    for t in ("task1", "task2", "task3"):
        for m in ("mice1", "mice2"):
            sel = [r for r in grows if r[0] == t and r[1].startswith(m)]
            n = sum(r[2] for r in sel)
            fm = sum(r[2] * r[3] for r in sel) / n
            om = sum(r[2] * r[4] for r in sel) / n
            mouse_rows.append([t, m, n, f(fm), f(om)])
    group_rows = [[t, g, n, f(a), f(b)] for t, g, n, a, b in grows]

    fig, ax = plt.subplots(figsize=(11, 3.0))
    labels = [f"{t[-1]}:{g.replace('session', 's')}" for t, g, *_ in grows]
    x = np.arange(len(grows))
    ax.bar(x - 0.2, [r[3] for r in grows], 0.4, color=C1, label="final")
    ax.bar(x + 0.2, [r[4] for r in grows], 0.4, color="#93c5fd", label="oracle@32（诊断用）")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, fontsize=7)
    ax.set_ylabel("本地 OCF")
    ax.set_title("heldout 按 任务:鼠/session 分组（本地 OCF，非 Kaggle）")
    ax.legend()
    group_img = fig_to_img(fig, "分组 heldout")

    return f"""
<section id="sec-c"><h2>C. v5 方案：<code>{esc(config['name'])}</code></h2>
<p>配置来自 <code>src/final_config.json</code>，流程按 <code>src/final.py</code> 的 <code>predict_task</code> 绘制。</p>
{p1}{p2}{p3}
<h3>C.1 Kaggle 成绩</h3>
<div class="kpis"><div class="kpi"><b>{v3:.5f}</b><span>v3 Public</span></div><div class="kpi hl"><b>{v5:.5f}</b><span>v5 Public（+{v5 - v3:.5f}）</span></div>
<div class="kpi"><b>{top:.5f}</b><span>榜首快照</span></div><div class="kpi"><b>{top - v5:.5f}</b><span>v5 距榜首</span></div></div>
<p class="muted">分数为用户回报（verified_online=false）。v5 同时改了 Task2 reranker 与 Task3 soft 融合，+0.71 不能归因到单个任务。榜单 CSV 截于 2026-11-23，早于 v5 提交。</p>
{lb_img}
<h3>C.2 本地固定公开 heldout（v5）</h3>
<div class="warn">本地 OCF 不是 Kaggle 分数。heldout 被历史实验反复使用，不是 pristine 独立确认。oracle@32 = 每行在生产排序前 32 个候选中取 OCF 最高者，使用公开 GT，仅作诊断。</div>
{table(["任务", "协议", "Tune final", "Heldout final", "Heldout oracle@32", "heldout 行数"], hrows)}
<p>三任务等权平均 heldout final = <b>{mean_final:.5f}</b>（仅用于同协议回归）。</p>
<h3>C.3 按鼠拆分（由 results.json 分组按样本数加权）</h3>
<div class="cols"><div>{table(["任务", "鼠", "n", "final", "oracle@32"], mouse_rows)}</div>
<details><summary>按 鼠/session 分组明细</summary>{table(["任务", "组", "n", "final", "oracle@32"], group_rows, "small")}</details></div>
{group_img}
</section>"""


# ---------------------------------------------------------------- D. predictions
def section_predictions(device, task_results):
    from src.evaluation import score_predictions

    parts = []
    checks = []
    dists = {}
    for t, (npz_path, _) in HELDOUT.items():
        if not npz_path.exists():
            return None, f"缺少 {npz_path}"
        with np.load(npz_path, allow_pickle=False) as z:
            pred, tgt, groups, tidx = z["predictions"], z["targets"], z["group_ids"], z["target_indices"]
        s = np.asarray(score_predictions(pred, tgt, device=device, batch_size=64), dtype=np.float64)
        recorded = task_results[t]["heldout"]["final"]["ocf_mean"]
        diff = abs(s.mean() - recorded)
        group_ok = all(abs(s[groups == g].mean() - v["ocf_mean"]) < 1e-6
                       for g, v in task_results[t]["heldout"]["final"]["groups"].items())
        checks.append((t, len(s), s.mean(), recorded, diff, group_ok))
        if diff > 1e-6 or not group_ok:
            return None, f"{t} 重算 OCF {s.mean():.6f} 与记录 {recorded:.6f} 不一致"
        dists[t] = s
        exact = np.array([np.array_equal(p, g) for p, g in zip(pred, tgt)])
        order = np.argsort(s)
        good = order[::-1][:8]
        mid = order[len(order) // 2 - 4: len(order) // 2 + 4]
        bad = order[:8]
        fig, axes = plt.subplots(6, 8, figsize=(9, 7.2))
        for r, (name, sel) in enumerate((("高分", good), ("中位", mid), ("低分", bad))):
            for c, i in enumerate(sel):
                a, b = axes[2 * r, c], axes[2 * r + 1, c]
                a.imshow(tgt[i], cmap="gray_r", vmin=0, vmax=255)
                b.imshow(pred[i], cmap="gray_r", vmin=0, vmax=255)
                a.set_title(f"GT #{tidx[i]}", fontsize=7)
                b.set_title(f"OCF {s[i]:.1f}", fontsize=7, color=C3 if s[i] > 50 else ("#dc2626" if s[i] < 10 else "k"))
                a.set_xticks([]); a.set_yticks([]); b.set_xticks([]); b.set_yticks([])
            axes[2 * r, 0].set_ylabel(f"{name}\nGT", fontsize=8)
            axes[2 * r + 1, 0].set_ylabel("预测", fontsize=8)
        fig.suptitle(f"{t} heldout：GT（上）与 v5 预测（下）", fontsize=10)
        img = fig_to_img(fig, f"{t} 示例", dpi=80)
        parts.append(f"<h3>D.{len(parts) + 1} {t}</h3><p>n={len(s)}，均值 {s.mean():.3f}，中位 {np.median(s):.2f}，"
                     f"P90 {np.percentile(s, 90):.2f}，预测与 GT 完全一致的行 {exact.sum()} / {len(s)} "
                     f"（{exact.mean() * 100:.2f}%）。</p>{img}")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
    for ax, (t, s), c in zip(axes, dists.items(), (C1, C2, C3)):
        ax.hist(s, bins=50, range=(0, 100), color=c)
        ax.axvline(s.mean(), color="k", ls="--", lw=1)
        ax.set_title(f"{t}：均值 {s.mean():.2f}")
        ax.set_xlabel("单样本 OCF")
    axes[0].set_ylabel("行数")
    hist_img = fig_to_img(fig, "OCF 分布")
    crow = [[t, n, f(a, 6), f(b, 6), f"{d:.1e}", "是" if g else "否"] for t, n, a, b, d, g in checks]
    body = f"""
<section id="sec-d"><h2>D. 真实预测示例（本地 heldout）</h2>
<p>数据：<code>test/.cache/evaluate_final_task1_v2/task1/heldout_final.npz</code>、<code>evaluate_final_task23_v1/task2/heldout_final.npz</code>、
<code>evaluate_final_task23_v1/resumed/task3/heldout_final.npz</code>。每个文件自带 <code>predictions / targets / group_ids / target_indices</code>，行内配对，不做任何外部 join。</p>
<div class="card"><h3>对齐校验</h3>
<p>用 <code>test/evaluate.py</code> 同一实现（<code>src.evaluation.score_predictions</code> → 官方 OCF）重算每行 OCF，与 results.json 中记录的 heldout final 均值及每个 鼠/session 分组均值比较；
另外确认 <code>targets</code> 与 <code>pictures.npz[target_indices]</code> 二值化后逐像素相等（构建时已检查）。</p>
{table(["任务", "行数", "重算均值", "记录均值", "|差|", "分组均值全部一致"], crow)}</div>
{hist_img}
{''.join(parts)}
<p class="muted">Task2/3 预测取自库内字图，OCF=100 的行即分类正确；低分行多是笔画结构差异大的字。Task1 GT 不在训练库，预测来自 NotoSansSC 渲染库。</p>
</section>"""
    return body, None


def check_targets(pictures):
    bad = []
    for t, (npz_path, _) in HELDOUT.items():
        if not npz_path.exists():
            continue
        with np.load(npz_path, allow_pickle=False) as z:
            gt = np.where(pictures[z["target_indices"]] >= 128, 255, 0).astype(np.uint8)
            if not np.array_equal(gt, z["targets"]):
                bad.append(t)
    return bad


# ---------------------------------------------------------------- E. roadmap (data-driven)
OUTCOMES = {
    "adopted": ("已进入生产", "#15803d"),
    "positive": ("正增益未采纳", "#86efac"),
    "negative": ("否定", "#dc2626"),
    "diagnostic": ("诊断/上界", "#7c3aed"),
    "infra": ("基础设施", "#94a3b8"),
}
TASKS = ("T1", "T2", "T3")
TASK_NAMES = {"T1": "未见字重建（20 repeat）", "T2": "已见字跨日 session5（4 repeat）", "T3": "已见字单 trial"}


def load_roadmap():
    data = json.loads(ROADMAP.read_text(encoding="utf-8"))
    fam = {f["key"] for f in data["families"]}
    for e in data["experiments"]:
        if e["family"] not in fam or e["outcome"] not in OUTCOMES:
            raise ValueError(f"roadmap.json 实验 {e['id']} 的 family/outcome 非法")
    data["experiments"].sort(key=lambda e: int(e["id"]))
    return data


def task_delta(e, t):
    for r in e.get("results", []):
        if r.get("task") == t and r.get("kind") == "final" and r.get("metric", "ocf") == "ocf":
            d = r.get("delta")
            if d is None and r.get("baseline") is not None and r.get("candidate") is not None:
                d = r["candidate"] - r["baseline"]
            if d is not None:
                return d, r
    return None, None


def result_lines(e, task=None):
    rows = [r for r in e.get("results", []) if task is None or r.get("task") == task]
    return "".join(
        f'<li>{esc(r.get("task", ""))} {esc(r.get("metric", "ocf"))} [{esc(r.get("kind", ""))}] '
        + " → ".join(f(v) for v in (r.get("baseline"), r.get("candidate")) if v is not None)
        + ("" if r.get("delta") is None else
           f' <span class="{"pos" if r["delta"] > 0 else "neg"}">Δ{r["delta"]:+.3f}</span>')
        + f'<br><span class="muted">{esc(r.get("protocol", ""))}</span></li>'
        for r in rows)


def outcome_dots(exps):
    return "".join(f'<span class="chip o-{k}" title="{OUTCOMES[k][0]}">{n}</span>'
                   for k in OUTCOMES if (n := sum(e["outcome"] == k for e in exps)))


def roadmap_tree(data):
    """Collapsible tree: task -> research direction -> experiment -> results/protocol."""
    exps = data["experiments"]
    roots = [(t, f"{t} · {TASK_NAMES[t]}", [e for e in exps if t in e.get("tasks", [])]) for t in TASKS]
    roots.append(("通用", "通用 · 基础设施 / 协议 / 跨任务诊断", [e for e in exps if not e.get("tasks")]))
    out = ['<div class="tree">']
    for t, label, mine in roots:
        adopted = [e for e in mine if e["outcome"] == "adopted"]
        head = (f'<b>{esc(label)}</b> <span class="muted">{len(mine)} 项</span> {outcome_dots(mine)}'
                + (f' <span class="muted">生产：{esc(", ".join(e["id"] for e in adopted))}</span>' if adopted else ""))
        out.append(f'<details class="lv1" open><summary>{head}</summary>')
        for fam in data["families"]:
            sub = [e for e in mine if e["family"] == fam["key"]]
            if not sub:
                continue
            deltas = [d for e in sub if (d := task_delta(e, t)[0] if t in TASKS else None) is not None]
            best = f' <span class="muted">最大 Δ {max(deltas):+.2f}</span>' if deltas else ""
            out.append(f'<details class="lv2"><summary>{esc(fam["label"])} <span class="muted">{len(sub)} 项</span> '
                       f'{outcome_dots(sub)}{best}</summary>')
            for e in sub:
                d = task_delta(e, t)[0] if t in TASKS else None
                dtxt = "" if d is None else f' <span class="{"pos" if d > 0 else "neg"}">{d:+.2f}</span>'
                others = [x for x in e.get("tasks", []) if x != t]
                also = f' <span class="muted">（亦涉及 {"/".join(others)}）</span>' if others else ""
                out.append(
                    f'<details class="lv3"><summary><span class="chip o-{e["outcome"]}">{OUTCOMES[e["outcome"]][0]}</span> '
                    f'<b>{esc(e["id"])}</b> {esc(e["title"])}{dtxt}{also}</summary>'
                    f'<div class="leaf"><div>{esc(e.get("method", ""))}</div>'
                    f'<ul>{result_lines(e, t if t in TASKS else None)}</ul>'
                    + (f'<div>{esc(e["note"])}</div>' if e.get("note") else "")
                    + f'<div class="muted"><code>{esc(e.get("dir", ""))}</code></div></div></details>')
            out.append("</details>")
        out.append("</details>")
    out.append("</div>")
    return "".join(out)


def roadmap_delta_chart(exps):
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.5 + 0.2 * max(
        sum(task_delta(e, t)[0] is not None for e in exps) for t in TASKS)), sharex=False)
    for ax, t in zip(axes, TASKS):
        rows = [(e, task_delta(e, t)[0]) for e in exps if task_delta(e, t)[0] is not None]
        y = np.arange(len(rows))[::-1]
        ax.barh(y, [d for _, d in rows], color=[OUTCOMES[e["outcome"]][1] for e, _ in rows])
        ax.set_yticks(y)
        ax.set_yticklabels([f"{e['id']} {e['title']}" for e, _ in rows], fontsize=7)
        ax.axvline(0, color="k", lw=0.8)
        for yy, (_, d) in zip(y, rows):
            ax.text(d, yy, f" {d:+.2f} ", va="center", ha="left" if d >= 0 else "right", fontsize=6.5)
        ax.set_title(f"{t}：各实验 Δ 本地 OCF（{len(rows)} 项）")
        ax.margins(x=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in OUTCOMES.values()]
    fig.legend(handles, [n for n, _ in OUTCOMES.values()], loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig_to_img(fig, "各任务实验增益")


def roadmap_production_chart(prod):
    rows = [p for p in prod if p.get("kaggle_public") is not None]
    held = [p for p in prod if p.get("heldout")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.0))
    ax = axes[0]
    ax.plot([p["name"] for p in rows], [p["kaggle_public"] for p in rows], "o-", color=C1)
    for p in rows:
        ax.annotate(f"{p['kaggle_public']:.3f}", (p["name"], p["kaggle_public"]), (0, 6),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set_title("Kaggle Public OCF 演进（用户回报）")
    ax = axes[1]
    x = np.arange(len(TASKS))
    width = 0.8 / max(1, len(held))
    for k, p in enumerate(held):
        ax.bar(x + k * width - 0.4 + width / 2, [p["heldout"].get(t) or 0 for t in TASKS], width, label=p["name"])
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS)
    ax.set_title("固定公开 heldout final（本地 OCF）")
    ax.legend(fontsize=8)
    return fig_to_img(fig, "生产版本演进")


def roadmap_ceiling_chart(ceil):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    ax = axes[0]
    items = ceil["T1"]
    ax.bar([i["label"] for i in items], [i["value"] for i in items], color=[C1, "#93c5fd", "#a78bfa", "#dc2626"][:len(items)])
    for k, i in enumerate(items):
        ax.text(k, i["value"] + 1, f"{i['value']:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([i["label"] for i in items], rotation=25, ha="right", fontsize=7)
    ax.set_title("Task1 上界")
    ax.set_ylabel("本地 OCF")
    ax = axes[1]
    m = ceil["T3_accuracy_model"]
    w, p0, target = m["wrong_mean"], m["top1"], m["target"]
    p = np.linspace(0, 0.5, 200)
    ax.plot(p * 100, 100 * p + w * (1 - p), color=C3)
    ax.scatter([p0 * 100], [100 * p0 + w * (1 - p0)], color="k", zorder=3)
    need = (target - w) / (100 - w)
    ax.axhline(target, color="#dc2626", ls="--", lw=1)
    ax.axvline(need * 100, color="#dc2626", ls=":", lw=1)
    ax.text(need * 100 + 1, target + 2, f"{target} 分需 p≈{need * 100:.0f}%", color="#dc2626", fontsize=8)
    ax.annotate(f"当前 top1 {p0 * 100:.2f}%", (p0 * 100, 100 * p0 + w * (1 - p0)), (25, 35), textcoords="offset points", fontsize=8,
                arrowprops={"arrowstyle": "->"})
    ax.set_xlabel("top1 准确率 p (%)")
    ax.set_title(f"Task3：final ≈ 100p + {w}(1−p)")
    ax = axes[2]
    reps, vals = ceil["T1_repeat_scaling"]["repeats"], ceil["T1_repeat_scaling"]["ocf"]
    ax.plot(reps, vals, "o-", color=C1, label="实测")
    a, b = np.polyfit(1 / np.sqrt(reps), vals, 1)
    xs = np.linspace(1, 20, 100)
    ax.plot(xs, a / np.sqrt(xs) + b, ":", color=CG, label="a/√n + b 外推")
    ax.text(20, a / np.sqrt(20) + b - 0.4, f"{a / np.sqrt(20) + b:.2f}", ha="right", fontsize=8)
    ax.set_xlabel("repeat 数")
    ax.set_title("Task1 repeat 缩放")
    ax.legend(fontsize=7)
    return fig_to_img(fig, "上界分析")


def section_roadmap():
    data = load_roadmap()
    exps = data["experiments"]
    counts = {k: sum(e["outcome"] == k for e in exps) for k in OUTCOMES}
    legend = "".join(f'<span class="chip o-{k}">{n}：{counts[k]}</span>' for k, (n, _) in OUTCOMES.items())
    summary = []
    for t in TASKS:
        mine = [e for e in exps if t in e.get("tasks", [])]
        best = max(((task_delta(e, t)[0], e) for e in mine if task_delta(e, t)[0] is not None),
                   key=lambda x: x[0], default=(None, None))
        adopted = [e["id"] for e in mine if e["outcome"] == "adopted"]
        summary.append([t, len(mine), sum(e["outcome"] == "negative" for e in mine), esc(", ".join(adopted) or "—"),
                        "—" if best[1] is None else f"{best[1]['id']} {esc(best[1]['title'])}（{best[0]:+.3f}）"])
    return f"""
<section id="sec-e"><h2>E. 实验探索路线图（00–{exps[-1]['id']}）</h2>
<p class="muted">数据唯一来源：<code>test/91_visual_report/roadmap.json</code>（更新于 {esc(data.get('updated', ''))}）。
新实验完成后在其中追加一条记录，再运行 <code>build_report.py --skip-predictions</code>（或完整运行）即可刷新本节全部图表。</p>
<div class="warn">全部为本地公开数据 OCF，不是 Kaggle 分数；不同实验基线与协议不同，Δ 只在同一实验内有意义，不可横向相加。</div>
<div>{legend}</div>
<h3>E.1 各任务概览</h3>
{table(["任务", "涉及实验", "否定", "进入生产的实验", "单项最大 Δ"], summary)}
<h3>E.2 探索树：任务 → 方向 → 实验</h3>
<p class="muted">点击节点折叠/展开；节点后的色块为各结论的实验数，数字为该任务下的 Δ 本地 OCF。跨任务实验在每个相关任务下各出现一次。
<button onclick="document.querySelectorAll('.tree details').forEach(d=>d.open=true)">全部展开</button>
<button onclick="document.querySelectorAll('.tree details.lv2,.tree details.lv3').forEach(d=>d.open=false)">折叠到任务</button></p>
{roadmap_tree(data)}
<h3>E.3 各任务实验增益</h3>{roadmap_delta_chart(exps)}
<h3>E.4 生产版本演进</h3>{roadmap_production_chart(data["production"])}
<h3>E.5 上界分析</h3>{roadmap_ceiling_chart(data["ceilings"])}
</section>"""


CSS = """
:root{--bg:#f8fafc;--fg:#0f172a;--muted:#64748b;--card:#fff;--line:#e2e8f0;--acc:#2563eb;--warn:#fef3c7;--warnb:#f59e0b}
*{box-sizing:border-box}body{margin:0;font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif;background:var(--bg);color:var(--fg);line-height:1.6}
.wrap{display:grid;grid-template-columns:220px 1fr;gap:24px;max-width:1320px;margin:0 auto;padding:16px}
nav{position:sticky;top:12px;align-self:start;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
nav a{display:block;color:var(--fg);text-decoration:none;padding:4px 0;font-size:14px}nav a:hover{color:var(--acc)}
nav .sub{padding-left:12px;font-size:12px;color:var(--muted)}
main{min-width:0}header{padding:8px 0 4px}h1{margin:0;font-size:26px}h2{border-bottom:2px solid var(--acc);padding-bottom:4px;margin-top:36px}
h3{margin-top:22px;font-size:17px}section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:6px 22px 18px;margin-bottom:20px}
img{max-width:100%;height:auto;display:block;margin:8px 0}code{background:#f1f5f9;padding:1px 4px;border-radius:4px;font-size:.9em}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
th{background:#f1f5f9;position:sticky;top:0}table.small{font-size:12px}
.muted{color:var(--muted);font-size:13px}.warn{background:var(--warn);border-left:4px solid var(--warnb);padding:8px 12px;border-radius:6px;font-size:14px;margin:10px 0}
.card{border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin:10px 0;background:#fcfdff}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start}.cols3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.hier,.pf{display:flex;flex-wrap:wrap;align-items:stretch;gap:6px}.hb,.pb{border:2px solid var(--acc);border-radius:8px;padding:8px 10px;background:#fff;min-width:120px;flex:1;font-weight:600;font-size:14px}
.hb small,.pb small{display:block;font-weight:400;color:var(--muted);font-size:12px}.arr{align-self:center;color:var(--muted);font-size:20px}
.pipe{margin:12px 0;border:1px solid var(--line);border-radius:10px;overflow:hidden}.pt{color:#fff;font-weight:700;padding:4px 12px}.pf{padding:10px}
.rep{margin:12px 0}.rrow{display:flex;gap:6px;margin:6px 0}.r{width:48px;text-align:center;padding:6px 0;border-radius:6px;color:#fff;font-weight:700}
.r.tr{background:var(--acc)}.r.t3{background:#16a34a}
.kpis{display:flex;flex-wrap:wrap;gap:12px}.kpi{flex:1;min-width:150px;border:1px solid var(--line);border-radius:10px;padding:10px;text-align:center}
.kpi b{display:block;font-size:22px}.kpi span{color:var(--muted);font-size:13px}.kpi.hl{border-color:var(--acc);background:#eff6ff}
.formula{font-size:17px;text-align:center;font-weight:700;margin:10px 0}
.pass{color:#16a34a;font-weight:700}.fail{color:#dc2626;font-weight:700}.diag{color:#7c3aed;font-weight:700}.pos{color:#16a34a}.neg{color:#dc2626}
details summary{cursor:pointer;color:var(--acc);margin:8px 0}
.chip{display:inline-block;border-radius:6px;padding:1px 6px;margin:2px;font-size:12px;border:1px solid transparent;cursor:default}
.o-adopted{background:#15803d;color:#fff;box-shadow:0 0 0 2px #0f172a inset}.o-positive{background:#dcfce7;border-color:#86efac}
.o-negative{background:#fee2e2;border-color:#fca5a5}.o-diagnostic{background:#ede9fe;border-color:#c4b5fd}.o-infra{background:#f1f5f9;border-color:#cbd5e1}
.tree details{margin:2px 0}.tree summary{color:var(--fg,#0f172a);padding:3px 4px;border-radius:6px}.tree summary:hover{background:#f1f5f9}
.tree .lv1{border:1px solid var(--line);border-radius:10px;padding:4px 8px;margin:8px 0}.tree .lv1>summary{font-size:15px}
.tree .lv2,.tree .lv3{margin-left:18px;border-left:2px solid var(--line);padding-left:10px}
.tree .leaf{margin:4px 0 8px 18px;font-size:13px}.tree .leaf ul{margin:4px 0;padding-left:18px}
@media (max-width:900px){.wrap{grid-template-columns:1fr}nav{position:static}.cols,.cols3{grid-template-columns:1fr}}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-predictions", action="store_true", help="omit section D (no GPU re-scoring)")
    args = parser.parse_args()
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(DATASET / "train" / "pictures.npz", allow_pickle=False) as z:
        pictures = z["images"]
    config = json.loads((ROOT / "src/final_config.json").read_text(encoding="utf-8"))
    kaggle = json.loads((ROOT / "test/90_submission/kaggle_scores.json").read_text(encoding="utf-8"))
    task_results = {}
    for t, (_, res) in HELDOUT.items():
        task_results[t] = json.loads(res.read_text(encoding="utf-8"))["tasks"][t]

    sec_a = section_dataset(pictures)
    sec_b = section_tasks()
    sec_c = section_v5(config, kaggle, task_results)
    bad = [] if args.skip_predictions else check_targets(pictures)
    if args.skip_predictions:
        sec_d, reason = None, "--skip-predictions"
    elif bad:
        sec_d, reason = None, f"targets 与 pictures[target_indices] 不一致：{bad}"
    else:
        sec_d, reason = section_predictions(device, task_results)
    if sec_d is None:
        sec_d = f'<section id="sec-d"><h2>D. 真实预测示例</h2><div class="warn">已省略（{esc(reason)}）。</div></section>'
        print("section D omitted:", reason)
    sec_e = section_roadmap()

    nav = """<nav><b>目录</b>
<a href="#sec-a">A. 数据集</a><a class="sub" href="#sec-a">层级 / 分块 / raster / PSTH</a>
<a href="#sec-b">B. 三个任务与 OCF</a>
<a href="#sec-c">C. v5 方案与成绩</a><a class="sub" href="#sec-c">pipeline / Kaggle / heldout / 按鼠</a>
<a href="#sec-d">D. 真实预测示例</a>
<a href="#sec-e">E. 实验探索路线图</a><a class="sub" href="#sec-e">探索树 / 增益 / 版本 / 上界</a></nav>"""
    doc = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SiClink 小鼠视觉重建 BCI：技术复盘可视化报告</title>
<style>{CSS}</style></head><body><div class="wrap">{nav}<main>
<header><h1>SiClink 小鼠视觉重建 BCI：技术复盘可视化报告</h1>
<p class="muted">生成脚本 <code>test/91_visual_report/build_report.py</code>。仅使用公开 train 数据与本地评估输出，不读取任何隐藏测试 GT。
本地 OCF ≠ Kaggle 分数；固定公开 heldout 已被历史实验复用。</p></header>
{sec_a}{sec_b}{sec_c}{sec_d}{sec_e}
</main></div></body></html>"""
    OUTPUT.write_text(doc, encoding="utf-8")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
