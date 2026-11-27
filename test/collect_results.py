"""Collect local OCF results without mixing evaluation protocols or score sources."""

import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "test"
TASKS = ("task1", "task2", "task3")
PROTOCOL_KEYS = (
    "task", "bin_ms", "split_seed", "picture_seed", "tune_seed", "subset_seed",
    "n_tune", "n_heldout", "tune", "heldout", "held_out", "evaluation_groups",
    "validation_session", "outer_folds", "target_day_excluded_from_base_training",
    "target_day_excluded_from_reranker_training",
)
EXPERIMENT_TASKS = {
    "24_forward_cnn": "task2",
    "25_rotating_sessions": "task2",
    "40_forward_cnn_day_readout": "task2",
}


def stable_hash(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def infer_task(data, path):
    for key in ("task", "config", "protocol", "split"):
        value = data.get(key)
        if isinstance(value, dict) and value.get("task") in TASKS:
            return value["task"]
        if key == "task" and value in TASKS:
            return value
    experiment = data.get("experiment", path.parent.name)
    if experiment in EXPERIMENT_TASKS:
        return EXPERIMENT_TASKS[experiment]
    for value in (experiment, path.name, data.get("scope"), data.get("method"), data.get("protocol")):
        if isinstance(value, str):
            match = re.search(r"task\s*[_-]?\s*([123])(?:\b|_)", value, re.IGNORECASE)
            if match:
                return f"task{match.group(1)}"
    return None


def method_prefix(data, path):
    experiment = data.get("experiment", path.parent.name)
    return experiment if path.name == "results.json" else f"{experiment}:{path.stem}"


def make_metric(value):
    if not isinstance(value, dict):
        return None
    ocf = value.get("ocf", value.get("ocf_mean"))
    if not isinstance(ocf, (int, float)) or isinstance(ocf, bool) or not math.isfinite(ocf):
        return None
    n = value.get("n")
    if n is None and isinstance(value.get("pictures"), list):
        n = len(value["pictures"])
    top1 = value.get("overall_top1", value.get("top1", value.get("top1_accuracy", value.get("actual_accuracy"))))
    return {"n": n, "local_ocf": float(ocf), "local_top1": top1}


def make_protocol(data, task):
    protocol = {"task": task}
    raw = data.get("protocol")
    if isinstance(raw, dict):
        for key in PROTOCOL_KEYS:
            if key in raw:
                protocol[key] = raw[key]
        for source, target in (("tune_picture_count", "n_tune"), ("heldout_picture_count", "n_heldout")):
            if source in raw and target not in protocol:
                protocol[target] = raw[source]
        if not any(key in raw for key in ("tune", "heldout", "held_out", "n_tune", "tune_picture_count")):
            protocol["protocol_sha256"] = stable_hash(raw)
    elif isinstance(raw, str):
        protocol["description"] = raw
    split = data.get("split")
    if isinstance(split, dict):
        protocol["split_sha256"] = stable_hash(split)
        for key in ("seed", "validation_session"):
            if key in split:
                protocol[key] = split[key]
    config = data.get("config", {})
    if isinstance(config, dict):
        for key in ("bin_ms", "grid"):
            if key in config and key not in protocol:
                protocol[key] = config[key]
        if not raw and not split:
            for key in ("group", "seed"):
                if key in config:
                    protocol[key] = config[key]
    for key in ("split_seed", "picture_seed", "tune_seed", "subset_seed"):
        if key in data and key not in protocol:
            protocol[key] = data[key]
    return protocol


def add_record(records, warnings, *, task, protocol, group, role, method, source, tune, heldout):
    if task not in TASKS or role not in ("final", "oracle") or (tune is None and heldout is None):
        return
    protocol = dict(protocol)
    protocol["task"] = task
    records.append({
        "task": task, "protocol": protocol, "group": str(group), "role": role,
        "method": method, "source": source, "tune": tune, "heldout": heldout,
    })
    if heldout is None:
        warnings.append(f"{source} :: {task}/{group}/{role}/{method}: heldout not evaluated; null.")


def legacy_protocol(data, task, metrics, split_name):
    config = data.get("config", {})
    return {
        "schema": "legacy_evaluation",
        "task": task,
        "split": split_name,
        "bin_ms": config.get("bin_ms"),
        "grid": config.get("grid"),
        "n": metrics.get("n"),
        "evaluation_groups": sorted(metrics.get("groups", {})),
    }


def collect_legacy(data, path, task, records, warnings):
    source = path.relative_to(ROOT).as_posix()
    prefix = method_prefix(data, path)
    evaluation = data.get("evaluation", {})
    variants = []
    if "ocf_mean" in evaluation:
        variants.append(("independent", evaluation))
    for variant in ("direct", "retrieval"):
        if variant in evaluation:
            variants.append((variant, evaluation[variant]))
    assignment = data.get("evaluation_assignment")
    if isinstance(assignment, dict) and "ocf_mean" in assignment:
        variants.append(("assignment", assignment))
    for variant, metrics in variants:
        add_record(
            records, warnings, task=task, protocol=legacy_protocol(data, task, metrics, "evaluation"),
            group=variant, role="final", method=prefix, source=source,
            tune=make_metric(metrics), heldout=None,
        )
    heldout = data.get("held_out", {})
    if not variants and isinstance(heldout, dict):
        for arm in ("baseline", "candidate"):
            metrics = heldout.get(arm)
            if isinstance(metrics, dict):
                add_record(
                    records, warnings, task=task, protocol=legacy_protocol(data, task, metrics, "heldout"),
                    group="overall", role="final", method=f"{prefix}:{arm}", source=source,
                    tune=None, heldout=make_metric(metrics),
                )


def metric_values(value):
    if not isinstance(value, dict):
        return {}
    for key in ("overall", "pooled"):
        if key in value:
            return value[key]
    return value


def role_metrics(value, default_role="final"):
    raw = metric_values(value)
    if not raw:
        return None, None
    if default_role == "oracle":
        return None, make_metric(raw)
    if "final_ocf" in raw or "oracle32_ocf" in raw:
        final = make_metric({"n": raw.get("n"), "ocf": raw.get("final_ocf"), "overall_top1": raw.get("final_accuracy")})
        oracle = make_metric({"n": raw.get("n"), "ocf": raw.get("oracle32_ocf")})
        return final, oracle
    if "top1_ocf" in raw or "oracle_top32_ocf" in raw:
        final = make_metric({"n": raw.get("n"), "ocf": raw.get("top1_ocf")})
        oracle = make_metric({"n": raw.get("n"), "ocf": raw.get("oracle_top32_ocf")})
        return final, oracle
    return make_metric(raw), None


def collect_arm_splits(container, task, role, prefix, source, protocol, records, warnings):
    tune = container.get("tune", {})
    heldout = container.get("heldout", container.get("held_out", {}))
    if not isinstance(tune, dict):
        return
    if not isinstance(heldout, dict):
        heldout = {}
    for arm in sorted(set(tune) | set(heldout)):
        if arm == "selected" and arm not in tune:
            continue
        tune_value = tune.get(arm)
        held_value = heldout.get(arm)
        tune_final, tune_oracle = role_metrics(tune_value, role)
        held_final, held_oracle = role_metrics(held_value, role)
        method = f"{prefix}:{arm}"
        if any(value is not None for value in (tune_final, tune_oracle, held_final, held_oracle)):
            add_record(records, warnings, task=task, protocol=protocol, group="overall", role="final",
                       method=method, source=source, tune=tune_final, heldout=held_final)
            add_record(records, warnings, task=task, protocol=protocol, group="overall", role="oracle",
                       method=method, source=source, tune=tune_oracle, heldout=held_oracle)
            continue
        if not isinstance(tune_value, dict) and not isinstance(held_value, dict):
            continue
        tune_blocks = tune_value if isinstance(tune_value, dict) else {}
        held_blocks = held_value if isinstance(held_value, dict) else {}
        for group in sorted(set(tune_blocks) | set(held_blocks)):
            tune_final, tune_oracle = role_metrics(tune_blocks.get(group), role)
            held_final, held_oracle = role_metrics(held_blocks.get(group), role)
            add_record(records, warnings, task=task, protocol=protocol, group=group, role="final",
                       method=method, source=source, tune=tune_final, heldout=held_final)
            add_record(records, warnings, task=task, protocol=protocol, group=group, role="oracle",
                       method=method, source=source, tune=tune_oracle, heldout=held_oracle)


def collect_standard(data, path, task, records, warnings):
    source = path.relative_to(ROOT).as_posix()
    prefix = method_prefix(data, path)
    protocol = make_protocol(data, task)
    if isinstance(data.get("tune"), dict):
        collect_arm_splits(data, task, "final", prefix, source, protocol, records, warnings)
        return
    locked = data.get("locked_selection", {})
    if isinstance(locked, dict) and isinstance(locked.get("tune"), dict):
        container = {"tune": locked["tune"], "heldout": data.get("heldout", {})}
        collect_arm_splits(container, task, "final", prefix, source, protocol, records, warnings)
        return
    if all(isinstance(data.get(role), dict) and isinstance(data[role].get("tune"), dict) for role in ("final", "oracle")):
        for role in ("final", "oracle"):
            collect_arm_splits(data[role], task, role, prefix, source, protocol, records, warnings)
        return
    for child_task in TASKS:
        child = data.get(child_task)
        if isinstance(child, dict) and isinstance(child.get("tune"), dict):
            collect_arm_splits(child, child_task, "final", prefix, source, make_protocol(data, child_task), records, warnings)
    if isinstance(data.get("heldout"), dict):
        collect_arm_splits({"tune": {}, "heldout": data["heldout"]}, task, "final", prefix, source, protocol, records, warnings)


def collect_candidate_tasks(data, path, records, warnings):
    source = path.relative_to(ROOT).as_posix()
    prefix = method_prefix(data, path)
    for task, values in data.get("tasks", {}).items():
        if task not in TASKS or not isinstance(values, dict):
            continue
        protocol = make_protocol(data, task)
        overall = values.get("overall")
        if isinstance(overall, dict) and "actual_top1_ocf" in overall:
            final = make_metric({"n": overall.get("n"), "ocf": overall["actual_top1_ocf"], "overall_top1": overall.get("actual_accuracy")})
            add_record(records, warnings, task=task, protocol=protocol, group="overall", role="final",
                       method=f"{prefix}:actual_top1", source=source, tune=final, heldout=None)
            for k, ocf in overall.get("oracle_ocf", {}).items():
                oracle = make_metric({"n": overall.get("n"), "ocf": ocf})
                add_record(records, warnings, task=task, protocol=protocol, group="overall", role="oracle",
                           method=f"{prefix}:oracle@{k}", source=source, tune=oracle, heldout=None)
        elif "final_top1_ocf" in values:
            if "sampling" in data:
                protocol["sampling"] = data["sampling"]
            final = make_metric({"n": values.get("n"), "ocf": values["final_top1_ocf"]})
            oracle = make_metric({"n": values.get("n"), "ocf": values.get("oracle32_ocf")})
            add_record(records, warnings, task=task, protocol=protocol, group="overall", role="final",
                       method=f"{prefix}:final_top1", source=source, tune=final, heldout=None)
            add_record(records, warnings, task=task, protocol=protocol, group="overall", role="oracle",
                       method=f"{prefix}:oracle32", source=source, tune=oracle, heldout=None)


def collect_rotating(data, path, task, records, warnings):
    source = path.relative_to(ROOT).as_posix()
    prefix = method_prefix(data, path)
    protocol = make_protocol(data, task)
    protocol["picture_ids_sha256"] = stable_hash(data.get("picture_ids"))
    protocol["evaluation"] = "macro mean over new session1/session2/session3 folds"
    for arm, values in data.get("new_macro_mean", {}).items():
        add_record(records, warnings, task=task, protocol=protocol, group="new_macro_mean", role="final",
                   method=f"{prefix}:{arm}", source=source, tune=make_metric(values.get("tune")),
                   heldout=make_metric(values.get("held_out")))


def collect_layout_teacher(data, path, task, records, warnings):
    source = path.relative_to(ROOT).as_posix()
    prefix = method_prefix(data, path)
    protocol = {
        "task": task, "schema": "layout_teacher_local_reused_split",
        "train_ids_sha256": stable_hash(data.get("train_ids")),
        "tune_ids_sha256": stable_hash(data.get("tune_ids")),
        "heldout_ids_sha256": stable_hash(data.get("held_out_ids")),
    }
    selected = data.get("selected")
    for arm in ("baseline", "student"):
        heldout = data.get("selected_held_out") if selected == arm and data.get("student_held_out_evaluated") else None
        add_record(records, warnings, task=task, protocol=protocol, group="overall", role="final",
                   method=f"{prefix}:{arm}", source=source, tune=make_metric(data.get(f"{arm}_tune")),
                   heldout=make_metric(heldout))
    for name, key in (("image_teacher_top1", "image_teacher_top1_ocf"), ("image_teacher_oracle@8", "image_teacher_oracle_at_8")):
        oracle = make_metric({"n": data.get("baseline_tune", {}).get("n"), "ocf": data.get(key)})
        add_record(records, warnings, task=task, protocol=protocol, group="overall", role="oracle",
                   method=f"{prefix}:{name}", source=source, tune=oracle, heldout=None)


def collect_nested_rerank(data, path, task, records, warnings):
    source = path.relative_to(ROOT).as_posix()
    prefix = method_prefix(data, path)
    protocol = make_protocol(data, task)
    macro = data.get("macro_mean_of_three_outer_days", {})
    for arm in sorted(set(macro.get("tune", {})) | set(macro.get("heldout", {}))):
        if not arm.startswith("delta_"):
            add_record(records, warnings, task=task, protocol=protocol, group="macro_mean_of_three_outer_days",
                       role="final", method=f"{prefix}:{arm}", source=source,
                       tune=make_metric(macro.get("tune", {}).get(arm)),
                       heldout=make_metric(macro.get("heldout", {}).get(arm)))


def fold_metric(value, ocf_key):
    if not isinstance(value, dict):
        return None
    metric = make_metric({
        "n": value.get("n"),
        "ocf": value.get(ocf_key),
        "overall_top1": value.get("final_top1") if ocf_key == "final_ocf" else None,
    })
    if metric is not None and ocf_key == "final_ocf":
        for key in ("recall_at_1", "recall_at_32"):
            if key in value:
                metric[key] = value[key]
    return metric


def collect_fold_metrics(values, group, prefix, source, protocol, records, warnings):
    for arm in ("existing_cnn", "day_readout", "constant_rate"):
        metrics = values.get(arm)
        if not isinstance(metrics, dict):
            continue
        method = f"{prefix}:{arm}"
        for role, ocf_key in (("final", "final_ocf"), ("oracle", "oracle32_ocf")):
            add_record(
                records, warnings, task="task2", protocol=protocol, group=group, role=role,
                method=method, source=source,
                tune=fold_metric(metrics.get("tune"), ocf_key),
                heldout=fold_metric(metrics.get("heldout"), ocf_key),
            )


def collect_day_readout(data, path, task, records, warnings):
    source = path.relative_to(ROOT).as_posix()
    prefix = method_prefix(data, path)
    protocol = make_protocol(data, task)
    protocol["schema"] = "leave_one_day_out_records_and_fold_pooled"
    protocol["picture_ids_sha256"] = stable_hash(data.get("picture_ids"))
    for fold in data.get("records", []):
        group = f"{fold.get('mouse')}/{fold.get('held_session')}"
        collect_fold_metrics(fold.get("metrics", {}), group, prefix, source, protocol, records, warnings)
    for fold, values in data.get("fold_pooled", {}).items():
        collect_fold_metrics(values, f"fold_pooled/{fold}", prefix, source, protocol, records, warnings)


def best_method(methods):
    split = "tune" if any(value["tune"] is not None for value in methods.values()) else "heldout"
    candidates = [(name, value) for name, value in methods.items() if value[split] is not None]
    if not candidates:
        return None
    name, value = min(candidates, key=lambda item: (-item[1][split]["local_ocf"], item[0]))
    return {
        "method": name, "selection_split": split, "selection_local_ocf": value[split]["local_ocf"],
        "tune": value["tune"], "heldout": value["heldout"], "source": value["source"],
    }


def build_summary(records, warnings):
    summary = {}
    for record in records:
        task = record["task"]
        protocol_id = stable_hash({"task": task, "protocol": record["protocol"]})[:16]
        protocols = summary.setdefault(task, {})
        entry = protocols.setdefault(protocol_id, {"protocol": record["protocol"], "groups": {}})
        if entry["protocol"] != record["protocol"]:
            warnings.append(f"Protocol id collision for {task}/{protocol_id}.")
        roles = entry["groups"].setdefault(record["group"], {
            "final": {"methods": {}, "best": None}, "oracle": {"methods": {}, "best": None},
        })
        methods = roles[record["role"]]["methods"]
        if record["method"] in methods:
            warnings.append(f"Duplicate method ignored: {task}/{protocol_id}/{record['group']}/{record['role']}/{record['method']}.")
            continue
        methods[record["method"]] = {"source": record["source"], "tune": record["tune"], "heldout": record["heldout"]}
    for protocols in summary.values():
        for entry in protocols.values():
            for roles in entry["groups"].values():
                for role in ("final", "oracle"):
                    roles[role]["methods"] = dict(sorted(roles[role]["methods"].items()))
                    roles[role]["best"] = best_method(roles[role]["methods"])
            entry["groups"] = dict(sorted(entry["groups"].items()))
    return {task: dict(sorted(summary.get(task, {}).items())) for task in TASKS}


def main():
    records = []
    warnings = []
    paths = (set(TEST.glob("*/results_*.json")) | set(TEST.glob("*/results.json"))
             | set(TEST.glob("*/*/results.json")) | set(TEST.glob("*/*/results_*.json")))
    for path in sorted(paths, key=lambda item: item.as_posix()):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("experiment") == "00_sanity":
            continue
        task = infer_task(data, path)
        before = len(records)
        if "evaluation" in data or ("held_out" in data and "config" in data and data.get("tune") is None):
            collect_legacy(data, path, task, records, warnings)
        elif "tasks" in data:
            collect_candidate_tasks(data, path, records, warnings)
        elif "new_macro_mean" in data:
            collect_rotating(data, path, task, records, warnings)
        elif "baseline_tune" in data and "student_tune" in data:
            collect_layout_teacher(data, path, task, records, warnings)
        elif "macro_mean_of_three_outer_days" in data:
            collect_nested_rerank(data, path, task, records, warnings)
        elif isinstance(data.get("records"), list) and isinstance(data.get("fold_pooled"), dict):
            collect_day_readout(data, path, task, records, warnings)
        else:
            collect_standard(data, path, task, records, warnings)
        if len(records) == before:
            warnings.append(f"{path.relative_to(ROOT).as_posix()}: no supported task-level local OCF metrics; excluded from rankings.")

    summary = build_summary(records, warnings)
    kaggle_path = TEST / "90_deliverables/90_submission/kaggle_scores.json"
    kaggle_scores = json.loads(kaggle_path.read_text(encoding="utf-8")) if kaggle_path.exists() else []
    best_kaggle = max((row["score"] for row in kaggle_scores), default=None)
    output = {
        "summary": summary,
        "kaggle_scores": kaggle_scores,
        "leaderboard_rank_1_snapshot": 54.01486,
        "best_kaggle_score": best_kaggle,
        "gap_to_rank_1_actual": None if best_kaggle is None else 54.01486 - best_kaggle,
        "score_sources": {
            "local": "summary contains only local OCF/top1 on public data; local OCF is not a Kaggle score.",
            "kaggle": "kaggle_scores is copied verbatim from test/90_deliverables/90_submission/kaggle_scores.json; no local value is substituted.",
            "ranking": "Best methods are ranked only within task + protocol + group + role. Tune is used when available; heldout-only groups are explicitly marked selection_split=heldout. Local scores are never mixed with Kaggle scores.",
        },
        "warnings": sorted(set(warnings)),
    }
    path = TEST / "results_summary.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {path}: {len(records)} records, {len(output['warnings'])} warnings")
    for task, protocols in summary.items():
        finals = sum(1 for entry in protocols.values() for groups in entry["groups"].values() if groups["final"]["best"])
        print(f"{task}: {len(protocols)} protocols, {finals} final/group rankings")


if __name__ == "__main__":
    main()
