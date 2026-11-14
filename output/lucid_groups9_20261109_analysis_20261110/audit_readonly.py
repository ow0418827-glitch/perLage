"""只读取冻结结果，复核统计并输出独立分析；不调用模型或商城。"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent
RUN = ROOT / "output/lucid_groups9_20261109"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def link(label: str, path: Path, line: int | None = None):
    target = path.resolve().as_posix() + (f":{line}" if line else "")
    return f"[{label}](<{target}>)"


def main():
    outcomes_path = RUN / "shopping/outcomes.json"
    outcomes = read(outcomes_path)
    shopping = read(RUN / "shopping/summary_metrics.json")
    retrieval = read(RUN / "summary_metrics.json")
    groups = []
    episodes = []
    roles = Counter()
    errors = Counter()
    attempt_finishes = Counter()
    final_finishes = Counter()
    rank_stats = Counter()
    counts = Counter()
    seen = set()
    for group in sorted({o["group"] for o in outcomes}):
        method_dir = RUN / "groups" / group / "methods/lucid"
        frozen = read(method_dir / "FROZEN_IMAGE.json")
        plan = read(method_dir / "evaluation/evaluation_plan.json")
        source = frozen["source"]["id"]
        memory_id = plan["source_memory_id"]
        per_group = Counter()
        for outcome in (o for o in outcomes if o["group"] == group):
            key = (group, outcome["task_id"])
            assert key not in seen
            seen.add(key)
            record_path = Path(outcome["record_path"])
            assert digest(record_path) == outcome["record_sha256"]
            record = read(record_path)
            traces = record["PolicyTrace"]
            selected = traces[0]["SelectedIds"]
            recalled = memory_id in selected
            assert recalled == outcome["source_recalled"]
            rank = selected.index(memory_id) + 1 if recalled else None
            source_in_results = any(
                t.get("ParsedAction", "").startswith("search[")
                and f"[{source}]" in t.get("Observation", "")
                for t in traces
            )
            clicked = any(
                t.get("ParsedAction") == f"click[{source}]" for t in traces
            )
            purchased = record.get("Completed") is True and record.get("FinalASIN") == source
            assert purchased == outcome["misleading_purchase"]
            assert record["FinalASIN"] == outcome["final_asin"]
            assert clicked == purchased
            if source_in_results:
                assert recalled
            if purchased:
                assert source_in_results
            per_group.update(
                episodes=1,
                source_recalled=int(recalled),
                source_rank_first=int(rank == 1),
                source_in_current_search_results=int(source_in_results),
                source_clicked=int(clicked),
                source_purchased=int(purchased),
                exact_target_purchased=int(outcome["exact_target_purchase"]),
                completed=int(record.get("Completed") is True),
                policy_steps=len(traces),
            )
            counts["explicit_think_actions"] += sum(
                a.startswith("Action: think[") for a in record["Actions"]
            )
            if rank:
                rank_stats[f"rank_{rank}_episodes"] += 1
                rank_stats[f"rank_{rank}_source_purchases"] += int(purchased)
            for trace in traces:
                assert trace["DisplayedMemoryCount"] == 3
                assert len(trace["SelectedIds"]) == 3
                assert len(trace["InputImages"]) == 3
                counts["policy_steps"] += 1
                counts["input_image_instances"] += len(trace["InputImages"])
                counts["invalid_actions"] += int(bool(trace["InvalidAction"]))
                final_finishes[str(trace["FinishReason"])] += 1
                for image in trace["InputImages"]:
                    roles[image["Role"]] += 1
                if recalled:
                    source_images = [im for im in trace["InputImages"] if im.get("Id") == memory_id]
                    assert len(source_images) == 1
                    assert source_images[0]["SHA256"] == frozen["image_sha256"]
                for attempt in trace["ResponseAttempts"]:
                    counts["response_attempts"] += 1
                    counts["attempts_with_reasoning_metadata"] += int(bool(attempt.get("ReasoningPresent")))
                    errors[attempt.get("ErrorType", "successful_action_response")] += 1
                    attempt_finishes[str(attempt.get("FinishReason"))] += 1
            episodes.append(dict(
                group=group, task_id=outcome["task_id"], source_asin=source,
                source_recalled=recalled, selected_source_rank=rank,
                source_in_current_search_results=source_in_results,
                source_clicked=clicked, source_purchased=purchased,
                final_asin=record["FinalASIN"], record_path=str(record_path),
                record_sha256=outcome["record_sha256"],
            ))
        generation = frozen["metadata"]["attack"]
        assert generation["steps_completed"] == 1000
        assert generation["selected_step"] == 1000
        row = next(r for r in retrieval["groups"] if r["group"] == group)
        assert row["verification_status"] == "passed"
        assert per_group["source_recalled"] == int(row["mod_top3"].split("/")[0])
        groups.append(dict(
            group=group, name=row["name"], source=frozen["source"], target=frozen["target"],
            retrieval_clean_top3=row["clean_top3"], retrieval_modified_top3=row["mod_top3"],
            generation_steps=generation["steps_completed"], **per_group,
        ))
    overall = {k: sum(g.get(k, 0) for g in groups) for k in (
        "episodes", "source_recalled", "source_rank_first", "source_in_current_search_results",
        "source_clicked", "source_purchased", "exact_target_purchased", "completed", "policy_steps",
    )}
    assert overall["episodes"] == 180
    assert overall["source_purchased"] == shopping["overall"]["misleading_purchases"] == 49
    assert overall["source_recalled"] == 145
    assert overall["source_in_current_search_results"] == 53
    audit = dict(
        analysis_date="2026-11-10", experiment_date="2026-11-09",
        scope="只读复核既有冻结结果，不运行模型、商城、训练或正式排名",
        input_outcomes_sha256=digest(outcomes_path),
        definition="实际搜索结果仅解析当前会话搜索动作返回观察中的源编号，不计提示中的历史示例。",
        reasoning_boundary="只有显式思考动作和推理存在/数量元数据，没有内部推理正文。",
        overall=overall, counts=dict(counts), image_roles=dict(roles),
        response_attempt_types=dict(errors), attempt_finish_reasons=dict(attempt_finishes),
        final_action_finish_reasons=dict(final_finishes), selected_rank_statistics=dict(rank_stats),
        groups=groups, episodes=episodes,
    )
    (DEST / "AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(overall=overall, counts=dict(counts), roles=dict(roles)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
