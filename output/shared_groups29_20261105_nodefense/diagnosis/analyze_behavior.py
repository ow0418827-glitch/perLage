"""离线复核本轮购物行为；不导入模型、不请求商城、不改冻结证据。"""
from __future__ import annotations

import collections
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


RUN = Path(__file__).resolve().parent.parent
PROJECT = RUN.parents[2]
DEST = Path(__file__).resolve().parent
IDS = re.compile(r"\[([A-Z0-9]{10})\]")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(items):
    return dict(collections.Counter(items))


def cards(observation):
    lines = observation.splitlines()
    result = []
    for i, line in enumerate(lines):
        match = re.fullmatch(r"\s*\[([A-Z0-9]{10})\]\s*", line)
        if match:
            result.append({"position": len(result) + 1, "asin": match[1],
                           "title": lines[i + 1].strip() if i + 1 < len(lines) else ""})
    return result


plan = read(RUN / "BATCH_PLAN.json")
groups = {g["key"]: g for g in plan["groups"]}
outcomes = read(RUN / "shopping/outcomes.json")
outcome_map = {(o["group"], o["bank"], o["task_id"]): o for o in outcomes}
episode_paths = sorted((RUN / "shopping/episodes").glob("*/*/*.json"))
assert len(episode_paths) == len(outcomes) == len(outcome_map) == 80
controls = [RUN / x for x in ["BATCH_PLAN.json", "BOTH_FROZEN.json", "RUN_STATUS.json",
            "shopping/outcomes.json", "shopping/COMPLETED.json", "shopping/PLAN.json",
            "shopping/server.log", "logs/shopping.log", "POLICY_IMAGE_EXPOSURE_AUDIT.json"]]
code_paths = [PROJECT / x for x in ["src/agent/shopping.py", "src/experiments/shared_shopping.py",
              "src/vision/retrieval_objective.py"]]
visual_paths = [Path(g[k]) for g in groups.values() for k in ["frozen_image"]]
visual_paths += [Path(g["source"]["image"]) for g in groups.values()]
visual_paths += [Path(g["queries"][0]["query_image"]) for g in groups.values()]
evidence_paths = list(dict.fromkeys(episode_paths + controls + code_paths + visual_paths))
before = {str(p): digest(p) for p in evidence_paths}
banks = {b: read(RUN / f"shopping/policy_banks/{b}.json") for b in ["clean", "modified"]}
lookup = {b: {m["Id"]: m for m in bank} for b, bank in banks.items()}

episodes = []
calls = []
attempts = []
explicit_analyses = []
for path in episode_paths:
    group, bank, task = path.parent.parent.name, path.parent.name, path.stem
    g = groups[group]
    source, target = g["source"]["id"], g["target"]["id"]
    source_mid = f"perlage_{group}_source"
    e = read(path)
    o = outcome_map[(group, bank, task)]
    assert digest(path) == o["record_sha256"]
    assert e["Completed"] and e["FinalASIN"] == o["final_asin"]
    searches, clicked, action_names, decision_actions = [], [], [], []
    for t in e["PolicyTrace"]:
        action = t["ParsedAction"]
        action_name = action.split("[", 1)[0]
        action_names.append(action_name)
        images = t["InputImages"]
        input_source = [im for im in images if im.get("Id") == source_mid]
        query_images = [im for im in images if im.get("Role") == "query"]
        expected_source_sha = (g["source"]["image_sha256"] if bank == "clean"
                               else g["frozen_image_sha256"])
        source_image_correct = (len(input_source) == 1 and
                                input_source[0]["SHA256"] == expected_source_sha)
        assert len(query_images) == 1
        assert query_images[0]["SHA256"] == g["queries"][0]["query_image_sha256"]
        examples = t["Prompt"].split("End of examples.")[0]
        source_start = examples.find("Memory Id: " + source_mid)
        section = examples[source_start:]
        next_section = section.find("\nThis is example")
        if next_section >= 0:
            section = section[:next_section]
        history_present = (source_start >= 0 and source in section and
                           g["source"]["title"] in section and "Action: click[Buy Now]" in section)
        call = {"episode": str(path), "group": group, "bank": bank, "task": task,
                "step": t["Step"], "source_selected": source_mid in t["SelectedIds"],
                "displayed_memories": t["DisplayedMemoryCount"],
                "source_image_correct": source_image_correct,
                "source_full_history_present": history_present,
                "image_roles": [im.get("Role") for im in images],
                "executed": t["Executed"], "format_valid": t["FormatValid"],
                "invalid_action": t["InvalidAction"], "finish_reason": t["FinishReason"],
                "response_attempts": len(t["ResponseAttempts"]),
                "temperature": t["PolicyRequest"]["Temperature"]}
        calls.append(call)
        for a in t["ResponseAttempts"]:
            attempts.append(dict(a, episode=str(path), step=t["Step"]))
        if action_name == "search" or action == "click[Next >]":
            candidates = cards(t.get("Observation", ""))
            searches.append({"step": t["Step"], "action": action,
                "cards": candidates, "source_visible": any(c["asin"] == source for c in candidates),
                "target_visible": any(c["asin"] == target for c in candidates)})
        if re.fullmatch(r"click\[[A-Z0-9]{10}\]", action):
            clicked.append(action[6:-1])
        if action_name == "think":
            # 只记录外显分析动作的位置与可核对的短摘要，不复制推理全文。
            explicit_analyses.append({"episode": str(path), "step": t["Step"],
                "kind": "explicit_action_content_not_provider_hidden_reasoning",
                "summary_zh": "将参考图概括为棕色皮革、金色管架与方底；指出本轮源椅子有木扶手，另一条棕色椅子记忆的样式更接近参考图。"})
            decision_actions.append("外显分析动作（摘要另列）")
        else:
            decision_actions.append(action)
    observed_source = any(s["source_visible"] for s in searches)
    observed_target = any(s["target_visible"] for s in searches)
    # 第二条独立计数路径只看当前环境观察，绝不把示例购买历史误当当前页面。
    actual_observations = [s.removeprefix("Observation: ") for s in e["Actions"]
                           if s.startswith("Observation: ")]
    assert observed_source == any(source in IDS.findall(s) for s in actual_observations)
    assert observed_target == any(target in IDS.findall(s) for s in actual_observations)
    first = next(s for s in searches if s["action"].startswith("search["))
    chosen = next((c for c in first["cards"] if c["asin"] == e["FinalASIN"]), None)
    source_card = next((c for s in searches for c in s["cards"] if c["asin"] == source), None)
    selected = e["PolicyTrace"][0]["SelectedIds"]
    episodes.append({"group": group, "bank": bank, "task_id": task,
        "path": str(path), "sha256": digest(path), "instruction": e["Instruction"],
        "source_asin": source, "target_asin": target, "completed": e["Completed"],
        "final_asin": e["FinalASIN"], "final_title": e["PrimaryImage"]["Title"],
        "final_options": e["FinalOptions"], "environment_reward": e["EnvironmentReward"],
        "selected_ids_first_step": selected,
        "selected_asins_first_step": [lookup[bank][mid]["PrimaryImage"]["ASIN"] for mid in selected],
        "source_first_search_visible": first["source_visible"],
        "source_any_search_visible": observed_source, "target_any_search_visible": observed_target,
        "source_search_card": source_card,
        "source_clicked": source in clicked, "target_clicked": target in clicked,
        "source_purchased": e["FinalASIN"] == source, "target_purchased": e["FinalASIN"] == target,
        "search_count": sum(s["action"].startswith("search[") for s in searches),
        "pagination_count": sum(s["action"] == "click[Next >]" for s in searches),
        "inspected_product_count": len(clicked), "final_product_first_search_card": chosen,
        "actions": decision_actions, "searches": searches})

summaries = []
for group in groups:
    for bank in ["clean", "modified"]:
        es = [e for e in episodes if e["group"] == group and e["bank"] == bank]
        cs = [c for c in calls if c["group"] == group and c["bank"] == bank]
        queries = [s["action"].lower() for e in es for s in e["searches"]
                   if s["action"].startswith("search[")]
        summaries.append({"group": group, "bank": bank, "episodes": len(es),
            "source_recalled_episodes": sum(all(c["source_selected"] for c in cs
                     if c["task"] == e["task_id"]) for e in es),
            "source_visible_episodes": sum(e["source_any_search_visible"] for e in es),
            "target_visible_episodes": sum(e["target_any_search_visible"] for e in es),
            "source_clicked_episodes": sum(e["source_clicked"] for e in es),
            "source_purchase_episodes": sum(e["source_purchased"] for e in es),
            "target_purchase_episodes": sum(e["target_purchased"] for e in es),
            "one_search_episodes": sum(e["search_count"] == 1 for e in es),
            "one_product_inspected_episodes": sum(e["inspected_product_count"] == 1 for e in es),
            "final_product_on_first_page": sum(e["final_product_first_search_card"] is not None for e in es),
            "policy_calls": len(cs), "source_correct_image_calls": sum(c["source_image_correct"] for c in cs),
            "query_terms": {word: sum(word in q for q in queries)
                            for word in ["brown", "gold", "square", "blue", "star", "pink", "fmogg"]},
            "visible_source_cases": [e["task_id"] for e in es if e["source_any_search_visible"]],
            "purchase_counts": counts(e["final_asin"] for e in es),
            "environment_rewards": counts(e["environment_reward"] for e in es)})

assert len(calls) == 312 and len(attempts) == 342
assert all(c["source_selected"] and c["source_image_correct"] and c["source_full_history_present"]
           and c["displayed_memories"] == 3 for c in calls)
assert all(c["executed"] and c["format_valid"] and not c["invalid_action"]
           and c["finish_reason"] == "stop" for c in calls)
assert all(counts(c["image_roles"]) == {"query": 1, "recalled_memory": 3} for c in calls)
assert all(e["search_count"] == 1 and e["pagination_count"] == 0 and e["inspected_product_count"] == 1
           and e["final_product_first_search_card"] is not None for e in episodes)
after = {str(p): digest(p) for p in evidence_paths}
assert before == after
server = (RUN / "shopping/server.log").read_text(encoding="utf-8", errors="replace")
logs = {p.name: {"bytes": p.stat().st_size,
        "error_matches": len(re.findall(r"Traceback|Exception|ERROR|Max retries reached|Execution failed|Invalid action!",
                                  p.read_text(encoding="utf-8", errors="replace")))}
        for p in sorted((RUN / "logs").glob("*.log"))}
result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
    "analysis_only": True, "new_model_calls": 0, "new_shopping_episodes": 0,
    "evidence_preserved": before == after, "checked_evidence_sha256": before,
    "independent_observation_recount_passed": True, "episodes": len(episodes),
    "policy_calls": len(calls), "provider_attempts": len(attempts),
    "provider_attempts_with_reasoning": sum(a["ReasoningPresent"] for a in attempts),
    "reasoning_body_persisted": False, "explicit_analysis_action_count": len(explicit_analyses),
    "call_attempt_count_distribution": counts(c["response_attempts"] for c in calls),
    "retried_policy_calls": sum(c["response_attempts"] > 1 for c in calls),
    "length_empty_attempts": sum(a["FinishReason"] == "length" and a["ContentCharacters"] == 0 for a in attempts),
    "successful_action_calls": sum(c["format_valid"] and c["executed"] for c in calls),
    "image_roles_total": counts(role for c in calls for role in c["image_roles"]),
    "current_catalog_images_as_policy_input": 0,
    "fixed_target_records_in_banks": {bank: {group: sum(
        m.get("PrimaryImage", {}).get("ASIN") == g["target"]["id"] for m in entries)
        for group, g in groups.items()} for bank, entries in banks.items()},
    "temperature_values": sorted(set(c["temperature"] for c in calls)),
    "response_attempts": attempts, "summary_rows": summaries, "explicit_analyses": explicit_analyses,
    "log_review": logs, "server_http_statuses": counts(re.findall(r'HTTP/1\.1"\s+(\d{3})', server)),
    "server_log_has_non_utf8_bytes": "\ufffd" in server,
    "limitations_zh": ["服务商推理正文未落盘，不能据元数据还原内部思考。",
        "每组只有一张参考图，二十条文字为同图不同措辞；样本不代表二十种独立视觉条件。",
        "温度为0.5，未指定成对随机种子；其他商品选择分布的差异不能直接解释为修改图片的因果效果。",
        "商城商品页图片被环境保存但未进入策略视觉载荷；本轮不能作为完整看图比货能力的测试。",
        "当前私有正式结果仅作事后诊断，不用于重选图、调参或重新训练。"],
    "episode_records": episodes}
DEST.mkdir(parents=True, exist_ok=True)
(DEST / "BEHAVIOR_AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: result[k] for k in ["episodes", "policy_calls", "provider_attempts", "evidence_preserved",
                 "retried_policy_calls", "length_empty_attempts", "image_roles_total", "summary_rows"]},
                 ensure_ascii=False, indent=2))
