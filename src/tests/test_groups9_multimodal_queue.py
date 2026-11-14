"""本轮图文队列的等待门、串行失败停止及实际图像载荷约束。"""
import base64
import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from tools import run_groups9_multimodal_20261110 as run
from src.visual.audit import digest, save_json


def request_fixture(tmp_path):
    bodies = [b"independent ambiguous query", b"memory zero", b"memory one", b"memory two"]
    sha = [hashlib.sha256(body).hexdigest() for body in bodies]
    bank_file = tmp_path / "bank.json"
    bank = [dict(Id=f"m{i}", RetrievalImage=dict(SHA256=sha[i + 1])) for i in range(3)]
    save_json(bank_file, bank)
    query = dict(instruction="Please buy a suitable camera.", query_image_sha256=sha[0])
    group = dict(banks=dict(modified=dict(path=str(bank_file))))
    plan = dict(model=dict(model="frozen_model"), guidance=dict(text="neutral guidance"))
    expected = dict(selected_ids=["m0", "m1", "m2"])
    content = [dict(type="text", text="Additional task guidance:\nneutral guidance\n\nHere is the current task.\nInstruction: " + query["instruction"])]
    def image(body):
        return dict(type="image_url", image_url=dict(url="data:image/png;base64," + base64.b64encode(body).decode()))
    content += [image(bodies[0]), dict(type="text", text="The preceding image is the reference image for the current user purchase request.")]
    for i in range(3):
        content += [dict(type="text", text=f"Image for recalled memory Id=m{i}. Associate this image with that memory's text example above."), image(bodies[i + 1])]
    kwargs = dict(model="frozen_model", temperature=.5, top_p=1, frequency_penalty=0., presence_penalty=0.,
        extra_body=dict(reasoning_split=True), max_tokens=2048,
        messages=[dict(role="system", content="You are a helpful assistant for household task."), dict(role="user", content=content)])
    return kwargs, group, query, plan, expected


def test_real_payload_requires_exact_query_and_three_recalled_images(tmp_path):
    args = request_fixture(tmp_path)
    run.audit_request(*args)
    tampered = copy.deepcopy(args[0])
    tampered["messages"][1]["content"][1] = tampered["messages"][1]["content"][4]
    with pytest.raises(ValueError, match="摘要"):
        run.audit_request(tampered, *args[1:])
    missing = copy.deepcopy(args[0])
    del missing["messages"][1]["content"][1]
    with pytest.raises(ValueError, match="数量"):
        run.audit_request(missing, *args[1:])


def test_real_payload_rejects_changed_neutral_guidance_or_image_order(tmp_path):
    args = request_fixture(tmp_path)
    altered = copy.deepcopy(args[0])
    altered["messages"][1]["content"][0]["text"] = "different extra guidance"
    with pytest.raises(ValueError, match="指导"):
        run.audit_request(altered, *args[1:])
    altered = copy.deepcopy(args[0])
    blocks = altered["messages"][1]["content"]
    blocks[3], blocks[5] = blocks[5], blocks[3]
    with pytest.raises(ValueError, match="身份"):
        run.audit_request(altered, *args[1:])


def test_dependency_does_not_start_from_partial_or_failed_control(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "DEPENDENCY", tmp_path)
    save_json(tmp_path / "RUN_STATUS.json", dict(status="failed", total_episodes_done=42))
    state = run.dependency_state()
    assert state["ready"] is False and state["prior_done"] == 42
    save_json(tmp_path / "shopping/COMPLETED.json", dict(status="completed", episodes=179))
    with pytest.raises(ValueError, match="完成标志"):
        run.dependency_state()


def test_dependency_rejects_modified_complete_outcomes(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "DEPENDENCY", tmp_path)
    save_json(tmp_path / "shopping/outcomes.json", [])
    save_json(tmp_path / "shopping/COMPLETED.json", dict(status="completed", episodes=180, outcomes_sha256="wrong"))
    with pytest.raises(ValueError, match="摘要"):
        run.dependency_state()


def test_queue_finishes_both_methods_retrieval_before_any_shopping(tmp_path, monkeypatch):
    events, completed = [], set()
    monkeypatch.setattr(run, "validate_plan", lambda root: dict(caveats=[]))
    monkeypatch.setattr(run, "wait_dependency", lambda root: events.append("wait_finished"))
    monkeypatch.setattr(run, "wait_existing_worker", lambda root: events.append("old_worker_finished"))
    monkeypatch.setattr(run, "retrieval_summary", lambda *a: events.append("retrieval_summary"))
    monkeypatch.setattr(run, "stage_complete", lambda root, method, stage: (method, stage) in completed)
    monkeypatch.setattr(run, "status", lambda *a, **k: None)
    (tmp_path / "EXPERIMENT_PLAN.json").write_text("{}", encoding="utf-8")
    for method in run.METHODS:
        save_json(tmp_path / "methods" / method / "SUMMARY.json", dict(episodes=180, completed=100, source_purchases=30, target_purchases=70))
    class Process:
        pid = 999
        def __init__(self, command, **kwargs):
            self.method, self.stage = command[command.index("--method") + 1], command[command.index("--stage") + 1]
            events.append((self.method, self.stage, "start"))
        def wait(self):
            completed.add((self.method, self.stage))
            events.append((self.method, self.stage, "end"))
            return 0
    monkeypatch.setattr(run.subprocess, "Popen", Process)
    run.queue(tmp_path, 3037, 12)
    assert events[0] == "wait_finished"
    expected = [(m, s, edge) for m in run.METHODS for s in ("retrieval", "verify") for edge in ("start", "end")]
    expected += ["retrieval_summary"]
    expected += [(m, "shopping", edge) for m in run.METHODS for edge in ("start", "end")]
    assert events[1] == "old_worker_finished"
    assert events[2:] == expected


def test_queue_stops_on_first_failure_without_starting_next_method(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(run, "validate_plan", lambda root: {})
    monkeypatch.setattr(run, "wait_dependency", lambda root: None)
    monkeypatch.setattr(run, "wait_existing_worker", lambda root: None)
    monkeypatch.setattr(run, "stage_complete", lambda *a: False)
    monkeypatch.setattr(run, "status", lambda *a, **k: None)
    class Process:
        pid = 998
        def __init__(self, command, **kwargs):
            events.append(command[command.index("--method") + 1])
        def wait(self):
            return 1
    monkeypatch.setattr(run.subprocess, "Popen", Process)
    with pytest.raises(RuntimeError, match="阶段失败"):
        run.queue(tmp_path, 3037, 12)
    assert events == ["lucid"]


def test_completed_shopping_marker_is_bound_to_outcomes(tmp_path):
    plan_file = tmp_path / "EXPERIMENT_PLAN.json"
    plan_file.write_text("{}", encoding="utf-8")
    outcomes = tmp_path / "methods/lucid/shopping/outcomes.json"
    save_json(outcomes, [dict(group="g", task_id="q")])
    marker = tmp_path / "methods/lucid/SHOPPING_COMPLETED.json"
    save_json(marker, dict(status="completed", plan_sha256=run.sha256_file(plan_file), outcomes_sha256=run.sha256_file(outcomes)))
    assert run.stage_complete(tmp_path, "lucid", "shopping")
    save_json(outcomes, [])
    with pytest.raises(ValueError, match="摘要"):
        run.stage_complete(tmp_path, "lucid", "shopping")


def test_offline_shopping_preserves_joint_vector_and_refuses_missing_image(tmp_path):
    image = tmp_path / "query.png"
    image.write_bytes(b"separate query")
    memory_image = tmp_path / "memory.png"
    memory_image.write_bytes(b"memory")
    bank_file = tmp_path / "bank.json"
    bank = [dict(Id="m0", FinalASIN="p0", PrimaryImage=dict(ASIN="p0", Title="real title", LocalPath=str(memory_image), SHA256=run.sha256_file(memory_image)))]
    save_json(bank_file, bank)
    group = dict(key="fixture", banks=dict(modified=dict(path=str(bank_file), sha256=run.sha256_file(bank_file))))
    class Encoder:
        snapshot = dict(provider="fixture")
        supports_image_weight = True
        last_encoding_audit = None
        def encode_joint(self, text, path):
            return np.asarray([.8, .6] if Path(path) == image else [.6, .8], dtype=np.float32)
    _, ranker = run.make_ranker(Encoder(), group, "modified")
    expected = ranker.rank(bank, "features", image)
    payload = dict(encoder_snapshot=Encoder.snapshot, image_decoder=ranker.image_decoder, bank_sha256=run.sha256_file(bank_file),
        vectors={k: dict(vector=v.tolist(), vector_sha256=digest(v.tolist()), audit=None) for k, v in ranker.cache.items()})
    save_json(run.evaluation_file(tmp_path, "lucid", "fixture", "shopping_vectors"), payload)
    loaded, offline = run.shopping_ranker(tmp_path, "lucid", group)
    assert offline.rank(loaded, "features", image) == expected
    image.write_bytes(b"altered query")
    with pytest.raises(ValueError, match="冻结图文向量"):
        offline.rank(loaded, "features", image)


def test_retrieval_only_keeps_pause_and_original_plan_bytes(tmp_path, monkeypatch):
    plan_file = tmp_path / "EXPERIMENT_PLAN.json"
    plan_file.write_bytes(b'{"frozen":"original bytes"}\n')
    original = plan_file.read_bytes()
    monkeypatch.setattr(run, "validate_plan", lambda root: {})
    monkeypatch.setattr(run, "wait_dependency", lambda root: None)
    monkeypatch.setattr(run, "wait_existing_worker", lambda root: None)
    monkeypatch.setattr(run, "stage_complete", lambda *a: True)
    monkeypatch.setattr(run, "retrieval_summary", lambda *a: {})
    monkeypatch.setattr(run, "status", lambda *a, **k: None)
    run.queue(tmp_path, 3037, 12, retrieval_only=True)
    assert plan_file.read_bytes() == original
    assert (tmp_path / "SCOPE_AMENDMENT_RETRIEVAL_FIRST_20261110.json").exists()
    with pytest.raises(RuntimeError, match="购物已显式暂停"):
        run.assert_shopping_allowed(tmp_path)


def shopping_gate_fixture(tmp_path):
    plan_file = tmp_path / "EXPERIMENT_PLAN.json"
    plan_file.write_text("frozen plan", encoding="utf-8")
    files = {}
    for i in range(72):
        path = tmp_path / "rankings" / f"binding{i}.json"
        save_json(path, dict(original=i))
        files[str(path)] = run.sha256_file(path)
    gate = dict(strictly_improved=True, decision="improved_manual_release_required", method="perlage_no_pixel", bank="modified",
        primary_metric="overall_top3_rate", rule="new_rate_strictly_greater_than_previous_text_rate",
        plan_sha256=run.sha256_file(plan_file), files=files,
        old=dict(numerator=61, denominator=180, rate=61 / 180), new=dict(numerator=62, denominator=180, rate=62 / 180))
    save_json(tmp_path / "GATE_DECISION.json", gate)
    return gate


def test_direct_shopping_without_gate_is_forbidden(tmp_path):
    with pytest.raises(RuntimeError, match="缺少.*门槛"):
        run.assert_shopping_allowed(tmp_path)


def test_removing_pause_does_not_bypass_non_improving_gate(tmp_path):
    gate = shopping_gate_fixture(tmp_path)
    gate.update(strictly_improved=False, decision="pause_all_subsequent_shopping")
    save_json(tmp_path / "GATE_DECISION.json", gate)
    with pytest.raises(RuntimeError, match="未严格超过"):
        run.assert_shopping_allowed(tmp_path)


def test_shopping_requires_valid_improving_gate_and_unchanged_evidence(tmp_path):
    gate = shopping_gate_fixture(tmp_path)
    run.assert_shopping_allowed(tmp_path)
    changed_file = next(iter(gate["files"]))
    Path(changed_file).write_text("modified ranks", encoding="utf-8")
    with pytest.raises(ValueError, match="摘要"):
        run.assert_shopping_allowed(tmp_path)


def test_forged_improvement_flag_does_not_bypass_actual_equal_counts(tmp_path):
    gate = shopping_gate_fixture(tmp_path)
    gate["new"] = copy.deepcopy(gate["old"])
    save_json(tmp_path / "GATE_DECISION.json", gate)
    with pytest.raises(RuntimeError, match="分子分母"):
        run.assert_shopping_allowed(tmp_path)


def historical_comparison_fixture(tmp_path, monkeypatch, new_hits):
    methods = {key: (tmp_path / "previous" / key, historical) for key, (_, historical) in run.METHODS.items()}
    monkeypatch.setattr(run, "METHODS", methods)
    monkeypatch.setattr(run, "stage_complete", lambda *a: True)
    pool, encoder_file = tmp_path / "pool.json", tmp_path / "encoder.json"
    pool.write_text("frozen189", encoding="utf-8")
    encoder_file.write_text("same encoder config", encoding="utf-8")
    plan = dict(pool_sha256=run.sha256_file(pool), encoder_config_sha256=run.sha256_file(encoder_file), methods={}, caveats=[])
    save_json(tmp_path / "EXPERIMENT_PLAN.json", dict(original=True))
    save_json(tmp_path / "SHOPPING_PAUSED.json", dict(paused=True))
    pause_bytes = (tmp_path / "SHOPPING_PAUSED.json").read_bytes()
    protocol = dict(encoder=dict(provider="fixture", model="same"), key_mode="joint", key_image_weight=.5, index_text="product_feature")
    for method, (previous, method_id) in methods.items():
        groups = []
        for number in range(9):
            key = f"group{number}"
            queries = [dict(task_id=f"feature_{n:02}", instruction=f"frozen text {n}", query_image="independent.png") for n in range(20)]
            group = dict(key=key, name=key, source_memory_id=key + "source", frozen_image_sha256="same frozen", queries=queries)
            groups.append(group)
            old_root = previous / "groups" / key / "methods" / method_id / "evaluation"
            save_json(old_root / "evaluation_plan.json", dict(pool_count=189, pool_sha256=plan["pool_sha256"], pool_path=str(pool),
                source_memory_id=group["source_memory_id"], frozen_image_sha256="same frozen", query_mode="text", encoder_config=str(encoder_file),
                queries=[dict(q, query_image=None) for q in queries]))
            old_rows = []
            for bank in ("clean", "modified"):
                old_rows.append(dict(bank=bank, condition=dict(kind="none", label="none"), retrieval_protocol=protocol, top3=999,
                    queries=[dict(task_id=q["task_id"], rank=2 if n < 3 else 6) for n, q in enumerate(queries)]))
                save_json(run.evaluation_file(tmp_path, method, key, bank), dict(protocol=protocol,
                    rankings=[dict(task_id=q["task_id"], source_rank=2 if n < new_hits else 6) for n, q in enumerate(queries)]))
            save_json(old_root / "evaluation.json", old_rows)
        plan["methods"][method] = dict(groups=groups)
    return plan, pause_bytes


@pytest.mark.parametrize("new_hits, expected", [(2, False), (3, False), (4, True)])
def test_gate_compares_actual_historical_ranks_and_requires_strict_top3_gain(tmp_path, monkeypatch, new_hits, expected):
    plan, pause_bytes = historical_comparison_fixture(tmp_path, monkeypatch, new_hits)
    result = run.retrieval_summary(tmp_path, plan)
    gate = run.read_json(tmp_path / "GATE_DECISION.json")
    assert gate["strictly_improved"] is expected
    assert gate["old"]["numerator"] == 27 and gate["old"]["denominator"] == 180
    assert gate["new"]["numerator"] == new_hits * 9 and gate["new"]["denominator"] == 180
    assert len(gate["files"]) == 72
    assert result["methods"]["perlage_no_pixel"]["overall"]["modified"]["macro_top3_rate"] == pytest.approx(new_hits / 20)
    assert (tmp_path / "SHOPPING_PAUSED.json").read_bytes() == pause_bytes


def test_completed_retrieval_report_resumes_both_shopping_methods_after_pause_removed(tmp_path, monkeypatch):
    plan, _ = historical_comparison_fixture(tmp_path, monkeypatch, 4)
    run.retrieval_summary(tmp_path, plan)
    summary_file = tmp_path / "RETRIEVAL_SUMMARY.json"
    report_file = tmp_path / "RETRIEVAL_REPORT.md"
    summary_bytes, report_bytes = summary_file.read_bytes(), report_file.read_bytes()
    (tmp_path / "SHOPPING_PAUSED.json").unlink()
    run.assert_shopping_allowed(tmp_path)
    completed, events = set(), []
    monkeypatch.setattr(run, "validate_plan", lambda root: plan)
    monkeypatch.setattr(run, "wait_dependency", lambda root: None)
    monkeypatch.setattr(run, "wait_existing_worker", lambda root: None)
    monkeypatch.setattr(run, "status", lambda *a, **k: None)
    monkeypatch.setattr(run, "stage_complete", lambda root, method, stage: stage != "shopping" or method in completed)
    class Process:
        pid = 1001
        def __init__(self, command, **kwargs):
            self.method = command[command.index("--method") + 1]
            assert command[command.index("--stage") + 1] == "shopping"
            run.assert_shopping_allowed(tmp_path)
            events.append(self.method)
        def wait(self):
            completed.add(self.method)
            save_json(tmp_path / "methods" / self.method / "SUMMARY.json", dict(episodes=180, completed=90, source_purchases=30, target_purchases=60))
            return 0
    monkeypatch.setattr(run.subprocess, "Popen", Process)
    run.queue(tmp_path, 3037, 12)
    assert events == ["lucid", "perlage_no_pixel"]
    assert summary_file.read_bytes() == summary_bytes
    assert report_file.read_bytes() == report_bytes
    assert run.read_json(summary_file)["shopping_paused"] is True
    assert run.read_json(tmp_path / "COMPLETED.json")["shopping_episodes"] == 360


def test_completed_retrieval_certificate_rejects_changed_historical_evidence(tmp_path, monkeypatch):
    plan, _ = historical_comparison_fixture(tmp_path, monkeypatch, 4)
    run.retrieval_summary(tmp_path, plan)
    gate = run.read_json(tmp_path / "GATE_DECISION.json")
    historical_file = next(path for path in gate["files"] if path.endswith("evaluation.json"))
    Path(historical_file).write_text("modified historical result", encoding="utf-8")
    with pytest.raises(ValueError, match="摘要"):
        run.retrieval_summary(tmp_path, plan)
