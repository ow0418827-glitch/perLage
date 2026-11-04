"""Opt-in round driver; core policy does not modify the legacy shopping loop."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, replace
from typing import Callable

from .adapters import ShoppingEnvironment
from .authorization import intersect_hypotheses
from .compiler import Op, ProcedureHint, StepOp
from .contracts import (
    AdmittedField, Assessment, Candidate, Contract, Observation, Outcome, Request,
    Status, Target,
)
from .gate import ExecutionGate
from .history import HistoryResolver
from .ledger import EvidenceLedger
from .model import DecisionModel
from .matching import normalize_text, resolve_suggested_options, values_match
from .requirements import canonical_field, required_option_types, requirements_for
from .providers import HmacVerifier, SourceProvider, UnavailableSource
from .scheduler import Scheduler


@dataclass(frozen=True)
class RunnerConfig:
    max_candidates: int = 5
    max_pages: int = 2
    max_environment_steps: int = 30
    max_detail_draws: int = 5
    max_policy_draws: int = 12
    max_evidence_retries: int = 2
    procedure_kl_budget: float = 0.005
    match_threshold: float = 0.95
    allow_uncalibrated_purchase: bool = False
    random_seed: int = 0

    def __post_init__(self) -> None:
        if min(self.max_candidates, self.max_pages, self.max_environment_steps,
               self.max_detail_draws, self.max_policy_draws) < 1 or self.procedure_kl_budget < 0:
            raise ValueError("Invalid run budget")
        if self.max_evidence_retries < 0:
            raise ValueError("Evidence retry budget cannot be negative")


class SafeRecallRunner:
    def __init__(self, environment: ShoppingEnvironment, model: DecisionModel,
                 *, source: SourceProvider | None = None,
                 verifier: HmacVerifier | None = None,
                 clock: Callable[[], float] = time.time,
                 config: RunnerConfig = RunnerConfig()):
        self.environment = environment
        self.model = model
        self.config = config
        self.history = HistoryResolver(source or UnavailableSource(),
                                       verifier or HmacVerifier({}), clock=clock)
        self.gate = ExecutionGate(score_threshold=config.match_threshold)
        self.ledger = EvidenceLedger()
        self.scheduler = Scheduler(config.procedure_kl_budget,
                                   config.max_policy_draws,
                                   random.Random(config.random_seed))
        self._audit: list[dict[str, object]] = []
        self._steps = 0
        self._detail_draws = 0
        self._reference_rng = random.Random(config.random_seed + 1)
        self._detail_binding = None
        self._detail_texts: dict[Op, str] = {}

    def _record(self, event: str, **fields: object) -> None:
        self._audit.append({"event": event, **fields})

    def _result(self, status: Status, reason: str, *,
                final_handle: str | None = None,
                final_options: dict[str, str] | None = None) -> Outcome:
        return Outcome(status, reason, final_handle, final_options or {},
                       self._steps, audit=tuple(self._audit))

    def _observed(self, observation: Observation) -> Observation:
        self._steps += 1
        self.ledger.add(observation)
        return observation

    def _target(self, contract: Contract) -> tuple[Target | None, str]:
        values = dict(contract.current_fields)
        admitted: dict[str, AdmittedField] = {}
        for reference in contract.references:
            result = self.history.resolve(contract, reference)
            self._record("history_admission", field=reference.field,
                         source_kind=reference.kind, status=result.reason)
            if result.field is None:
                return None, result.reason
            if reference.field in values and values[reference.field] != result.field.value:
                return None, "current_history_conflict"
            values[reference.field] = result.field.value
            admitted[reference.field] = result.field
        return Target(contract.subject, values, admitted, contract.version), "accepted"

    def _history_still_valid(self, contract: Contract, target: Target) -> bool:
        for reference in contract.references:
            refreshed = self.history.resolve(contract, reference)
            original = target.admitted.get(reference.field)
            if (refreshed.field is None or original is None
                    or (refreshed.field.source_id, refreshed.field.source_version,
                        refreshed.field.value, refreshed.field.scope,
                        refreshed.field.source_kind, refreshed.field.purpose) !=
                    (original.source_id, original.source_version, original.value,
                     original.scope, original.source_kind, original.purpose)):
                self._record("history_changed", field=reference.field,
                             status=refreshed.reason)
                return False
        return True

    def _choose(self, allowed: tuple, preferred: tuple, *, phase: str):
        if self.scheduler.draws < self.scheduler.max_draws:
            operation, trace = self.scheduler.choose_preferences(allowed, preferred)
        else:
            # After the statistical budget, use the common reference only.
            operation = self._reference_rng.choice(allowed)
            reference = {op.value: 1 / len(allowed) for op in allowed}
            trace = {"operation": operation.value, "reference": reference,
                     "actual": reference, "kl": 0.0, "allocated": 0.0,
                     "path_allocated": self.scheduler.spent_upper_bound}
        self._record("procedure_draw", phase=phase, **trace)
        return operation

    def _detail(self, observation: Observation, allowed: tuple[Op, ...],
                hint: ProcedureHint, *, missing: bool) -> tuple[Observation, Op | None]:
        if (not allowed or self._detail_draws >= self.config.max_detail_draws
                or self._steps + 3 >= self.config.max_environment_steps):
            return observation, None
        plan = hint.program.runtime_plan(required_missing=missing,
            has_options=bool(observation.option_types), allowed_detail_tabs=allowed)
        op = self._choose(allowed, plan.preferred_tabs, phase="detail")
        subpage, current = self.environment.detail(op.value)
        self._steps += 2
        self._detail_draws += 1
        if (current.page != "item" or subpage.handle != observation.handle
                or current.handle != observation.handle
                or current.page_number != observation.page_number
                or current.selected_options != observation.selected_options):
            raise RuntimeError("Detail operation changed the current candidate binding")
        binding = (current.handle, tuple(sorted(current.selected_options.items())),
                   current.price, current.text)
        if binding != self._detail_binding:
            self._detail_texts = {}
        self._detail_binding = binding
        self._detail_texts[op] = subpage.text
        observation = replace(current, text=current.text + "\n" +
                              "\n".join(self._detail_texts.values()))
        self.ledger.add(observation)
        return observation, op

    def _assess(self, request: Request, target: Target,
                observation: Observation) -> Assessment:
        assessment = self.model.assess(request, target, observation)
        self._record("assessment", handle=observation.handle,
            required_fields=sorted(requirements_for(request, target).required_fields),
            observed_fields=dict(assessment.observed_fields),
            evidence_spans={k: v[:400] for k, v in assessment.evidence_spans.items()},
            missing_fields=list(assessment.missing_fields),
            conflict=assessment.conflict, conflict_fields=list(assessment.conflict_fields),
            suggested_options=dict(assessment.suggested_options),
            visible_options=dict(observation.option_types),
            selected_options=dict(observation.selected_options), score=assessment.match_score)
        return assessment

    def _current_target_options(self, target: Target, observation: Observation) -> dict[str, str]:
        suggestions = {}
        types = required_option_types(target, observation.option_types)
        for field, value in target.values.items():
            field = canonical_field(field)
            if field == "size" and target.admitted.get("size") and (
                    target.admitted["size"].scope.get("kind") == "body_measurement"):
                continue  # Actual current size-chart evidence is required first.
            candidates = [(kind, label) for label, kind in observation.option_types.items()
                if normalize_text(kind) in types and
                (canonical_field(kind) == field or field in {"variant", "volume", "capacity"}
                 and (field == "variant" or canonical_field(kind) == "size"))
                and values_match(field, value, label)]
            if len(candidates) == 1:
                kind, label = candidates[0]
                suggestions[kind] = label
        return suggestions

    def _select_options(self, target: Target, observation: Observation,
                        suggestions: dict[str, str]) -> tuple[Observation, bool, dict[str, str]]:
        self._record("typed_action", operation="check_options", handle=observation.handle,
                     parameter_source="current_target_and_current_page")
        required_types = required_option_types(target, observation.option_types)
        ignored = {k: v for k, v in suggestions.items() if normalize_text(k) not in required_types}
        if ignored:
            self._record("optional_options_ignored", handle=observation.handle, suggestions=ignored)
        try:
            resolved = resolve_suggested_options(observation, suggestions,
                                                required_types=required_types)
        except ValueError:
            self._record("option_rejected", handle=observation.handle,
                         reason="not_visible_or_ambiguous", suggestions=suggestions)
            return observation, False, suggestions
        changed = False
        for option_type, label in sorted(resolved.items()):
            if observation.selected_options.get(option_type) == label:
                continue
            if self._steps + 2 >= self.config.max_environment_steps:
                break
            prior = observation
            observation = self._observed(self.environment.select(label))
            self._steps += 1
            if (observation.page != "item" or observation.handle != prior.handle
                    or observation.page_number != prior.page_number):
                raise RuntimeError("Option operation changed the current candidate binding")
            self._detail_texts = {}
            self._detail_binding = None
            changed = True
            self._record("typed_action", operation="check_options", handle=observation.handle,
                         option_type=option_type, label=label, parameter_source="current_page")
        return observation, changed, resolved

    def _inspect(self, request: Request, target: Target,
                 observation: Observation, hint: ProcedureHint
                 ) -> tuple[Observation, Assessment]:
        self._detail_binding = None
        self._detail_texts = {}
        seen_tabs: set[Op] = set()
        observation, tab = self._detail(observation, tuple(Op), hint, missing=False)
        if tab:
            seen_tabs.add(tab)
        plan = hint.program.runtime_plan(required_missing=False,
                                        has_options=bool(observation.option_types))
        preferred = tuple(step.operation for step in plan.steps
                          if step.operation in {StepOp.COMPARE, StepOp.CHECK_OPTIONS})[:1]
        first = self._choose((StepOp.COMPARE, StepOp.CHECK_OPTIONS), preferred,
                             phase="mandatory_checks")
        if first == StepOp.CHECK_OPTIONS:
            observation, pre_changed, _ = self._select_options(
                target, observation, self._current_target_options(target, observation))
            if pre_changed:
                seen_tabs.clear()
        assessment = self._assess(request, target, observation)
        self._record("typed_action", operation="compare", handle=observation.handle,
                     parameter_source="current_request_and_current_page")
        observation, changed, options = self._select_options(
            target, observation, dict(assessment.suggested_options))
        if changed:
            seen_tabs.clear()
            assessment = self._assess(request, target, observation)
            # Normalize without another click: a revised suggestion must pass
            # the gate in the actual selected state, not silently flip options.
            try:
                options = resolve_suggested_options(observation, assessment.suggested_options,
                    required_types=required_option_types(target, observation.option_types))
            except ValueError:
                options = dict(assessment.suggested_options)
        assessment = replace(assessment, suggested_options=options)
        for attempt in range(self.config.max_evidence_retries):
            verdict = self.gate.check(request, target, observation, assessment, self.ledger)
            if verdict.allowed or not verdict.required_missing:
                break
            observation, tab = self._detail(observation, tuple(op for op in Op if op not in seen_tabs),
                                            hint, missing=True)
            if tab is None:
                break
            seen_tabs.add(tab)
            self._record("evidence_retry", handle=observation.handle, attempt=attempt + 1,
                         required_missing=list(verdict.required_missing), tab=tab.value)
            assessment = self._assess(request, target, observation)
            observation, changed, options = self._select_options(
                target, observation, dict(assessment.suggested_options))
            if changed:
                seen_tabs.clear()
                assessment = self._assess(request, target, observation)
                try:
                    options = resolve_suggested_options(observation, assessment.suggested_options,
                        required_types=required_option_types(target, observation.option_types))
                except ValueError:
                    options = dict(assessment.suggested_options)
            assessment = replace(assessment, suggested_options=options)
        return observation, assessment

    def run(self, request: Request, *,
            ordinary_memory: tuple[str, ...] = ()) -> Outcome:
        # A runner instance may be reused, but authorization evidence and
        # statistical budgets belong to one episode only.
        self.ledger = EvidenceLedger()
        self.scheduler = Scheduler(self.config.procedure_kl_budget,
                                   self.config.max_policy_draws,
                                   random.Random(self.config.random_seed))
        self._audit = []
        self._steps = 0
        self._detail_draws = 0
        self._reference_rng = random.Random(self.config.random_seed + 1)
        try:
            return self._run_once(request, ordinary_memory)
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            # A malformed model reply, failed observation, or exhausted model
            # budget must never fall through to a purchase call.
            self._record("episode_failure", error=type(exc).__name__)
            return self._result(Status.BLOCKED, "episode_validation_failed")

    def _run_once(self, request: Request,
                  ordinary_memory: tuple[str, ...]) -> Outcome:
        self._observed(self.environment.reset())
        try:
            hypotheses = self.model.intent(request)
        except (ValueError, KeyError, TypeError) as exc:
            self._record("intent_error", error=type(exc).__name__)
            return self._result(Status.NEEDS_CLARIFICATION, "intent_unresolved")
        contract = intersect_hypotheses(request.subject, hypotheses, request.text)
        self._record("authorization", reference_fields=[
            ref.field for ref in contract.references],
            unresolved=list(contract.unresolved))
        if contract.unresolved:
            return self._result(Status.NEEDS_CLARIFICATION,
                                "authorization_ambiguous")
        target, reason = self._target(contract)
        if target is None:
            return self._result(Status.NEEDS_EVIDENCE, reason)
        if not target.values and not request.image_bytes:
            return self._result(Status.NEEDS_CLARIFICATION,
                                "target_not_identifiable_from_current_request")
        try:
            hint = self.model.compile_procedure(ordinary_memory)
            if not isinstance(hint, ProcedureHint):
                raise ValueError("Compiler did not return a typed program")
        except (ValueError, KeyError, TypeError):
            hint = ProcedureHint(())
            self._record("procedure_rejected")
        self._record("procedure_compiled", **hint.serialize())
        query = contract.search_terms or target.values.get("target_identity", "")
        if not query:
            try:
                query = self.model.query(request, target)
            except (ValueError, KeyError, TypeError) as exc:
                self._record("query_error", error=type(exc).__name__)
                return self._result(Status.NEEDS_EVIDENCE, "search_query_unresolved")
        self._record("typed_action", operation="search", query=query,
                     parameter_source="current_authorized_target")
        if self._steps + 1 >= self.config.max_environment_steps:
            return self._result(Status.BUDGET_EXHAUSTED, "environment_step_limit")
        try:
            search = self._observed(self.environment.search(query))
        except ValueError:
            return self._result(Status.BLOCKED, "search_not_allowed")
        if search.page != "search":
            return self._result(Status.BLOCKED, "search_state_invalid")
        candidates: list[Candidate] = []
        pending_evidence: set[str] = set()
        inspected = 0
        pages = 0
        while pages < self.config.max_pages and inspected < self.config.max_candidates:
            pages += 1
            visible_ids = tuple(dict.fromkeys(search.visible_ids))
            for handle in visible_ids:
                if (inspected >= self.config.max_candidates
                        or self._steps + 4 >= self.config.max_environment_steps):
                    break
                inspected += 1
                try:
                    observation = self._observed(self.environment.open(handle))
                except ValueError:
                    self._record("candidate_rejected", reason="invalid_current_handle")
                    continue
                if observation.page != "item" or observation.handle != handle:
                    return self._result(Status.BLOCKED, "candidate_binding_invalid")
                self._record("typed_action", operation="inspect", handle=handle,
                             parameter_source="current_search_result")
                try:
                    observation, assessment = self._inspect(request, target,
                                                            observation, hint)
                except (ValueError, KeyError, TypeError) as exc:
                    self._record("candidate_rejected", handle=handle,
                                 reason="assessment_invalid",
                                 error=type(exc).__name__)
                    search = self._observed(self.environment.back())
                    continue
                verdict = self.gate.check(request, target, observation,
                                          assessment, self.ledger)
                self._record("candidate", handle=handle,
                             score=assessment.match_score, decision=verdict.reason,
                             required_missing=list(verdict.required_missing),
                             optional_missing=list(verdict.optional_missing),
                             conflict_fields=list(verdict.conflict_fields))
                pending_evidence.update(verdict.required_missing)
                if verdict.allowed:
                    candidates.append(Candidate(handle, observation, assessment))
                search = self._observed(self.environment.back())
            if (not search.has_next or pages >= self.config.max_pages
                    or inspected >= self.config.max_candidates
                    or self._steps + 5 >= self.config.max_environment_steps):
                break
            search = self._observed(self.environment.next_page())
            self._record("typed_action", operation="continue_search",
                         parameter_source="current_search_page", page=search.page_number)
        if not candidates:
            status = (Status.BUDGET_EXHAUSTED if self._steps + 4 >=
                      self.config.max_environment_steps else Status.NO_MATCH)
            if status != Status.BUDGET_EXHAUSTED and pending_evidence:
                self._record("unresolved_evidence", fields=sorted(pending_evidence))
                return self._result(Status.NEEDS_EVIDENCE, "required_evidence_missing")
            return self._result(status, "no_eligible_current_candidate")
        candidates.sort(key=lambda item: (-item.assessment.match_score, item.handle))
        for choice in candidates:
            if self._steps + 4 >= self.config.max_environment_steps:
                break
            while search.page_number > choice.observation.page_number:
                if self._steps + 3 > self.config.max_environment_steps:
                    return self._result(Status.BUDGET_EXHAUSTED, "submit_budget_exhausted")
                search = self._observed(self.environment.previous_page())
            while search.page_number < choice.observation.page_number:
                if self._steps + 3 > self.config.max_environment_steps:
                    return self._result(Status.BUDGET_EXHAUSTED, "submit_budget_exhausted")
                search = self._observed(self.environment.next_page())
            # Reopen and recheck. Search-result order, an old handle, or a
            # previous option/price snapshot cannot authorize a purchase.
            if self._steps + 2 > self.config.max_environment_steps:
                return self._result(Status.BUDGET_EXHAUSTED, "submit_budget_exhausted")
            current = self._observed(self.environment.open(choice.handle))
            if current.page != "item" or current.handle != choice.handle:
                return self._result(Status.BLOCKED, "submit_candidate_binding_invalid")
            try:
                current, assessment = self._inspect(request, target, current,
                                                    ProcedureHint(()))
            except (ValueError, KeyError, TypeError) as exc:
                self._record("submit_rejected", handle=choice.handle,
                             reason="assessment_invalid",
                             error=type(exc).__name__)
                search = self._observed(self.environment.back())
                continue
            verdict = self.gate.check(request, target, current, assessment,
                                      self.ledger)
            if not verdict.allowed:
                self._record("submit_rejected", handle=choice.handle,
                             reason=verdict.reason)
                search = self._observed(self.environment.back())
                continue
            if not self._history_still_valid(contract, target):
                return self._result(Status.NEEDS_EVIDENCE,
                                    "authorized_history_changed")
            if not self.config.allow_uncalibrated_purchase:
                self._record("submit_rejected", handle=choice.handle,
                             reason="complete_policy_risk_not_calibrated")
                return self._result(Status.BLOCKED,
                                    "complete_policy_risk_not_calibrated")
            if self._steps + 1 > self.config.max_environment_steps:
                return self._result(Status.BUDGET_EXHAUSTED, "submit_budget_exhausted")
            # The adapter checks version, handle and selected options again at
            # the actual irreversible environment call.
            try:
                done, actual_handle = self.environment.buy(current)
            except ValueError:
                return self._result(Status.BLOCKED, "submit_state_changed")
            self._steps += 1
            if not done or actual_handle != choice.handle:
                return self._result(Status.BLOCKED, "purchase_result_unverified")
            self._record("purchase", handle=actual_handle)
            return self._result(Status.PURCHASED, "current_evidence_accepted",
                                final_handle=actual_handle,
                                final_options=dict(current.selected_options))
        return self._result(Status.BUDGET_EXHAUSTED, "submit_budget_exhausted")
