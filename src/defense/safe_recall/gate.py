"""Independent task-field, evidence, option and page-state execution gate."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

from .contracts import Assessment, Observation, Request, Target
from .ledger import EvidenceLedger
from .matching import (
    normalize_text, parse_volume, resolve_suggested_options, selected_option, values_match,
    value_supported_in_span,
)
from .requirements import canonical_field, required_option_types, requirements_for


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str
    required_missing: tuple[str, ...] = ()
    optional_missing: tuple[str, ...] = ()
    conflict_fields: tuple[str, ...] = ()
    details: tuple[str, ...] = ()


class ExecutionGate:
    def __init__(self, *, score_threshold: float = 0.95):
        if not 0 <= score_threshold <= 1:
            raise ValueError("Score threshold outside [0, 1]")
        self.score_threshold = score_threshold

    def check(self, request: Request, target: Target, observation: Observation,
              assessment: Assessment, ledger: EvidenceLedger) -> GateDecision:
        requirements = requirements_for(request, target)
        required_missing, optional_missing = requirements.split_missing(
            assessment.missing_fields)

        def reject(reason: str, *, missing: tuple[str, ...] = (),
                   conflicts: tuple[str, ...] = (), details: tuple[str, ...] = ()) -> GateDecision:
            return GateDecision(False, reason, missing, optional_missing,
                                conflicts, details)

        if observation.page != "item" or not observation.handle:
            return reject("not_on_current_item")
        if ledger.current_item(observation.handle) != observation:
            return reject("stale_candidate_evidence")
        if request.image_bytes and not observation.image_bytes:
            return reject("current_product_image_missing",
                          missing=("product_identity",),
                          details=("current_reference_identity_evidence_missing",))
        reported_conflicts = tuple(getattr(assessment, "conflict_fields", ()))
        blocking_conflicts, _ = requirements.split_conflicts(reported_conflicts)
        if blocking_conflicts:
            return reject("target_field_conflict", conflicts=blocking_conflicts,
                          details=("conflict_in_required_or_core_identity_field",))
        if assessment.conflict and not reported_conflicts:
            return reject("unspecified_candidate_conflict",
                          conflicts=("unspecified",),
                          details=("legacy_conflict_requires_field_evidence",))
        if required_missing:
            return reject("required_evidence_missing", missing=required_missing,
                          details=("task_required_fields_not_established",))
        if assessment.match_score < self.score_threshold:
            return reject("absolute_match_score_below_threshold")
        observed_fields = {canonical_field(key): value
                           for key, value in assessment.observed_fields.items()}
        evidence_spans = {canonical_field(key): value
                          for key, value in assessment.evidence_spans.items()}
        if (len(observed_fields) != len(assessment.observed_fields)
                or len(evidence_spans) != len(assessment.evidence_spans)):
            return reject("ambiguous_assessment_fields")
        required_types = required_option_types(target, observation.option_types)
        try:
            suggestions = resolve_suggested_options(
                observation, assessment.suggested_options,
                required_types=required_types)
        except ValueError:
            return reject("option_not_currently_visible",
                          details=("required_suggestion_has_no_unique_current_option",))

        for original_field, required in target.values.items():
            field = canonical_field(original_field)
            if field == "max_price":
                if observation.price is None:
                    return reject("current_price_missing", missing=(field,))
                try:
                    maximum = float(required)
                except (ValueError, TypeError):
                    return reject("invalid_budget")
                if not math.isfinite(maximum) or maximum < 0:
                    return reject("invalid_budget")
                if not math.isfinite(observation.price) or observation.price < 0:
                    return reject("current_price_missing", missing=(field,))
                if observation.price > maximum:
                    return reject("budget_exceeded", conflicts=(field,))
                continue
            admitted = target.admitted.get(original_field)
            if field == "size" and admitted:
                scope = admitted.scope
                if scope.get("kind") == "body_measurement":
                    option = next((label for kind, label in suggestions.items()
                                   if normalize_text(kind) == "size"), None)
                    chart = [(label, bounds) for label, bounds in assessment.size_chart.items()
                             if option and normalize_text(label) == normalize_text(option)]
                    span = assessment.evidence_spans.get("size_chart", "")
                    if (not option or len(chart) != 1
                            or not ledger.cited_here(observation, span)):
                        return reject("current_size_chart_missing", missing=(field,))
                    bounds = chart[0][1]
                    quoted_numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", span)]
                    if (option.casefold() not in span.casefold()
                            or scope["dimension"].casefold() not in span.casefold()
                            or len(quoted_numbers) < 2
                            or quoted_numbers[-2:] != list(bounds)):
                        return reject("size_chart_citation_invalid", missing=(field,))
                    if not (bounds[0] <= float(scope["min_cm"])
                            and float(scope["max_cm"]) <= bounds[1]):
                        return reject("body_range_not_covered", conflicts=(field,))
                    continue
                if scope.get("kind") == "brand_label":
                    brand = scope.get("brand", "")
                    observed_brand = observed_fields.get("brand", "")
                    if (normalize_text(observed_brand) != normalize_text(brand)
                            or not ledger.cited_here(observation, evidence_spans.get("brand", ""))
                            or not value_supported_in_span("brand", observed_brand,
                                                          evidence_spans.get("brand", ""))):
                        return reject("size_brand_scope_mismatch", conflicts=("brand",))
            observed = observed_fields.get(field)
            if observed is None:
                return reject("target_field_not_matched:" + field, missing=(field,),
                              details=("required_current_value_missing",))
            if not values_match(field, required, observed):
                return reject("target_field_not_matched:" + field, conflicts=(field,),
                              details=("current_value_differs_from_task",))
            span = evidence_spans.get(field, "")
            if not ledger.cited_here(observation, span):
                return reject("uncited_current_evidence:" + field, missing=(field,))
            if not value_supported_in_span(field, observed, span):
                reason = ("volume_citation_not_supporting_value"
                          if parse_volume(observed) is not None else "citation_not_supporting_value")
                return reject(reason, missing=(field,),
                              details=("normalized_value_requires_matching_current_citation",))

        for option_type, label in suggestions.items():
            selected = selected_option(observation, option_type)
            if selected is None or selected.label != label:
                return reject("required_option_not_selected",
                              details=("required_current_suggestion_not_selected",))

        target_fields = {canonical_field(field): value
                         for field, value in target.values.items()}
        for field, option_kind in (("size", "size"), ("volume", "size"),
                                   ("scent", "scent"), ("color", "color"),
                                   ("style", "style"), ("flavor", "flavor")):
            if field not in target_fields:
                continue
            visible = [(label, kind) for label, kind in observation.option_types.items()
                       if normalize_text(kind) == option_kind
                       or canonical_field(kind) == field
                       or (field == "size" and parse_volume(target_fields[field]) is not None
                           and canonical_field(kind) == "volume")]
            if not visible:
                continue
            scope = target.admitted.get(field)
            if field == "size" and scope and scope.scope.get("kind") == "body_measurement":
                selected = selected_option(observation, visible[0][1])
                suggested = next((label for kind, label in suggestions.items()
                                  if normalize_text(kind) == "size"), None)
                if selected is None or selected.label != suggested:
                    return reject("size_chart_option_not_selected", missing=(field,))
                continue
            selected = selected_option(observation, visible[0][1])
            if selected is None:
                return reject("target_" + field + "_option_not_selected", missing=(field,))
            if not values_match(field, target_fields[field], selected.label):
                return reject("target_" + field + "_option_not_selected", conflicts=(field,))
        if "variant" in target_fields:
            matching_types = {kind for label, kind in observation.option_types.items()
                              if values_match("variant", target_fields["variant"], label)}
            if matching_types and not any(
                (selected := selected_option(observation, kind)) is not None
                and values_match("variant", target_fields["variant"], selected.label)
                for kind in matching_types
            ):
                return reject("target_variant_option_not_selected",
                              missing=("variant",))
        return GateDecision(True, "eligible", optional_missing=optional_missing)
