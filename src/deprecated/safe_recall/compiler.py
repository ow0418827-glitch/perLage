"""Finite, typed historical procedures with no product or purchase capability.

The grammar preserves search/inspection/comparison order, while all argument
slots refer to objects owned by the current runner. A slot name is not evidence
of trust: the executor must bind it to its existing target/current candidate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


MAX_PROCEDURE_STEPS = 12


class Op(str, Enum):
    ATTRIBUTES = "Attributes"
    DESCRIPTION = "Description"
    FEATURES = "Features"


class StepOp(str, Enum):
    SEARCH = "search"
    INSPECT = "inspect"
    COMPARE = "compare"
    CHECK_OPTIONS = "check_options"
    DETAIL = "detail"
    CONTINUE_SEARCH = "continue_search"


class Slot(str, Enum):
    CURRENT_TARGET = "current_target"
    CURRENT_CANDIDATE = "current_candidate"


class Guard(str, Enum):
    ALWAYS = "always"
    REQUIRED_MISSING = "required_missing"
    HAS_OPTIONS = "has_options"


_ARGUMENTS = {
    StepOp.SEARCH: (Slot.CURRENT_TARGET,),
    StepOp.INSPECT: (Slot.CURRENT_CANDIDATE,),
    StepOp.COMPARE: (Slot.CURRENT_TARGET, Slot.CURRENT_CANDIDATE),
    StepOp.CHECK_OPTIONS: (Slot.CURRENT_TARGET, Slot.CURRENT_CANDIDATE),
    StepOp.DETAIL: (Slot.CURRENT_CANDIDATE,),
    StepOp.CONTINUE_SEARCH: (Slot.CURRENT_TARGET,),
}
_MANDATORY_CHECKS = (StepOp.COMPARE, StepOp.CHECK_OPTIONS)


@dataclass(frozen=True)
class ProcedureStep:
    operation: StepOp
    arguments: tuple[Slot, ...]
    guard: Guard = Guard.ALWAYS
    tab: Op | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, StepOp):
            raise ValueError("Procedure operation must be a typed enum")
        if (not isinstance(self.arguments, tuple)
                or any(not isinstance(slot, Slot) for slot in self.arguments)
                or self.arguments != _ARGUMENTS[self.operation]):
            raise ValueError("Procedure arguments must be exact current-state slots")
        if not isinstance(self.guard, Guard):
            raise ValueError("Procedure guard must be a typed enum")
        if self.operation == StepOp.DETAIL:
            if not isinstance(self.tab, Op):
                raise ValueError("Detail steps require a permitted typed tab")
        elif self.tab is not None:
            raise ValueError("Only a detail step can specify a tab")

    def active(self, *, required_missing: bool, has_options: bool) -> bool:
        if self.guard == Guard.REQUIRED_MISSING:
            return required_missing
        if self.guard == Guard.HAS_OPTIONS:
            return has_options
        return True

    def serialize(self) -> dict[str, object]:
        result: dict[str, object] = {
            "op": self.operation.value,
            "args": [slot.value for slot in self.arguments],
            "guard": self.guard.value,
        }
        if self.tab is not None:
            result["tab"] = self.tab.value
        return result


@dataclass(frozen=True)
class RuntimePlan:
    """Symbolic suggestions, never an authorization or direct environment call.

    Search suggestions cannot supply a query; inspect cannot choose a handle;
    check_options cannot supply a label. All bindings and budgets remain in the
    runner. check_order always contains both protected checks, including when a
    historical program is empty or its guards are false.
    """

    steps: tuple[ProcedureStep, ...]
    preferred_tabs: tuple[Op, ...]
    check_order: tuple[StepOp, ...]
    search_requested: bool
    continue_search_requested: bool


@dataclass(frozen=True)
class ProcedureProgram:
    steps: tuple[ProcedureStep, ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.steps, tuple)
                or len(self.steps) > MAX_PROCEDURE_STEPS
                or any(not isinstance(step, ProcedureStep) for step in self.steps)):
            raise ValueError("A procedure must have at most twelve typed steps")

    @property
    def required_checks(self) -> tuple[StepOp, ...]:
        # This is runner policy, not something a memory may weaken.
        return _MANDATORY_CHECKS

    def serialize(self) -> dict[str, object]:
        return {"steps": [step.serialize() for step in self.steps]}

    def runtime_plan(self, *, required_missing: bool, has_options: bool,
                     allowed_detail_tabs: tuple[Op, ...] = tuple(Op)) -> RuntimePlan:
        if (not isinstance(required_missing, bool)
                or not isinstance(has_options, bool)
                or any(not isinstance(tab, Op) for tab in allowed_detail_tabs)):
            raise ValueError("Runtime state must come from typed current evidence")
        steps = tuple(
            step for step in self.steps
            if step.active(required_missing=required_missing,
                           has_options=has_options)
            and (step.operation != StepOp.DETAIL
                 or step.tab in allowed_detail_tabs)
        )
        tabs = tuple(dict.fromkeys(step.tab for step in steps
                                   if step.operation == StepOp.DETAIL))
        ordered = tuple(dict.fromkeys(
            [step.operation for step in steps
             if step.operation in _MANDATORY_CHECKS] + list(_MANDATORY_CHECKS)))
        return RuntimePlan(
            steps, tabs, ordered,
            any(step.operation == StepOp.SEARCH for step in steps),
            any(step.operation == StepOp.CONTINUE_SEARCH for step in steps),
        )

    def preferred_tabs(self, missing: bool, has_options: bool,
                       allowed_detail_tabs: tuple[Op, ...] = tuple(Op)
                       ) -> tuple[Op, ...]:
        return self.runtime_plan(
            required_missing=missing, has_options=has_options,
            allowed_detail_tabs=allowed_detail_tabs).preferred_tabs

    def check_order(self, missing: bool, has_options: bool) -> tuple[StepOp, ...]:
        return self.runtime_plan(required_missing=missing,
                                 has_options=has_options).check_order


@dataclass(frozen=True)
class ProcedureHint:
    # The original field and positional constructor remain compatible.
    preferred: tuple[Op, ...] = ()
    program: ProcedureProgram | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.preferred, tuple)
                or any(not isinstance(tab, Op) for tab in self.preferred)):
            raise ValueError("Preferred tabs must be typed enums")
        preferred = tuple(dict.fromkeys(self.preferred))
        if self.program is None:
            program = ProcedureProgram(tuple(
                ProcedureStep(StepOp.DETAIL, (Slot.CURRENT_CANDIDATE,), tab=tab)
                for tab in preferred))
        elif isinstance(self.program, ProcedureProgram):
            program = self.program
            program_tabs = tuple(dict.fromkeys(
                step.tab for step in program.steps
                if step.operation == StepOp.DETAIL))
            if preferred and preferred != program_tabs:
                raise ValueError("Preferred tabs disagree with typed program")
            preferred = program_tabs
        else:
            raise ValueError("Procedure program must be pre-typed")
        object.__setattr__(self, "preferred", preferred)
        object.__setattr__(self, "program", program)

    def serialize(self) -> dict[str, object]:
        return self.program.serialize()


def parse_procedure_payload(payload: Mapping[str, object]) -> ProcedureHint:
    """Parse a complete allow-listed payload, rejecting every unknown field.

    Model strings are converted to enum members only after structural checks.
    There are no literals, claimed provenance, references, purchase steps,
    branches, loop targets, thresholds, or opaque/free-form action parameters.
    A legacy preferred-only payload becomes a finite detail-only procedure.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("Procedure payload must be an object")
    if set(payload) == {"preferred"}:
        raw = payload["preferred"]
        if (not isinstance(raw, list) or len(raw) > MAX_PROCEDURE_STEPS
                or any(not isinstance(value, str) for value in raw)):
            raise ValueError("Legacy preferred payload is invalid")
        return ProcedureHint(tuple(dict.fromkeys(Op(value) for value in raw)))
    if set(payload) != {"steps"}:
        raise ValueError("Unknown or missing procedure payload fields")
    raw_steps = payload["steps"]
    if not isinstance(raw_steps, list) or len(raw_steps) > MAX_PROCEDURE_STEPS:
        raise ValueError("Procedure must be a bounded step list")
    steps: list[ProcedureStep] = []
    for raw in raw_steps:
        if not isinstance(raw, Mapping):
            raise ValueError("Procedure step must be an object")
        if not isinstance(raw.get("op"), str):
            raise ValueError("Procedure operation must be an enum string")
        operation = StepOp(raw["op"])
        expected_keys = {"op", "args", "guard"}
        if operation == StepOp.DETAIL:
            expected_keys.add("tab")
        if set(raw) != expected_keys:
            raise ValueError("Unknown or missing procedure step fields")
        raw_arguments = raw["args"]
        if (not isinstance(raw_arguments, list)
                or any(not isinstance(slot, str) for slot in raw_arguments)
                or not isinstance(raw["guard"], str)):
            raise ValueError("Procedure arguments and guard must be symbolic")
        tab = raw.get("tab")
        if operation == StepOp.DETAIL and not isinstance(tab, str):
            raise ValueError("Detail tab must be an enum string")
        steps.append(ProcedureStep(
            operation, tuple(Slot(slot) for slot in raw_arguments),
            Guard(raw["guard"]), Op(tab) if tab is not None else None))
    return ProcedureHint(program=ProcedureProgram(tuple(steps)))


def compile_procedure(records: Iterable[object]) -> ProcedureHint:
    """Merge only validated typed procedures, never raw memory commands.

    Exact duplicate steps are collapsed and the merged plan is capped at twelve
    steps. Repetition cannot grow an unbounded execution path. Raw text, images,
    dictionaries, and arbitrary objects deliberately grant no capability.
    """
    steps: list[ProcedureStep] = []
    for record in records:
        if isinstance(record, ProcedureHint):
            incoming = record.program.steps
        elif isinstance(record, ProcedureProgram):
            incoming = record.steps
        elif isinstance(record, ProcedureStep):
            incoming = (record,)
        else:
            continue
        for step in incoming:
            if step not in steps and len(steps) < MAX_PROCEDURE_STEPS:
                steps.append(step)
    return ProcedureHint(program=ProcedureProgram(tuple(steps)))
