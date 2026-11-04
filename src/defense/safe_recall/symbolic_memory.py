"""Produce an entity-free shadow of a historical action trace.

This is a conservative syntax transformation, not an attack classifier or
entity-recognition model. All historical parameter values and free prose are
discarded. Only public operations, page structure and typed parameter slots
remain; source authenticity is never inferred from the transformed trace.
"""

from __future__ import annotations

import json

_TABS = {"Attributes", "Description", "Features"}
_OPTION_TYPES = {"size", "scent", "color", "style", "flavor", "material"}


def symbolize_trace(text: str) -> str:
    events: list[dict[str, object]] = []
    for line in text[:64000].splitlines()[:1024]:
        if line.startswith("Action: "):
            command = line[len("Action: "):].strip()
            verb, separator, argument = command.partition("[")
            if not separator or not argument.endswith("]"):
                continue
            argument = argument[:-1]
            if verb == "search":
                events.append({"op": "search", "args": ["current_target"]})
            elif verb == "think":
                events.append({"op": "reason", "args": ["current_target", "current_candidate"]})
            elif verb == "click":
                if argument in _TABS:
                    events.append({"op": "detail", "args": ["current_candidate"], "tab": argument})
                elif argument == "Buy Now":
                    # A terminal event describes history; it grants no action.
                    events.append({"op": "historical_end"})
                elif argument in {"Next >", "Back to Search"}:
                    events.append({"op": "continue_search", "args": ["current_target"]})
                elif argument == "< Prev":
                    events.append({"op": "return_to_current_page"})
                elif len(argument) == 10 and argument.isascii() and argument.isalnum():
                    events.append({"op": "inspect", "args": ["current_candidate"]})
                else:
                    events.append({"op": "check_options", "args": ["current_target", "current_candidate"]})
        else:
            # Retain a public option *type*, never its historical labels.
            heading, separator, _values = line.strip().partition("[")
            heading = heading.strip().casefold()
            if separator and heading in _OPTION_TYPES:
                events.append({"op": "visible_option_type", "type": heading})
    return json.dumps({"events": events[:64]}, ensure_ascii=False, separators=(",", ":"))


def symbolize_memories(memories: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(symbolize_trace(text) for text in memories[:5])
