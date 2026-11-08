"""Bounded display normalization and unique current-page option resolution."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import re
from typing import Mapping
import unicodedata

from .contracts import Observation


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


@dataclass(frozen=True)
class Volume:
    milliliters: Decimal
    pack_count: int


_VOLUME = re.compile(
    r"(?P<number>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>fl\.?\s*oz\.?|fluid\s+ounces?|milliliters?|millilitres?|ml|liters?|litres?|l|毫升|升)"
    r"(?:\s*\(?\s*pack\s+of\s+(?P<pack>\d+)\s*\)?)?",
    re.IGNORECASE,
)
_FACTORS = {"l": Decimal(1000), "liter": Decimal(1000),
            "liters": Decimal(1000), "litre": Decimal(1000),
            "litres": Decimal(1000), "升": Decimal(1000)}


def parse_volume(value: str) -> Volume | None:
    match = _VOLUME.fullmatch(normalize_text(value))
    if match is None:
        return None
    amount = Decimal(match.group("number"))
    pack = int(match.group("pack") or 1)
    if amount <= 0 or pack <= 0:
        return None
    unit = normalize_text(match.group("unit")).replace(".", "")
    factor = (Decimal("29.5735295625") if unit.startswith(("fl", "fluid"))
              else _FACTORS.get(unit, Decimal(1)))
    return Volume(amount * factor, pack)


def values_match(field: str, required: str, observed: str) -> bool:
    if normalize_text(required) == normalize_text(observed):
        return True
    # Numeric tolerance is restricted to labels carrying an explicit volume
    # unit; clothing labels such as M, L or size 38 cannot pass this branch.
    if normalize_text(field) not in {"size", "volume", "capacity", "容量"}:
        return False
    desired, actual = parse_volume(required), parse_volume(observed)
    if desired is None or actual is None or desired.pack_count != actual.pack_count:
        return False
    # 0.1% only covers display/rounding differences, not a different package.
    return abs(desired.milliliters - actual.milliliters) <= (
        max(desired.milliliters, actual.milliliters) * Decimal("0.001"))


def volume_supported_in_span(field: str, observed: str, span: str) -> bool:
    """A converted volume must occur in the actual citation, not only a claim."""
    if (normalize_text(field) not in {"size", "volume", "capacity", "容量"}
            or parse_volume(observed) is None):
        return True
    normalized_span = normalize_text(span)
    for match in _VOLUME.finditer(normalized_span):
        # Avoid treating the prefix of an arbitrary word as a unit.
        if (match.start() and normalized_span[match.start() - 1].isalnum()) or (
                match.end() < len(normalized_span)
                and normalized_span[match.end()].isalnum()):
            continue
        if values_match(field, observed, match.group(0)):
            return True
    return False


def value_supported_in_span(field: str, observed: str, span: str) -> bool:
    """Tie a normalized claim to its actual current-page quotation."""
    if normalize_text(field) in {"size", "volume", "capacity", "容量"} and parse_volume(observed):
        return volume_supported_in_span(field, observed, span)
    claimed = normalize_text(observed)
    return bool(claimed and re.search(r"(?<!\w)" + re.escape(claimed) + r"(?!\w)",
                                     normalize_text(span)))


@dataclass(frozen=True)
class VisibleOption:
    option_type: str
    label: str


def resolve_visible_option(option_types: Mapping[str, str], option_type: str,
                           label: str) -> VisibleOption | None:
    matches = [VisibleOption(actual_type, actual_label)
               for actual_label, actual_type in option_types.items()
               if normalize_text(actual_type) == normalize_text(option_type)
               and normalize_text(actual_label) == normalize_text(label)]
    return matches[0] if len(matches) == 1 else None


def resolve_suggested_options(observation: Observation,
                              suggestions: Mapping[str, str], *,
                              required_types: frozenset[str] | None = None
                              ) -> dict[str, str]:
    result: dict[str, str] = {}
    seen: set[str] = set()
    for option_type, label in suggestions.items():
        normalized_type = normalize_text(option_type)
        if required_types is not None and normalized_type not in required_types:
            continue
        visible = resolve_visible_option(observation.option_types, option_type, label)
        if visible is None or normalized_type in seen:
            raise ValueError("Suggested option is not uniquely visible on the current page")
        seen.add(normalized_type)
        result[visible.option_type] = visible.label
    return result


def selected_option(observation: Observation, option_type: str) -> VisibleOption | None:
    choices = [(actual_type, label)
               for actual_type, label in observation.selected_options.items()
               if normalize_text(actual_type) == normalize_text(option_type)]
    if len(choices) != 1:
        return None
    return resolve_visible_option(observation.option_types, *choices[0])
