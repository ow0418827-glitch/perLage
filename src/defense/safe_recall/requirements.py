"""Task-owned field necessity, independent of candidate/model suggestions.

A candidate may report unknown attributes, but cannot enlarge the user's goal.
These rules do not infer an unmentioned size, scent or variant from a page.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .contracts import Request, Target
from .matching import normalize_text, parse_volume, values_match


_ALIASES = {
    "identity": "product_identity", "product": "product_identity",
    "商品身份": "product_identity", "商品": "product_identity",
    "品牌": "brand", "品类": "category", "型号": "model",
    "尺码": "size", "容量": "volume", "capacity": "volume",
    "香型": "scent", "颜色": "color", "colour": "color",
    "款式": "style", "规格": "variant", "预算": "max_price",
}
_CORE_CONFLICTS = frozenset({
    "product_identity", "target_identity", "brand", "category", "model",
})


def canonical_field(name: str) -> str:
    value = normalize_text(name).replace(" ", "_")
    return _ALIASES.get(value, value)


@dataclass(frozen=True)
class RequirementSet:
    required_fields: frozenset[str]
    core_conflict_fields: frozenset[str] = _CORE_CONFLICTS

    def split_missing(self, fields: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        canonical = {canonical_field(field) for field in fields}
        return (tuple(sorted(canonical & self.required_fields)),
                tuple(sorted(canonical - self.required_fields)))

    def split_conflicts(self, fields: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        canonical = {canonical_field(field) for field in fields}
        blocking = self.required_fields | self.core_conflict_fields
        return (tuple(sorted(canonical & blocking)),
                tuple(sorted(canonical - blocking)))


def requirements_for(request: Request, target: Target) -> RequirementSet:
    required = {canonical_field(field) for field in target.values}
    if request.image_bytes:
        # Identity must still match the current reference image. This does not
        # turn every dropdown offered by the seller into a required attribute.
        required.add("product_identity")
    return RequirementSet(frozenset(required))


def required_option_types(target: Target,
                          option_types: Mapping[str, str]) -> frozenset[str]:
    """Return only option types needed by the existing task contract.

    A generic variant is mapped through this page's actual visible labels.
    Extra page attributes never create new task requirements.
    """
    fields = {canonical_field(field): value for field, value in target.values.items()}
    required = set()
    direct = {"size": "size", "volume": "size", "scent": "scent",
              "color": "color", "style": "style", "flavor": "flavor"}
    for field, option_type in direct.items():
        if field in fields:
            required.add(option_type)
    for label, option_type in option_types.items():
        if "variant" in fields and values_match("variant", fields["variant"], label):
            required.add(normalize_text(option_type))
        # Some pages name a volume selector Capacity or Volume rather than Size.
        if canonical_field(option_type) in fields:
            required.add(normalize_text(option_type))
        if ("size" in fields and parse_volume(fields["size"]) is not None
                and canonical_field(option_type) == "volume"):
            required.add(normalize_text(option_type))
    return frozenset(required)
