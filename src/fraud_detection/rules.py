from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class RuleOperator(StrEnum):
    """Supported comparison operators for declarative rule predicates."""

    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL = "<="
    IN = "in"
    NOT_IN = "not_in"


class RuleAction(StrEnum):
    """Decision actions assignable by declarative rules."""

    ALLOW = "ALLOW"
    CHALLENGE = "CHALLENGE"
    DENY = "DENY"


class RuleError(ValueError):
    """Base exception for rule syntax, compilation, and evaluation errors."""


@dataclass(frozen=True)
class RuleCondition:
    """A deterministic predicate condition comparing a named record field against a target value."""

    field: str
    operator: RuleOperator
    value: Any

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field.strip():
            raise RuleError("RuleCondition field must be a non-empty string.")
        raw_op: Any = self.operator
        if not isinstance(raw_op, RuleOperator):
            try:
                object.__setattr__(self, "operator", RuleOperator(raw_op))
            except ValueError as exc:
                valid = [op.value for op in RuleOperator]
                raise RuleError(
                    f"Unsupported rule operator {raw_op!r}. Supported operators: {valid}."
                ) from exc
        if self.operator in (
            RuleOperator.IN,
            RuleOperator.NOT_IN,
        ) and not isinstance(self.value, (list, tuple, set, frozenset)):
            raise RuleError(
                f"Operator {self.operator.value!r} requires a collection of target values."
            )

    def evaluate(self, record: Mapping[str, Any]) -> bool:
        """Safely evaluate this predicate against a record mapping in linear bounded time."""
        if self.field not in record:
            return False
        val = record[self.field]
        if val is None:
            return False

        op = self.operator
        try:
            if op == RuleOperator.EQUALS:
                return bool(val == self.value)
            if op == RuleOperator.NOT_EQUALS:
                return bool(val != self.value)
            if op in (
                RuleOperator.GREATER_THAN,
                RuleOperator.GREATER_THAN_OR_EQUAL,
                RuleOperator.LESS_THAN,
                RuleOperator.LESS_THAN_OR_EQUAL,
            ):
                num_val = float(val)
                target_num = float(self.value)
                if op == RuleOperator.GREATER_THAN:
                    return num_val > target_num
                if op == RuleOperator.GREATER_THAN_OR_EQUAL:
                    return num_val >= target_num
                if op == RuleOperator.LESS_THAN:
                    return num_val < target_num
                return num_val <= target_num
            if op == RuleOperator.IN:
                collection = self.value if isinstance(self.value, Collection) else [self.value]
                return val in collection
            if op == RuleOperator.NOT_IN:
                collection = self.value if isinstance(self.value, Collection) else [self.value]
                return val not in collection
        except (ValueError, TypeError):
            return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize condition to a JSON-compatible dictionary."""
        val = list(self.value) if isinstance(self.value, (set, frozenset, tuple)) else self.value
        return {
            "field": self.field,
            "operator": self.operator.value,
            "value": val,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RuleCondition:
        """Deserialize and validate condition from mapping."""
        if not isinstance(data, Mapping):
            raise RuleError("RuleCondition data must be a dictionary.")
        if "field" not in data or "operator" not in data or "value" not in data:
            raise RuleError("RuleCondition requires 'field', 'operator', and 'value'.")
        return cls(
            field=str(data["field"]),
            operator=RuleOperator(data["operator"]),
            value=data["value"],
        )


@dataclass(frozen=True)
class DecisionRule:
    """A prioritized business rule executing an action if all conditions are met."""

    rule_id: str
    name: str
    action: RuleAction
    conditions: tuple[RuleCondition, ...]
    priority: int = 100
    enabled: bool = True
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, str) or not self.rule_id.strip():
            raise RuleError("DecisionRule rule_id must be a non-empty string.")
        if not isinstance(self.name, str) or not self.name.strip():
            raise RuleError("DecisionRule name must be a non-empty string.")
        raw_act: Any = self.action
        if not isinstance(raw_act, RuleAction):
            try:
                object.__setattr__(self, "action", RuleAction(raw_act))
            except ValueError as exc:
                valid = [a.value for a in RuleAction]
                raise RuleError(
                    f"Unsupported rule action {raw_act!r}. Supported actions: {valid}."
                ) from exc
        if not self.conditions:
            raise RuleError("DecisionRule must contain at least one RuleCondition.")
        if (
            not isinstance(self.priority, int)
            or isinstance(self.priority, bool)
            or self.priority < 0
        ):
            raise RuleError("DecisionRule priority must be a non-negative integer.")

    def evaluate(self, record: Mapping[str, Any]) -> bool:
        """Evaluate whether all conditions match for the given record."""
        if not self.enabled:
            return False
        return all(cond.evaluate(record) for cond in self.conditions)

    def to_dict(self) -> dict[str, Any]:
        """Serialize rule to a JSON-compatible dictionary."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "action": self.action.value,
            "priority": self.priority,
            "enabled": self.enabled,
            "reason": self.reason,
            "conditions": [cond.to_dict() for cond in self.conditions],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DecisionRule:
        """Deserialize and validate rule from mapping."""
        if not isinstance(data, Mapping):
            raise RuleError("DecisionRule data must be a dictionary.")
        req = ("rule_id", "name", "action", "conditions")
        for key in req:
            if key not in data:
                raise RuleError(f"DecisionRule missing required key {key!r}.")

        raw_conds = data["conditions"]
        if not isinstance(raw_conds, (list, tuple)):
            raise RuleError("DecisionRule conditions must be a list of condition objects.")

        conds = tuple(RuleCondition.from_dict(c) for c in raw_conds)
        return cls(
            rule_id=str(data["rule_id"]),
            name=str(data["name"]),
            action=RuleAction(data["action"]),
            priority=int(data.get("priority", 100)),
            enabled=bool(data.get("enabled", True)),
            reason=str(data.get("reason", "")),
            conditions=conds,
        )


@dataclass(frozen=True)
class RuleEvaluationResult:
    """The outcome of evaluating a RuleSet against a single record."""

    matched: bool
    matched_rule_id: str | None = None
    matched_rule_name: str | None = None
    action: RuleAction | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize evaluation result to dictionary."""
        return {
            "matched": self.matched,
            "matched_rule_id": self.matched_rule_id,
            "matched_rule_name": self.matched_rule_name,
            "action": self.action.value if self.action is not None else None,
            "reason": self.reason,
        }


class RulePrecedence(StrEnum):
    """Precedence ordering between declarative rules and model risk tiers."""

    RULES_OVERRIDE_MODEL = "rules_override_model"
    MODEL_OVERRIDES_RULES = "model_overrides_rules"


@dataclass(frozen=True)
class RuleSet:
    """An ordered, validated collection of DecisionRules evaluated by priority."""

    rules: tuple[DecisionRule, ...] = ()

    def __post_init__(self) -> None:
        # Sort rules stably by priority ascending (lower number = higher precedence)
        sorted_rules = tuple(sorted(self.rules, key=lambda r: r.priority))
        object.__setattr__(self, "rules", sorted_rules)

        # Check rule_id uniqueness
        seen_ids: set[str] = set()
        for rule in self.rules:
            if rule.rule_id in seen_ids:
                raise RuleError(f"Duplicate rule_id {rule.rule_id!r} found in RuleSet.")
            seen_ids.add(rule.rule_id)

    def evaluate_record(self, record: Mapping[str, Any]) -> RuleEvaluationResult:
        """Evaluate a single record against the rules in priority order."""
        for rule in self.rules:
            if rule.evaluate(record):
                return RuleEvaluationResult(
                    matched=True,
                    matched_rule_id=rule.rule_id,
                    matched_rule_name=rule.name,
                    action=rule.action,
                    reason=rule.reason or f"Matched rule {rule.rule_id}: {rule.name}",
                )
        return RuleEvaluationResult(matched=False, reason="No rules matched.")

    def evaluate_records(self, records: list[Mapping[str, Any]]) -> list[RuleEvaluationResult]:
        """Evaluate a sequence of records against the rules."""
        return [self.evaluate_record(rec) for rec in records]

    def to_dict(self) -> dict[str, Any]:
        """Serialize RuleSet to dictionary."""
        return {
            "rules": [rule.to_dict() for rule in self.rules],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RuleSet:
        """Deserialize and validate RuleSet from mapping."""
        if not isinstance(data, Mapping) or "rules" not in data:
            raise RuleError("RuleSet data must be a dictionary containing 'rules' list.")
        raw_rules = data["rules"]
        if not isinstance(raw_rules, (list, tuple)):
            raise RuleError("RuleSet 'rules' must be a list.")
        rules = tuple(DecisionRule.from_dict(r) for r in raw_rules)
        return cls(rules=rules)

    def to_json(self, indent: int = 2) -> str:
        """Serialize RuleSet to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> RuleSet:
        """Deserialize RuleSet from JSON string."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuleError(f"Invalid JSON for RuleSet: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def load_file(cls, path: str | Path) -> RuleSet:
        """Load and parse a RuleSet from a JSON file."""
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Rules file not found: {file_path}")
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuleError(f"Failed to read rules file {file_path}: {exc}") from exc
        return cls.from_json(content)

    def save_file(self, path: str | Path) -> None:
        """Atomically persist RuleSet to a JSON file."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp_file = dest.with_name(f".{dest.name}.tmp")
        temp_file.write_text(self.to_json() + "\n", encoding="utf-8")
        temp_file.replace(dest)
