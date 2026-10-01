from __future__ import annotations

from pathlib import Path

import pytest

from fraud_detection.rules import (
    DecisionRule,
    RuleAction,
    RuleCondition,
    RuleError,
    RuleOperator,
    RulePrecedence,
    RuleSet,
)


def test_rule_operator_and_action_enums() -> None:
    assert RuleOperator.EQUALS.value == "=="
    assert RuleOperator.NOT_EQUALS.value == "!="
    assert RuleOperator.GREATER_THAN.value == ">"
    assert RuleOperator.GREATER_THAN_OR_EQUAL.value == ">="
    assert RuleOperator.LESS_THAN.value == "<"
    assert RuleOperator.LESS_THAN_OR_EQUAL.value == "<="
    assert RuleOperator.IN.value == "in"
    assert RuleOperator.NOT_IN.value == "not_in"

    assert RuleAction.ALLOW.value == "ALLOW"
    assert RuleAction.CHALLENGE.value == "CHALLENGE"
    assert RuleAction.DENY.value == "DENY"

    assert RulePrecedence.RULES_OVERRIDE_MODEL.value == "rules_override_model"
    assert RulePrecedence.MODEL_OVERRIDES_RULES.value == "model_overrides_rules"


def test_rule_condition_evaluation() -> None:
    # Numeric comparisons
    cond_gt = RuleCondition("Amount", RuleOperator.GREATER_THAN, 1000.0)
    assert cond_gt.evaluate({"Amount": 1500.0}) is True
    assert cond_gt.evaluate({"Amount": 1000.0}) is False
    assert cond_gt.evaluate({"Amount": 500.0}) is False
    assert cond_gt.evaluate({"Amount": "2000.5"}) is True  # numeric string coercion
    assert cond_gt.evaluate({"Amount": "not_a_number"}) is False
    assert cond_gt.evaluate({"Other": 100.0}) is False
    assert cond_gt.evaluate({"Amount": None}) is False

    cond_gte = RuleCondition("Amount", RuleOperator.GREATER_THAN_OR_EQUAL, 1000.0)
    assert cond_gte.evaluate({"Amount": 1000.0}) is True
    assert cond_gte.evaluate({"Amount": 999.9}) is False

    cond_lt = RuleCondition("Amount", RuleOperator.LESS_THAN, 10.0)
    assert cond_lt.evaluate({"Amount": 5.0}) is True
    assert cond_lt.evaluate({"Amount": 10.0}) is False

    cond_lte = RuleCondition("Amount", RuleOperator.LESS_THAN_OR_EQUAL, 10.0)
    assert cond_lte.evaluate({"Amount": 10.0}) is True
    assert cond_lte.evaluate({"Amount": 10.1}) is False

    # Equality and inequality
    cond_eq = RuleCondition("currency", RuleOperator.EQUALS, "USD")
    assert cond_eq.evaluate({"currency": "USD"}) is True
    assert cond_eq.evaluate({"currency": "EUR"}) is False

    cond_neq = RuleCondition("currency", RuleOperator.NOT_EQUALS, "USD")
    assert cond_neq.evaluate({"currency": "EUR"}) is True
    assert cond_neq.evaluate({"currency": "USD"}) is False

    # In and Not In
    cond_in = RuleCondition("channel", RuleOperator.IN, ["web", "mobile"])
    assert cond_in.evaluate({"channel": "web"}) is True
    assert cond_in.evaluate({"channel": "pos"}) is False

    cond_not_in = RuleCondition("country", RuleOperator.NOT_IN, ["US", "CA"])
    assert cond_not_in.evaluate({"country": "GB"}) is True
    assert cond_not_in.evaluate({"country": "US"}) is False


def test_rule_condition_validation_and_serialization() -> None:
    # Invalid field
    with pytest.raises(RuleError, match="field must be a non-empty string"):
        RuleCondition("", RuleOperator.EQUALS, 10)

    # Invalid operator
    with pytest.raises(RuleError, match="Unsupported rule operator"):
        RuleCondition("Amount", "invalid_op", 10)  # type: ignore[arg-type]

    # In operator requires collection
    with pytest.raises(RuleError, match="requires a collection"):
        RuleCondition("channel", RuleOperator.IN, "not_a_collection")

    # Serialization roundtrip
    cond = RuleCondition("V1", RuleOperator.GREATER_THAN, 2.5)
    d = cond.to_dict()
    assert d == {"field": "V1", "operator": ">", "value": 2.5}
    cond_deser = RuleCondition.from_dict(d)
    assert cond_deser == cond

    # from_dict invalid
    with pytest.raises(RuleError, match="must be a dictionary"):
        RuleCondition.from_dict(["not", "dict"])  # type: ignore[arg-type]
    with pytest.raises(RuleError, match="requires 'field', 'operator', and 'value'"):
        RuleCondition.from_dict({"field": "Amount"})


def test_decision_rule_evaluation() -> None:
    cond1 = RuleCondition("Amount", RuleOperator.GREATER_THAN, 5000.0)
    cond2 = RuleCondition("country", RuleOperator.EQUALS, "NG")
    rule = DecisionRule(
        rule_id="R001",
        name="High Value High Risk Country",
        action=RuleAction.DENY,
        conditions=(cond1, cond2),
        priority=10,
        reason="Exceeds $5000 in high risk country",
    )

    # Both conditions match
    assert rule.evaluate({"Amount": 6000.0, "country": "NG"}) is True
    # Only one condition matches
    assert rule.evaluate({"Amount": 6000.0, "country": "US"}) is False
    assert rule.evaluate({"Amount": 1000.0, "country": "NG"}) is False

    # Disabled rule
    disabled_rule = DecisionRule(
        rule_id="R002",
        name="Disabled Rule",
        action=RuleAction.CHALLENGE,
        conditions=(cond1,),
        enabled=False,
    )
    assert disabled_rule.evaluate({"Amount": 10000.0}) is False


def test_decision_rule_validation_and_serialization() -> None:
    cond = RuleCondition("Amount", RuleOperator.GREATER_THAN, 100.0)

    # Invalid rule_id / name
    with pytest.raises(RuleError, match="rule_id must be a non-empty string"):
        DecisionRule("", "name", RuleAction.DENY, (cond,))
    with pytest.raises(RuleError, match="name must be a non-empty string"):
        DecisionRule("R1", "", RuleAction.DENY, (cond,))

    # Invalid action
    with pytest.raises(RuleError, match="Unsupported rule action"):
        DecisionRule("R1", "name", "INVALID_ACTION", (cond,))  # type: ignore[arg-type]

    # Empty conditions
    with pytest.raises(RuleError, match="at least one RuleCondition"):
        DecisionRule("R1", "name", RuleAction.ALLOW, ())

    # Invalid priority
    with pytest.raises(RuleError, match="priority must be a non-negative integer"):
        DecisionRule("R1", "name", RuleAction.ALLOW, (cond,), priority=-1)

    # Serialization roundtrip
    rule = DecisionRule(
        rule_id="R10",
        name="VIP Allow",
        action=RuleAction.ALLOW,
        conditions=(cond,),
        priority=5,
        enabled=True,
        reason="VIP customer fast-track",
    )
    d = rule.to_dict()
    assert d["rule_id"] == "R10"
    assert d["action"] == "ALLOW"
    assert d["priority"] == 5

    deser = DecisionRule.from_dict(d)
    assert deser == rule

    with pytest.raises(RuleError, match="missing required key"):
        DecisionRule.from_dict({"rule_id": "R1"})


def test_ruleset_priority_ordering_and_evaluation() -> None:
    # Rule 1: priority 50 -> ALLOW if Amount < 1.0
    r1 = DecisionRule(
        rule_id="R_LOW",
        name="Micro-transaction Allow",
        action=RuleAction.ALLOW,
        conditions=(RuleCondition("Amount", RuleOperator.LESS_THAN, 1.0),),
        priority=50,
    )
    # Rule 2: priority 10 -> DENY if Amount > 10000.0
    r2 = DecisionRule(
        rule_id="R_HIGH",
        name="Massive Transaction Deny",
        action=RuleAction.DENY,
        conditions=(RuleCondition("Amount", RuleOperator.GREATER_THAN, 10000.0),),
        priority=10,
    )
    # Rule 3: priority 20 -> CHALLENGE if channel == "crypto"
    r3 = DecisionRule(
        rule_id="R_CRYPTO",
        name="Crypto Channel Challenge",
        action=RuleAction.CHALLENGE,
        conditions=(RuleCondition("channel", RuleOperator.EQUALS, "crypto"),),
        priority=20,
    )

    ruleset = RuleSet((r1, r2, r3))
    # Check that rules were sorted by priority: r2 (10), r3 (20), r1 (50)
    assert [r.rule_id for r in ruleset.rules] == ["R_HIGH", "R_CRYPTO", "R_LOW"]

    # Match R_HIGH
    res_high = ruleset.evaluate_record({"Amount": 15000.0, "channel": "crypto"})
    assert res_high.matched is True
    assert res_high.matched_rule_id == "R_HIGH"
    assert res_high.action == RuleAction.DENY

    # Match R_CRYPTO (Amount 500 does not match R_HIGH)
    res_crypto = ruleset.evaluate_record({"Amount": 500.0, "channel": "crypto"})
    assert res_crypto.matched is True
    assert res_crypto.matched_rule_id == "R_CRYPTO"
    assert res_crypto.action == RuleAction.CHALLENGE

    # Match R_LOW
    res_low = ruleset.evaluate_record({"Amount": 0.5, "channel": "card"})
    assert res_low.matched is True
    assert res_low.matched_rule_id == "R_LOW"
    assert res_low.action == RuleAction.ALLOW

    # No match
    res_none = ruleset.evaluate_record({"Amount": 100.0, "channel": "card"})
    assert res_none.matched is False
    assert res_none.action is None
    assert res_none.matched_rule_id is None

    # Batch evaluate
    batch_res = ruleset.evaluate_records(
        [
            {"Amount": 15000.0},
            {"Amount": 50.0},
        ]
    )
    assert len(batch_res) == 2
    assert batch_res[0].matched is True
    assert batch_res[1].matched is False


def test_ruleset_duplicate_ids_rejected() -> None:
    cond = RuleCondition("Amount", RuleOperator.GREATER_THAN, 10.0)
    r1 = DecisionRule("DUP", "Name 1", RuleAction.DENY, (cond,))
    r2 = DecisionRule("DUP", "Name 2", RuleAction.ALLOW, (cond,))
    with pytest.raises(RuleError, match="Duplicate rule_id 'DUP'"):
        RuleSet((r1, r2))


def test_ruleset_io_and_persistence(tmp_path: Path) -> None:
    cond = RuleCondition("V1", RuleOperator.LESS_THAN, -3.0)
    rule = DecisionRule(
        rule_id="R_ANOMALY",
        name="Severe V1 Anomaly",
        action=RuleAction.DENY,
        conditions=(cond,),
        priority=1,
        reason="V1 score far below normal range",
    )
    ruleset = RuleSet((rule,))

    # JSON serialization
    json_str = ruleset.to_json()
    assert "R_ANOMALY" in json_str
    deser = RuleSet.from_json(json_str)
    assert deser == ruleset

    # Save and load file
    file_path = tmp_path / "rules" / "fraud_rules.json"
    ruleset.save_file(file_path)
    assert file_path.is_file()

    loaded = RuleSet.load_file(file_path)
    assert loaded == ruleset

    # Missing file
    with pytest.raises(FileNotFoundError):
        RuleSet.load_file(tmp_path / "nonexistent.json")

    # Invalid JSON
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{corrupt json", encoding="utf-8")
    with pytest.raises(RuleError, match="Invalid JSON"):
        RuleSet.load_file(bad_file)
