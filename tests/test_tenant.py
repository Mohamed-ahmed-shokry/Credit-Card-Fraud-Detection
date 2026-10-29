"""Unit tests for multi-tenant domain models, validation, and execution context."""

from __future__ import annotations

import pytest

from fraud_detection.tenant import (
    RuleInheritanceMode,
    TenantContext,
    TenantError,
    TenantNotFoundError,
    TenantRiskProfile,
    TenantStatus,
    TenantSuspendedError,
    TenantTier,
    TenantValidationError,
)


def test_tenant_tier_and_status_enums() -> None:
    """Validate string enum values."""
    assert TenantTier.FREE.value == "free"
    assert TenantTier.STANDARD.value == "standard"
    assert TenantTier.PREMIUM.value == "premium"
    assert TenantTier.ENTERPRISE.value == "enterprise"

    assert TenantStatus.ACTIVE.value == "active"
    assert TenantStatus.SUSPENDED.value == "suspended"
    assert TenantStatus.DEPRECATED.value == "deprecated"

    assert RuleInheritanceMode.OVERRIDE_ONLY.value == "override_only"
    assert RuleInheritanceMode.INHERIT_AND_OVERRIDE.value == "inherit_and_override"
    assert RuleInheritanceMode.ISOLATED.value == "isolated"


def test_tenant_risk_profile_defaults_and_custom() -> None:
    """Validate default and custom risk profile construction."""
    profile = TenantRiskProfile()
    assert profile.decline_threshold == 0.85
    assert profile.challenge_threshold == 0.60
    assert profile.review_threshold == 0.35
    assert profile.max_amount_per_tx is None
    assert profile.velocity_tx_limit_1h is None
    assert profile.chargeback_rate_limit == 0.015
    assert profile.rule_inheritance_mode == RuleInheritanceMode.INHERIT_AND_OVERRIDE

    custom = TenantRiskProfile(
        decline_threshold=0.90,
        challenge_threshold=0.70,
        review_threshold=0.50,
        max_amount_per_tx=5000.0,
        velocity_tx_limit_1h=50,
        chargeback_rate_limit=0.01,
        rule_inheritance_mode=RuleInheritanceMode.OVERRIDE_ONLY,
    )
    assert custom.max_amount_per_tx == 5000.0
    assert custom.velocity_tx_limit_1h == 50


def test_tenant_risk_profile_validation_errors() -> None:
    """Validate threshold bounds and order constraints."""
    with pytest.raises(TenantValidationError, match="decline_threshold must be within"):
        TenantRiskProfile(decline_threshold=1.5)

    with pytest.raises(TenantValidationError, match="review_threshold must be within"):
        TenantRiskProfile(review_threshold=-0.1)

    with pytest.raises(TenantValidationError, match="Threshold order constraint violated"):
        TenantRiskProfile(
            review_threshold=0.70,
            challenge_threshold=0.50,
            decline_threshold=0.80,
        )

    with pytest.raises(TenantValidationError, match="max_amount_per_tx must be positive"):
        TenantRiskProfile(max_amount_per_tx=0.0)

    with pytest.raises(TenantValidationError, match="velocity_tx_limit_1h must be positive"):
        TenantRiskProfile(velocity_tx_limit_1h=-5)


def test_tenant_context_creation_and_activity_check() -> None:
    """Validate tenant context lifecycle and status checks."""
    ctx = TenantContext(tenant_id="tenant-alpha-01", tier=TenantTier.PREMIUM)
    assert ctx.tenant_id == "tenant-alpha-01"
    assert ctx.tier == TenantTier.PREMIUM
    assert ctx.status == TenantStatus.ACTIVE
    assert ctx.is_active is True
    ctx.check_active()

    suspended = TenantContext(
        tenant_id="tenant-suspended",
        status=TenantStatus.SUSPENDED,
    )
    assert suspended.is_active is False
    with pytest.raises(TenantSuspendedError, match="is suspended and cannot process"):
        suspended.check_active()


def test_tenant_context_invalid_id_and_rate_limit() -> None:
    """Validate tenant context identifiers and rate limit bounds."""
    with pytest.raises(TenantValidationError, match="Invalid tenant_id"):
        TenantContext(tenant_id="")

    with pytest.raises(TenantValidationError, match="Invalid tenant_id"):
        TenantContext(tenant_id="invalid space id")

    with pytest.raises(TenantValidationError, match="Invalid tenant_id"):
        TenantContext(tenant_id="a" * 65)

    with pytest.raises(TenantValidationError, match="rate_limit_rpm must be positive"):
        TenantContext(tenant_id="valid-id", rate_limit_rpm=0)


def test_tenant_error_hierarchy() -> None:
    """Validate tenant exception inheritance."""
    assert issubclass(TenantNotFoundError, TenantError)
    assert issubclass(TenantValidationError, TenantError)
    assert issubclass(TenantSuspendedError, TenantError)
