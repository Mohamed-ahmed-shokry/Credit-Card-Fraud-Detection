"""Unit tests for multi-tenant domain models, validation, and execution context."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from fraud_detection.rules import (
    DecisionRule,
    RuleAction,
    RuleCondition,
    RuleOperator,
    RuleSet,
)
from fraud_detection.tenant import (
    RuleInheritanceMode,
    TenantContext,
    TenantError,
    TenantNotFoundError,
    TenantRegistry,
    TenantRegistryConfig,
    TenantRiskProfile,
    TenantStatus,
    TenantSuspendedError,
    TenantTier,
    TenantValidationError,
    resolve_tenant_ruleset,
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


def test_tenant_registry_initialization_and_default() -> None:
    """Validate registry startup and default tenant lookup."""
    reg = TenantRegistry()
    assert reg.count() == 1
    default_ctx = reg.get("default")
    assert default_ctx.tenant_id == "default"
    assert default_ctx.tier == TenantTier.STANDARD
    assert default_ctx.status == TenantStatus.ACTIVE

    # Fallback to default
    fallback_ctx = reg.get("non-existent-tenant")
    assert fallback_ctx.tenant_id == "default"


def test_tenant_registry_strict_isolation() -> None:
    """Validate strict isolation mode denies unknown tenants without fallback."""
    config = TenantRegistryConfig(strict_isolation=True)
    reg = TenantRegistry(config=config)
    with pytest.raises(TenantNotFoundError, match="Tenant 'unknown' not found"):
        reg.get("unknown")


def test_tenant_registry_register_and_overwrite() -> None:
    """Validate registering tenants and overwrite behavior."""
    reg = TenantRegistry()
    t1 = TenantContext(tenant_id="tenant-acme", tier=TenantTier.ENTERPRISE)
    reg.register(t1)
    assert reg.count() == 2
    assert reg.get("tenant-acme").tier == TenantTier.ENTERPRISE

    with pytest.raises(TenantValidationError, match="already registered"):
        reg.register(t1, overwrite=False)

    updated_t1 = TenantContext(tenant_id="tenant-acme", tier=TenantTier.PREMIUM)
    reg.register(updated_t1, overwrite=True)
    assert reg.get("tenant-acme").tier == TenantTier.PREMIUM


def test_tenant_registry_capacity_bounding_and_lru() -> None:
    """Validate capacity bounding with LRU eviction preserving default tenant."""
    config = TenantRegistryConfig(max_tenants=3, default_tenant_id="default")
    reg = TenantRegistry(config=config)
    # Default is already registered, count=1
    reg.register(TenantContext(tenant_id="t-1"))
    reg.register(TenantContext(tenant_id="t-2"))
    assert reg.count() == 3

    # Access t-1 so t-2 becomes LRU
    assert reg.get("t-1").tenant_id == "t-1"

    # Register t-3, should evict t-2
    reg.register(TenantContext(tenant_id="t-3"))
    assert reg.count() == 3

    # Default and t-1 should still exist, t-2 should fallback to default (or raise if strict)
    strict_config = TenantRegistryConfig(max_tenants=3, strict_isolation=True)
    strict_reg = TenantRegistry(config=strict_config)
    strict_reg.register(TenantContext(tenant_id="a"))
    strict_reg.register(TenantContext(tenant_id="b"))
    strict_reg.get("a")  # Access a
    strict_reg.register(TenantContext(tenant_id="c"))  # Should evict b

    assert strict_reg.get("a").tenant_id == "a"
    assert strict_reg.get("c").tenant_id == "c"
    with pytest.raises(TenantNotFoundError):
        strict_reg.get("b")


def test_tenant_registry_lifecycle_and_updates() -> None:
    """Validate status update, profile update, and deletion."""
    reg = TenantRegistry()
    reg.register(TenantContext(tenant_id="tenant-beta", tier=TenantTier.STANDARD))

    # Update status
    updated = reg.update_status("tenant-beta", TenantStatus.SUSPENDED)
    assert updated.status == TenantStatus.SUSPENDED
    assert reg.get("tenant-beta").status == TenantStatus.SUSPENDED

    # Update risk profile
    new_profile = TenantRiskProfile(decline_threshold=0.95, challenge_threshold=0.75)
    updated_profile = reg.update_risk_profile("tenant-beta", new_profile)
    assert updated_profile.risk_profile.decline_threshold == 0.95

    # Delete
    assert reg.delete("tenant-beta") is True
    assert reg.delete("tenant-beta") is False

    # Prevent deleting default tenant
    with pytest.raises(TenantValidationError, match="Cannot delete system default"):
        reg.delete("default")


def test_tenant_registry_multithreaded_concurrency() -> None:
    """Validate concurrent multi-threaded register and lookup without data race."""
    reg = TenantRegistry(config=TenantRegistryConfig(max_tenants=500))

    def worker(worker_id: int) -> None:
        for i in range(20):
            t_id = f"worker-{worker_id}-t-{i}"
            reg.register(TenantContext(tenant_id=t_id), overwrite=True)
            ctx = reg.get(t_id)
            assert ctx.tenant_id == t_id

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, w) for w in range(8)]
        for f in futures:
            f.result()

    assert reg.count() > 10


def test_tenant_registry_manifest_export_and_import() -> None:
    """Validate serialization roundtrip through export_manifest and import_manifest."""
    reg = TenantRegistry(config=TenantRegistryConfig(max_tenants=50))
    reg.register(
        TenantContext(
            tenant_id="tenant-x",
            tier=TenantTier.PREMIUM,
            status=TenantStatus.ACTIVE,
            risk_profile=TenantRiskProfile(
                decline_threshold=0.92,
                challenge_threshold=0.68,
                review_threshold=0.42,
                max_amount_per_tx=12000.0,
                velocity_tx_limit_1h=100,
                chargeback_rate_limit=0.012,
                rule_inheritance_mode=RuleInheritanceMode.OVERRIDE_ONLY,
                feature_overrides={"geo_velocity": 0.5},
                metadata={"region": "EMEA"},
            ),
            rate_limit_rpm=5000,
            tags=("vip", "finance"),
        )
    )

    manifest = reg.export_manifest()
    assert manifest["max_tenants"] == 50
    assert manifest["tenant_count"] == 2

    imported = TenantRegistry.import_manifest(manifest)
    assert imported.count() == 2
    tx = imported.get("tenant-x")
    assert tx.tier == TenantTier.PREMIUM
    assert tx.risk_profile.decline_threshold == 0.92
    assert tx.risk_profile.max_amount_per_tx == 12000.0
    assert tx.risk_profile.rule_inheritance_mode == RuleInheritanceMode.OVERRIDE_ONLY
    assert tx.rate_limit_rpm == 5000
    assert tx.tags == ("vip", "finance")


def test_resolve_tenant_ruleset_override_only() -> None:
    """Validate OVERRIDE_ONLY mode ignores global rules completely."""
    global_rule = DecisionRule(
        rule_id="RULE_GLOBAL_1",
        name="Global rule",
        conditions=(
            RuleCondition(field="Amount", operator=RuleOperator.GREATER_THAN, value=100.0),
        ),
        action=RuleAction.CHALLENGE,
        priority=10,
    )
    global_ruleset = RuleSet(rules=(global_rule,))

    custom_rule = DecisionRule(
        rule_id="RULE_TENANT_1",
        name="Tenant custom rule",
        conditions=(
            RuleCondition(field="Amount", operator=RuleOperator.GREATER_THAN, value=500.0),
        ),
        action=RuleAction.DENY,
        priority=5,
    )
    profile = TenantRiskProfile(
        rule_inheritance_mode=RuleInheritanceMode.OVERRIDE_ONLY,
        custom_rules=(custom_rule,),
    )
    ctx = TenantContext(tenant_id="tenant-override", risk_profile=profile)

    resolved = resolve_tenant_ruleset(ctx, global_ruleset)
    assert len(resolved) == 1
    assert resolved.rules[0].rule_id == "RULE_TENANT_1"


def test_resolve_tenant_ruleset_isolated() -> None:
    """Validate ISOLATED mode isolates tenant namespace from global rules."""
    global_rule = DecisionRule(
        rule_id="RULE_GLOBAL_1",
        name="Global rule",
        conditions=(
            RuleCondition(field="Amount", operator=RuleOperator.GREATER_THAN, value=100.0),
        ),
        action=RuleAction.CHALLENGE,
        priority=10,
    )
    global_ruleset = RuleSet(rules=(global_rule,))

    custom_rule = DecisionRule(
        rule_id="RULE_ISO_1",
        name="Isolated rule",
        conditions=(
            RuleCondition(field="Amount", operator=RuleOperator.GREATER_THAN, value=1000.0),
        ),
        action=RuleAction.DENY,
        priority=1,
    )
    profile = TenantRiskProfile(
        rule_inheritance_mode=RuleInheritanceMode.ISOLATED,
        custom_rules=(custom_rule,),
    )
    ctx = TenantContext(tenant_id="tenant-iso", risk_profile=profile)

    resolved = resolve_tenant_ruleset(ctx, global_ruleset)
    assert len(resolved) == 1
    assert resolved.rules[0].rule_id == "RULE_ISO_1"


def test_resolve_tenant_ruleset_inherit_and_override() -> None:
    """Validate INHERIT_AND_OVERRIDE mode merges rules and overrides matching rule_ids."""
    global_rule_1 = DecisionRule(
        rule_id="RULE_BASE_1",
        name="Base rule 1",
        conditions=(RuleCondition(field="Amount", operator=RuleOperator.GREATER_THAN, value=50.0),),
        action=RuleAction.CHALLENGE,
        priority=20,
    )
    global_rule_2 = DecisionRule(
        rule_id="RULE_BASE_2",
        name="Base rule 2",
        conditions=(
            RuleCondition(field="Amount", operator=RuleOperator.GREATER_THAN, value=100.0),
        ),
        action=RuleAction.DENY,
        priority=30,
    )
    global_ruleset = RuleSet(rules=(global_rule_1, global_rule_2))

    # Tenant overrides RULE_BASE_1 with a different threshold and action
    override_rule_1 = DecisionRule(
        rule_id="RULE_BASE_1",
        name="Tenant override of base rule 1",
        conditions=(RuleCondition(field="Amount", operator=RuleOperator.GREATER_THAN, value=75.0),),
        action=RuleAction.DENY,
        priority=15,
    )
    # Tenant adds a new rule
    new_rule_tenant = DecisionRule(
        rule_id="RULE_TENANT_SPECIFIC",
        name="Tenant specific rule",
        conditions=(RuleCondition(field="Amount", operator=RuleOperator.LESS_THAN, value=1.0),),
        action=RuleAction.CHALLENGE,
        priority=5,
    )

    profile = TenantRiskProfile(
        rule_inheritance_mode=RuleInheritanceMode.INHERIT_AND_OVERRIDE,
        custom_rules=(override_rule_1, new_rule_tenant),
    )
    ctx = TenantContext(tenant_id="tenant-hybrid", risk_profile=profile)

    resolved = resolve_tenant_ruleset(ctx, global_ruleset)
    # Should contain 3 rules: RULE_TENANT_SPECIFIC (p=5), override RULE_BASE_1 (p=15),
    # and RULE_BASE_2 (p=30)
    assert len(resolved) == 3
    rule_ids = [r.rule_id for r in resolved.rules]
    assert rule_ids == ["RULE_TENANT_SPECIFIC", "RULE_BASE_1", "RULE_BASE_2"]

    # Check that RULE_BASE_1 has the tenant's overridden action
    overridden_rule = next(r for r in resolved.rules if r.rule_id == "RULE_BASE_1")
    assert overridden_rule.action == RuleAction.DENY
    assert overridden_rule.priority == 15
    assert overridden_rule.name == "Tenant override of base rule 1"
