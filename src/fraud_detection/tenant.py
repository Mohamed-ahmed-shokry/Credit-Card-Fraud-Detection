"""Multi-tenant policy partitioning, dynamic rule federation, and account isolation.

Provides tenant-scoped risk profiles with customizable decision thresholds,
declarative rule inheritance modes, thread-safe tenant registry with LRU capacity
bounding, and isolated runtime context for multi-organization fraud prevention.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from fraud_detection.rules import DecisionRule, RuleSet

__all__ = [
    "RuleInheritanceMode",
    "TenantContext",
    "TenantError",
    "TenantNotFoundError",
    "TenantRegistry",
    "TenantRegistryConfig",
    "TenantRiskProfile",
    "TenantStatus",
    "TenantSuspendedError",
    "TenantTier",
    "TenantValidationError",
    "resolve_tenant_ruleset",
]

_TENANT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]{1,64}$")


class TenantError(Exception):
    """Base exception for all tenant-related failures."""


class TenantNotFoundError(TenantError):
    """Raised when a requested tenant ID does not exist in the registry."""


class TenantValidationError(TenantError):
    """Raised when a tenant configuration or profile fails validation."""


class TenantSuspendedError(TenantError):
    """Raised when an operation is attempted on an inactive/suspended tenant."""


class TenantTier(StrEnum):
    """Service and operational tier for a tenant."""

    FREE = "free"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class RuleInheritanceMode(StrEnum):
    """Rule resolution and federation inheritance mode.

    OVERRIDE_ONLY: Evaluates only tenant custom rules; global rules are ignored.
    INHERIT_AND_OVERRIDE: Evaluates tenant rules first; falls back to global rules
        if tenant rules do not match or when combining federated rules.
    ISOLATED: Strictly evaluates tenant custom rules with isolated rule namespaces.
    """

    OVERRIDE_ONLY = "override_only"
    INHERIT_AND_OVERRIDE = "inherit_and_override"
    ISOLATED = "isolated"


class TenantStatus(StrEnum):
    """Lifecycle status of a tenant account."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEPRECATED = "deprecated"


@dataclass(frozen=True)
class TenantRiskProfile:
    """Configurable risk and threshold parameters tailored to a specific tenant."""

    decline_threshold: float = 0.85
    challenge_threshold: float = 0.60
    review_threshold: float = 0.35
    max_amount_per_tx: float | None = None
    velocity_tx_limit_1h: int | None = None
    chargeback_rate_limit: float = 0.015
    rule_inheritance_mode: RuleInheritanceMode = RuleInheritanceMode.INHERIT_AND_OVERRIDE
    custom_rules: tuple[DecisionRule, ...] = field(default_factory=tuple)
    feature_overrides: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate threshold bounds and order constraints."""
        for name, val in [
            ("decline_threshold", self.decline_threshold),
            ("challenge_threshold", self.challenge_threshold),
            ("review_threshold", self.review_threshold),
            ("chargeback_rate_limit", self.chargeback_rate_limit),
        ]:
            if not (0.0 <= val <= 1.0):
                raise TenantValidationError(f"{name} must be within [0.0, 1.0], got {val}")

        if not (self.review_threshold <= self.challenge_threshold <= self.decline_threshold):
            raise TenantValidationError(
                f"Threshold order constraint violated: review ({self.review_threshold}) "
                f"<= challenge ({self.challenge_threshold}) <= decline ({self.decline_threshold})"
            )

        if self.max_amount_per_tx is not None and self.max_amount_per_tx <= 0.0:
            raise TenantValidationError(
                f"max_amount_per_tx must be positive, got {self.max_amount_per_tx}"
            )

        if self.velocity_tx_limit_1h is not None and self.velocity_tx_limit_1h <= 0:
            raise TenantValidationError(
                f"velocity_tx_limit_1h must be positive, got {self.velocity_tx_limit_1h}"
            )


@dataclass(frozen=True)
class TenantContext:
    """Runtime execution context representing the active tenant."""

    tenant_id: str
    tier: TenantTier = TenantTier.STANDARD
    status: TenantStatus = TenantStatus.ACTIVE
    risk_profile: TenantRiskProfile = field(default_factory=TenantRiskProfile)
    rate_limit_rpm: int = 1000
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate tenant identifier and rate limit."""
        if not self.tenant_id or not _TENANT_ID_PATTERN.match(self.tenant_id):
            raise TenantValidationError(
                f"Invalid tenant_id '{self.tenant_id}': must be 1-64 alphanumeric, dash, "
                "dot, or underscore characters"
            )
        if self.rate_limit_rpm <= 0:
            raise TenantValidationError(
                f"rate_limit_rpm must be positive, got {self.rate_limit_rpm}"
            )

    @property
    def is_active(self) -> bool:
        """Return True if the tenant is active and permitted to process transactions."""
        return self.status == TenantStatus.ACTIVE

    def check_active(self) -> None:
        """Raise TenantSuspendedError if the tenant is not active."""
        if not self.is_active:
            raise TenantSuspendedError(
                f"Tenant '{self.tenant_id}' is {self.status.value} and cannot process requests"
            )


@dataclass(frozen=True)
class TenantRegistryConfig:
    """Configuration settings for the thread-safe tenant registry."""

    max_tenants: int = 10_000
    default_tenant_id: str = "default"
    allow_fallback_to_default: bool = True
    strict_isolation: bool = False

    def __post_init__(self) -> None:
        """Validate registry parameters."""
        if self.max_tenants <= 0:
            raise TenantValidationError(f"max_tenants must be positive, got {self.max_tenants}")
        if not self.default_tenant_id or not _TENANT_ID_PATTERN.match(self.default_tenant_id):
            raise TenantValidationError(f"Invalid default_tenant_id '{self.default_tenant_id}'")


class TenantRegistry:
    """Thread-safe, capacity-bounded registry storing active tenant contexts.

    Supports dynamic tenant onboarding, lifecycle updates, LRU capacity bounding,
    and fallback resolution.
    """

    def __init__(self, config: TenantRegistryConfig | None = None) -> None:
        self._config = config or TenantRegistryConfig()
        self._lock = threading.RLock()
        self._tenants: OrderedDict[str, TenantContext] = OrderedDict()

        # Seed the default tenant
        default_ctx = TenantContext(
            tenant_id=self._config.default_tenant_id,
            tier=TenantTier.STANDARD,
            status=TenantStatus.ACTIVE,
            risk_profile=TenantRiskProfile(),
        )
        self._tenants[self._config.default_tenant_id] = default_ctx

    @property
    def config(self) -> TenantRegistryConfig:
        """Return the immutable registry configuration."""
        return self._config

    def register(self, context: TenantContext, overwrite: bool = False) -> None:
        """Register a new tenant context with thread-safe capacity bounding.

        Raises TenantValidationError if tenant already exists and overwrite is False.
        """
        with self._lock:
            tenant_id = context.tenant_id
            if tenant_id in self._tenants and not overwrite:
                raise TenantValidationError(
                    f"Tenant '{tenant_id}' is already registered. Set overwrite=True to update."
                )

            # Capacity bounding: if at capacity, evict least recently used non-default tenant
            if tenant_id not in self._tenants and len(self._tenants) >= self._config.max_tenants:
                evicted_key = None
                for key in self._tenants:
                    if key != self._config.default_tenant_id:
                        evicted_key = key
                        break
                if evicted_key:
                    del self._tenants[evicted_key]

            self._tenants[tenant_id] = context
            self._tenants.move_to_end(tenant_id)

    def get(self, tenant_id: str) -> TenantContext:
        """Retrieve a tenant context by ID, updating LRU order.

        Falls back to default tenant if allowed and not in strict_isolation mode.
        Raises TenantNotFoundError if tenant is not found and fallback is disabled/unavailable.
        """
        with self._lock:
            if tenant_id in self._tenants:
                self._tenants.move_to_end(tenant_id)
                return self._tenants[tenant_id]

            if self._config.allow_fallback_to_default and not self._config.strict_isolation:
                return self._tenants[self._config.default_tenant_id]

            raise TenantNotFoundError(f"Tenant '{tenant_id}' not found in registry")

    def update_status(self, tenant_id: str, status: TenantStatus) -> TenantContext:
        """Update the lifecycle status of a registered tenant."""
        with self._lock:
            if tenant_id not in self._tenants:
                raise TenantNotFoundError(f"Tenant '{tenant_id}' not found for status update")

            existing = self._tenants[tenant_id]
            updated = TenantContext(
                tenant_id=existing.tenant_id,
                tier=existing.tier,
                status=status,
                risk_profile=existing.risk_profile,
                rate_limit_rpm=existing.rate_limit_rpm,
                created_at=existing.created_at,
                tags=existing.tags,
            )
            self._tenants[tenant_id] = updated
            self._tenants.move_to_end(tenant_id)
            return updated

    def update_risk_profile(self, tenant_id: str, profile: TenantRiskProfile) -> TenantContext:
        """Update the risk profile of a registered tenant."""
        with self._lock:
            if tenant_id not in self._tenants:
                raise TenantNotFoundError(f"Tenant '{tenant_id}' not found for profile update")

            existing = self._tenants[tenant_id]
            updated = TenantContext(
                tenant_id=existing.tenant_id,
                tier=existing.tier,
                status=existing.status,
                risk_profile=profile,
                rate_limit_rpm=existing.rate_limit_rpm,
                created_at=existing.created_at,
                tags=existing.tags,
            )
            self._tenants[tenant_id] = updated
            self._tenants.move_to_end(tenant_id)
            return updated

    def delete(self, tenant_id: str) -> bool:
        """Delete a tenant context from the registry.

        The default tenant cannot be deleted.
        """
        with self._lock:
            if tenant_id == self._config.default_tenant_id:
                raise TenantValidationError(f"Cannot delete system default tenant '{tenant_id}'")
            if tenant_id in self._tenants:
                del self._tenants[tenant_id]
                return True
            return False

    def list_tenants(self, limit: int | None = None) -> list[TenantContext]:
        """Return a snapshot list of registered tenant contexts."""
        with self._lock:
            tenants = list(self._tenants.values())
            if limit is not None and limit >= 0:
                return tenants[:limit]
            return tenants

    def count(self) -> int:
        """Return total count of registered tenants."""
        with self._lock:
            return len(self._tenants)

    def export_manifest(self) -> dict[str, Any]:
        """Serialize registered tenant profiles into a portable dictionary manifest."""
        with self._lock:
            tenants_data = [
                {
                    "tenant_id": t.tenant_id,
                    "tier": t.tier.value,
                    "status": t.status.value,
                    "rate_limit_rpm": t.rate_limit_rpm,
                    "created_at": t.created_at,
                    "tags": list(t.tags),
                    "risk_profile": {
                        "decline_threshold": t.risk_profile.decline_threshold,
                        "challenge_threshold": t.risk_profile.challenge_threshold,
                        "review_threshold": t.risk_profile.review_threshold,
                        "max_amount_per_tx": t.risk_profile.max_amount_per_tx,
                        "velocity_tx_limit_1h": t.risk_profile.velocity_tx_limit_1h,
                        "chargeback_rate_limit": t.risk_profile.chargeback_rate_limit,
                        "rule_inheritance_mode": t.risk_profile.rule_inheritance_mode.value,
                        "feature_overrides": dict(t.risk_profile.feature_overrides),
                        "metadata": dict(t.risk_profile.metadata),
                    },
                }
                for t in self._tenants.values()
            ]
            return {
                "max_tenants": self._config.max_tenants,
                "default_tenant_id": self._config.default_tenant_id,
                "allow_fallback_to_default": self._config.allow_fallback_to_default,
                "strict_isolation": self._config.strict_isolation,
                "tenant_count": len(tenants_data),
                "tenants": tenants_data,
            }

    @classmethod
    def import_manifest(cls, data: dict[str, Any]) -> TenantRegistry:
        """Create and populate a TenantRegistry from a serialized manifest."""
        config = TenantRegistryConfig(
            max_tenants=int(data.get("max_tenants", 10_000)),
            default_tenant_id=str(data.get("default_tenant_id", "default")),
            allow_fallback_to_default=bool(data.get("allow_fallback_to_default", True)),
            strict_isolation=bool(data.get("strict_isolation", False)),
        )
        registry = cls(config=config)
        for t_dict in data.get("tenants", []):
            rp_dict = t_dict.get("risk_profile", {})
            risk_profile = TenantRiskProfile(
                decline_threshold=float(rp_dict.get("decline_threshold", 0.85)),
                challenge_threshold=float(rp_dict.get("challenge_threshold", 0.60)),
                review_threshold=float(rp_dict.get("review_threshold", 0.35)),
                max_amount_per_tx=rp_dict.get("max_amount_per_tx"),
                velocity_tx_limit_1h=rp_dict.get("velocity_tx_limit_1h"),
                chargeback_rate_limit=float(rp_dict.get("chargeback_rate_limit", 0.015)),
                rule_inheritance_mode=RuleInheritanceMode(
                    rp_dict.get("rule_inheritance_mode", "inherit_and_override")
                ),
                feature_overrides=dict(rp_dict.get("feature_overrides", {})),
                metadata=dict(rp_dict.get("metadata", {})),
            )
            context = TenantContext(
                tenant_id=str(t_dict["tenant_id"]),
                tier=TenantTier(t_dict.get("tier", "standard")),
                status=TenantStatus(t_dict.get("status", "active")),
                risk_profile=risk_profile,
                rate_limit_rpm=int(t_dict.get("rate_limit_rpm", 1000)),
                created_at=str(t_dict.get("created_at", "")),
                tags=tuple(t_dict.get("tags", ())),
            )
            registry.register(context, overwrite=True)
        return registry


def resolve_tenant_ruleset(
    tenant_context: TenantContext, global_ruleset: RuleSet | None = None
) -> RuleSet:
    """Resolve and federate a RuleSet for a specific tenant based on its inheritance mode.

    Modes:
      - OVERRIDE_ONLY: Ignores global rules; uses only the tenant's custom rules.
      - ISOLATED: Strictly evaluates tenant custom rules with isolated rule namespaces.
      - INHERIT_AND_OVERRIDE: Merges global rules with tenant custom rules, replacing
        global rules when a tenant rule matches an existing rule_id.
    """
    mode = tenant_context.risk_profile.rule_inheritance_mode
    custom_rules = tenant_context.risk_profile.custom_rules

    if mode in (RuleInheritanceMode.OVERRIDE_ONLY, RuleInheritanceMode.ISOLATED):
        return RuleSet(rules=custom_rules)

    # INHERIT_AND_OVERRIDE
    if not global_ruleset or not global_ruleset.rules:
        return RuleSet(rules=custom_rules)

    tenant_rule_map = {r.rule_id: r for r in custom_rules}
    merged_rules: list[DecisionRule] = []

    # Add global rules that are not overridden by tenant
    for g_rule in global_ruleset.rules:
        if g_rule.rule_id in tenant_rule_map:
            merged_rules.append(tenant_rule_map.pop(g_rule.rule_id))
        else:
            merged_rules.append(g_rule)

    # Append any remaining new tenant rules
    merged_rules.extend(tenant_rule_map.values())

    return RuleSet(rules=tuple(merged_rules))
