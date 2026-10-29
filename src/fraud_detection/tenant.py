"""Multi-tenant policy partitioning, dynamic rule federation, and account isolation.

Provides tenant-scoped risk profiles with customizable decision thresholds,
declarative rule inheritance modes, thread-safe tenant registry with LRU capacity
bounding, and isolated runtime context for multi-organization fraud prevention.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from fraud_detection.rules import DecisionRule

__all__ = [
    "RuleInheritanceMode",
    "TenantContext",
    "TenantError",
    "TenantNotFoundError",
    "TenantRiskProfile",
    "TenantStatus",
    "TenantSuspendedError",
    "TenantTier",
    "TenantValidationError",
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
