"""Synthetic Identity Fraud (SIF) Detection, PII Inconsistency & Identity Verification Engine.

Provides domain models, SSA historical allocation verification, Death Master File screening,
PII element fragmentation, synthetic velocity clustering, authorized user abuse detection,
and composite risk decisioning with automated case/challenge escalation.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


class SifRiskLevel(StrEnum):
    """Categorical risk tiers for synthetic identity assessments."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SifAction(StrEnum):
    """Policy action decisions determined by synthetic identity evaluations."""

    APPROVE = "approve"
    STEP_UP_IDENTITY_PROOFING = "step_up_identity_proofing"
    MANUAL_REVIEW = "manual_review"
    DECLINE_BLOCK = "decline_block"


class SifAnomalyType(StrEnum):
    """Taxonomy of synthetic identity anomalies and red flags."""

    SSN_DOB_MISMATCH = "ssn_dob_mismatch"
    DECEASED_SSN_MATCH = "deceased_ssn_match"
    INVALID_SSN_STRUCTURE = "invalid_ssn_structure"
    UNISSUED_SSN_AREA = "unissued_ssn_area"
    SYNTHETIC_VELOCITY_BURST = "synthetic_velocity_burst"
    PII_ELEMENT_FRAGMENTATION = "pii_element_fragmentation"
    AUTHORIZED_USER_ABUSE = "authorized_user_abuse"
    HIGH_RISK_ADDRESS_DROP = "high_risk_address_drop"
    BURNER_PHONE_DETECTED = "burner_phone_detected"


class SsnValidationStatus(StrEnum):
    """Validation outcomes for SSN issuance and consistency audits."""

    VALID = "valid"
    INVALID_FORMAT = "invalid_format"
    RESERVED_OR_SPECIAL = "reserved_or_special"
    UNISSUED_AREA = "unissued_area"
    HIGH_GROUP_EXCEEDED = "high_group_exceeded"
    ISSUED_POST_RANDOMIZATION = "issued_post_randomization"
    DOB_INCONSISTENT = "dob_inconsistent"
    DECEASED = "deceased"


@dataclass(frozen=True)
class AuthorizedUserRecord:
    """Authorized user added or requested on an applicant credit profile."""

    user_id: str
    name: str
    relationship: str = "other"
    added_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    ssn_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "relationship": self.relationship,
            "added_at": self.added_at.isoformat(),
            "ssn_hash": self.ssn_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuthorizedUserRecord:
        added_raw = data.get("added_at")
        added_at = (
            datetime.fromisoformat(added_raw)
            if isinstance(added_raw, str)
            else (added_raw if isinstance(added_raw, datetime) else datetime.now(UTC))
        )
        return cls(
            user_id=str(data.get("user_id", "")),
            name=str(data.get("name", "")),
            relationship=str(data.get("relationship", "other")),
            added_at=added_at,
            ssn_hash=str(data.get("ssn_hash", "")),
        )


@dataclass
class IdentityProfile:
    """Credit applicant or customer identity attributes evaluated for synthetic fraud."""

    applicant_id: str
    ssn: str
    first_name: str
    last_name: str
    dob: str  # YYYY-MM-DD
    address: str
    phone: str
    email: str
    credit_line_requested: float = 5000.0
    authorized_users: list[AuthorizedUserRecord] = field(default_factory=list)
    tenant_id: str = "default"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def ssn_clean(self) -> str:
        """Strip hyphens, spaces, and non-digits from SSN."""
        return re.sub(r"\D", "", self.ssn)

    def ssn_masked(self) -> str:
        """Return masked SSN format (XXX-XX-1234)."""
        clean = self.ssn_clean()
        if len(clean) == 9:
            return f"XXX-XX-{clean[-4:]}"
        return "XXX-XX-XXXX"

    def ssn_hash(self) -> str:
        """Return deterministic SHA-256 hash of cleaned SSN."""
        clean = self.ssn_clean()
        return hashlib.sha256(clean.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicant_id": self.applicant_id,
            "ssn": self.ssn_masked(),
            "ssn_hash": self.ssn_hash(),
            "first_name": self.first_name,
            "last_name": self.last_name,
            "dob": self.dob,
            "address": self.address,
            "phone": self.phone,
            "email": self.email,
            "credit_line_requested": self.credit_line_requested,
            "authorized_users": [u.to_dict() for u in self.authorized_users],
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IdentityProfile:
        created_raw = data.get("created_at")
        created_at = (
            datetime.fromisoformat(created_raw)
            if isinstance(created_raw, str)
            else (created_raw if isinstance(created_raw, datetime) else datetime.now(UTC))
        )
        raw_users = data.get("authorized_users", [])
        authorized_users = [
            AuthorizedUserRecord.from_dict(u) if isinstance(u, dict) else u for u in raw_users
        ]
        return cls(
            applicant_id=str(data.get("applicant_id", "")),
            ssn=str(data.get("ssn", "")),
            first_name=str(data.get("first_name", "")),
            last_name=str(data.get("last_name", "")),
            dob=str(data.get("dob", "")),
            address=str(data.get("address", "")),
            phone=str(data.get("phone", "")),
            email=str(data.get("email", "")),
            credit_line_requested=float(data.get("credit_line_requested", 5000.0)),
            authorized_users=authorized_users,
            tenant_id=str(data.get("tenant_id", "default")),
            created_at=created_at,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class SsnValidationResult:
    """Detailed result of historical SSA allocation and consistency validation."""

    is_valid: bool
    status: SsnValidationStatus
    area_number: int
    group_number: int
    serial_number: int
    state_issued: str
    estimated_issuance_year_range: tuple[int, int] | None = None
    is_deceased: bool = False
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "status": self.status.value,
            "area_number": self.area_number,
            "group_number": self.group_number,
            "serial_number": self.serial_number,
            "state_issued": self.state_issued,
            "estimated_issuance_year_range": (
                list(self.estimated_issuance_year_range)
                if self.estimated_issuance_year_range
                else None
            ),
            "is_deceased": self.is_deceased,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SsnValidationResult:
        year_range = data.get("estimated_issuance_year_range")
        parsed_range = (
            (int(year_range[0]), int(year_range[1]))
            if isinstance(year_range, (list, tuple)) and len(year_range) == 2
            else None
        )
        return cls(
            is_valid=bool(data.get("is_valid", False)),
            status=SsnValidationStatus(data.get("status", "valid")),
            area_number=int(data.get("area_number", 0)),
            group_number=int(data.get("group_number", 0)),
            serial_number=int(data.get("serial_number", 0)),
            state_issued=str(data.get("state_issued", "Unknown")),
            estimated_issuance_year_range=parsed_range,
            is_deceased=bool(data.get("is_deceased", False)),
            details=str(data.get("details", "")),
        )


@dataclass
class SifAssessment:
    """Full synthetic identity risk assessment report for an identity profile."""

    assessment_id: str = field(default_factory=lambda: f"sif_{uuid4().hex[:12]}")
    applicant_id: str = ""
    tenant_id: str = "default"
    risk_score: float = 0.0
    risk_level: SifRiskLevel = SifRiskLevel.LOW
    action: SifAction = SifAction.APPROVE
    triggered_anomalies: list[SifAnomalyType] = field(default_factory=list)
    anomaly_breakdown: dict[str, float] = field(default_factory=dict)
    identity_fragmentation_score: float = 0.0
    ssn_validation: SsnValidationResult | None = None
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "applicant_id": self.applicant_id,
            "tenant_id": self.tenant_id,
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level.value,
            "action": self.action.value,
            "triggered_anomalies": [a.value for a in self.triggered_anomalies],
            "anomaly_breakdown": {k: round(v, 4) for k, v in self.anomaly_breakdown.items()},
            "identity_fragmentation_score": round(self.identity_fragmentation_score, 4),
            "ssn_validation": (self.ssn_validation.to_dict() if self.ssn_validation else None),
            "evaluated_at": self.evaluated_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SifAssessment:
        eval_raw = data.get("evaluated_at")
        evaluated_at = (
            datetime.fromisoformat(eval_raw)
            if isinstance(eval_raw, str)
            else (eval_raw if isinstance(eval_raw, datetime) else datetime.now(UTC))
        )
        anomalies = [SifAnomalyType(a) for a in data.get("triggered_anomalies", [])]
        ssn_val_raw = data.get("ssn_validation")
        ssn_validation = (
            SsnValidationResult.from_dict(ssn_val_raw) if isinstance(ssn_val_raw, dict) else None
        )
        return cls(
            assessment_id=str(data.get("assessment_id", "")),
            applicant_id=str(data.get("applicant_id", "")),
            tenant_id=str(data.get("tenant_id", "default")),
            risk_score=float(data.get("risk_score", 0.0)),
            risk_level=SifRiskLevel(data.get("risk_level", "low")),
            action=SifAction(data.get("action", "approve")),
            triggered_anomalies=anomalies,
            anomaly_breakdown=dict(data.get("anomaly_breakdown", {})),
            identity_fragmentation_score=float(data.get("identity_fragmentation_score", 0.0)),
            ssn_validation=ssn_validation,
            evaluated_at=evaluated_at,
            metadata=dict(data.get("metadata", {})),
        )
