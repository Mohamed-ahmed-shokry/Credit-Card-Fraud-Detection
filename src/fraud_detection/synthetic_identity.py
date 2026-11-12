"""Synthetic Identity Fraud (SIF) Detection, PII Inconsistency & Identity Verification Engine.

Provides domain models, SSA historical allocation verification, Death Master File screening,
PII element fragmentation, synthetic velocity clustering, authorized user abuse detection,
and composite risk decisioning with automated case/challenge escalation.
"""

from __future__ import annotations

from enum import StrEnum


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
