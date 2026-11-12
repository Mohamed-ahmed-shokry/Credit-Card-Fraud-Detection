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

# SSA Area to State mapping (pre-2011 historical allocation)
SSA_AREA_ALLOCATION: list[tuple[int, int, str, int, int]] = [
    (1, 3, "New Hampshire", 1936, 2011),
    (4, 7, "Maine", 1936, 2011),
    (8, 9, "Vermont", 1936, 2011),
    (10, 34, "Massachusetts", 1936, 2011),
    (35, 39, "Rhode Island", 1936, 2011),
    (40, 49, "Connecticut", 1936, 2011),
    (50, 134, "New York", 1936, 2011),
    (135, 158, "New Jersey", 1936, 2011),
    (159, 211, "Pennsylvania", 1936, 2011),
    (212, 220, "Maryland", 1936, 2011),
    (221, 222, "Delaware", 1936, 2011),
    (223, 231, "Virginia", 1936, 2011),
    (232, 232, "North Carolina", 1936, 2011),
    (233, 236, "West Virginia", 1936, 2011),
    (237, 246, "North Carolina", 1936, 2011),
    (247, 251, "South Carolina", 1936, 2011),
    (252, 260, "Georgia", 1936, 2011),
    (261, 267, "Florida", 1936, 2011),
    (268, 302, "Ohio", 1936, 2011),
    (303, 317, "Indiana", 1936, 2011),
    (318, 361, "Illinois", 1936, 2011),
    (362, 386, "Michigan", 1936, 2011),
    (387, 399, "Wisconsin", 1936, 2011),
    (400, 415, "Kentucky", 1936, 2011),
    (416, 424, "Tennessee", 1936, 2011),
    (425, 428, "Mississippi", 1936, 2011),
    (429, 432, "Arkansas", 1936, 2011),
    (433, 439, "Louisiana", 1936, 2011),
    (440, 448, "Oklahoma", 1936, 2011),
    (449, 467, "Texas", 1936, 2011),
    (468, 477, "Minnesota", 1936, 2011),
    (478, 485, "Iowa", 1936, 2011),
    (486, 500, "Missouri", 1936, 2011),
    (501, 502, "North Dakota", 1936, 2011),
    (503, 504, "South Dakota", 1936, 2011),
    (505, 508, "Nebraska", 1936, 2011),
    (509, 515, "Kansas", 1936, 2011),
    (516, 517, "Montana", 1936, 2011),
    (518, 519, "Idaho", 1936, 2011),
    (520, 520, "Wyoming", 1936, 2011),
    (521, 524, "Colorado", 1936, 2011),
    (525, 525, "New Mexico", 1936, 2011),
    (526, 527, "Arizona", 1936, 2011),
    (528, 529, "Utah", 1936, 2011),
    (530, 530, "Nevada", 1936, 2011),
    (531, 539, "Washington", 1936, 2011),
    (540, 544, "Oregon", 1936, 2011),
    (545, 573, "California", 1936, 2011),
    (574, 574, "Alaska", 1957, 2011),
    (575, 576, "Hawaii", 1936, 2011),
    (577, 579, "District of Columbia", 1936, 2011),
    (580, 580, "Virgin Islands", 1936, 2011),
    (581, 584, "Puerto Rico", 1936, 2011),
    (586, 586, "Pacific Islands", 1951, 2011),
    (700, 728, "Railroad Retirement", 1936, 1963),
]

# Unissued area numbers prior to June 25, 2011 randomization
PRE_2011_UNISSUED_AREAS: set[int] = (
    {0, 666}
    | set(range(587, 600))
    | set(range(640, 650))
    | set(range(667, 700))
    | set(range(729, 900))
)

# Simulated Death Master File (DMF) records for high-risk validation
DEFAULT_DECEASED_RECORDS: dict[str, dict[str, str]] = {
    # 9-digit cleaned SSN -> metadata
    "042129999": {"name": "Arthur Pendelton", "dod": "2018-04-12", "state": "CT"},
    "111223333": {"name": "Eleanor Vance", "dod": "2015-11-03", "state": "NY"},
    "318445555": {"name": "Walter White", "dod": "2020-09-07", "state": "IL"},
    "449998888": {"name": "Henry Higgins", "dod": "2012-01-20", "state": "TX"},
    "545001122": {"name": "Clara Bow", "dod": "2016-08-14", "state": "CA"},
}


def get_group_order(group: int) -> int:
    """Return the SSA group issuance progression order (1 to 99)."""
    if 1 <= group <= 9 and group % 2 == 1:
        return (group + 1) // 2  # 1, 3, 5, 7, 9 -> 1, 2, 3, 4, 5
    if 10 <= group <= 98 and group % 2 == 0:
        return 5 + (group - 8) // 2  # 10, 12, ... 98 -> 6, 7, ... 50
    if 2 <= group <= 8 and group % 2 == 0:
        return 50 + group // 2  # 02, 04, 06, 08 -> 51, 52, 53, 54
    if 11 <= group <= 99 and group % 2 == 1:
        return 54 + (group - 9) // 2  # 11, 13, ... 99 -> 55, 56, ... 99
    return 0


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


class SsnValidator:
    """Validates US Social Security Numbers against SSA allocation history and death registries."""

    def __init__(
        self,
        deceased_records: dict[str, dict[str, str]] | None = None,
        unissued_areas: set[int] | None = None,
    ) -> None:
        self._deceased_records: dict[str, dict[str, str]] = dict(
            DEFAULT_DECEASED_RECORDS if deceased_records is None else deceased_records
        )
        self._unissued_areas: set[int] = set(
            PRE_2011_UNISSUED_AREAS if unissued_areas is None else unissued_areas
        )

    def add_deceased_record(self, ssn: str, name: str, dod: str, state: str = "US") -> None:
        clean = re.sub(r"\D", "", ssn)
        if len(clean) == 9:
            self._deceased_records[clean] = {"name": name, "dod": dod, "state": state}

    def is_deceased(self, ssn: str) -> tuple[bool, dict[str, str] | None]:
        clean = re.sub(r"\D", "", ssn)
        record = self._deceased_records.get(clean)
        return (record is not None, record)

    def lookup_area(self, area: int) -> tuple[str, int, int] | None:
        for start, end, state, from_yr, to_yr in SSA_AREA_ALLOCATION:
            if start <= area <= end:
                return (state, from_yr, to_yr)
        return None

    def validate(self, ssn: str, dob: str | None = None) -> SsnValidationResult:
        clean = re.sub(r"\D", "", ssn)
        if len(clean) != 9:
            return SsnValidationResult(
                is_valid=False,
                status=SsnValidationStatus.INVALID_FORMAT,
                area_number=0,
                group_number=0,
                serial_number=0,
                state_issued="Unknown",
                details="SSN must contain exactly 9 numeric digits",
            )

        area = int(clean[:3])
        group = int(clean[3:5])
        serial = int(clean[5:9])

        if group == 0 or serial == 0:
            return SsnValidationResult(
                is_valid=False,
                status=SsnValidationStatus.RESERVED_OR_SPECIAL,
                area_number=area,
                group_number=group,
                serial_number=serial,
                state_issued="Unknown",
                details="Group number 00 and serial number 0000 are invalid and reserved",
            )

        if area == 0 or area == 666 or area >= 900:
            return SsnValidationResult(
                is_valid=False,
                status=SsnValidationStatus.RESERVED_OR_SPECIAL,
                area_number=area,
                group_number=group,
                serial_number=serial,
                state_issued="Unknown",
                details=f"Area number {area:03d} is reserved or prohibited",
            )

        deceased_match, dec_info = self.is_deceased(clean)
        if deceased_match and dec_info:
            d_name = dec_info.get("name", "Unknown")
            d_dod = dec_info.get("dod", "Unknown")
            return SsnValidationResult(
                is_valid=False,
                status=SsnValidationStatus.DECEASED,
                area_number=area,
                group_number=group,
                serial_number=serial,
                state_issued=dec_info.get("state", "Unknown"),
                is_deceased=True,
                details=f"SSN matched Death Master File: {d_name} (DOD: {d_dod})",
            )

        birth_year: int | None = None
        if dob:
            try:
                birth_year = int(dob.split("-")[0])
            except (ValueError, IndexError):
                birth_year = None

        area_info = self.lookup_area(area)
        if area_info is None:
            if area in self._unissued_areas:
                if birth_year is not None and birth_year < 2011:
                    detail_msg = (
                        f"Area {area:03d} was unissued prior to 2011 randomization, "
                        f"but applicant birth year is {birth_year}"
                    )
                    return SsnValidationResult(
                        is_valid=False,
                        status=SsnValidationStatus.UNISSUED_AREA,
                        area_number=area,
                        group_number=group,
                        serial_number=serial,
                        state_issued="Unissued Pre-2011",
                        details=detail_msg,
                    )
                return SsnValidationResult(
                    is_valid=True,
                    status=SsnValidationStatus.ISSUED_POST_RANDOMIZATION,
                    area_number=area,
                    group_number=group,
                    serial_number=serial,
                    state_issued="Randomized Post-2011",
                    estimated_issuance_year_range=(2011, 2026),
                    details="SSN area issued under SSA post-June 25, 2011 randomization",
                )
            return SsnValidationResult(
                is_valid=True,
                status=SsnValidationStatus.VALID,
                area_number=area,
                group_number=group,
                serial_number=serial,
                state_issued="Unknown (Unmapped)",
                details="Area number outside standard historical tables",
            )

        state, from_yr, to_yr = area_info

        # Check special retired series like Railroad Retirement (1936-1963)
        if state == "Railroad Retirement" and birth_year is not None and birth_year > 1963:
            rr_detail = (
                "SSN area belonged to Railroad Retirement board discontinued in 1963, "
                f"but applicant born in {birth_year}"
            )
            return SsnValidationResult(
                is_valid=False,
                status=SsnValidationStatus.DOB_INCONSISTENT,
                area_number=area,
                group_number=group,
                serial_number=serial,
                state_issued=state,
                estimated_issuance_year_range=(from_yr, to_yr),
                details=rr_detail,
            )

        return SsnValidationResult(
            is_valid=True,
            status=SsnValidationStatus.VALID,
            area_number=area,
            group_number=group,
            serial_number=serial,
            state_issued=state,
            estimated_issuance_year_range=(from_yr, to_yr),
            details=f"Valid pre-randomization SSN issued in {state}",
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
