"""Synthetic Identity Fraud (SIF) Detection, PII Inconsistency & Identity Verification Engine.

Provides domain models, SSA historical allocation verification, Death Master File screening,
PII element fragmentation, synthetic velocity clustering, authorized user abuse detection,
and composite risk decisioning with automated case/challenge escalation.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from fraud_detection.cases import CaseManager
    from fraud_detection.challenge import ChallengeManager
    from fraud_detection.events import EventBus

from fraud_detection.aml import jaro_winkler_similarity, levenshtein_similarity

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


@dataclass(frozen=True)
class PiiFragmentationMatch:
    """Individual conflict or overlap indicating PII splicing across applicant profiles."""

    matched_applicant_id: str
    overlap_field: str
    conflicting_field: str
    similarity: float
    risk_penalty: float
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched_applicant_id": self.matched_applicant_id,
            "overlap_field": self.overlap_field,
            "conflicting_field": self.conflicting_field,
            "similarity": round(self.similarity, 4),
            "risk_penalty": round(self.risk_penalty, 4),
            "description": self.description,
        }


class PiiInconsistencyEngine:
    """Evaluates cross-application PII element fragmentation and pairwise similarity anomalies."""

    def __init__(
        self,
        name_similarity_threshold: float = 0.70,
        address_similarity_threshold: float = 0.85,
    ) -> None:
        self.name_similarity_threshold = name_similarity_threshold
        self.address_similarity_threshold = address_similarity_threshold

    def evaluate_inconsistency(
        self,
        target: IdentityProfile,
        historical_profiles: list[IdentityProfile],
    ) -> tuple[float, list[PiiFragmentationMatch], list[SifAnomalyType]]:
        """Compute PII fragmentation penalty and detect conflicting historical identities."""
        matches: list[PiiFragmentationMatch] = []
        anomalies: set[SifAnomalyType] = set()
        total_penalty = 0.0

        target_name = f"{target.first_name} {target.last_name}".strip().lower()
        target_ssn_hash = target.ssn_hash()
        target_phone = re.sub(r"\D", "", target.phone)
        target_email = target.email.strip().lower()
        target_address = target.address.strip().lower()

        for h in historical_profiles:
            if h.applicant_id == target.applicant_id:
                continue

            h_name = f"{h.first_name} {h.last_name}".strip().lower()
            h_ssn_hash = h.ssn_hash()
            h_phone = re.sub(r"\D", "", h.phone)
            h_email = h.email.strip().lower()
            h_address = h.address.strip().lower()

            name_sim = jaro_winkler_similarity(target_name, h_name)

            # 1. Same SSN, different Name
            if target_ssn_hash == h_ssn_hash:
                if name_sim < self.name_similarity_threshold:
                    penalty = 0.85
                    total_penalty += penalty
                    anomalies.add(SifAnomalyType.PII_ELEMENT_FRAGMENTATION)
                    matches.append(
                        PiiFragmentationMatch(
                            matched_applicant_id=h.applicant_id,
                            overlap_field="ssn",
                            conflicting_field="name",
                            similarity=name_sim,
                            risk_penalty=penalty,
                            description=(
                                f"Same SSN used by distinct name '{h.first_name} {h.last_name}' "
                                f"(similarity: {name_sim:.2f})"
                            ),
                        )
                    )
                # Check DOB with same SSN
                if target.dob and h.dob and target.dob != h.dob:
                    penalty = 0.60
                    total_penalty += penalty
                    anomalies.add(SifAnomalyType.SSN_DOB_MISMATCH)
                    matches.append(
                        PiiFragmentationMatch(
                            matched_applicant_id=h.applicant_id,
                            overlap_field="ssn",
                            conflicting_field="dob",
                            similarity=0.0,
                            risk_penalty=penalty,
                            description=f"Same SSN registered with conflicting DOB '{h.dob}'",
                        )
                    )

            # 2. Similar/Same Name & DOB, but different SSN
            elif name_sim >= 0.90 and target.dob and h.dob and target.dob == h.dob:
                penalty = 0.75
                total_penalty += penalty
                anomalies.add(SifAnomalyType.PII_ELEMENT_FRAGMENTATION)
                matches.append(
                    PiiFragmentationMatch(
                        matched_applicant_id=h.applicant_id,
                        overlap_field="name_and_dob",
                        conflicting_field="ssn",
                        similarity=name_sim,
                        risk_penalty=penalty,
                        description="Identical Name and DOB associated with a different SSN",
                    )
                )

            # 3. Same Phone, different SSN and Name
            if (
                target_phone
                and h_phone
                and target_phone == h_phone
                and target_ssn_hash != h_ssn_hash
                and name_sim < self.name_similarity_threshold
            ):
                penalty = 0.35
                total_penalty += penalty
                matches.append(
                    PiiFragmentationMatch(
                        matched_applicant_id=h.applicant_id,
                        overlap_field="phone",
                        conflicting_field="identity",
                        similarity=name_sim,
                        risk_penalty=penalty,
                        description=f"Shared phone number across distinct identity '{h_name}'",
                    )
                )

            # 4. Same Email, different SSN and Name
            if (
                target_email
                and h_email
                and target_email == h_email
                and target_ssn_hash != h_ssn_hash
                and name_sim < self.name_similarity_threshold
            ):
                penalty = 0.35
                total_penalty += penalty
                matches.append(
                    PiiFragmentationMatch(
                        matched_applicant_id=h.applicant_id,
                        overlap_field="email",
                        conflicting_field="identity",
                        similarity=name_sim,
                        risk_penalty=penalty,
                        description=f"Shared email address across distinct identity '{h_name}'",
                    )
                )

            # 5. Same Address, different last name and different SSN
            if target_address and h_address:
                addr_sim = levenshtein_similarity(target_address, h_address)
                if (
                    addr_sim >= self.address_similarity_threshold
                    and target.last_name.lower() != h.last_name.lower()
                    and target_ssn_hash != h_ssn_hash
                ):
                    penalty = 0.25
                    total_penalty += penalty
                    desc = f"Shared address '{h.address}' with unrelated last name '{h.last_name}'"
                    matches.append(
                        PiiFragmentationMatch(
                            matched_applicant_id=h.applicant_id,
                            overlap_field="address",
                            conflicting_field="last_name",
                            similarity=addr_sim,
                            risk_penalty=penalty,
                            description=desc,
                        )
                    )

        fragmentation_score = min(1.0, total_penalty)
        return (fragmentation_score, matches, list(anomalies))


DROP_ADDRESS_PATTERNS: list[str] = [
    r"\bp\.?o\.?\s*box\b",
    r"\bpost\s*office\s*box\b",
    r"\bpmb\s*#?\d+\b",
    r"\bsuite\s*pmb\b",
    r"\bmailbox\s*etc\b",
    r"\bparcel\s*pickup\b",
    r"\bc/o\s*drop\b",
    r"\bvmf\s*box\b",
    r"\bcommercial\s*mail\b",
]

BURNER_PHONE_PATTERNS: list[str] = [
    r"^(?:1)?(?:800|888|877|866|855|844|833)\d{7}$",
    r"^(?:1)?[2-9]\d{2}55501\d{2}$",
    r"^(?:1)?(\d)\1{6,}$",
]


class SyntheticIdentityTracker:
    """Sliding-window velocity and anomaly tracker for synthetic identity patterns."""

    def __init__(
        self,
        velocity_window_hours: float = 24.0,
        max_applications_per_ssn: int = 2,
        max_applications_per_phone: int = 3,
        max_applications_per_address: int = 4,
    ) -> None:
        self.velocity_window = timedelta(hours=velocity_window_hours)
        self.max_applications_per_ssn = max_applications_per_ssn
        self.max_applications_per_phone = max_applications_per_phone
        self.max_applications_per_address = max_applications_per_address
        self._lock = threading.RLock()

        self._ssn_history: dict[str, deque[datetime]] = defaultdict(deque)
        self._phone_history: dict[str, deque[datetime]] = defaultdict(deque)
        self._address_history: dict[str, deque[datetime]] = defaultdict(deque)

        self._compiled_drop_patterns = [re.compile(p, re.IGNORECASE) for p in DROP_ADDRESS_PATTERNS]
        self._compiled_phone_patterns = [re.compile(p) for p in BURNER_PHONE_PATTERNS]

    def is_drop_address(self, address: str) -> bool:
        """Check if address matches known mail drops, freight forwarders, or PO Boxes."""
        normalized = address.strip()
        return any(p.search(normalized) for p in self._compiled_drop_patterns)

    def is_burner_phone(self, phone: str) -> bool:
        """Check if phone number matches toll-free, fake, or repeated burner patterns."""
        clean = re.sub(r"\D", "", phone)
        return any(p.match(clean) for p in self._compiled_phone_patterns)

    def is_authorized_user_abuse(self, profile: IdentityProfile) -> tuple[bool, str]:
        """Detect credit piggybacking / authorized user manipulation."""
        users = profile.authorized_users
        if not users:
            return (False, "")

        count = len(users)
        # More than 2 authorized users on an initial application is high risk
        unrelated_count = sum(
            1
            for u in users
            if u.relationship.lower() in {"other", "broker", "investor", "unknown", "none"}
            or not u.relationship
        )

        if count >= 3 or (count >= 2 and unrelated_count >= 1):
            msg = (
                f"Elevated authorized user count ({count}) "
                f"with {unrelated_count} non-household relations"
            )
            return (True, msg)
        return (False, "")

    def check_and_record(
        self,
        profile: IdentityProfile,
        now: datetime | None = None,
    ) -> tuple[float, list[SifAnomalyType], dict[str, Any]]:
        """Evaluate synthetic velocity, drop address, burner phone, and piggybacking."""
        current_time = now if now is not None else profile.created_at
        anomalies: list[SifAnomalyType] = []
        breakdown: dict[str, float] = {}
        risk_score = 0.0

        # 1. Drop address check
        if self.is_drop_address(profile.address):
            anomalies.append(SifAnomalyType.HIGH_RISK_ADDRESS_DROP)
            breakdown["high_risk_address_drop"] = 0.35
            risk_score += 0.35

        # 2. Burner phone check
        if self.is_burner_phone(profile.phone):
            anomalies.append(SifAnomalyType.BURNER_PHONE_DETECTED)
            breakdown["burner_phone_detected"] = 0.40
            risk_score += 0.40

        # 3. Authorized user abuse check
        is_abuse, abuse_reason = self.is_authorized_user_abuse(profile)
        if is_abuse:
            anomalies.append(SifAnomalyType.AUTHORIZED_USER_ABUSE)
            breakdown["authorized_user_abuse"] = 0.50
            risk_score += 0.50

        # 4. Sliding-window velocity check
        ssn_key = profile.ssn_hash()
        phone_key = re.sub(r"\D", "", profile.phone)
        addr_key = profile.address.strip().lower()

        cutoff = current_time - self.velocity_window

        with self._lock:
            # Clean and check SSN velocity
            ssn_q = self._ssn_history[ssn_key]
            while ssn_q and ssn_q[0] < cutoff:
                ssn_q.popleft()
            if len(ssn_q) >= self.max_applications_per_ssn:
                anomalies.append(SifAnomalyType.SYNTHETIC_VELOCITY_BURST)
                breakdown["ssn_velocity_burst"] = 0.60
                risk_score += 0.60
            ssn_q.append(current_time)

            # Clean and check Phone velocity
            if phone_key:
                phone_q = self._phone_history[phone_key]
                while phone_q and phone_q[0] < cutoff:
                    phone_q.popleft()
                if (
                    len(phone_q) >= self.max_applications_per_phone
                    and SifAnomalyType.SYNTHETIC_VELOCITY_BURST not in anomalies
                ):
                    anomalies.append(SifAnomalyType.SYNTHETIC_VELOCITY_BURST)
                    breakdown["phone_velocity_burst"] = 0.45
                    risk_score += 0.45
                phone_q.append(current_time)

            # Clean and check Address velocity
            if addr_key:
                addr_q = self._address_history[addr_key]
                while addr_q and addr_q[0] < cutoff:
                    addr_q.popleft()
                if len(addr_q) >= self.max_applications_per_address:
                    if SifAnomalyType.HIGH_RISK_ADDRESS_DROP not in anomalies:
                        anomalies.append(SifAnomalyType.HIGH_RISK_ADDRESS_DROP)
                    breakdown["address_velocity_spike"] = 0.40
                    risk_score += 0.40
                addr_q.append(current_time)

        details = {
            "breakdown": breakdown,
            "abuse_reason": abuse_reason,
            "drop_address": SifAnomalyType.HIGH_RISK_ADDRESS_DROP in anomalies,
            "burner_phone": SifAnomalyType.BURNER_PHONE_DETECTED in anomalies,
        }
        return (min(1.0, risk_score), anomalies, details)


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


class SyntheticIdentityStore:
    """Thread-safe in-memory store for identity profiles and synthetic fraud assessments."""

    def __init__(self, max_records: int = 50_000) -> None:
        self.max_records = max_records
        self._lock = threading.RLock()

        self._profiles: dict[str, IdentityProfile] = {}
        self._profiles_by_ssn: dict[str, list[str]] = defaultdict(list)
        self._profiles_by_phone: dict[str, list[str]] = defaultdict(list)
        self._profiles_by_email: dict[str, list[str]] = defaultdict(list)
        self._profiles_by_tenant: dict[str, list[str]] = defaultdict(list)

        self._assessments: dict[str, SifAssessment] = {}
        self._assessments_by_applicant: dict[str, list[str]] = defaultdict(list)
        self._assessments_by_tenant: dict[str, list[str]] = defaultdict(list)

    def add_profile(self, profile: IdentityProfile) -> None:
        """Store or update an applicant identity profile with secondary indexing."""
        with self._lock:
            app_id = profile.applicant_id
            if len(self._profiles) >= self.max_records and app_id not in self._profiles:
                # Evict oldest record
                oldest_key = next(iter(self._profiles))
                self.remove_profile(oldest_key)

            self._profiles[app_id] = profile

            ssn_h = profile.ssn_hash()
            if app_id not in self._profiles_by_ssn[ssn_h]:
                self._profiles_by_ssn[ssn_h].append(app_id)

            phone_clean = re.sub(r"\D", "", profile.phone)
            if phone_clean and app_id not in self._profiles_by_phone[phone_clean]:
                self._profiles_by_phone[phone_clean].append(app_id)

            email_norm = profile.email.strip().lower()
            if email_norm and app_id not in self._profiles_by_email[email_norm]:
                self._profiles_by_email[email_norm].append(app_id)

            tenant = profile.tenant_id
            if app_id not in self._profiles_by_tenant[tenant]:
                self._profiles_by_tenant[tenant].append(app_id)

    def remove_profile(self, applicant_id: str) -> None:
        with self._lock:
            prof = self._profiles.pop(applicant_id, None)
            if prof:
                ssn_h = prof.ssn_hash()
                if applicant_id in self._profiles_by_ssn[ssn_h]:
                    self._profiles_by_ssn[ssn_h].remove(applicant_id)

                phone_clean = re.sub(r"\D", "", prof.phone)
                if applicant_id in self._profiles_by_phone[phone_clean]:
                    self._profiles_by_phone[phone_clean].remove(applicant_id)

                email_norm = prof.email.strip().lower()
                if applicant_id in self._profiles_by_email[email_norm]:
                    self._profiles_by_email[email_norm].remove(applicant_id)

                tenant = prof.tenant_id
                if applicant_id in self._profiles_by_tenant[tenant]:
                    self._profiles_by_tenant[tenant].remove(applicant_id)

    def get_profile(self, applicant_id: str) -> IdentityProfile | None:
        with self._lock:
            return self._profiles.get(applicant_id)

    def find_profiles_by_ssn(self, ssn_hash: str) -> list[IdentityProfile]:
        with self._lock:
            app_ids = self._profiles_by_ssn.get(ssn_hash, [])
            return [self._profiles[aid] for aid in app_ids if aid in self._profiles]

    def find_profiles_by_phone(self, phone: str) -> list[IdentityProfile]:
        clean = re.sub(r"\D", "", phone)
        with self._lock:
            app_ids = self._profiles_by_phone.get(clean, [])
            return [self._profiles[aid] for aid in app_ids if aid in self._profiles]

    def find_profiles_by_email(self, email: str) -> list[IdentityProfile]:
        norm = email.strip().lower()
        with self._lock:
            app_ids = self._profiles_by_email.get(norm, [])
            return [self._profiles[aid] for aid in app_ids if aid in self._profiles]

    def list_profiles(
        self, tenant_id: str | None = None, limit: int = 100
    ) -> list[IdentityProfile]:
        with self._lock:
            if tenant_id:
                app_ids = self._profiles_by_tenant.get(tenant_id, [])
                return [self._profiles[aid] for aid in reversed(app_ids) if aid in self._profiles][
                    :limit
                ]
            return list(reversed(list(self._profiles.values())))[:limit]

    def add_assessment(self, assessment: SifAssessment) -> None:
        with self._lock:
            aid = assessment.assessment_id
            if len(self._assessments) >= self.max_records and aid not in self._assessments:
                oldest_key = next(iter(self._assessments))
                self._assessments.pop(oldest_key, None)

            self._assessments[aid] = assessment
            self._assessments_by_applicant[assessment.applicant_id].append(aid)
            self._assessments_by_tenant[assessment.tenant_id].append(aid)

    def get_assessment(self, assessment_id: str) -> SifAssessment | None:
        with self._lock:
            return self._assessments.get(assessment_id)

    def list_assessments(
        self, tenant_id: str | None = None, limit: int = 100
    ) -> list[SifAssessment]:
        with self._lock:
            if tenant_id:
                aids = self._assessments_by_tenant.get(tenant_id, [])
                return [
                    self._assessments[aid] for aid in reversed(aids) if aid in self._assessments
                ][:limit]
            return list(reversed(list(self._assessments.values())))[:limit]

    @property
    def profile_count(self) -> int:
        with self._lock:
            return len(self._profiles)

    @property
    def assessment_count(self) -> int:
        with self._lock:
            return len(self._assessments)

    def clear(self) -> None:
        with self._lock:
            self._profiles.clear()
            self._profiles_by_ssn.clear()
            self._profiles_by_phone.clear()
            self._profiles_by_email.clear()
            self._profiles_by_tenant.clear()
            self._assessments.clear()
            self._assessments_by_applicant.clear()
            self._assessments_by_tenant.clear()


class SyntheticIdentityEngine:
    """Composite scoring and decision engine for Synthetic Identity Fraud detection."""

    def __init__(
        self,
        store: SyntheticIdentityStore | None = None,
        ssn_validator: SsnValidator | None = None,
        pii_engine: PiiInconsistencyEngine | None = None,
        tracker: SyntheticIdentityTracker | None = None,
        case_manager: CaseManager | None = None,
        challenge_manager: ChallengeManager | None = None,
        event_bus: EventBus | None = None,
        step_up_threshold: float = 0.35,
        manual_review_threshold: float = 0.65,
        block_threshold: float = 0.85,
    ) -> None:
        self.store = store or SyntheticIdentityStore()
        self.ssn_validator = ssn_validator or SsnValidator()
        self.pii_engine = pii_engine or PiiInconsistencyEngine()
        self.tracker = tracker or SyntheticIdentityTracker()
        self.case_manager = case_manager
        self.challenge_manager = challenge_manager
        self.event_bus = event_bus

        self.step_up_threshold = step_up_threshold
        self.manual_review_threshold = manual_review_threshold
        self.block_threshold = block_threshold

    def evaluate(
        self,
        profile: IdentityProfile,
        now: datetime | None = None,
    ) -> SifAssessment:
        """Execute full synthetic identity evaluation and determine policy routing."""
        eval_time = now if now is not None else profile.created_at
        anomalies: list[SifAnomalyType] = []
        breakdown: dict[str, float] = {}

        # 1. SSN issuance and death registry validation
        ssn_val = self.ssn_validator.validate(profile.ssn, dob=profile.dob)
        ssn_penalty = 0.0

        if ssn_val.is_deceased:
            anomalies.append(SifAnomalyType.DECEASED_SSN_MATCH)
            ssn_penalty = 1.0
            breakdown["deceased_ssn"] = 1.0
        elif not ssn_val.is_valid:
            if ssn_val.status == SsnValidationStatus.DOB_INCONSISTENT:
                anomalies.append(SifAnomalyType.SSN_DOB_MISMATCH)
                ssn_penalty = 0.85
                breakdown["ssn_dob_mismatch"] = 0.85
            elif ssn_val.status == SsnValidationStatus.UNISSUED_AREA:
                anomalies.append(SifAnomalyType.UNISSUED_SSN_AREA)
                ssn_penalty = 0.80
                breakdown["unissued_ssn_area"] = 0.80
            elif ssn_val.status in (
                SsnValidationStatus.INVALID_FORMAT,
                SsnValidationStatus.RESERVED_OR_SPECIAL,
            ):
                anomalies.append(SifAnomalyType.INVALID_SSN_STRUCTURE)
                ssn_penalty = 0.75
                breakdown["invalid_ssn_structure"] = 0.75

        # 2. PII element fragmentation against existing profiles
        existing_profiles = self.store.list_profiles(tenant_id=profile.tenant_id, limit=500)
        frag_score, _matches, frag_anomalies = self.pii_engine.evaluate_inconsistency(
            profile, existing_profiles
        )
        for a in frag_anomalies:
            if a not in anomalies:
                anomalies.append(a)
        if frag_score > 0.0:
            breakdown["pii_fragmentation"] = round(frag_score, 4)

        # 3. Synthetic velocity, drop address, burner phone, authorized user abuse
        vel_score, vel_anomalies, vel_details = self.tracker.check_and_record(
            profile, now=eval_time
        )
        for a in vel_anomalies:
            if a not in anomalies:
                anomalies.append(a)
        for k, v in vel_details.get("breakdown", {}).items():
            breakdown[k] = round(float(v), 4)

        # 4. Calibrated Composite Risk Scoring
        if ssn_val.is_deceased:
            risk_score = 1.0
        else:
            # Weighted aggregation
            risk_score = min(
                1.0,
                max(
                    0.0,
                    ssn_penalty * 0.40 + frag_score * 0.45 + vel_score * 0.35,
                ),
            )
            if (
                SifAnomalyType.PII_ELEMENT_FRAGMENTATION in anomalies
                or SifAnomalyType.SSN_DOB_MISMATCH in anomalies
            ):
                risk_score = max(risk_score, 0.70)
            elif (
                SifAnomalyType.BURNER_PHONE_DETECTED in anomalies
                or SifAnomalyType.HIGH_RISK_ADDRESS_DROP in anomalies
            ):
                risk_score = max(risk_score, 0.40)
            elif SifAnomalyType.AUTHORIZED_USER_ABUSE in anomalies:
                risk_score = max(risk_score, 0.55)

        # 5. Policy Decision Routing
        if risk_score >= self.block_threshold or ssn_val.is_deceased:
            action = SifAction.DECLINE_BLOCK
            risk_level = SifRiskLevel.CRITICAL
        elif risk_score >= self.manual_review_threshold:
            action = SifAction.MANUAL_REVIEW
            risk_level = SifRiskLevel.HIGH
        elif risk_score >= self.step_up_threshold:
            action = SifAction.STEP_UP_IDENTITY_PROOFING
            risk_level = SifRiskLevel.MEDIUM
        else:
            action = SifAction.APPROVE
            risk_level = SifRiskLevel.LOW

        assessment = SifAssessment(
            applicant_id=profile.applicant_id,
            tenant_id=profile.tenant_id,
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            action=action,
            triggered_anomalies=anomalies,
            anomaly_breakdown=breakdown,
            identity_fragmentation_score=round(frag_score, 4),
            ssn_validation=ssn_val,
            evaluated_at=eval_time,
            metadata={
                "credit_line_requested": profile.credit_line_requested,
                "authorized_users_count": len(profile.authorized_users),
            },
        )

        # Escalate to CaseManager on MANUAL_REVIEW
        if action == SifAction.MANUAL_REVIEW and self.case_manager:
            with contextlib.suppress(Exception):
                self.case_manager.create_case(
                    transaction_id=f"sif_{profile.applicant_id}",
                    tenant_id=profile.tenant_id,
                    fraud_score=risk_score,
                    metadata={
                        "sif_assessment_id": assessment.assessment_id,
                        "anomalies": [a.value for a in anomalies],
                    },
                )

        # Trigger Challenge on STEP_UP_IDENTITY_PROOFING
        if action == SifAction.STEP_UP_IDENTITY_PROOFING and self.challenge_manager:
            with contextlib.suppress(Exception):
                from fraud_detection.challenge import ChallengeMethod

                target = profile.phone or profile.email
                method = ChallengeMethod.SMS_OTP if profile.phone else ChallengeMethod.EMAIL_OTP
                self.challenge_manager.create_challenge(
                    transaction_id=f"sif_{profile.applicant_id}",
                    user_id=profile.applicant_id,
                    tenant_id=profile.tenant_id,
                    destination=target,
                    method=method,
                    metadata={"sif_assessment_id": assessment.assessment_id},
                )

        # Save to store
        self.store.add_profile(profile)
        self.store.add_assessment(assessment)

        # Publish domain events
        if self.event_bus:
            with contextlib.suppress(Exception):
                from fraud_detection.events import EventType, FraudDomainEvent

                self.event_bus.publish(
                    FraudDomainEvent(
                        event_type=EventType.SIF_APPLICATION_EVALUATED,
                        payload={
                            "assessment_id": assessment.assessment_id,
                            "applicant_id": profile.applicant_id,
                            "risk_score": assessment.risk_score,
                            "risk_level": assessment.risk_level.value,
                            "action": assessment.action.value,
                        },
                        tenant_id=profile.tenant_id,
                    ),
                    synchronous=True,
                )

                if anomalies:
                    self.event_bus.publish(
                        FraudDomainEvent(
                            event_type=EventType.SIF_ANOMALY_DETECTED,
                            payload={
                                "assessment_id": assessment.assessment_id,
                                "applicant_id": profile.applicant_id,
                                "anomalies": [a.value for a in anomalies],
                                "risk_score": assessment.risk_score,
                            },
                            tenant_id=profile.tenant_id,
                        ),
                        synchronous=True,
                    )

                if action == SifAction.DECLINE_BLOCK:
                    self.event_bus.publish(
                        FraudDomainEvent(
                            event_type=EventType.SIF_APPLICATION_DECLINED,
                            payload={
                                "assessment_id": assessment.assessment_id,
                                "applicant_id": profile.applicant_id,
                                "risk_score": assessment.risk_score,
                            },
                            tenant_id=profile.tenant_id,
                        ),
                        synchronous=True,
                    )
                elif action == SifAction.APPROVE:
                    self.event_bus.publish(
                        FraudDomainEvent(
                            event_type=EventType.SIF_IDENTITY_CONFIRMED,
                            payload={
                                "assessment_id": assessment.assessment_id,
                                "applicant_id": profile.applicant_id,
                                "risk_score": assessment.risk_score,
                            },
                            tenant_id=profile.tenant_id,
                        ),
                        synchronous=True,
                    )

        return assessment
