from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


def to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


class ApiModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",
        use_enum_values=True,
    )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AnswerabilityStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"


class CuratorialRole(str, Enum):
    INTRODUCTION = "opening"
    HISTORICAL_CONTEXT = "context"
    CORE_EVIDENCE = "core_evidence"
    COUNTERPOINT = "contrast"
    SYNTHESIS = "synthesis"


ROLE_LABELS: dict[str, str] = {
    CuratorialRole.INTRODUCTION.value: "引入",
    CuratorialRole.HISTORICAL_CONTEXT.value: "历史背景",
    CuratorialRole.CORE_EVIDENCE.value: "核心证据",
    CuratorialRole.COUNTERPOINT.value: "对照／其他声音",
    CuratorialRole.SYNTHESIS.value: "综合与延伸",
}


class SentenceType(str, Enum):
    INSTITUTION_FACT = "institution_fact"
    VISUAL_OBSERVATION = "visual_observation"
    SYSTEM_INFERENCE = "system_inference"
    UNCERTAIN = "uncertain"


class ExhibitionStatus(str, Enum):
    """Lifecycle states.

    Only ``DRAFT``/``GENERATING``/``READY`` appear on the visitor path. The
    review states are retained so existing stored records stay loadable and so
    the internal ``/_dev`` tooling keeps working, but the public flow no longer
    passes through them (01b §2.3).
    """

    DRAFT = "draft"
    GENERATING = "generating"
    READY = "ready"
    AUTO_VALIDATED = "auto_validated"
    REVIEW_PENDING = "review_pending"
    PUBLISHED = "published"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


# States a visitor may never mutate.
LOCKED_STATUSES = {
    ExhibitionStatus.REVIEW_PENDING.value,
    ExhibitionStatus.PUBLISHED.value,
    ExhibitionStatus.WITHDRAWN.value,
}


class EvidenceChunk(ApiModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    source_title: str = Field(min_length=1)
    source_location: str = "Institution object record"
    supports: str = ""
    reviewed: bool = False
    review_status: str = "pending"
    verification: str | None = None
    # The quoted text has its own rights scope. It must not inherit an image
    # licence merely because both came from the same object record.
    license: str | None = None
    rights_uri: str | None = None
    source_kind: str | None = None


class EvidenceDepth(str, Enum):
    """How much independent institution prose backs an object.

    ``THIN`` objects (notably every Met record, which has no curatorial
    description field) carry only metadata restatement, so they are barred from
    core-evidence curatorial roles. See 01b §1.4.
    """

    FULL = "full"
    THIN = "thin"


class MuseumObject(ApiModel):
    id: str = Field(min_length=1)
    source_id: str | None = None
    accession_number: str = ""
    title: str = Field(min_length=1)
    title_original: str | None = None
    date: str = ""
    date_earliest: int | None = None
    date_latest: int | None = None
    maker: str = ""
    medium: str = ""
    type: str = "Collection object"
    culture: str = ""
    creator: str | None = None
    material: str | None = None
    place: str | None = None
    culture_display: str | None = None
    description: str | None = None
    image_url: str = Field(min_length=1)
    image_url_large: str | None = None
    object_url: str = Field(min_length=1)
    # ``rights`` remains required for frozen v2/v3 collections and existing
    # clients. New records additionally distinguish the reusable asset/data
    # scopes below; missing structured fields mean unspecified, not inherited.
    rights: str = Field(min_length=1)
    rights_uri: str | None = None
    image_license: str | None = None
    image_rights_uri: str | None = None
    metadata_license: str | None = None
    metadata_rights_uri: str | None = None
    curatorial_text_license: str | None = None
    curatorial_text_rights_uri: str | None = None
    alt_text: str = ""
    alt_text_source: str = "metadata_fallback"
    themes: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    institution: str = ""
    institution_id: str = ""
    department: str = ""
    classification: str = ""
    credit_line: str = ""
    culture_pack_ids: list[str] = Field(default_factory=list)
    evidence_domain_ids: list[str] = Field(default_factory=list)
    relation_facets: list[str] = Field(default_factory=list)
    evidence_depth: EvidenceDepth = EvidenceDepth.THIN

    @model_validator(mode="before")
    @classmethod
    def hydrate_routing_compatibility(cls, value: Any) -> Any:
        """Read both v2 ``themes`` and v3 evidence-domain records.

        ``themes`` remains a serialized compatibility copy during the v3
        transition.  Keeping both directions here also makes old exhibitions
        and newly imported collection objects interchangeable.
        """
        if not isinstance(value, dict):
            return value
        hydrated = dict(value)
        domains = hydrated.get("evidenceDomainIds") or hydrated.get(
            "evidence_domain_ids"
        )
        themes = hydrated.get("themes")
        if domains and not themes:
            hydrated["themes"] = domains
        elif themes and not domains:
            hydrated["evidenceDomainIds"] = themes
        return hydrated

    @property
    def routing_domain_ids(self) -> list[str]:
        return self.evidence_domain_ids or self.themes

    @property
    def supports_core_evidence(self) -> bool:
        return self.evidence_depth == EvidenceDepth.FULL.value


class ObjectSummary(ApiModel):
    id: str
    title: str
    image_url: str
    object_url: str
    rights: str
    rights_uri: str | None = None
    image_license: str | None = None
    image_rights_uri: str | None = None
    metadata_license: str | None = None
    metadata_rights_uri: str | None = None
    curatorial_text_license: str | None = None
    curatorial_text_rights_uri: str | None = None
    alt_text: str | None = None
    date: str | None = None
    medium: str | None = None
    # Alternatives are query-scoped retrieval results, not arbitrary catalogue
    # neighbours.  Keep the retrieval trace so an editor can audit why a
    # replacement was offered and the backend can preserve the relevant source
    # excerpt if that replacement is accepted.
    retrieval_score: float | None = Field(default=None, ge=0)
    matched_anchor_terms: list[str] = Field(default_factory=list)
    matched_evidence_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_object(cls, obj: MuseumObject) -> "ObjectSummary":
        return cls(
            id=obj.id,
            title=obj.title,
            image_url=obj.image_url,
            object_url=obj.object_url,
            rights=obj.rights,
            rights_uri=obj.rights_uri,
            image_license=obj.image_license,
            image_rights_uri=obj.image_rights_uri,
            metadata_license=obj.metadata_license,
            metadata_rights_uri=obj.metadata_rights_uri,
            curatorial_text_license=obj.curatorial_text_license,
            curatorial_text_rights_uri=obj.curatorial_text_rights_uri,
            alt_text=obj.alt_text,
            date=obj.date,
            medium=obj.medium,
        )


# The language an exhibition is written in. It is fixed when the exhibition is
# generated, not when it is read: labels and curatorial prose are produced once
# and stored, so a reader who switches language later gets a translated
# interface around prose that stays in the language it was written in.
SiteLanguage = Literal["zh", "en"]


class AgendaInput(ApiModel):
    question: str = Field(min_length=3, max_length=500)
    prior_knowledge: str = Field(min_length=1, max_length=40)
    duration_minutes: Literal[5, 10, 15]
    personal_connection: str | None = Field(default=None, max_length=500)
    excluded_topics: list[str] = Field(default_factory=list, max_length=20)
    collection_id: str | None = None
    language: SiteLanguage = "zh"

    @field_validator("question", "prior_knowledge")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("excluded_topics")
    @classmethod
    def normalize_excluded_topics(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]


class CoverageSummary(ApiModel):
    collection_id: str
    collection_name: str
    eligible_object_count: int
    matched_object_count: int
    evidence_domain_ids: list[str] = Field(default_factory=list)
    candidate_object_ids: list[str] = Field(default_factory=list)


class AgendaCheckResponse(ApiModel):
    status: AnswerabilityStatus
    can_generate: bool
    exhibition_theme: str = Field(min_length=1, max_length=120)
    answerable_part: str | None = None
    gaps: list[str] = Field(default_factory=list)
    supported_aspects: list[str] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)
    evidence_roles: list[str] = Field(default_factory=list)
    collection_version: str
    recommended_questions: list[str] = Field(default_factory=list)
    coverage: CoverageSummary


class LabelSentence(ApiModel):
    id: str
    text: str = Field(min_length=1)
    type: SentenceType
    evidence_ids: list[str] = Field(min_length=1)


class ExhibitionItem(ApiModel):
    id: str
    object: MuseumObject
    role: CuratorialRole
    role_label: str
    # Chinese title shown in the hall. Institutions catalogue in English, but
    # the exhibition is read in Chinese, so this carries either the object's own
    # Chinese title or a model translation. The English original stays on the
    # object record and in the source panel.
    display_title: str = ""
    sub_question: str
    why_selected: str
    relation: str
    label_sentences: list[LabelSentence] = Field(min_length=1)
    alternatives: list[ObjectSummary] = Field(default_factory=list)
    order: int = Field(ge=0)


class VersionInfo(ApiModel):
    model: str
    labels_model: str | None = None
    provider: str
    prompt: str = "p0-2026-08-06"
    collection: str
    validator: str = "p0-1"


class ValidationIssue(ApiModel):
    code: str
    message: str
    item_id: str | None = None
    sentence_id: str | None = None


class ValidationCheck(ApiModel):
    key: str
    label: str
    passed: bool
    detail: str


class ValidationResult(ApiModel):
    passed: bool
    checked_at: datetime = Field(default_factory=utc_now)
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    checks: list[ValidationCheck] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)


class Revision(ApiModel):
    id: str
    kind: str
    before: dict[str, Any]
    after: dict[str, Any]
    created_at: datetime = Field(default_factory=utc_now)


class ReviewRecord(ApiModel):
    decision: Literal["approved", "rejected"]
    reviewer: str
    note: str | None = None
    evidence_review_confirmed: bool = False
    reviewed_at: datetime = Field(default_factory=utc_now)


class ExhibitionPoster(ApiModel):
    status: Literal["idle", "generating", "ready", "failed"] = "idle"
    background_url: str | None = None
    alt_text: str | None = None
    prompt_summary: str | None = None
    provider: str | None = None
    model: str | None = None
    generated_by: str | None = None
    size: str | None = None
    generated_at: datetime | None = None
    is_ai_generated: Literal[True] = True
    error_code: str | None = None


class VisitorMotivation(str, Enum):
    """Falk (2009) identity-related visit motivations, reduced to four.

    Drives narrative register and information density, not object selection.
    """

    EXPLORER = "explorer"          # 有个问题想弄明白
    RECHARGER = "recharger"        # 随便逛逛，放松
    FACILITATOR = "facilitator"    # 带着人来
    PROFESSIONAL = "professional"  # 有基础，想看深的


class VisitorPace(str, Enum):
    """Véron & Levasseur (1983) circulation styles.

    Derived from the chosen duration; drives hall size and camera pacing.
    """

    GRASSHOPPER = "grasshopper"  # 跳跃，只看少数亮点
    BUTTERFLY = "butterfly"      # 频繁改变方向，大部分展品都停
    ANT = "ant"                  # 慢而近，几乎每件都看


MOTIVATION_LABELS: dict[str, str] = {
    VisitorMotivation.EXPLORER.value: "有个问题想弄明白",
    VisitorMotivation.RECHARGER.value: "随便逛逛，放松一下",
    VisitorMotivation.FACILITATOR.value: "我带着人来",
    VisitorMotivation.PROFESSIONAL.value: "有点基础，想看深的",
}

# duration -> (item count, chapter count, pace, label character budget)
# Label budgets follow Serrell (1997): visitors read far less than designers
# assume, and a 5-minute visitor will not read a 220-character label.
DURATION_PLAN: dict[int, tuple[int, int, str, int]] = {
    5: (5, 2, VisitorPace.GRASSHOPPER.value, 80),
    10: (8, 3, VisitorPace.BUTTERFLY.value, 140),
    15: (12, 4, VisitorPace.ANT.value, 220),
}


class VisitorProfile(ApiModel):
    """Structured outcome of the curator interview."""

    curiosity_domain_id: str | None = None
    curiosity_label: str = ""
    free_form_question: str | None = Field(default=None, max_length=500)
    # The one thing the visitor most wants answered. It is collected only when
    # the opening turn did not already contain a visitor-written question.
    open_question: str | None = Field(default=None, max_length=300)
    motivation: VisitorMotivation = VisitorMotivation.EXPLORER
    prior_knowledge: Literal["none", "some", "familiar"] = "none"
    duration_minutes: Literal[5, 10, 15] = 10
    excluded_topics: list[str] = Field(default_factory=list, max_length=20)
    companion: str | None = None
    language: SiteLanguage = "zh"

    @property
    def plan(self) -> tuple[int, int, str, int]:
        return DURATION_PLAN[self.duration_minutes]

    @property
    def item_count(self) -> int:
        return self.plan[0]

    @property
    def chapter_count(self) -> int:
        return self.plan[1]

    @property
    def pace(self) -> str:
        return self.plan[2]

    @property
    def label_max_chars(self) -> int:
        return self.plan[3]

    def to_agenda(self, collection_id: str | None = None) -> "AgendaInput":
        """Bridge to the existing retrieval and answerability path.

        A specific open question outranks the opening topic: it is the sharper
        statement of what the visitor came to find out, and it is what the
        retrieval should be anchored on.
        """
        fallback = (
            "How do these objects relate to one another"
            if self.language == "en"
            else "这些藏品之间有什么关系"
        )
        question = (
            self.open_question
            or self.free_form_question
            or self.curiosity_label
            or fallback
        ).strip()
        return AgendaInput(
            question=question,
            prior_knowledge=self.prior_knowledge,
            duration_minutes=self.duration_minutes,
            excluded_topics=self.excluded_topics,
            collection_id=collection_id,
            language=self.language,
        )


class Chapter(ApiModel):
    id: str
    order: int = Field(ge=0)
    title: str = Field(min_length=1)
    lead_in: str = ""
    item_ids: list[str] = Field(default_factory=list)
    space_hint: str | None = None


class Epilogue(ApiModel):
    text: str = ""
    open_questions: list[str] = Field(default_factory=list)
    material_boundary: list[str] = Field(default_factory=list)


class CuratorialAudience(ApiModel):
    """Non-identifying visitor settings that shape interpretation.

    The private exhibition record keeps these settings so the curatorial
    choices can be audited.  They are deliberately omitted from the public
    projection because a published exhibition must not disclose a visitor's
    interview profile or exclusions.
    """

    motivation: VisitorMotivation
    prior_knowledge: Literal["none", "some", "familiar"]
    duration_minutes: Literal[5, 10, 15]
    excluded_topics: list[str] = Field(default_factory=list, max_length=20)
    voice: str = Field(min_length=1)
    density: str = Field(min_length=1)


class CuratorialClaim(ApiModel):
    """A public-facing interpretive claim and its allowed source anchors."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    confidence: Literal["supported", "provisional", "uncertain"] = "provisional"


class CuratorialObjectDecision(ApiModel):
    """Why one selected object occupies one position in the argument."""

    item_id: str = Field(min_length=1)
    object_id: str = Field(min_length=1)
    role: CuratorialRole
    selection_rationale: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class CuratorialExcludedCandidate(ApiModel):
    """A bounded audit sample of relevant candidates not selected."""

    object_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class CuratorialEthics(ApiModel):
    """Review state, never an inference that silence means clearance."""

    provenance_status: Literal["not_reviewed", "unknown", "partial", "documented"] = (
        "not_reviewed"
    )
    provenance_notes: list[str] = Field(default_factory=list)
    cultural_sensitivity_status: Literal[
        "not_reviewed", "unknown", "no_flags_after_review", "flags_present"
    ] = "not_reviewed"
    cultural_sensitivity: list[str] = Field(default_factory=list)
    # ``None`` means not assessed. False is reserved for an actual review that
    # concluded source-community review is not required.
    community_review_required: bool | None = None
    community_review_status: Literal[
        "not_assessed", "not_required", "required", "completed"
    ] = "not_assessed"
    community_review_notes: list[str] = Field(default_factory=list)


class CuratorialInterpretationPolicy(ApiModel):
    fact_policy: str = Field(min_length=1)
    inference_policy: str = Field(min_length=1)
    uncertainty_policy: str = Field(min_length=1)
    external_knowledge_allowed: Literal[False] = False


class CuratorialEvaluationTarget(ApiModel):
    id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    method: Literal["visitor_prompt", "comprehension_check", "expert_review"]


class CuratorialRetrievalRecord(ApiModel):
    method: str = Field(min_length=1)
    version: str = Field(min_length=1)
    candidate_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)


class CuratorialBrief(ApiModel):
    """Versioned intermediate contract between retrieval and public prose."""

    schema_version: Literal["curatorial-brief/v1"] = "curatorial-brief/v1"
    status: Literal["deterministic", "model_refined"] = "deterministic"
    generated_at: datetime = Field(default_factory=utc_now)
    visitor_inquiry: str = Field(min_length=1, max_length=500)
    audience: CuratorialAudience
    big_idea: CuratorialClaim
    key_messages: list[CuratorialClaim] = Field(min_length=1, max_length=5)
    critical_questions: list[str] = Field(min_length=2, max_length=4)
    objects: list[CuratorialObjectDecision] = Field(min_length=1, max_length=12)
    excluded_candidates: list[CuratorialExcludedCandidate] = Field(
        default_factory=list, max_length=5
    )
    ethics: CuratorialEthics = Field(default_factory=CuratorialEthics)
    interpretation_policy: CuratorialInterpretationPolicy
    evaluation_targets: list[CuratorialEvaluationTarget] = Field(
        min_length=1, max_length=6
    )
    retrieval: CuratorialRetrievalRecord


class SpaceDesignSpec(ApiModel):
    """Model-authored parameters consumed by a deterministic 3D renderer.

    The model never emits code (01b decision A). Every field is clamped
    server-side before it reaches the client.
    """

    space_form: Literal["hall", "cloister", "corridor", "pavilion"] = "hall"
    wall_color: str = "#e8e3da"
    floor_color: str = "#4a4038"
    accent_color: str = "#8c6b3f"
    ceiling_color: str = "#f2efe9"
    light_temperature: int = Field(default=3400, ge=2200, le=6500)
    light_intensity: float = Field(default=0.85, ge=0.25, le=1.6)
    mood: Literal["warm_dim", "neutral", "cool_bright", "dramatic"] = "warm_dim"
    frame_style: Literal["thin_dark", "wide_gold", "scroll_hanging", "vitrine"] = "thin_dark"
    pacing: VisitorPace = VisitorPace.BUTTERFLY
    typography: Literal["serif", "song", "sans"] = "serif"

    @field_validator("wall_color", "floor_color", "accent_color", "ceiling_color")
    @classmethod
    def validate_hex(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
            raise ValueError("must be a #rrggbb colour")
        return value.lower()


class Exhibition(ApiModel):
    id: str
    slug: str | None = None
    title: str
    subtitle: str = ""
    question: str
    exhibition_theme: str = Field(min_length=1, max_length=120)
    curatorial_thesis: str
    core_answer: str
    sub_questions: list[str] = Field(min_length=2, max_length=4)
    items: list[ExhibitionItem]
    chapters: list[Chapter] = Field(default_factory=list)
    epilogue: Epilogue = Field(default_factory=Epilogue)
    # Optional only so pre-v1 stored exhibitions remain readable. Every new
    # profile-driven generation creates this before the first model call.
    curatorial_brief: CuratorialBrief | None = None
    space_design: SpaceDesignSpec = Field(default_factory=SpaceDesignSpec)
    visitor_profile: VisitorProfile | None = None
    coverage_limits: list[str] = Field(default_factory=list)
    status: ExhibitionStatus = ExhibitionStatus.DRAFT
    versions: VersionInfo
    validation: ValidationResult | None = None
    agenda: AgendaInput
    collection_id: str
    evidence_domain_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "evidenceDomainId",
            "evidence_domain_id",
            "themeId",
            "theme_id",
        ),
        serialization_alias="evidenceDomainId",
    )
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    revisions: list[Revision] = Field(default_factory=list)
    review: ReviewRecord | None = None
    poster: ExhibitionPoster | None = None

    @model_validator(mode="before")
    @classmethod
    def hydrate_legacy_exhibition_theme(cls, value: Any) -> Any:
        """Keep pre-field local demo records readable after the schema upgrade."""
        if not isinstance(value, dict):
            return value
        if value.get("exhibitionTheme") or value.get("exhibition_theme"):
            return value
        question = str(value.get("question", "")).strip()
        if not question:
            return value
        hydrated = dict(value)
        hydrated["exhibitionTheme"] = question.rstrip("？?。.!！") or question
        return hydrated


class PublicEvidenceChunk(ApiModel):
    id: str
    text: str
    source_title: str
    source_url: str
    source_location: str
    supports: str
    review_status: str
    license: str | None = None
    rights_uri: str | None = None
    source_kind: str | None = None


class PublicMuseumObject(ApiModel):
    id: str
    accession_number: str
    title: str
    title_original: str | None = None
    date: str
    maker: str | None = None
    culture: str | None = None
    medium: str
    type: str
    image_url: str
    object_url: str
    rights: str
    rights_uri: str | None = None
    image_license: str | None = None
    image_rights_uri: str | None = None
    metadata_license: str | None = None
    metadata_rights_uri: str | None = None
    curatorial_text_license: str | None = None
    curatorial_text_rights_uri: str | None = None
    alt_text: str
    evidence: list[PublicEvidenceChunk]


class PublicExhibitionItem(ApiModel):
    id: str
    object: PublicMuseumObject
    role: CuratorialRole
    role_label: str
    sub_question: str
    why_selected: str
    relation: str
    label_sentences: list[LabelSentence]


class PublicVersionInfo(ApiModel):
    model: str
    labels_model: str | None = None
    prompt: str
    collection: str
    validator: str


class PublicValidationResult(ApiModel):
    passed: bool
    checks: list[ValidationCheck]
    blocking_issues: list[str]
    checked_at: datetime


class PublicCuratorialBrief(ApiModel):
    """Privacy-safe curatorial reasoning exposed on a published share page."""

    schema_version: Literal["curatorial-brief/v1"]
    status: Literal["deterministic", "model_refined"]
    generated_at: datetime
    visitor_inquiry: str
    big_idea: CuratorialClaim
    key_messages: list[CuratorialClaim]
    critical_questions: list[str]
    objects: list[CuratorialObjectDecision]
    ethics: CuratorialEthics
    interpretation_policy: CuratorialInterpretationPolicy
    evaluation_targets: list[CuratorialEvaluationTarget]
    retrieval: CuratorialRetrievalRecord

    @classmethod
    def from_brief(
        cls,
        brief: CuratorialBrief,
        public_item_ids: dict[str, str],
    ) -> "PublicCuratorialBrief":
        return cls(
            schema_version=brief.schema_version,
            status=brief.status,
            generated_at=brief.generated_at,
            visitor_inquiry=brief.visitor_inquiry,
            big_idea=brief.big_idea,
            key_messages=brief.key_messages,
            critical_questions=brief.critical_questions,
            objects=[
                decision.model_copy(
                    update={"item_id": public_item_ids[decision.item_id]}
                )
                for decision in brief.objects
                if decision.item_id in public_item_ids
            ],
            ethics=brief.ethics,
            interpretation_policy=brief.interpretation_policy,
            evaluation_targets=brief.evaluation_targets,
            retrieval=brief.retrieval,
        )


class PublicationReview(ApiModel):
    status: Literal["approved"] = "approved"
    reviewed_at: datetime


class PublicExhibition(ApiModel):
    id: str
    slug: str
    title: str
    question: str
    exhibition_theme: str
    curatorial_thesis: str
    core_answer: str
    sub_questions: list[str]
    curatorial_brief: PublicCuratorialBrief | None = None
    items: list[PublicExhibitionItem]
    coverage_limits: list[str]
    status: Literal["published"] = "published"
    versions: PublicVersionInfo
    validation: PublicValidationResult
    publication_review: PublicationReview
    poster: ExhibitionPoster | None = None
    updated_at: datetime

    @classmethod
    def from_exhibition(cls, exhibition: Exhibition) -> "PublicExhibition":
        if not exhibition.slug or exhibition.status != ExhibitionStatus.PUBLISHED.value:
            raise ValueError("Only a published exhibition can be converted to a public response")
        if not exhibition.review or not exhibition.review.evidence_review_confirmed:
            raise ValueError("A published exhibition must include explicit evidence review confirmation")
        public_items: list[PublicExhibitionItem] = []
        for item_index, item in enumerate(exhibition.items, start=1):
            public_object = PublicMuseumObject(
                id=item.object.id,
                accession_number=item.object.accession_number,
                title=item.object.title,
                title_original=item.object.title_original,
                date=item.object.date,
                maker=item.object.maker or item.object.creator,
                culture=item.object.culture or None,
                medium=item.object.medium or item.object.material or "",
                type=item.object.type,
                image_url=item.object.image_url,
                object_url=item.object.object_url,
                rights=item.object.rights,
                rights_uri=item.object.rights_uri,
                image_license=item.object.image_license,
                image_rights_uri=item.object.image_rights_uri,
                metadata_license=item.object.metadata_license,
                metadata_rights_uri=item.object.metadata_rights_uri,
                curatorial_text_license=item.object.curatorial_text_license,
                curatorial_text_rights_uri=item.object.curatorial_text_rights_uri,
                alt_text=item.object.alt_text,
                evidence=[
                    PublicEvidenceChunk(
                        id=evidence.id,
                        text=evidence.text,
                        source_title=evidence.source_title,
                        source_url=evidence.source_url,
                        source_location=evidence.source_location,
                        supports=evidence.supports,
                        review_status=evidence.review_status,
                        license=evidence.license,
                        rights_uri=evidence.rights_uri,
                        source_kind=evidence.source_kind,
                    )
                    for evidence in item.object.evidence
                ],
            )
            public_items.append(
                PublicExhibitionItem(
                    id=f"item-{item_index}",
                    object=public_object,
                    role=item.role,
                    role_label=item.role_label,
                    sub_question=item.sub_question,
                    why_selected=item.why_selected,
                    relation=item.relation,
                    label_sentences=[
                        LabelSentence(
                            id=f"item-{item_index}-sentence-{sentence_index}",
                            text=sentence.text,
                            type=sentence.type,
                            evidence_ids=sentence.evidence_ids,
                        )
                        for sentence_index, sentence in enumerate(item.label_sentences, start=1)
                    ],
                )
            )
        validation = exhibition.validation or ValidationResult(passed=False)
        public_item_ids = {
            item.id: f"item-{item_index}"
            for item_index, item in enumerate(exhibition.items, start=1)
        }
        return cls(
            id=exhibition.slug,
            slug=exhibition.slug,
            title=exhibition.title,
            question=exhibition.question,
            exhibition_theme=exhibition.exhibition_theme,
            curatorial_thesis=exhibition.curatorial_thesis,
            core_answer=exhibition.core_answer,
            sub_questions=exhibition.sub_questions,
            curatorial_brief=(
                PublicCuratorialBrief.from_brief(
                    exhibition.curatorial_brief, public_item_ids
                )
                if exhibition.curatorial_brief
                else None
            ),
            items=public_items,
            coverage_limits=exhibition.coverage_limits,
            versions=PublicVersionInfo(
                model=exhibition.versions.model,
                labels_model=exhibition.versions.labels_model,
                prompt=exhibition.versions.prompt,
                collection=exhibition.versions.collection,
                validator=exhibition.versions.validator,
            ),
            validation=PublicValidationResult(
                passed=validation.passed,
                checks=validation.checks,
                blocking_issues=validation.blocking_issues,
                checked_at=validation.checked_at,
            ),
            publication_review=PublicationReview(reviewed_at=exhibition.review.reviewed_at),
            poster=(
                exhibition.poster
                if exhibition.poster and exhibition.poster.status == "ready"
                else None
            ),
            updated_at=exhibition.updated_at,
        )


class GenerateExhibitionRequest(ApiModel):
    agenda: AgendaInput


class ItemsPatchRequest(ApiModel):
    operation: Literal["reorder", "replace"] | None = None
    action: Literal["reorder", "replace"] | None = None
    item_ids: list[str] | None = None
    target_item_id: str | None = None
    replacement_object_id: str | None = None
    item_id: str | None = None
    object_id: str | None = None
    direction: Literal["up", "down"] | None = None

    @model_validator(mode="after")
    def normalize_patch_fields(self) -> "ItemsPatchRequest":
        self.operation = self.operation or self.action
        self.target_item_id = self.target_item_id or self.item_id
        self.replacement_object_id = self.replacement_object_id or self.object_id
        if self.operation is None:
            raise ValueError("operation or action is required")
        return self


class FocusPatchRequest(ApiModel):
    focus: str = Field(min_length=3, max_length=500)


class ReviewRequest(ApiModel):
    reviewer: str = Field(default="demo-admin", min_length=1, max_length=100)
    note: str | None = Field(default=None, max_length=1000)
    reason: str | None = Field(default=None, max_length=1000)
    evidence_review_confirmed: bool = False

    @model_validator(mode="after")
    def map_reason_to_note(self) -> "ReviewRequest":
        if not self.note and self.reason:
            self.note = self.reason
        return self


class EpilogueChatTurn(ApiModel):
    """One completed, client-supplied turn from the optional epilogue chat.

    History is deliberately small and is never persisted by the API.  Roles
    are constrained so a client cannot smuggle a second system message into
    the provider conversation.
    """

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    # A current visitor message may be 800 characters; responses can be a
    # little longer.  Keeping history turns at 1,200 avoids rejecting the next
    # request while the overall eight-turn cap still bounds provider input.
    content: str = Field(min_length=1, max_length=1200)

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class EpilogueChatRequest(ApiModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=800)
    history: list[EpilogueChatTurn] = Field(default_factory=list, max_length=8)
    open_question_id: str | None = Field(default=None, max_length=80)
    open_question_text: str | None = Field(
        default=None,
        max_length=300,
        validation_alias=AliasChoices(
            "openQuestionText",
            "openQuestion",
            "open_question_text",
            "open_question",
        ),
        serialization_alias="openQuestionText",
    )

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("open_question_id", "open_question_text")
    @classmethod
    def strip_optional_anchor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None

    @model_validator(mode="after")
    def validate_completed_history(self) -> "EpilogueChatRequest":
        if len(self.history) % 2:
            raise ValueError("history must contain complete user/assistant pairs")
        for index, turn in enumerate(self.history):
            expected = "user" if index % 2 == 0 else "assistant"
            if turn.role != expected:
                raise ValueError("history must alternate user and assistant turns")
        return self


class EpilogueChatCitation(ApiModel):
    item_id: str
    label: str
    object_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list, max_length=8)


class EpilogueChatResponse(ApiModel):
    answer: str
    citations: list[EpilogueChatCitation] = Field(default_factory=list, max_length=5)
    suggested_prompts: list[str] = Field(min_length=2, max_length=2)
    mode: Literal["deepseek", "local_fallback"]
    notice: str | None = None
    open_question_id: str | None = None
    anchored_question: str | None = None


class EventName(str, Enum):
    AGENDA_SUBMITTED = "agenda_submitted"
    ANSWERABILITY_FAILED = "answerability_failed"
    GENERATION_STARTED = "generation_started"
    GENERATION_COMPLETED = "generation_completed"
    GENERATION_FAILED = "generation_failed"
    POSTER_GENERATION_STARTED = "poster_generation_started"
    POSTER_GENERATION_COMPLETED = "poster_generation_completed"
    POSTER_GENERATION_FAILED = "poster_generation_failed"
    OBJECT_VIEWED = "object_viewed"
    SOURCE_OPENED = "source_opened"
    WHY_SELECTED_OPENED = "why_selected_opened"
    OBJECT_REPLACED = "object_replaced"
    ITEM_REORDERED = "item_reordered"
    FOCUS_CHANGED = "focus_changed"
    VALIDATION_FAILED = "validation_failed"
    VALIDATION_PASSED = "validation_passed"
    EXHIBITION_COMPLETED = "exhibition_completed"
    PUBLISHED = "published"
    SHARED = "shared"
    FEEDBACK_SUBMITTED = "feedback_submitted"
    # Hall navigation. These are product-behaviour descriptors for the
    # visit-depth measures in 01b §10.1 — never evidence of learning.
    HALL_ENTERED = "hall_entered"
    HALL_EXITED = "hall_exited"
    HALL_FALLBACK_2D = "hall_fallback_2d"
    CHAPTER_ENTERED = "chapter_entered"
    EPILOGUE_REACHED = "epilogue_reached"
    FREE_WALK_ENTERED = "free_walk_entered"
    INSTITUTION_PAGE_OPENED = "institution_page_opened"
    AUDIO_GUIDE_ENABLED = "audio_guide_enabled"
    AUDIO_GUIDE_DISABLED = "audio_guide_disabled"
    EPILOGUE_CHAT_OPENED = "epilogue_chat_opened"
    EPILOGUE_CHAT_MESSAGE_SENT = "epilogue_chat_message_sent"
    EPILOGUE_CHAT_CLOSED = "epilogue_chat_closed"


class EventCreate(ApiModel):
    session_id: str | None = Field(default=None, min_length=6, max_length=128)
    event: EventName
    exhibition_id: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class EventLog(EventCreate):
    id: str
    created_at: datetime = Field(default_factory=utc_now)


class EventAccepted(ApiModel):
    accepted: bool = True
    id: str
    session_id: str
    event: EventName
    created_at: datetime


class AnalyticsResponse(ApiModel):
    demo_only: bool = True
    event_counts: dict[str, int]
    exhibition_status_counts: dict[str, int]
    total_events: int
    total_exhibitions: int
    published_exhibitions: int
    collection_objects: int
    reviewed_objects: int
    exhibitions: int
    published: int
    events: dict[str, int]
    note: str = "描述性产品行为统计，不代表学习成效。"


class ExhibitionListResponse(ApiModel):
    items: list[Exhibition]
    total: int


class ErrorDetail(ApiModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Curator interview
# --------------------------------------------------------------------------


class InterviewQuestionId(str, Enum):
    CURIOSITY = "curiosity"
    MOTIVATION = "motivation"
    CUSTOM_QUESTION = "custom_question"
    PRIOR_KNOWLEDGE = "prior_knowledge"
    DURATION = "duration"
    NEGOTIATION = "negotiation"
    OPEN_QUESTION = "open_question"
    EXCLUSIONS = "exclusions"


class InterviewOption(ApiModel):
    value: str
    label: str
    hint: str | None = None


class InterviewQuestion(ApiModel):
    id: InterviewQuestionId
    prompt: str
    options: list[InterviewOption] = Field(default_factory=list)
    allow_free_text: bool = False
    free_text_placeholder: str | None = None
    skippable: bool = False
    multi_select: bool = False
    # 1-based position out of `total_steps`, for the chat progress affordance.
    step: int = 1
    total_steps: int = 5


class InterviewTurn(ApiModel):
    question_id: InterviewQuestionId
    prompt: str
    answer_value: str | None = None
    answer_label: str | None = None
    free_text: str | None = None
    skipped: bool = False
    answered_at: datetime = Field(default_factory=utc_now)
    # What the curator said back before moving on. Absent when the model was
    # unavailable and the deterministic path had nothing worth adding.
    curator_reply: str | None = Field(default=None, max_length=300)


class InterviewAnswer(ApiModel):
    question_id: InterviewQuestionId
    value: str | None = None
    free_text: str | None = Field(default=None, max_length=500)
    skipped: bool = False


class InterviewState(ApiModel):
    id: str
    collection_id: str | None = None
    complete: bool = False
    profile: VisitorProfile = Field(default_factory=VisitorProfile)
    transcript: list[InterviewTurn] = Field(default_factory=list)
    next_question: InterviewQuestion | None = None
    # Set when the corpus cannot fully answer a free-form question, so the agent
    # negotiates in conversation instead of refusing (01b decision C).
    negotiation_note: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


# --------------------------------------------------------------------------
# Curation pipeline jobs
# --------------------------------------------------------------------------


class JobStepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class JobStep(ApiModel):
    key: str
    title: str
    detail: str
    status: JobStepStatus = JobStepStatus.PENDING
    # One concrete sentence about what this step actually found, shown to the
    # visitor when the step completes. This is the point of the visible list.
    finding: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class GenerationJob(ApiModel):
    id: str
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    progress: int = Field(default=0, ge=0, le=100)
    stage: str = ""
    steps: list[JobStep] = Field(default_factory=list)
    exhibition_id: str | None = None
    # Stable, machine-readable failure category. ``error`` remains the
    # visitor-facing Chinese recovery message.
    error_code: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class GenerateFromProfileRequest(ApiModel):
    interview_id: str | None = None
    profile: VisitorProfile | None = None
    collection_id: str | None = None
