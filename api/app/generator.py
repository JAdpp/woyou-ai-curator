from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Awaitable, Callable
from uuid import uuid4

from . import curation
from .collections import (
    BM25_RETRIEVAL_METHOD,
    BM25_RETRIEVAL_VERSION,
    HYBRID_RETRIEVAL_METHOD,
    HYBRID_RETRIEVAL_VERSION,
    CollectionDataError,
    CollectionRepository,
    LoadedCollection,
    QuestionPolicy,
    SearchResult,
    _requests_cross_cultural,
)
from .config import Settings
from .images import ImageCache, ImageFetchError
from .models import (
    AgendaCheckResponse,
    AgendaInput,
    AnswerabilityStatus,
    CoverageSummary,
    CuratorialRole,
    EvidenceDepth,
    Exhibition,
    ExhibitionItem,
    ExhibitionStatus,
    LabelSentence,
    MuseumObject,
    ObjectSummary,
    ROLE_LABELS,
    Revision,
    SentenceType,
    VersionInfo,
    VisitorProfile,
    utc_now,
)
from .providers.deepseek import DeepSeekProvider, ProviderError, VisionImage
from .validator import REQUIRED_ROLES, validate_exhibition


logger = logging.getLogger(__name__)

# (step_key, human-readable finding) -> awaited by the job runner
StepEmitter = Callable[[str, str], Awaitable[None]]


@dataclass(frozen=True)
class GenerationContext:
    collection: LoadedCollection
    results: list[SearchResult]
    selected: list[MuseumObject]
    remaining_results: list[SearchResult]
    question_card_limits: list[str]
    evidence_domain_id: str | None
    policy: QuestionPolicy | None
    exhibition_theme: str


ROLE_ORDER = [
    CuratorialRole.INTRODUCTION,
    CuratorialRole.HISTORICAL_CONTEXT,
    CuratorialRole.CORE_EVIDENCE,
    CuratorialRole.COUNTERPOINT,
    CuratorialRole.SYNTHESIS,
]


UNSUPPORTED_BOUNDARY_PATTERNS = (
    r"拍卖|市场价|估价|价格",
    r"治疗|疗效|幸福感|焦虑",
    r"生成.*(?:古代|仿古).*(?:文物|藏品|图片)",
    r"(?:CMA|克利夫兰).*(?:故宫|大英博物馆).*(?:优劣|排名|全部)",
    r"王朝衰亡.*完全|完全.*王朝衰亡",
)


PARTIAL_SCOPE_PATTERNS = (
    r"所有|全部|一定",
    r"直接导致|完全由|唯一原因",
    r"精确复原|完整复原",
    r"真实劳动条件",
    r"全部政治和经济原因",
)


@dataclass(frozen=True)
class CulturalCoverageObligation:
    """A named culture/place the visitor explicitly asks the comparison to include.

    This is deliberately a small reviewed vocabulary rather than free model
    extraction.  Only controlled catalogue origin fields may satisfy a named
    obligation; a description that merely mentions another place is not enough
    to claim that the exhibition contains an object from that culture.
    """

    label_zh: str
    question_patterns: tuple[str, ...]
    origin_patterns: tuple[str, ...] = ()
    culture_pack_ids: tuple[str, ...] = ()


CULTURAL_COVERAGE_OBLIGATIONS: tuple[CulturalCoverageObligation, ...] = (
    CulturalCoverageObligation("中国", (r"中国|中华|\bchinese\b|\bchina\b",), (r"\bchina\b|\bchinese\b",)),
    CulturalCoverageObligation("日本", (r"日本|\bjapan(?:ese)?\b",), (r"\bjapan(?:ese)?\b",)),
    CulturalCoverageObligation("韩国／朝鲜", (r"韩国|朝鲜|\bkorea(?:n)?\b",), (r"\bkorea(?:n)?\b",)),
    CulturalCoverageObligation("越南", (r"越南|\bvietnam(?:ese)?\b",), (r"\bvietnam(?:ese)?\b",)),
    CulturalCoverageObligation("印度", (r"印度|\bindia(?:n)?\b",), (r"\bindia(?:n)?\b",)),
    CulturalCoverageObligation("伊朗／波斯", (r"伊朗|波斯|\biran(?:ian)?\b|\bpersia(?:n)?\b",), (r"\biran(?:ian)?\b|\bpersia(?:n)?\b",)),
    CulturalCoverageObligation("代尔夫特", (r"代尔夫特|\bdelft\b",), (r"\bdelft\b",)),
    CulturalCoverageObligation("荷兰", (r"荷兰|尼德兰|\bnetherlands\b|\bdutch\b|\bholland\b",), (r"\bnetherlands\b|\bdutch\b|\bholland\b",)),
    CulturalCoverageObligation("埃及", (r"埃及|\begypt(?:ian)?\b",), (r"\begypt(?:ian)?\b",)),
    CulturalCoverageObligation("希腊", (r"希腊|\bgree(?:ce|k)\b",), (r"\bgree(?:ce|k)\b",)),
    CulturalCoverageObligation("罗马", (r"罗马|\broman\b|\brome\b",), (r"\broman\b|\brome\b",)),
    CulturalCoverageObligation("墨西哥", (r"墨西哥|\bmexic(?:o|an)\b",), (r"\bmexic(?:o|an)\b",)),
    CulturalCoverageObligation("秘鲁", (r"秘鲁|\bperu(?:vian)?\b",), (r"\bperu(?:vian)?\b",)),
    CulturalCoverageObligation("非洲", (r"非洲|\bafrica(?:n)?\b",), culture_pack_ids=("africa",)),
    CulturalCoverageObligation("美洲", (r"美洲|\bamericas?\b",), culture_pack_ids=("americas",)),
    CulturalCoverageObligation("欧洲", (r"欧洲|\beurope(?:an)?\b",), culture_pack_ids=("europe",)),
    CulturalCoverageObligation("东亚", (r"东亚|\beast asia(?:n)?\b",), culture_pack_ids=("east_asia",)),
    CulturalCoverageObligation("东南亚", (r"东南亚|\bsoutheast asia(?:n)?\b",), culture_pack_ids=("southeast_asia",)),
    CulturalCoverageObligation("西亚／北非", (r"西亚|中东|北非|\bwest asia(?:n)?\b|\bmiddle east(?:ern)?\b|\bnorth africa(?:n)?\b",), culture_pack_ids=("west_asia_north_africa",)),
    CulturalCoverageObligation("大洋洲／太平洋", (r"大洋洲|太平洋文化|\boceania(?:n)?\b|\bpacific island",), culture_pack_ids=("oceania",)),
)


class ExhibitionGenerator:
    def __init__(
        self,
        settings: Settings,
        collections: CollectionRepository,
        provider: DeepSeekProvider | None = None,
        image_cache: ImageCache | None = None,
    ) -> None:
        self.settings = settings
        self.collections = collections
        self.provider = provider or DeepSeekProvider(settings)
        self.image_cache = image_cache

    async def _generate_model_json(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        *,
        stage: str,
        timeout_seconds: float,
        vision_images: list[VisionImage] | None = None,
    ) -> dict[str, Any]:
        """Run one model stage under a real wall-clock budget.

        Provider-level HTTP timeouts are retained as a second boundary.  The
        stage deadline is deliberately shorter for labels, which are optional
        refinements over a complete deterministic exhibition.
        """

        started = perf_counter()
        outcome = "failed"
        try:
            if vision_images is not None:
                visual_generate = getattr(
                    self.provider, "generate_json_with_images", None
                )
                operation = (
                    visual_generate(system_prompt, user_payload, vision_images)
                    if callable(visual_generate)
                    else self.provider.generate_json(system_prompt, user_payload)
                )
            else:
                operation = self.provider.generate_json(system_prompt, user_payload)
            result = await asyncio.wait_for(
                operation,
                timeout=timeout_seconds,
            )
            outcome = "completed"
            return result
        except asyncio.TimeoutError as exc:
            outcome = "timed_out"
            raise ProviderError(
                f"model stage {stage} exceeded its wall-clock timeout"
            ) from exc
        finally:
            logger.info(
                "curation model stage=%s outcome=%s elapsed=%.3fs budget=%.1fs",
                stage,
                outcome,
                perf_counter() - started,
                timeout_seconds,
            )

    @staticmethod
    def _retrieval_contract(results: list[SearchResult]) -> tuple[str, str]:
        """Record the retrieval route that actually supplied this candidate set."""

        sources = {
            source for result in results for source in result.retrieval_sources
        }
        if any(source.startswith("dense") for source in sources) or "evidence_rerank" in sources:
            return HYBRID_RETRIEVAL_METHOD, HYBRID_RETRIEVAL_VERSION
        return BM25_RETRIEVAL_METHOD, BM25_RETRIEVAL_VERSION

    @staticmethod
    def probe_answerability(
        collections: CollectionRepository, agenda: AgendaInput
    ) -> AgendaCheckResponse:
        """Run the gate without constructing a generator.

        Used by the interview to decide whether to open a negotiation turn, so
        it must not require a model provider or any settings.
        """
        probe = ExhibitionGenerator.__new__(ExhibitionGenerator)
        probe.collections = collections
        probe.provider = None  # type: ignore[assignment]
        probe.settings = None  # type: ignore[assignment]
        return ExhibitionGenerator.check_agenda(probe, agenda)

    def check_agenda(self, agenda: AgendaInput) -> AgendaCheckResponse:
        collection = self.collections.get(agenda.collection_id)
        results = self.collections.search(agenda, collection)
        eligible_objects = self.collections.eligible_objects(collection)
        eligible_count = len(eligible_objects)
        policy_match = self.collections.match_question_policy(collection, agenda.question)
        policy = policy_match[0] if policy_match else None
        direct_results = self._direct_results(results)
        # A free-form subject is already hard-gated by retrieval.  Forcing it
        # into whichever broad routing domain happens to dominate (often
        # “materials” for ceramic animals) silently drops relevant cultures.
        evidence_domain_id = policy.evidence_domain_id if policy else None
        domain_direct = self._filter_evidence_domain(direct_results, evidence_domain_id)
        matched = domain_direct or direct_results
        exhibition_theme = self._exhibition_theme(agenda.question, policy_match)
        policy_objects: list[MuseumObject] = []
        if policy and policy.starter_object_ids:
            eligible_by_id = {obj.id: obj for obj in eligible_objects}
            policy_objects = [
                eligible_by_id[object_id]
                for object_id in policy.starter_object_ids
                if object_id in eligible_by_id
            ]

        # Reviewed question cards/regression policies are a frozen retrieval
        # route, so evaluate them before the free-query hit-count gate.  Their
        # starter IDs remain valid even when catalogue wording shares no token
        # with the Chinese question.
        if policy:
            status = AnswerabilityStatus(policy.status)
            gaps = list(policy.coverage_limits)
            if policy.rationale and status != AnswerabilityStatus.SUPPORTED:
                gaps.insert(0, policy.rationale)
            if status == AnswerabilityStatus.SUPPORTED and policy.starter_object_ids:
                if len(policy_objects) != 5:
                    status = AnswerabilityStatus.PARTIALLY_SUPPORTED
                    gaps.insert(0, "预验证问题卡的五件角色藏品不完整，当前禁止生成。")
                matched = [
                    SearchResult(obj=obj, score=float("inf"))
                    for obj in policy_objects
                ]
            elif status == AnswerabilityStatus.SUPPORTED and policy.evidence_domain_id:
                result_by_id = {result.obj.id: result for result in results}
                domain_pool = [
                    result_by_id.get(obj.id, SearchResult(obj=obj, score=1.0))
                    for obj in eligible_objects
                    if policy.evidence_domain_id in obj.themes
                ]
                if len(domain_pool) < 5:
                    status = AnswerabilityStatus.PARTIALLY_SUPPORTED
                    gaps.insert(0, "预验证证据域当前不足五件可用藏品，当前禁止生成。")
                matched = domain_pool
            if not gaps and status == AnswerabilityStatus.PARTIALLY_SUPPORTED:
                gaps = ["馆藏只能支持问题的局部方面，不能形成五角色证据链。"]
            if not gaps and status == AnswerabilityStatus.UNSUPPORTED:
                gaps = ["该问题超出单一馆藏和当前产品边界。"]
        elif len(results) < 5:
            status = AnswerabilityStatus.UNSUPPORTED
            gaps = ["与问题主题直接匹配的可用藏品不足五件；系统不会用无关藏品补位。"]
        elif any(re.search(pattern, agenda.question, re.IGNORECASE) for pattern in UNSUPPORTED_BOUNDARY_PATTERNS):
            status = AnswerabilityStatus.UNSUPPORTED
            gaps = ["该问题需要馆藏之外的数据、效果证据或产品明确禁止的能力。"]
        elif any(re.search(pattern, agenda.question, re.IGNORECASE) for pattern in PARTIAL_SCOPE_PATTERNS):
            status = (
                AnswerabilityStatus.PARTIALLY_SUPPORTED
                if matched
                else AnswerabilityStatus.UNSUPPORTED
            )
            gaps = ["馆藏可支持局部对象比较，但不能支持总体化、排他性或完整因果结论。"]
        elif len(matched) >= 5:
            status = AnswerabilityStatus.SUPPORTED
            gaps = []
        elif len(matched) >= 1:
            status = AnswerabilityStatus.PARTIALLY_SUPPORTED
            gaps = [f"当前只有 {len(matched)} 件藏品与问题形成较强的直接匹配，少于五件。"]
        else:
            status = AnswerabilityStatus.UNSUPPORTED
            gaps = ["馆藏元数据与证据片段不足以支持这个问题，系统不会用模型常识补全。"]

        coverage_obligations = self._cultural_coverage_obligations(agenda.question)
        missing_obligations = [
            obligation
            for obligation in coverage_obligations
            if not any(
                self._object_satisfies_cultural_obligation(result.obj, obligation)
                for result in matched
            )
        ]
        if missing_obligations:
            missing_labels = "、".join(
                f"“{obligation.label_zh}”" for obligation in missing_obligations
            )
            coverage_gap = (
                f"当前直接证据候选缺少{missing_labels}的馆藏对象，"
                "无法按原问题完成这组跨文化比较。"
            )
            if status == AnswerabilityStatus.SUPPORTED:
                status = AnswerabilityStatus.PARTIALLY_SUPPORTED
            if coverage_gap not in gaps:
                gaps.insert(0, coverage_gap)

        evidence_domain_ids: list[str] = []
        for result in matched[:10]:
            for domain_id in result.obj.themes:
                if domain_id not in evidence_domain_ids:
                    evidence_domain_ids.append(domain_id)
        if (
            policy
            and policy.evidence_domain_id
            and policy.evidence_domain_id not in evidence_domain_ids
        ):
            evidence_domain_ids.insert(0, policy.evidence_domain_id)
        supported_aspects = self._supported_aspects(exhibition_theme, matched)
        if coverage_obligations:
            covered_labels = [
                obligation.label_zh
                for obligation in coverage_obligations
                if obligation not in missing_obligations
            ]
            if covered_labels:
                supported_aspects.append("已覆盖文化／地点：" + "、".join(covered_labels))
        can_generate = status == AnswerabilityStatus.SUPPORTED
        return AgendaCheckResponse(
            status=status,
            can_generate=can_generate,
            exhibition_theme=exhibition_theme,
            answerable_part=(
                "可用馆藏能够组成五件藏品的证据链。"
                if can_generate
                else ("馆藏只能支持问题的局部方面。" if matched else None)
            ),
            gaps=gaps,
            supported_aspects=supported_aspects,
            coverage_gaps=gaps,
            evidence_roles=[ROLE_LABELS[role.value] for role in ROLE_ORDER],
            collection_version=collection.version,
            recommended_questions=self.collections.recommend_questions(collection),
            coverage=CoverageSummary(
                collection_id=collection.id,
                collection_name=collection.name,
                eligible_object_count=eligible_count,
                matched_object_count=len(matched),
                evidence_domain_ids=evidence_domain_ids[:8],
                candidate_object_ids=[result.obj.id for result in matched[:10]],
            ),
        )

    @staticmethod
    def _direct_results(results: list[SearchResult]) -> list[SearchResult]:
        # The repository gate guarantees topicality; this tier distinguishes a
        # direct evidence chain from a long, weak tail.  It is intentionally
        # relative because BM25 scores vary with corpus size and term rarity.
        positive = [result for result in results if result.score > 0]
        if not positive:
            return []
        threshold = positive[0].score * 0.55
        return [result for result in positive if result.score >= threshold]

    @staticmethod
    def _filter_evidence_domain(
        results: list[SearchResult], evidence_domain_id: str | None
    ) -> list[SearchResult]:
        if not evidence_domain_id:
            return list(results)
        return [
            result for result in results if evidence_domain_id in result.obj.themes
        ]

    @staticmethod
    def _dominant_evidence_domain(results: list[SearchResult]) -> str | None:
        totals: dict[str, float] = {}
        for result in results:
            for domain_id in result.obj.themes:
                totals[domain_id] = totals.get(domain_id, 0.0) + max(result.score, 1.0)
        return max(totals, key=totals.get) if totals else None

    @staticmethod
    def _exhibition_theme(
        question: str,
        policy_match: tuple[QuestionPolicy, float] | None,
    ) -> str:
        """Return a query-specific theme without inventing a broader topic.

        Exact, curated policies reuse their reviewed wording. Near matches and
        free-form questions stay anchored to the visitor's own text.
        """
        source = (
            policy_match[0].question
            if policy_match and policy_match[1] == 1.0
            else question
        )
        compact = re.sub(r"\s+", " ", source).strip().rstrip("？?。.!！")
        if len(compact) > 116:
            compact = compact[:115].rstrip() + "…"
        return compact or "基于当前问题的馆藏证据"

    @staticmethod
    def _supported_aspects(
        exhibition_theme: str, matched: list[SearchResult]
    ) -> list[str]:
        if not matched:
            return []
        aspects = [f"问题焦点：{exhibition_theme}"]
        types = list(
            dict.fromkeys(result.obj.type for result in matched if result.obj.type)
        )
        if types:
            aspects.append("藏品类型：" + "、".join(types[:3]))
        aspects.append(f"可定位机构记录：{len(matched)} 件候选藏品")
        return aspects[:4]

    @staticmethod
    def _cultural_coverage_obligations(
        question: str,
    ) -> list[CulturalCoverageObligation]:
        """Return named comparison legs only when the question names two or more.

        Generic requests such as “各国文化中的狗” intentionally remain under
        the diversity re-ranker.  A question that explicitly names China, Iran
        and Delft, however, must contain evidence for all three or negotiate a
        narrower question before generation.
        """

        obligations = [
            obligation
            for obligation in CULTURAL_COVERAGE_OBLIGATIONS
            if any(
                re.search(pattern, question, re.IGNORECASE)
                for pattern in obligation.question_patterns
            )
        ]
        return obligations if len(obligations) >= 2 else []

    @staticmethod
    def _object_satisfies_cultural_obligation(
        obj: MuseumObject,
        obligation: CulturalCoverageObligation,
    ) -> bool:
        """Match only catalogue origin fields or reviewed broad culture packs."""

        if set(obj.culture_pack_ids) & set(obligation.culture_pack_ids):
            return True
        controlled_origin = " ".join(
            value
            for value in (
                obj.culture,
                obj.culture_display,
                obj.place,
                obj.creator,
                obj.maker,
            )
            if value
        )
        return any(
            re.search(pattern, controlled_origin, re.IGNORECASE)
            for pattern in obligation.origin_patterns
        )

    async def generate(self, agenda: AgendaInput) -> Exhibition:
        check = self.check_agenda(agenda)
        if not check.can_generate:
            raise CollectionDataError(
                "QUESTION_UNSUPPORTED",
                "The current collection cannot support a five-object evidence chain for this question.",
                status=check.status,
                coverageGaps=check.coverage_gaps,
                recommendedQuestions=check.recommended_questions,
            )
        context = self._context(agenda)
        exhibition = self._deterministic_exhibition(agenda, context)
        if self.provider.configured:
            try:
                model_output = await self.provider.generate_json(
                    self._system_prompt(), self._model_payload(agenda, context)
                )
                exhibition = self._apply_model_output(exhibition, model_output)
                exhibition.versions.provider = "deepseek"
            except (ProviderError, ValueError, TypeError, KeyError):
                exhibition.coverage_limits.append(
                    "The model output was unusable; this visit falls back to a "
                    "deterministic template built only from the selected "
                    "collection evidence."
                    if agenda.language == "en"
                    else "模型输出不可用；本次展示采用仅基于已选馆藏证据的确定性中文模板。"
                )
                exhibition.versions.provider = "deterministic_fallback"

        exhibition.validation = validate_exhibition(exhibition)
        exhibition.status = (
            ExhibitionStatus.AUTO_VALIDATED
            if exhibition.validation.passed
            else ExhibitionStatus.DRAFT
        )
        exhibition.updated_at = utc_now()
        return exhibition

    async def generate_from_profile(
        self,
        profile: "VisitorProfile",
        collection_id: str | None = None,
        emit: "StepEmitter | None" = None,
        on_frame_ready: "Callable[[Exhibition], Awaitable[None]] | None" = None,
    ) -> Exhibition:
        """Build a chaptered, walkable exhibition from an interview profile.

        ``emit`` receives ``(step_key, finding)`` as each stage lands so the
        visitor's todo list can report what actually happened rather than a
        synthetic percentage.

        ``on_frame_ready`` fires after the model frame has fixed the public title
        and subtitle, but before the independent per-object label calls. The caller
        can therefore compose the final title into the poster while labels run
        concurrently, without freezing the deterministic skeleton title.
        """

        async def step(key: str, finding: str) -> None:
            if emit is not None:
                await emit(key, finding)

        agenda = profile.to_agenda(collection_id)
        collection = self.collections.get(collection_id)
        # Every finding below is on screen for the whole generation, so it is
        # written in the language the visit was curated in.
        en = profile.language == "en"

        await step(
            "profile",
            (
                f"{profile.duration_minutes} minutes, {profile.item_count} objects, "
                f"one route in {profile.chapter_count} segments."
                if en
                else f"{curation.REGISTER[profile.motivation]['density']}信息密度，"
                f"{profile.duration_minutes} 分钟，{profile.item_count} 件展品，"
                f"连续展线分为 {profile.chapter_count} 个叙事区段。"
            ),
        )

        # -- retrieval ---------------------------------------------------
        all_results = self.collections.search(agenda, collection)
        domain_id = profile.curiosity_domain_id
        pool = self._filter_evidence_domain(all_results, domain_id)
        # A narrow free-text question can starve the pool; widen before failing.
        if len(pool) < profile.item_count:
            pool = self._filter_evidence_domain(all_results, None)
        if len(pool) < profile.item_count:
            raise CollectionDataError(
                "COLLECTION_DATA_INSUFFICIENT",
                "Not enough eligible objects remain after exclusions to fill this exhibition.",
                requestedItemCount=profile.item_count,
                availableObjectCount=len(pool),
            )

        objects = curation.order_for_narrative(
            pool,
            profile.item_count,
            prefer_culture_diversity=_requests_cross_cultural(agenda.question),
        )
        full_depth = sum(
            1 for obj in objects if obj.evidence_depth == EvidenceDepth.FULL.value
        )
        topic = profile.curiosity_label or profile.free_form_question or (
            "these objects" if profile.language == "en" else "这批藏品"
        )
        eligible = len(self.collections.eligible_objects(collection))
        await step(
            "retrieve",
            (
                f"From {eligible} eligible objects, {len(pool)} relate to "
                f"“{topic}”; {len(objects)} selected, {full_depth} of them with "
                "an institution-written description."
                if en
                else f"从 {eligible} 件可用藏品中筛出 "
                f"{len(pool)} 件与「{topic}」相关，选定 {len(objects)} 件，"
                f"其中 {full_depth} 件带机构撰写的说明。"
            ),
        )

        # -- deterministic skeleton --------------------------------------
        exhibition = self._profile_skeleton(
            profile, agenda, collection, objects, domain_id, pool
        )

        # -- model pass ---------------------------------------------------
        # Two calls, not one. The frame is small and returns quickly, so the
        # visitor sees a real title within seconds; the per-object label calls
        # are independent and run concurrently, so total latency is the slowest
        # chapter rather than the sum of all of them.
        model_applied = False
        labelled_items = 0
        if self.provider is not None and self.provider.configured:
            plan = curation.CurationPlan(
                profile=profile,
                objects=objects,
                roles=[item.role for item in exhibition.items],
                chapter_sizes=[len(chapter.item_ids) for chapter in exhibition.chapters],
                evidence_domain_id=domain_id,
                collection_id=collection.id,
                collection_version=collection.version,
                institution=collection.institution,
                candidate_count=len(pool),
                retrieval_method=self._retrieval_contract(pool)[0],
                retrieval_version=self._retrieval_contract(pool)[1],
            )
            try:
                frame = await self._generate_model_json(
                    curation.frame_prompt(exhibition.agenda.language),
                    curation.frame_payload(
                        plan,
                        exhibition.items,
                        exhibition.chapters,
                        exhibition.curatorial_brief,
                    ),
                    stage="frame",
                    timeout_seconds=min(
                        self.settings.deepseek_timeout_seconds,
                        self.settings.deepseek_frame_timeout_seconds,
                    ),
                )
                if not isinstance(frame.get("curatorialBrief"), dict):
                    raise ValueError("frame output missing required curatorialBrief")
                exhibition = curation.apply_frame(exhibition, frame)
                exhibition.versions.provider = "deepseek"
                model_applied = True
            except (ProviderError, ValueError, TypeError, KeyError) as error:
                logger.warning(
                    "curation frame call failed (%s: %s); using deterministic text",
                    type(error).__name__,
                    error,
                )
                exhibition.coverage_limits.append(
                    "本次策展文本由确定性模板生成；模型输出不可用，展品与来源不受影响。"
                )
                exhibition.versions.provider = "deterministic_fallback"

        if on_frame_ready is not None:
            await on_frame_ready(exhibition)

        subtitle_fallback = (
            "one thread through the current collection"
            if en
            else "基于当前馆藏的一条线索"
        )
        await step(
            "theme",
            (
                f"“{exhibition.title}” — {exhibition.subtitle or subtitle_fallback}"
                if en
                else f"《{exhibition.title}》——{exhibition.subtitle or subtitle_fallback}"
            ),
        )
        await step(
            "chapters",
            (", " if en else "、").join(
                (
                    f"{chapter.title} ({len(chapter.item_ids)})"
                    if en
                    else f"{chapter.title}（{len(chapter.item_ids)} 件）"
                )
                for chapter in exhibition.chapters
            ),
        )

        if model_applied:
            labelled_items = await self._write_labels(exhibition, profile)
            if labelled_items:
                exhibition.versions.labels_model = self.settings.deepseek_labels_model

        bound = sum(len(item.label_sentences) for item in exhibition.items)
        named = sum(1 for item in exhibition.items if item.display_title)
        if model_applied and labelled_items:
            detail = (
                f"{labelled_items}/{len(exhibition.items)} objects written by the visual model"
                if en
                else f"{labelled_items}/{len(exhibition.items)} 件由视觉模型撰写"
            )
        elif model_applied:
            detail = (
                "the model returned no labels; deterministic template used"
                if en
                else "模型未返回展签，改用确定性模板"
            )
        else:
            detail = "deterministic template used" if en else "使用确定性模板"
        await step(
            "labels",
            (
                f"{bound} label sentences for {len(exhibition.items)} objects, "
                f"{named} with a display title; {detail}."
                if en
                else f"为 {len(exhibition.items)} 件展品写了 {bound} 条展签，"
                f"{named} 件有中文展品名；{detail}。"
            ),
        )
        await step(
            "epilogue",
            exhibition.epilogue.text[:60] + ("…" if len(exhibition.epilogue.text) > 60 else ""),
        )
        await step(
            "space",
            (
                f"{exhibition.space_design.space_form} plan, one continuous route, "
                f"{len(exhibition.chapters)} segments, accent "
                f"{exhibition.space_design.accent_color}, {exhibition.space_design.mood} lighting."
                if en
                else f"{exhibition.space_design.space_form} 形制，一条连续展线，"
                f"{len(exhibition.chapters)} 个叙事区段，"
                f"主色 {exhibition.space_design.accent_color}，{exhibition.space_design.mood} 光照。"
            ),
        )

        exhibition.validation = validate_exhibition(exhibition)
        exhibition.status = (
            ExhibitionStatus.READY
            if exhibition.validation.passed
            else ExhibitionStatus.DRAFT
        )
        exhibition.updated_at = utc_now()
        return exhibition


    async def _write_labels(
        self, exhibition: Exhibition, profile: "VisitorProfile"
    ) -> int:
        """Write one image-grounded public label per object, concurrently.

        A single-object call prevents a bad image or malformed response from
        collapsing a whole chapter, and removes ambiguity about which pixels
        belong to which object.  Curatorial relations remain in the frame; the
        wall label itself needs only this object's image and catalogue record.
        """
        by_id = {item.id: item for item in exhibition.items}
        prompt = curation.labels_prompt(profile.label_max_chars, profile.language)
        chapter_by_item = {
            item_id: chapter
            for chapter in exhibition.chapters
            for item_id in chapter.item_ids
        }
        visual_provider = callable(
            getattr(self.provider, "generate_json_with_images", None)
        )

        async def prepare_image(item: ExhibitionItem) -> VisionImage | None:
            visual = curation.image_evidence(item.object)
            if not visual_provider or visual is None or self.image_cache is None:
                return None
            try:
                payload, _cached = await asyncio.to_thread(
                    self.image_cache.get,
                    visual.source_url,
                    1024,
                )
                return VisionImage(
                    object_id=item.object.id,
                    evidence_id=visual.id,
                    payload=payload,
                )
            except ImageFetchError as error:
                logger.warning(
                    "visual label image unavailable object=%s code=%s",
                    item.object.id,
                    error.code,
                )
                return None

        async def write(item: ExhibitionItem) -> bool:
            chapter = chapter_by_item.get(item.id)
            if chapter is None:
                return False
            vision_image = await prepare_image(item)
            available_visual_ids = (
                {vision_image.evidence_id} if vision_image is not None else set()
            )
            try:
                payload = curation.labels_payload(
                    profile,
                    chapter,
                    [item],
                    exhibition.curatorial_brief,
                    available_visual_evidence_ids=available_visual_ids,
                )
                allowed_evidence_by_object = {
                    str(raw_item["objectId"]): {
                        str(chunk["id"])
                        for chunk in raw_item.get("evidence", [])
                        if isinstance(chunk, dict) and chunk.get("id")
                    }
                    for raw_item in payload.get("items", [])
                    if isinstance(raw_item, dict) and raw_item.get("objectId")
                }
                visual_evidence_by_object = (
                    {item.object.id: vision_image.evidence_id}
                    if vision_image is not None
                    else {}
                )
                output = await self._generate_model_json(
                    prompt,
                    payload,
                    stage=f"labels:{item.id}",
                    timeout_seconds=min(
                        self.settings.deepseek_timeout_seconds,
                        self.settings.deepseek_labels_timeout_seconds,
                    ),
                    vision_images=(
                        [vision_image]
                        if visual_provider and vision_image is not None
                        else ([] if visual_provider else None)
                    ),
                )
                return curation.apply_labels(
                    [item],
                    output,
                    profile.label_max_chars,
                    allowed_evidence_by_object=allowed_evidence_by_object,
                    visual_evidence_by_object=visual_evidence_by_object,
                ) > 0
            except (ProviderError, ValueError, TypeError, KeyError) as error:
                logger.warning(
                    "visual label for object '%s' failed (%s: %s); keeping catalogue fallback",
                    item.object.id,
                    type(error).__name__,
                    error,
                )
                return False

        results = await asyncio.gather(
            *(write(item) for item in by_id.values())
        )
        return sum(1 for ok in results if ok)

    def _profile_skeleton(
        self,
        profile: "VisitorProfile",
        agenda: AgendaInput,
        collection: LoadedCollection,
        objects: list[MuseumObject],
        domain_id: str | None,
        candidate_results: list[SearchResult],
    ) -> Exhibition:
        """Structurally complete exhibition using only institution text."""
        objects, roles = curation.plan_roles(objects)
        sizes = curation.chapter_sizes(len(objects), profile.chapter_count)
        theme = profile.curiosity_label or self._exhibition_theme(agenda.question, None)

        used = {obj.id for obj in objects}
        # Alternatives must come from the same query-local, hard-gated RAG
        # result set as the selected objects.  Pulling from the whole catalogue
        # here used to leak unrelated records into otherwise grounded exhibits.
        alternatives_pool = self._alternative_summaries(
            candidate_results, used_ids=used, limit=40
        )

        items: list[ExhibitionItem] = []
        sub_questions = self._sub_questions(agenda.question)
        for index, (obj, role) in enumerate(zip(objects, roles, strict=True)):
            exhibition_obj = curation.with_collection_image_evidence(obj)
            item_id = str(uuid4())
            role_label = ROLE_LABELS[role]
            rotated = alternatives_pool[index * 3 :] + alternatives_pool[: index * 3]
            items.append(
                ExhibitionItem(
                    id=item_id,
                    object=exhibition_obj,
                    role=role,
                    role_label=role_label,
                    display_title=curation.display_title(exhibition_obj),
                    sub_question=sub_questions[index % len(sub_questions)],
                    why_selected=(
                        "馆方记录提供了可核对的题名、年代或材料信息，"
                        f"可用来追问：{sub_questions[index % len(sub_questions)]}"
                    ),
                    relation=self._relation(index, role_label),
                    label_sentences=curation.label_sentences(
                        exhibition_obj, item_id, role_label, profile.label_max_chars
                    ),
                    alternatives=rotated[:3],
                    order=index,
                )
            )

        limits = [
            f"本展使用 {collection.institution} 的公开馆藏记录（数据版本 {collection.version}）。",
            "机构英文原文保持原样；中文策展关系属于系统推断，不冒充机构原文。",
            "材料之外的人物关系、年代因果与历史结论不会由模型常识补写。",
        ]
        thin = [obj for obj in objects if obj.evidence_depth == EvidenceDepth.THIN.value]
        if thin:
            limits.append(
                f"其中 {len(thin)} 件来自不提供策展说明字段的机构，只承担背景或对照位置。"
            )

        exhibition = Exhibition(
            id=str(uuid4()),
            title=f"{theme}：一条用馆藏搭的线索",
            subtitle="",
            question=agenda.question,
            exhibition_theme=theme[:116],
            curatorial_thesis=f"围绕“{theme}”，把公开馆藏记录组织成一条可以走完的展线。",
            core_answer=(
                f"这不是关于“{theme}”的唯一说法，而是一条由 {len(objects)} 件馆藏材料支撑、"
                "每句都能回到机构记录的解释路径。"
            ),
            sub_questions=sub_questions,
            items=items,
            chapters=[],
            coverage_limits=limits,
            status=ExhibitionStatus.GENERATING,
            versions=VersionInfo(
                model=self.settings.deepseek_model if self.settings else "deterministic",
                provider="deterministic",
                prompt="v4-vision-public-copy-2026-08-28",
                collection=collection.version,
                validator="p0-2",
            ),
            agenda=agenda,
            visitor_profile=profile,
            collection_id=collection.id,
            evidence_domain_id=domain_id,
            space_design=curation.build_space_design(profile),
        )
        exhibition.chapters = curation.build_chapters(items, sizes, theme)
        exhibition.epilogue = curation.build_epilogue(profile, limits, theme)
        plan = curation.CurationPlan(
            profile=profile,
            objects=[item.object for item in items],
            roles=[item.role for item in items],
            chapter_sizes=[len(chapter.item_ids) for chapter in exhibition.chapters],
            evidence_domain_id=domain_id,
            collection_id=collection.id,
            collection_version=collection.version,
            institution=collection.institution,
            candidate_count=len(candidate_results),
            retrieval_method=self._retrieval_contract(candidate_results)[0],
            retrieval_version=self._retrieval_contract(candidate_results)[1],
        )
        exhibition.curatorial_brief = curation.build_curatorial_brief(
            plan,
            agenda.question,
            exhibition.items,
            exhibition.chapters,
            candidate_results,
        )
        exhibition.curatorial_thesis = exhibition.curatorial_brief.big_idea.text
        exhibition.core_answer = "".join(
            message.text
            if message.text.endswith(("。", "！", "？"))
            else message.text + "。"
            for message in exhibition.curatorial_brief.key_messages
        )
        exhibition.sub_questions = list(
            exhibition.curatorial_brief.critical_questions
        )
        return exhibition

    def _refresh_curatorial_brief(
        self,
        exhibition: Exhibition,
        candidate_results: list[SearchResult],
    ) -> None:
        """Rebuild the deterministic contract after an editor changes the plan."""

        profile = exhibition.visitor_profile
        if profile is None or exhibition.curatorial_brief is None:
            return

        # An edit invalidates the provenance of every model-written selection
        # rationale and adjacency claim, including items that did not move.
        # Rebuild the full decision layer before marking this Brief
        # deterministic; carrying model prose forward would be mixed
        # provenance under a misleading status.
        for index, item in enumerate(exhibition.items):
            item.why_selected = self._why_selected(
                item.role_label, item.sub_question
            )
            item.relation = self._relation(index, item.role_label)

        collection = self.collections.get(exhibition.collection_id)
        retrieval_method, retrieval_version = self._retrieval_contract(candidate_results)
        plan = curation.CurationPlan(
            profile=profile,
            objects=[item.object for item in exhibition.items],
            roles=[item.role for item in exhibition.items],
            chapter_sizes=[len(chapter.item_ids) for chapter in exhibition.chapters],
            evidence_domain_id=exhibition.evidence_domain_id,
            collection_id=collection.id,
            collection_version=collection.version,
            institution=collection.institution,
            candidate_count=len(candidate_results),
            retrieval_method=retrieval_method,
            retrieval_version=retrieval_version,
        )
        exhibition.curatorial_brief = curation.build_curatorial_brief(
            plan,
            exhibition.question,
            exhibition.items,
            exhibition.chapters,
            candidate_results,
        )
        exhibition.curatorial_thesis = exhibition.curatorial_brief.big_idea.text
        exhibition.core_answer = "".join(
            message.text
            if message.text.endswith(("。", "！", "？"))
            else message.text + "。"
            for message in exhibition.curatorial_brief.key_messages
        )
        exhibition.sub_questions = list(
            exhibition.curatorial_brief.critical_questions
        )

    async def refocus(self, exhibition: Exhibition, focus: str) -> Exhibition:
        candidate_agenda = exhibition.agenda.model_copy(update={"question": focus.strip()})
        check = self.check_agenda(candidate_agenda)
        if not check.can_generate:
            raise CollectionDataError(
                "QUESTION_UNSUPPORTED",
                "The revised focus cannot be supported by five objects in this collection.",
                status=check.status,
                coverageGaps=check.coverage_gaps,
                recommendedQuestions=check.recommended_questions,
            )
        revised_context = self._context(candidate_agenda)
        if (
            exhibition.evidence_domain_id
            and revised_context.evidence_domain_id
            and exhibition.evidence_domain_id != revised_context.evidence_domain_id
        ):
            raise CollectionDataError(
                "FOCUS_EVIDENCE_DOMAIN_CHANGE_REQUIRES_NEW_EXHIBITION",
                "Changing to a different corpus evidence domain requires generating a new exhibition.",
                currentEvidenceDomain=exhibition.evidence_domain_id,
                requestedEvidenceDomain=revised_context.evidence_domain_id,
            )

        selected_ids = {item.object.id for item in exhibition.items}
        revised_result_ids = {result.obj.id for result in revised_context.results}
        unsupported_selected_ids = sorted(selected_ids - revised_result_ids)
        if unsupported_selected_ids:
            raise CollectionDataError(
                "FOCUS_REQUIRES_NEW_EXHIBITION",
                "The current selected objects do not all support the revised focus; generate a new exhibition.",
                unsupportedSelectedObjectIds=unsupported_selected_ids,
            )
        before = {"question": exhibition.question, "title": exhibition.title}
        exhibition.question = focus.strip()
        exhibition.agenda = candidate_agenda
        exhibition.evidence_domain_id = (
            revised_context.evidence_domain_id or exhibition.evidence_domain_id
        )
        exhibition.exhibition_theme = revised_context.exhibition_theme
        exhibition.title = self._title(focus)
        exhibition.curatorial_thesis = self._thesis(focus)
        exhibition.core_answer = self._core_answer(focus)
        exhibition.sub_questions = self._sub_questions(focus)
        for index, item in enumerate(exhibition.items):
            item.sub_question = exhibition.sub_questions[index % len(exhibition.sub_questions)]
            item.why_selected = self._why_selected(item.role_label, item.sub_question)
            item.relation = self._relation(index, item.role_label)
            item.label_sentences = self._label_sentences(item.object, item.id, item.role_label, item.sub_question)
        self._refresh_curatorial_brief(exhibition, revised_context.results)
        exhibition.revisions.append(
            Revision(
                id=str(uuid4()),
                kind="focus_changed",
                before=before,
                after={"question": exhibition.question, "title": exhibition.title},
            )
        )
        self._reset_after_edit(exhibition)
        return exhibition

    def replace_item(
        self,
        exhibition: Exhibition,
        target_item_id: str,
        replacement_object_id: str,
    ) -> Exhibition:
        target = next((item for item in exhibition.items if item.id == target_item_id), None)
        if target is None:
            raise CollectionDataError(
                "ITEM_NOT_FOUND", f"Exhibition item '{target_item_id}' was not found."
            )
        if any(item.object.id == replacement_object_id for item in exhibition.items):
            raise CollectionDataError(
                "DUPLICATE_OBJECT", "The replacement object is already used in this exhibition."
            )
        allowed_alternatives = {alternative.id for alternative in target.alternatives}
        if replacement_object_id not in allowed_alternatives:
            raise CollectionDataError(
                "REPLACEMENT_NOT_ALLOWED",
                "The replacement must come from this item's current theme-and-role alternative set.",
                replacementObjectId=replacement_object_id,
                allowedObjectIds=sorted(allowed_alternatives),
            )
        collection = self.collections.get(exhibition.collection_id)
        eligible = {obj.id for obj in self.collections.require_generation_ready(collection)}
        current_results = self.collections.search(exhibition.agenda, collection)
        result_by_id = {result.obj.id: result for result in current_results}
        replacement_result = result_by_id.get(replacement_object_id)
        if replacement_object_id not in eligible:
            raise CollectionDataError(
                "REPLACEMENT_NOT_ELIGIBLE",
                "The replacement object is missing required evidence, image, rights, or institution-link data.",
                replacementObjectId=replacement_object_id,
            )
        if replacement_result is None:
            raise CollectionDataError(
                "REPLACEMENT_NO_LONGER_RELEVANT",
                "The replacement no longer passes the current question's hard relevance gate.",
                replacementObjectId=replacement_object_id,
            )
        replacement = self._prioritized_object(replacement_result)
        previous_object = target.object
        previous_alternatives = list(target.alternatives)
        before = {"itemId": target.id, "objectId": previous_object.id}
        target.object = replacement
        target.label_sentences = self._label_sentences(
            replacement, target.id, target.role_label, target.sub_question
        )
        target.why_selected = self._why_selected(target.role_label, target.sub_question)
        used_ids = {item.object.id for item in exhibition.items}
        previous_result = result_by_id.get(previous_object.id)
        previous_summary = (
            self._summary_from_result(previous_result)
            if previous_result is not None
            else ObjectSummary.from_object(previous_object)
        )
        next_alternatives = [previous_summary] + [
            alternative
            for alternative in previous_alternatives
            if alternative.id != replacement_object_id
        ]
        target.alternatives = [
            alternative for alternative in next_alternatives if alternative.id not in used_ids
        ][:3]
        self._refresh_curatorial_brief(exhibition, current_results)
        exhibition.revisions.append(
            Revision(
                id=str(uuid4()),
                kind="object_replaced",
                before=before,
                after={"itemId": target.id, "objectId": replacement.id},
            )
        )
        self._reset_after_edit(exhibition)
        return exhibition

    def reorder_items(
        self,
        exhibition: Exhibition,
        item_ids: list[str] | None = None,
        item_id: str | None = None,
        direction: str | None = None,
    ) -> Exhibition:
        before_ids = [item.id for item in exhibition.items]
        if item_ids is not None:
            if len(item_ids) != len(exhibition.items) or set(item_ids) != set(before_ids):
                raise CollectionDataError(
                    "INVALID_REORDER",
                    "itemIds must contain every current exhibition item exactly once.",
                )
            by_id = {item.id: item for item in exhibition.items}
            exhibition.items = [by_id[current_id] for current_id in item_ids]
        elif item_id and direction:
            index = next((i for i, item in enumerate(exhibition.items) if item.id == item_id), None)
            if index is None:
                raise CollectionDataError("ITEM_NOT_FOUND", f"Exhibition item '{item_id}' was not found.")
            destination = index - 1 if direction == "up" else index + 1
            if destination < 0 or destination >= len(exhibition.items):
                raise CollectionDataError("INVALID_REORDER", "The item cannot move farther in that direction.")
            exhibition.items[index], exhibition.items[destination] = (
                exhibition.items[destination],
                exhibition.items[index],
            )
        else:
            raise CollectionDataError(
                "INVALID_REORDER", "Provide itemIds, or provide itemId with direction 'up' or 'down'."
            )

        chapter_sizes = [
            len(chapter.item_ids)
            for chapter in sorted(exhibition.chapters, key=lambda chapter: chapter.order)
        ]
        cursor = 0
        for chapter, size in zip(
            sorted(exhibition.chapters, key=lambda chapter: chapter.order),
            chapter_sizes,
            strict=True,
        ):
            chapter.item_ids = [
                item.id for item in exhibition.items[cursor : cursor + size]
            ]
            cursor += size

        for index, item in enumerate(exhibition.items):
            item.order = index
            item.relation = self._relation(index, item.role_label)
        collection = self.collections.get(exhibition.collection_id)
        self._refresh_curatorial_brief(
            exhibition,
            self.collections.search(exhibition.agenda, collection),
        )
        exhibition.revisions.append(
            Revision(
                id=str(uuid4()),
                kind="item_reordered",
                before={"itemIds": before_ids},
                after={"itemIds": [item.id for item in exhibition.items]},
            )
        )
        self._reset_after_edit(exhibition)
        return exhibition

    def _context(self, agenda: AgendaInput) -> GenerationContext:
        collection = self.collections.get(agenda.collection_id)
        all_results = self.collections.search(agenda, collection)
        eligible_by_id = {
            obj.id: obj for obj in self.collections.require_generation_ready(collection)
        }
        policy_match = self.collections.match_question_policy(collection, agenda.question)
        policy = policy_match[0] if policy_match else None
        direct_results = self._direct_results(all_results)
        evidence_domain_id = policy.evidence_domain_id if policy else None
        if policy and policy.starter_object_ids:
            starter_objects = [
                eligible_by_id[item_id]
                for item_id in policy.starter_object_ids
                if item_id in eligible_by_id
            ]
            if len(starter_objects) != 5:
                raise CollectionDataError(
                    "QUESTION_CARD_INCOMPLETE",
                    "The matched question card does not currently have five eligible starter objects.",
                    policyId=policy.policy_id,
                    eligibleStarterObjectIds=[obj.id for obj in starter_objects],
                )
            selected = starter_objects
            results = self._filter_evidence_domain(all_results, evidence_domain_id)
        elif policy and policy.status == AnswerabilityStatus.SUPPORTED.value:
            results = self._filter_evidence_domain(all_results, evidence_domain_id)
            selected = self._diverse_selection(
                results,
                5,
                prefer_culture_diversity=_requests_cross_cultural(agenda.question),
            )
        else:
            results = self._filter_evidence_domain(direct_results, evidence_domain_id)
            selected = self._diverse_selection(
                results,
                5,
                prefer_culture_diversity=_requests_cross_cultural(agenda.question),
            )
        selected_ids = {obj.id for obj in selected}
        remaining_results = [
            result for result in results if result.obj.id not in selected_ids
        ]
        return GenerationContext(
            collection=collection,
            results=results,
            selected=selected,
            remaining_results=remaining_results,
            question_card_limits=policy.coverage_limits if policy else [],
            evidence_domain_id=evidence_domain_id,
            policy=policy,
            exhibition_theme=self._exhibition_theme(agenda.question, policy_match),
        )

    @staticmethod
    def _prioritized_object(result: SearchResult) -> MuseumObject:
        """Return an object with query-matched institution evidence first."""
        if not result.matched_evidence_ids:
            return result.obj
        positions = {
            evidence_id: index
            for index, evidence_id in enumerate(result.matched_evidence_ids)
        }
        ordered = sorted(
            result.obj.evidence,
            key=lambda chunk: (positions.get(chunk.id, len(positions)), chunk.id),
        )
        return result.obj.model_copy(update={"evidence": ordered})

    @classmethod
    def _summary_from_result(cls, result: SearchResult) -> ObjectSummary:
        """Keep an alternative's query relevance and evidence trace."""
        return ObjectSummary.from_object(result.obj).model_copy(
            update={
                "retrieval_score": result.score,
                "matched_anchor_terms": list(result.matched_anchor_terms),
                "matched_evidence_ids": list(result.matched_evidence_ids),
            }
        )

    @classmethod
    def _alternative_summaries(
        cls,
        results: list[SearchResult],
        *,
        used_ids: set[str],
        limit: int,
    ) -> list[ObjectSummary]:
        """Build ranked alternatives only from the active RAG candidate set."""
        return [
            cls._summary_from_result(result)
            for result in results
            if result.obj.id not in used_ids
        ][:limit]

    @staticmethod
    def _diverse_selection(
        results: list[SearchResult],
        count: int,
        *,
        prefer_culture_diversity: bool = False,
    ) -> list[MuseumObject]:
        remaining = list(results)
        selected: list[MuseumObject] = []
        seen_themes: set[str] = set()
        seen_media: set[str] = set()
        seen_culture_packs: set[str] = set()
        seen_culture_roots: set[str] = set()
        seen_institutions: set[str] = set()
        strongest = max((result.score for result in remaining), default=1.0)
        available_culture_packs = {
            pack for result in remaining for pack in result.obj.culture_pack_ids
        }

        def culture_root(obj: MuseumObject) -> str:
            # Importers use both ASCII and full-width punctuation.  The first
            # catalogue segment is the broad culture/place label; dynasties and
            # periods after it should not turn “Egypt” into two cultures.
            root = re.split(r"[,;\uFF0C\uFF1B]", obj.culture or "", maxsplit=1)[0]
            return re.sub(r"\s+", " ", root).strip().casefold()

        available_culture_roots = {
            root
            for result in remaining
            if (root := culture_root(result.obj))
        }

        def similarity(candidate: MuseumObject, chosen: MuseumObject) -> float:
            candidate_features = {
                *candidate.themes,
                *candidate.culture_pack_ids,
                culture_root(candidate),
                candidate.medium.casefold() if candidate.medium else "",
                candidate.type.casefold() if candidate.type else "",
            } - {""}
            chosen_features = {
                *chosen.themes,
                *chosen.culture_pack_ids,
                culture_root(chosen),
                chosen.medium.casefold() if chosen.medium else "",
                chosen.type.casefold() if chosen.type else "",
            } - {""}
            union = candidate_features | chosen_features
            return (
                len(candidate_features & chosen_features) / len(union)
                if union
                else 0.0
            )

        while remaining and len(selected) < count:
            candidates = remaining
            if prefer_culture_diversity and len(seen_culture_roots) < min(
                count, len(available_culture_roots)
            ):
                unseen = [
                    result
                    for result in remaining
                    if culture_root(result.obj)
                    and culture_root(result.obj) not in seen_culture_roots
                ]
                if unseen:
                    unseen_pack = [
                        result
                        for result in unseen
                        if set(result.obj.culture_pack_ids) - seen_culture_packs
                    ]
                    candidates = unseen_pack or unseen

            def selection_score(result: SearchResult) -> tuple[float, str]:
                if prefer_culture_diversity:
                    relevance = result.score / strongest if strongest else 0.0
                    novelty = (
                        1.0
                        if set(result.obj.culture_pack_ids) - seen_culture_packs
                        else 0.0
                    )
                    culture_novelty = (
                        1.0
                        if culture_root(result.obj)
                        and culture_root(result.obj) not in seen_culture_roots
                        else 0.0
                    )
                    redundancy = max(
                        (similarity(result.obj, obj) for obj in selected),
                        default=0.0,
                    )
                    institution_novelty = (
                        1.0
                        if result.obj.institution_id
                        and result.obj.institution_id not in seen_institutions
                        else 0.0
                    )
                    # MMR: relevance remains primary, while pack novelty is
                    # strong enough to keep “across cultures” from returning
                    # five near-duplicates from the largest source pack.
                    utility = (
                        0.68 * relevance
                        + 0.34 * culture_novelty
                        + 0.12 * novelty
                        + 0.06 * institution_novelty
                        - 0.22 * redundancy
                    )
                    return utility, result.obj.id
                utility = (
                    result.score
                    + (1.5 if not set(result.obj.themes) & seen_themes else 0)
                    + (0.75 if result.obj.medium and result.obj.medium not in seen_media else 0)
                    + (0.9 if set(result.obj.culture_pack_ids) - seen_culture_packs else 0)
                    + (
                        0.35
                        if result.obj.institution_id
                        and result.obj.institution_id not in seen_institutions
                        else 0
                    )
                )
                return utility, result.obj.id

            best = max(
                candidates,
                key=selection_score,
            )
            selected.append(ExhibitionGenerator._prioritized_object(best))
            seen_themes.update(best.obj.themes)
            if best.obj.medium:
                seen_media.add(best.obj.medium)
            seen_culture_packs.update(best.obj.culture_pack_ids)
            if root := culture_root(best.obj):
                seen_culture_roots.add(root)
            if best.obj.institution_id:
                seen_institutions.add(best.obj.institution_id)
            remaining.remove(best)
        if len(selected) < count:
            raise CollectionDataError(
                "QUESTION_UNSUPPORTED",
                "Fewer than five objects match the question after evidence and exclusion checks.",
                matchedObjectCount=len(selected),
                requiredObjectCount=count,
            )
        return selected

    def _deterministic_exhibition(
        self, agenda: AgendaInput, context: GenerationContext
    ) -> Exhibition:
        exhibition_id = str(uuid4())
        sub_questions = self._sub_questions(agenda.question)
        alternative_pool = self._alternative_summaries(
            context.remaining_results,
            used_ids={obj.id for obj in context.selected},
            limit=len(context.remaining_results),
        )
        items: list[ExhibitionItem] = []
        for index, (obj, role) in enumerate(zip(context.selected, ROLE_ORDER, strict=True)):
            item_id = str(uuid4())
            role_label = ROLE_LABELS[role.value]
            sub_question = sub_questions[index % len(sub_questions)]
            role_alternatives = []
            if alternative_pool:
                rotated = alternative_pool[index:] + alternative_pool[:index]
                role_alternatives = rotated[:3]
            items.append(
                ExhibitionItem(
                    id=item_id,
                    object=obj,
                    role=role,
                    role_label=role_label,
                    sub_question=sub_question,
                    why_selected=self._why_selected(role_label, sub_question),
                    relation=self._relation(index, role_label),
                    label_sentences=self._label_sentences(obj, item_id, role_label, sub_question),
                    alternatives=role_alternatives,
                    order=index,
                )
            )
        pending_evidence = any(
            not evidence.reviewed for obj in context.selected for evidence in obj.evidence
        )
        limits = [
            f"本展只使用 {context.collection.institution} 的馆藏记录（数据版本 {context.collection.version}）。",
            "英文馆方题名与证据原文保持原样；中文策展关系属于系统推断，不冒充机构原文。",
            "材料之外的人物关系、年代因果与历史结论不会由模型常识补写。",
        ]
        limits.extend(limit for limit in context.question_card_limits if limit not in limits)
        if pending_evidence:
            limits.append("部分来源片段为 source-exact-match，但仍等待人工内容审核。")
        return Exhibition(
            id=exhibition_id,
            title=self._title(agenda.question),
            question=agenda.question,
            exhibition_theme=context.exhibition_theme,
            curatorial_thesis=self._thesis(agenda.question),
            core_answer=self._core_answer(agenda.question),
            sub_questions=sub_questions,
            items=items,
            coverage_limits=limits,
            status=ExhibitionStatus.DRAFT,
            versions=VersionInfo(
                model=self.settings.deepseek_model,
                provider="deterministic",
                collection=context.collection.version,
            ),
            agenda=agenda,
            collection_id=context.collection.id,
            evidence_domain_id=context.evidence_domain_id,
        )

    @staticmethod
    def _title(question: str) -> str:
        compact = re.sub(r"\s+", " ", question).strip().rstrip("？?")
        return f"从馆藏证据看：{compact[:42]}"

    @staticmethod
    def _thesis(question: str) -> str:
        return f"围绕“{question}”，把馆方记录组织为引入、背景、核心证据、对照声音与综合延伸五个可核查位置。"

    @staticmethod
    def _core_answer(question: str) -> str:
        return f"围绕“{question}”，这里呈现一条由五件馆藏材料支持、可逐句核查并可由访客修改的解释路径。"

    @staticmethod
    def _sub_questions(question: str) -> list[str]:
        return [
            f"馆方记录为“{question}”提供了哪些直接材料？",
            "这些藏品在年代、材料、用途或表达方式上形成什么关系？",
            "哪件藏品能够提供对照、限制或其他声音？",
        ]

    @staticmethod
    def _why_selected(role_label: str, sub_question: str) -> str:
        del role_label
        return (
            "馆方记录提供了可核对的题名、年代或材料信息，"
            f"可用来追问：{sub_question}"
        )

    @staticmethod
    def _relation(index: int, role_label: str) -> str:
        del role_label
        if index == 0:
            return "先从一件有明确馆方记录的实物开始，确认我们究竟在比较什么。"
        return "与前一件并看时，先比较年代、材料、用途或形象是否真的相同。"

    @staticmethod
    def _label_sentences(
        obj: MuseumObject, item_id: str, role_label: str, sub_question: str
    ) -> list[LabelSentence]:
        del sub_question
        return curation.label_sentences(obj, item_id, role_label, 140)

    @staticmethod
    def _system_prompt() -> str:
        return (
            "你是证据受限的数字遗产微策展助手。只能使用输入 objects 及其 evidence，"
            "不得补写馆方材料之外的人名、年代、因果或价值判断。馆方英文原文不得翻译后冒充原文。"
            "输出单个 JSON 对象，包含 title, curatorialThesis, coreAnswer, subQuestions, coverageLimits, items。"
            "items 必须严格使用给定的五个 objectId 与 role；每个 labelSentence 必须包含 text, type, evidenceIds，"
            "且 evidenceIds 只能来自同一对象。不要生成或改写 institution_fact；馆方事实句由系统保留原文。"
            "你只能补充明确标为 system_inference 或 uncertain 的关系文本。"
        )

    @staticmethod
    def _model_payload(agenda: AgendaInput, context: GenerationContext) -> dict[str, Any]:
        return {
            "agenda": agenda.model_dump(mode="json", by_alias=True),
            "requiredRoles": [role.value for role in ROLE_ORDER],
            "objects": [obj.model_dump(mode="json", by_alias=True) for obj in context.selected],
        }

    @staticmethod
    def _apply_model_output(exhibition: Exhibition, output: dict[str, Any]) -> Exhibition:
        for key, attr in (
            ("title", "title"),
            ("curatorialThesis", "curatorial_thesis"),
            ("coreAnswer", "core_answer"),
        ):
            value = output.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Model output is missing {key}")
            setattr(exhibition, attr, value.strip())

        sub_questions = output.get("subQuestions")
        if not isinstance(sub_questions, list) or not 2 <= len(sub_questions) <= 3:
            raise ValueError("Model output must include two or three subQuestions")
        exhibition.sub_questions = [str(value).strip() for value in sub_questions]
        coverage_limits = output.get("coverageLimits")
        if isinstance(coverage_limits, list):
            exhibition.coverage_limits.extend(
                str(value).strip() for value in coverage_limits if str(value).strip()
            )

        raw_items = output.get("items")
        if not isinstance(raw_items, list) or len(raw_items) != 5:
            raise ValueError("Model output must include exactly five items")
        existing = {item.object.id: item for item in exhibition.items}
        seen_roles: set[str] = set()
        seen_objects: set[str] = set()
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                raise ValueError("Each model item must be an object")
            object_id = str(raw_item.get("objectId", ""))
            role = str(raw_item.get("role", ""))
            if object_id not in existing or role not in REQUIRED_ROLES:
                raise ValueError("Model changed a selected object or required role")
            if object_id in seen_objects or role in seen_roles:
                raise ValueError("Model duplicated an object or role")
            item = existing[object_id]
            if str(item.role) != role:
                raise ValueError("Model reassigned a fixed curatorial role")
            evidence_ids = {evidence.id for evidence in item.object.evidence}
            sentences = raw_item.get("labelSentences")
            if not isinstance(sentences, list) or not sentences:
                raise ValueError("Model item has no labelSentences")
            parsed_sentences = [
                sentence.model_copy(deep=True)
                for sentence in item.label_sentences
                if sentence.type == SentenceType.INSTITUTION_FACT.value
            ]
            for index, sentence in enumerate(sentences):
                if not isinstance(sentence, dict):
                    raise ValueError("Model label sentence is invalid")
                references = sentence.get("evidenceIds")
                if not isinstance(references, list) or not references or not set(references) <= evidence_ids:
                    raise ValueError("Model sentence has an invalid evidence binding")
                sentence_type = sentence.get("type", SentenceType.SYSTEM_INFERENCE.value)
                if sentence_type == SentenceType.INSTITUTION_FACT.value:
                    raise ValueError("The model is not allowed to create or overwrite institution facts")
                if sentence_type not in {
                    SentenceType.SYSTEM_INFERENCE.value,
                    SentenceType.UNCERTAIN.value,
                }:
                    raise ValueError("Model sentence type must be system_inference or uncertain")
                parsed_sentences.append(
                    LabelSentence(
                        id=f"{item.id}-m{index + 1}",
                        text=str(sentence.get("text", "")).strip(),
                        type=sentence_type,
                        evidence_ids=references,
                    )
                )
            item.sub_question = str(raw_item.get("subQuestion", item.sub_question)).strip()
            item.why_selected = str(raw_item.get("whySelected", item.why_selected)).strip()
            item.relation = str(raw_item.get("relation", item.relation)).strip()
            item.label_sentences = parsed_sentences
            seen_objects.add(object_id)
            seen_roles.add(role)
        return exhibition

    @staticmethod
    def _reset_after_edit(exhibition: Exhibition) -> None:
        exhibition.status = ExhibitionStatus.DRAFT
        exhibition.slug = None
        exhibition.review = None
        exhibition.validation = validate_exhibition(exhibition)
        if exhibition.validation.passed:
            exhibition.status = ExhibitionStatus.AUTO_VALIDATED
        exhibition.updated_at = utc_now()
