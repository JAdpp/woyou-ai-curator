from __future__ import annotations

import asyncio
import contextvars
import logging
import re
import threading
import unicodedata
from concurrent.futures import Future
from dataclasses import asdict, dataclass, field, replace
from copy import deepcopy
from functools import partial
from hashlib import sha256
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
    _query_plan,
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
    LocalizedObjectMetadata,
    MuseumObject,
    ObjectSummary,
    ROLE_LABELS,
    Revision,
    SentenceType,
    VersionInfo,
    VisitorMotivation,
    VisitorProfile,
    utc_now,
)
from .providers.deepseek import DeepSeekProvider, ProviderError, VisionImage
from .query_plan_review import (
    parse_query_plan_review,
    query_plan_review_payload,
    query_plan_review_prompt,
    unreviewed_query_plan,
)
from .retrieval_agent import (
    AGENTIC_RETRIEVAL_METHOD,
    AGENTIC_RETRIEVAL_VERSION,
    EXPANDABLE_REASONS,
    EXPANSION_REASON_NONE,
    RetrievalAudit,
    RetrievalQueryPlan,
    audit_payload,
    audit_prompt,
    fuse_search_results,
    parse_audit,
    parse_query_plan,
    query_plan_payload,
    query_plan_prompt,
)
from .retrieval_filters import FilterSpec
from .set_coverage import set_coverage, set_coverage_gap
from .validator import REQUIRED_ROLES, validate_exhibition
from .visual_evidence import audit_visual_candidates, visual_core_proof, prewarm_visual_candidates
from .models import VisualCoreEvidence
from .curatorial_copy_review import (
    copy_review_payload, copy_review_prompt, merge_copy_review_batches,
    concise_copy_review_prompt, copy_decisions_to_patches,
    neutralize_object_review_batch, parse_copy_review_batch, split_copy_review_payload,
    recover_auxiliary_copy_batch,
)


logger = logging.getLogger(__name__)


class _BoundedRetrievalExecutor:
    """No-queue, process-wide admission for non-interruptible catalogue calls.

    Cancelling an await cannot stop synchronous DNS, HTTP or native kernels.
    A timed-out worker therefore keeps its slot until the real call exits.
    Daemon workers are deliberately independent of asyncio's default executor:
    asyncio.run teardown and CLI process exit must not join a stuck retrieval.
    At most max_workers background calls exist; overload fails without queuing.
    """

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers
        self._slots = threading.BoundedSemaphore(max_workers)

    def submit(self, call: Callable[[], Any]) -> Future:
        if not self._slots.acquire(blocking=False):
            raise CollectionDataError(
                "RETRIEVAL_CAPACITY_EXHAUSTED",
                "All bounded catalogue workers are occupied; retry after an active search finishes.",
                maxWorkers=self.max_workers,
            )
        future: Future = Future()
        context = contextvars.copy_context()

        def run() -> None:
            try:
                if not future.set_running_or_notify_cancel():
                    return
                try:
                    result = context.run(call)
                except BaseException as error:
                    future.set_exception(error)
                else:
                    future.set_result(result)
            finally:
                self._slots.release()

        worker = threading.Thread(target=run, name="catalogue-retrieval", daemon=True)
        try:
            worker.start()
        except BaseException:
            self._slots.release()
            raise
        return future


_RETRIEVAL_EXECUTOR = _BoundedRetrievalExecutor()

QUESTION_CARD_RETRIEVAL_METHOD = "reviewed_question_card_starters"
QUESTION_CARD_RETRIEVAL_VERSION = "question-card-v1"

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


@dataclass(frozen=True)
class InitialRetrievalOutcome:
    results: list[SearchResult]
    query_plan: RetrievalQueryPlan | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgenticRetrievalOutcome:
    results: list[SearchResult]
    audit_applied: bool = False
    expanded_queries: tuple[str, ...] = ()
    # Frozen question cards force their reviewed five-object spine, while the
    # wider query pool remains available for honest in-theme alternatives.
    forced_object_ids: tuple[str, ...] = ()
    answerability: str | None = None
    interpretation: str = ""
    coverage_gap: str = ""
    expansion_reason: str = EXPANSION_REASON_NONE
    failure_code: str | None = None
    # Optional catalogue expansion is best-effort. Its failure must remain
    # observable without overwriting the mandatory first audit or making a
    # semantic unsupported verdict look like an infrastructure outage.
    warning_code: str | None = None
    warning_detail: str = ""
    stage_results: dict[str, list[SearchResult]] = field(default_factory=dict)
    audit_diagnostics: dict[str, Any] = field(default_factory=dict)
    exhibition_set_requirements: tuple[dict[str, Any], ...] = ()


def merge_pool_audits(first: RetrievalAudit, recovered: RetrievalAudit, language: str) -> RetrievalAudit:
    """Merge disjoint evidence windows without promoting either local gap."""
    merged = list(first.accepted)
    seen = {result.obj.id for result in merged}
    for result in recovered.accepted:
        if result.obj.id not in seen:
            merged.append(result)
            seen.add(result.obj.id)
    status = (AnswerabilityStatus.SUPPORTED.value
              if recovered.answerability == AnswerabilityStatus.SUPPORTED.value
              else AnswerabilityStatus.PARTIALLY_SUPPORTED.value if merged
              else AnswerabilityStatus.UNSUPPORTED.value)
    gap = ("Some related objects were verified, but a complete response to the original question is not yet established."
           if language == "en" else "本次已核验到部分相关藏品，但尚未确认能完整回应原问题。")
    if not merged:
        gap = ("The reviewed windows have not yielded verified objects; this is not an exhaustive catalogue search."
               if language == "en" else "本次已审核的候选窗口尚未留下合格对象；这不是对全部馆藏的穷尽检索。")
    return replace(first, accepted=merged, answerability=status,
                   coverage_gap="" if status == AnswerabilityStatus.SUPPORTED.value else gap,
                   search_queries=recovered.search_queries or first.search_queries,
                   expansion_reason=recovered.expansion_reason,
                   condition_checks=first.condition_checks + recovered.condition_checks,
                   condition_rejections=first.condition_rejections + recovered.condition_rejections)


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
    r"(?:为什么|为何).{0,24}(?:喜欢|偏爱)",
    r"直接导致|完全由|唯一原因",
    r"精确复原|完整复原",
    r"真实劳动条件",
    r"全部政治和经济原因",
)


AFFECTIVE_BROWSE_PATTERN = re.compile(
    r"(?:我|最近|今天|现在).{0,18}(?:累|疲惫|压力|烦|焦虑|想放松|"
    r"想休息|静一静|慢一点)|(?:不想|不要|别).{0,10}(?:被催|赶时间|"
    r"按年代讲)|\b(?:tired|overwhelmed|stressed|need\s+(?:a\s+)?break|"
    r"slow\s+down)\b",
    re.IGNORECASE,
)


def _is_affective_browse_profile(profile: VisitorProfile, question: str) -> bool:
    """Route a recharge preference as browsing, not a historical claim.

    The original words remain the public inquiry and frame input. Only object
    retrieval becomes browse/diversity selection, with a visible boundary that
    the objects are not proven to cause relaxation or therapeutic effects.
    """

    return bool(
        profile.motivation == VisitorMotivation.RECHARGER
        and AFFECTIVE_BROWSE_PATTERN.search(question)
    )


BRONZE_REPAIR_EVIDENCE = re.compile(
    r"\b(?:conservation\s+(?:treatment|work|history|campaign|project)|"
    r"treatment\s+(?:report|history)|(?:was|were|has\s+been|have\s+been)\s+"
    r"(?:conserved|restored|repaired|cleaned|stabili[sz]ed)|restored\s+by|"
    r"repaired\s+by|corrosion|corroded|rust(?:ed|ing)?|stabili[sz](?:ed|ation)|"
    r"repair(?:ed|s)?|cleaning)\b",
    re.IGNORECASE,
)
RITUAL_PRACTICE_EVIDENCE = re.compile(
    r"\b(?:consecrat(?:ed|ion)|active\s+(?:worship|ritual)|"
    r"(?:used|use)\s+(?:in|for)\s+(?:worship|ritual|ceremon(?:y|ies)|devotion)|"
    r"worship(?:ped)?|devotion(?:al)?|prayer|pilgrimage|offering(?:s)?|"
    r"funerary\s+(?:ritual|use)|ritual\s+(?:use|object|implement|practice))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PredicateAssessment:
    status: AnswerabilityStatus
    matched: list[SearchResult]
    gap: str


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


CANONICAL_ORIGIN_REGIONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "east_asia",
        re.compile(
            r"中国|日本|韩国|朝鲜|东亚|\b(?:china|chinese|japan|japanese|korea|korean|tibet|tibetan|mongol(?:ia|ian)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "south_asia",
        re.compile(
            r"南亚|印度|\b(?:india|indian|pakistan|pakistani|sri lanka|nepal|nepalese|mughal)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "southeast_asia",
        re.compile(
            r"东南亚|越南|泰国|柬埔寨|缅甸|印尼|\b(?:southeast asia|vietnam(?:ese)?|thailand|thai|siam|cambodia(?:n)?|myanmar|burma|burmese|indonesia(?:n)?|philippines?|filipino)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "west_asia_north_africa",
        re.compile(
            r"西亚|中东|北非|伊朗|波斯|埃及|\b(?:west asia|middle east|north africa|iran(?:ian)?|persia(?:n)?|iraq(?:i)?|syria(?:n)?|turk(?:ey|ish)|ottoman|egypt(?:ian)?|morocco|moroccan|algeria(?:n)?|tunisia(?:n)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "africa",
        re.compile(
            r"非洲|\b(?:africa(?:n)?|nigeria(?:n)?|congo(?:lese)?|ghana(?:ian)?|mali(?:an)?|ethiopia(?:n)?|senegal(?:ese)?|kenya(?:n)?|tanzania(?:n)?|uganda(?:n)?|côte d['’]ivoire|ivorian)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "americas",
        re.compile(
            r"美洲|墨西哥|秘鲁|\b(?:americas?|american|united states|mexic(?:o|an)|aztec|maya(?:n)?|peru(?:vian)?|andean|brazil(?:ian)?|canada|canadian|colombia(?:n)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "europe",
        re.compile(
            r"欧洲|希腊|罗马|\b(?:europe(?:an)?|greece|greek|roman|rome|italy|italian|france|french|germany|german|netherlands|dutch|holland|britain|british|england|english|spain|spanish|portugal|portuguese|russia(?:n)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "oceania",
        re.compile(
            r"大洋洲|太平洋|\b(?:oceania(?:n)?|pacific island|australia(?:n)?|new zealand|maori|papua new guinea)\b",
            re.IGNORECASE,
        ),
    ),
)

CANONICAL_ORIGIN_PACKS = frozenset(
    region for region, _pattern in CANONICAL_ORIGIN_REGIONS
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
                audit_generate = getattr(
                    self.provider,
                    "generate_retrieval_audit_json",
                    None,
                )
                query_plan_generate = getattr(
                    self.provider,
                    "generate_retrieval_query_plan_json",
                    None,
                )
                if (stage.startswith("retrieval_plan:") or stage.startswith("retrieval_plan_review")) and callable(
                    query_plan_generate
                ):
                    operation = query_plan_generate(system_prompt, user_payload)
                elif (stage.startswith("retrieval_audit") or stage.startswith("frame_review")) and callable(
                    audit_generate
                ):
                    operation = audit_generate(system_prompt, user_payload)
                else:
                    operation = self.provider.generate_json(
                        system_prompt, user_payload
                    )
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

    async def _generate_retrieval_audit_json(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        *,
        stage: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Retry one transport failure without resetting the audit budget.

        A retry is the same evidence request, not a second opinion about an
        unfavorable verdict. HTTP refusals, invalid output and timeouts still
        propagate to the existing fail-closed/optional-expansion handling.
        """
        audit_deadline = perf_counter() + timeout_seconds
        try:
            return await self._generate_model_json(
                system_prompt, user_payload, stage=stage, timeout_seconds=timeout_seconds,
            )
        except ProviderError as error:
            remaining = audit_deadline - perf_counter()
            if error.code != "provider_network_error" or remaining < 1.0:
                raise
            logger.warning(
                "Retrieval audit transport retry stage=%s remaining=%.3fs", stage, remaining,
            )
            return await self._generate_model_json(
                system_prompt, user_payload, stage=f"{stage}:transport_retry",
                timeout_seconds=remaining,
            )

    @staticmethod
    async def _run_retrieval_work(
        call: Callable[[], Any],
        *,
        timeout_seconds: float,
    ) -> Any:
        if timeout_seconds <= 0:
            raise asyncio.TimeoutError
        future = _RETRIEVAL_EXECUTOR.submit(call)
        return await asyncio.wait_for(
            asyncio.wrap_future(future), timeout=timeout_seconds
        )

    async def _rerank_async(
        self,
        question: str,
        results: list[SearchResult],
        *,
        deadline: float,
    ) -> list[SearchResult]:
        """Optional synchronous reranking must not block deadline delivery."""

        if not isinstance(self.collections, CollectionRepository):
            return results
        try:
            return await self._run_retrieval_work(
                partial(self.collections.rerank_results, question, results, deadline=deadline),
                timeout_seconds=max(0.0, deadline - perf_counter()),
            )
        except (asyncio.TimeoutError, CollectionDataError) as error:
            logger.warning("Optional fused reranking unavailable: %s", error)
            return results

    async def _search_async(
        self,
        agenda: AgendaInput,
        collection: LoadedCollection,
        *,
        deadline: float | None = None,
        filters: FilterSpec | None = None,
    ) -> list[SearchResult]:
        """Run CPU/mmap retrieval without blocking the API event loop."""

        retrieval_budget = float(
            getattr(
                self.settings,
                "rag_retrieval_timeout_seconds",
                self.settings.rag_llm_audit_timeout_seconds,
            )
        )
        timeout_seconds = (
            retrieval_budget
            if deadline is None
            else max(0.0, deadline - perf_counter())
        )
        if timeout_seconds <= 0.0:
            raise CollectionDataError(
                "RETRIEVAL_SEARCH_TIMEOUT",
                "Collection retrieval had no wall-clock budget remaining.",
                timeoutSeconds=0.0,
            )
        try:
            if isinstance(self.collections, CollectionRepository):
                search_method = self.collections.search
                search_code = getattr(search_method, "__code__", None)
                accepts_kwargs = bool(
                    search_code is not None and search_code.co_flags & 0x08
                )
                search_kwargs: dict[str, Any] = {}
                if accepts_kwargs or (
                    search_code is not None
                    and "deadline" in search_code.co_varnames
                ):
                    search_kwargs["deadline"] = deadline
                if accepts_kwargs or (
                    search_code is not None
                    and "filters" in search_code.co_varnames
                ):
                    search_kwargs["filters"] = filters
                search_call = partial(
                    search_method,
                    agenda,
                    collection,
                    **search_kwargs,
                )
            else:
                search_call = partial(
                    self.collections.search,
                    agenda,
                    collection,
                )
            return await self._run_retrieval_work(
                search_call,
                timeout_seconds=timeout_seconds,
            )
        except asyncio.TimeoutError as error:
            raise CollectionDataError(
                "RETRIEVAL_SEARCH_TIMEOUT",
                "Collection retrieval exceeded its wall-clock budget.",
                timeoutSeconds=timeout_seconds,
            ) from error

    async def prepare_initial_retrieval(
        self,
        agenda: AgendaInput,
        collection: LoadedCollection,
        *,
        deadline: float | None = None,
    ) -> InitialRetrievalOutcome:
        """Plan explicit constraints before the first expensive catalogue call.

        Shared by visitor generation and evaluation. The planner receives only
        the visitor's question, never benchmark rules or expected object IDs.
        When the active provider supports planning, a non-browse request must
        obtain a valid draft before recall. A planning outage is a service
        failure, not evidence that the collection cannot answer the question.
        Legacy providers without this capability retain their existing path.
        """

        started = perf_counter()
        deadline = deadline or (
            started + float(self.settings.rag_retrieval_timeout_seconds)
        )
        plan: RetrievalQueryPlan | None = None
        diagnostics: dict[str, Any] = {"planningStatus": "not_available", "planningAttempts": 0}
        browse_all = _query_plan(
            agenda.question, getattr(collection, "concept_aliases", {})
        ).browse_all
        can_plan = bool(
            self.settings.rag_llm_audit_enabled
            and self.provider is not None
            and self.provider.configured
            and getattr(self.provider, "supports_retrieval_audit", False)
            and getattr(self.provider, "supports_retrieval_query_planning", False)
            and self.settings.rag_agentic_max_queries > 0
        )
        if browse_all:
            diagnostics["planningStatus"] = "browse_all"
        elif can_plan:
            audit_floor = min(
                float(self.settings.rag_llm_audit_timeout_seconds),
                max(10.0, self.settings.rag_llm_audit_timeout_seconds * 0.75),
            )
            budget = min(float(self.settings.rag_planning_timeout_seconds), max(0.0, deadline - perf_counter() - audit_floor))
            if budget >= 2.0:
                planner_deadline = min(deadline - audit_floor, perf_counter() + budget)
                # Draft and its one possible transport retry share eight seconds;
                # fidelity review uses the remaining configured sub-budget, all
                # within the unchanged retrieval deadline and audit reserve.
                draft_deadline = min(planner_deadline, perf_counter() + 8.0)
                for attempt in range(2):
                    remaining = max(0.0, draft_deadline - perf_counter())
                    if remaining < 1.0:
                        break
                    diagnostics["planningAttempts"] = attempt + 1
                    try:
                        output = await self._generate_model_json(
                            query_plan_prompt(agenda.language),
                            query_plan_payload(agenda.question, agenda.language),
                            stage=f"retrieval_plan:{attempt + 1}",
                            timeout_seconds=remaining,
                        )
                        if not isinstance(output, dict):
                            raise ValueError("query_plan_response_not_object")
                        plan = parse_query_plan(
                            output,
                            question=agenda.question,
                            max_queries=min(self.settings.rag_agentic_max_queries, 5),
                        )
                        diagnostics["planningStatus"] = (
                            "applied" if plan.valid else "invalid_contract"
                        )
                        break
                    except (ProviderError, ValueError, TypeError, KeyError) as error:
                        diagnostics["planningStatus"] = "unavailable"
                        diagnostics["planningError"] = type(error).__name__
                        diagnostics["planningErrorCode"] = getattr(error, "code", "invalid_contract")
                        # One transient transport retry shares the original
                        # eight-second draft budget. Billing/API refusals,
                        # malformed output and timeout are not retryable here.
                        retryable = (
                            attempt == 0
                            and isinstance(error, ProviderError)
                            and error.code == "provider_network_error"
                        )
                        logger.warning("Initial catalogue planning unavailable: %s", error)
                        if not retryable:
                            break
                if plan is not None and plan.valid:
                    review_started = perf_counter()
                    remaining = max(0.0, planner_deadline - review_started)
                    review = unreviewed_query_plan(plan, "shared_planning_deadline")
                    diagnostics["planningReviewAttempts"] = 0
                    if remaining >= 0.5:
                        diagnostics["planningReviewAttempts"] = 1
                        try:
                            output = await self._generate_model_json(
                                query_plan_review_prompt(agenda.language),
                                query_plan_review_payload(agenda.question, plan, agenda.language),
                                stage="retrieval_plan_review",
                                timeout_seconds=remaining,
                            )
                            review = parse_query_plan_review(
                                output, question=agenda.question, draft=plan,
                                max_queries=min(self.settings.rag_agentic_max_queries, 5),
                            )
                            repair_budget = min(5.0, max(0.0, planner_deadline - perf_counter()))
                            if not review.reviewed and repair_budget >= 2.0:
                                # One schema/declared-scope repair, still inside
                                # the same configured planning deadline. It
                                # cannot see candidates or optimise for quota.
                                diagnostics["planningReviewInitial"] = dict(review.diagnostics)
                                diagnostics["planningReviewAttempts"] = 2
                                repair_payload = query_plan_review_payload(agenda.question, plan, agenda.language)
                                repair_payload.update({"previousReview": output,
                                                       "contractFailure": review.diagnostics.get("reason")})
                                repaired_output = await self._generate_model_json(
                                    query_plan_review_prompt(agenda.language)
                                    + "\nOne contract repair: the prior review failed the declared-scope/schema validation shown in contractFailure. Re-read the original question independently, follow the scope policy, and return a complete consistent review. Remove stale set entries only when replaced by the correctly scoped original goal; never drop a visitor goal merely to make the schema valid. No retrieval candidates are available.",
                                    repair_payload, stage="retrieval_plan_review_repair",
                                    timeout_seconds=repair_budget,
                                )
                                review = parse_query_plan_review(
                                    repaired_output, question=agenda.question, draft=plan,
                                    max_queries=min(self.settings.rag_agentic_max_queries, 5),
                                )
                        except (ProviderError, ValueError, TypeError, KeyError) as error:
                            review = unreviewed_query_plan(plan, "review_unavailable")
                            diagnostics["planningReviewError"] = type(error).__name__
                            diagnostics["planningReviewErrorCode"] = getattr(error, "code", "invalid_contract")
                            logger.warning("Catalogue plan fidelity review unavailable: %s", error)
                    # A well-formed draft is not a reviewed visitor contract.
                    # Retrying a service failure must not silently curate a
                    # simpler question from that draft.
                    plan = review.plan if review.reviewed else None
                    diagnostics["planningReview"] = review.diagnostics
                    diagnostics["planningReviewElapsedMs"] = round((perf_counter() - review_started) * 1000, 2)
            else:
                diagnostics["planningStatus"] = "budget_exhausted"
        diagnostics["planElapsedMs"] = round((perf_counter() - started) * 1000, 2)
        if can_plan and not browse_all and not (plan is not None and plan.valid):
            # Do not erase the visitor's per-object requirements by continuing
            # with an empty condition contract after a provider failure. This
            # also lets the benchmark count an explicit service failure rather
            # than treating unrelated, unconditionally accepted objects as a
            # successful retrieval. The source-fidelity review is required too.
            diagnostics["queryPlanApplied"] = False
            raise CollectionDataError(
                "RETRIEVAL_PLAN_UNAVAILABLE",
                (
                    "The exhibition search-planning service is temporarily unavailable; "
                    "object selection has not started. Your question is preserved. Please "
                    "retry shortly; this does not mean the collection lacks relevant material."
                    if agenda.language == "en" else
                    "策展检索规划服务暂时不可用，尚未开始选品。你的问题仍然保留，"
                    "请稍后直接重试；这不表示馆藏缺少相关资料。"
                ),
                serviceStage="retrieval_planning",
                retryable=True,
                planningDiagnostics=diagnostics,
            )
        filters = self._recall_filters(plan)
        diagnostics["filterSpec"] = asdict(filters)
        diagnostics["catalogueTypeHints"] = list(plan.catalogue_type_hints) if plan else []
        diagnostics["typeFilterPolicy"] = "semantic_kind_audited_not_literal_catalogue_type"
        diagnostics["materialAuditConditions"] = list(plan.filters.materials) if plan else []
        diagnostics["materialFilterPolicy"] = "natural_language_material_verified_in_evidence_audit"
        diagnostics["queryPlanApplied"] = bool(plan and plan.valid)
        search_started = perf_counter()
        results = await self._search_async(
            agenda, collection, deadline=deadline, filters=filters
        )
        diagnostics["searchElapsedMs"] = round(
            (perf_counter() - search_started) * 1000, 2
        )
        return InitialRetrievalOutcome(results, plan, diagnostics)

    async def _agentic_retrieve(
        self,
        agenda: AgendaInput,
        collection: LoadedCollection,
        initial_results: list[SearchResult],
        *,
        required_count: int,
        deadline: float | None = None,
        initial_query_plan: RetrievalQueryPlan | None = None,
        planning_attempted: bool = False,
    ) -> AgenticRetrievalOutcome:
        stages: dict[str, list[SearchResult]] = {}
        diagnostics: dict[str, Any] = {}
        prewarm_task = None
        if (initial_query_plan and initial_query_plan.evidence_mode in {"visual_observation", "open_exploration"}
                and self.settings.rag_visual_audit_enabled
                and callable(getattr(self.image_cache, "get", None))):
            prewarm_sample = self._audit_candidate_sample(
                initial_results,
                top_k=min(max(required_count * 4, 20), self.settings.rag_llm_audit_top_k, 32),
                cross_cultural=_requests_cross_cultural(agenda.question),
                cultural_obligations=self._cultural_coverage_obligations(agenda.question),
                query_batches=[],
            )
            prewarm_task = asyncio.create_task(prewarm_visual_candidates(
                [result.obj for result in prewarm_sample[:self.settings.rag_visual_audit_top_k]],
                self.image_cache, max_candidates=self.settings.rag_visual_audit_top_k,
                timeout_seconds=min(12.0, max(0.001, deadline - perf_counter())) if deadline else 12.0,
            ))
        try:
            outcome = await self._agentic_retrieve_impl(
                agenda, collection, initial_results,
                required_count=required_count,
                deadline=deadline,
                initial_query_plan=initial_query_plan,
                planning_attempted=planning_attempted,
                stage_results=stages,
                audit_diagnostics=diagnostics,
                prewarm_task=prewarm_task,
            )
        finally:
            if prewarm_task is not None and not prewarm_task.done():
                prewarm_task.cancel()
                await asyncio.gather(prewarm_task, return_exceptions=True)
        requirements = tuple(diagnostics.get("exhibitionSetRequirements", (
            initial_query_plan.exhibition_set_requirements if initial_query_plan else ()
        )))
        coverage = set_coverage(requirements, outcome.results, question=agenda.question)
        if diagnostics.get("poolRecoveryFailure") and not outcome.warning_code:
            outcome = replace(outcome, warning_code="RETRIEVAL_AUDIT_UNAVAILABLE",
                              warning_detail=("Additional evidence review did not finish."
                                              if agenda.language == "en" else "本次补充证据复审尚未完成。"))
        if (outcome.warning_code and not outcome.failure_code
                and (len(outcome.results) < required_count or not coverage["satisfied"]
                     or any(not any(self._object_satisfies_cultural_obligation(result.obj, obligation)
                                    for result in outcome.results)
                            for obligation in self._cultural_coverage_obligations(agenda.question))
                     or outcome.answerability == AnswerabilityStatus.UNSUPPORTED.value)):
            # Optional recovery becomes necessary when the retained evidence
            # cannot form this visit. An interrupted audit is then a service
            # failure, not evidence that the question has no answer.
            outcome = replace(outcome, failure_code=outcome.warning_code,
                              coverage_gap=outcome.warning_detail or (
                                  "Evidence review did not finish. Retry the same question."
                                  if agenda.language == "en" else
                                  "本次证据复审尚未完成，请直接重试原问题；这不表示馆藏缺少相关资料。"
                              ))
        if requirements:
            diagnostics["auditedSetCoverage"] = coverage
            if not coverage["satisfied"] and not outcome.failure_code:
                outcome = replace(outcome, answerability=AnswerabilityStatus.UNSUPPORTED.value,
                                  coverage_gap=set_coverage_gap(coverage, agenda.language))
        return replace(outcome, stage_results=stages, audit_diagnostics=diagnostics,
                       exhibition_set_requirements=requirements)

    async def _agentic_retrieve_impl(
        self,
        agenda: AgendaInput,
        collection: LoadedCollection,
        initial_results: list[SearchResult],
        *,
        required_count: int,
        deadline: float | None = None,
        initial_query_plan: RetrievalQueryPlan | None = None,
        planning_attempted: bool = False,
        stage_results: dict[str, list[SearchResult]],
        audit_diagnostics: dict[str, Any] | None = None,
        prewarm_task: asyncio.Task | None = None,
    ) -> AgenticRetrievalOutcome:
        """Audit hybrid recall and run at most one bounded expansion loop.

        The deterministic/embedding repository remains the source of every
        object and evidence ID. The model can only accept IDs it was shown and
        can only suggest additional search strings; it cannot inject an object
        or turn its own background knowledge into museum evidence.
        """

        # A browse-all visit intentionally has no topical predicate to audit.
        # It enters the separate diversity/narrative selector below; asking the
        # relevance model whether objects "answer" a request to surprise the
        # visitor would manufacture an unsupported theme.
        if initial_results and all(
            "browse_all" in result.retrieval_sources
            for result in initial_results
        ):
            return AgenticRetrievalOutcome(results=initial_results)

        policy_match = self.collections.match_question_policy(
            collection, agenda.question
        )
        # Only a supported public question card with enough explicit starter
        # objects is a frozen positive route. Regression labels never exempt a
        # query from runtime relevance audit. Exact reviewed negative/partial
        # policies remain hard boundaries so the profile path cannot bypass
        # answerability merely because raw retrieval found many topical rows.
        if policy_match is not None and policy_match[1] == 1.0:
            policy = policy_match[0]
            if policy.status != AnswerabilityStatus.SUPPORTED.value:
                return AgenticRetrievalOutcome(
                    results=initial_results,
                    answerability=policy.status,
                    coverage_gap=(
                        policy.rationale
                        or "；".join(policy.coverage_limits)
                        or (
                            "The reviewed question boundary does not support generation as asked."
                            if agenda.language == "en"
                            else "经审定的问题边界不支持按当前问法生成展览。"
                        )
                    ),
                )
            if policy.source == "question_card" and policy.starter_object_ids:
                eligible_by_id = {
                    obj.id: obj
                    for obj in self.collections.require_generation_ready(collection)
                }
                starters = [
                    eligible_by_id[object_id]
                    for object_id in policy.starter_object_ids
                    if object_id in eligible_by_id
                ]
                if len(starters) >= required_count:
                    starters = starters[:required_count]
                    initial_by_id = {
                        result.obj.id: result for result in initial_results
                    }
                    starter_results: list[SearchResult] = []
                    for rank, obj in enumerate(starters):
                        base = initial_by_id.get(obj.id)
                        if base is None:
                            starter_results.append(
                                SearchResult(
                                    obj=obj,
                                    score=float(len(starters) - rank),
                                    retrieval_sources=("question_card",),
                                )
                            )
                        else:
                            starter_results.append(
                                replace(
                                    base,
                                    retrieval_sources=tuple(
                                        dict.fromkeys(
                                            (*base.retrieval_sources, "question_card")
                                        )
                                    ),
                                )
                            )
                    starter_ids = tuple(obj.id for obj in starters)
                    remaining = [
                        result
                        for result in initial_results
                        if result.obj.id not in set(starter_ids)
                    ]
                    return AgenticRetrievalOutcome(
                        results=[*starter_results, *remaining],
                        forced_object_ids=starter_ids,
                        answerability=AnswerabilityStatus.SUPPORTED.value,
                    )

        provider_can_audit = bool(
            self.provider is not None
            and self.provider.configured
            and getattr(self.provider, "supports_retrieval_audit", False)
        )
        provider_can_plan_queries = bool(
            provider_can_audit
            and getattr(
                self.provider,
                "supports_retrieval_query_planning",
                False,
            )
            and callable(
                getattr(
                    self.provider,
                    "generate_retrieval_query_plan_json",
                    None,
                )
            )
        )
        unaudited_dense_only = any(
            "bm25" not in result.retrieval_sources
            and any(
                source.startswith("dense")
                for source in result.retrieval_sources
            )
            for result in initial_results
        )
        lexical_fallback = [
            result
            for result in initial_results
            if "bm25" in result.retrieval_sources
        ]
        safe_fallback = lexical_fallback if unaudited_dense_only else initial_results
        # Every non-browse, non-frozen visitor question reaches the same
        # evidence audit. A familiar alias only improves recall; it cannot
        # certify that the rest of a free-form question is supported. This
        # also prevents degraded BM25 mode from treating one generic matched
        # word (for example "place") as approval of an unknown subject.
        semantic_audit_required = True
        if not self.settings.rag_llm_audit_enabled:
            return AgenticRetrievalOutcome(
                results=[] if semantic_audit_required else safe_fallback,
                failure_code=(
                    "RETRIEVAL_AUDIT_UNAVAILABLE"
                    if semantic_audit_required
                    else None
                ),
                coverage_gap=(
                    "The semantic relevance audit is disabled; this open or ambiguous query cannot be approved from similarity or title words alone."
                    if agenda.language == "en"
                    else "语义相关性审查已停用；开放问法或有歧义的题名不能只靠相似度／词面命中获准。"
                )
                if semantic_audit_required or unaudited_dense_only
                else "",
            )
        if not provider_can_audit:
            return AgenticRetrievalOutcome(
                results=[] if semantic_audit_required else safe_fallback,
                failure_code=(
                    "RETRIEVAL_AUDIT_UNAVAILABLE"
                    if semantic_audit_required
                    else None
                ),
                coverage_gap=(
                    "This open or ambiguous query requires an evidence relevance audit, but the audit model is unavailable."
                    if agenda.language == "en"
                    else "开放问法或有歧义的题名需要证据相关性审查；本次审查模型不可用，未让词面近邻进入展览。"
                )
                if semantic_audit_required or unaudited_dense_only
                else "",
            )

        # A source-quote audit is deliberately stricter than nearest-neighbour
        # retrieval. Give a five-object exhibition enough reviewed alternatives
        # for some honest rejections while retaining the configured/token cap.
        top_k = min(max(required_count * 4, 20), self.settings.rag_llm_audit_top_k, 32)
        max_queries = min(self.settings.rag_agentic_max_queries, 5)
        started = perf_counter()
        retrieval_budget = float(
            getattr(
                self.settings,
                "rag_retrieval_timeout_seconds",
                self.settings.rag_llm_audit_timeout_seconds,
            )
        )
        retrieval_deadline = deadline or (started + retrieval_budget)
        per_audit_cap = float(self.settings.rag_llm_audit_timeout_seconds)
        query_plan = initial_query_plan
        query_batches: list[list[SearchResult]] = []
        stage_results["structured_retrieval"] = list(initial_results)

        def remaining_time() -> float:
            return max(0.0, retrieval_deadline - perf_counter())

        visual_report = None
        audit_diagnostics = audit_diagnostics if audit_diagnostics is not None else {}

        async def run_audit(
            candidates: list[SearchResult],
            *,
            pass_number: int,
            timeout_seconds: float,
        ) -> RetrievalAudit:
            nonlocal visual_report
            candidates = self._dedupe_audit_candidates(candidates)
            audited_candidates = self._audit_candidate_sample(
                candidates,
                top_k=top_k,
                cross_cultural=_requests_cross_cultural(agenda.question),
                cultural_obligations=self._cultural_coverage_obligations(
                    agenda.question
                ),
                query_batches=query_batches,
            )
            stage_results[f"audit_sample_{pass_number}"] = list(audited_candidates)
            if pass_number == 1 and prewarm_task is not None:
                try:
                    audit_diagnostics["visualPrewarm"] = await asyncio.wait_for(
                        prewarm_task, timeout=max(0.001, min(6.0, remaining_time() - 20.0)),
                    )
                except (asyncio.TimeoutError, OSError, ValueError):
                    audit_diagnostics["visualPrewarm"] = {"status": "deadline_exceeded"}
            if (pass_number == 1 and query_plan is not None
                    and query_plan.evidence_mode in {"visual_observation", "open_exploration"}
                    and self.settings.rag_visual_audit_enabled and self.image_cache is not None):
                visual_budget = min(self.settings.rag_visual_audit_timeout_seconds,
                                    max(0.0, remaining_time() - min(15.0, per_audit_cap)))
                if visual_budget >= 2.0:
                    visual_predicates = [
                        {"id": f"p{index + 1}", "kind": "visual", "text": text}
                        for index, text in enumerate(query_plan.mandatory_predicates)
                        if f"p{index + 1}" in query_plan.visual_predicate_ids
                    ]
                    if query_plan.catalogue_type_hints:
                        visual_predicates.append({"id": "object_kind", "kind": "object_kind",
                                                  "text": " or ".join(query_plan.catalogue_type_hints)})
                    visual_predicates.extend(
                        {"id": requirement["id"], "kind": "visual", "text": requirement["text"]}
                        for requirement in query_plan.exhibition_set_requirements
                        if requirement["evidenceScope"] == "visible_features_or_record"
                    )
                    visual_report = await audit_visual_candidates(
                        agenda.question, [result.obj for result in audited_candidates],
                        self.provider, self.image_cache,
                        max_candidates=self.settings.rag_visual_audit_top_k,
                        timeout_seconds=visual_budget, visual_predicates=visual_predicates,
                    )
                    audit_diagnostics["visualPreselection"] = visual_report.to_payload()
            visual_sources = visual_report.to_payload() if visual_report is not None else None
            output = await self._generate_retrieval_audit_json(
                audit_prompt(agenda.language) + f"\nSTRICT OUTPUT BUDGET: conditionEvidence replaces legacy predicateEvidence; do not emit predicateEvidence. Return at most {max(required_count, 10)} best accepted objects, preserving required comparison legs before near-duplicates. Each supportingQuote is an exact concise span (normally 10-100 characters), never the whole record. Omit prose reasons for accepted objects. Rejected objects need only objectId and the failed condition check; omit unknown checks. Never truncate the JSON or weaken an admission condition to fit the budget.",
                audit_payload(
                    agenda.question,
                    audited_candidates,
                    required_count=required_count,
                    top_k=top_k,
                    pass_number=pass_number,
                    mandatory_predicates=(
                        query_plan.mandatory_predicates if query_plan else ()
                    ),
                    pool_coverage_legs=(
                        query_plan.pool_coverage_legs if query_plan else ()
                    ),
                    selection_constraints=(
                        query_plan.selection_constraints if query_plan else ()
                    ),
                    evidence_mode=(
                        query_plan.evidence_mode if query_plan else "record_explanation"
                    ),
                    catalogue_type_hints=(
                        query_plan.catalogue_type_hints if query_plan else ()
                    ),
                    explicit_materials=query_plan.filters.materials if query_plan else (),
                    strict_conditions=True,
                    visual_sources=visual_sources,
                    visual_predicate_ids=query_plan.visual_predicate_ids if query_plan else (),
                    exhibition_set_requirements=query_plan.exhibition_set_requirements if query_plan else (),
                ),
                stage=f"retrieval_audit:{pass_number}",
                timeout_seconds=min(timeout_seconds, remaining_time()),
            )
            audit = parse_audit(
                output,
                audited_candidates,
                question=agenda.question,
                max_expansion_queries=max_queries,
                mandatory_predicates=(
                    query_plan.mandatory_predicates if query_plan else ()
                ),
                catalogue_type_hints=query_plan.catalogue_type_hints if query_plan else (),
                explicit_materials=query_plan.filters.materials if query_plan else (),
                strict_conditions=True,
                evidence_mode=query_plan.evidence_mode if query_plan else "record_explanation",
                visual_sources=visual_sources,
                visual_predicate_ids=query_plan.visual_predicate_ids if query_plan else (),
                exhibition_set_requirements=query_plan.exhibition_set_requirements if query_plan else (),
            )
            audit_diagnostics["exhibitionSetRequirements"] = list(
                query_plan.exhibition_set_requirements if query_plan else ()
            )
            audit_diagnostics[f"pass{pass_number}"] = {
                "valid": audit.valid, "conditions": list(audit.condition_checks),
                "rejections": list(audit.condition_rejections),
                "acceptedObjectIds": [result.obj.id for result in audit.accepted],
                "answerability": audit.answerability,
                "windowCoverageGap": audit.coverage_gap,
                "coverageGapScope": "this_audit_window_only",
            }
            if visual_report is not None and query_plan is not None:
                annotated = []
                for result in audit.accepted:
                    visual = visual_report.by_object_id.get(result.obj.id)
                    proof = (visual_core_proof(visual, evidence_mode=query_plan.evidence_mode)
                             if visual is not None else None)
                    if proof and "visual_condition_source_bound" in result.retrieval_sources:
                        proof["questionSha256"] = sha256(agenda.question.encode("utf-8")).hexdigest()
                        obj = result.obj.model_copy(update={
                            "visual_core_evidence": VisualCoreEvidence.model_validate(proof),
                        })
                        result = replace(result, obj=obj)
                    annotated.append(result)
                audit = replace(audit, accepted=annotated)
            return audit

        # When the visitor's words produce too few direct catalogue matches,
        # translate them into a few atomic catalogue expressions before the
        # first evidence audit. This is a recall aid, not an approval shortcut:
        # planned-query candidates are fused with the original ranking and
        # must still be accepted against source IDs by ``run_audit``.
        preplanned_queries: list[str] = []
        # BM25 hit count is not a precision signal: a broad word such as
        # "writing" can produce hundreds of cabinets and manuscripts while
        # missing all three inscription functions the visitor named. Every
        # open, non-frozen question therefore receives one bounded semantic
        # query plan before evidence audit. The plan only improves recall; it
        # cannot approve objects or bypass source-ID validation.
        if (query_plan is not None or provider_can_plan_queries) and max_queries > 0:
            remaining = remaining_time()
            mandatory_audit_floor = min(
                per_audit_cap,
                max(10.0, per_audit_cap * 0.75),
            )
            planner_budget = min(
                8.0,
                per_audit_cap,
                max(0.0, remaining - mandatory_audit_floor),
            )
            if query_plan is not None or (not planning_attempted and planner_budget >= 2.0):
                if query_plan is None:
                    try:
                        raw_plan = await self._generate_model_json(
                            query_plan_prompt(agenda.language),
                            query_plan_payload(agenda.question, agenda.language),
                            stage="retrieval_plan:1",
                            timeout_seconds=planner_budget,
                        )
                        query_plan = parse_query_plan(
                            raw_plan,
                            question=agenda.question,
                            max_queries=max_queries,
                        )
                    except (ProviderError, ValueError, TypeError, KeyError) as error:
                        logger.warning(
                            "optional retrieval query plan unavailable (%s: %s)",
                            type(error).__name__,
                            error,
                        )
                        query_plan = None

                if (
                    query_plan is not None
                    and query_plan.valid
                    and query_plan.in_collection_scope
                    and not query_plan.filters.empty
                    and not planning_attempted
                ):
                    try:
                        initial_results = await self._search_async(
                            agenda,
                            collection,
                            deadline=retrieval_deadline,
                            filters=self._recall_filters(query_plan),
                        )
                    except CollectionDataError as error:
                        logger.warning(
                            "controlled catalogue filters could not be applied "
                            "(%s: %s)",
                            error.code,
                            error,
                        )
                        return AgenticRetrievalOutcome(
                            results=[],
                            failure_code=error.code,
                            coverage_gap=(
                                "The explicit catalogue constraints could not be applied safely."
                                if agenda.language == "en"
                                else "无法安全执行问题中的明确馆藏筛选条件，本次未忽略条件继续生成。"
                            ),
                        )

                if (
                    query_plan is not None
                    and query_plan.valid
                    and query_plan.in_collection_scope
                    and query_plan.search_queries
                ):
                    atomic_query_texts = list(query_plan.search_queries)
                    # Atomic axes protect precision, but their conjunction can
                    # hide records whose prose expresses the relationship
                    # across several fields. For plans with three or more
                    # axes, reserve one remaining slot for a combined semantic
                    # query. Collection search treats an over-specified atomic
                    # expression as evidence-backed dense recall, after which
                    # the same source-bound audits still decide admission.
                    semantic_synthesis = query_plan.semantic_query
                    if not semantic_synthesis and len(atomic_query_texts) >= 3:
                        semantic_synthesis = re.sub(
                            r"\s+",
                            " ",
                            " ".join(atomic_query_texts),
                        ).strip()[:180]
                    semantic_synthesis = semantic_synthesis.strip()[:240]
                    if semantic_synthesis.casefold() in {
                        query.casefold() for query in atomic_query_texts
                    }:
                        semantic_synthesis = ""
                    if semantic_synthesis and len(atomic_query_texts) >= max_queries:
                        # Explicit axes own the bounded slots. A synthesis is
                        # optional recall and must not silently replace the
                        # visitor's final comparison leg at the query cap.
                        semantic_synthesis = ""
                    planned_query_texts = [
                        *([semantic_synthesis] if semantic_synthesis else []),
                        *atomic_query_texts,
                    ][:max_queries]
                    remaining = remaining_time()
                    mandatory_audit_reserve = min(
                        per_audit_cap,
                        max(mandatory_audit_floor, remaining * 0.55),
                    )
                    search_budget = min(
                        8.0,
                        max(0.0, remaining - mandatory_audit_reserve),
                    )
                    if search_budget >= 0.5:
                        planned_agendas = [
                            agenda.model_copy(update={"question": query})
                            for query in planned_query_texts
                        ]
                        planning_deadline = perf_counter() + search_budget

                        def search_planned_queries() -> list[list[SearchResult]]:
                            batch_search = getattr(
                                self.collections,
                                "search_many",
                                None,
                            )
                            if callable(batch_search):
                                if isinstance(
                                    self.collections,
                                    CollectionRepository,
                                ):
                                    batches = batch_search(
                                        planned_agendas,
                                        collection,
                                        deadline=planning_deadline,
                                        atomic=True,
                                        filters=self._recall_filters(query_plan),
                                    )
                                    if semantic_synthesis and batches:
                                        batches[0] = [
                                            replace(
                                                result,
                                                retrieval_sources=tuple(
                                                    dict.fromkeys(
                                                        (
                                                            *result.retrieval_sources,
                                                            "agentic_semantic_synthesis",
                                                        )
                                                    )
                                                ),
                                            )
                                            for result in batches[0]
                                        ]
                                    return batches
                                return batch_search(planned_agendas, collection)
                            return [
                                self.collections.search(planned, collection)
                                for planned in planned_agendas
                            ]

                        try:
                            planned_batches = await self._run_retrieval_work(
                                search_planned_queries,
                                timeout_seconds=search_budget,
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                "optional pre-audit catalogue planning search timed out"
                            )
                            planned_batches = []
                        except CollectionDataError as error:
                            logger.warning(
                                "optional pre-audit catalogue planning search skipped "
                                "(%s: %s)",
                                type(error).__name__,
                                error,
                            )
                            planned_batches = []
                        except ValueError as error:
                            logger.warning(
                                "optional pre-audit catalogue planning search skipped "
                                "(%s: %s)",
                                type(error).__name__,
                                error,
                            )
                            planned_batches = []

                        nonempty_batches = [
                            batch for batch in planned_batches if batch
                        ]
                        query_batches.extend(nonempty_batches)
                        if nonempty_batches:
                            initial_results = fuse_search_results(
                                [initial_results, *nonempty_batches],
                                limit=self.settings.rag_max_results,
                                rrf_k=self.settings.rag_rrf_k,
                            )
                            initial_results = self._query_coverage_order(
                                initial_results,
                                nonempty_batches,
                                top_k=top_k,
                            )
                            initial_results = await self._rerank_async(
                                agenda.question,
                                initial_results,
                                deadline=retrieval_deadline,
                            )
                            preplanned_queries.extend(
                                query
                                for query, batch in zip(
                                    planned_query_texts,
                                    planned_batches,
                                    strict=False,
                                )
                                if batch
                            )
                elif query_plan is not None and not query_plan.valid:
                    logger.warning(
                        "optional retrieval query plan returned an invalid contract"
                    )

        stage_results["pre_audit"] = list(initial_results)
        # Reserve time for a second audit only when the first pass asks for
        # expansion. Most well-formed queries complete after this single call.
        remaining = remaining_time()
        if remaining < 1.0:
            return AgenticRetrievalOutcome(
                results=[],
                failure_code="RETRIEVAL_AUDIT_UNAVAILABLE",
                coverage_gap=(
                    "The shared retrieval deadline expired before relevance audit."
                    if agenda.language == "en"
                    else "共享检索时限在相关性审查开始前已用尽。"
                ),
            )
        # The first audit is mandatory; expansion is conditional. Give the
        # mandatory pass all currently available time up to its own cap, then
        # decide from the real remaining budget whether a requested expansion
        # can run. Reserving most of the deadline for a hypothetical second
        # pass caused valid no-expansion audits to time out prematurely.
        first_budget = min(per_audit_cap, remaining)
        try:
            first = await run_audit(
                initial_results,
                pass_number=1,
                timeout_seconds=first_budget,
            )
        except (ProviderError, ValueError, TypeError, KeyError) as error:
            logger.warning(
                "retrieval audit unavailable (%s: %s); failing closed",
                type(error).__name__,
                error,
            )
            return AgenticRetrievalOutcome(
                results=[],
                failure_code="RETRIEVAL_AUDIT_UNAVAILABLE",
                coverage_gap=(
                    "The semantic relevance audit was unavailable, so no unreviewed candidate was allowed into the exhibition."
                    if agenda.language == "en"
                    else "语义相关性审查暂不可用；本次没有让未经审查的候选进入展览。"
                ),
            )
        if not first.valid:
            logger.warning(
                "retrieval audit returned an invalid contract; failing closed"
            )
            return AgenticRetrievalOutcome(
                results=[],
                failure_code="RETRIEVAL_AUDIT_INVALID",
                coverage_gap=(
                    "The semantic relevance audit returned an invalid contract, so no unreviewed candidate was allowed into the exhibition."
                    if agenda.language == "en"
                    else "语义相关性审查返回无效结构；本次没有让未经审查的候选进入展览。"
                ),
            )
        def apply_set_gate(audit: RetrievalAudit) -> RetrievalAudit:
            requirements = query_plan.exhibition_set_requirements if query_plan else ()
            coverage = set_coverage(requirements, audit.accepted, question=agenda.question)
            if coverage["satisfied"]:
                return audit
            # A pool may supply enough relevant background objects without
            # supplying the visitor's core observation. Keep those legitimate
            # objects, but do not let their count bypass bounded recovery.
            return replace(
                audit,
                answerability=(AnswerabilityStatus.PARTIALLY_SUPPORTED.value
                               if audit.answerability == AnswerabilityStatus.SUPPORTED.value
                               else audit.answerability),
                coverage_gap=set_coverage_gap(coverage, agenda.language),
                expansion_reason="predicate_evidence_gap",
            )

        first = self._enforce_audited_object_count(
            first,
            required_count=required_count,
            language=agenda.language,
        )
        first = self._enforce_audited_cultural_coverage(agenda, first)
        first = apply_set_gate(first)
        should_expand = bool(first.search_queries) and (
            first.expansion_reason in EXPANDABLE_REASONS
        )
        # Inspect the already-paid-for recall pool before issuing more network
        # searches. A low-ranked, source-rich comparison leg may never have
        # entered the first bounded window. Similarity is not approval: every
        # newly surfaced object still goes through the same evidence audit.
        first_sample_ids = {
            result.obj.id for result in stage_results.get("audit_sample_1", [])
        }
        unreviewed_pool = [
            result for result in initial_results
            if result.obj.id not in first_sample_ids
        ]
        missing_obligations = [
            obligation
            for obligation in self._cultural_coverage_obligations(agenda.question)
            if not any(self._object_satisfies_cultural_obligation(result.obj, obligation)
                       for result in first.accepted)
        ]
        recoverable_pool_gap = bool(unreviewed_pool) and (
            len(first.accepted) < required_count or missing_obligations
            or not set_coverage(
                query_plan.exhibition_set_requirements if query_plan else (),
                first.accepted, question=agenda.question,
            )["satisfied"]
        )
        if recoverable_pool_gap and remaining_time() >= min(10.0, per_audit_cap):
            # Focus the second window on missing named regions first, followed
            # by unseen source-rich candidates. Never spend the same window
            # re-approving the first pass's accepted objects.
            unreviewed_pool.sort(key=lambda result: (
                not any(self._object_satisfies_cultural_obligation(result.obj, leg)
                        for leg in missing_obligations),
                not any(chunk.source_kind == "institution_curatorial_text"
                        and len(chunk.text.strip()) >= 40 for chunk in result.obj.evidence),
            ))
            try:
                recovered = await run_audit(
                    unreviewed_pool, pass_number=2,
                    timeout_seconds=min(per_audit_cap, remaining_time()),
                )
                if recovered.valid:
                    first = merge_pool_audits(first, recovered, agenda.language)
                    first = self._enforce_audited_object_count(
                        first, required_count=required_count, language=agenda.language,
                    )
                    first = self._enforce_audited_cultural_coverage(agenda, first)
                    first = apply_set_gate(first)
                    stage_results["pool_recovery_accepted"] = list(first.accepted)
                    should_expand = bool(first.search_queries) and (
                        first.expansion_reason in EXPANDABLE_REASONS
                    )
                else:
                    audit_diagnostics["poolRecoveryFailure"] = {"code": "invalid_contract"}
            except (ProviderError, ValueError, TypeError, KeyError) as error:
                audit_diagnostics["poolRecoveryFailure"] = {"code": "unavailable", "errorType": type(error).__name__}
                logger.warning("Unseen-pool evidence audit unavailable; keeping first pass: %s", error)
        named_cultural_obligations = self._cultural_coverage_obligations(
            agenda.question
        )
        generic_cultural_roots = {
            root
            for result in first.accepted
            if (root := self._canonical_object_origin(result.obj))
        }
        generic_cultural_minimum_met = bool(
            not named_cultural_obligations
            and _requests_cross_cultural(agenda.question)
            and len(generic_cultural_roots) >= 3
        )
        usable_partial = bool(
            len(first.accepted) >= required_count
            and set_coverage(
                query_plan.exhibition_set_requirements if query_plan else (),
                first.accepted, question=agenda.question,
            )["satisfied"]
            and first.answerability
            == AnswerabilityStatus.PARTIALLY_SUPPORTED.value
            and (
                first.expansion_reason != "missing_cultural_leg"
                or generic_cultural_minimum_met
            )
        )
        if (
            len(first.accepted) >= required_count
            and first.answerability == AnswerabilityStatus.SUPPORTED.value
        ) or usable_partial or not should_expand:
            return AgenticRetrievalOutcome(
                results=first.accepted,
                audit_applied=True,
                expanded_queries=tuple(preplanned_queries),
                answerability=first.answerability,
                interpretation=first.interpretation,
                coverage_gap=first.coverage_gap,
                expansion_reason=first.expansion_reason,
            )

        result_sets = [initial_results]
        used_queries: list[str] = list(preplanned_queries)
        preplanned_keys = {query.casefold() for query in preplanned_queries}
        remaining_query_slots = max(0, max_queries - len(used_queries))
        planned_queries = tuple(
            query
            for query in first.search_queries
            if query.casefold() not in preplanned_keys
        )[:remaining_query_slots]
        # Encode all expansion queries in one provider batch. They still pass
        # independently through BM25, semantic floors, evidence reranking and
        # the second audit; batching removes repeated query-model startup. The
        # worker checks a cooperative deadline between scans (the active ONNX
        # kernel itself cannot be interrupted by asyncio cancellation).
        remaining = remaining_time()
        minimum_atomic_search = 2.0
        minimum_second_audit = min(
            per_audit_cap,
            max(8.0, per_audit_cap * 0.65),
        )
        second_audit_reserve = min(
            per_audit_cap,
            max(minimum_second_audit, remaining * 0.50),
        )
        search_budget = remaining - second_audit_reserve
        if planned_queries and search_budget < minimum_atomic_search:
            warning_detail = (
                "The shared retrieval deadline left no time for the requested catalogue expansion."
                if agenda.language == "en"
                else "共享检索时限不足以执行证据审查提出的扩展检索。"
            )
            logger.warning("optional agentic expansion skipped: %s", warning_detail)
            return AgenticRetrievalOutcome(
                results=first.accepted,
                audit_applied=True,
                expanded_queries=tuple(used_queries),
                answerability=first.answerability,
                interpretation=first.interpretation,
                coverage_gap=first.coverage_gap,
                expansion_reason=first.expansion_reason,
                warning_code="RETRIEVAL_SEARCH_TIMEOUT",
                warning_detail=warning_detail,
            )
        if planned_queries:
            expanded_agendas = [
                agenda.model_copy(update={"question": query})
                for query in planned_queries
            ]
            expansion_deadline = perf_counter() + search_budget

            def search_expansions() -> list[list[SearchResult]]:
                batch_search = getattr(self.collections, "search_many", None)
                if callable(batch_search):
                    if isinstance(self.collections, CollectionRepository):
                        return batch_search(
                            expanded_agendas,
                            collection,
                            deadline=expansion_deadline,
                            atomic=True,
                            filters=(
                                self._recall_filters(query_plan)
                                if query_plan is not None
                                else FilterSpec()
                            ),
                        )
                    return batch_search(expanded_agendas, collection)
                return [
                    self.collections.search(expanded_agenda, collection)
                    for expanded_agenda in expanded_agendas
                ]

            try:
                expanded_batches = await self._run_retrieval_work(
                    search_expansions,
                    timeout_seconds=search_budget,
                )
            except asyncio.TimeoutError as error:
                logger.warning(
                    "batched agentic retrieval timed out (%s: %s)",
                    type(error).__name__,
                    error,
                )
                return AgenticRetrievalOutcome(
                    results=first.accepted,
                    audit_applied=True,
                    expanded_queries=tuple(used_queries),
                    answerability=first.answerability,
                    interpretation=first.interpretation,
                    coverage_gap=first.coverage_gap,
                    expansion_reason=first.expansion_reason,
                    warning_code="RETRIEVAL_SEARCH_TIMEOUT",
                    warning_detail=(
                        "The bounded catalogue expansion timed out; the original question was retained."
                        if agenda.language == "en"
                        else "扩展馆藏检索超时；原问题已保留，本次没有把超时误报为主题不受支持。"
                    ),
                )
            except CollectionDataError as error:
                if error.code == "RETRIEVAL_SEARCH_TIMEOUT":
                    logger.warning(
                        "batched agentic retrieval reached its cooperative deadline"
                    )
                    return AgenticRetrievalOutcome(
                        results=first.accepted,
                        audit_applied=True,
                        expanded_queries=tuple(used_queries),
                        answerability=first.answerability,
                        interpretation=first.interpretation,
                        coverage_gap=first.coverage_gap,
                        expansion_reason=first.expansion_reason,
                        warning_code="RETRIEVAL_SEARCH_TIMEOUT",
                        warning_detail=(
                            "The bounded catalogue expansion timed out; the original question was retained."
                            if agenda.language == "en"
                            else "扩展馆藏检索超时；原问题已保留，本次没有把超时误报为主题不受支持。"
                        ),
                    )
                logger.warning(
                    "agentic retrieval batch skipped (%s: %s)",
                    type(error).__name__,
                    error,
                )
                expanded_batches = []
            except ValueError as error:
                logger.warning(
                    "agentic retrieval batch skipped (%s: %s)",
                    type(error).__name__,
                    error,
                )
                expanded_batches = []
            for query, expanded in zip(
                planned_queries,
                expanded_batches,
                strict=False,
            ):
                if expanded:
                    result_sets.append(expanded)
                    used_queries.append(query)

        if len(result_sets) == 1:
            return AgenticRetrievalOutcome(
                results=first.accepted,
                audit_applied=True,
                expanded_queries=tuple(used_queries),
                answerability=first.answerability,
                interpretation=first.interpretation,
                coverage_gap=first.coverage_gap,
                expansion_reason=first.expansion_reason,
            )

        fused = fuse_search_results(
            result_sets,
            limit=self.settings.rag_max_results,
            rrf_k=self.settings.rag_rrf_k,
        )
        query_batches.extend(expanded_batches)
        fused = self._query_coverage_order(
            fused,
            expanded_batches,
            top_k=top_k,
        )
        fused = await self._rerank_async(
            agenda.question,
            fused,
            deadline=retrieval_deadline,
        )
        # Optional expansion must not erase evidence the mandatory first pass
        # already verified.  RRF can demote a first-pass object when it appears
        # only in the original wording but not in a translated query.  Keep
        # those objects at the front of the second audit window; they are still
        # re-audited and therefore receive no automatic approval.
        first_accepted_ids = {result.obj.id for result in first.accepted}
        if first_accepted_ids:
            fused = [
                *first.accepted,
                *[
                    result
                    for result in fused
                    if result.obj.id not in first_accepted_ids
                ],
            ][: self.settings.rag_max_results]
        stage_results["expanded_retrieval"] = list(fused)
        remaining_budget = remaining_time()
        if remaining_budget < 1.0:
            warning_detail = (
                "The shared retrieval deadline expired before expanded evidence could be audited."
                if agenda.language == "en"
                else "扩展候选尚未完成证据复审，共享检索时限已用尽。"
            )
            logger.warning("optional expanded evidence audit skipped: %s", warning_detail)
            return AgenticRetrievalOutcome(
                results=first.accepted,
                audit_applied=True,
                expanded_queries=tuple(used_queries),
                answerability=first.answerability,
                interpretation=first.interpretation,
                coverage_gap=first.coverage_gap,
                expansion_reason=first.expansion_reason,
                warning_code="RETRIEVAL_AUDIT_UNAVAILABLE",
                warning_detail=warning_detail,
            )
        try:
            second = await run_audit(
                fused,
                pass_number=3 if "audit_sample_2" in stage_results else 2,
                timeout_seconds=min(per_audit_cap, remaining_budget),
            )
        except (ProviderError, ValueError, TypeError, KeyError) as error:
            logger.warning(
                "expanded retrieval audit unavailable (%s: %s); retaining first audit",
                type(error).__name__,
                error,
            )
            return AgenticRetrievalOutcome(
                results=first.accepted,
                audit_applied=True,
                expanded_queries=tuple(used_queries),
                answerability=first.answerability,
                interpretation=first.interpretation,
                coverage_gap=first.coverage_gap,
                expansion_reason=first.expansion_reason,
                warning_code="RETRIEVAL_AUDIT_UNAVAILABLE",
                warning_detail=(
                    "Expanded candidates could not be audited, so they were not used."
                    if agenda.language == "en"
                    else "扩展候选未能完成证据复审，因此没有进入展览。"
                ),
            )
        if not second.valid:
            logger.warning(
                "expanded retrieval audit returned an invalid contract; retaining first audit"
            )
            return AgenticRetrievalOutcome(
                results=first.accepted,
                audit_applied=True,
                expanded_queries=tuple(used_queries),
                answerability=first.answerability,
                interpretation=first.interpretation,
                coverage_gap=first.coverage_gap,
                expansion_reason=first.expansion_reason,
                warning_code="RETRIEVAL_AUDIT_INVALID",
                warning_detail=(
                    "The expanded evidence audit returned an invalid contract."
                    if agenda.language == "en"
                    else "扩展证据复审返回了无效结构。"
                ),
            )
        second = self._enforce_audited_object_count(
            second,
            required_count=required_count,
            language=agenda.language,
        )
        chosen = self._enforce_audited_cultural_coverage(agenda, second)
        chosen = apply_set_gate(chosen)
        return AgenticRetrievalOutcome(
            results=chosen.accepted,
            audit_applied=True,
            expanded_queries=tuple(used_queries),
            answerability=chosen.answerability,
            interpretation=chosen.interpretation or first.interpretation,
            # An empty current gap is meaningful; never resurrect a superseded
            # negative claim from an earlier, smaller audit window.
            coverage_gap=chosen.coverage_gap,
            expansion_reason=first.expansion_reason,
        )

    @staticmethod
    def _recall_filters(plan: RetrievalQueryPlan | None) -> FilterSpec:
        """Natural-language material classes are not interoperable field codes.

        Keep date, origin and other explicit constraints, but defer material
        entailment to the strict per-object audit. A superclass such as plant
        fibre otherwise excludes rattan before the model can see its record.
        The original plan is unchanged and all its material conditions still
        bind admission. Explicit FilterSpec API callers are unaffected.
        """
        return replace(plan.filters, materials=()) if plan and plan.valid else FilterSpec()

    @staticmethod
    def _dedupe_audit_candidates(
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        """Collapse exact duplicated institution descriptions for auditing.

        Museum APIs often expose a parent work plus separately addressable
        parts with the same explanatory text. Showing all of them to the model
        spends a bounded audit window without adding another evidence case.
        Only long, exact normalized descriptions from the same institution are
        folded; distinct records, short tombstones and later selection remain
        untouched.
        """

        seen: set[tuple[str, str]] = set()
        unique: list[SearchResult] = []
        for result in candidates:
            description = re.sub(
                r"\s+",
                " ",
                result.obj.description or "",
            ).strip().casefold()
            key = (result.obj.institution_id, description)
            if len(description) >= 120 and key in seen:
                continue
            if len(description) >= 120:
                seen.add(key)
            unique.append(result)
        return unique

    @staticmethod
    def _query_coverage_order(
        fused: list[SearchResult],
        query_batches: list[list[SearchResult]],
        *,
        top_k: int,
    ) -> list[SearchResult]:
        """Reserve audit-window candidates from every atomic query axis.

        RRF correctly rewards consensus, but several independent evidence legs
        are not expected to share objects. Without a small per-query reserve,
        generic candidates common to all rankings can fill the whole audit
        window. This only reorders already-retrieved candidates; the evidence
        auditor still decides relevance and validates every source ID.
        """

        batches = [batch for batch in query_batches if batch]
        if not batches or len(fused) <= 1 or top_k <= 1:
            return fused
        fused_by_id = {result.obj.id: result for result in fused}
        # A single precise query may be the only route to the requested
        # predicate, so expose enough of its head to build a five-object room.
        # With several independent axes, retain the smaller per-axis reserve
        # so no one query can consume the whole audit window.
        if len(batches) == 1:
            quotas = [min(5, max(1, top_k // 3))]
        else:
            ordinary_quota = 2 if top_k >= len(batches) * 2 + 3 else 1
            quotas = [
                min(4, max(ordinary_quota, top_k // 5))
                if any(
                    "agentic_semantic_synthesis" in result.retrieval_sources
                    for result in batch[:5]
                )
                else ordinary_quota
                for batch in batches
            ]
        reserve = min(top_k - 1, sum(quotas))
        head_count = max(1, top_k - reserve)
        selected = list(fused[:head_count])
        selected_ids = {result.obj.id for result in selected}

        for offset in range(max(quotas, default=0)):
            for batch, quota in zip(batches, quotas, strict=True):
                if offset >= quota:
                    continue
                candidate = next(
                    (
                        fused_by_id[result.obj.id]
                        for result in batch[offset:]
                        if result.obj.id in fused_by_id
                        and result.obj.id not in selected_ids
                    ),
                    None,
                )
                if candidate is not None:
                    selected.append(candidate)
                    selected_ids.add(candidate.obj.id)
                if len(selected) >= top_k:
                    break
            if len(selected) >= top_k:
                break

        selected.extend(
            result for result in fused if result.obj.id not in selected_ids
        )
        return selected

    @classmethod
    def _audit_candidate_sample(
        cls,
        candidates: list[SearchResult],
        *,
        top_k: int,
        cross_cultural: bool,
        cultural_obligations: list[CulturalCoverageObligation] | None = None,
        query_batches: list[list[SearchResult]] | None = None,
    ) -> list[SearchResult]:
        """Keep retrieval rank while reserving room for origin diversity.

        Dense/BM25 fusion often places many records from one well-described
        institution first. For a cross-cultural question, showing only that
        head to the auditor makes missing cultures unrecoverable even when they
        are already in the recall pool. This sampler changes no relevance
        score and approves nothing; it only exposes a bounded, more informative
        candidate set to the evidence auditor.
        """

        obligations = cultural_obligations or []
        if top_k <= 0:
            return []
        if len(candidates) <= top_k:
            return candidates[:top_k]
        by_id = {result.obj.id: result for result in candidates}
        candidate_rank = {result.obj.id: rank for rank, result in enumerate(candidates)}

        def quote_ready(result: SearchResult) -> bool:
            matched = set(result.matched_evidence_ids)
            return any(
                chunk.source_kind != "institution_provenance"
                and len((chunk.text or "").strip()) >= 40
                and (chunk.id in matched or chunk.source_kind == "institution_curatorial_text")
                for chunk in result.obj.evidence
            )
        protected: list[SearchResult] = []
        protected_ids: set[str] = set()

        def protect(result: SearchResult | None) -> None:
            if (
                result is not None
                and result.obj.id not in protected_ids
                and len(protected) < max(0, top_k - 1)
            ):
                protected.append(result)
                protected_ids.add(result.obj.id)

        # Reserve after global reranking and deduplication. Query association
        # is a recall reason, not proof of relevance; current evidence and
        # scores remain unchanged and all candidates still need audit.
        for obligation in obligations:
            if not any(
                cls._object_satisfies_cultural_obligation(result.obj, obligation)
                for result in protected
            ):
                protect(next((
                    result for result in candidates
                    if cls._object_satisfies_cultural_obligation(result.obj, obligation)
                ), None))
        # One regional slot can be consumed by a thin high-rank title while a
        # directly described object from that same region is hidden. Preserve
        # a second, quote-ready representative per explicitly requested leg.
        for obligation in obligations:
            rich_leg = [result for result in candidates
                        if result.obj.id not in protected_ids
                        and cls._object_satisfies_cultural_obligation(result.obj, obligation)
                        and quote_ready(result)]
            if rich_leg:
                protect(min(rich_leg, key=lambda result: (
                    not any(chunk.source_kind == "institution_curatorial_text"
                            for chunk in result.obj.evidence),
                    -(result.evidence_score if result.evidence_score is not None else -1.0),
                    candidate_rank[result.obj.id],
                )))
        batches = [batch for batch in (query_batches or []) if batch]
        axis_quota = 2 if top_k >= len(batches) * 2 + len(protected) + 3 else 1
        batch_ranks = [
            {result.obj.id: rank for rank, result in enumerate(batch[:50])}
            for batch in batches
        ]
        for offset in range(axis_quota):
            for index, batch in enumerate(batches):
                if offset == 1 and len(batches) > 1:
                    # The same generic head may occur in every query. Reserve
                    # the second slot for a candidate ranked disproportionately
                    # well by this axis, without treating uniqueness as proof.
                    # Inspect only a bounded head; the auditor still validates
                    # the actual relation and exact institutional evidence.
                    def axis_advantage(result: SearchResult) -> tuple[float, int]:
                        rank = batch_ranks[index].get(result.obj.id, 50)
                        own = 1.0 / (rank + 1)
                        others = [
                            1.0 / (ranks[result.obj.id] + 1) if result.obj.id in ranks else 0.0
                            for position, ranks in enumerate(batch_ranks) if position != index
                        ]
                        return own - max(others, default=0.0), -rank

                    distinct = [
                        result for result in batch[:50]
                        if result.obj.id in by_id and result.obj.id not in protected_ids
                        and axis_advantage(result)[0] > 0
                    ]
                    if distinct:
                        protect(by_id[max(distinct, key=axis_advantage).obj.id])
                        continue
                protect(next((
                    by_id[result.obj.id] for result in batch[offset:]
                    if result.obj.id in by_id and result.obj.id not in protected_ids
                ), None))
        # Keep most of the highest-ranked candidates. Four reserved slots are
        # enough to surface three or more comparison regions without letting
        # low-ranked catalogue geography crowd out semantic precision.
        diversity_slots = (
            min(
                max(4, len(obligations)),
                max(1, top_k // 3),
            )
            if cross_cultural
            else 0
        )
        # Every open-vocabulary question, not only a cross-cultural one, needs
        # quote-ready candidates. A thin title match can rank highly yet be
        # impossible to admit under the exact-source contract. Reserve a small
        # window for candidates with a matched institution row or substantial
        # curatorial evidence. This exposes evidence; it never approves it.
        candidate_rank = {
            result.obj.id: rank for rank, result in enumerate(candidates)
        }

        def quote_ready(result: SearchResult) -> bool:
            matched = set(result.matched_evidence_ids)
            for chunk in result.obj.evidence:
                if chunk.source_kind == "institution_provenance":
                    continue
                text = (chunk.text or "").strip()
                if len(text) < 40:
                    continue
                if chunk.id in matched or chunk.source_kind == "institution_curatorial_text":
                    return True
            return False

        source_rich = [result for result in candidates if quote_ready(result)]
        source_rich.sort(
            key=lambda result: (
                not bool(result.matched_evidence_ids),
                -(result.evidence_score or -1.0),
                candidate_rank[result.obj.id],
                result.obj.id,
            )
        )
        evidence_slots = min(
            len(source_rich),
            max(0, min(6, top_k // 3)),
        )
        head_count = max(1, top_k - len(protected) - diversity_slots - evidence_slots)
        selected = list(candidates[:head_count])
        selected_ids = {result.obj.id for result in selected}
        for result in protected:
            if result.obj.id not in selected_ids and len(selected) < top_k:
                selected.append(result)
                selected_ids.add(result.obj.id)
        added_evidence = 0
        for result in source_rich:
            if len(selected) >= top_k or added_evidence >= evidence_slots:
                break
            if result.obj.id in selected_ids:
                continue
            selected.append(result)
            selected_ids.add(result.obj.id)
            added_evidence += 1
            if added_evidence >= evidence_slots:
                break
        seen_origins = {
            origin
            for result in selected
            if (origin := cls._canonical_object_origin(result.obj))
        }
        # A generic origin-diversity slot can still miss a culture explicitly
        # named by the visitor. Reserve the first available topical candidate
        # for every named leg before filling generic regional diversity.
        for obligation in obligations:
            if len(selected) >= top_k:
                break
            if any(
                cls._object_satisfies_cultural_obligation(
                    result.obj,
                    obligation,
                )
                for result in selected
            ):
                continue
            candidate = next(
                (
                    result
                    for result in candidates[head_count:]
                    if result.obj.id not in selected_ids
                    and cls._object_satisfies_cultural_obligation(
                        result.obj,
                        obligation,
                    )
                ),
                None,
            )
            if candidate is not None:
                selected.append(candidate)
                selected_ids.add(candidate.obj.id)
                if origin := cls._canonical_object_origin(candidate.obj):
                    seen_origins.add(origin)
        for result in candidates[head_count:]:
            if len(selected) >= top_k:
                break
            origin = cls._canonical_object_origin(result.obj)
            if origin and origin not in seen_origins:
                selected.append(result)
                selected_ids.add(result.obj.id)
                seen_origins.add(origin)
        for result in candidates[head_count:]:
            if len(selected) >= top_k:
                break
            if result.obj.id not in selected_ids:
                selected.append(result)
                selected_ids.add(result.obj.id)
        return selected

    def _enforce_audited_cultural_coverage(
        self,
        agenda: AgendaInput,
        audit: RetrievalAudit,
    ) -> RetrievalAudit:
        """Do not let five objects from one culture answer a comparison.

        The model judges semantic relevance; this deterministic postcondition
        checks whether the accepted pool actually contains the comparison legs
        named by the visitor.  It never adds objects or upgrades a model
        decision, and therefore remains a constraint rather than a topic rule.
        """

        if audit.answerability != AnswerabilityStatus.SUPPORTED.value:
            return audit

        named = self._cultural_coverage_obligations(agenda.question)
        missing = [
            obligation
            for obligation in named
            if not any(
                self._object_satisfies_cultural_obligation(result.obj, obligation)
                for result in audit.accepted
            )
        ]
        gap = ""
        if missing:
            labels = "、".join(obligation.label_zh for obligation in missing)
            gap = (
                f"The audited pool is missing requested cultural leg(s): {labels}."
                if agenda.language == "en"
                else f"语义审查后的候选仍缺少明确要求的文化比较腿：{labels}。"
            )
        elif not named and _requests_cross_cultural(agenda.question):
            culture_buckets: set[str] = set()
            for result in audit.accepted:
                origin = self._canonical_object_origin(result.obj)
                if origin:
                    culture_buckets.add(origin)
            if len(culture_buckets) < 3:
                gap = (
                    "The audited pool covers fewer than three cultural regions, so it cannot support the requested cross-cultural comparison."
                    if agenda.language == "en"
                    else "语义审查后的候选不足三个文化区域，不能支撑所要求的跨文化比较。"
                )

        if not gap:
            return audit
        combined_gap = " ".join(
            part for part in (audit.coverage_gap.strip(), gap) if part
        )[:500]
        return replace(
            audit,
            answerability=AnswerabilityStatus.PARTIALLY_SUPPORTED.value,
            coverage_gap=combined_gap,
        )

    @staticmethod
    def _enforce_audited_object_count(
        audit: RetrievalAudit,
        *,
        required_count: int,
        language: str,
    ) -> RetrievalAudit:
        """A model cannot declare support with too few evidence-bound IDs."""

        if (
            audit.answerability != AnswerabilityStatus.SUPPORTED.value
            or len(audit.accepted) >= required_count
        ):
            return audit
        count_gap = (
            f"Only {len(audit.accepted)} evidence-bound objects passed audit; "
            f"the visit requires {required_count}."
            if language == "en"
            else f"只有 {len(audit.accepted)} 件藏品通过证据绑定审查，"
            f"当前参观需要 {required_count} 件。"
        )
        return replace(
            audit,
            answerability=AnswerabilityStatus.PARTIALLY_SUPPORTED.value,
            coverage_gap=" ".join(
                part for part in (audit.coverage_gap.strip(), count_gap) if part
            )[:500],
        )

    @staticmethod
    def _retrieval_contract(results: list[SearchResult]) -> tuple[str, str]:
        """Record the retrieval route that actually supplied this candidate set."""

        sources = {
            source for result in results for source in result.retrieval_sources
        }
        if "llm_relevance_audit" in sources:
            return AGENTIC_RETRIEVAL_METHOD, AGENTIC_RETRIEVAL_VERSION
        if "question_card" in sources:
            return QUESTION_CARD_RETRIEVAL_METHOD, QUESTION_CARD_RETRIEVAL_VERSION
        if (
            any(source.startswith("dense") for source in sources)
            or sources.intersection(
                {"bm25_evidence", "evidence_rerank", "qwen3_rerank"}
            )
        ):
            return HYBRID_RETRIEVAL_METHOD, HYBRID_RETRIEVAL_VERSION
        return BM25_RETRIEVAL_METHOD, BM25_RETRIEVAL_VERSION

    @staticmethod
    def probe_answerability(
        collections: CollectionRepository,
        agenda: AgendaInput,
        *,
        audit_available: bool = False,
    ) -> AgendaCheckResponse:
        """Run the gate without constructing a generator.

        Used by the interview to decide whether to open a negotiation turn, so
        it must not construct a model provider. The caller supplies the real
        runtime capability: when an audit is unavailable, dense-only nearest
        neighbours are removed before the interview makes any promise.
        """
        probe = ExhibitionGenerator.__new__(ExhibitionGenerator)
        probe.collections = collections
        probe.provider = None  # type: ignore[assignment]
        probe.settings = None  # type: ignore[assignment]
        probe._audit_available_override = audit_available
        collection = collections.get(agenda.collection_id)
        results = collections.search(agenda, collection)
        return ExhibitionGenerator.check_agenda(
            probe,
            agenda,
            results_override=results,
        )

    @staticmethod
    def _results_with_predicate_evidence(
        results: list[SearchResult],
        pattern: re.Pattern[str],
    ) -> list[SearchResult]:
        """Keep only objects whose institution text supports the asked action.

        Acquisition snippets are excluded: a credit line can document how the
        museum received an object, but it cannot support a conservation,
        ritual-practice or present-day community claim.
        """

        qualified: list[SearchResult] = []
        for result in results:
            evidence_ids = [
                chunk.id
                for chunk in result.obj.evidence
                if chunk.source_kind != "institution_provenance"
                and pattern.search(" ".join(filter(None, (chunk.text, chunk.supports))))
            ]
            if evidence_ids:
                qualified.append(
                    replace(
                        result,
                        matched_evidence_ids=tuple(dict.fromkeys(evidence_ids)),
                    )
                )
        return qualified

    @classmethod
    def _predicate_assessment(
        cls,
        question: str,
        results: list[SearchResult],
    ) -> PredicateAssessment | None:
        """Gate a few high-cost predicates that topic counts cannot answer."""

        bronze = re.search(r"青铜(?:器)?|\bbronze\b", question, re.IGNORECASE)
        corrosion = re.search(
            r"生锈|锈蚀|腐蚀|\b(?:corrosion|rust(?:ed|ing)?)\b",
            question,
            re.IGNORECASE,
        )
        repair = re.search(
            r"修复|修补|保育|保护处理|\b(?:restor(?:ation|ed)?|conserv(?:ation|ed)?|repair(?:ed)?|treat(?:ment|ed)?)\b",
            question,
            re.IGNORECASE,
        )
        if bronze and corrosion and repair:
            matched = cls._results_with_predicate_evidence(
                results,
                BRONZE_REPAIR_EVIDENCE,
            )
            matched = [
                result
                for result in matched
                if re.search(
                    r"\b(?:bronze|copper\s+alloy)\b|青铜|铜合金",
                    " ".join(
                        filter(
                            None,
                            (
                                result.obj.title,
                                result.obj.medium,
                                result.obj.material,
                                result.obj.type,
                                result.obj.classification,
                            ),
                        )
                    ),
                    re.IGNORECASE,
                )
            ]
            return PredicateAssessment(
                status=AnswerabilityStatus.PARTIALLY_SUPPORTED,
                matched=matched,
                gap=(
                    "馆藏可定位少数青铜器的已记录锈蚀或修补，但不能据此解释锈蚀机理，"
                    "也不足以概括不同文化的修复方法。可改问“馆方如何记录这些器物的修补／保存痕迹？”"
                ),
            )

        museum_context = re.search(
            r"入馆|进入博物馆|博物馆|馆藏|展陈|展示|\bmuseum\b",
            question,
            re.IGNORECASE,
        )
        sacredness = re.search(
            r"神圣性|神圣|\bsacredness\b|\b(?:remain|still)\s+sacred\b",
            question,
            re.IGNORECASE,
        )
        continuing_claim = re.search(
            r"能否|还能|是否|保持|失去|延续|改变|吗|\b(?:remain|still|continue|lose)\b",
            question,
            re.IGNORECASE,
        )
        if museum_context and sacredness and continuing_claim:
            matched = cls._results_with_predicate_evidence(
                results,
                RITUAL_PRACTICE_EVIDENCE,
            )
            return PredicateAssessment(
                status=(
                    AnswerabilityStatus.PARTIALLY_SUPPORTED
                    if matched
                    else AnswerabilityStatus.UNSUPPORTED
                ),
                matched=matched,
                gap=(
                    "馆方记录可支持历史礼仪用途的比较，但不能由物件记录判断今天是否仍被相关社群视为神圣；"
                    "这需要社群观点与实际展陈语境材料。"
                ),
            )

        colonial_or_looted = re.search(
            r"殖民(?:时期|主义|统治)?|被殖民|\bcolonial(?:ism)?\b|\bcolonized\b|\blooted\b|掠夺|流失",
            question,
            re.IGNORECASE,
        )
        normative = re.search(
            r"该不该|应不应该|是否应该|要不要|\bshould\b",
            question,
            re.IGNORECASE,
        )
        restitution = re.search(
            r"归还|返还|遣返|\brepatriat(?:e|ion|ed)\b|\brestitut(?:ion|e|ed)\b|\breturn\b",
            question,
            re.IGNORECASE,
        )
        if colonial_or_looted and normative and restitution:
            return PredicateAssessment(
                status=AnswerabilityStatus.UNSUPPORTED,
                matched=[],
                gap=(
                    "馆藏可呈现部分来源与入藏记录，但不能仅凭这些记录裁定是否归还；"
                    "该判断需要原属社群、权属与采集情境、法律及机构政策材料。"
                ),
            )

        return None

    def _hard_generation_boundary(
        self,
        agenda: AgendaInput,
        collection: LoadedCollection,
        results: list[SearchResult],
    ) -> PredicateAssessment | None:
        """Return only evidence boundaries that more retrieval cannot repair.

        Raw candidate count and missing cultural legs are intentionally absent:
        the bounded agent may recover those with translated or decomposed
        queries. Exact reviewed negative policies, prohibited requests and
        predicates requiring external legal/clinical/community evidence cannot
        be repaired by finding more visually similar museum objects.
        """

        policy_match = self.collections.match_question_policy(
            collection, agenda.question
        )
        if policy_match is not None and policy_match[1] == 1.0:
            policy = policy_match[0]
            if policy.status != AnswerabilityStatus.SUPPORTED.value:
                return PredicateAssessment(
                    status=AnswerabilityStatus(policy.status),
                    matched=results,
                    gap=(
                        policy.rationale
                        or "；".join(policy.coverage_limits)
                        or "经审定的问题边界不支持按当前问法生成展览。"
                    ),
                )

        # A fuzzy question-card match may supply useful retrieval vocabulary,
        # but it cannot erase an added legal, conservation or living-sacredness
        # predicate. Only the exact negative policy above is a complete reviewed
        # boundary in its own right.
        predicate = self._predicate_assessment(agenda.question, results)
        if predicate is not None:
            return predicate

        if any(
            re.search(pattern, agenda.question, re.IGNORECASE)
            for pattern in UNSUPPORTED_BOUNDARY_PATTERNS
        ):
            return PredicateAssessment(
                status=AnswerabilityStatus.UNSUPPORTED,
                matched=[],
                gap="该问题需要馆藏之外的数据、效果证据或产品明确禁止的能力。",
            )
        # Broad or causal wording is repairable by narrowing the answer. Let
        # the evidence audit return partially_supported with an explicit
        # boundary instead of rejecting before QueryPlan and retrieval run.
        return None

    def check_agenda(
        self,
        agenda: AgendaInput,
        *,
        results_override: list[SearchResult] | None = None,
    ) -> AgendaCheckResponse:
        collection = self.collections.get(agenda.collection_id)
        results = (
            results_override
            if results_override is not None
            else self.collections.search(agenda, collection)
        )
        policy_match = self.collections.match_question_policy(
            collection, agenda.question
        )
        policy_similarity = policy_match[1] if policy_match is not None else 0.0
        exact_policy = (
            policy_match[0]
            if policy_match is not None and policy_similarity == 1.0
            else None
        )
        audit_available_override = getattr(
            self, "_audit_available_override", None
        )
        audit_available = (
            bool(audit_available_override)
            if audit_available_override is not None
            else bool(
                self.provider is not None
                and self.provider.configured
                and getattr(self.provider, "supports_retrieval_audit", False)
                and self.settings is not None
                and self.settings.rag_llm_audit_enabled
            )
        )
        query_plan = _query_plan(
            agenda.question,
            getattr(collection, "concept_aliases", {}),
        )
        required_count = VisitorProfile(
            durationMinutes=agenda.duration_minutes
        ).item_count
        eligible_objects = self.collections.eligible_objects(collection)
        eligible_ids = {obj.id for obj in eligible_objects}
        exact_starter_count = (
            sum(
                object_id in eligible_ids
                for object_id in exact_policy.starter_object_ids
            )
            if exact_policy is not None
            else 0
        )
        policy_usable_without_audit = bool(
            exact_policy is not None
            and (
                (
                    exact_policy.source == "question_card"
                    and exact_policy.status == AnswerabilityStatus.SUPPORTED.value
                    and exact_starter_count >= required_count
                )
                or exact_policy.status != AnswerabilityStatus.SUPPORTED.value
            )
        )
        audit_required_but_unavailable = bool(
            not audit_available
            and not query_plan.browse_all
            and not policy_usable_without_audit
        )
        if not audit_available:
            if query_plan.browse_all:
                results = [
                    result
                    for result in results
                    if "browse_all" in result.retrieval_sources
                ]
            elif audit_required_but_unavailable:
                results = []
            else:
                results = [
                    result
                    for result in results
                    if "bm25" in result.retrieval_sources
                    or "question_card" in result.retrieval_sources
                    or "browse_all" in result.retrieval_sources
                ]
        eligible_count = len(eligible_objects)
        # A near match is useful retrieval vocabulary, not a reviewed answer to
        # the visitor's changed question.  Only exact policy text may supply a
        # frozen status, evidence domain or starter set.  This keeps an added
        # legal/conservation/living-practice predicate from inheriting the
        # answerability of a simpler public question card.
        policy = (
            exact_policy
            if exact_policy is not None
            and policy_usable_without_audit
            else None
        )
        predicate = (
            None
            if policy is not None
            else self._predicate_assessment(agenda.question, results)
        )
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
                if len(policy_objects) < required_count:
                    status = AnswerabilityStatus.PARTIALLY_SUPPORTED
                    gaps.insert(
                        0,
                        f"当前参观时长需要 {required_count} 件藏品，"
                        f"预验证问题卡只有 {len(policy_objects)} 件可用角色藏品。",
                    )
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
        elif predicate is not None:
            status = predicate.status
            matched = predicate.matched
            gaps = [predicate.gap]
        elif audit_required_but_unavailable:
            status = AnswerabilityStatus.UNSUPPORTED
            gaps = [
                (
                    "The evidence relevance audit is temporarily unavailable. "
                    "This does not mean the collection lacks the topic; retry when the audit service is available."
                    if agenda.language == "en"
                    else "证据相关性审查暂不可用；这不代表馆藏没有这个主题，请在审查服务恢复后重试。"
                )
            ]
        elif len(results) < 5:
            status = AnswerabilityStatus.UNSUPPORTED
            gaps = ["与问题主题直接匹配的可用藏品不足五件；系统不会用无关藏品补位。"]
        elif any(re.search(pattern, agenda.question, re.IGNORECASE) for pattern in UNSUPPORTED_BOUNDARY_PATTERNS):
            status = AnswerabilityStatus.UNSUPPORTED
            gaps = ["该问题需要馆藏之外的数据、效果证据或产品明确禁止的能力。"]
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
        result_sources = {
            source for result in results for source in result.retrieval_sources
        }
        frozen_question_card = bool(
            policy is not None
            and policy_similarity == 1.0
            and policy.source == "question_card"
            and status == AnswerabilityStatus.SUPPORTED
            and len(policy_objects) >= required_count
        )
        exact_negative_policy = bool(
            policy is not None
            and policy_similarity == 1.0
            and status != AnswerabilityStatus.SUPPORTED
        )
        if frozen_question_card:
            decision_basis = "reviewed_question_card"
        elif exact_negative_policy:
            decision_basis = "reviewed_policy"
        elif predicate is not None:
            decision_basis = "predicate_boundary"
        elif results and all("browse_all" in result.retrieval_sources for result in results):
            decision_basis = "browse"
        elif audit_required_but_unavailable:
            decision_basis = "audit_unavailable"
        elif "dense_open_query" in result_sources and audit_available:
            decision_basis = "open_dense_provisional"
        elif can_generate and audit_available:
            # Known lexical aliases and non-frozen regression positives still
            # enter the same runtime evidence audit as open dense candidates.
            # Calling their hit count "reviewed" here would disagree with the
            # generation path and recreate a late-failure UX.
            decision_basis = "runtime_audit_provisional"
        else:
            decision_basis = "lexical_retrieval"
        requires_runtime_audit = bool(
            can_generate
            and decision_basis
            in {"open_dense_provisional", "runtime_audit_provisional"}
        )
        return AgendaCheckResponse(
            status=status,
            can_generate=can_generate,
            requires_runtime_audit=requires_runtime_audit,
            decision_basis=decision_basis,
            exhibition_theme=exhibition_theme,
            answerable_part=(
                "初步召回到足量候选，仍需逐件核查馆方证据。"
                if requires_runtime_audit
                else "可用馆藏能够组成五件藏品的证据链。"
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
        """Match one leg from controlled origin, never from auxiliary facets.

        ``culturePackIds`` is multi-valued in the imported corpus: a Chinese
        object can carry an auxiliary Europe or Americas facet. Such a facet
        cannot make the same object stand in for two origins. A single pack is
        used only when the institution supplied no controlled origin text.
        """

        controlled_origin = " ".join(
            value
            for value in (
                obj.culture,
                obj.culture_display,
                obj.place,
            )
            if value
        )
        if any(
            re.search(pattern, controlled_origin, re.IGNORECASE)
            for pattern in obligation.origin_patterns
        ):
            return True
        if controlled_origin and any(
            re.search(pattern, controlled_origin, re.IGNORECASE)
            for pattern in obligation.question_patterns
        ):
            return True
        if controlled_origin and obligation.culture_pack_ids:
            controlled_regions = {
                region
                for region, pattern in CANONICAL_ORIGIN_REGIONS
                if pattern.search(controlled_origin)
            }
            if controlled_regions.intersection(obligation.culture_pack_ids):
                return True
        return bool(
            not controlled_origin
            and len(obj.culture_pack_ids) == 1
            and obj.culture_pack_ids[0] in obligation.culture_pack_ids
        )

    def _ensure_final_cultural_coverage(
        self,
        agenda: AgendaInput,
        selected: list[MuseumObject],
        pool: list[SearchResult],
        *,
        allow_repair: bool,
    ) -> list[MuseumObject]:
        """Keep audited comparison legs in the actual room, not only its pool."""

        obligations = self._cultural_coverage_obligations(agenda.question)
        if not obligations:
            if not _requests_cross_cultural(agenda.question):
                return selected
            origins = {
                value
                for obj in selected
                if (value := self._canonical_object_origin(obj))
            }
            if len(origins) >= 3:
                return selected
            raise CollectionDataError(
                "CROSS_CULTURAL_SELECTION_INSUFFICIENT",
                "The final room retained fewer than three audited cultural regions.",
                selectedRegionCount=len(origins),
                requiredRegionCount=3,
            )

        working = list(selected)

        def covered(objects: list[MuseumObject]) -> set[int]:
            return {
                index
                for index, obligation in enumerate(obligations)
                if any(
                    self._object_satisfies_cultural_obligation(obj, obligation)
                    for obj in objects
                )
            }

        if allow_repair:
            chosen_ids = {obj.id for obj in working}
            for result in pool:
                before = covered(working)
                if len(before) == len(obligations):
                    break
                candidate = result.obj
                candidate_legs = {
                    index
                    for index, obligation in enumerate(obligations)
                    if self._object_satisfies_cultural_obligation(
                        candidate,
                        obligation,
                    )
                }
                if candidate.id in chosen_ids or not (candidate_legs - before):
                    continue
                full_count = sum(
                    obj.evidence_depth == EvidenceDepth.FULL.value for obj in working
                )
                victim_index = next(
                    (
                        index
                        for index in range(len(working) - 1, -1, -1)
                        if (
                            working[index].evidence_depth
                            != EvidenceDepth.FULL.value
                            or candidate.evidence_depth == EvidenceDepth.FULL.value
                            or full_count > 1
                        )
                        and len(
                            covered(
                                [
                                    candidate if offset == index else obj
                                    for offset, obj in enumerate(working)
                                ]
                            )
                        )
                        > len(before)
                    ),
                    None,
                )
                if victim_index is None:
                    continue
                chosen_ids.remove(working[victim_index].id)
                working[victim_index] = self._prioritized_object(result)
                chosen_ids.add(candidate.id)

        final_covered = covered(working)
        if len(final_covered) != len(obligations):
            missing = "、".join(
                obligation.label_zh
                for index, obligation in enumerate(obligations)
                if index not in final_covered
            )
            raise CollectionDataError(
                "CROSS_CULTURAL_SELECTION_INSUFFICIENT",
                "The final room did not retain every explicitly requested cultural leg.",
                missingCulturalLegs=missing,
            )
        return working

    def _ensure_final_set_coverage(
        self, agenda: AgendaInput, selected: list[MuseumObject],
        results: list[SearchResult], requirements: tuple[dict[str, Any], ...], *,
        allow_repair: bool = True,
    ) -> tuple[list[MuseumObject], dict[str, Any]]:
        """Retain source-checked witnesses in the actual itinerary, not just recall.

        Repairs replace only with audited objects. Each swap improves set-goal
        coverage without sacrificing another goal, named culture, or the last
        full record. If bounded repair cannot satisfy the contract, fail closed.
        """
        working = list(selected)
        obligations = self._cultural_coverage_obligations(agenda.question)

        def coverage(objects):
            return set_coverage(requirements, results, question=agenda.question,
                                selected_ids={obj.id for obj in objects})

        def progress(report):
            return [min(row["minWitnesses"], row["witnessCount"])
                    for row in report["requirements"]]

        def cultural_axes(objects):
            return {index for index, obligation in enumerate(obligations)
                    if any(self._object_satisfies_cultural_obligation(obj, obligation)
                           for obj in objects)}

        current = coverage(working)
        if allow_repair and not current["satisfied"]:
            # At most three requirements, each with at most three witnesses.
            # Reconsider the pool after every successful swap, but never search
            # combinatorially or keep cycling among equal-quality replacements.
            for _ in range(sum(row["minWitnesses"] for row in requirements)):
                if current["satisfied"]:
                    break
                before = progress(current)
                protected_cultures = cultural_axes(working)
                roots_before = {self._canonical_object_origin(obj) for obj in working} - {""}
                full_before = sum(obj.evidence_depth == EvidenceDepth.FULL.value for obj in working)
                best = None
                best_progress = sum(before)
                selected_ids = {obj.id for obj in working}
                for result in results:
                    if result.obj.id in selected_ids or not result.set_witnesses:
                        continue
                    candidate = self._prioritized_object(result)
                    for index in range(len(working) - 1, -1, -1):
                        trial = [candidate if offset == index else obj
                                 for offset, obj in enumerate(working)]
                        after_coverage = coverage(trial)
                        after = progress(after_coverage)
                        if sum(after) <= best_progress or any(a < b for a, b in zip(after, before)):
                            continue
                        if not protected_cultures.issubset(cultural_axes(trial)):
                            continue
                        if full_before and not any(obj.evidence_depth == EvidenceDepth.FULL.value for obj in trial):
                            continue
                        if not obligations and _requests_cross_cultural(agenda.question):
                            roots_after = {self._canonical_object_origin(obj) for obj in trial} - {""}
                            if len(roots_after) < min(3, len(roots_before)):
                                continue
                        best = (trial, after_coverage)
                        best_progress = sum(after)
                if best is None:
                    break
                working, current = best
        if not current["satisfied"]:
            raise CollectionDataError(
                "QUESTION_UNSUPPORTED_AFTER_AUDIT",
                "The final selected set does not contain the required source-checked core witnesses.",
                coverageGap=set_coverage_gap(current, agenda.language),
                exhibitionSetCoverage=current,
            )
        return working, current

    @staticmethod
    def _canonical_object_origin(obj: MuseumObject) -> str:
        """Return one normalized catalogue-backed culture region per object."""

        controlled_origin = next(
            (
                value
                for value in (obj.culture, obj.culture_display, obj.place)
                if value and value.strip()
            ),
            "",
        )
        if controlled_origin:
            compact = re.sub(r"\s+", " ", controlled_origin).strip()
            for region, pattern in CANONICAL_ORIGIN_REGIONS:
                if pattern.search(compact):
                    return region
        pack_roots = {
            pack.strip().casefold()
            for pack in obj.culture_pack_ids
            if pack.strip().casefold() in CANONICAL_ORIGIN_PACKS
        }
        if len(pack_roots) == 1:
            return next(iter(pack_roots))
        # Unknown catalogue strings are not cultural regions. A date, street,
        # city, workshop or period label may occupy a source's culture/place
        # field; returning that raw value falsely turns five records from one
        # country into five "regions". Unknowns therefore remain uncounted
        # until a controlled region pattern or one unambiguous culture pack is
        # available.
        return ""

    async def generate(self, agenda: AgendaInput) -> Exhibition:
        collection = self.collections.get(agenda.collection_id)
        retrieval_deadline = perf_counter() + float(
            self.settings.rag_retrieval_timeout_seconds
        )
        initial = await self.prepare_initial_retrieval(
            agenda,
            collection,
            deadline=retrieval_deadline,
        )
        initial_results = initial.results
        hard_boundary = self._hard_generation_boundary(
            agenda,
            collection,
            initial_results,
        )
        if hard_boundary is not None:
            raise CollectionDataError(
                "QUESTION_UNSUPPORTED",
                "The current collection cannot support a five-object evidence chain for this question.",
                status=hard_boundary.status,
                coverageGaps=[hard_boundary.gap],
                recommendedQuestions=self.collections.recommend_questions(collection),
            )
        retrieval_outcome = await self._agentic_retrieve(
            agenda,
            collection,
            initial_results,
            required_count=5,
            deadline=retrieval_deadline,
            initial_query_plan=initial.query_plan,
            planning_attempted=True,
        )
        if retrieval_outcome.failure_code:
            raise CollectionDataError(
                retrieval_outcome.failure_code,
                "The semantic relevance audit did not complete successfully.",
                coverageGap=retrieval_outcome.coverage_gap,
            )
        if retrieval_outcome.answerability is not None and (
            len(retrieval_outcome.results) < 5
            or retrieval_outcome.answerability
            == AnswerabilityStatus.UNSUPPORTED.value
        ):
            raise CollectionDataError(
                "QUESTION_UNSUPPORTED_AFTER_AUDIT",
                "The relevance audit could not verify five collection objects for this question.",
                availableObjectCount=len(retrieval_outcome.results),
                answerability=retrieval_outcome.answerability,
                coverageGap=retrieval_outcome.coverage_gap or None,
                expandedQueries=list(retrieval_outcome.expanded_queries),
            )
        if len(retrieval_outcome.results) < 5:
            raise CollectionDataError(
                (
                    "QUESTION_UNSUPPORTED_AFTER_AUDIT"
                    if retrieval_outcome.audit_applied
                    else "QUESTION_UNSUPPORTED"
                ),
                "The retrieval gate could not verify five collection objects for this question.",
                availableObjectCount=len(retrieval_outcome.results),
                coverageGap=retrieval_outcome.coverage_gap or None,
                expandedQueries=list(retrieval_outcome.expanded_queries),
            )
        context = self._context(agenda, all_results=retrieval_outcome.results)
        if (
            retrieval_outcome.answerability
            == AnswerabilityStatus.PARTIALLY_SUPPORTED.value
        ):
            partial_boundary = retrieval_outcome.coverage_gap or (
                "馆藏记录只能支持问题中的具体对象案例；本展不把这些案例扩大为普遍结论。"
            )
            context = replace(
                context,
                question_card_limits=list(
                    dict.fromkeys(
                        [*context.question_card_limits, partial_boundary]
                    )
                ),
            )
        final_selected = self._ensure_final_cultural_coverage(
            agenda,
            context.selected,
            context.results,
            allow_repair=not bool(retrieval_outcome.forced_object_ids),
        )
        final_selected, final_set_coverage = self._ensure_final_set_coverage(
            agenda, final_selected, context.results, retrieval_outcome.exhibition_set_requirements,
            allow_repair=not bool(retrieval_outcome.forced_object_ids),
        )
        retrieval_outcome.audit_diagnostics["finalSetCoverage"] = final_set_coverage
        final_ids = {obj.id for obj in final_selected}
        context = replace(
            context,
            selected=final_selected,
            remaining_results=[
                result
                for result in context.results
                if result.obj.id not in final_ids
            ],
        )
        exhibition = self._deterministic_exhibition(agenda, context)
        exhibition.exhibition_set_coverage = final_set_coverage
        if (
            retrieval_outcome.coverage_gap
            and retrieval_outcome.coverage_gap not in exhibition.coverage_limits
        ):
            # A pool-level audit note may describe candidates not finally
            # selected. Keep its exact wording in the trace, not as a factual
            # assertion about the finished exhibition.
            exhibition.coverage_limits.append(
                "This route uses the verified examples retrieved for this visit; it does not establish exhaustive cultural coverage."
                if agenda.language == "en" else "本展呈现本次检索并核实的例子，不声称涵盖相关文化的全部情况。"
            )
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
        preference_browse = _is_affective_browse_profile(profile, agenda.question)
        retrieval_agenda = (
            agenda.model_copy(
                update={
                    "question": (
                        "surprise me"
                        if profile.language == "en"
                        else "随便带我逛逛"
                    )
                }
            )
            if preference_browse
            else agenda
        )
        collection = self.collections.get(collection_id)
        retrieval_deadline = perf_counter() + float(
            self.settings.rag_retrieval_timeout_seconds
        )
        initial = await self.prepare_initial_retrieval(
            retrieval_agenda,
            collection,
            deadline=retrieval_deadline,
        )
        initial_results = initial.results
        editorial_constraints = tuple(getattr(initial.query_plan, "editorial_constraints", ()))
        # Only non-recoverable boundaries fail before the model. Sparse recall
        # and missing cultural legs continue into agentic query expansion.
        hard_boundary = self._hard_generation_boundary(
            retrieval_agenda,
            collection,
            initial_results,
        )
        if hard_boundary is not None:
            raise CollectionDataError(
                "QUESTION_UNSUPPORTED",
                "The current collection cannot support the requested evidence chain.",
                status=hard_boundary.status,
                coverageGaps=[hard_boundary.gap],
                recommendedQuestions=self.collections.recommend_questions(collection),
            )
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
        retrieval_outcome = await self._agentic_retrieve(
            retrieval_agenda,
            collection,
            initial_results,
            required_count=profile.item_count,
            deadline=retrieval_deadline,
            initial_query_plan=initial.query_plan,
            planning_attempted=True,
        )
        if retrieval_outcome.failure_code:
            raise CollectionDataError(
                retrieval_outcome.failure_code,
                "The semantic relevance audit did not complete successfully.",
                coverageGap=retrieval_outcome.coverage_gap,
            )
        all_results = retrieval_outcome.results
        domain_id = profile.curiosity_domain_id
        pool = self._filter_evidence_domain(all_results, domain_id)
        # A narrow free-text question can starve the pool; widen before failing.
        if len(pool) < profile.item_count:
            pool = self._filter_evidence_domain(all_results, None)
        if retrieval_outcome.answerability is not None and (
            retrieval_outcome.answerability
            == AnswerabilityStatus.UNSUPPORTED.value
        ):
            raise CollectionDataError(
                "QUESTION_UNSUPPORTED_AFTER_AUDIT",
                "The retrieval gate found topical objects but insufficient evidence for the question's requested claim.",
                requestedItemCount=profile.item_count,
                availableObjectCount=len(pool),
                answerability=retrieval_outcome.answerability,
                coverageGap=retrieval_outcome.coverage_gap or None,
                expandedQueries=list(retrieval_outcome.expanded_queries),
            )
        if len(pool) < profile.item_count:
            raise CollectionDataError(
                (
                    "QUESTION_UNSUPPORTED_AFTER_AUDIT"
                    if retrieval_outcome.audit_applied
                    else "COLLECTION_DATA_INSUFFICIENT"
                ),
                (
                    "The relevance audit could not verify enough collection objects for this question."
                    if retrieval_outcome.audit_applied
                    else "Not enough eligible objects remain after exclusions to fill this exhibition."
                ),
                requestedItemCount=profile.item_count,
                availableObjectCount=len(pool),
                coverageGap=retrieval_outcome.coverage_gap or None,
                expandedQueries=list(retrieval_outcome.expanded_queries),
            )

        if retrieval_outcome.forced_object_ids:
            pool_by_id = {result.obj.id: result for result in pool}
            forced_results = [
                pool_by_id[object_id]
                for object_id in retrieval_outcome.forced_object_ids
                if object_id in pool_by_id
            ]
            if len(forced_results) != profile.item_count:
                raise CollectionDataError(
                    "QUESTION_CARD_INCOMPLETE",
                    "The reviewed question-card spine is not available in the current filtered pool.",
                    requiredObjectIds=list(retrieval_outcome.forced_object_ids),
                    availableObjectIds=[result.obj.id for result in forced_results],
                )
            objects = [
                self._prioritized_object(result) for result in forced_results
            ]
        else:
            objects = curation.order_for_narrative(
                pool,
                profile.item_count,
                prefer_culture_diversity=_requests_cross_cultural(
                    retrieval_agenda.question
                ),
            )
        objects = self._ensure_final_cultural_coverage(
            retrieval_agenda,
            objects,
            pool,
            allow_repair=not bool(retrieval_outcome.forced_object_ids),
        )
        objects, final_set_coverage = self._ensure_final_set_coverage(
            retrieval_agenda, objects, pool, retrieval_outcome.exhibition_set_requirements,
            allow_repair=not bool(retrieval_outcome.forced_object_ids),
        )
        retrieval_outcome.audit_diagnostics["finalSetCoverage"] = final_set_coverage
        full_depth = sum(
            1 for obj in objects if obj.evidence_depth == EvidenceDepth.FULL.value
        )
        topic = profile.curiosity_label or profile.free_form_question or (
            "these objects" if profile.language == "en" else "这批藏品"
        )
        eligible = len(self.collections.eligible_objects(collection))
        if retrieval_outcome.audit_applied:
            audit_detail = (
                f" Semantic audit retained {len(all_results)} directly relevant objects"
                if en
                else f"；语义审查保留 {len(all_results)} 件直接相关藏品"
            )
            if retrieval_outcome.expanded_queries:
                audit_detail += (
                    f" after {len(retrieval_outcome.expanded_queries)} bounded query expansions."
                    if en
                    else f"，并执行 {len(retrieval_outcome.expanded_queries)} 条受限扩展检索。"
                )
            elif en:
                audit_detail += "."
            else:
                audit_detail += "。"
        else:
            audit_detail = ""
        if preference_browse:
            retrieval_finding = (
                f"From {eligible} eligible objects, {len(objects)} were selected "
                "as an exploratory, unhurried route. This is a preference, not "
                "evidence that an object has a therapeutic effect."
                if en
                else f"从 {eligible} 件可用藏品中选定 {len(objects)} 件，"
                "编成一条可慢慢看的探索路线；这是对当下偏好的回应，"
                "不声称藏品具有疗愈或放松效果。"
            )
        else:
            retrieval_finding = (
                f"From {eligible} eligible objects, {len(pool)} relate to "
                f"“{topic}”; {len(objects)} selected, {full_depth} of them with "
                f"an institution-written description.{audit_detail}"
                if en
                else f"从 {eligible} 件可用藏品中筛出 "
                f"{len(pool)} 件与「{topic}」相关，选定 {len(objects)} 件，"
                f"其中 {full_depth} 件带机构撰写的说明{audit_detail}"
            )
        await step("retrieve", retrieval_finding)

        # -- deterministic skeleton --------------------------------------
        exhibition = self._profile_skeleton(
            profile, agenda, collection, objects, domain_id, pool
        )
        exhibition.exhibition_set_coverage = final_set_coverage
        if (
            retrieval_outcome.coverage_gap
            and retrieval_outcome.coverage_gap not in exhibition.coverage_limits
        ):
            exhibition.coverage_limits.append(
                "This route uses the verified examples retrieved for this visit; it does not establish exhaustive cultural coverage."
                if en else "本展呈现本次检索并核实的例子，不声称涵盖相关文化的全部情况。"
            )
        if preference_browse:
            preference_limit = (
                "This exploratory route responds to the visitor's desired pace; "
                "it does not claim a therapeutic or universal emotional effect."
                if en
                else "这条探索路线回应的是参观者当下想放慢节奏的偏好；"
                "不声称任何藏品具有疗愈性或普遍情绪效果。"
            )
            if preference_limit not in exhibition.coverage_limits:
                exhibition.coverage_limits.append(preference_limit)

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
            # All selected objects eventually need image-grounded labels,
            # including record_explanation visits that correctly skipped the
            # optional preselection visual audit. Use frame latency to warm
            # their public images without adding a serial wait or a job budget.
            selected_image_prewarm = (
                asyncio.create_task(prewarm_visual_candidates(
                    [item.object for item in exhibition.items], self.image_cache,
                    max_candidates=12, timeout_seconds=12.0,
                ))
                if self.image_cache is not None else None
            )
            try:
                frame_deadline = perf_counter() + min(
                    self.settings.deepseek_timeout_seconds,
                    self.settings.deepseek_frame_timeout_seconds,
                )
                frame = await self._generate_model_json(
                    curation.frame_prompt(exhibition.agenda.language) + "\nSource boundaries apply to titles and relations too: depicted location is not maker perspective; do not substitute a famous city for the documented location or transfer a neighbouring object's city. Overlapping date intervals cannot establish earlier/later. A material inventory does not map materials to specific parts or prove how decoration was applied. No images are supplied to this frame task: do not invent detailed visible features such as facial parts, motif anatomy or construction details. Those belong to the later single-object visual label pass. Frame text should point the visitor to documented, concrete things to notice, not anticipate what unseen pixels show. Use chapter membership exactly as supplied. If uncertain, point to one documented detail of the object itself rather than inventing a comparison.",
                    {**curation.frame_payload(
                        plan,
                        exhibition.items,
                        exhibition.chapters,
                        exhibition.curatorial_brief,
                        evidence_boundaries=exhibition.coverage_limits,
                    ), "visitorQuestion": agenda.question,
                     "editorialConstraints": list(editorial_constraints), "exhibitionSetGoals": [
                        {"id": row["id"], "visitorGoal": row["sourceQuote"],
                         "witnessObjectIds": row["witnessObjectIds"],
                         "boundary": "This is an editorial focus and witness routing, not new institution evidence. Guide attention to these objects without extending their source claims to all objects."}
                        for row in final_set_coverage["requirements"]
                    ]},
                    stage="frame",
                    timeout_seconds=min(
                        self.settings.deepseek_timeout_seconds,
                        max(0.001, self.settings.deepseek_frame_timeout_seconds
                            - min(24.0, self.settings.deepseek_frame_timeout_seconds * 0.45)),
                    ),
                )
                if not isinstance(frame.get("curatorialBrief"), dict):
                    raise ValueError("frame output missing required curatorialBrief")
                # Generated inventories are not published: navigation is bound
                # to actual stop IDs again after title translation. Do not let
                # discarded lead-in prose invalidate the remaining frame.
                frame = deepcopy(frame)
                raw_chapters = frame.get("chapters")
                if isinstance(raw_chapters, list) and len(raw_chapters) == len(exhibition.chapters):
                    for raw, chapter in zip(raw_chapters, exhibition.chapters):
                        if isinstance(raw, dict):
                            # Navigation is rebuilt from final IDs and translated
                            # titles. Do not review text that is never published.
                            raw.pop("leadIn", None)
                chapter_structure = [
                    {"index": index, "itemCount": len(chapter.item_ids),
                     "objectIds": [item.object.id for item in exhibition.items
                                   if item.id in chapter.item_ids]}
                    for index, chapter in enumerate(exhibition.chapters)
                ]
                review_payload = copy_review_payload(
                    frame, objects, question=agenda.question, language=agenda.language,
                    chapter_structure=chapter_structure,
                    editorial_constraints=editorial_constraints,
                )
                reviewed = await self._review_frame_copy(
                    frame, review_payload, language=agenda.language,
                    deadline=frame_deadline,
                )
                if not reviewed.review_passed:
                    raise ValueError(f"public frame copy review did not pass: {reviewed.status}")
                frame = reviewed.frame
                framed_exhibition = curation.apply_frame(exhibition, frame)
                frame_validation = validate_exhibition(framed_exhibition)
                blocking_copy_codes = {
                    issue.code
                    for issue in frame_validation.errors
                    if issue.code == "PUBLIC_COPY_ITEM_COUNT_MISMATCH"
                }
                if blocking_copy_codes:
                    raise ValueError(
                        "frame output states an item count that disagrees with "
                        "the selected exhibition structure"
                    )
                exhibition = framed_exhibition
                exhibition.versions.provider = "deepseek"
                if reviewed.locally_neutralized_fields:
                    exhibition.coverage_limits.append(
                        "少量策展解释未完成来源核对，已替换为中性观察提示或明确的核对状态说明；这些局部降级内容不作为历史事实。"
                    )
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
            finally:
                if selected_image_prewarm is not None:
                    if not selected_image_prewarm.done():
                        selected_image_prewarm.cancel()
                    # Cancellation applies to async waiters. Any already active
                    # image download retains its bounded worker and original
                    # 12-second cooperative deadline; it starts no new work.
                    warmed = await asyncio.gather(selected_image_prewarm, return_exceptions=True)
                    if isinstance(warmed[0], dict):
                        logger.info("selected image prewarm cached=%s count=%s elapsed=%s",
                                    warmed[0].get("cachedCount", 0), len(exhibition.items),
                                    warmed[0].get("elapsedSeconds", 0))

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

        # A usable deterministic frame is enough context for source-bound
        # labels. Frame rejection must not suppress an independent visual pass.
        labels_attempted = self.provider is not None and self.provider.configured
        if labels_attempted:
            labelled_items = await self._write_labels(
                exhibition, profile, editorial_constraints=editorial_constraints,
            )
            if labelled_items:
                exhibition.versions.labels_model = self.settings.deepseek_labels_model
                if not model_applied:
                    exhibition.coverage_limits = [
                        ("本次展览框架使用确定性模板；展签单独生成并保留来源约束。"
                         if limit == "本次策展文本由确定性模板生成；模型输出不可用，展品与来源不受影响。"
                         else limit)
                        for limit in exhibition.coverage_limits
                    ]

        # Navigation is a rendering of the final selected stops, not another
        # opportunity for the model to invent chapter membership or materials.
        curation.bind_chapter_navigation(exhibition)
        bound = sum(len(item.label_sentences) for item in exhibition.items)
        named = sum(1 for item in exhibition.items if item.display_title)
        if labelled_items:
            detail = (
                f"{labelled_items}/{len(exhibition.items)} objects written by the visual model"
                if en
                else f"{labelled_items}/{len(exhibition.items)} 件由视觉模型撰写"
            )
        elif labels_attempted:
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


    async def _review_frame_copy(self, frame, payload, *, language: str, deadline: float):
        """Review small field groups, bounded by the original frame deadline.

        Each group keeps all selected sources for comparisons. Two concurrent
        reviews limit provider pressure; queued work gets no fresh timeout.
        Nothing is applied until all groups satisfy the same source contract.
        """
        batches = split_copy_review_payload(payload)
        slots = asyncio.Semaphore(2)
        prompt = concise_copy_review_prompt(language)
        locally_neutralized: set[int] = set()
        auxiliary_neutralized: set[str] = set()

        async def review(index, batch):
            async with slots:
                # Keep the model-facing request/repair in one wire protocol.
                # The v1 payload and compiled patches are internal validation
                # objects, not examples for a v2 model response.
                wire_batch = ({**batch, "schemaVersion": "curatorial-copy-decisions-v2"}
                              if language != "en" else batch)
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    raise ProviderError("frame review deadline exhausted")
                raw_output = await self._generate_model_json(
                    prompt, wire_batch, stage=f"frame_review:batch{index}",
                    timeout_seconds=min(12.0, remaining),
                )
                output = copy_decisions_to_patches(raw_output, batch)
                initial_output = output
                parsed = parse_copy_review_batch(output, frame, batch)
                if parsed.status == "invalid" and deadline - perf_counter() >= 2.0:
                    output = await self._generate_model_json(
                        prompt + "\nCorrect the previous response's protocol/source-binding errors only. Return the complete response in the SAME schema required above, including unchanged keep decisions. invalid_noop_change means replacement repeated the original: actually remove/correct the suspected unsupported assertion, not just repeat its correction reason or declare it resolved. Use exact continuous source quotes and this batch's fields and allowedEvidenceIds. Neutralize unsupported facts; do not reverse an unresolved semantic verdict.",
                        {**wire_batch, "contractErrors": list(parsed.errors), "previousReview": raw_output},
                        stage=f"frame_review_repair:batch{index}",
                        timeout_seconds=min(8.0, deadline - perf_counter()),
                    )
                output = copy_decisions_to_patches(output, batch)
                final_check = parse_copy_review_batch(output, frame, batch)
                if not final_check.review_passed:
                    neutral = neutralize_object_review_batch(batch)
                    if neutral is not None:
                        locally_neutralized.add(index)
                        logger.info("frame object copy locally neutralized batch=%s fields=%s review_status=%s",
                                    index, batch["expectedFieldCount"], final_check.status)
                        return neutral
                    recovered = recover_auxiliary_copy_batch(output, initial_output, frame, batch)
                    if recovered is not None:
                        safe_output, paths = recovered
                        auxiliary_neutralized.update(paths)
                        return safe_output
                return output

        tasks = [asyncio.create_task(review(index, batch)) for index, batch in enumerate(batches)]
        try:
            outputs = await asyncio.wait_for(
                asyncio.gather(*tasks), max(0.001, deadline - perf_counter()),
            )
        except asyncio.TimeoutError as error:
            raise ProviderError("frame review deadline exhausted") from error
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        result = merge_copy_review_batches(outputs, frame, batches)
        # This narrowly bounds degradation; failed core thesis/title/chapter
        # reviews still invalidate the frame. Never label notices as supported
        # historical key messages just because their original bindings exist.
        if len(auxiliary_neutralized) > 2:
            return replace(result, frame=deepcopy(frame), review_passed=False, status="invalid",
                           errors=("too_many_unreviewed_auxiliary_fields",))
        if result.review_passed:
            for path in auxiliary_neutralized:
                if path.startswith("/curatorialBrief/keyMessages/"):
                    result.frame["curatorialBrief"]["keyMessages"][int(path.split("/")[3])]["confidence"] = "uncertain"
        count = (sum(batches[index]["expectedFieldCount"] for index in locally_neutralized)
                 + len(auxiliary_neutralized))
        if result.review_passed and count:
            result = replace(result, status="bounded_local_fallback", locally_neutralized_fields=count)
        return result

    async def _write_labels(
        self, exhibition: Exhibition, profile: "VisitorProfile", *,
        editorial_constraints: tuple[str, ...] = (),
    ) -> int:
        """Write and independently review one source-bound label per object.

        A single-object call prevents a bad image or malformed response from
        collapsing a whole chapter, and removes ambiguity about which pixels
        belong to which object.  Curatorial relations remain in the frame; the
        wall label itself needs only this object's image and catalogue record.
        """
        from .visual_evidence import _IMAGE_EXECUTOR, _cached_image_only
        from .label_review import (
            VISUAL_SENTENCE_CONTRACT,
            WRITER_BUDGET_FRACTION,
            checked_label_candidate,
            generate_label_with_transport_retry,
            label_review_payload,
            label_review_prompt,
        )

        # This is the already allocated label-stage budget, not a fresh budget
        # after image fetching. It remains independent of frame success.
        label_budget = min(self.settings.deepseek_timeout_seconds,
                           self.settings.deepseek_labels_timeout_seconds)
        label_deadline = perf_counter() + label_budget
        image_deadline = perf_counter() + min(8.0, label_budget * 0.25)
        image_slots = asyncio.Semaphore(4)
        by_id = {item.id: item for item in exhibition.items}
        prompt = (curation.labels_prompt(profile.label_max_chars, profile.language) + VISUAL_SENTENCE_CONTRACT
                  + "\neditorialConstraints are visitor-requested wording/evidence boundaries, not object facts or instructions to override source rules. Apply them to every label where relevant. For requested known/unknown distinctions, describe the supplied record's limits without inventing missing knowledge or proving historical absence.")
        review_prompt = label_review_prompt(profile.label_max_chars, profile.language)
        chapter_by_item = {
            item_id: chapter
            for chapter in exhibition.chapters
            for item_id in chapter.item_ids
        }
        visual_provider = callable(
            getattr(self.provider, "generate_json_with_images", None)
        )

        def unavailable_label(item: ExhibitionItem, *, missing_image: bool = False) -> None:
            # Never keep an unreviewed writer paragraph after a timeout or a
            # critic failure. This is a visible degraded state, not a passed
            # visual label; original catalogue fields remain intact.
            source_ids = [chunk.id for chunk in curation.catalogue_evidence(item.object)]
            if missing_image:
                text = ("The image could not be loaded for this label; the catalogue fields and source remain available."
                        if profile.language == "en" else "本次未能载入图像，暂不描述画面细节；可先查看馆藏信息与原始来源。")
            else:
                text = ("This label's source review did not finish; generated interpretation is withheld. The catalogue record remains available."
                        if profile.language == "en" else "这件藏品的展签尚未完成来源核对，暂不展示生成解读；可查看馆方著录与原始来源。")
            item.label_sentences = [LabelSentence(
                id=str(uuid4()), text=text, type=SentenceType.UNCERTAIN,
                evidence_ids=source_ids[:1],
            )]

        async def prepare_image(item: ExhibitionItem) -> VisionImage | None:
            visual = curation.image_evidence(item.object)
            if not visual_provider or visual is None or self.image_cache is None:
                return None
            try:
                payload = _cached_image_only(self.image_cache, visual.source_url, 1024)
                if payload is None:
                    async with image_slots:
                        if perf_counter() >= image_deadline:
                            return None
                        call = (partial(self.image_cache.get, visual.source_url, 1024,
                                        deadline=image_deadline)
                                if getattr(self.image_cache, "supports_deadline", False)
                                else partial(self.image_cache.get, visual.source_url, 1024))
                        future = _IMAGE_EXECUTOR.submit(call)
                        payload, _cached = await asyncio.wait_for(
                            asyncio.wrap_future(future), max(0.001, image_deadline - perf_counter()),
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
            except (asyncio.TimeoutError, OSError):
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
                if perf_counter() >= label_deadline:
                    unavailable_label(item, missing_image=visual_provider and vision_image is None)
                    return False
                payload = curation.labels_payload(
                    profile,
                    chapter,
                    [item],
                    exhibition.curatorial_brief,
                    available_visual_evidence_ids=available_visual_ids,
                )
                payload["visitorQuestion"] = exhibition.agenda.question
                payload["editorialConstraints"] = list(editorial_constraints)
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
                writer_started = perf_counter()
                output = await generate_label_with_transport_retry(
                    self._generate_model_json,
                    prompt,
                    payload,
                    stage=f"labels:{item.id}",
                    deadline=writer_started + (label_deadline - writer_started) * WRITER_BUDGET_FRACTION,
                    request_timeout=self.settings.deepseek_timeout_seconds,
                    vision_images=(
                        [vision_image]
                        if visual_provider and vision_image is not None
                        else ([] if visual_provider else None)
                    ),
                )
                # Reject structurally invalid drafts before paying for review,
                # but keep both prose and translations on a disposable copy.
                checked_label_candidate(
                    item, output, max_chars=profile.label_max_chars,
                    allowed_evidence_by_object=allowed_evidence_by_object,
                    visual_evidence_by_object=visual_evidence_by_object,
                    # A well-formed draft may still contain a semantic error;
                    # give the critic its one chance to remove that claim.
                    enforce_visual_scope=False,
                )
                review_payload = label_review_payload(
                    item, payload, output, vision_image,
                    max_chars=profile.label_max_chars, language=profile.language,
                )
                if perf_counter() >= label_deadline:
                    raise ProviderError("label source review deadline exhausted")
                reviewed = await generate_label_with_transport_retry(
                    self._generate_model_json,
                    review_prompt,
                    review_payload,
                    stage=f"labels_review:{item.id}",
                    deadline=label_deadline,
                    request_timeout=self.settings.deepseek_timeout_seconds,
                    # Reuse exactly the writer's image, with no second fetch
                    # or opportunity to swap a nearby object's pixels.
                    vision_images=([vision_image] if vision_image is not None
                                   else ([] if visual_provider else None)),
                )
                candidate = checked_label_candidate(
                    item, reviewed, max_chars=profile.label_max_chars,
                    # The critic may see extra original context (e.g. the
                    # photographed underside) that was absent from the draft.
                    # Only IDs actually supplied in this review are allowed.
                    allowed_evidence_by_object={item.object.id: {
                        chunk["id"] for chunk in review_payload["items"][0]["evidence"]
                    }},
                    visual_evidence_by_object=visual_evidence_by_object,
                )
                if visual_provider and vision_image is None:
                    # Missing pixels must not be laundered into a visual claim
                    # by labelling that claim "system_inference" instead.
                    # Keep only *reviewed* field translations, not speculative
                    # prose. This still does not count as a reviewed visual label.
                    item.localized_metadata = candidate.localized_metadata
                    unavailable_label(item, missing_image=True)
                    return False
                item.display_title = candidate.display_title
                item.localized_metadata = candidate.localized_metadata
                item.label_sentences = candidate.label_sentences
                return True
            except (ProviderError, ValueError, TypeError, KeyError) as error:
                logger.warning(
                    "visual label source review for object '%s' failed (%s: %s); using explicit fallback",
                    item.object.id,
                    type(error).__name__,
                    error,
                )
                unavailable_label(item, missing_image=visual_provider and vision_image is None)
                return False

        async def translate_tombstones() -> dict[str, Any] | None:
            # Shares the label stage's deadline; it must never outlive it.
            remaining = label_deadline - perf_counter()
            if remaining <= 0.05:
                return None
            try:
                return await self._generate_model_json(
                    curation.TOMBSTONES_PROMPT,
                    curation.tombstones_payload(list(by_id.values())),
                    stage="tombstones",
                    timeout_seconds=min(20.0, remaining),
                )
            except Exception as error:  # noqa: BLE001 - optional; labels stand without it
                logger.warning("tombstone translation failed (%s: %s)", type(error).__name__, error)
                return None

        # Chinese titles and tombstones must not depend on each label passing
        # its visual review; this runs alongside and only fills what is left.
        tombstones = (
            asyncio.create_task(translate_tombstones())
            if profile.language != "en"
            else None
        )
        results = await asyncio.gather(
            *(write(item) for item in by_id.values())
        )
        if tombstones is not None:
            output = await tombstones
            if output is not None:
                curation.apply_tombstone_translations(list(by_id.values()), output)
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
        # A full visitor question is context, not a poster headline. If the
        # optional frame cannot be safely used, this neutral short title still
        # fits the entrance image; the original question remains unchanged.
        if len(theme) > 32:
            theme = "A closer look" if profile.language == "en" else "从你的好奇出发"

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
                        exhibition_obj,
                        item_id,
                        role_label,
                        profile.label_max_chars,
                        profile.language,
                    ),
                    alternatives=rotated[:3],
                    order=index,
                )
            )

        limits = [
            f"本展使用 {'、'.join(dict.fromkeys(obj.institution for obj in objects if obj.institution)) or collection.institution} 的公开馆藏记录（数据版本 {collection.version}）。",
            "机构英文原文保持原样；中文策展关系属于系统推断，不冒充机构原文。",
            "材料之外的人物关系、年代因果与历史结论不会由模型常识补写。",
        ]
        thin = [obj for obj in objects if not obj.supports_core_evidence]
        if thin:
            limits.append(
                f"其中 {len(thin)} 件的馆藏记录缺少详细说明，只承担背景或对照位置。"
            )
        if any(obj.visual_core_evidence for obj in objects):
            limits.append("本展部分核心例证来自实际查看的馆藏图像，仅支持可见形态的比较；图像不证明历史原因、材质或人物身份。")

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
                prompt="v6-hybrid-source-review-2026-09-06-rc11",
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
        collection = self.collections.get(candidate_agenda.collection_id)
        retrieval_deadline = perf_counter() + float(
            self.settings.rag_retrieval_timeout_seconds
        )
        initial = await self.prepare_initial_retrieval(
            candidate_agenda,
            collection,
            deadline=retrieval_deadline,
        )
        initial_results = initial.results
        hard_boundary = self._hard_generation_boundary(
            candidate_agenda,
            collection,
            initial_results,
        )
        if hard_boundary is not None:
            raise CollectionDataError(
                "QUESTION_UNSUPPORTED",
                "The revised focus cannot be supported by five objects in this collection.",
                status=hard_boundary.status,
                coverageGaps=[hard_boundary.gap],
                recommendedQuestions=self.collections.recommend_questions(collection),
            )
        retrieval_outcome = await self._agentic_retrieve(
            candidate_agenda,
            collection,
            initial_results,
            required_count=len(exhibition.items),
            deadline=retrieval_deadline,
            initial_query_plan=initial.query_plan,
            planning_attempted=True,
        )
        if retrieval_outcome.failure_code:
            raise CollectionDataError(
                retrieval_outcome.failure_code,
                "The revised focus could not complete semantic evidence audit.",
                coverageGap=retrieval_outcome.coverage_gap,
            )
        if (
            retrieval_outcome.answerability is not None
            and retrieval_outcome.answerability
            != AnswerabilityStatus.SUPPORTED.value
        ) or len(retrieval_outcome.results) < len(exhibition.items):
            raise CollectionDataError(
                "QUESTION_UNSUPPORTED_AFTER_AUDIT",
                "The revised focus did not retain enough audited object evidence.",
                answerability=retrieval_outcome.answerability,
                availableObjectCount=len(retrieval_outcome.results),
                coverageGap=retrieval_outcome.coverage_gap or None,
            )
        revised_context = self._context(
            candidate_agenda,
            all_results=retrieval_outcome.results,
        )
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

        selected_in_order = tuple(item.object.id for item in exhibition.items)
        selected_ids = set(selected_in_order)
        # A reviewed question card freezes a five-role evidence spine, not just
        # an unordered bag of objects. Refocus does not silently reorder the
        # current visit, so both membership and order must already match.
        if (
            retrieval_outcome.forced_object_ids
            and selected_in_order != retrieval_outcome.forced_object_ids
        ):
            raise CollectionDataError(
                "FOCUS_REQUIRES_NEW_EXHIBITION",
                "The revised focus is a reviewed question card whose required evidence spine differs from the current objects.",
                currentSelectedObjectIds=sorted(selected_ids),
                requiredStarterObjectIds=sorted(
                    retrieval_outcome.forced_object_ids
                ),
            )
        revised_result_ids = {result.obj.id for result in revised_context.results}
        unsupported_selected_ids = sorted(selected_ids - revised_result_ids)
        if unsupported_selected_ids:
            raise CollectionDataError(
                "FOCUS_REQUIRES_NEW_EXHIBITION",
                "The current selected objects do not all support the revised focus; generate a new exhibition.",
                unsupportedSelectedObjectIds=unsupported_selected_ids,
            )
        _, final_set_coverage = self._ensure_final_set_coverage(
            candidate_agenda, [item.object for item in exhibition.items], revised_context.results,
            retrieval_outcome.exhibition_set_requirements, allow_repair=False,
        )
        before = {"question": exhibition.question, "title": exhibition.title}
        exhibition.exhibition_set_coverage = final_set_coverage
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
            item.label_sentences = self._label_sentences(
                item.object,
                item.id,
                item.role_label,
                item.sub_question,
                exhibition.agenda.language,
            )
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
        target.display_title = curation.display_title(replacement)
        target.localized_metadata = LocalizedObjectMetadata()
        target.label_sentences = self._label_sentences(
            replacement,
            target.id,
            target.role_label,
            target.sub_question,
            exhibition.agenda.language,
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

    def _context(
        self,
        agenda: AgendaInput,
        *,
        all_results: list[SearchResult] | None = None,
    ) -> GenerationContext:
        collection = self.collections.get(agenda.collection_id)
        results_were_supplied = all_results is not None
        if all_results is None:
            all_results = self.collections.search(agenda, collection)
        eligible_by_id = {
            obj.id: obj for obj in self.collections.require_generation_ready(collection)
        }
        policy_match = self.collections.match_question_policy(collection, agenda.question)
        # Only the exact reviewed question may force starter objects or a
        # frozen evidence domain. A fuzzy match can help retrieval upstream,
        # but substituting its starters here would bypass the LLM-audited
        # candidate set for the visitor's materially different question.
        policy = (
            policy_match[0]
            if policy_match is not None and policy_match[1] == 1.0
            else None
        )
        # Runtime retrieval already applied the source-ID-bound evidence audit.
        # A relative BM25 threshold must not discard those accepted objects a
        # second time merely because tokenization or query fusion changed their
        # numeric score. Only an internally fetched, unaudited pool needs the
        # legacy direct-score heuristic.
        direct_results = (
            list(all_results)
            if results_were_supplied
            else self._direct_results(all_results)
        )
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
        selected_title_families: dict[str, int] = {}
        title_family_cache: dict[str, str] = {}
        strongest = max((result.score for result in remaining), default=1.0)
        available_culture_packs = {
            pack for result in remaining for pack in result.obj.culture_pack_ids
        }

        def culture_root(obj: MuseumObject) -> str:
            # Use the same canonical regions as the evidence audit gate. Raw
            # labels such as China/Japan/Korea are cultures, but for the
            # product's “cross-cultural” minimum they are one East Asia region.
            return ExhibitionGenerator._canonical_object_origin(obj)

        def title_family(obj: MuseumObject) -> str:
            """Collapse trivial article/plural variants for selection only."""

            if obj.id in title_family_cache:
                return title_family_cache[obj.id]
            value = unicodedata.normalize(
                "NFKC", (obj.title or obj.title_original or "")
            ).casefold()
            english: list[str] = []
            for token in re.findall(r"[a-z0-9]+", value):
                if token in {"a", "an", "the", "of", "in", "on", "from"}:
                    continue
                if token.endswith("ies") and len(token) > 4:
                    token = token[:-3] + "y"
                elif token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
                    token = token[:-1]
                english.append(token)
            chinese = "".join(re.findall(r"[\u3400-\u9fff]+", value))
            family = "|".join(filter(None, (" ".join(english), chinese)))
            title_family_cache[obj.id] = family
            return family

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

            # Relevance can legitimately produce several variants from one
            # series, but a five-object room should not become four near-
            # identical catalogue titles when other audited evidence exists.
            # Cap a normalized title family at two until no alternative remains;
            # later cultural-leg repair may still override this soft selector.
            non_repeating = [
                result
                for result in candidates
                if selected_title_families.get(title_family(result.obj), 0) < 2
                or not title_family(result.obj)
            ]
            if non_repeating:
                candidates = non_repeating

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
            if family := title_family(best.obj):
                selected_title_families[family] = (
                    selected_title_families.get(family, 0) + 1
                )
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
                    label_sentences=self._label_sentences(
                        obj,
                        item_id,
                        role_label,
                        sub_question,
                        agenda.language,
                    ),
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

    # Deterministic floors for when the frame call fails. They used to quote
    # the internal sub-question ("哪件藏品能够提供对照、限制或其他声音？") and
    # tell every object to "compare", which read as boilerplate. They now say
    # only what is true of any recorded object.
    @staticmethod
    def _why_selected(role_label: str, sub_question: str) -> str:
        del role_label, sub_question
        return "馆方记录写明了它的题名、年代和材料，都可以在“来源”里逐条核对。"

    @staticmethod
    def _relation(index: int, role_label: str) -> str:
        del role_label
        if index == 0:
            return "先看清它是什么、用什么做的，再往下走。"
        return "先读它自己的馆方记录，别急着和前一件归成一类。"

    @staticmethod
    def _label_sentences(
        obj: MuseumObject,
        item_id: str,
        role_label: str,
        sub_question: str,
        language: str = "zh",
    ) -> list[LabelSentence]:
        del sub_question
        return curation.label_sentences(obj, item_id, role_label, 140, language)

    @staticmethod
    def _system_prompt() -> str:
        return (
            "你是证据受限的数字遗产微策展助手。只能使用输入 objects 及其 evidence，"
            "不得补写馆方材料之外的人名、年代、因果或价值判断。馆方英文原文不得翻译后冒充原文。"
            "输出单个 JSON 对象，包含 title, curatorialThesis, coreAnswer, subQuestions, coverageLimits, items。"
            "items 必须严格使用给定的五个 objectId 与 role；每个 labelSentence 必须包含 text, type, evidenceIds，"
            "且 evidenceIds 只能来自同一对象。不要生成或改写 institution_fact；馆方事实句由系统保留原文。"
            "你只能补充明确标为 system_inference 或 uncertain 的关系文本。"
            "输入 evidenceBoundaries 若非空，标题、论点与展签都不得越过这些边界。"
        )

    @staticmethod
    def _model_payload(agenda: AgendaInput, context: GenerationContext) -> dict[str, Any]:
        return {
            "agenda": agenda.model_dump(mode="json", by_alias=True),
            "requiredRoles": [role.value for role in ROLE_ORDER],
            "objects": [obj.model_dump(mode="json", by_alias=True) for obj in context.selected],
            "evidenceBoundaries": list(context.question_card_limits),
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
