"""Run an auditable DeepSeek-assisted review over a scoped frozen qrel pool.

This is deliberately *not* an evaluator and never edits retrieval_eval_v1 or
the human-review revision log. It reads only the local review API's pending
candidate DTOs, asks DeepSeek for evidence-bound draft suggestions, and (only
with ``--apply``) appends them to the separate AI-suggestion SQLite store.
Default mode is dry-run.

Examples::

    python scripts/ai_qrel_review.py --category visual_motif --dry-run
    python scripts/ai_qrel_review.py --query-id 001-visual-motif --apply --resume

Use a small scoped dry-run and inspect its JSONL report before an apply run.
The script refuses an unscoped run, skips all existing judgements by default,
and cannot write directly to the frozen questions/qrels files.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, TextIO
from uuid import UUID, uuid5

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "api"))

from app.config import Settings  # noqa: E402
from app.images import ImageCache, ImageFetchError  # noqa: E402
from app.providers.deepseek import VisionImage  # noqa: E402
from app.providers.deepseek import DeepSeekProvider, ProviderError  # noqa: E402
from app.qrel_suggestions import AiSuggestionStore  # noqa: E402


SCRIPT_VERSION = "ai-qrel-review-v2"
NOTE_PREFIX = "AI代审，待复核："
REQUEST_NAMESPACE = UUID("625ef75b-d77d-5273-843e-2d8ce88a0424")
EVIDENCE_VERDICTS = {
    "supports",
    "insufficient",
    "contradicts",
    "uncertain",
    "not_applicable",
}
ANSWERABILITY = {"supported", "partially_supported", "unsupported"}


class QrelReviewApi(Protocol):
    async def list_questions(self) -> dict[str, Any]: ...
    async def question(self, query_id: str) -> dict[str, Any]: ...


class LocalQrelReviewApi:
    """Small API client. The review service itself owns frozen-data validation."""

    def __init__(self, base_url: str, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def _request(
        self, method: str, path: str, *, json_body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        # This client is intentionally local-only.  Ignore machine-level proxy
        # settings so Windows proxy/PAC configuration cannot turn a loopback
        # request into an upstream 502.
        async with httpx.AsyncClient(timeout=self.timeout_seconds, trust_env=False) as client:
            response = await client.request(method, f"{self.base_url}{path}", json=json_body)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"QREL review API returned a non-object for {path}")
        return data

    async def list_questions(self) -> dict[str, Any]:
        return await self._request("GET", "/api/admin/retrieval-eval/reviews/questions")

    async def question(self, query_id: str) -> dict[str, Any]:
        return await self._request(
            "GET", f"/api/admin/retrieval-eval/reviews/questions/{query_id}"
        )



class JsonProvider(Protocol):
    configured: bool

    async def generate_retrieval_audit_json(
        self, system_prompt: str, user_payload: dict[str, Any]
    ) -> dict[str, Any]: ...

    async def generate_qrel_vision_json(
        self, system_prompt: str, user_payload: dict[str, Any], images: Sequence[VisionImage]
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ReviewScope:
    categories: tuple[str, ...] = ()
    query_ids: tuple[str, ...] = ()
    query_start: str | None = None
    query_end: str | None = None
    limit: int | None = None

    def includes(self, summary: Mapping[str, Any]) -> bool:
        query_id = str(summary.get("queryId", ""))
        if self.categories and str(summary.get("category", "")) not in self.categories:
            return False
        if self.query_ids and query_id not in self.query_ids:
            return False
        if self.query_start and query_id < self.query_start:
            return False
        if self.query_end and query_id > self.query_end:
            return False
        return True


@dataclass
class ReviewRun:
    dry_run: bool = True
    resume: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)
    event_sink: Callable[[dict[str, Any]], None] | None = field(
        default=None, repr=False
    )

    def record(self, **event: Any) -> None:
        self.events.append(event)
        if self.event_sink is not None:
            self.event_sink(event)


ProgressSink = Callable[[int, int, str, str, str | None], None]


class IncrementalJsonlReport:
    """Append and fsync-friendly flush every event so interrupted runs survive."""

    def __init__(self, path: Path, *, dry_run: bool, resume: bool) -> None:
        self.path = path
        self.dry_run = dry_run
        self.resume = resume
        self._handle: TextIO | None = None

    def __enter__(self) -> "IncrementalJsonlReport":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Append is deliberate: a resumed run must retain the prior audit tail.
        self._handle = self.path.open("a", encoding="utf-8", newline="\n")
        return self

    def __exit__(self, *_args: Any) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def record(self, event: Mapping[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("incremental report is not open")
        payload = {
            "schemaVersion": 1,
            "scriptVersion": SCRIPT_VERSION,
            "dryRun": self.dry_run,
            "resume": self.resume,
            **event,
        }
        self._handle.write(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        )
        # Flushing every event is slightly stronger than the required per-question
        # durability and keeps already-persisted candidate suggestions auditable.
        self._handle.flush()


def _note(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("model note must be a non-empty Chinese short note")
    text = value.strip()
    if len(text) > 300:
        raise ValueError("model note exceeds 300 characters")
    return text if text.startswith(NOTE_PREFIX) else f"{NOTE_PREFIX}{text}"


def _object_prompt(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Build a label-free candidate view; do not pass pooled qrel metadata."""

    obj = candidate.get("object")
    evidence = candidate.get("evidence")
    if not isinstance(obj, dict) or not isinstance(evidence, list):
        raise ValueError("review API candidate is missing object or evidence")
    # Object DTO includes many presentational fields; these are the bounded
    # catalogue facts that help relevance review. All evidence remains intact.
    metadata = {
        key: obj.get(key)
        for key in (
            "id", "title", "titleOriginal", "date", "maker", "medium", "type",
            "culture", "creator", "material", "place", "cultureDisplay", "description",
            "institution", "institutionId", "department", "classification", "themes", "tags",
        )
        if obj.get(key) not in (None, "", [], {})
    }
    return {"objectId": candidate.get("objectId"), "metadata": metadata, "evidence": evidence}


CANDIDATE_PROMPT_VERSION = "candidate-text-v2"
FINALIZATION_PROMPT_VERSION = "finalization-text-v2"
VISION_PROMPT_VERSION = "vision-v2"


CANDIDATE_SYSTEM_PROMPT = """你是馆藏检索评测的 AI 代审助手。仅根据给定问题、每件候选的馆藏元数据和该对象的机构证据判断其是否有助于回答问题。不要猜测、不要使用常识补全，也不要把主题相近当作证据充分。

严格只输出 JSON 对象：
{"decisions":[{"objectId":"...","relevance":0|1|2|3,"evidenceVerdict":"supports|insufficient|contradicts|uncertain|not_applicable","supportingEvidenceIds":["仅该对象提供的 evidence id"],"note":"简短中文理由"}]}

每一个输入 objectId 必须恰好有一个决定。relevance=0 无关；1 边缘或只主题相近；2 有用但有限；3 直接且充分。只有 relevance 为 2 或 3 时 evidenceVerdict 才可以是 supports；supports 必须列出至少一个证据 ID。不得输出问题之外的字段。"""

FINALIZATION_SYSTEM_PROMPT = """你是馆藏检索评测的 AI 代审助手。根据问题、问题类别、必需文化区域及已完成的逐对象审阅，判断当前候选集合对于回答问题的总体可答性。不要把未被证据支持的推断写成支持。若没有 relevance>=2 且 evidenceVerdict=supports 的候选，不能判 supported；跨文化问题的任一 requiredCulturalLegs 若没有这样的候选覆盖，也不能判 supported。

严格只输出 JSON 对象：
{"expectedAnswerability":"supported|partially_supported|unsupported","note":"简短中文理由"}
不得输出其他字段。"""

VISION_SYSTEM_PROMPT = """你是馆藏检索评测的视觉复核助手。只对每个 objectId 所附的馆藏图像做视觉层面复核，并结合给定的文本初审与机构证据。图像无法证明的历史、作者、年代或因果关系不得据此提升判断。严格只输出 JSON 对象：
{"decisions":[{"objectId":"...","relevance":0|1|2|3,"evidenceVerdict":"supports|insufficient|contradicts|uncertain|not_applicable","supportingEvidenceIds":["该对象证据"],"note":"简短中文理由"}]}
必须每件图像恰好一个决定，不能输出其他字段。"""

VISUAL_CATEGORIES = {"visual_motif", "open_theme", "ambiguous_out_of_scope"}


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _suggestion_run_id(
    benchmark_id: str,
    text_model: str,
    vision_model: str,
    scope: ReviewScope,
) -> str:
    """Bind a run to every input that can change an AI suggestion."""

    identity = {
        "scriptVersion": SCRIPT_VERSION,
        "benchmarkId": benchmark_id,
        "textModel": text_model,
        "visionModel": vision_model,
        "promptVersions": {
            "candidate": CANDIDATE_PROMPT_VERSION,
            "finalization": FINALIZATION_PROMPT_VERSION,
            "vision": VISION_PROMPT_VERSION,
        },
        "promptHashes": {
            "candidate": _prompt_sha256(CANDIDATE_SYSTEM_PROMPT),
            "finalization": _prompt_sha256(FINALIZATION_SYSTEM_PROMPT),
            "vision": _prompt_sha256(VISION_SYSTEM_PROMPT),
        },
        "scope": {
            "categories": scope.categories,
            "queryIds": scope.query_ids,
            "queryStart": scope.query_start,
            "queryEnd": scope.query_end,
            "limit": scope.limit,
        },
    }
    identity_key = json.dumps(identity, ensure_ascii=False, sort_keys=True)
    return str(uuid5(REQUEST_NAMESPACE, identity_key))


def validate_candidate_decision(
    raw: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    if set(raw) != {"relevance", "evidenceVerdict", "supportingEvidenceIds", "note"}:
        raise ValueError("model candidate output must contain exactly the required fields")
    relevance = raw["relevance"]
    verdict = raw["evidenceVerdict"]
    evidence_ids = raw["supportingEvidenceIds"]
    if isinstance(relevance, bool) or not isinstance(relevance, int) or relevance not in {0, 1, 2, 3}:
        raise ValueError("model relevance must be an integer from 0 to 3")
    if not isinstance(verdict, str) or verdict not in EVIDENCE_VERDICTS:
        raise ValueError("model evidenceVerdict is invalid")
    if not isinstance(evidence_ids, list) or any(not isinstance(item, str) for item in evidence_ids):
        raise ValueError("model supportingEvidenceIds must be a string list")
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("model supportingEvidenceIds must not contain duplicates")
    allowed = {
        item.get("id") for item in candidate.get("evidence", []) if isinstance(item, dict)
    }
    if any(item not in allowed for item in evidence_ids):
        raise ValueError("model cited evidence outside the candidate object")
    if verdict == "supports" and (relevance < 2 or not evidence_ids):
        raise ValueError("supports requires relevance 2/3 and owned evidence")
    if relevance < 2 and verdict == "supports":
        raise ValueError("relevance 0/1 cannot claim supports")
    return {
        "relevance": relevance,
        "evidenceVerdict": verdict,
        "supportingEvidenceIds": evidence_ids,
        "note": _note(raw["note"]),
    }


def validate_finalization(raw: Mapping[str, Any]) -> dict[str, Any]:
    if set(raw) != {"expectedAnswerability", "note"}:
        raise ValueError("model finalization output must contain exactly the required fields")
    answerability = raw["expectedAnswerability"]
    if not isinstance(answerability, str) or answerability not in ANSWERABILITY:
        raise ValueError("model expectedAnswerability is invalid")
    return {"expectedAnswerability": answerability, "note": _note(raw["note"])}


def _safe_error_code(error: BaseException) -> str:
    """Return an auditable code without leaking provider bodies or credentials."""

    if isinstance(error, ProviderError):
        return str(error.code or "provider_error")
    return type(error).__name__


def _visual_failure_decision(
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    """A missing visual check can never retain a text-only grade of three."""

    bounded = dict(decision)
    if bounded.get("relevance") == 3:
        bounded["relevance"] = 2
        note = str(bounded.get("note", NOTE_PREFIX)).rstrip()
        suffix = "视觉复核失败，暂降为最高 2 级。"
        if suffix not in note:
            bounded["note"] = f"{note} {suffix}"[:300]
    return bounded


def _completed_candidate(
    candidate: Mapping[str, Any], judgment: Mapping[str, Any]
) -> dict[str, Any]:
    """Keep finalization grounded in the object and its cultural coverage."""

    obj = candidate.get("object")
    metadata = {
        key: obj.get(key)
        for key in (
            "title",
            "titleOriginal",
            "date",
            "maker",
            "culture",
            "cultureDisplay",
            "place",
            "type",
            "classification",
        )
        if isinstance(obj, dict) and obj.get(key) not in (None, "", [], {})
    }
    legs = candidate.get("culturalLegs")
    return {
        "objectId": candidate.get("objectId"),
        "culturalLegs": [leg for leg in legs if isinstance(leg, str)]
        if isinstance(legs, list)
        else [],
        "metadata": metadata,
        "judgment": dict(judgment),
    }


def _enforce_finalization_consistency(
    finalization: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    category: Any,
    required_cultural_legs: Sequence[str],
) -> tuple[dict[str, Any], list[str]]:
    """Deterministically gate unsupported or culturally incomplete `supported`."""

    decision = dict(finalization)
    if decision.get("expectedAnswerability") != "supported":
        return decision, []
    supported = [
        item
        for item in candidates
        if isinstance(item.get("judgment"), Mapping)
        and item["judgment"].get("relevance") in {2, 3}
        and item["judgment"].get("evidenceVerdict") == "supports"
    ]
    risks: list[str] = []
    if not supported:
        decision["expectedAnswerability"] = "unsupported"
        risks.append("consistency_downgraded:no_supported_candidate")
    elif category == "cross_cultural":
        covered = {
            leg
            for item in supported
            for leg in item.get("culturalLegs", [])
            if isinstance(leg, str)
        }
        missing = [leg for leg in required_cultural_legs if leg not in covered]
        if missing:
            decision["expectedAnswerability"] = "partially_supported"
            risks.append(
                "consistency_downgraded:missing_cultural_legs:" + ",".join(missing)
            )
    if risks:
        note = str(decision.get("note", NOTE_PREFIX)).rstrip()
        suffix = "一致性门禁已下调总体可答性。"
        if suffix not in note:
            decision["note"] = f"{note} {suffix}"[:300]
    return decision, risks


def validate_candidate_batch(
    raw: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    if set(raw) != {"decisions"} or not isinstance(raw["decisions"], list):
        raise ValueError("batch output must be exactly a decisions list")
    expected = {str(candidate["objectId"]) for candidate in candidates}
    decisions: dict[str, dict[str, Any]] = {}
    for item in raw["decisions"]:
        if not isinstance(item, dict) or not isinstance(item.get("objectId"), str):
            raise ValueError("batch decision lacks objectId")
        object_id = item["objectId"]
        if object_id not in expected or object_id in decisions:
            raise ValueError("batch decision objectId is not in its candidate set")
        candidate = next(candidate for candidate in candidates if candidate["objectId"] == object_id)
        decisions[object_id] = validate_candidate_decision(
            {key: value for key, value in item.items() if key != "objectId"}, candidate
        )
    if set(decisions) != expected:
        raise ValueError("batch output must decide every candidate exactly once")
    return decisions


def _repair_visual_support_without_evidence(
    raw: Mapping[str, Any],
    text_decisions: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], dict[str, list[str]]]:
    """Repair one bounded vision-only contract error without inventing evidence.

    Vision models occasionally identify a visible motif as ``supports`` while
    returning no institution evidence id. Reuse owned evidence only when the
    already-validated text decision supplied it; otherwise retain the visual
    relevance signal at no more than grade 2 and downgrade the verdict to
    ``insufficient``. All other malformed output still fails closed in the
    normal validator.
    """

    decisions = raw.get("decisions")
    if set(raw) != {"decisions"} or not isinstance(decisions, list):
        return raw, {}
    repaired: list[Any] = []
    risks: dict[str, list[str]] = {}
    for value in decisions:
        if not isinstance(value, dict):
            repaired.append(value)
            continue
        item = dict(value)
        object_id = item.get("objectId")
        evidence_ids = item.get("supportingEvidenceIds")
        relevance = item.get("relevance")
        if (
            isinstance(object_id, str)
            and item.get("evidenceVerdict") == "supports"
            and isinstance(evidence_ids, list)
            and not evidence_ids
        ):
            text_decision = text_decisions.get(object_id, {})
            text_evidence = text_decision.get("supportingEvidenceIds")
            if (
                text_decision.get("evidenceVerdict") == "supports"
                and isinstance(text_evidence, list)
                and text_evidence
            ):
                item["supportingEvidenceIds"] = list(text_evidence)
                risks[object_id] = ["vision_support_reused_text_evidence"]
            else:
                item["evidenceVerdict"] = "insufficient"
                if isinstance(relevance, int) and not isinstance(relevance, bool):
                    item["relevance"] = min(relevance, 2)
                note = str(item.get("note", "")).strip()
                suffix = "图像匹配但没有可归属的机构证据，暂按证据不足处理。"
                item["note"] = f"{note} {suffix}".strip()[:300]
                risks[object_id] = ["vision_support_without_owned_evidence"]
        repaired.append(item)
    return {"decisions": repaired}, risks


def _confidence_and_risks(decision: Mapping[str, Any], category: Any) -> tuple[str, list[str]]:
    relevance = decision.get("relevance")
    verdict = decision.get("evidenceVerdict")
    confidence = "high" if relevance == 3 and verdict == "supports" else "medium" if relevance == 2 else "low"
    risks = ["ai_draft", "needs_human_confirmation"]
    if confidence == "low":
        risks.append("low_confidence")
    if category == "visual_motif" or (
        category in {"open_theme", "ambiguous_out_of_scope"} and relevance is not None and relevance >= 1
    ):
        # Deterministic policy marker; never rely on a model to volunteer it.
        risks.append("visual_review_required")
    return confidence, risks


def _suggestion_record(
    *, kind: str, run_id: str, benchmark_id: str, query_id: str, object_id: str | None,
    model: str, prompt_version: str, prompt: str, model_input: Mapping[str, Any],
    decision: Mapping[str, Any], category: Any, modalities: str = "metadata,evidence",
    extra_risk_flags: Sequence[str] = (), validation_status: str = "validated_text_only",
) -> dict[str, Any]:
    confidence, risk_flags = _confidence_and_risks(decision, category)
    risk_flags = list(dict.fromkeys([*risk_flags, *extra_risk_flags]))
    return {
        "suggestionRunId": run_id, "kind": kind, "queryId": query_id, "objectId": object_id,
        "provider": "deepseek", "model": model, "promptVersion": prompt_version,
        "promptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "inputModality": modalities, "inputSha256": hashlib.sha256(json.dumps(model_input, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
        "confidenceBand": confidence, "riskFlags": risk_flags, "validationStatus": validation_status,
        "suggestion": dict(decision),
    }


async def _prepare_vision_images(
    image_cache: ImageCache,
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[Mapping[str, Any], VisionImage, str]], dict[str, str]]:
    """Fetch only frozen-object images through the existing bounded cache."""

    prepared: list[tuple[Mapping[str, Any], VisionImage, str]] = []
    failures: dict[str, str] = {}
    for candidate in candidates:
        object_id = str(candidate["objectId"])
        obj = candidate.get("object")
        image_url = obj.get("imageUrl") if isinstance(obj, dict) else None
        evidence = candidate.get("evidence")
        evidence_id = next((str(item.get("id")) for item in evidence if isinstance(item, dict) and item.get("id")), f"{object_id}:image") if isinstance(evidence, list) else f"{object_id}:image"
        if not isinstance(image_url, str) or not image_url:
            failures[object_id] = "image_unavailable"
            continue
        try:
            payload, _ = await asyncio.to_thread(image_cache.get, image_url, 512)
        except ImageFetchError as error:
            failures[object_id] = f"image_fetch_failed:{error.code}"
            continue
        except (OSError, httpx.HTTPError) as error:
            failures[object_id] = f"image_fetch_failed:{_safe_error_code(error)}"
            continue
        prepared.append((candidate, VisionImage(object_id=object_id, evidence_id=evidence_id, payload=payload), hashlib.sha256(payload).hexdigest()))
    return prepared, failures


async def run_review(
    api: QrelReviewApi,
    provider: JsonProvider,
    scope: ReviewScope,
    *,
    suggestion_store: AiSuggestionStore | None = None,
    image_cache: ImageCache | None = None,
    dry_run: bool = True,
    resume: bool = False,
    event_sink: Callable[[dict[str, Any]], None] | None = None,
    progress_sink: ProgressSink | None = None,
) -> ReviewRun:
    if not provider.configured:
        raise RuntimeError("DEEPSEEK_API_KEY is required for AI qrel review")
    listed = await api.list_questions()
    benchmark_id = listed.get("benchmarkId")
    questions = listed.get("questions")
    if not isinstance(benchmark_id, str) or not isinstance(questions, list):
        raise RuntimeError("local qrel API returned an invalid questions response")
    model = str(getattr(provider, "model", "deepseek-configured-model"))
    vision_model = str(
        getattr(provider, "labels_model", "deepseek-configured-vision-model")
    )
    run_id = _suggestion_run_id(benchmark_id, model, vision_model, scope)
    selected = [item for item in questions if isinstance(item, dict) and scope.includes(item)]
    if scope.limit is not None:
        selected = selected[: scope.limit]
    run = ReviewRun(dry_run=dry_run, resume=resume, event_sink=event_sink)
    total = len(selected)
    for index, summary in enumerate(selected, start=1):
        query_id = str(summary["queryId"])
        if progress_sink is not None:
            progress_sink(index, total, query_id, "started", None)
        if resume and summary.get("status") == "complete":
            run.record(kind="question", queryId=query_id, status="skipped_complete")
            if progress_sink is not None:
                progress_sink(index, total, query_id, "skipped", "human complete")
            continue
        try:
            detail = await api.question(query_id)
        except (httpx.HTTPError, asyncio.TimeoutError) as error:
            code = _safe_error_code(error)
            run.record(
                kind="question",
                queryId=query_id,
                status="error",
                stage="question_fetch",
                errorCode=code,
                riskFlags=[f"error:question_fetch:{code}"],
            )
            if progress_sink is not None:
                progress_sink(index, total, query_id, "error", f"question_fetch:{code}")
            continue
        candidates = detail.get("candidates")
        if not isinstance(candidates, list):
            raise RuntimeError(f"local qrel API returned invalid candidates for {query_id}")
        completed: list[dict[str, Any]] = []
        generated: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]] = {}
        all_prior, latest_finalization = (
            suggestion_store.latest_for_question(query_id)
            if suggestion_store and resume
            else ({}, None)
        )
        # A prompt/model/scope change creates a new run id. Never silently mix
        # suggestions produced by an older review contract into this run.
        prior_suggestions = {
            object_id: suggestion
            for object_id, suggestion in all_prior.items()
            if suggestion.get("suggestionRunId") == run_id
        }
        prior_finalization = (
            latest_finalization
            if latest_finalization
            and latest_finalization.get("suggestionRunId") == run_id
            else None
        )
        missing: list[dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise RuntimeError(f"invalid candidate DTO for {query_id}")
            existing = candidate.get("judgment")
            if existing is not None:
                # Human judgment is always authoritative and is never written by
                # this script or replaced by a suggestion.
                completed.append(_completed_candidate(candidate, existing))
                run.record(
                    kind="candidate",
                    queryId=query_id,
                    objectId=candidate.get("objectId"),
                    status="skipped_existing_human_judgment",
                )
                continue
            if str(candidate.get("objectId")) in prior_suggestions:
                prior = prior_suggestions[str(candidate.get("objectId"))]
                run.record(
                    kind="candidate",
                    queryId=query_id,
                    objectId=candidate.get("objectId"),
                    status="skipped_existing_ai_suggestion",
                    suggestionRunId=prior.get("suggestionRunId"),
                    suggestionId=prior.get("suggestionId"),
                    decision={
                        key: prior.get(key)
                        for key in (
                            "relevance",
                            "evidenceVerdict",
                            "supportingEvidenceIds",
                            "note",
                        )
                    },
                    riskFlags=prior.get("riskFlags", []),
                    provider=prior.get("provider"),
                    model=prior.get("model"),
                    promptVersion=prior.get("promptVersion"),
                    promptSha256=prior.get("promptHash"),
                    inputSha256=prior.get("inputHash"),
                    validationStatus=prior.get("validationStatus"),
                    modalities=prior.get("modalities", []),
                )
                completed.append(_completed_candidate(candidate, prior))
                continue
            missing.append(candidate)

        # One deterministic text request per question (normally all twelve
        # pending candidates), not one costly serial request per object.
        if missing:
            text_input = {
                "question": summary.get("question"),
                "candidates": [_object_prompt(candidate) for candidate in missing],
            }
            try:
                raw_text = await provider.generate_retrieval_audit_json(
                    CANDIDATE_SYSTEM_PROMPT, text_input
                )
            except (ProviderError, httpx.HTTPError, asyncio.TimeoutError) as error:
                code = _safe_error_code(error)
                run.record(
                    kind="question",
                    queryId=query_id,
                    status="error",
                    stage="candidate_text",
                    errorCode=code,
                    riskFlags=[f"error:candidate_text:{code}"],
                )
                if progress_sink is not None:
                    progress_sink(
                        index, total, query_id, "error", f"candidate_text:{code}"
                    )
                continue
            text_decisions = validate_candidate_batch(raw_text, missing)
            for candidate in missing:
                object_id = str(candidate["objectId"])
                decision = text_decisions[object_id]
                candidate_input = {
                    "question": summary.get("question"),
                    "candidate": _object_prompt(candidate),
                    "batchInputSha256": hashlib.sha256(json.dumps(text_input, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
                }
                generated[object_id] = (candidate, decision, candidate_input, [])
                completed.append(_completed_candidate(candidate, decision))

        category = summary.get("category")
        visual_targets = [
            item[0] for item in generated.values()
            if category == "visual_motif" or (
                category in {"open_theme", "ambiguous_out_of_scope"}
                and (item[1]["relevance"] >= 1 or "visual_confirmation_recommended" in _confidence_and_risks(item[1], category)[1])
            )
        ]
        if visual_targets and image_cache is None:
            for candidate in visual_targets:
                object_id = str(candidate["objectId"])
                item, decision, model_input, risks = generated[object_id]
                generated[object_id] = (
                    item,
                    _visual_failure_decision(decision),
                    model_input,
                    [*risks, "error:vision:cache_unavailable"],
                )
        elif visual_targets:
            prepared, image_failures = await _prepare_vision_images(image_cache, visual_targets)
            for object_id, failure in image_failures.items():
                item, decision, model_input, risks = generated[object_id]
                generated[object_id] = (
                    item,
                    _visual_failure_decision(decision),
                    model_input,
                    [*risks, f"error:image:{failure}"],
                )
            for start in range(0, len(prepared), 4):
                batch = prepared[start : start + 4]
                batch_candidates = [item[0] for item in batch]
                vision_input = {
                    "question": summary.get("question"),
                    "candidates": [
                        {"objectId": candidate["objectId"], "textDecision": generated[str(candidate["objectId"])][1], "candidate": _object_prompt(candidate), "imageSha256": image_hash}
                        for candidate, _image, image_hash in batch
                    ],
                }
                try:
                    raw_vision = await provider.generate_qrel_vision_json(
                        VISION_SYSTEM_PROMPT, vision_input, [item[1] for item in batch]
                    )
                except (ProviderError, httpx.HTTPError, asyncio.TimeoutError) as error:
                    code = _safe_error_code(error)
                    for candidate, _image, _hash in batch:
                        object_id = str(candidate["objectId"])
                        item, decision, model_input, risks = generated[object_id]
                        generated[object_id] = (
                            item,
                            _visual_failure_decision(decision),
                            model_input,
                            [*risks, f"error:vision:{code}"],
                        )
                    continue
                # Invalid model output is a contract failure, not a recoverable
                # transport failure. Let validation abort so the prompt/schema
                # can be repaired before more suggestions are stored.
                text_decisions = {
                    str(candidate["objectId"]): generated[str(candidate["objectId"])][1]
                    for candidate in batch_candidates
                }
                repaired_vision, repair_risks = _repair_visual_support_without_evidence(
                    raw_vision, text_decisions
                )
                visual_decisions = validate_candidate_batch(
                    repaired_vision, batch_candidates
                )
                for candidate, _image, image_hash in batch:
                    object_id = str(candidate["objectId"])
                    _candidate, _text_decision, text_input, risks = generated[object_id]
                    generated[object_id] = (
                        candidate,
                        visual_decisions[object_id],
                        {"textInput": text_input, "visionInput": vision_input, "imageSha256": image_hash},
                        [*risks, *repair_risks.get(object_id, []), "vision_reviewed"],
                    )

        for object_id, (candidate, decision, model_input, extra_risks) in generated.items():
            visual_done = "vision_reviewed" in extra_risks
            record = _suggestion_record(
                kind="candidate", run_id=run_id, benchmark_id=benchmark_id, query_id=query_id, object_id=object_id,
                model=vision_model if visual_done else model,
                prompt_version=(
                    f"{CANDIDATE_PROMPT_VERSION}+{VISION_PROMPT_VERSION}"
                    if visual_done
                    else CANDIDATE_PROMPT_VERSION
                ),
                prompt=(CANDIDATE_SYSTEM_PROMPT + VISION_SYSTEM_PROMPT) if visual_done else CANDIDATE_SYSTEM_PROMPT,
                model_input=model_input, decision=decision, category=category,
                modalities="metadata,evidence,image" if visual_done else "metadata,evidence",
                extra_risk_flags=extra_risks,
                validation_status=(
                    "validated_vision"
                    if visual_done
                    else "validated_text_only_with_visual_error"
                    if any(flag.startswith("error:") for flag in extra_risks)
                    else "validated_text_only"
                ),
            )
            persisted: dict[str, Any] | None = None
            if not dry_run:
                if suggestion_store is None:
                    raise RuntimeError("AI suggestion persistence requires a local suggestion store")
                persisted = suggestion_store.append(record)
            audit_event: dict[str, Any] = {
                "kind": "candidate",
                "queryId": query_id,
                "objectId": object_id,
                "status": "dry_run" if dry_run else "suggested",
                "suggestionRunId": run_id,
                "suggestionId": persisted.get("suggestionId") if persisted else None,
                "decision": decision,
                "provider": record["provider"],
                "model": record["model"],
                "promptVersion": record["promptVersion"],
                "promptSha256": record["promptSha256"],
                "inputSha256": record["inputSha256"],
                "validationStatus": record["validationStatus"],
                "riskFlags": record["riskFlags"],
                "modalities": record["inputModality"].split(","),
            }
            image_sha256 = model_input.get("imageSha256")
            if isinstance(image_sha256, (str, dict)):
                audit_event["imageSha256"] = image_sha256
            run.record(**audit_event)

        completed = [
            _completed_candidate(
                generated[str(item["objectId"])][0],
                generated[str(item["objectId"])][1],
            )
            if str(item["objectId"]) in generated
            else item
            for item in completed
        ]

        if len(completed) != len(candidates):
            run.record(kind="finalization", queryId=query_id, status="skipped_incomplete")
            if progress_sink is not None:
                progress_sink(index, total, query_id, "partial", "candidate set incomplete")
            continue
        human_finalization = detail.get("finalization")
        if human_finalization is not None:
            run.record(
                kind="finalization",
                queryId=query_id,
                status="skipped_existing_human_finalization",
            )
            if progress_sink is not None:
                progress_sink(index, total, query_id, "skipped", "human finalized")
            continue
        if prior_finalization is not None:
            run.record(
                kind="finalization",
                queryId=query_id,
                status="skipped_existing_ai_suggestion",
                suggestionRunId=prior_finalization.get("suggestionRunId"),
                suggestionId=prior_finalization.get("suggestionId"),
                decision={
                    key: prior_finalization.get(key)
                    for key in ("expectedAnswerability", "note")
                },
                riskFlags=prior_finalization.get("riskFlags", []),
                provider=prior_finalization.get("provider"),
                model=prior_finalization.get("model"),
                promptVersion=prior_finalization.get("promptVersion"),
                promptSha256=prior_finalization.get("promptHash"),
                inputSha256=prior_finalization.get("inputHash"),
                validationStatus=prior_finalization.get("validationStatus"),
                modalities=prior_finalization.get("modalities", []),
            )
            if progress_sink is not None:
                progress_sink(index, total, query_id, "resumed", "already stored")
            continue
        required_cultural_legs = [
            leg
            for leg in summary.get("requiredCulturalLegs", [])
            if isinstance(leg, str)
        ]
        final_input = {
            "question": summary.get("question"),
            "category": category,
            "requiredCulturalLegs": required_cultural_legs,
            "candidateJudgments": completed,
        }
        try:
            raw_finalization = await provider.generate_retrieval_audit_json(
                FINALIZATION_SYSTEM_PROMPT, final_input
            )
        except (ProviderError, httpx.HTTPError, asyncio.TimeoutError) as error:
            code = _safe_error_code(error)
            run.record(
                kind="finalization",
                queryId=query_id,
                status="error",
                stage="finalization_text",
                errorCode=code,
                riskFlags=[f"error:finalization_text:{code}"],
            )
            if progress_sink is not None:
                progress_sink(
                    index, total, query_id, "partial", f"finalization_text:{code}"
                )
            continue
        finalization = validate_finalization(raw_finalization)
        finalization, consistency_risks = _enforce_finalization_consistency(
            finalization,
            completed,
            category=category,
            required_cultural_legs=required_cultural_legs,
        )
        final_record = _suggestion_record(kind="finalization", run_id=run_id, benchmark_id=benchmark_id, query_id=query_id, object_id=None, model=model, prompt_version=FINALIZATION_PROMPT_VERSION, prompt=FINALIZATION_SYSTEM_PROMPT, model_input=final_input, decision=finalization, category=category, extra_risk_flags=consistency_risks, validation_status="validated_consistency_downgraded" if consistency_risks else "validated_text_only")
        persisted_finalization: dict[str, Any] | None = None
        if not dry_run:
            if suggestion_store is None:
                raise RuntimeError("AI suggestion persistence requires a local suggestion store")
            persisted_finalization = suggestion_store.append(final_record)
        run.record(
            kind="finalization",
            queryId=query_id,
            status="dry_run" if dry_run else "suggested",
            suggestionRunId=run_id,
            suggestionId=(
                persisted_finalization.get("suggestionId")
                if persisted_finalization
                else None
            ),
            decision=finalization,
            provider=final_record["provider"],
            model=final_record["model"],
            promptVersion=final_record["promptVersion"],
            promptSha256=final_record["promptSha256"],
            inputSha256=final_record["inputSha256"],
            validationStatus=final_record["validationStatus"],
            riskFlags=final_record["riskFlags"],
            modalities=final_record["inputModality"].split(","),
        )
        if progress_sink is not None:
            progress_sink(
                index,
                total,
                query_id,
                "completed",
                f"{len(generated)} new, {len(prior_suggestions)} resumed",
            )
    return run


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", action="append", default=[], help="Frozen pending-question category; repeatable")
    parser.add_argument("--query-id", action="append", default=[], help="Frozen queryId; repeatable")
    parser.add_argument("--query-start", help="Inclusive lexical queryId lower bound")
    parser.add_argument("--query-end", help="Inclusive lexical queryId upper bound")
    parser.add_argument("--limit", type=int, help="Maximum selected questions")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Persist tentative AI suggestions only; human judgements are never written")
    mode.add_argument("--dry-run", action="store_true", help="Generate and report suggestions without persisting them (default)")
    parser.add_argument("--resume", action="store_true", help="Resume a prior scoped run by skipping existing candidate/finalization revisions")
    parser.add_argument("--report", type=Path, help="Write JSONL decision audit records")
    args = parser.parse_args(argv)
    if not (args.category or args.query_id or args.query_start or args.query_end):
        parser.error("scope is required: pass --category, --query-id, or --query-start/--query-end")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.query_start and args.query_end and args.query_start > args.query_end:
        parser.error("--query-start must not be after --query-end")
    return args


async def _main_async(
    args: argparse.Namespace,
    *,
    event_sink: Callable[[dict[str, Any]], None] | None = None,
    progress_sink: ProgressSink | None = None,
) -> ReviewRun:
    settings = Settings.from_env()
    provider = DeepSeekProvider(settings)
    return await run_review(
        LocalQrelReviewApi(args.api_base_url),
        provider,
        ReviewScope(tuple(args.category), tuple(args.query_id), args.query_start, args.query_end, args.limit),
        suggestion_store=AiSuggestionStore(settings.qrel_review_dataset_dir, settings.qrel_suggestion_db_path),
        image_cache=ImageCache(
            settings.store_path.parent / "cache" / "objects",
            limit_bytes=settings.image_cache_limit_mb * 1024 * 1024,
        ),
        dry_run=not args.apply,
        resume=args.resume,
        event_sink=event_sink,
        progress_sink=progress_sink,
    )


def _stdout_progress(
    index: int,
    total: int,
    query_id: str,
    status: str,
    detail: str | None,
) -> None:
    suffix = f" ({detail})" if detail else ""
    print(f"[{index}/{total}] {query_id}: {status}{suffix}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = (
        IncrementalJsonlReport(
            args.report, dry_run=not args.apply, resume=args.resume
        )
        if args.report
        else None
    )
    if report is not None:
        report.__enter__()
        report.record(
            {
                "kind": "run",
                "status": "started",
                "scope": {
                    "categories": args.category,
                    "queryIds": args.query_id,
                    "queryStart": args.query_start,
                    "queryEnd": args.query_end,
                    "limit": args.limit,
                },
            }
        )
    try:
        run = asyncio.run(
            _main_async(
                args,
                event_sink=report.record if report is not None else None,
                progress_sink=_stdout_progress,
            )
        )
    except (ProviderError, httpx.HTTPError, RuntimeError, ValueError) as error:
        if report is not None:
            report.record(
                {
                    "kind": "run",
                    "status": "failed",
                    "errorCode": _safe_error_code(error),
                }
            )
        print(f"AI qrel review failed: {error}", file=sys.stderr)
        return 2
    else:
        if report is not None:
            report.record(
                {
                    "kind": "run",
                    "status": "completed",
                    "eventCount": len(run.events),
                }
            )
        error_count = sum(
            1 for event in run.events if event.get("status") == "error"
        )
        print(
            f"AI qrel review complete: {len(run.events)} audit events, "
            f"{error_count} recoverable errors.",
            flush=True,
        )
        return 0
    finally:
        if report is not None:
            report.__exit__(None, None, None)


if __name__ == "__main__":
    raise SystemExit(main())
