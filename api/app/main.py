from __future__ import annotations

import re
from contextlib import asynccontextmanager
from hashlib import sha256
from typing import Annotated
from uuid import uuid4

from fastapi import Cookie, Depends, FastAPI, Header, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import asyncio
import json
import logging

from fastapi.responses import StreamingResponse

from .admin_reports import (
    DataAuditReport,
    DescriptiveStatisticsExport,
    build_data_audit,
    build_descriptive_export,
)
from .audio_guide import (
    AudioGuideKind,
    AudioGuideReferenceError,
    AudioGuideService,
)
from .collections import (
    BM25_RETRIEVAL_METHOD,
    BM25_RETRIEVAL_VERSION,
    HYBRID_RETRIEVAL_METHOD,
    HYBRID_RETRIEVAL_VERSION,
    CollectionDataError,
    CollectionRepository,
)
from .config import Settings
from .epilogue_chat import EpilogueChatReferenceError, EpilogueChatService
from .generator import ExhibitionGenerator
from .images import ImageCache, ImageFetchError
from . import interview_voice
from .interview import MIN_DOMAIN_OBJECTS, InterviewService, domain_choices_for
from .jobs import JobStore
from .models import (
    AgendaCheckResponse,
    AgendaInput,
    AnalyticsResponse,
    EventCreate,
    EventAccepted,
    EventLog,
    EventName,
    EpilogueChatRequest,
    EpilogueChatResponse,
    Exhibition,
    ExhibitionPoster,
    ExhibitionStatus,
    FocusPatchRequest,
    GenerateExhibitionRequest,
    GenerateFromProfileRequest,
    GenerationJob,
    InterviewAnswer,
    InterviewQuestionId,
    InterviewState,
    ItemsPatchRequest,
    LOCKED_STATUSES,
    PublicExhibition,
    ReviewRecord,
    ReviewRequest,
    ValidationResult,
    VisitorProfile,
    utc_now,
)
from .store import ExhibitionStore
from .validator import validate_exhibition
from .providers.aliyun_image import (
    AliyunImageProvider,
    AliyunImageProviderError,
    PosterContext,
)
from .providers.aliyun_tts import AliyunTtsProvider, AliyunTtsProviderError


def _event(
    store: ExhibitionStore,
    name: EventName,
    exhibition_id: str | None = None,
    session_id: str = "system-generated",
    **parameters: object,
) -> EventLog:
    return store.save_event(
        EventLog(
            id=str(uuid4()),
            session_id=session_id,
            event=name,
            exhibition_id=exhibition_id,
            parameters=parameters,
        )
    )


def _slug(exhibition: Exhibition) -> str:
    ascii_title = re.sub(r"[^a-z0-9]+", "-", exhibition.title.casefold()).strip("-")
    prefix = ascii_title[:48].strip("-") or "exhibition"
    return f"{prefix}-{exhibition.id[:8]}"


def _poster_seed(exhibition_id: str) -> int:
    return int.from_bytes(sha256(exhibition_id.encode("utf-8")).digest()[:4], "big") % (
        2**31
    )


logger = logging.getLogger("app")


_INTERRUPTED_GENERATION_NOTE = (
    "上一次生成任务在完成前被中断；该记录保留为草稿，不代表已完成展览。"
)


def _mark_interrupted_generation(
    store: ExhibitionStore, exhibition_id: str
) -> Exhibition | None:
    """Turn a persisted in-progress skeleton into an honest recoverable draft."""

    try:
        exhibition = store.get_exhibition(exhibition_id)
    except KeyError:
        return None
    if exhibition.status != ExhibitionStatus.GENERATING.value:
        return exhibition
    exhibition.status = ExhibitionStatus.DRAFT
    if _INTERRUPTED_GENERATION_NOTE not in exhibition.coverage_limits:
        exhibition.coverage_limits.append(_INTERRUPTED_GENERATION_NOTE)
    if exhibition.poster and exhibition.poster.status == "generating":
        exhibition.poster = ExhibitionPoster(
            status="failed",
            provider=exhibition.poster.provider,
            model=exhibition.poster.model,
            generated_by=exhibition.poster.generated_by,
            size=exhibition.poster.size,
            error_code="generation_interrupted",
        )
    exhibition.updated_at = utc_now()
    return store.save_exhibition(exhibition)


_EPILOGUE_CHAT_EVENTS = {
    EventName.EPILOGUE_CHAT_OPENED.value,
    EventName.EPILOGUE_CHAT_MESSAGE_SENT.value,
    EventName.EPILOGUE_CHAT_CLOSED.value,
}


def _safe_epilogue_event_parameters(
    event: EventName | str, parameters: dict[str, object]
) -> dict[str, object]:
    """Whitelist non-text analytics for the optional post-visit chat.

    The generic event endpoint predates chat.  Filtering here means even a
    modified client cannot persist message/history text under one of the chat
    event names.
    """

    event_value = event.value if isinstance(event, EventName) else str(event)
    if event_value not in _EPILOGUE_CHAT_EVENTS:
        return parameters

    cleaned: dict[str, object] = {}
    turn_count = parameters.get("turnCount")
    if isinstance(turn_count, int) and not isinstance(turn_count, bool):
        cleaned["turnCount"] = max(0, min(turn_count, 8))

    if event_value == EventName.EPILOGUE_CHAT_MESSAGE_SENT.value:
        selected = parameters.get("selectedPrompt")
        if isinstance(selected, bool):
            cleaned["selectedPrompt"] = selected
        elif isinstance(selected, str) and re.fullmatch(
            r"open-question-[1-9]", selected
        ):
            cleaned["selectedPrompt"] = selected

    if event_value in {
        EventName.EPILOGUE_CHAT_OPENED.value,
        EventName.EPILOGUE_CHAT_CLOSED.value,
    }:
        source = parameters.get("source")
        if isinstance(source, str) and re.fullmatch(r"[a-zA-Z0-9_-]{1,32}", source):
            cleaned["source"] = source

    if event_value == EventName.EPILOGUE_CHAT_CLOSED.value:
        reason = parameters.get("reason")
        if reason in {"collapse", "skip"}:
            cleaned["reason"] = reason

    return cleaned


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    # Uvicorn only configures its own loggers, so without this the app's
    # warnings — including a silently-degraded model pass — go nowhere.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    collections = CollectionRepository(
        settings.collections_dir,
        default_collection_id=settings.default_collection_id,
        rag_mode=settings.rag_mode,
        dense_index_dir=settings.rag_index_dir,
        embedding_model_cache_dir=settings.rag_model_cache_dir,
        embedding_model=settings.rag_embedding_model,
        dense_top_k=settings.rag_dense_top_k,
        dense_min_score=settings.rag_dense_min_score,
        evidence_min_score=settings.rag_evidence_min_score,
        rrf_k=settings.rag_rrf_k,
        hybrid_max_results=settings.rag_max_results,
    )
    store = ExhibitionStore(
        mode=settings.store_mode,
        path=settings.store_path if settings.store_mode == "json" else None,
    )
    image_cache = ImageCache(
        settings.store_path.parent / "cache" / "objects",
        limit_bytes=settings.image_cache_limit_mb * 1024 * 1024,
    )
    generator = ExhibitionGenerator(
        settings=settings,
        collections=collections,
        image_cache=image_cache,
    )
    epilogue_chat_service = EpilogueChatService(generator.provider)
    image_provider: AliyunImageProvider | None = None
    image_provider_error: AliyunImageProviderError | None = None
    if settings.aliyun_image_api_key and settings.aliyun_image_api_host:
        try:
            image_provider = AliyunImageProvider(
                api_key=settings.aliyun_image_api_key,
                api_host=settings.aliyun_image_api_host,
                output_dir=settings.aliyun_image_output_dir,
                model=settings.aliyun_image_model,
                size=settings.aliyun_image_size,
                timeout_seconds=settings.aliyun_image_timeout_seconds,
                font_path=settings.aliyun_image_font_path,
            )
        except AliyunImageProviderError as exc:
            image_provider_error = exc
    tts_provider: AliyunTtsProvider | None = None
    tts_provider_error: AliyunTtsProviderError | None = None
    if settings.aliyun_tts_api_key and settings.aliyun_tts_api_host:
        try:
            tts_provider = AliyunTtsProvider(
                api_key=settings.aliyun_tts_api_key,
                api_host=settings.aliyun_tts_api_host,
                model=settings.aliyun_tts_model,
                voice=settings.aliyun_tts_voice,
                instruction=settings.aliyun_tts_instruction,
                timeout_seconds=settings.aliyun_tts_timeout_seconds,
                max_audio_bytes=settings.aliyun_tts_max_audio_bytes,
            )
        except AliyunTtsProviderError as exc:
            tts_provider_error = exc
    audio_guide_service = (
        AudioGuideService(tts_provider, settings.aliyun_tts_output_dir)
        if tts_provider is not None
        else None
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Jobs are process-local. Only a server/test-client startup proves that
        # this process owns the store; module import must remain read-only.
        # At that point persisted GENERATING records are orphans from the
        # previous process and can be recovered honestly as drafts.
        recovered_ids: list[str] = []
        for persisted in store.list_exhibitions():
            if persisted.status == ExhibitionStatus.GENERATING.value:
                recovered = _mark_interrupted_generation(store, persisted.id)
                if recovered is not None:
                    recovered_ids.append(recovered.id)
        if recovered_ids:
            logger.warning(
                "recovered %d interrupted generation record(s) as draft",
                len(recovered_ids),
            )
        yield

    app = FastAPI(
        title="问展 / Inquiry Curator API",
        version="0.1.0",
        description="Evidence-traceable heritage micro-curation P0 API.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount(
        "/generated/posters",
        StaticFiles(directory=str(settings.aliyun_image_output_dir), check_dir=False),
        name="generated-posters",
    )
    interviews = InterviewService(collections)
    interview_sessions: dict[str, InterviewState] = {}
    jobs = JobStore(max_job_seconds=settings.generation_job_timeout_seconds)
    poster_background_tasks: set[asyncio.Task[bool]] = set()
    audio_prewarm_tasks: set[asyncio.Task[None]] = set()
    app.state.settings = settings
    app.state.collections = collections
    app.state.store = store
    app.state.generator = generator
    app.state.epilogue_chat_service = epilogue_chat_service
    app.state.image_provider = image_provider
    app.state.image_provider_error = image_provider_error
    app.state.tts_provider = tts_provider
    app.state.tts_provider_error = tts_provider_error
    app.state.audio_guide_service = audio_guide_service
    app.state.interviews = interviews
    app.state.interview_sessions = interview_sessions
    app.state.jobs = jobs
    app.state.poster_background_tasks = poster_background_tasks
    app.state.audio_prewarm_tasks = audio_prewarm_tasks
    app.state.image_cache = image_cache

    @app.exception_handler(CollectionDataError)
    async def collection_error_handler(_, exc: CollectionDataError) -> JSONResponse:
        not_found_codes = {"COLLECTION_NOT_FOUND", "ITEM_NOT_FOUND"}
        status_code = 404 if exc.code in not_found_codes else 422
        return JSONResponse(
            status_code=status_code,
            content={
                "detail": exc.message,
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    def require_admin(
        x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        if settings.app_env == "development" and not x_admin_token and not authorization:
            # The local demo UI has no login screen yet. Production never takes this branch.
            return
        if (
            settings.app_env == "production"
            and settings.admin_review_token == "development-only-admin-token"
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ADMIN_REVIEW_TOKEN must be configured in production.",
            )
        bearer = None
        if authorization and authorization.lower().startswith("bearer "):
            bearer = authorization[7:].strip()
        if (x_admin_token or bearer) != settings.admin_review_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="A valid demo administrator review token is required.",
            )

    def require_editor_access(x_editor_token: str | None) -> None:
        if settings.app_env != "production":
            return
        if not settings.editor_access_token:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="EDITOR_ACCESS_TOKEN must be configured before production draft access is enabled.",
            )
        if x_editor_token != settings.editor_access_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="A valid editor access token is required for draft access.",
            )

    def ensure_mutable(exhibition: Exhibition) -> None:
        if exhibition.status in LOCKED_STATUSES:
            raise HTTPException(
                status_code=409,
                detail="This version is locked. Withdraw it or create a new draft version before editing or revalidating.",
            )

    @app.get("/api/images/{object_id:path}")
    def get_object_image(object_id: str, w: int = 1024, large: bool = False) -> Response:
        """Serve a texture-sized, CORS-clean copy of an institution image.

        Only URLs already present in the frozen collection are fetchable, so
        this cannot be used as an open proxy.
        """
        target: str | None = None
        for collection in collections.list():
            for obj in collection.objects:
                if obj.id == object_id:
                    target = (obj.image_url_large or obj.image_url) if large else obj.image_url
                    break
            if target:
                break
        if not target:
            raise HTTPException(status_code=404, detail="Object image not found.")

        try:
            payload, cached = image_cache.get(target, w)
        except ImageFetchError as error:
            raise HTTPException(
                status_code=502,
                detail={"code": error.code, "message": error.message},
            ) from error

        return Response(
            content=payload,
            media_type="image/webp",
            headers={
                # Frozen collection data means a given object image is immutable.
                "Cache-Control": "public, max-age=604800, immutable",
                "Access-Control-Allow-Origin": "*",
                "X-Image-Cache": "hit" if cached else "miss",
            },
        )

    @app.get("/api/collection/highlights")
    def collection_highlights(limit: int = 24, language: str = "zh") -> dict[str, object]:
        """A spread of real objects for the landing page.

        Drawn round-robin across coverage domains so the wall shows the actual
        range of the collection rather than whichever domain happens to be
        largest. Selection is deterministic, so the landing image set is stable
        between reloads.
        """
        collection = collections.get()
        by_domain: dict[str, list[object]] = {}
        for obj in collection.objects:
            if not obj.evidence or not obj.routing_domain_ids:
                continue
            by_domain.setdefault(obj.routing_domain_ids[0], []).append(obj)

        picked: list[object] = []
        index = 0
        domains = sorted(by_domain)
        while len(picked) < min(limit, len(collection.objects)) and domains:
            progressed = False
            for domain_id in domains:
                pool = by_domain[domain_id]
                if index < len(pool):
                    picked.append(pool[index])
                    progressed = True
                    if len(picked) >= limit:
                        break
            if not progressed:
                break
            index += 1

        # Per-domain summary so the landing page can say what is actually in
        # here. Without it a visitor is asked to start a conversation with no
        # idea what the collection can talk about.
        domain_summaries = []
        for domain_id, label_hint in domain_choices_for(collection, language).items():
            pool = by_domain.get(domain_id, [])
            if len(pool) < MIN_DOMAIN_OBJECTS:
                continue
            label, hint = label_hint
            domain_summaries.append(
                {
                    "id": domain_id,
                    "label": label,
                    "hint": hint,
                    "objectCount": len(pool),
                    "samples": [
                        {"id": obj.id, "title": obj.title_original or obj.title}
                        for obj in pool[:3]
                    ],
                }
            )
        domain_summaries.sort(key=lambda entry: -entry["objectCount"])

        dated = [
            (obj.date or "") for obj in collection.objects if obj.date
        ]
        types: dict[str, int] = {}
        institution_counts: dict[str, int] = {}
        institution_names: dict[str, str] = {}
        institution_image_licenses: dict[str, set[str]] = {}
        institution_metadata_licenses: dict[str, set[str]] = {}
        institution_text_licenses: dict[str, set[str]] = {}
        for obj in collection.objects:
            if obj.type:
                types[obj.type] = types.get(obj.type, 0) + 1
            institution_id = obj.institution_id or obj.institution
            if institution_id:
                institution_counts[institution_id] = institution_counts.get(institution_id, 0) + 1
                institution_names[institution_id] = obj.institution or institution_id
                if obj.image_license:
                    institution_image_licenses.setdefault(institution_id, set()).add(obj.image_license)
                if obj.metadata_license:
                    institution_metadata_licenses.setdefault(institution_id, set()).add(obj.metadata_license)
                if obj.curatorial_text_license:
                    institution_text_licenses.setdefault(institution_id, set()).add(
                        obj.curatorial_text_license
                    )

        institution_source_urls = {
            "cma": "https://www.clevelandart.org/open-access",
            "met": "https://www.metmuseum.org/policies/image-resources",
            "aic": "https://www.artic.edu/open-access/open-access-images",
        }

        return {
            "collectionId": collection.id,
            "collectionVersion": collection.version,
            "objectCount": len(collection.objects),
            "institutions": sorted(
                {obj.institution for obj in collection.objects if obj.institution}
            ),
            "institutionSummaries": [
                {
                    "id": institution_id,
                    "name": institution_names[institution_id],
                    "objectCount": institution_counts[institution_id],
                    "sourceUrl": institution_source_urls.get(institution_id),
                    "imageLicenses": sorted(institution_image_licenses.get(institution_id, set())),
                    "metadataLicenses": sorted(
                        institution_metadata_licenses.get(institution_id, set())
                    ),
                    "curatorialTextLicenses": sorted(
                        institution_text_licenses.get(institution_id, set())
                    ),
                }
                for institution_id in sorted(institution_counts)
            ],
            "domains": domain_summaries,
            "topTypes": [
                {"label": name, "count": count}
                for name, count in sorted(types.items(), key=lambda pair: -pair[1])[:8]
            ],
            "datedObjectCount": len(dated),
            "items": [
                {
                    "id": obj.id,
                    "title": obj.title_original or obj.title,
                    "altText": obj.alt_text,
                    "institution": obj.institution,
                }
                for obj in picked
            ],
        }

    @app.get("/health")
    def health() -> dict[str, object]:
        available = collections.list()
        retrieval: dict[str, object]
        try:
            default_collection = collections.get()
            retrieval_status = collections.retrieval_status(default_collection)
            retrieval = {
                "method": (
                    HYBRID_RETRIEVAL_METHOD
                    if retrieval_status.mode == "hybrid"
                    else BM25_RETRIEVAL_METHOD
                ),
                "version": (
                    HYBRID_RETRIEVAL_VERSION
                    if retrieval_status.mode == "hybrid"
                    else BM25_RETRIEVAL_VERSION
                ),
                "mode": retrieval_status.mode,
                "available": retrieval_status.available,
                "reason": retrieval_status.reason,
                "model": retrieval_status.model,
                "fingerprint": retrieval_status.fingerprint,
                "collectionId": default_collection.id,
                "collectionVersion": default_collection.version,
            }
        except CollectionDataError as error:
            retrieval = {
                "method": BM25_RETRIEVAL_METHOD,
                "version": BM25_RETRIEVAL_VERSION,
                "mode": "bm25",
                "available": False,
                "reason": error.message,
                "model": settings.rag_embedding_model,
                "fingerprint": None,
                "collectionId": None,
                "collectionVersion": None,
            }
        return {
            "status": "ok",
            "collections": len(available),
            "storeMode": settings.store_mode,
            "modelConfigured": bool(settings.deepseek_api_key),
            "posterModelConfigured": image_provider is not None,
            "ttsModelConfigured": audio_guide_service is not None,
            "retrieval": retrieval,
        }

    @app.post("/api/agenda/check", response_model=AgendaCheckResponse)
    def check_agenda(agenda: AgendaInput) -> AgendaCheckResponse:
        result = generator.check_agenda(agenda)
        _event(
            store,
            EventName.AGENDA_SUBMITTED,
            collectionId=result.coverage.collection_id,
            answerability=result.status,
        )
        if not result.can_generate:
            _event(
                store,
                EventName.ANSWERABILITY_FAILED,
                collectionId=result.coverage.collection_id,
                answerability=result.status,
            )
        return result

    # ------------------------------------------------------------------
    # Curator interview
    # ------------------------------------------------------------------

    @app.post("/api/interview/start", response_model=InterviewState)
    def start_interview(
        collection_id: str | None = None, language: str = "zh"
    ) -> InterviewState:
        state = interviews.start(collection_id, language)
        interview_sessions[state.id] = state
        _event(store, EventName.AGENDA_SUBMITTED, interviewId=state.id, phase="started")
        return state

    @app.get("/api/interview/{interview_id}", response_model=InterviewState)
    def get_interview(interview_id: str) -> InterviewState:
        state = interview_sessions.get(interview_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Interview session not found.")
        return state

    @app.post("/api/interview/{interview_id}/answer", response_model=InterviewState)
    async def answer_interview(interview_id: str, answer: InterviewAnswer) -> InterviewState:
        state = interview_sessions.get(interview_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Interview session not found.")
        before = len(state.transcript)
        updated = interviews.answer(state, answer)
        # A stale answer is ignored by the state machine and leaves the
        # transcript untouched; there is nothing to speak to in that case.
        if len(updated.transcript) > before:
            _voice_over_interview(updated)
        interview_sessions[interview_id] = updated
        return updated

    def _voice_over_interview(state: InterviewState) -> None:
        """Attach the curator's spoken reply, and ground the next options in it.

        This path is deliberately local and deterministic: the next question
        must not wait on an AI prose call. The original visitor question is
        carried into every reply so an automatically inferred domain can never
        replace it conversationally.
        """
        turn = state.transcript[-1]
        next_question = state.next_question
        # `==`, not `is`: ApiModel sets use_enum_values, so this id is a plain
        # string that compares equal to the member but is not identical to it.
        wants_suggestions = (
            next_question is not None
            and next_question.id == InterviewQuestionId.OPEN_QUESTION
        )
        collection = collections.get(state.collection_id)
        language = state.profile.language
        visitor_question = (
            state.profile.open_question or state.profile.free_form_question or ""
        )
        voice = interview_voice.compose_immediate(
            question_id=turn.question_id,
            answer_label=turn.answer_label,
            free_text=turn.free_text,
            skipped=turn.skipped,
            topic=state.profile.curiosity_label,
            visitor_question=visitor_question,
            available_domains=interviews.available_domains(collection, language),
            want_suggestions=wants_suggestions,
            language=language,
        )
        turn.curator_reply = voice.reply
        if (
            wants_suggestions
            and voice.suggestions
            and next_question is not None
            # A second tab may have answered this question while the model was
            # composing. Never let a late response rewind the state machine.
            and state.next_question is next_question
        ):
            suggested_question = InterviewService.open_question_question(
                state.profile.curiosity_label, voice.suggestions, language
            )
            suggested_question.step = next_question.step
            suggested_question.total_steps = next_question.total_steps
            state.next_question = suggested_question

    # ------------------------------------------------------------------
    # Curation pipeline
    # ------------------------------------------------------------------

    async def _prewarm_lobby_audio_quietly(exhibition_id: str) -> None:
        """Populate the first guide segment without delaying a finished exhibition.

        The guide remains an optional enhancement: failures here are logged for
        operators but never change either the completed job or its exhibition.
        Looking the service up on ``app.state`` also keeps the runtime and test
        injection boundary identical to the public audio-guide endpoint.
        """

        service: AudioGuideService | None = app.state.audio_guide_service
        if service is None:
            return
        try:
            exhibition = store.get_exhibition(exhibition_id)
            await service.get_audio(exhibition, "lobby")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - audio must stay best-effort
            logger.warning(
                "Lobby-audio prewarm failed for exhibition %s: %s",
                exhibition_id,
                type(exc).__name__,
            )

    def _schedule_lobby_audio_prewarm(exhibition_id: str) -> None:
        """Retain a non-blocking TTS task until it resolves or fails safely."""

        if app.state.audio_guide_service is None:
            return
        task = asyncio.create_task(_prewarm_lobby_audio_quietly(exhibition_id))
        audio_prewarm_tasks.add(task)
        task.add_done_callback(audio_prewarm_tasks.discard)

    async def _run_curation(profile: VisitorProfile, collection_id: str | None):
        """Return the coroutine a job runs, closing over the request payload."""

        async def work(job_store: JobStore, job_id: str) -> None:
            job_store.start_step(job_id, "profile")
            poster_task: asyncio.Task[bool] | None = None
            staged_exhibition_id: str | None = None

            async def start_final_poster(exhibition: Exhibition) -> None:
                """Start the poster after the frame pass fixes its final title.

                It can still run alongside the independent per-object label
                calls, without freezing a deterministic pre-frame title into
                the composed image.
                """
                nonlocal poster_task, staged_exhibition_id
                staged_exhibition_id = exhibition.id
                store.save_exhibition(exhibition)
                if image_provider is None:
                    job_store.skip_step(
                        job_id, "poster", "未配置图像模型或中文字体，展厅使用馆藏图像回退。"
                    )
                    return
                job_store.start_step(job_id, "poster")
                poster_task = asyncio.create_task(_generate_poster_quietly(exhibition.id))
                poster_background_tasks.add(poster_task)
                poster_task.add_done_callback(poster_background_tasks.discard)

            try:
                exhibition = await generator.generate_from_profile(
                    profile,
                    collection_id=collection_id,
                    emit=job_store.emitter(job_id),
                    on_frame_ready=start_final_poster,
                )

                if poster_task is not None:
                    try:
                        if poster_task.done():
                            ready = await poster_task
                        elif settings.generation_poster_wait_seconds > 0:
                            ready = await asyncio.wait_for(
                                asyncio.shield(poster_task),
                                timeout=settings.generation_poster_wait_seconds,
                            )
                        else:
                            raise asyncio.TimeoutError
                        job_store.finish_step(
                            job_id,
                            "poster",
                            "横版展览海报已生成：AI 主题主视觉与准确题名已合成，不作为馆藏或证据。"
                            if ready
                            else "海报未生成，展厅使用馆藏图像回退。",
                        )
                    except asyncio.TimeoutError:
                        # Do not cancel an optional asset or the entire valid
                        # exhibition. The retained task updates the poster
                        # record when it finishes.
                        job_store.skip_step(
                            job_id,
                            "poster",
                            "策展已完成；海报继续在后台生成，稍后会自动写入展览。",
                        )

                # The poster writes onto the stored record, so merge its latest
                # state back rather than overwriting it with the in-memory copy.
                stored_exhibition = None
                try:
                    stored_exhibition = store.get_exhibition(exhibition.id)
                except KeyError:
                    pass
                if stored_exhibition is not None:
                    if stored_exhibition.poster is not None:
                        exhibition.poster = stored_exhibition.poster
                    # Poster generation may finish after text validation. Never
                    # let the older in-memory timestamp move the persisted
                    # exhibition backwards when those records are merged.
                    exhibition.updated_at = max(
                        exhibition.updated_at, stored_exhibition.updated_at
                    )
                saved = store.save_exhibition(exhibition)

                job_store.complete(job_id, saved.id)
                # First narration is intentionally not a pipeline step.  It
                # starts only after the visit is complete and can overlap with
                # the client entering the hall, so an upstream TTS delay or
                # outage never lengthens curation progress.
                _schedule_lobby_audio_prewarm(saved.id)
                _event(
                    store,
                    EventName.GENERATION_COMPLETED,
                    saved.id,
                    provider=saved.versions.provider,
                    itemCount=len(saved.items),
                    chapterCount=len(saved.chapters),
                )
            except asyncio.CancelledError:
                if staged_exhibition_id is not None:
                    _mark_interrupted_generation(store, staged_exhibition_id)
                _event(
                    store,
                    EventName.GENERATION_FAILED,
                    staged_exhibition_id,
                    reason="job_cancelled_or_timed_out",
                )
                raise
            except Exception as error:
                if staged_exhibition_id is not None:
                    _mark_interrupted_generation(store, staged_exhibition_id)
                _event(
                    store,
                    EventName.GENERATION_FAILED,
                    staged_exhibition_id,
                    reason=type(error).__name__,
                )
                raise

        return work

    async def _generate_poster_quietly(exhibition_id: str) -> bool:
        """Best-effort poster generation that never raises into the job."""
        try:
            await generate_exhibition_poster(exhibition_id)
            refreshed = store.get_exhibition(exhibition_id)
            return bool(refreshed.poster and refreshed.poster.status == "ready")
        except Exception:  # noqa: BLE001 - a missing poster is not a curation failure
            return False

    @app.post(
        "/api/exhibitions/generate",
        response_model=GenerationJob,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def generate_exhibition(request: GenerateFromProfileRequest) -> GenerationJob:
        profile = request.profile
        if profile is None and request.interview_id:
            state = interview_sessions.get(request.interview_id)
            if state is None:
                raise HTTPException(status_code=404, detail="Interview session not found.")
            profile = state.profile
        if profile is None:
            raise HTTPException(
                status_code=422,
                detail="Provide either interviewId or an explicit visitor profile.",
            )

        _event(store, EventName.GENERATION_STARTED, motivation=profile.motivation)
        job = jobs.create(profile.language)
        jobs.run(job.id, await _run_curation(profile, request.collection_id))
        return job

    @app.get("/api/jobs/{job_id}", response_model=GenerationJob)
    def get_job(job_id: str) -> GenerationJob:
        try:
            return jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found.") from exc

    @app.get("/api/jobs/{job_id}/stream")
    async def stream_job(job_id: str) -> StreamingResponse:
        try:
            jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found.") from exc

        async def events():
            while True:
                try:
                    job = jobs.get(job_id)
                except KeyError:
                    break
                payload = job.model_dump(mode="json", by_alias=True)
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                if job.status in {"completed", "failed"}:
                    break
                await jobs.wait_for_change(job_id, timeout=15)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                # Nginx buffers SSE by default and would defeat the whole point.
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/exhibitions/generate-sync", response_model=Exhibition)
    async def generate_exhibition_sync(request: GenerateExhibitionRequest) -> Exhibition:
        """Legacy synchronous path, retained for the offline regression suite."""
        _event(store, EventName.GENERATION_STARTED)
        try:
            exhibition = await generator.generate(request.agenda)
        except CollectionDataError:
            _event(store, EventName.GENERATION_FAILED)
            raise
        saved = store.save_exhibition(exhibition)
        _event(
            store,
            EventName.GENERATION_COMPLETED,
            saved.id,
            provider=saved.versions.provider,
        )
        return saved

    @app.get("/api/exhibitions/{exhibition_id}", response_model=Exhibition)
    def get_exhibition(
        exhibition_id: str,
        x_editor_token: Annotated[str | None, Header(alias="X-Editor-Token")] = None,
    ) -> Exhibition:
        require_editor_access(x_editor_token)
        try:
            return store.get_exhibition(exhibition_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Exhibition not found.") from exc

    @app.post(
        "/api/exhibitions/{exhibition_id}/epilogue-chat",
        response_model=EpilogueChatResponse,
    )
    async def epilogue_chat(
        exhibition_id: str,
        request: EpilogueChatRequest,
        response: Response,
        x_editor_token: Annotated[str | None, Header(alias="X-Editor-Token")] = None,
    ) -> EpilogueChatResponse:
        """Continue an optional, evidence-bounded discussion with 彦远.

        Chat text is request-scoped: neither the current message nor supplied
        history is written to the exhibition store or analytics events.
        """

        require_editor_access(x_editor_token)
        try:
            exhibition = store.get_exhibition(exhibition_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Exhibition not found.") from exc

        service: EpilogueChatService = app.state.epilogue_chat_service
        try:
            reply = await service.reply(exhibition, request)
        except EpilogueChatReferenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": exc.code, "message": exc.message},
            ) from exc
        response.headers["Cache-Control"] = "no-store"
        return reply

    @app.get("/api/exhibitions/{exhibition_id}/audio-guide")
    async def get_exhibition_audio_guide(
        exhibition_id: str,
        kind: AudioGuideKind,
        ref: str | None = None,
        x_editor_token: Annotated[str | None, Header(alias="X-Editor-Token")] = None,
    ) -> FileResponse:
        """Return one trusted, content-addressed narration segment.

        The route intentionally has no arbitrary text input: narration is
        rebuilt from the stored exhibition and the requested stop reference.
        A TTS outage never mutates or removes the exhibition; clients may keep
        the hall usable and fall back to local speech synthesis.
        """

        require_editor_access(x_editor_token)
        try:
            exhibition = store.get_exhibition(exhibition_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Exhibition not found.") from exc

        service: AudioGuideService | None = app.state.audio_guide_service
        if service is None:
            configured_error: AliyunTtsProviderError | None = app.state.tts_provider_error
            error_code = configured_error.code if configured_error else "not_configured"
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": error_code,
                    "message": (
                        "Configure DASHSCOPE_API_KEY and ALIYUN_TTS_API_HOST on the "
                        "server to enable the professional audio guide. The exhibition "
                        "remains available without it."
                    ),
                },
            )

        try:
            generated = await service.get_audio(exhibition, kind, ref)
        except AudioGuideReferenceError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={"code": exc.code, "message": exc.message},
            ) from exc
        except AliyunTtsProviderError as exc:
            logger.warning(
                "TTS generation failed for exhibition %s: %s", exhibition_id, exc.code
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": exc.code,
                    "message": (
                        "The professional audio guide is temporarily unavailable. "
                        "The exhibition remains available without it."
                    ),
                },
            ) from exc

        return FileResponse(
            path=str(generated.path),
            media_type="audio/mpeg",
            headers={
                "X-Audio-Cache": generated.cache_status,
                "X-Audio-Model": generated.model,
                "X-Audio-Voice": generated.voice,
                # The route is stop-addressed, not content-hashed. Revalidate
                # after an editor changes a label while keeping the server's
                # content-addressed MP3 cache available for a fast HIT.
                "Cache-Control": "private, max-age=0, must-revalidate",
            },
        )

    @app.post("/api/exhibitions/{exhibition_id}/poster", response_model=Exhibition)
    async def generate_exhibition_poster(
        exhibition_id: str,
        force: bool = False,
        x_editor_token: Annotated[str | None, Header(alias="X-Editor-Token")] = None,
    ) -> Exhibition:
        require_editor_access(x_editor_token)
        try:
            exhibition = store.get_exhibition(exhibition_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Exhibition not found.") from exc
        ensure_mutable(exhibition)

        if exhibition.poster and exhibition.poster.status == "ready" and not force:
            return exhibition
        if exhibition.poster and exhibition.poster.status == "generating":
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "poster_generation_in_progress",
                    "message": "The exhibition poster is already being generated.",
                },
            )

        provider: AliyunImageProvider | None = app.state.image_provider
        if provider is None:
            configured_error: AliyunImageProviderError | None = (
                app.state.image_provider_error
            )
            error_code = configured_error.code if configured_error else "not_configured"
            exhibition.poster = ExhibitionPoster(
                status="failed",
                provider="aliyun-model-studio",
                model=settings.aliyun_image_model,
                generated_by=f"Alibaba Cloud Model Studio · {settings.aliyun_image_model}",
                size=settings.aliyun_image_size,
                error_code=error_code,
            )
            exhibition.updated_at = utc_now()
            # Poster work may outlive the curation job. Merge only the poster
            # field so a stale GENERATING snapshot cannot overwrite READY.
            latest = store.get_exhibition(exhibition.id)
            latest.poster = exhibition.poster
            latest.updated_at = exhibition.updated_at
            store.save_exhibition(latest)
            _event(
                store,
                EventName.POSTER_GENERATION_FAILED,
                exhibition.id,
                provider="aliyun-model-studio",
                errorCode=error_code,
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "code": error_code,
                    "message": (
                        "Configure DASHSCOPE_API_KEY and ALIYUN_IMAGE_API_HOST on the "
                        "server before generating an exhibition poster."
                    ),
                },
            )

        exhibition.poster = ExhibitionPoster(
            status="generating",
            provider="aliyun-model-studio",
            model=settings.aliyun_image_model,
            generated_by=f"Alibaba Cloud Model Studio · {settings.aliyun_image_model}",
            size=settings.aliyun_image_size,
        )
        exhibition.updated_at = utc_now()
        store.save_exhibition(exhibition)
        _event(
            store,
            EventName.POSTER_GENERATION_STARTED,
            exhibition.id,
            provider="aliyun-model-studio",
            model=settings.aliyun_image_model,
        )

        try:
            generated = await provider.generate_poster(
                PosterContext(
                    exhibition_id=exhibition.id,
                    exhibition_theme=exhibition.exhibition_theme,
                    title=exhibition.title,
                    core_question=exhibition.question,
                    subtitle=exhibition.subtitle,
                    seed=_poster_seed(exhibition.id),
                )
            )
            relative_path = generated.image_path.resolve().relative_to(
                settings.aliyun_image_output_dir.resolve()
            )
        except (AliyunImageProviderError, OSError, ValueError) as exc:
            if isinstance(exc, AliyunImageProviderError):
                error_code = exc.code
                retryable = exc.retryable
            else:
                error_code = "storage_error"
                retryable = False
            exhibition.poster = ExhibitionPoster(
                status="failed",
                provider="aliyun-model-studio",
                model=settings.aliyun_image_model,
                generated_by=f"Alibaba Cloud Model Studio · {settings.aliyun_image_model}",
                size=settings.aliyun_image_size,
                error_code=error_code,
            )
            exhibition.updated_at = utc_now()
            latest = store.get_exhibition(exhibition.id)
            latest.poster = exhibition.poster
            latest.updated_at = exhibition.updated_at
            store.save_exhibition(latest)
            _event(
                store,
                EventName.POSTER_GENERATION_FAILED,
                exhibition.id,
                provider="aliyun-model-studio",
                errorCode=error_code,
                retryable=retryable,
            )
            raise HTTPException(
                status_code=502,
                detail={
                    "code": error_code,
                    "message": (
                        "The AI poster could not be generated; the exhibition remains "
                        "available with its collection-image fallback."
                    ),
                    "retryable": retryable,
                },
            ) from exc

        exhibition.poster = ExhibitionPoster(
            status="ready",
            background_url=f"/generated/posters/{relative_path.as_posix()}",
            alt_text=(
                f"AI 生成的横版展览海报，题名“{exhibition.title}”，"
                f"主题为“{exhibition.exhibition_theme}”；文字由系统准确排版，"
                "主视觉不代表任何馆藏实物。"
            ),
            prompt_summary=(
                f"由 {generated.model} 生成与“{exhibition.exhibition_theme}”相关的无字主题主视觉；"
                f"系统随后叠加准确题名“{exhibition.title}”和“卧游 · AI策展人彦远”，"
                "未复制具体馆藏。"
            ),
            provider=generated.provider,
            model=generated.model,
            generated_by=f"Alibaba Cloud Model Studio · {generated.model}",
            size=generated.size,
            generated_at=utc_now(),
            error_code=None,
        )
        exhibition.updated_at = utc_now()
        latest = store.get_exhibition(exhibition.id)
        latest.poster = exhibition.poster
        latest.updated_at = exhibition.updated_at
        saved = store.save_exhibition(latest)
        _event(
            store,
            EventName.POSTER_GENERATION_COMPLETED,
            exhibition.id,
            provider=generated.provider,
            model=generated.model,
            size=generated.size,
        )
        return saved

    @app.patch("/api/exhibitions/{exhibition_id}/items", response_model=Exhibition)
    def patch_items(
        exhibition_id: str,
        request: ItemsPatchRequest,
        x_editor_token: Annotated[str | None, Header(alias="X-Editor-Token")] = None,
    ) -> Exhibition:
        require_editor_access(x_editor_token)
        try:
            exhibition = store.get_exhibition(exhibition_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Exhibition not found.") from exc
        ensure_mutable(exhibition)
        if request.operation == "replace":
            if not request.target_item_id or not request.replacement_object_id:
                raise HTTPException(
                    status_code=422,
                    detail="replace requires itemId/targetItemId and objectId/replacementObjectId.",
                )
            exhibition = generator.replace_item(
                exhibition, request.target_item_id, request.replacement_object_id
            )
            event_name = EventName.OBJECT_REPLACED
        else:
            exhibition = generator.reorder_items(
                exhibition,
                item_ids=request.item_ids,
                item_id=request.item_id,
                direction=request.direction,
            )
            event_name = EventName.ITEM_REORDERED
        saved = store.save_exhibition(exhibition)
        _event(store, event_name, saved.id)
        return saved

    @app.patch("/api/exhibitions/{exhibition_id}/focus", response_model=Exhibition)
    async def patch_focus(
        exhibition_id: str,
        request: FocusPatchRequest,
        x_editor_token: Annotated[str | None, Header(alias="X-Editor-Token")] = None,
    ) -> Exhibition:
        require_editor_access(x_editor_token)
        try:
            exhibition = store.get_exhibition(exhibition_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Exhibition not found.") from exc
        ensure_mutable(exhibition)
        saved = store.save_exhibition(await generator.refocus(exhibition, request.focus))
        _event(store, EventName.FOCUS_CHANGED, saved.id)
        return saved

    @app.post("/api/exhibitions/{exhibition_id}/validate", response_model=Exhibition)
    def validate_route(
        exhibition_id: str,
        x_editor_token: Annotated[str | None, Header(alias="X-Editor-Token")] = None,
    ) -> Exhibition:
        require_editor_access(x_editor_token)
        try:
            exhibition = store.get_exhibition(exhibition_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Exhibition not found.") from exc
        ensure_mutable(exhibition)
        exhibition.validation = validate_exhibition(exhibition)
        exhibition.status = (
            ExhibitionStatus.AUTO_VALIDATED
            if exhibition.validation.passed
            else ExhibitionStatus.DRAFT
        )
        exhibition.updated_at = utc_now()
        saved = store.save_exhibition(exhibition)
        _event(
            store,
            EventName.VALIDATION_PASSED if saved.validation and saved.validation.passed else EventName.VALIDATION_FAILED,
            saved.id,
        )
        return saved

    @app.post("/api/exhibitions/{exhibition_id}/publish", response_model=Exhibition)
    def request_publish(
        exhibition_id: str,
        x_editor_token: Annotated[str | None, Header(alias="X-Editor-Token")] = None,
    ) -> Exhibition:
        require_editor_access(x_editor_token)
        try:
            exhibition = store.get_exhibition(exhibition_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Exhibition not found.") from exc
        ensure_mutable(exhibition)
        exhibition.validation = validate_exhibition(exhibition)
        if not exhibition.validation.passed:
            exhibition.status = ExhibitionStatus.DRAFT
            store.save_exhibition(exhibition)
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "VALIDATION_FAILED",
                    "blockingIssues": exhibition.validation.blocking_issues,
                },
            )
        exhibition.status = ExhibitionStatus.REVIEW_PENDING
        exhibition.slug = None
        exhibition.review = None
        exhibition.updated_at = utc_now()
        return store.save_exhibition(exhibition)

    @app.get(
        "/api/admin/exhibitions",
        response_model=list[Exhibition],
        dependencies=[Depends(require_admin)],
    )
    def list_admin_exhibitions() -> list[Exhibition]:
        return store.list_exhibitions()

    @app.get(
        "/api/admin/data-audit",
        response_model=DataAuditReport,
        dependencies=[Depends(require_admin)],
    )
    def data_audit() -> DataAuditReport:
        return build_data_audit(collections)

    @app.post(
        "/api/admin/exhibitions/{exhibition_id}/approve",
        response_model=Exhibition,
        dependencies=[Depends(require_admin)],
    )
    def approve_exhibition(
        exhibition_id: str, request: ReviewRequest | None = None
    ) -> Exhibition:
        request = request or ReviewRequest()
        try:
            exhibition = store.get_exhibition(exhibition_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Exhibition not found.") from exc
        if exhibition.status != ExhibitionStatus.REVIEW_PENDING.value:
            raise HTTPException(status_code=409, detail="Only review-pending exhibitions can be approved.")
        if not request.evidence_review_confirmed:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "EVIDENCE_REVIEW_CONFIRMATION_REQUIRED",
                    "message": "Confirm that the five selected objects and their cited evidence were checked before approval.",
                },
            )
        if not request.note or len(request.note.strip()) < 10:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "REVIEW_NOTE_REQUIRED",
                    "message": "Record a short note confirming what was checked for the five selected objects.",
                },
            )
        exhibition.validation = validate_exhibition(exhibition)
        if not exhibition.validation.passed:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "VALIDATION_FAILED",
                    "blockingIssues": exhibition.validation.blocking_issues,
                },
            )
        exhibition.status = ExhibitionStatus.PUBLISHED
        exhibition.slug = _slug(exhibition)
        exhibition.review = ReviewRecord(
            decision="approved",
            reviewer=request.reviewer,
            note=request.note,
            evidence_review_confirmed=True,
        )
        exhibition.updated_at = utc_now()
        saved = store.save_exhibition(exhibition)
        _event(store, EventName.PUBLISHED, saved.id, reviewer=request.reviewer)
        return saved

    @app.post(
        "/api/admin/exhibitions/{exhibition_id}/reject",
        response_model=Exhibition,
        dependencies=[Depends(require_admin)],
    )
    def reject_exhibition(
        exhibition_id: str, request: ReviewRequest | None = None
    ) -> Exhibition:
        request = request or ReviewRequest()
        try:
            exhibition = store.get_exhibition(exhibition_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Exhibition not found.") from exc
        if exhibition.status != ExhibitionStatus.REVIEW_PENDING.value:
            raise HTTPException(status_code=409, detail="Only review-pending exhibitions can be rejected.")
        exhibition.status = ExhibitionStatus.REJECTED
        exhibition.slug = None
        exhibition.review = ReviewRecord(
            decision="rejected",
            reviewer=request.reviewer,
            note=request.note,
            evidence_review_confirmed=False,
        )
        exhibition.updated_at = utc_now()
        return store.save_exhibition(exhibition)

    @app.post(
        "/api/admin/exhibitions/{exhibition_id}/withdraw",
        response_model=Exhibition,
        dependencies=[Depends(require_admin)],
    )
    def withdraw_exhibition(exhibition_id: str) -> Exhibition:
        try:
            exhibition = store.get_exhibition(exhibition_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Exhibition not found.") from exc
        if exhibition.status != ExhibitionStatus.PUBLISHED.value:
            raise HTTPException(
                status_code=409,
                detail="Only published exhibitions can be withdrawn.",
            )
        exhibition.status = ExhibitionStatus.WITHDRAWN
        exhibition.slug = None
        exhibition.updated_at = utc_now()
        return store.save_exhibition(exhibition)

    @app.get("/e/{slug}", response_model=PublicExhibition)
    def get_published_exhibition(slug: str) -> PublicExhibition:
        try:
            return PublicExhibition.from_exhibition(store.get_by_slug(slug))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Published exhibition not found.") from exc

    @app.get("/api/public/exhibitions/{slug}", response_model=PublicExhibition)
    def get_published_exhibition_api(slug: str) -> PublicExhibition:
        return get_published_exhibition(slug)

    @app.post("/api/events", response_model=EventAccepted)
    def record_event(
        payload: EventCreate,
        response: Response,
        ic_session: Annotated[str | None, Cookie()] = None,
    ) -> EventAccepted:
        session_id = payload.session_id or ic_session or f"anon-{uuid4()}"
        if not ic_session:
            response.set_cookie(
                "ic_session",
                session_id,
                max_age=30 * 24 * 60 * 60,
                httponly=True,
                samesite="lax",
                secure=settings.app_env == "production",
            )
        event = EventLog(
            id=str(uuid4()),
            session_id=session_id,
            event=payload.event,
            exhibition_id=payload.exhibition_id,
            parameters=_safe_epilogue_event_parameters(
                payload.event, payload.parameters
            ),
        )
        saved = store.save_event(event)
        return EventAccepted(
            id=saved.id,
            session_id=saved.session_id or session_id,
            event=saved.event,
            created_at=saved.created_at,
        )

    @app.get(
        "/api/admin/analytics",
        response_model=AnalyticsResponse,
        dependencies=[Depends(require_admin)],
    )
    def analytics() -> AnalyticsResponse:
        loaded = collections.list()
        objects = [obj for collection in loaded for obj in collection.objects]
        reviewed = sum(
            1 for obj in objects if obj.evidence and all(chunk.reviewed for chunk in obj.evidence)
        )
        return store.analytics(collection_objects=len(objects), reviewed_objects=reviewed)

    @app.get(
        "/api/admin/analytics/export",
        response_model=DescriptiveStatisticsExport,
        dependencies=[Depends(require_admin)],
    )
    def export_analytics(response: Response) -> DescriptiveStatisticsExport:
        response.headers["Content-Disposition"] = (
            'attachment; filename="inquiry-curator-descriptive-statistics.json"'
        )
        return build_descriptive_export(
            collections,
            store,
            api_version=app.version,
        )

    return app


app = create_app()
