"""One real visitor-path smoke, isolated from the shared exhibition store.

Does not alter product sources or disable optional generation stages. An
explicit --relay may route Aliyun HTTPS through an approved temporary relay;
without it the process keeps the operator's network environment unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUESTION = "我想看看带花卉纹样的陶瓷器，留意花朵和器物形状怎样相配"


def stage_budget_overrides(settings_class) -> dict[str, str]:
    """Use shipped dataclass defaults, never the operator's loaded .env values."""
    names = (
        "deepseek_timeout_seconds", "deepseek_frame_timeout_seconds",
        "deepseek_labels_timeout_seconds", "rag_retrieval_timeout_seconds",
        "rag_planning_timeout_seconds", "deepseek_query_review_thinking",
        "generation_poster_wait_seconds", "generation_job_timeout_seconds",
    )
    defaults = {field.name: field.default for field in fields(settings_class)}
    return {name.upper(): str(defaults[name]) for name in names}


def process_environment(output_dir: Path, relay: str | None) -> dict[str, str]:
    """Return only process-scoped overrides; no relay means no proxy mutation."""
    result = {
        "RAG_MODE": "hybrid", "STORE_MODE": "json",
        "STORE_PATH": str(output_dir / "store.json"),
        "RAG_TRACE_DIR": str(output_dir / "retrieval-traces"),
        "ALIYUN_IMAGE_OUTPUT_DIR": str(output_dir / "generated-posters"),
        "ALIYUN_TTS_OUTPUT_DIR": str(output_dir / "generated-audio"),
    }
    if relay:
        result["HTTPS_PROXY"] = relay
        # Derived from frozen global_open image URLs; does not redirect the
        # DeepSeek or institution image requests through the optional relay.
        result["NO_PROXY"] = ",".join((
            "api.deepseek.com", "localhost", "127.0.0.1", "testserver",
            "images.metmuseum.org", "openaccess-cdn.clevelandart.org", "www.artic.edu",
        ))
    return result


async def pending_asset_tasks(app) -> dict[str, int]:
    return {
        name: sum(not task.done() for task in tuple(getattr(app.state, name, ())))
        for name in ("poster_background_tasks", "audio_prewarm_tasks")
    }


def wait_for_optional_assets(client, app, seconds: float) -> dict:
    """Observe existing tasks only: never start or retry any paid generation."""
    started = time.perf_counter()
    snapshots = []
    previous = None
    while True:
        pending = client.portal.call(pending_asset_tasks, app)
        if pending != previous:
            snapshots.append({"elapsedSeconds": time.perf_counter() - started, **pending})
            previous = pending
        remaining = seconds - (time.perf_counter() - started)
        if not any(pending.values()) or remaining <= 0:
            return {
                "budgetSeconds": seconds, "elapsedSeconds": time.perf_counter() - started,
                "snapshots": snapshots, "pendingAtClientTeardown": pending,
                "allObservedTasksSettled": not any(pending.values()),
                "boundary": "Settled does not mean successful. Pending isolated-client tasks may be cancelled at teardown; no task is retried here.",
            }
        time.sleep(min(0.5, remaining))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--relay", default=None,
                        help="Optional approved HTTPS relay; omitted leaves proxy environment unchanged")
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--image-cache-dir", type=Path,
                        help="Optional shared cache of public collection images; never shares exhibition stores")
    parser.add_argument("--job-timeout-seconds", type=float)
    parser.add_argument("--default-stage-budgets", action="store_true",
                        help="Use shipped Settings dataclass stage and job defaults in this process only")
    parser.add_argument("--optional-assets-wait-seconds", type=float, default=0.0,
                        help="After curation, observe existing poster/lobby-TTS tasks for 0–120 seconds; no new generation")
    args = parser.parse_args()
    if not 0 <= args.optional_assets_wait_seconds <= 120:
        parser.error("--optional-assets-wait-seconds must be between 0 and 120")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise RuntimeError("This smoke already has a result; do not silently rerun it")
    sys.path.insert(0, str(ROOT / "api"))
    # Importing config can load .env, but does not instantiate Settings. Apply
    # explicit process overrides afterwards and only then call from_env().
    from app.config import Settings
    process_overrides = process_environment(output_dir, args.relay)
    if args.image_cache_dir is not None:
        process_overrides["IMAGE_CACHE_DIR"] = str(args.image_cache_dir.resolve())
    if args.default_stage_budgets:
        process_overrides.update(stage_budget_overrides(Settings))
    if args.job_timeout_seconds is not None:
        if args.job_timeout_seconds <= 0:
            raise ValueError("job timeout must be positive")
        process_overrides["GENERATION_JOB_TIMEOUT_SECONDS"] = str(args.job_timeout_seconds)
    os.environ.update(process_overrides)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(output_dir / "runtime.log", encoding="utf-8"), logging.StreamHandler()],
    )
    from fastapi.testclient import TestClient
    from app.main import create_app

    def hashes():
        return {
            str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "api" / "app").rglob("*.py"))
        }

    settings = Settings.from_env()
    before_hashes = hashes()
    state = {
        "startedAt": datetime.now(timezone.utc).isoformat(),
        "question": args.question,
        "scope": "one real visitor generation in an isolated in-process API; deployment network, server load and browser behavior are not established by this run",
        "processOverrides": process_overrides,
        "runtime": {
            "ragMode": settings.rag_mode,
            "embeddingProvider": settings.rag_embedding_provider,
            "embeddingModel": settings.rag_embedding_model,
            "rerankEnabled": settings.rag_rerank_enabled,
            "rerankModel": settings.rag_rerank_model,
            "textModel": settings.deepseek_model,
            "labelsModel": settings.deepseek_labels_model,
            "retrievalTimeoutSeconds": settings.rag_retrieval_timeout_seconds,
            "planningTimeoutSeconds": settings.rag_planning_timeout_seconds,
            "queryReviewThinking": settings.deepseek_query_review_thinking,
            "jobTimeoutSeconds": settings.generation_job_timeout_seconds,
        },
        "modelStages": [], "audioStages": [], "jobSnapshots": [], "initialRetrieval": None, "audit": None,
        "sourceHashesBefore": before_hashes,
    }

    def save():
        result_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    started = time.perf_counter()
    app = create_app(settings)
    generator = app.state.generator
    original_initial = generator.prepare_initial_retrieval
    original_audit = generator._agentic_retrieve
    original_model = generator._generate_model_json

    async def observe_initial(*args, **kwargs):
        clock = time.perf_counter()
        try:
            result = await original_initial(*args, **kwargs)
            state["initialRetrieval"] = {
                "elapsedSeconds": time.perf_counter() - clock,
                "diagnostics": result.diagnostics,
                "queryPlan": asdict(result.query_plan) if result.query_plan else None,
                "objectIds": [item.obj.id for item in result.results],
            }
            return result
        except Exception as error:
            state["initialRetrieval"] = {"errorType": type(error).__name__, "error": str(error), "elapsedSeconds": time.perf_counter() - clock}
            raise

    async def observe_audit(*args, **kwargs):
        clock = time.perf_counter()
        result = await original_audit(*args, **kwargs)
        state["audit"] = {
            "elapsedSeconds": time.perf_counter() - clock,
            "answerability": result.answerability,
            "interpretation": result.interpretation,
            "coverageGap": result.coverage_gap,
            "failureCode": result.failure_code,
            "warningCode": result.warning_code,
            "warningDetail": result.warning_detail,
            "expandedQueries": list(result.expanded_queries),
            "diagnostics": getattr(result, "audit_diagnostics", {}),
            "accepted": [{"objectId": item.obj.id, "evidenceIds": list(item.matched_evidence_ids)} for item in result.results],
            "stageObjectIds": {stage: [item.obj.id for item in items] for stage, items in result.stage_results.items()},
        }
        return result

    async def observe_model(*args, **kwargs):
        record = {
            "stage": kwargs.get("stage"), "timeoutSeconds": kwargs.get("timeout_seconds"),
            "imageObjectIds": [item.object_id for item in (kwargs.get("vision_images") or [])],
        }
        state["modelStages"].append(record)
        clock = time.perf_counter()
        try:
            value = await original_model(*args, **kwargs)
            record["status"] = "completed"
            record["output"] = value
            return value
        except Exception as error:
            record.update({"status": "failed", "errorType": type(error).__name__, "error": str(error)})
            raise
        finally:
            record["elapsedSeconds"] = time.perf_counter() - clock

    # Transparent observations only; all calls delegate to the frozen methods.
    generator.prepare_initial_retrieval = observe_initial
    generator._agentic_retrieve = observe_audit
    generator._generate_model_json = observe_model
    audio_service = app.state.audio_guide_service
    if audio_service is not None:
        original_audio = audio_service.get_audio

        async def observe_audio(*args, **kwargs):
            record = {"kind": args[1] if len(args) > 1 else kwargs.get("kind"), "status": "running"}
            state["audioStages"].append(record)
            clock = time.perf_counter()
            try:
                value = await original_audio(*args, **kwargs)
                from app.providers.aliyun_tts import is_valid_mp3
                record.update({"status": "completed", "path": str(value.path),
                               "cacheStatus": value.cache_status,
                               "validMp3": is_valid_mp3(value.path.read_bytes()),
                               "bytes": value.path.stat().st_size})
                return value
            except asyncio.CancelledError:
                record["status"] = "cancelled_at_observer_teardown"
                raise
            except Exception as error:
                record.update({"status": "failed", "errorType": type(error).__name__,
                               "errorCode": getattr(error, "code", None)})
                raise
            finally:
                record["elapsedSeconds"] = time.perf_counter() - clock

        audio_service.get_audio = observe_audio
    try:
        with TestClient(app) as client:
            request = {
                "collectionId": "global_open",
                "profile": {
                    "curiosityDomainId": "other", "freeFormQuestion": args.question,
                    "motivation": "explorer", "priorKnowledge": "some",
                    "durationMinutes": 5, "language": "zh",
                },
            }
            state["request"] = request
            response = client.post("/api/exhibitions/generate", json=request)
            state["createHttpStatus"] = response.status_code
            state["createResponse"] = response.json()
            state["createElapsedSeconds"] = time.perf_counter() - started
            if response.status_code != 202:
                state["outcome"] = "request_rejected"
                return 1
            job_id = response.json()["id"]
            stop_at = time.perf_counter() + settings.generation_job_timeout_seconds + 90
            previous = None
            while time.perf_counter() < stop_at:
                job = client.get(f"/api/jobs/{job_id}").json()
                signature = json.dumps(job, sort_keys=True)
                if signature != previous:
                    state["jobSnapshots"].append({"elapsedSeconds": time.perf_counter() - started, "job": job})
                    previous = signature
                    save()
                if job["status"] in {"completed", "failed"}:
                    state["job"] = job
                    break
                time.sleep(1)
            else:
                state["outcome"] = "observer_deadline"
                return 1
            state["generationElapsedSeconds"] = time.perf_counter() - started
            if job["status"] == "failed":
                state["outcome"] = "generation_failed"
                return 1
            exhibition = app.state.store.get_exhibition(job["exhibitionId"])
            poster_at_completion = exhibition.poster.status if exhibition.poster else "not_started"
            state["optionalAssetsObservation"] = wait_for_optional_assets(
                client, app, args.optional_assets_wait_seconds,
            )
            exhibition = app.state.store.get_exhibition(job["exhibitionId"])
            if exhibition.poster and exhibition.poster.status == "ready" and exhibition.poster.background_url:
                poster_url = exhibition.poster.background_url
                if poster_url.startswith("/generated/posters/"):
                    asset = client.get(poster_url)
                    state["posterAssetRead"] = {"httpStatus": asset.status_code,
                                                "contentType": asset.headers.get("content-type"),
                                                "bytes": len(asset.content)}
            state["exhibition"] = exhibition.model_dump(mode="json", by_alias=True)
            citation_errors = []
            sentence_count = 0
            for item in exhibition.items:
                allowed = {e.id for e in item.object.evidence}
                for sentence in item.label_sentences:
                    sentence_count += 1
                    for evidence_id in sentence.evidence_ids:
                        if evidence_id not in allowed:
                            citation_errors.append({"itemId": item.id, "sentenceId": sentence.id, "evidenceId": evidence_id})
            state["verification"] = {
                "itemCount": len(exhibition.items),
                "uniqueObjectCount": len({item.object.id for item in exhibition.items}),
                "labelSentenceCount": sentence_count,
                "citationOwnershipErrors": citation_errors,
                "validatorPassed": bool(exhibition.validation and exhibition.validation.passed),
                "coverageLimits": exhibition.coverage_limits,
                "posterStatusAtCompletion": poster_at_completion,
                "posterStatusAfterObservation": exhibition.poster.status if exhibition.poster else "not_started",
                "optionalBackgroundStages": "not disabled; only existing poster/lobby-TTS tasks observed; chapter/artwork/epilogue audio and browser playback remain separate checks",
            }
            state["outcome"] = "completed" if state["verification"]["validatorPassed"] and len(exhibition.items) == 5 and not citation_errors else "completed_with_validation_issues"
            return 0 if state["outcome"] == "completed" else 1
    except Exception as error:
        state["outcome"] = "smoke_exception"
        state["error"] = {"type": type(error).__name__, "message": str(error)}
        logging.exception("Isolated smoke failed")
        return 1
    finally:
        state["elapsedSeconds"] = time.perf_counter() - started
        state["completedAt"] = datetime.now(timezone.utc).isoformat()
        state["sourceHashesAfter"] = hashes()
        state["sourceUnchanged"] = state["sourceHashesAfter"] == before_hashes
        save()
        print(json.dumps({"outcome": state.get("outcome"), "elapsedSeconds": state["elapsedSeconds"], "resultPath": str(result_path), "verification": state.get("verification"), "job": state.get("job")}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
