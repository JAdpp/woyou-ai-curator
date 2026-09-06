"""Frozen visitor questions through interview and retrieval, not generation.

Default: inventory only, with no business-module import, directory creation or
provider call. --execute explicitly permits the configured embedding, rerank,
planning and evidence/vision audit calls. Every execution needs a new directory.
No frame, labels, poster, TTS, server mutation or case-level retry is performed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, fields
from datetime import datetime, timezone
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "data/qa/visitor_release_fresh_20260906_rc7.json"
BEHAVIORS = {"retrieve_or_explain_gap", "clarify", "out_of_scope"}
PROFILE_FIELDS = {
    "curiosityDomainId", "curiosityLabel", "motivation", "priorKnowledge",
    "durationMinutes", "excludedTopics", "companion", "language",
}
SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,100}\Z")


class ProbeError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_error(error: BaseException) -> dict[str, str | None]:
    """Never persist arbitrary provider exception messages/URLs/headers."""
    code = getattr(error, "code", None)
    return {"type": type(error).__name__,
            "code": code if isinstance(code, str) and SAFE_IDENTIFIER.fullmatch(code) else None}


def load_cases(path: Path, case_ids: list[str] | None = None,
               expected_sha256: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    raw_bytes = path.read_bytes()
    digest = sha256(raw_bytes).hexdigest()
    if expected_sha256 and digest != expected_sha256.lower():
        raise ValueError("question_file_hash_mismatch")
    raw = json.loads(raw_bytes)
    if raw.get("status") != "frozen_before_first_run":
        raise ValueError("question_file_not_frozen")
    indexed = {}
    for case in raw["cases"]:
        key = case["id"]
        if (not isinstance(key, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,99}", key)
                or key in indexed or case.get("expectedBehavior") not in BEHAVIORS):
            raise ValueError("invalid_case_identity_or_behavior")
        question = case.get("question")
        if not isinstance(question, str) or not question.strip() or len(question) > 500:
            raise ValueError("invalid_original_question")
        profile = {**raw.get("defaultProfile", {}), **case.get("profile", {})}
        if set(profile) - PROFILE_FIELDS:
            raise ValueError("profile_contains_non_product_fields")
        # The fixed smoke uses the product's five-minute/five-object plan.
        if profile.get("durationMinutes", 5) != 5:
            raise ValueError("probe_requires_five_minute_product_profile")
        profile.setdefault("durationMinutes", 5)
        profile.setdefault("language", "zh")
        indexed[key] = {"id": key, "question": question, "profile": profile,
                        # Used only by this observer to select its entry path.
                        # It is never included in an app/provider payload.
                        "expectedBehavior": case["expectedBehavior"]}
    selected = case_ids if case_ids is not None else list(indexed)
    if not selected or len(set(selected)) != len(selected) or any(key not in indexed for key in selected):
        raise ValueError("unknown_or_duplicate_case_id")
    return digest, [indexed[key] for key in selected]


def product_input(case: dict[str, Any]) -> dict[str, Any]:
    """Strict projection: benchmark intent/checks/expected behavior stay out."""
    return {"question": case["question"],
            "profile": {key: value for key, value in case["profile"].items() if key in PROFILE_FIELDS}}


def source_hashes() -> dict[str, str]:
    paths = [*sorted((ROOT / "api/app").rglob("*.py")), Path(__file__).resolve()]
    return {path.relative_to(ROOT).as_posix(): sha256(path.read_bytes()).hexdigest() for path in paths}


def process_environment(output: Path, settings_class) -> dict[str, str]:
    defaults = {field.name: field.default for field in fields(settings_class)}
    runtime = output / "runtime"
    values = {
        "RAG_MODE": "hybrid", "RAG_LLM_AUDIT_ENABLED": "true", "RAG_VISUAL_AUDIT_ENABLED": "true",
        "STORE_MODE": "json", "STORE_PATH": str(runtime / "store.json"),
        "IMAGE_CACHE_DIR": str(runtime / "cache/objects"),
        "RAG_FILTER_INDEX_DIR": str(runtime / "cache/filters"),
        "RAG_TRACE_DIR": str(runtime / "retrieval-traces"),
        "ALIYUN_IMAGE_OUTPUT_DIR": str(runtime / "unused-posters"),
        "ALIYUN_TTS_OUTPUT_DIR": str(runtime / "unused-audio"),
    }
    # Apply shipped budgets together so an older .env's frame budget cannot
    # shrink or reject the requested 70-second retrieval setting on import.
    for name in ("rag_retrieval_timeout_seconds", "rag_llm_audit_timeout_seconds",
                 "deepseek_timeout_seconds", "deepseek_frame_timeout_seconds",
                 "deepseek_labels_timeout_seconds", "generation_poster_wait_seconds",
                 "generation_job_timeout_seconds"):
        values[name.upper()] = str(defaults[name])
    # Frozen dense matrices and existing local model artifacts remain read-only
    # inputs. The serving loader uses mmap='r', allow_download=False, no build.
    # No proxy or provider credential is changed by this probe.
    return values


@contextmanager
def environment_overrides(values: dict[str, str]):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def interview_entry(client, product: dict[str, Any], *, complete_generic: bool,
                    record: dict[str, Any], save) -> None:
    """Public route only; never invent an answer to a scope negotiation."""
    entry = {"snapshots": [], "completeGenericProfile": complete_generic,
             "generationRequested": False}
    record["interview"] = entry

    def request(path: str, *, body=None, params=None, phase: str):
        response = client.post(path, json=body, params=params)
        snapshot = {"phase": phase, "httpStatus": response.status_code}
        if 200 <= response.status_code < 300:
            state = response.json()
            snapshot["state"] = state
        else:
            state = None
            snapshot["error"] = {"type": "InterviewHttpError", "code": f"HTTP_{response.status_code}"}
        entry["snapshots"].append(snapshot)
        save()
        if state is None:
            raise ProbeError("INTERVIEW_HTTP_ERROR")
        return state

    state = request("/api/interview/start", phase="start", params={
        "collection_id": "global_open", "language": product["profile"].get("language", "zh")})
    state = request(f"/api/interview/{state['id']}/answer", phase="after_opening", body={
        "questionId": "curiosity", "freeText": product["question"]})
    entry["openingNextQuestionId"] = (state.get("nextQuestion") or {}).get("id")
    profile = product["profile"]
    generic_answers = {
        "motivation": {"value": profile.get("motivation", "explorer")},
        "prior_knowledge": {"value": profile.get("priorKnowledge", "some")},
        "duration": {"value": str(profile.get("durationMinutes", 5))},
        "exclusions": ({"freeText": "、".join(profile["excludedTopics"])}
                       if profile.get("excludedTopics") else {"value": "none"}),
    }
    seen = set()
    if complete_generic:
        for _ in range(7):
            next_question = state.get("nextQuestion") or {}
            key = next_question.get("id")
            if state.get("complete") or key not in generic_answers:
                break
            if key in seen:
                entry["repeatedGenericQuestion"] = key
                break
            seen.add(key)
            state = request(f"/api/interview/{state['id']}/answer", phase=f"after_{key}",
                            body={"questionId": key, **generic_answers[key]})
    next_question = state.get("nextQuestion") or {}
    entry.update({
        "complete": bool(state.get("complete")), "nextQuestionId": next_question.get("id"),
        "stoppedForVisitorChoice": next_question.get("id") in {"custom_question", "open_question", "negotiation"},
        "finalProfile": state.get("profile"), "finalNegotiationNote": state.get("negotiationNote"),
        "originalQuestionPreserved": product["question"] in {
            (state.get("profile") or {}).get("freeFormQuestion"),
            (state.get("profile") or {}).get("openQuestion")},
        "boundary": "Snapshots, not an automatic semantic pass. No suggested replacement topic was chosen.",
    })
    save()


def candidate_payload(result, diagnostics: dict[str, Any]) -> dict[str, Any]:
    checks = []
    for stage, report in diagnostics.items():
        if stage.startswith("pass") and isinstance(report, dict):
            checks.extend({"auditPass": stage, **check} for check in report.get("conditions", [])
                          if check.get("objectId") == result.obj.id)
    return {"objectId": result.obj.id, "score": result.score,
            "object": result.obj.model_dump(mode="json", by_alias=True),
            "matchedEvidenceIds": list(result.matched_evidence_ids),
            "matchedAnchorTerms": list(result.matched_anchor_terms),
            "fieldScores": dict(result.field_scores), "retrievalSources": list(result.retrieval_sources),
            "denseScore": result.dense_score, "evidenceScore": result.evidence_score,
            "conditionChecks": checks, "setWitnesses": list(result.set_witnesses)}


async def retrieval_only(app, product: dict[str, Any], record: dict[str, Any], save) -> None:
    from app.models import VisitorProfile

    generator = app.state.generator
    profile = VisitorProfile.model_validate({**product["profile"], "freeFormQuestion": product["question"]})
    agenda = profile.to_agenda("global_open")
    if agenda.question != product["question"] or profile.item_count != 5:
        raise ProbeError("PROBE_PRODUCT_INPUT_CHANGED")
    record["retrievalRequest"] = {"agenda": agenda.model_dump(mode="json", by_alias=True),
                                   "requiredCount": profile.item_count}
    collection = app.state.collections.get("global_open")
    started = perf_counter()
    deadline = started + float(app.state.settings.rag_retrieval_timeout_seconds)
    record["retrievalBudgetSeconds"] = app.state.settings.rag_retrieval_timeout_seconds
    original_model = generator._generate_model_json
    record["modelStages"] = []

    async def observe_model(*args, **kwargs):
        stage = {"stage": kwargs.get("stage"), "timeoutSeconds": kwargs.get("timeout_seconds"),
                 "imageObjectIds": [image.object_id for image in (kwargs.get("vision_images") or ())]}
        record["modelStages"].append(stage)
        clock = perf_counter()
        try:
            value = await original_model(*args, **kwargs)
            stage.update(status="completed", output=value)
            return value
        except BaseException as error:
            stage.update(status="failed", error=safe_error(error))
            raise
        finally:
            stage["elapsedSeconds"] = perf_counter() - clock
            save()

    generator._generate_model_json = observe_model
    try:
        initial_started = perf_counter()
        initial = await generator.prepare_initial_retrieval(agenda, collection, deadline=deadline)
        record["initialRetrieval"] = {
            "elapsedSeconds": perf_counter() - initial_started,
            "queryPlan": asdict(initial.query_plan) if initial.query_plan else None,
            "diagnostics": initial.diagnostics, "objectIds": [row.obj.id for row in initial.results]}
        save()
        audit_started = perf_counter()
        audit = await generator._agentic_retrieve(
            agenda, collection, initial.results, required_count=profile.item_count,
            deadline=deadline, initial_query_plan=initial.query_plan, planning_attempted=True)
        record["audit"] = {
            "elapsedSeconds": perf_counter() - audit_started, "applied": audit.audit_applied,
            "answerability": audit.answerability, "interpretation": audit.interpretation,
            "coverageGap": audit.coverage_gap, "failureCode": audit.failure_code,
            "warningCode": audit.warning_code, "expandedQueries": list(audit.expanded_queries),
            "exhibitionSetRequirements": list(audit.exhibition_set_requirements),
            "diagnostics": audit.audit_diagnostics,
            "accepted": [candidate_payload(row, audit.audit_diagnostics) for row in audit.results],
            "stageObjectIds": {stage: [row.obj.id for row in rows] for stage, rows in audit.stage_results.items()},
        }
        record["status"] = "retrieval_failed" if audit.failure_code else "retrieval_returned"
    except BaseException as error:
        # The plan-unavailable error owns useful structured diagnostics, but its
        # arbitrary message/details (which can contain upstream URLs) stay out.
        details = getattr(error, "details", {})
        if isinstance(details, dict) and isinstance(details.get("planningDiagnostics"), dict):
            record["initialRetrieval"] = {"queryPlan": None, "diagnostics": details["planningDiagnostics"]}
        raise
    finally:
        generator._generate_model_json = original_model
        record["retrievalElapsedSeconds"] = perf_counter() - started
        save()


def write_json(path: Path, value: Any, secrets: tuple[str, ...] = ()) -> None:
    serialized = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    for secret in secrets:
        serialized = serialized.replace(secret, "[REDACTED]")
    path.write_text(serialized + "\n", encoding="utf-8")


def execute(args, cases: list[dict[str, Any]], question_digest: str) -> int:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    before = source_hashes()
    state = {"schemaVersion": "visitor-retrieval-probe/v1", "startedAt": now(),
             "status": "starting", "questionFileSha256": question_digest,
             "questionFile": str(args.questions.resolve()), "sourceHashesBefore": before,
             "caseIds": [case["id"] for case in cases], "cases": [],
             "scope": "Public interview then retrieval-only. No selected exhibition, labels, poster, TTS, browser or production readiness claim.",
             "attemptPolicy": "One execution per case. Product-internal bounded retries remain unchanged; no observer-level retry.",
             "oraclePolicy": "Only original question and whitelisted profile reach the app. Expected behavior selects observer phase only."}
    secrets: tuple[str, ...] = ()

    def save_run():
        write_json(output / "result.json", state, secrets)

    save_run()
    old_logging_disable = logging.root.manager.disable
    # Provider error logging may contain URLs. Only our fixed summaries go to
    # stdout; complete structured diagnostics go to the isolated artifacts.
    logging.disable(logging.CRITICAL)
    try:
        sys.path.insert(0, str(ROOT / "api"))
        from app.config import Settings
        overrides = process_environment(output, Settings)
        with environment_overrides(overrides):
            settings = Settings.from_env()
            secrets = tuple(value for field in fields(settings)
                            if "api_key" in field.name and isinstance(value := getattr(settings, field.name), str) and len(value) >= 8)
            from fastapi.testclient import TestClient
            from app.main import create_app
            app = create_app(settings)
            if not (app.state.generator.provider.configured
                    and getattr(app.state.generator.provider, "supports_retrieval_query_planning", False)):
                raise ProbeError("QUERY_PLANNING_PROVIDER_REQUIRED")
            state["runtime"] = {
                "processOverrides": overrides, "ragMode": settings.rag_mode,
                "textModel": settings.deepseek_model, "visualModel": settings.deepseek_labels_model,
                "embeddingProvider": settings.rag_embedding_provider, "embeddingModel": settings.rag_embedding_model,
                "rerankEnabled": settings.rag_rerank_enabled, "rerankModel": settings.rag_rerank_model,
                "visualAuditEnabled": settings.rag_visual_audit_enabled,
                "retrievalTimeoutSeconds": settings.rag_retrieval_timeout_seconds,
                "denseIndexRoot": str(settings.rag_index_dir),
                "inputPolicy": "Versioned dense indexes/local model artifacts read-only; store/images/FTS/traces isolated; no dense rebuild or download.",
            }
            collection = app.state.collections.get("global_open")
            state["collection"] = {"id": collection.id, "version": collection.version,
                                   "objectsSha256": collection.objects_sha256, "objectCount": len(collection.objects)}
            manager = app.state.collections._dense_manager
            index_path = manager.expected_path(collection)
            state["runtime"]["denseManifestSha256"] = (
                sha256((index_path / "manifest.json").read_bytes()).hexdigest()
                if (index_path / "manifest.json").is_file() else None)
            dense_status = manager.status(collection)
            state["runtime"]["denseStatusBefore"] = {
                "available": dense_status.available, "mode": dense_status.mode,
                "model": dense_status.model, "fingerprint": dense_status.fingerprint}
            save_run()
            if not dense_status.available:
                raise ProbeError("DENSE_INDEX_UNAVAILABLE")
            with TestClient(app) as client:
                for case in cases:
                    if source_hashes() != before:
                        raise ProbeError("SOURCE_CHANGED_BETWEEN_CASES")
                    case_dir = output / case["id"]
                    case_dir.mkdir(exist_ok=False)
                    product = product_input(case)
                    record = {"id": case["id"], **product, "startedAt": now(), "status": "running",
                              "questionSha256": sha256(product["question"].encode("utf-8")).hexdigest(),
                              "expectedBehaviorForOfflineReview": case["expectedBehavior"],
                              "generationRequested": False}

                    def save_case():
                        write_json(case_dir / "result.json", record, secrets)

                    save_case()
                    started = perf_counter()
                    try:
                        routing_only = case["expectedBehavior"] in {"clarify", "out_of_scope"}
                        interview_entry(client, product, complete_generic=routing_only, record=record, save=save_case)
                        if routing_only:
                            record["status"] = "interview_observed"
                        else:
                            client.portal.call(retrieval_only, app, product, record, save_case)
                    except Exception as error:
                        record.update(status="failed", error=safe_error(error))
                    finally:
                        record.update(completedAt=now(), elapsedSeconds=perf_counter()-started)
                        save_case()
                        summary = {"id": case["id"], "status": record["status"],
                                   "elapsedSeconds": round(record["elapsedSeconds"], 3),
                                   "answerability": record.get("audit", {}).get("answerability"),
                                   "acceptedCount": len(record.get("audit", {}).get("accepted", [])),
                                   "nextQuestionId": record.get("interview", {}).get("nextQuestionId")}
                        state["cases"].append(summary)
                        save_run()
                        print(json.dumps(summary, ensure_ascii=False), flush=True)
            if source_hashes() != before:
                raise ProbeError("SOURCE_CHANGED_DURING_CASE")
            state["status"] = "completed"
            return 1 if any(case["status"] in {"failed", "retrieval_failed"} for case in state["cases"]) else 0
    except Exception as error:
        state.update(status="failed", error=safe_error(error))
        return 1
    finally:
        state["sourceHashesAfter"] = source_hashes()
        state["sourceUnchanged"] = state["sourceHashesAfter"] == before
        state["completedAt"] = now()
        save_run()
        logging.disable(old_logging_disable)
        print(json.dumps({"status": state["status"], "completedCases": len(state["cases"]),
                          "sourceUnchanged": state["sourceUnchanged"]}), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--questions-sha256", help="Optional exact frozen file SHA-256 guard")
    parser.add_argument("--case", action="append", help="Repeat for a subset; omission inventories/runs all frozen cases")
    parser.add_argument("--output", type=Path, help="New once-only output directory; required with --execute")
    parser.add_argument("--execute", action="store_true", help="Permit real configured retrieval/vision provider calls")
    args = parser.parse_args(argv)
    try:
        digest, cases = load_cases(args.questions, args.case, args.questions_sha256)
    except (OSError, ValueError, KeyError, TypeError):
        parser.error("Invalid frozen questions, case IDs, profile fields or SHA-256")
    if args.output is not None and args.output.exists():
        parser.error("Output directory already exists; preserve that attempt and choose a new directory")
    if not args.execute:
        print(json.dumps({"execute": False, "paidCalls": 0, "questionFileSha256": digest,
                          "cases": [{"id": case["id"], "expectedBehavior": case["expectedBehavior"]} for case in cases]}))
        return 0
    if args.output is None:
        parser.error("--execute requires --output pointing to a new directory")
    return execute(args, cases, digest)


if __name__ == "__main__":
    raise SystemExit(main())
