"""Record real local interview transitions without calling paid providers."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exercise-scope-replies", action="store_true",
                        help="Also simulate explicit rewrite, offered direction and skip; never generate")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if (output / "result.json").exists():
        parser.error("result exists; keep every attempt in a new directory")
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "api"))
    from app.config import Settings
    from app.main import create_app
    from fastapi.testclient import TestClient

    settings = replace(Settings.from_env(), rag_mode="bm25", rag_llm_audit_enabled=False,
                       deepseek_api_key=None, aliyun_image_api_key=None, aliyun_tts_api_key=None,
                       store_mode="json", store_path=output / "store.json")
    app = create_app(settings)
    holdout = json.loads((ROOT / "data/qa/visitor_release_holdout_20260906.json").read_text(encoding="utf-8"))
    questions = [{"id": case["id"], "question": case["question"], "kind": "frozen_holdout"}
                 for case in holdout["cases"] if case["id"] in {"release-09-it-was-fashionable", "release-10-realism-choice"}]
    questions += [
        {"id": "scope-control-more-special", "question": "我想看更特别的那一类作品", "kind": "new_ambiguous_control"},
        {"id": "scope-control-that-kind", "question": "给我看看那样的东西", "kind": "new_ambiguous_control"},
        {"id": "scope-control-explicit-subject", "question": "我想看看画面里的打铁工人，留意他们怎样配合劳动", "kind": "explicit_no_extra_question_control"},
        {"id": "scope-control-casual", "question": "没想好，随便带我逛逛", "kind": "casual_no_extra_question_control"},
    ]
    reports = []
    source_hash = hashlib.sha256((ROOT / "api/app/interview.py").read_bytes()).hexdigest()
    with TestClient(app) as client:
        for case in questions:
            started = client.post("/api/interview/start", params={"collection_id": "global_open"})
            started.raise_for_status()
            state = started.json()
            snapshots = [{"phase": "start", "state": state}]
            response = client.post(f"/api/interview/{state['id']}/answer",
                                   json={"questionId": "curiosity", "freeText": case["question"]})
            response.raise_for_status()
            state = response.json()
            snapshots.append({"phase": "after_opening", "state": state})
            after_opening = state.get("nextQuestion") or {}
            scope_reply = None
            if args.exercise_scope_replies and after_opening.get("id") == "custom_question" and after_opening.get("skippable"):
                if case["id"] == "release-09-it-was-fashionable":
                    scope_reply = {"freeText": "我说的是欧洲18世纪人物穿的衣服，想看衣领和袖子"}
                elif case["id"] == "release-10-realism-choice":
                    scope_reply = {"freeText": "我指画得像实物的静物画，想看看阴影与表面质感"}
                elif case["id"] == "scope-control-more-special":
                    scope_reply = {"value": next(option["value"] for option in after_opening["options"]
                                                  if option["value"].startswith("__scope_direction__:"))}
                else:
                    scope_reply = {"skipped": True}
                response = client.post(f"/api/interview/{state['id']}/answer",
                                       json={"questionId": "custom_question", **scope_reply})
                response.raise_for_status()
                state = response.json()
                snapshots.append({"phase": "after_scope_reply", "state": state})
            # Continue only through generic profile questions. Stop at the
            # first request for scope/evidence clarification, never generate.
            for _ in range(4):
                question = state.get("nextQuestion") or {}
                choice = {"motivation": "recharger", "prior_knowledge": "none", "duration": "5"}.get(question.get("id"))
                if choice is None:
                    break
                response = client.post(f"/api/interview/{state['id']}/answer",
                                       json={"questionId": question["id"], "value": choice})
                response.raise_for_status()
                state = response.json()
                snapshots.append({"phase": f"after_{question['id']}", "state": state})
            reports.append({**case, "openingNextQuestionId": after_opening.get("id"),
                            "openingNextPrompt": after_opening.get("prompt"),
                            "simulatedScopeReply": scope_reply,
                            "snapshots": snapshots})
    result = {"schemaVersion": "interview-scope-smoke/v1", "checkedAt": datetime.now(timezone.utc).isoformat(),
              "paidProviderCalls": 0, "providersDisabledInIsolatedSettings": True,
              "path": "POST /api/interview/start then POST /api/interview/{id}/answer; no /generate",
              "sourceSha256": source_hash, "cases": reports}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output / "result.json"), "cases": [
        {key: case[key] for key in ("id", "openingNextQuestionId", "openingNextPrompt")} for case in reports]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
