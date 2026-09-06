"""One diagnostic vision call for a pending immutable blind-review batch."""
import argparse
import asyncio
import json
from pathlib import Path

import review_recovered_pool_extension as pool


async def run(args):
    output = args.pool_dir / f"contract-diagnostic-{args.query_id}.json"
    if output.exists():
        raise ValueError("diagnostic exists; inspect it rather than silently repeating the call")
    packages = pool.read_jsonl(args.pool_dir / "blind-inputs.jsonl")
    package = next(p for p in packages if p["queryId"] == args.query_id)
    dataset = pool.FrozenRetrievalEvalV1(pool.ROOT / "data/qa/retrieval_eval_v1")
    settings = pool.review.Settings.from_env()
    provider = pool.review.DeepSeekProvider(settings)
    candidates = package["payload"]["candidates"][:4]
    cache = pool.review.ImageCache(settings.store_path.parent / "cache" / "objects",
                                   limit_bytes=settings.image_cache_limit_mb * 1024 * 1024)
    dtos = [{"objectId": c["objectId"], "object": dataset.objects[c["objectId"]], "evidence": c["evidence"]}
            for c in candidates]
    prepared, failed = await pool.review._prepare_vision_images(cache, dtos)
    if failed:
        raise ValueError("diagnostic image preparation failed")
    payload = {**package["payload"], "candidates": candidates}
    raw = await provider.generate_qrel_vision_json(pool.VISION_PROMPT, payload, [item[1] for item in prepared])
    result = {"queryId": args.query_id, "imageCount": len(prepared), "inputSha256": pool.digest(payload),
              "expectedObjectIds": [c["objectId"] for c in candidates], "rawModelResponse": raw,
              "scope": "diagnostic only; no review label or existing journal changed"}
    try:
        pool.review.validate_candidate_batch(raw, candidates)
        result["validation"] = "passed"
    except ValueError as error:
        result["validation"] = "failed"
        result["validationError"] = str(error)
    pool.write_json(output, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--query-id", required=True)
    asyncio.run(run(parser.parse_args()))
