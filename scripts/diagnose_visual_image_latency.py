"""Bounded, model-free GET diagnostics for three URLs already in a smoke report."""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))
from app.images import AIC_USER_AGENT, USER_AGENT, ImageCache
from app.models import MuseumObject
from app.visual_evidence import prewarm_visual_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prewarm", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    candidates = report["audit"]["diagnostics"]["visualPreselection"]["results"]
    by_host = {}
    for candidate in candidates:
        by_host.setdefault(urllib.parse.urlsplit(candidate["sourceUrl"]).hostname, candidate)
    chosen = list(by_host.values())[:3]
    probe_cache = ImageCache(ROOT / "api/runtime/cache/objects")
    run_cache = ImageCache(args.report.parent / "cache/objects")

    if args.prewarm:
        objects = [MuseumObject(id=row["objectId"], title=row["objectId"], image_url=row["sourceUrl"],
                                object_url=row["sourceUrl"], rights="frozen source image") for row in chosen]
        first = asyncio.run(prewarm_visual_candidates(objects, probe_cache, timeout_seconds=12))
        started = time.perf_counter()
        second = asyncio.run(prewarm_visual_candidates(objects, probe_cache, timeout_seconds=5))
        output = {"scope": "same three frozen URLs; real bounded prewarm and warm-cache repeat; no models",
                  "first": first, "repeat": second, "repeatWallSeconds": round(time.perf_counter() - started, 3),
                  "retained1024": [obj.id for obj in objects if probe_cache.has_cached(obj.image_url, 1024)]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    def measure(candidate):
        url = candidate["sourceUrl"]
        started = time.perf_counter()
        row = {"objectId": candidate["objectId"], "sourceUrl": url,
               "existingMainCacheEdges": [edge for edge in (512, 1024, 1536) if probe_cache._path(url, edge).exists()],
               "existingRunCacheEdges": [edge for edge in (512, 1024, 1536) if run_cache._path(url, edge).exists()]}
        headers = {"User-Agent": USER_AGENT, "Accept": "image/*"}
        if urllib.parse.urlsplit(url).hostname == "www.artic.edu":
            headers["AIC-User-Agent"] = AIC_USER_AGENT
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=12) as response:
                row["headersSeconds"] = round(time.perf_counter() - started, 3)
                row["status"] = response.status
                raw = response.read(12 * 1024 * 1024 + 1)
            row["downloadSeconds"] = round(time.perf_counter() - started, 3)
            row["bytes"] = len(raw)
            decoded = time.perf_counter()
            with Image.open(io.BytesIO(raw)) as image:
                row["dimensions"] = list(image.size)
                image.load()
                image = image.convert("RGB")
                image.thumbnail((512, 512), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                image.save(buffer, format="WEBP", quality=82, method=4)
                row["webp512Bytes"] = len(buffer.getvalue())
            row["decodeResizeSeconds"] = round(time.perf_counter() - decoded, 3)
        except Exception as error:
            row["errorType"] = type(error).__name__
            row["error"] = str(error)[:250]
        row["totalSeconds"] = round(time.perf_counter() - started, 3)
        return row

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(measure, chosen))
    output = {"scope": "three frozen-candidate URLs; one museum GET each; no paid models", "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
