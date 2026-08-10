"""Multi-institution open-access importer.

Builds the merged, frozen collection the demo generates from:

    data/collections/chinese_art_open/
      ├─ manifest.json            collection identity, licence, domain manifest
      ├─ objects.json             normalized objects (the runtime corpus)
      ├─ question_cards.json      pre-validated starter questions per domain
      ├─ regression_questions.json offline answerability regression set
      ├─ rights_audit.md          per-institution licence and attribution audit
      └─ raw/snapshots/<ts>/      verbatim API responses + SHA-256

Sources: Cleveland Museum of Art, Art Institute of Chicago, The Metropolitan
Museum of Art. All objects are CC0 / public domain with a resolving image and a
stable institution object page.

Usage (PowerShell):
    .\\.venv\\Scripts\\python.exe .\\scripts\\import_collections.py
    .\\.venv\\Scripts\\python.exe .\\scripts\\import_collections.py --cma 120 --aic 200 --met 200
    .\\.venv\\Scripts\\python.exe .\\scripts\\import_collections.py --skip-image-check
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import domains as domain_rules  # noqa: E402
from sources import aic, cma, met  # noqa: E402
from sources.base import (  # noqa: E402
    COLLECTIONS_ROOT,
    sha256_bytes,
    IMPORTER_VERSION,
    SCHEMA_VERSION,
    HttpClient,
    Snapshot,
    SourceObject,
    dedupe_objects,
    utc_now,
    verify_images,
    write_json,
)

COLLECTION_ID = "chinese_art_open"
COLLECTION_DIR = COLLECTIONS_ROOT / COLLECTION_ID
RAW_ROOT = COLLECTION_DIR / "raw"

ADAPTERS = {
    "cma": (cma, 0.4),
    # AIC asks anonymous clients to stay at or below 60 requests/minute.
    "aic": (aic, 1.0),
    # The Met serves single-object reads cheaply and this import needs many of
    # them; still well under any published ceiling.
    "met": (met, 0.15),
}


def log(message: str) -> None:
    print(message, flush=True)


# --------------------------------------------------------------------------
# Question cards and regression set
# --------------------------------------------------------------------------

DOMAIN_QUESTIONS: dict[str, list[str]] = {
    "landscape-brush": [
        "中国山水作品为什么不能只被理解为对自然风景的写生记录？",
        "画家反复仿效古代大师时，如何同时表达传承、选择与创新？",
    ],
    "calligraphy-inscription": [
        "题跋、诗句与书法怎样改变一件绘画作品的观看顺序和意义？",
        "器物上的铭文和纸上的书法，承担的功能有什么不同？",
    ],
    "ritual-bronze": [
        "商周青铜礼器的造型与纹饰，如何体现当时的等级与祭祀秩序？",
        "一件礼器从祭祀现场进入博物馆展柜，意义发生了哪些变化？",
    ],
    "ceramics-glaze": [
        "釉色与烧造技术的变化，如何改变了陶瓷器的使用场景与观看方式？",
        "青花瓷的钴料与纹样，能说明哪些跨地区的物质流动？",
    ],
    "buddhist-devotion": [
        "佛教造像的姿态、手印与材质，如何服务于具体的礼拜与观想方式？",
        "同一尊菩萨在不同时代的形象差异，反映了哪些信仰实践的改变？",
    ],
    "funerary-afterlife": [
        "墓葬中的俑像与明器，透露出古人对身后世界的哪些想象？",
        "随葬品的选择与摆放，如何反映死者生前的身份与家族意图？",
    ],
    "court-daily-life": [
        "服饰、家具与文房用具，如何在日常生活中标记身份与礼制？",
        "宫廷器用与民间用品在材料与工艺上的差别说明了什么？",
    ],
    "trade-exchange": [
        "沿丝路与海路流动的器物，如何把外来母题转化为本地表达？",
        "外销瓷的器形与纹样，为谁而做，又反映了谁的趣味？",
    ],
}

# Deliberately out of scope: these must come back unsupported so the gate is
# demonstrably doing something, not just always saying yes.
UNSUPPORTED_QUESTIONS = [
    "这些藏品今天在拍卖市场上大约值多少钱？",
    "参观这个展览能降低我的焦虑水平吗？",
    "请生成一件仿古青铜器的图片作为展品。",
    "克利夫兰、大英博物馆和故宫的中国收藏，哪一家整体质量排名最高？",
    "青铜礼器的衰落是否完全由王朝更替直接导致？",
]

PARTIAL_QUESTIONS = [
    "所有中国陶瓷的釉色配方分别是什么？",
    "能否精确复原这件器物出土时的全部墓葬布局？",
    "这些藏品能说明当时工匠的真实劳动条件吗？",
    "青花瓷的兴起完全由海外贸易需求唯一决定吗？",
    "请列出唐代全部政治和经济原因。",
]


def build_question_cards(
    objects: list[SourceObject], distribution: dict[str, int], version: str
) -> dict[str, Any]:
    by_domain: dict[str, list[SourceObject]] = {}
    for obj in objects:
        for domain_id in obj.themes:
            by_domain.setdefault(domain_id, []).append(obj)

    cards: list[dict[str, Any]] = []
    for domain in domain_rules.DOMAINS:
        pool = by_domain.get(domain.id, [])
        if len(pool) < domain_rules.MIN_OBJECTS_PER_DOMAIN:
            continue
        # Starters must be able to carry a core-evidence role.
        full_depth = [obj for obj in pool if obj.evidence_depth == "full"]
        starters = [obj.id for obj in (full_depth or pool)[:5]]
        for index, question in enumerate(DOMAIN_QUESTIONS.get(domain.id, [])):
            cards.append(
                {
                    "id": f"{domain.id}-{index + 1}",
                    "question": question,
                    "evidenceDomainId": domain.id,
                    "coverageStatus": "supported" if len(full_depth) >= 5 else "partially_supported",
                    "reviewStatus": "source_exact_match",
                    "starterObjectIds": starters if len(full_depth) >= 5 else [],
                    "coverageLimits": [
                        f"本域当前有 {len(pool)} 件可用藏品，其中 {len(full_depth)} 件带机构撰写的说明文本。",
                        "馆方英文原文保持原样；中文策展关系属于系统推断。",
                    ],
                }
            )
    return {
        "schemaVersion": SCHEMA_VERSION,
        "collectionVersion": version,
        "generatedAt": utc_now(),
        "note": "问题卡按证据覆盖域生成；覆盖域是语料路由标签，不是访客端固定主题。",
        "cards": cards,
    }


def build_regression_questions(cards: dict[str, Any], version: str) -> dict[str, Any]:
    questions: list[dict[str, Any]] = []
    for card in cards["cards"]:
        questions.append(
            {
                "id": f"reg-{card['id']}",
                "question": card["question"],
                "expectedStatus": card["coverageStatus"],
                "evidenceDomainId": card["evidenceDomainId"],
            }
        )
    for index, question in enumerate(PARTIAL_QUESTIONS):
        questions.append(
            {
                "id": f"reg-partial-{index + 1}",
                "question": question,
                "expectedStatus": "partially_supported",
                "rationale": "问题要求总体化、排他性或完整因果结论，超出单件对象证据能支持的范围。",
            }
        )
    for index, question in enumerate(UNSUPPORTED_QUESTIONS):
        questions.append(
            {
                "id": f"reg-unsupported-{index + 1}",
                "question": question,
                "expectedStatus": "unsupported",
                "rationale": "问题需要馆藏之外的数据、效果证据或产品明确禁止的能力。",
            }
        )
    return {
        "schemaVersion": SCHEMA_VERSION,
        "collectionVersion": version,
        "generatedAt": utc_now(),
        "questionCount": len(questions),
        "questions": questions,
    }


def rights_audit(objects: list[SourceObject], version: str) -> str:
    by_institution: dict[str, list[SourceObject]] = {}
    for obj in objects:
        by_institution.setdefault(obj.institution, []).append(obj)

    lines = [
        f"# 权利与归属审计 — {COLLECTION_ID} {version}",
        "",
        f"生成时间：{utc_now()}",
        f"对象总数：{len(objects)}",
        "",
        "## 纳入门槛",
        "",
        "- 机构公开声明为 CC0 / Public Domain",
        "- 有可解析的机构托管图片（导入时 HEAD 校验通过）",
        "- 有稳定的机构对象页 URL",
        "- 至少 2 条可定位证据片段",
        "",
        "## 分机构明细",
        "",
        "| 机构 | 对象数 | 权利声明 | 机构撰写说明 | 机构撰写替代文本 |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for institution, group in sorted(by_institution.items()):
        full = sum(1 for obj in group if obj.evidence_depth == "full")
        authored_alt = sum(1 for obj in group if obj.alt_text_source == "institution_authored")
        lines.append(
            f"| {institution} | {len(group)} | {group[0].rights} | {full} | {authored_alt} |"
        )

    lines += [
        "",
        "## 已知边界",
        "",
        "- The Met 不提供策展说明字段，其证据片段均为著录信息重述而非机构叙述文本；",
        "  因此 Met 对象的 `evidenceDepth` 恒为 `thin`，不承担核心证据策展角色。",
        "- 替代文本标记为 `metadata_fallback` 的对象由元数据合成，未经视觉复核。",
        "- 文物主体只使用机构原图。本项目不生成仿古文物，也不改画机构图像。",
        "- 图片默认热链机构 CDN；本地只缓存实际展出过的对象（见 01b §1.2）。",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def emit(objects: list[SourceObject], version: str, snapshot_ids: dict[str, str]) -> None:
    distribution = Counter(domain_id for obj in objects for domain_id in obj.themes)
    institution_distribution = Counter(obj.institution for obj in objects)
    depth_distribution = Counter(obj.evidence_depth for obj in objects)

    write_json(
        COLLECTION_DIR / "objects.json",
        [obj.to_json() for obj in objects],
    )

    cards = build_question_cards(objects, dict(distribution), version)
    write_json(COLLECTION_DIR / "question_cards.json", cards)
    write_json(
        COLLECTION_DIR / "regression_questions.json",
        build_regression_questions(cards, version),
    )

    write_json(
        COLLECTION_DIR / "manifest.json",
        {
            "schemaVersion": SCHEMA_VERSION,
            "importerVersion": IMPORTER_VERSION,
            "id": COLLECTION_ID,
            "name": "东亚艺术开放馆藏（三馆合并）",
            "institution": "Cleveland Museum of Art · Art Institute of Chicago · The Metropolitan Museum of Art",
            "version": version,
            "generatedAt": utc_now(),
            "objectCount": len(objects),
            "license": {
                "images": "CC0 1.0 / Public Domain, institution-hosted originals only",
                "metadata": "CC0 1.0",
                "licenseUrl": "https://creativecommons.org/publicdomain/zero/1.0/",
            },
            "sourceUrl": "https://www.clevelandart.org/open-access",
            "institutionDistribution": dict(institution_distribution),
            "evidenceDepthDistribution": dict(depth_distribution),
            "evidenceDomains": domain_rules.domain_manifest(dict(distribution)),
            "snapshotIds": snapshot_ids,
            "boundaries": [
                "覆盖域为语料路由标签，不是访客端固定策展主题。",
                "evidenceDepth=thin 的对象不得承担核心证据策展角色。",
                "机构原文直引保持原样，不翻译后冒充原文。",
            ],
        },
    )

    (COLLECTION_DIR / "rights_audit.md").write_text(
        rights_audit(objects, version), encoding="utf-8"
    )


def rebuild_from_snapshots() -> tuple[list[SourceObject], dict[str, str]]:
    """Re-map the newest stored snapshot without touching any API.

    Raw responses are archived verbatim on every run, so mapping, domain
    routing and image validation can be re-run offline after a code fix. This
    is what makes a failed late stage cheap to recover from — the expensive
    part is the thousands of institution requests, not the mapping.
    """
    snapshots_root = RAW_ROOT / "snapshots"
    if not snapshots_root.exists():
        raise SystemExit(f"no snapshots to rebuild from under {snapshots_root}")

    collected: list[SourceObject] = []
    snapshot_ids: dict[str, str] = {}
    mappers = {"cma": cma._map, "aic": aic._map, "met": met._map}

    for snapshot_dir in sorted(snapshots_root.iterdir(), reverse=True):
        for source_id, mapper in mappers.items():
            source_dir = snapshot_dir / source_id / "objects"
            if not source_dir.exists() or source_id in snapshot_ids:
                continue
            snapshot_ids[source_id] = snapshot_dir.name
            count = 0
            for record_path in sorted(source_dir.glob("*.json")):
                raw_bytes = record_path.read_bytes()
                try:
                    record = json.loads(raw_bytes.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                relative = record_path.relative_to(RAW_ROOT.parent).as_posix()
                digest = sha256_bytes(raw_bytes)
                # CMA's mapper also needs its API URL; the others derive it.
                mapped = (
                    mapper(record, relative, digest, cma.API_ROOT)
                    if source_id == "cma"
                    else mapper(record, relative, digest)
                )
                if mapped:
                    collected.append(mapped)
                    count += 1
            log(f"[{source_id}] re-mapped {count} objects from snapshot {snapshot_dir.name}")
        if len(snapshot_ids) == len(mappers):
            break

    if not collected:
        raise SystemExit("snapshots contained no mappable records")
    return collected, snapshot_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cma", type=int, default=320, help="CMA target object count")
    parser.add_argument("--aic", type=int, default=520, help="AIC target object count")
    parser.add_argument("--met", type=int, default=520, help="Met target object count")
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Skip HEAD validation of every image URL (faster, less safe)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(ADAPTERS),
        help="Import only these institutions",
    )
    parser.add_argument(
        "--from-snapshot",
        action="store_true",
        help="Re-map the newest stored raw snapshot instead of calling the APIs",
    )
    args = parser.parse_args()

    targets = {"cma": args.cma, "aic": args.aic, "met": args.met}
    selected = args.only or sorted(ADAPTERS)

    collected: list[SourceObject] = []
    snapshot_ids: dict[str, str] = {}

    if args.from_snapshot:
        collected, snapshot_ids = rebuild_from_snapshots()
    else:
        for source_id in selected:
            module, interval = ADAPTERS[source_id]
            client = HttpClient(min_interval=interval)
            if source_id == "aic":
                client.extra_headers["AIC-User-Agent"] = (
                    "Inquiry-Curator/2.0 (academic research demo; jadppcc@gmail.com)"
                )
            snapshot = Snapshot(RAW_ROOT, source_id)
            snapshot_ids[source_id] = snapshot.id
            log(f"[{source_id}] fetching up to {targets[source_id]} objects…")
            fetched = module.fetch(client, snapshot, targets[source_id], log)
            snapshot.finalize()
            log(f"[{source_id}] mapped {len(fetched)} objects")
            collected.extend(fetched)

    log(f"merged {len(collected)} objects; deduplicating…")
    collected = dedupe_objects(collected)
    log(f"after dedupe: {len(collected)}")

    log("assigning coverage domains…")
    routed: list[SourceObject] = []
    for obj in collected:
        obj.themes = domain_rules.assign(obj)
        if obj.themes:
            routed.append(obj)
    log(f"routed into domains: {len(routed)} (dropped {len(collected) - len(routed)} unroutable)")

    if not args.skip_image_check:
        log("validating image URLs…")
        client = HttpClient(min_interval=0.05)
        before = len(routed)
        routed = verify_images(routed, client, log)
        log(f"images ok: {len(routed)} (dropped {before - len(routed)})")

    version = f"{utc_now()[:10].replace('-', '')}-{len(routed)}"
    emit(routed, version, snapshot_ids)

    distribution = Counter(domain_id for obj in routed for domain_id in obj.themes)
    log("")
    log(f"wrote {COLLECTION_DIR}")
    log(f"collection version: {version}")
    log(f"objects: {len(routed)}")
    for domain in domain_rules.DOMAINS:
        count = distribution.get(domain.id, 0)
        flag = "ok " if count >= domain_rules.MIN_OBJECTS_PER_DOMAIN else "LOW"
        log(f"  [{flag}] {domain.id:<26} {count:>5}  {domain.label}")
    depth = Counter(obj.evidence_depth for obj in routed)
    log(f"evidence depth: full={depth.get('full', 0)} thin={depth.get('thin', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
