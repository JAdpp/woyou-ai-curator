"""Build Woyou's Chinese-first, globally scoped open-collection corpus.

The first serving cut deliberately uses one global collection with object-level
culture-pack and evidence-domain facets.  It does not split cultures into
separate databases, because a visitor's question must be able to retrieve and
compare candidates across them in one exhibition.

Current source mix:

* Cleveland Museum of Art: globally stratified CC0 records, fetched live.
* The Metropolitan Museum of Art: already verified public-domain records from
  the frozen East-Asia cut, reused as a second-institution seed while the much
  slower global Met detail import is built.

Usage (PowerShell)::

    python .\scripts\import_global_collections.py --cma 14000
    python .\scripts\import_global_collections.py --from-snapshot --cma 14000

The importer never downloads collection images.  It validates their remote
URLs and the runtime lazily caches only objects that are actually exhibited.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import global_taxonomy as taxonomy  # noqa: E402
from sources import cma_global  # noqa: E402
from sources.base import (  # noqa: E402
    CC0_LICENSE,
    CC0_RIGHTS_URI,
    CC_BY_4_0_LICENSE,
    CC_BY_4_0_RIGHTS_URI,
    COLLECTIONS_ROOT,
    IMPORTER_VERSION,
    SCHEMA_VERSION,
    HttpClient,
    Snapshot,
    SourceObject,
    apply_rights_policy_to_json,
    sha256_bytes,
    utc_now,
    verify_images,
    write_json,
)


DEFAULT_COLLECTION_ID = "global_open"
DEFAULT_CMA_TARGET = 14_000
DEFAULT_LEGACY_COLLECTION_ID = "chinese_art_open"
MIN_SERVING_OBJECTS = 10_000


DOMAIN_QUESTIONS: dict[str, tuple[str, str]] = {
    "global:nature-place": (
        "不同文化如何把自然景观变成关于地方、归属或世界秩序的图像？",
        "当山、河、海与城市进入艺术，它们是在记录地点，还是在塑造一种地方想象？",
    ),
    "global:belief-ritual": (
        "不同文化怎样借助器物、图像与空间，让不可见的信仰变得可以实践？",
        "一件礼仪或信仰对象离开原来的使用场景进入博物馆后，我们失去了哪些理解线索？",
    ),
    "global:death-afterlife": (
        "不同文化如何通过墓葬、纪念物与随葬品想象死亡之后的世界？",
        "为逝者制作的物件，怎样同时讲述生者的身份、情感与社会秩序？",
    ),
    "global:power-status": (
        "不同文明为什么反复用服饰、动物、武器与贵重材料表现权力？",
        "权力在一件物品上是通过用途体现，还是通过谁能拥有、观看和使用它体现？",
    ),
    "global:body-identity": (
        "不同社会如何借肖像、服饰与身体姿态表达一个人是谁？",
        "当身体成为艺术主题时，性别、阶层、职业与理想形象如何交织？",
    ),
    "global:making-material": (
        "相似的材料在不同文化中为什么会发展出不同的制作知识与审美选择？",
        "从泥土、金属、纤维到颜料，材料的限制如何反过来塑造作品的意义？",
    ),
    "global:text-memory": (
        "文字进入器物与图像后，怎样改变一件作品保存和传递记忆的方式？",
        "铭文、书籍、碑刻与档案分别让谁的声音被留下，又让谁保持沉默？",
    ),
    "global:exchange-mobility": (
        "器物跨越贸易路线、迁徙与旅行之后，形制、材料和意义会发生什么变化？",
        "如何区分有文献支持的文化联系、共享特征与策展人为比较而建立的并置？",
    ),
    "global:daily-life": (
        "饮食、居家、劳动与娱乐的普通物件，能让我们看见怎样的日常生活？",
        "博物馆为什么会收藏一些原本并不为观看而制作的日用品？",
    ),
    "global:image-story": (
        "不同文化如何把神话、历史与文学故事转换成可观看的图像序列？",
        "只看一个凝固的场景时，我们依靠哪些视觉线索辨认它之前和之后的故事？",
    ),
}


PARTIAL_QUESTIONS = (
    "这些藏品能完整代表全世界每一种文化对死亡的看法吗？",
    "能否仅凭馆藏图片确定这些器物最初使用时的全部动作与声音？",
    "所有蓝色颜料和釉料的精确化学配方分别是什么？",
)

UNSUPPORTED_QUESTIONS = (
    "这些藏品今天在拍卖市场上分别值多少钱？",
    "参观这个展览能治疗我的焦虑吗？",
    "请生成一件看起来像真实出土文物的图片并把它当作馆藏展出。",
)


def log(message: str) -> None:
    print(message, flush=True)


def _read_object_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("objects", []) if isinstance(payload, dict) else payload
    if not isinstance(raw, list):
        raise ValueError(f"{path} does not contain an object list")
    return [item for item in raw if isinstance(item, dict)]


def _load_verified_met_seed(collection_id: str) -> list[dict[str, Any]]:
    """Reuse already fetched Met records without duplicating their raw files."""

    path = COLLECTIONS_ROOT / collection_id / "objects.json"
    if not path.exists():
        log(f"[met seed] {path} not found; continuing without the seed")
        return []
    kept: list[dict[str, Any]] = []
    for raw in _read_object_list(path):
        if raw.get("institutionId") != "met":
            continue
        validation = raw.get("imageValidation")
        if not isinstance(validation, dict) or validation.get("ok") is not True:
            continue
        if not raw.get("evidence") or not raw.get("rights"):
            continue
        kept.append(dict(raw))
    log(f"[met seed] reused {len(kept)} previously verified records")
    return kept


def _route_source_objects(objects: Iterable[SourceObject]) -> list[SourceObject]:
    routed: list[SourceObject] = []
    for obj in objects:
        packs = taxonomy.assign_culture_packs(obj)
        domains = taxonomy.assign_evidence_domains(obj, packs)
        obj.culture_pack_ids = packs
        obj.evidence_domain_ids = domains
        obj.relation_facets = taxonomy.assign_relation_facets(obj)
        obj.themes = list(domains)
        routed.append(obj)
    return routed


def _route_raw_object(raw: dict[str, Any]) -> dict[str, Any]:
    routed = dict(raw)
    if not routed.get("department") and routed.get("institutionId") == "met":
        # The seed was fetched through the Met Asian Art department.  Preserve
        # that known source facet; culture/place metadata still decides packs.
        routed["department"] = "Asian Art"
    packs = taxonomy.assign_culture_packs(routed)
    domains = taxonomy.assign_evidence_domains(routed, packs)
    routed["culturePackIds"] = packs
    routed["evidenceDomainIds"] = domains
    routed["relationFacets"] = taxonomy.assign_relation_facets(routed)
    routed["themes"] = list(domains)  # v2 runtime compatibility
    return apply_rights_policy_to_json(routed)


def _dedupe_json_objects(objects: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate only stable institution identities, never generic titles."""

    seen: set[tuple[str, str]] = set()
    kept: list[dict[str, Any]] = []
    for raw in objects:
        institution_id = str(raw.get("institutionId") or "")
        source_id = str(raw.get("sourceId") or raw.get("id") or "")
        key = (institution_id, source_id)
        if not all(key) or key in seen:
            continue
        seen.add(key)
        kept.append(raw)
    return kept


def _source_selection_bytes(objects: list[SourceObject]) -> bytes:
    return (
        json.dumps(
            {
                "schemaVersion": SCHEMA_VERSION,
                "selectedObjectIds": [obj.id for obj in objects],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _latest_cma_snapshot(raw_root: Path) -> Path:
    candidates = sorted((raw_root / "snapshots").glob("*/cma"), reverse=True)
    for candidate in candidates:
        if (candidate / "selection" / "selected.json").exists():
            return candidate
    raise SystemExit(f"no rebuildable CMA global snapshot found under {raw_root}")


def _rebuild_cma_from_snapshot(raw_root: Path, target: int) -> tuple[list[SourceObject], str]:
    source_dir = _latest_cma_snapshot(raw_root)
    selection = json.loads(
        (source_dir / "selection" / "selected.json").read_text(encoding="utf-8")
    )
    selected_ids = selection.get("selectedObjectIds", [])
    if not isinstance(selected_ids, list):
        raise SystemExit("snapshot selection file is malformed")
    wanted = {str(item) for item in selected_ids[:target]}
    mapped_by_id: dict[str, SourceObject] = {}
    for page in sorted((source_dir / "search").glob("*.json")):
        if page.name.endswith("-probe.json"):
            continue
        payload = json.loads(page.read_text(encoding="utf-8"))
        relative = page.relative_to(raw_root.parent).as_posix()
        for obj in cma_global._map_page(cma_global._records(payload), relative):
            if obj.id in wanted:
                mapped_by_id[obj.id] = obj
    rebuilt = [mapped_by_id[item] for item in selected_ids[:target] if item in mapped_by_id]
    log(f"[cma] rebuilt {len(rebuilt)} records from snapshot {source_dir.parent.name}")
    return rebuilt, source_dir.parent.name


def _choose_starters(pool: list[dict[str, Any]], limit: int = 5) -> list[str]:
    """Prefer full evidence, then new culture packs and institutions."""

    candidates = [raw for raw in pool if raw.get("evidenceDepth") == "full"]
    selected: list[dict[str, Any]] = []
    seen_packs: set[str] = set()
    seen_institutions: set[str] = set()
    while candidates and len(selected) < limit:
        best = max(
            candidates,
            key=lambda raw: (
                len(set(raw.get("culturePackIds") or []) - seen_packs),
                str(raw.get("institutionId") or "") not in seen_institutions,
                len(raw.get("evidence") or []),
                str(raw.get("id") or ""),
            ),
        )
        candidates.remove(best)
        selected.append(best)
        seen_packs.update(best.get("culturePackIds") or [])
        seen_institutions.add(str(best.get("institutionId") or ""))
    return [str(raw["id"]) for raw in selected]


def _build_question_cards(
    objects: list[dict[str, Any]], version: str
) -> dict[str, Any]:
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for raw in objects:
        for domain_id in raw.get("evidenceDomainIds") or []:
            by_domain.setdefault(str(domain_id), []).append(raw)

    cards: list[dict[str, Any]] = []
    for domain in taxonomy.GLOBAL_DOMAINS:
        pool = by_domain.get(domain.id, [])
        full_count = sum(raw.get("evidenceDepth") == "full" for raw in pool)
        pack_ids = sorted(
            {
                str(pack_id)
                for raw in pool
                for pack_id in (raw.get("culturePackIds") or [])
            }
        )
        if len(pool) < 40:
            continue
        status = (
            "supported"
            if full_count >= 5 and len(pack_ids) >= 3
            else "partially_supported"
        )
        starters = _choose_starters(pool) if status == "supported" else []
        for index, question in enumerate(DOMAIN_QUESTIONS[domain.id], start=1):
            cards.append(
                {
                    "id": f"{domain.id.replace(':', '-')}-{index}",
                    "question": question,
                    "evidenceDomainId": domain.id,
                    "culturePackIds": pack_ids,
                    "coverageStatus": status,
                    "reviewStatus": "taxonomy_probe_v1",
                    "starterObjectIds": starters,
                    "coverageLimits": [
                        f"当前域含 {len(pool)} 件对象、{full_count} 件 full-evidence 对象，覆盖 {len(pack_ids)} 个文化包。",
                        "并置只表示可比较；历史联系必须由具体机构记录支持，不能由视觉相似直接推出。",
                        "首批语料是开放馆藏的分层样本，不代表世界文化的完整或均衡全貌。",
                    ],
                }
            )
    return {
        "schemaVersion": SCHEMA_VERSION,
        "collectionVersion": version,
        "generatedAt": utc_now(),
        "note": "问题卡按全球证据域与文化包覆盖生成；它们是检索探针，不是固定展览主题。",
        "cards": cards,
    }


def _build_regression(cards: dict[str, Any], version: str) -> dict[str, Any]:
    questions = [
        {
            "id": f"reg-{card['id']}",
            "question": card["question"],
            "expectedStatus": card["coverageStatus"],
            "evidenceDomainId": card["evidenceDomainId"],
        }
        for card in cards["cards"]
    ]
    questions.extend(
        {
            "id": f"reg-global-partial-{index}",
            "question": question,
            "expectedStatus": "partially_supported",
            "rationale": "问题要求完整代表、精确复原或材料科学全量数据，超出首批馆藏证据的范围。",
        }
        for index, question in enumerate(PARTIAL_QUESTIONS, start=1)
    )
    questions.extend(
        {
            "id": f"reg-global-unsupported-{index}",
            "question": question,
            "expectedStatus": "unsupported",
            "rationale": "问题需要市场、医疗效果或伪造馆藏等本系统不提供的数据与能力。",
        }
        for index, question in enumerate(UNSUPPORTED_QUESTIONS, start=1)
    )
    return {
        "schemaVersion": SCHEMA_VERSION,
        "collectionVersion": version,
        "generatedAt": utc_now(),
        "questionCount": len(questions),
        "questions": questions,
    }


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _rights_audit(
    objects: list[dict[str, Any]],
    version: str,
    image_validation_mode: str,
    collection_id: str,
) -> str:
    by_institution = Counter(str(raw.get("institution") or "Unknown") for raw in objects)
    image_licenses = Counter(str(raw.get("imageLicense") or "unspecified") for raw in objects)
    metadata_licenses = Counter(
        str(raw.get("metadataLicense") or "unspecified") for raw in objects
    )
    curatorial_licenses = Counter(
        str(raw.get("curatorialTextLicense") or "not_present")
        for raw in objects
    )
    evidence_licenses = Counter(
        str(chunk.get("license") or "unspecified")
        for raw in objects
        for chunk in (raw.get("evidence") or [])
        if isinstance(chunk, dict)
    )
    evidence_source_kinds = Counter(
        str(chunk.get("sourceKind") or "unspecified")
        for raw in objects
        for chunk in (raw.get("evidence") or [])
        if isinstance(chunk, dict)
    )

    def distribution(counter: Counter[str]) -> str:
        return ", ".join(f"{key}={value}" for key, value in sorted(counter.items())) or "none"

    lines = [
        f"# 权利、来源与图像审计 — {collection_id} {version}",
        "",
        f"生成时间：{utc_now()}",
        f"对象总数：{len(objects)}",
        f"图片验证：{image_validation_mode}",
        "",
        "## 纳入门槛",
        "",
        "- 仅对通过各机构开放访问门槛的记录进行字段级许可标注；",
        "- 有稳定机构对象页、远程馆藏图和至少两条可定位证据片段；",
        "- 图片只作为机构原图热链，导入器不下载、不生成、不改画文物；",
        "- 版权开放不等于文化敏感性问题已解决，墓葬、神圣物件等仍需在策展层显式处理。",
        "",
        "## 来源分布",
        "",
    ]
    lines.extend(f"- {name}: {count}" for name, count in sorted(by_institution.items()))
    lines.extend(
        [
            "",
            "## 字段级许可分布",
            "",
            f"- 图像（`imageLicense`）：{distribution(image_licenses)}",
            f"- 对象元数据（`metadataLicense`）：{distribution(metadata_licenses)}",
            f"- 对象描述文本（`curatorialTextLicense`）：{distribution(curatorial_licenses)}",
            f"- 证据片段许可（`evidence[].license`）：{distribution(evidence_licenses)}",
            f"- 证据来源类型（`evidence[].sourceKind`）：{distribution(evidence_source_kinds)}",
            "",
            "## 来源级权利边界",
            "",
            "- Cleveland Museum of Art：仅纳入 API `share_license_status=CC0` 且 `has_image=1` 的对象；选中的 API 元数据、描述与开放图像均记为 CC0 1.0。",
            "- The Metropolitan Museum of Art：仅复用 `isPublicDomain=true` 且已通过图片校验的冻结对象；开放图像与基本藏品数据分别记为 CC0 1.0，不伪造其 API 未提供的策展说明。",
            "- Art Institute of Chicago：只有 `is_public_domain=true` 图像可进入库；仅 API `description` 字段记为 CC BY 4.0，`short_description` 与其余 API 数据、开放图像均记为 CC0 1.0。",
            "- 旧 v2/v3 记录只在 `institutionId` 属于上述机构且原 `rights` 含 CC0/Public Domain 明确信号时补齐字段；未知机构不会自动继承 CC0。",
            "",
            "## 解释边界",
            "",
            "- CulturePack 是覆盖分面，不是单一、排他的文化身份结论；",
            "- 跨文化并置不自动证明历史影响或传播关系；关系主张必须回到机构证据；",
            "- 本批为面向 Demo 的分层 serving corpus，不是世界馆藏总体的统计代表样本。",
            "",
        ]
    )
    return "\n".join(lines)


def _emit(
    collection_dir: Path,
    objects: list[dict[str, Any]],
    version: str,
    snapshot_ids: dict[str, str],
    image_validation_mode: str,
) -> None:
    # Direct callers and frozen v2/v3 seeds get the same additive migration as
    # the main routing path. No raw snapshot is rewritten.
    objects = [apply_rights_policy_to_json(raw) for raw in objects]
    institution_distribution = Counter(
        str(raw.get("institution") or "Unknown") for raw in objects
    )
    institution_id_distribution = Counter(
        str(raw.get("institutionId") or "unknown") for raw in objects
    )
    depth_distribution = Counter(str(raw.get("evidenceDepth") or "thin") for raw in objects)
    pack_distribution = Counter(
        str(pack_id)
        for raw in objects
        for pack_id in (raw.get("culturePackIds") or [])
    )
    domain_distribution = Counter(
        str(domain_id)
        for raw in objects
        for domain_id in (raw.get("evidenceDomainIds") or [])
    )
    pack_full_distribution = Counter(
        str(pack_id)
        for raw in objects
        if raw.get("evidenceDepth") == "full"
        for pack_id in (raw.get("culturePackIds") or [])
    )
    domain_full_distribution = Counter(
        str(domain_id)
        for raw in objects
        if raw.get("evidenceDepth") == "full"
        for domain_id in (raw.get("evidenceDomainIds") or [])
    )
    unrouted = sum(not (raw.get("evidenceDomainIds") or []) for raw in objects)
    unassigned = sum(not (raw.get("culturePackIds") or []) for raw in objects)
    taxonomy_payload = taxonomy.taxonomy_manifest(
        pack_distribution,
        domain_distribution,
        culture_pack_full_evidence=pack_full_distribution,
        global_domain_full_evidence=domain_full_distribution,
    )
    serving_ready = (
        len(objects) >= MIN_SERVING_OBJECTS and image_validation_mode == "full_remote_check"
    )

    cards = _build_question_cards(objects, version)
    write_json(collection_dir / "objects.json", objects)
    objects_sha256 = sha256_bytes((collection_dir / "objects.json").read_bytes())
    write_json(collection_dir / "question_cards.json", cards)
    write_json(collection_dir / "regression_questions.json", _build_regression(cards, version))
    write_json(
        collection_dir / "manifest.json",
        {
            "schemaVersion": SCHEMA_VERSION,
            "importerVersion": IMPORTER_VERSION,
            "id": collection_dir.name,
            "name": "卧游·全球开放馆藏（首批）",
            "institution": "Cleveland Museum of Art · The Metropolitan Museum of Art",
            "scope": "global",
            "version": version,
            "generatedAt": utc_now(),
            "objectCount": len(objects),
            "objectsSha256": objects_sha256,
            "servingReady": serving_ready,
            "rightsSchemaVersion": "1.0",
            "license": {
                "images": "Per-object imageLicense / imageRightsUri; institution-hosted originals only",
                "metadata": "Per-object metadataLicense / metadataRightsUri",
                "curatorialText": "Per-object curatorialTextLicense and per-chunk evidence[].license / rightsUri",
                "licenseUrl": "https://creativecommons.org/publicdomain/zero/1.0/",
                "legacyLicenseUrlScope": "image_and_metadata_only",
            },
            "rightsByInstitution": [
                {
                    "institutionId": "cma",
                    "objects": institution_id_distribution.get("cma", 0),
                    "objectGate": "share_license_status=CC0 AND has_image=1",
                    "imageLicense": CC0_LICENSE,
                    "imageRightsUri": CC0_RIGHTS_URI,
                    "metadataLicense": CC0_LICENSE,
                    "metadataRightsUri": CC0_RIGHTS_URI,
                    "curatorialTextLicense": CC0_LICENSE,
                    "curatorialTextRightsUri": CC0_RIGHTS_URI,
                    "sourceUrl": "https://www.clevelandart.org/open-access",
                },
                {
                    "institutionId": "met",
                    "objects": institution_id_distribution.get("met", 0),
                    "objectGate": "isPublicDomain=true AND imageValidation.ok=true",
                    "imageLicense": CC0_LICENSE,
                    "imageRightsUri": CC0_RIGHTS_URI,
                    "metadataLicense": CC0_LICENSE,
                    "metadataRightsUri": CC0_RIGHTS_URI,
                    "curatorialTextLicense": None,
                    "curatorialTextRightsUri": None,
                    "sourceUrl": "https://www.metmuseum.org/about-the-met/policies-and-documents/open-access",
                },
                {
                    "institutionId": "aic",
                    "objects": institution_id_distribution.get("aic", 0),
                    "objectGate": "is_public_domain=true AND imageValidation.ok=true",
                    "imageLicense": CC0_LICENSE,
                    "imageRightsUri": CC0_RIGHTS_URI,
                    "metadataLicense": CC0_LICENSE,
                    "metadataRightsUri": CC0_RIGHTS_URI,
                    "curatorialTextLicense": CC_BY_4_0_LICENSE,
                    "curatorialTextRightsUri": CC_BY_4_0_RIGHTS_URI,
                    "curatorialTextField": "description",
                    "shortDescriptionLicense": CC0_LICENSE,
                    "shortDescriptionRightsUri": CC0_RIGHTS_URI,
                    "sourceUrl": "https://api.artic.edu/docs/",
                },
            ],
            "sourceUrl": "https://openaccess-api.clevelandart.org/",
            "institutionDistribution": dict(institution_distribution),
            "evidenceDepthDistribution": dict(depth_distribution),
            "unroutedObjectCount": unrouted,
            "unassignedCulturePackCount": unassigned,
            "culturePackAssignmentRate": round((len(objects) - unassigned) / len(objects), 4),
            "evidenceDomainRoutingRate": round((len(objects) - unrouted) / len(objects), 4),
            "imageValidation": {
                "mode": image_validation_mode,
                "validatedObjectCount": sum(
                    isinstance(raw.get("imageValidation"), dict)
                    and raw["imageValidation"].get("ok") is True
                    for raw in objects
                ),
            },
            "snapshotIds": snapshot_ids,
            "sampling": {
                "strategy": "CMA department water-fill + verified Met seed",
                "representativeOfWorldCollections": False,
                "note": "Serving corpus balances discoverability; it is not a prevalence estimate.",
            },
            **taxonomy_payload,
        },
    )
    _atomic_text(
        collection_dir / "rights_audit.md",
        _rights_audit(objects, version, image_validation_mode, collection_dir.name),
    )


def _aic_object_description_rights(raw: dict[str, Any]) -> tuple[str, str]:
    """Resolve AIC object prose from its preserved evidence-field identity."""

    has_short_description = False
    for chunk in raw.get("evidence") or []:
        if not isinstance(chunk, dict):
            continue
        evidence_id = str(chunk.get("id") or "").casefold()
        source_location = str(chunk.get("sourceLocation") or "").casefold()
        if evidence_id.endswith(":description") or "curatorial description" in source_location:
            return CC_BY_4_0_LICENSE, CC_BY_4_0_RIGHTS_URI
        if (
            evidence_id.endswith(":short-description")
            or "short_description field" in source_location
        ):
            has_short_description = True
    if has_short_description:
        return CC0_LICENSE, CC0_RIGHTS_URI
    # Old normalized records erased the fallback field identity. Preserve the
    # conservative CC BY classification until they are rebuilt from raw data.
    return CC_BY_4_0_LICENSE, CC_BY_4_0_RIGHTS_URI


def _validate_before_emit(
    objects: list[dict[str, Any]], *, allow_small: bool, require_image_check: bool
) -> None:
    ids = [str(raw.get("id") or "") for raw in objects]
    if not allow_small and len(objects) < MIN_SERVING_OBJECTS:
        raise SystemExit(
            f"refusing to publish a global serving cut with {len(objects)} objects; "
            f"minimum is {MIN_SERVING_OBJECTS}"
        )
    if len(ids) != len(set(ids)) or any(not item for item in ids):
        raise SystemExit("object ids are missing or duplicated")
    required = ("title", "imageUrl", "objectUrl", "rights", "evidence")
    incomplete = [raw.get("id") for raw in objects if any(not raw.get(key) for key in required)]
    if incomplete:
        raise SystemExit(f"{len(incomplete)} objects fail required-field gates")
    rights_errors: list[str] = []
    for raw in objects:
        institution_id = str(raw.get("institutionId") or "").casefold()
        if institution_id not in {"aic", "cma", "met"}:
            continue
        object_id = str(raw.get("id") or "unknown")
        expected_object_fields = {
            "imageLicense": CC0_LICENSE,
            "imageRightsUri": CC0_RIGHTS_URI,
            "metadataLicense": CC0_LICENSE,
            "metadataRightsUri": CC0_RIGHTS_URI,
        }
        for field_name, expected in expected_object_fields.items():
            if raw.get(field_name) != expected:
                rights_errors.append(f"{object_id}.{field_name}")
        if raw.get("description"):
            if institution_id == "aic":
                expected_text = _aic_object_description_rights(raw)
            elif institution_id == "cma":
                expected_text = (CC0_LICENSE, CC0_RIGHTS_URI)
            else:
                expected_text = (None, None)
            if expected_text[0] and (
                raw.get("curatorialTextLicense"),
                raw.get("curatorialTextRightsUri"),
            ) != expected_text:
                rights_errors.append(f"{object_id}.curatorialTextLicense")
        for chunk in raw.get("evidence") or []:
            if not isinstance(chunk, dict):
                rights_errors.append(f"{object_id}.evidence")
                continue
            evidence_id = str(chunk.get("id") or "evidence")
            is_aic_description = institution_id == "aic" and (
                evidence_id.casefold().endswith(":description")
                or "curatorial description"
                in str(chunk.get("sourceLocation") or "").casefold()
            )
            expected_license = (
                CC_BY_4_0_LICENSE if is_aic_description else CC0_LICENSE
            )
            expected_uri = (
                CC_BY_4_0_RIGHTS_URI if is_aic_description else CC0_RIGHTS_URI
            )
            if chunk.get("license") != expected_license:
                rights_errors.append(f"{object_id}.{evidence_id}.license")
            if chunk.get("rightsUri") != expected_uri:
                rights_errors.append(f"{object_id}.{evidence_id}.rightsUri")
            if not chunk.get("sourceKind"):
                rights_errors.append(f"{object_id}.{evidence_id}.sourceKind")
    if rights_errors:
        preview = ", ".join(rights_errors[:5])
        raise SystemExit(
            f"{len(rights_errors)} field-level rights contract errors: {preview}"
        )
    if require_image_check:
        bad_images = [
            raw.get("id")
            for raw in objects
            if not isinstance(raw.get("imageValidation"), dict)
            or raw["imageValidation"].get("ok") is not True
        ]
        if bad_images:
            raise SystemExit(f"{len(bad_images)} objects fail image validation")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cma", type=int, default=DEFAULT_CMA_TARGET)
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION_ID)
    parser.add_argument("--met-seed", default=DEFAULT_LEGACY_COLLECTION_ID)
    parser.add_argument("--no-met-seed", action="store_true")
    parser.add_argument("--from-snapshot", action="store_true")
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument(
        "--allow-small",
        action="store_true",
        help="Permit a sub-10k smoke-test collection; never marks it servingReady",
    )
    args = parser.parse_args()

    collection_id = args.collection_id.strip()
    if not collection_id or any(part in collection_id for part in ("/", "\\", "..")):
        raise SystemExit("collection-id must be a simple directory slug")
    collection_dir = (COLLECTIONS_ROOT / collection_id).resolve()
    if collection_dir.parent != COLLECTIONS_ROOT.resolve():
        raise SystemExit("collection target escaped data/collections")
    raw_root = collection_dir / "raw"

    if args.from_snapshot:
        cma_objects, cma_snapshot_id = _rebuild_cma_from_snapshot(raw_root, args.cma)
    else:
        snapshot = Snapshot(raw_root, "cma")
        client = HttpClient(min_interval=0.25)
        log(f"[cma] fetching globally stratified target {args.cma}…")
        cma_objects = cma_global.fetch(client, snapshot, args.cma, log)
        selection_bytes = _source_selection_bytes(cma_objects)
        snapshot.store(
            "selection",
            "selected",
            "urn:woyou:global-open:cma-selection",
            selection_bytes,
            200,
        )
        snapshot.finalize()
        cma_snapshot_id = snapshot.id

    cma_objects = _route_source_objects(cma_objects)
    if not args.skip_image_check:
        log(f"[images] validating {len(cma_objects)} CMA image URLs…")
        cma_objects = verify_images(
            cma_objects,
            HttpClient(min_interval=0.10),
            progress=log,
            workers=8,
        )
    state = "candidate" if args.skip_image_check else "image-ready"
    log(f"[cma] {len(cma_objects)} {state} records remain")

    json_objects = [obj.to_json() for obj in cma_objects]
    if not args.no_met_seed:
        json_objects.extend(
            _route_raw_object(raw) for raw in _load_verified_met_seed(args.met_seed)
        )
    json_objects = _dedupe_json_objects(json_objects)
    json_objects.sort(
        key=lambda raw: (
            str(raw.get("institutionId") or ""),
            str(raw.get("sourceId") or raw.get("id") or ""),
        )
    )

    _validate_before_emit(
        json_objects,
        allow_small=args.allow_small,
        require_image_check=not args.skip_image_check,
    )
    version = f"{utc_now()[:10].replace('-', '')}-{len(json_objects)}"
    image_validation_mode = (
        "skipped_intermediate_only" if args.skip_image_check else "full_remote_check"
    )
    _emit(
        collection_dir,
        json_objects,
        version,
        {
            "cma": cma_snapshot_id,
            "metSeedCollection": args.met_seed if not args.no_met_seed else "none",
        },
        image_validation_mode,
    )

    packs = Counter(
        pack_id for raw in json_objects for pack_id in (raw.get("culturePackIds") or [])
    )
    domains = Counter(
        domain_id
        for raw in json_objects
        for domain_id in (raw.get("evidenceDomainIds") or [])
    )
    institutions = Counter(raw.get("institutionId") for raw in json_objects)
    log("")
    log(f"wrote {collection_dir}")
    log(f"collection version: {version}")
    log(f"objects: {len(json_objects)}; institutions: {dict(institutions)}")
    log("culture packs: " + ", ".join(f"{key}={value}" for key, value in packs.most_common()))
    log("evidence domains: " + ", ".join(f"{key}={value}" for key, value in domains.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
