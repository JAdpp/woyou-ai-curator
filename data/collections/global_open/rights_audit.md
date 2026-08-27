# 权利、来源与图像审计 — global_open 20260826-17246

生成时间：2026-08-26T20:02:13Z
对象总数：17246
图片验证：full_remote_check

## 纳入门槛

- 仅对通过各机构开放访问门槛的记录进行字段级许可标注；
- 有稳定机构对象页、远程馆藏图和至少两条可定位证据片段；
- 图片只作为机构原图热链，导入器不下载、不生成、不改画文物；
- 版权开放不等于文化敏感性问题已解决，墓葬、神圣物件等仍需在策展层显式处理。

## 来源分布

- Art Institute of Chicago: 1000
- Cleveland Museum of Art: 15200
- The Metropolitan Museum of Art: 1046

## 字段级许可分布

- 图像（`imageLicense`）：CC0 1.0=17246
- 对象元数据（`metadataLicense`）：CC0 1.0=17246
- 对象描述文本（`curatorialTextLicense`）：CC BY 4.0=219, CC0 1.0=5551, not_present=11476
- 证据片段许可（`evidence[].license`）：CC BY 4.0=219, CC0 1.0=61401
- 证据来源类型（`evidence[].sourceKind`）：institution_curatorial_text=9050, institution_metadata=35324, institution_provenance=17246

## 来源级权利边界

- Cleveland Museum of Art：仅纳入 API `share_license_status=CC0` 且 `has_image=1` 的对象；选中的 API 元数据、描述与开放图像均记为 CC0 1.0。
- The Metropolitan Museum of Art：仅复用 `isPublicDomain=true` 且已通过图片校验的冻结对象；开放图像与基本藏品数据分别记为 CC0 1.0，不伪造其 API 未提供的策展说明。
- Art Institute of Chicago：只有 `is_public_domain=true` 图像可进入库；仅 API `description` 字段记为 CC BY 4.0，`short_description` 与其余 API 数据、开放图像均记为 CC0 1.0。
- 旧 v2/v3 记录只在 `institutionId` 属于上述机构且原 `rights` 含 CC0/Public Domain 明确信号时补齐字段；未知机构不会自动继承 CC0。

## 解释边界

- CulturePack 是覆盖分面，不是单一、排他的文化身份结论；
- 跨文化并置不自动证明历史影响或传播关系；关系主张必须回到机构证据；
- 本批为面向 Demo 的分层 serving corpus，不是世界馆藏总体的统计代表样本。
