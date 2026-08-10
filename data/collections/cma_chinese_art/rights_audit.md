# CMA Chinese Art 种子集权利与来源审计

> 数据快照：`20260806T044524Z`  
> 访问时间：`2026-08-06T04:45:24Z`  
> 数据性质：受限设计种子，不是整馆藏，也不是已完成人工内容审核的数据集。

## 结论

- 数据只来自 Cleveland Museum of Art（CMA）官方 Open Access API。
- 导入门槛固定为 `department=Chinese Art`、`cc0`、`has_image=1`，并再次逐件检查 `share_license_status == CC0` 与 `images.web.url`。
- CMA 官方说明：CC0 数据和公共领域藏品图像可免费用于商业与非商业用途；API 不需要 key 或 token。
- 本种子集没有生成、重绘或改变任何文物主体图像。展示图全部指向 CMA 官方 Open Access CDN。
- CC0 不免除可能存在的第三方商标、隐私或其他权利风险；正式公开前仍应保留逐件权利复核和撤下机制。

## 官方来源

- Open Access 政策：https://www.clevelandart.org/open-access
- API 文档：https://openaccess-api.clevelandart.org/
- 使用条款：https://www.clevelandart.org/terms-and-conditions
- CC0 1.0：https://creativecommons.org/publicdomain/zero/1.0/

## 快照与筛选

- 实时发现接口返回的 Chinese Art + CC0 + image 候选总数：2454
- `landscape-writing` 证据门槛候选：185
- `objects-ritual` 证据门槛候选（排重前）：226
- 冻结对象：60 件
- 主题分布：`landscape-writing` 30 件；`objects-ritual` 30 件
- 图片 URL 本次可访问检查通过：60/60

每件对象必须同时具有：机构详情页、CC0 web 图、机构 description、至少一条 citation、至少一条 provenance。脚本保留每个官方原始 JSON 响应、访问时间与 SHA-256；运行时对象通过 `raw_record_path` 和 `source_record_sha256` 回指原响应。

## 图像处理边界

- 默认展示 `images.web.url`；同时保留 CMA 的 print/full URL 供人工审核，不把大图自动下载进仓库。
- CMA 的 `images.annotation` 为空时，脚本只生成基于题名/年代/材质的文字回退，并标记 `metadata_fallback_pending_visual_review`；这不是人工核对的视觉描述。
- 正式发布前需要人工完成替代文本视觉审核、失效链接复查和权利状态复查。

## 证据处理边界

- `metadata`、`description`、`provenance` 片段均可定位到对象详情页及保存的 CMA API JSON 路径。
- 所有片段目前标记为 `pending_human_review`；`source_exact_match` 仅说明文本可在官方原响应中核对，不等于馆藏专家已经审核其策展用途。
- `citation` 只证明 CMA 记录列出了该书目，不能在没有取得并阅读原出版物时支持出版物中的内容性主张。
- 当前问题卡和 30 题回归集是结构化开发夹具，仍需遗产专家确认覆盖与文化适切性。

## 推荐归属标注

CC0 不强制署名，但 Demo 建议始终展示：`Creator. Title, Date. The Cleveland Museum of Art. Accession Number. CMA object URL.`，同时提供机构详情页与 CC0 链接。
