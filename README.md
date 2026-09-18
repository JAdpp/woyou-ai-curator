# 卧游 / Woyou

从一次简短的策展人对话出发，为每位访客生成一座有主题、有章节、有空间叙事的个性化 3D 虚拟展厅。展品与说明可追溯到博物馆公开馆藏。

**命名**：「卧游」出自宗炳（375–443）《画山水序》「澄怀观道，卧以游之」——把山水画挂在墙上躺着神游。这是中文里对「通过图像远游」最早、也最准确的说法，比任何借来的技术隐喻都更贴合虚拟展厅要做的事。

AI 策展人叫 **彦远**，取自张彦远（约 815–877）——《历代名画记》（847）的作者，第一个把画作组织成论证而不是清单的人，正是这个 agent 的工作。

它是研究型 Demo，不是已验证学习效果的产品，也不替代专业策展。

当前公开版本为 **RC11**，2026-09-07（UTC+8）已按用户授权推送并上线：[访问卧游](http://47.89.246.208:8081/)。线上实际运行 Qwen 混合检索，馆藏 17,246 件；真实生成、海报／音频资源、历史数据保全和剩余边界见 [上线验收记录](data/qa/RC11_上线验收_20260907.md)。本次发布不表示暂停的开放题总体质量评测已达标。

## 访客闭环

```
落地页 → 策展人访谈（≤6 轮） → 可见的策展流水线（8 步） → 3D 展厅 → 分享
                                                        ↘ 2D 无障碍版本
```

- **落地页**：首屏是一面由真实开放馆藏图拼成的正交网格墙（数据来自 `/api/collection/highlights`），配宗炳的原句与实时馆藏计数；随后依次给出一组跨文化策展案例、四项已实现能力、十个证据域的真实覆盖，以及三家数据来源机构的致谢与逐字段许可边界。案例把题名、年代、材料等馆方事实和系统提出的联系分开，不把视觉相似直接写成传播证据。
- **策展人访谈**：后端状态机驱动的对话，选项由馆藏实际能路由的覆盖域生成，因此不会承诺馆藏答不了的主题。访谈收集兴趣、参观倾向、熟悉程度、时长与排除内容，并提供「跳过访谈用默认设置」直达入口。
- **可见的策展流水线**：8 个步骤通过 SSE 实时推送（代理缓冲时自动降级为轮询），每步完成时回吐一句具体发现；海报生成与文本生成并行。等待区右侧提供使用两件真实馆藏图像的 3×3 滑块拼图，支持点击、方向键、重置与换图；它只消磨等待时间，不参与推荐或策展判断。
- **3D 展厅**：程序化生成一条连续展线，章节作为沿途的叙事区段；导览镜头与自由行走双模可随时互切且不丢参观进度。展签默认缩小为浮动窗口，可拖动、用方向键移动、调整大小并收起，不再固定遮住藏品。
- **2D 无障碍版本**：`?view=text`，承担全部键盘与读屏职责，同时是无 WebGL / 弱机 / 分享链接的呈现方式。

## 访客画像如何影响展览

访谈字段映射到具体的编排参数（依据见 01b §3.1）：

| 访谈问题 | 依据 | 影响 |
| --- | --- | --- |
| 来访动机 | Falk (2009) 身份动机 | 叙事语气、信息密度、空间形制与配色预设 |
| 浏览时长 | Véron & Levasseur (1983) 动线类型 | 5/10/15 分钟 → 5/8/12 件展品、2/3/4 个叙事区段、展线长度与镜头节奏 |
| 了解程度 | — | 术语与背景解释的详略 |
| 展签字数上限 | Serrell (1997) 停留时长 | 80 / 140 / 220 字 |

## 展览语言：中文优先

博物馆用英文著录，但展览是给中文读者看的，所以：

- **展品名**：优先用机构自己的中文题名（`titleOriginal`）；没有的由模型意译成中文展品名，存进 `ExhibitionItem.displayTitle`。英文原题不隐藏，降为副题并保留在「来源」面板里。
- **展签**：每件展品由 `deepseek-flash` 阅读真实馆藏图像，并结合该件藏品的馆方文字写 2–3 句中文展签。图像句只描述可见的颜色、轮廓、构图或表面状态；年代、材料、用途等只能来自馆方著录。所有句子合计受 80 / 140 / 220 字预算约束。
- **机构原文永不改写**：机构原文只在「来源」中出示，不混进公众展签；模型写的图像观察与馆方文字解释分别绑定 `collection_image` 和馆方文字证据 ID。
- **简繁转换是确定性的**：模型在写古典艺术文本时会漂移到繁体，而界面是简体，混排读起来就是坏的。所以 `to_simplified()`（OpenCC t2s）对**所有系统生成文本**做转换，绝不碰机构原文。这样「输出简体」不依赖模型是否听话。

## 两阶段模型写作，不是一次性生成

一次性索要整场展览（主题＋章节＋结语＋空间＋所有展签）会让模型跑过三分钟。现在拆成：

1. **策展框架调用**：先由程序建立 `CuratorialBrief`，再为每件展品传入最多 3 条、每条最多 600 字符的受控馆方证据摘要。模型输出题名、命题、分论点、对象入选理由、章节与结语；命题、分论点和对象决策都必须返回 `evidenceIds`。未知、隐藏或跨对象引用不会获准；核心框架审核失败仍整帧回退，少量补充字段可按 RC11 协议移除失败断言并明确标记局部降级。
2. **视觉展签调用**：每件展品单独调用 `deepseek-flash`，并发执行。后端先通过馆藏图片缓存下载并统一成 1024 px WebP，再以内嵌图像发送；因此 AIC 等需要专用请求头的图片也不依赖模型代抓取。单件调用同时接收该件馆方文字和同一份策展命题，但不得把内部角色、入选理由或流程术语写给观众。某张图或某次响应失败时，只保留该件的确定性馆方信息句，不拖垮其他展品。

框架仍先于展签返回，因此海报可以在逐件视觉展签写作期间并行生成。旧版“主题 47s、全流程 116s”的记录来自不传证据摘要的提示词，加入 `CuratorialBrief` 后需要重新测量，不能沿用为当前性能结论。

### CuratorialBrief：检索与文案之间的策展任务书

每个新的访客展览在模型写作前都会保存版本化的 `curatorial-brief/v1`，其中包括：访客问题与非识别性受众设置、Big Idea、带证据 ID 的 Key Messages、关键问题、每件展品的角色／入选理由／前后关系、最多 5 件未入选候选及原因、解释政策、评估目标，以及实际执行的检索方法和版本。它让题名、展签、TTS 和后续编辑共用同一套策展论证，而不是让模型在选物之后临时编一个故事。

私有工作记录保留受众设置和排除偏好；公开分享投影会删除这些字段，并把内部 item UUID 重映射为公开编号。provenance、文化敏感性和来源社群审阅默认都是 `not_reviewed`／`not_assessed`，没有记录绝不等于已经排除风险。

证据 ID 白名单能证明“这条文字指向了一份真实且允许使用的输入材料”，但**不能自动证明句子在语义上被材料蕴含**。模型精炼后的 Brief 会保留“语义蕴含仍待专业复核”的警告；公开发布仍需要领域审阅，涉及敏感遗产时还需要相应社群或文化持有者审阅。本 Demo 不把结构化追溯包装成专业策展认证。

## 馆藏推荐：可审计的混合 RAG（第一阶段）

候选链路是 `冻结馆藏 → 受控 QueryPlan → 本地硬过滤 → 对象／证据稀疏与向量召回 → RRF → Qwen 重排 → DeepSeek 证据审查 → CuratorialBrief`。运行时不临时抓取博物馆 API，也不让模型凭常识先挑藏品：

1. planner 在首次昂贵检索前产生类型化 `QueryPlan`，只允许年代、文化包、机构、材料、有图、权利与证据深度等白名单硬条件。硬约束在版本化的只读 SQLite 派生库中用绑定参数执行，模型不能直接提交 SQL。自然语言物件名称另存为 `catalogueTypeHints`，交由语义检索和审核确认，避免把“篮子”直接与某馆的 `Textile`、`Tools and Equipment` 等大类字段作字面等同；显式 API `FilterSpec.object_types` 仍支持严格类型过滤。材料及显式类型按完整词组匹配复合著录，例如 `paper` 可匹配 `Ink on paper`，但 `stone` 不匹配 `limestone`；明确过滤后不超过 60 件的小结果集另保留过滤召回通道，仍须经过相关性与证据审查。
2. hybrid 候选链先用 SQLite FTS5 对题名、路由、元数据与正文做有界对象 BM25 召回，再用原字段化 BM25 重施题名、排除词和全部锚点门控；另一条独立的证据 BM25 直接定位 61,620 条馆方 `evidence[]`。完整全库字段化 BM25 仍是 shadow 实际交付的基线。中英文别名只用于召回加权和歧义保护，**不再充当访客问题白名单**。
3. `qwen3.7-text-embedding` 以 768 维向量分别索引对象文档和馆方证据；向量在运维命令中预先生成，请求期间只计算查询向量。稀疏与向量排名经 RRF 融合后，`qwen3-rerank` 再根据访客问题和机构证据重排有界候选；相似度和重排分都不直接等于“证据支持”。
4. 进入展览前，DeepSeek 对有界候选做证据级相关性审查。全局重排后重新保留明确问题轴及文化区域的审核名额。模型只能返回本轮看过的对象 ID 和同对象、非 provenance 的证据 ID；缺失、猜测、跨对象或入藏来源引用都不能计入五件证据链。审核区分开放观展、视觉观察与事实解释：馆方文字描述的可见形态可用于明确标注的策展解读，但不能独自证明历史因果；文本审核不会把图像 URL 当作已经看图。必要时最多执行一次保持原意的扩展与复审；部分支持且满足展品数量与文化覆盖约束时可生成带限制说明的展览，否则明确说明不足，不拿最近邻补位。

第一阶段保留 MiniLM 384 维索引作为离线基线，不把它当成新的生产默认。Qwen 对象与证据矩阵仍是版本化 `.npy`，运行时以只读 mmap 做精确余弦扫描；SQLite manifest 和 dense manifest 都绑定馆藏 ID、版本与 `objects.json` SHA-256，请求期间不会重建索引。预计算文档、访客查询与重排候选会发送到配置的阿里云北京业务空间；检索 trace 只在本地保存规范化问题的 SHA-256、候选 ID、通道分数、延迟与错误，不重复写入问题原文。

`RAG_MODE=shadow` 会运行新候选链并记录对照 trace，但仍向后续链路交付 BM25 结果；run JSONL 会把实际交付的 `results` 与单查询 `candidateResults` 分开，不能把前者误当成混合排名。离线质量对照显式使用 `--rag-mode hybrid` 生成候选 run；上线前需通过门槛并实时核验服务器配置。`RAG_MODE=hybrid` 才会真正交付新排名，`RAG_MODE=bm25` 是应急和无密钥降级开关。示例配置默认使用 `shadow`；**当前不因“代码已接入”就认定已达到 cutover 门槛**。索引缺失、过期或损坏会可观测地回退 BM25；LLM 审查超时或返回无效结构时，未经审查的 dense-only 近邻仍会被排除。

2026-09-06 的混合检索核心为 `hybrid-rag-v4`，规划审核层为 `agentic-rag-v5.1`。应用与评测共享同一仓库工厂、provider 与重排配置，评测另存规划、检索、审核样本及接受集，并隔离超时后台任务的晚到 trace，不把初始召回分数与最终展览质量混算。V5.1 仅在原审核时限内补一次传输连接失败重试，不重试余额不足、HTTP 拒绝、无效输出或超时，也不反复询问模型以改变不利判定。137 题授权 AI 二审及恢复运行新增候选盲审已完成，但不是独立人工金标准；池外未审对象仍按未标注处理。当前进度、真实 API 故障与恢复验收见 [RAG 优化验收记录](data/qa/RAG_优化验收_20260906.md)。

## 不可配置的硬约束

这两条不接受模型或参数覆盖：

- **每场展览至少一件 `contrast` 展品**，防止个性化塌缩成熟悉性过滤泡。
- **`evidenceDepth = thin` 的展品不得承担 `core_evidence` 角色**。The Met 不提供策展说明字段，其证据片段是著录信息重述而非机构叙述文本，因此只承担背景/对照位置。选择阶段会保证候选集中至少有一件 `full` 展品，排位阶段会为它让出内部位置。

## 馆藏数据

当前默认馆藏是 `data/collections/global_open/`：**17,246 件**具有公开图片的 CC0 / Public Domain 对象，来自 [Cleveland Museum of Art Open Access](https://openaccess-api.clevelandart.org/)、[The Metropolitan Museum of Art Open Access](https://metmuseum.github.io/) 与 [Art Institute of Chicago API](https://api.artic.edu/docs/)。这是面向 Demo 的全球首批 serving corpus，不是对世界文化或各机构馆藏规模的代表性抽样。

纳入门槛：机构公开声明 CC0 / Public Domain、图片可解析、有稳定机构对象页，并保留至少 2 条可定位证据片段。当前 **17,246 / 17,246** 件均完成远程图片校验，`objects.json` 的 SHA-256 与 manifest 一致。

机构分布：Cleveland Museum of Art 15,200 件，The Met 1,046 件，Art Institute of Chicago 1,000 件。证据深度：6,126 件 `full`（含机构说明）/ 11,120 件 `thin`（主要是权威著录字段）。

文化包是可重叠的语料分面，不把对象的文化身份压成唯一标签。下表是冻结导入快照的原始计数；运行时 `controlled-origin-v2` 根据明确的 culture/place 字段修正了 837 件印度对象的东南亚误标，并为另外 2 件补充南亚标签，保留不确定及跨地区记录，不覆盖原始 JSON。实时检索分面与下表可能不同：

| 文化包 | 件数 | 文化包 | 件数 |
| --- | ---: | --- | ---: |
| 欧洲 | 6,875 | 东亚 | 3,741 |
| 美洲 | 2,765 | 西亚与北非 | 1,571 |
| 东南亚 | 1,309 | 南亚 | 692 |
| 非洲 | 480 | 大洋洲 | 103 |

访谈和检索使用十个跨文化证据域；它们是语料路由标签，**不是**系统预设的展览主题：

| 证据域 | 件数 | 证据域 | 件数 |
| --- | ---: | --- | ---: |
| 材料与制作 | 12,051 | 图像、观看与故事 | 6,609 |
| 身体与身份 | 2,312 | 书写与记忆 | 2,070 |
| 权力与身份秩序 | 1,753 | 日常生活 | 1,486 |
| 信仰与仪式 | 740 | 自然与地方 | 655 |
| 死亡与来世 | 437 | 交流与流动 | 118 |

原始 API 响应、访问时间与 SHA-256 保存在 `raw/snapshots/`，因此映射逻辑改动后可离线重建而无需重新抓取。

`data/collections/chinese_art_open/` 的 1,817 件东亚旧库仍保留，用于回归比较和回退；运行时默认馆藏通过 `DEFAULT_COLLECTION_ID=global_open` 显式选择，不依赖目录排序。

Art Institute of Chicago 已按部门、文化区与材料分层纳入 1,000 件。导入和运行时图片请求使用 AIC [官方文档建议](https://api.artic.edu/docs/)的 `AIC-User-Agent`，不伪造 `Referer`、不使用 Cookie，也不执行 Cloudflare challenge。许可逐字段记录：公有领域图像和除 `description` 外的作品 API 数据记为 CC0 1.0，`description` 及其证据片段单独记为 CC BY 4.0；`short_description` 保留其原始字段身份，按其余 API 数据记为 CC0 1.0。AIC 线上搜索的实际结果窗口为 1,000 条，导入器以稳定 ID 游标分窗，而不是只取排序靠前的 1,000 条。

图片策略见 01b §1.2：元数据全部本地冻结，图片默认热链机构 CDN，只对**实际展出过**的对象缓存 1024px WebP（`api/runtime/cache/objects/`，LRU 默认上限 256 MB，可通过 `IMAGE_CACHE_LIMIT_MB` 调整）。

### 图片代理

`GET /api/images/{objectId}?w=1024` 服务端取图、缩放、转 WebP 并加 CORS 头。这不是可选优化：Cleveland 的 CDN 不发 `Access-Control-Allow-Origin`，WebGL 无法采样被污染的贴图。它同时把 640 KB 的原图压到约 90 KB。代理只会取冻结馆藏里已有的 URL，不能被当作开放代理使用。

## 本地启动（Windows PowerShell）

环境要求：Node.js 20+、Python 3.10+。

```powershell
cd "D:\WHU PhD\博3\ICHEC\inquiry-curator"
npm.cmd install
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\api\requirements.txt
Copy-Item .\.env.example .\.env
# 在 .env 填入 DASHSCOPE_API_KEY 后，预先构建两个版本化派生索引
npm.cmd run rag:index:qwen
npm.cmd run rag:index:filters
```

在 `.env` 填服务端配置。不要把密钥放进任何 `NEXT_PUBLIC_*` 变量，也不要提交 `.env`。如果只需要无密钥启动，先把 `RAG_MODE` 改为 `bm25`；这会关闭云端 embedding 与 rerank，但不影响本地字段化 BM25 基线。

```powershell
npm.cmd run dev
```

访客端 [http://localhost:3000](http://localhost:3000)，API 健康检查 [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)，内部数据工具 [http://localhost:3000/dev/admin](http://localhost:3000/dev/admin)（生产构建下 404）。137 题开放检索候选的人工审核工作台位于 [http://localhost:3000/dev/admin/qrels](http://localhost:3000/dev/admin/qrels)；它采用单人本地审核模式，打开页面即可使用，无需令牌或审阅者代号。审核页及其写入 API 在生产环境中均返回 404。

## 配置

```text
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_LABELS_MODEL=deepseek-flash
DEEPSEEK_TIMEOUT_SECONDS=90
DEEPSEEK_FRAME_TIMEOUT_SECONDS=80
DEEPSEEK_LABELS_TIMEOUT_SECONDS=45
GENERATION_POSTER_WAIT_SECONDS=2
GENERATION_JOB_TIMEOUT_SECONDS=180
DASHSCOPE_API_KEY=
ALIYUN_TEXT_API_HOST=https://llm-nwypztqdwtzyt9zd.cn-beijing.maas.aliyuncs.com
ALIYUN_IMAGE_API_HOST=
ALIYUN_IMAGE_MODEL=qwen-image-3.0-pro
ALIYUN_IMAGE_SIZE=1536*864
ALIYUN_TTS_API_HOST=
ALIYUN_TTS_MODEL=qwen-audio-3.0-tts-plus
ALIYUN_TTS_VOICE=qwen-audio-3.0-tts-plus-longyulianrong
ALIYUN_TTS_INSTRUCTION=请使用专业、克制、清晰的博物馆导览播音主持声线，语速稍慢，停连自然，避免夸张表演。
ADMIN_REVIEW_TOKEN=change-before-public-deployment
EDITOR_ACCESS_TOKEN=change-before-public-deployment
QREL_REVIEW_DATASET_DIR=data/qa/retrieval_eval_v1
QREL_REVIEW_DB_PATH=api/runtime/qrel-reviews/retrieval_eval_v1/reviews.sqlite3
PSEUDONYMOUS_EVENT_RETENTION_DAYS=30
DEFAULT_COLLECTION_ID=global_open
IMAGE_CACHE_LIMIT_MB=256
RAG_MODE=shadow
RAG_EMBEDDING_PROVIDER=aliyun
RAG_EMBEDDING_MODEL=qwen3.7-text-embedding
RAG_EMBEDDING_DIMENSION=768
RAG_INDEX_DIR=api/runtime/cache/rag
RAG_MODEL_CACHE_DIR=api/runtime/cache/fastembed
RAG_DENSE_TOP_K=200
RAG_DENSE_MIN_SCORE=0.28
RAG_EVIDENCE_MIN_SCORE=0.30
RAG_RRF_K=60
RAG_MAX_RESULTS=250
RAG_RERANK_ENABLED=true
RAG_RERANK_MODEL=qwen3-rerank
RAG_RERANK_CANDIDATE_COUNT=60
RAG_RERANK_TOP_N=24
RAG_RERANK_TIMEOUT_SECONDS=6
RAG_RERANK_MAX_ATTEMPTS=1
RAG_STRUCTURED_FILTERS_ENABLED=true
RAG_FILTER_INDEX_DIR=api/runtime/cache/filters
RAG_EVIDENCE_BM25_TOP_K=120
RAG_TRACE_DIR=api/runtime/traces/retrieval
RAG_RETRIEVAL_TIMEOUT_SECONDS=40
RAG_LLM_AUDIT_ENABLED=true
RAG_LLM_AUDIT_TOP_K=18
RAG_LLM_AUDIT_TIMEOUT_SECONDS=22
RAG_AGENTIC_MAX_QUERIES=5
```

开放问题的完整细节见上文“馆藏推荐”。两份 `.env.example` 都使用北京业务空间专属域名，不使用通用端点；密钥始终留空。`RAG_MODE=shadow` 是评测期配置，不是对 `hybrid` 质量已经达标的声明。

初始召回、证据审查以及必要时的批量扩展／复审默认共用 40 秒墙钟预算；策展框架和并行展签调用默认分别受 80 秒、45 秒墙钟预算约束。加上海报等候 2 秒及预留 10 秒，177 秒小于默认任务时限 180 秒；已有 `.env` 的自定义值仍优先。同步检索和融合重排使用进程共享、最多 4 个后台调用且无等待队列的执行器：超时返回后仍在运行的调用继续占用名额，直到实际退出；容量满返回 `RETRIEVAL_CAPACITY_EXHAUSTED`，不把它误报成主题无藏品。该预算不等于包括冷加载、所有策展阶段及资源生成在内的整场展览总耗时。

相关性审查是自由问题的生成前置条件：若它超时或不可用，系统会明确失败关闭，并提示重试，而不会把未经审核的 BM25／向量近邻当成馆藏证据。只有审查已经通过之后，策展框架或展签模型失败才会回退到确定性文案。海报最多额外等待 2 秒，之后转入后台继续生成，不再阻塞可浏览展览。180 秒只作为任务状态机的最后安全兜底；若进程中断，已经写入的 `generating` 骨架会恢复成明确的草稿，而不会永久显示生成中。模型失败与回退原因会写入日志，避免静默降级。

阿里云图像生成需同时配置 `DASHSCOPE_API_KEY` 与该 Key 所属业务空间的 `ALIYUN_IMAGE_API_HOST`（北京与新加坡端点不可混用）。海报采用 `qwen-image-3.0-pro` 生成 1536×864 横版主题主视觉：服务端先在本地把访客主题路由为白名单内的纯英文视觉母题，模型不会收到原始题名、问题、中文或自由输入，只负责生成与主题相关的无字编辑视觉；它不得复制或伪造具体馆藏，也不得生成文字、Logo 或水印。随后服务端使用 Pillow 将最终中文标题、短副标题及「卧游 · AI 策展人彦远」确定性排入同一张 PNG，避免模型错字与伪文字。页面持续标注主题画面由 AI 生成、文字由系统排版且不代表馆藏实物。瞬时网络失败会受控重试一次；仍失败或浏览器加载失败时，2D 与 3D 入口改用本展馆藏公开图像与同一标题模板，不留空白海报位。

展厅导览语音使用北京地域的千问 `qwen-audio-3.0-tts-plus` 非实时 TTS。配置同一服务端 `DASHSCOPE_API_KEY`，并把该 Key 所属北京业务空间专属域名填入 `ALIYUN_TTS_API_HOST`；若图像和语音确实位于同一北京业务空间，留空时会回退到 `ALIYUN_IMAGE_API_HOST`。默认声线为 `qwen-audio-3.0-tts-plus-longyulianrong`，以专业、克制、清晰、稍慢且停连自然的博物馆播音风格生成 24 kHz MP3，并写入 AIGC 音频标识。接口为 `GET /api/exhibitions/{exhibition_id}/audio-guide?kind=lobby|chapter|artwork|epilogue&ref=...`；章节 `ref` 使用章节 ID，展品 `ref` 使用展览条目 ID。服务端只从已保存展览重建讲解词，不接收任意待朗读文本；缓存键包含模型、声线、风格指令、格式与讲解文本，避免修改展签后误播旧音频。未配置或临时生成失败时接口返回 503，但展览与展厅仍可浏览，客户端可明确回退到设备语音。

## 数据更新

```powershell
# 首次构建全球馆藏：实时抓取并完整校验远程图片
npm.cmd run data:import:global

# 从全部已冻结快照重建基础集、CMA 补充集和 AIC 切片；仍会重新验证 serving 图片
npm.cmd run data:rebuild:global

# 只重建 14,000 件 CMA 基础集与 Met 种子（通常不要单独使用）
npm.cmd run data:rebuild:global:base

# 为当前全球馆藏补充 CMA 的南亚/东南亚对象
npm.cmd run data:supplement:global

# 按部门、文化区与材料分层补充 1,000 件 AIC 公有领域对象
npm.cmd run data:supplement:aic

# 馆藏版本、文本配方或 Qwen embedding 变化后重建向量索引
npm.cmd run rag:index:qwen

# 馆藏版本或结构化字段映射变化后重建 SQLite/FTS5 派生索引
npm.cmd run rag:index:filters

# 保留的 MiniLM 离线基线
npm.cmd run rag:index:minilm

# 旧东亚馆藏的导入与离线重建
npm.cmd run data:import
npm.cmd run data:rebuild
```

导入器每次创建新的原始快照，不静默覆盖来源证据；输出文件用原子替换写入。完整离线重建会依次恢复基础集、CMA 南亚／东南亚补充快照与 AIC 分层切片，避免只运行基础导入器时悄然减少馆藏。更换数据版本后应重跑回归集。全球导入器按机构部门分层取样，并保留未命中当前证据域但满足权利、图片和证据门槛的对象，避免把分类规则误当成纳入门槛。

## 质量检查

```powershell
npm.cmd run test
```

依次运行 `typecheck` → `test:web`（展览空间、许可呈现与拼图逻辑单测）→ `test:api` → `qa:regression` → `qa:diverse`。

`qa:regression` 完全离线（不读云模型密钥）。它使用一个只能回传本轮候选 object/evidence ID 的合成审核器，检查审查后编排的结构约束；**这不是语义相关性质量证明**。它检查两件事：

1. 答复性门控是否仍复现每一条既定标签，且仍会拒绝（不能变成永远说 yes）；
2. **覆盖域 × 时长 × 动机**的 48 种组合是否都生成结构合法、可走完的展览——角色齐备、对照声音仍在、`thin` 展品未占核心证据位、展签未超字数预算。

它还输出个性化差异指标：不同画像所选展品集合的平均 Jaccard 距离（当前 **0.94**，阈值 0.50）。这只能说明选择集合不同，不能直接推出用户感知到更个性化或学到了更多。

最近一次结果：

```
collection      global_open @ 20260826-17246
objects         17246 loaded, 17246 with evidence
evidence depth  6126 full / 11120 thin
audit fixture   synthetic candidate-bound IDs (not semantic QA)
ok    answerability   26 questions, {'supported': 20, 'partially_supported': 3, 'unsupported': 3}
ok    exhibitions     48 generated, roles {opening:48, context:128, core_evidence:128, contrast:48, synthesis:48}
ok    personalisation mean Jaccard distance 0.94
```

真实 embedding＋LLM 证据审核另行运行，不混入 keyless CI：

```powershell
npm.cmd run qa:open-rag
```

该套件会核对审核是否只接受本轮展示过的对象／证据 ID、跨文化候选池与最终五件展厅是否都保留要求的文化腿，以及 OOD／伦理／完整因果问法是否诚实降级。它依赖真实模型服务，因此同时记录耗时与外部响应漂移，不能拿一次绿灯代替冻结模型上的 P95 或人工内容评审。

第一阶段另有一套 250 问的冻结检索评测集。先验证数据集指纹和 ID，再对已冻结的 run JSONL 计算 Recall@50、nDCG@10、MRR@10、Success@5、文化腿覆盖、证据支持、错误 `supported` 与分阶段延迟：

```powershell
npm.cmd run qa:retrieval:validate
npm.cmd run test:retrieval-eval
npm.cmd run test:retrieval-dataset
npm.cmd run test:retrieval-run
npm.cmd run qa:retrieval:run -- --run-name bm25-v1 --rag-mode bm25 --resume
npm.cmd run qa:retrieval:run -- --run-name phase1-qwen-v1 --rag-mode hybrid --resume
npm.cmd run qa:retrieval:compare -- --run bm25=artifacts/qa/retrieval-runs/retrieval_eval_v1-bm25-v1.jsonl --run phase1=artifacts/qa/retrieval-runs/retrieval_eval_v1-phase1-qwen-v1.jsonl
```

`qa:retrieval:run` 以生产检索器生成可续跑、逐问落盘的 run；默认不调 DeepSeek，只有显式加 `--audit` 才记录答复性和接受证据。shadow run 的 `results` 始终是服务基线，候选另存为 `candidateResults`；当前评分器只评分 `results`，所以候选质量比较必须使用上例的离线 hybrid run。`qa:retrieval:compare` 只评分运行文件，不在评分时调模型或修改 qrels；报告写入本地忽略的 `artifacts/qa/`。当前确定性字段匹配是程序化 gold，开放主题候选只是待人工审阅的 pooled silver，不得冒充人工金标。切换 `hybrid` 前至少要确认：Recall@50 相对 BM25 提升≥15%、nDCG@10 提升≥10%、精确题名／作者问法下降≤2%、引用错配为 0、错误 `supported` <2%、要求的文化腿覆盖≥90%，并完成冻结环境的 P95 延迟检查。v1 只有 100 个字段型 scored gold；137 个开放问题尚待人工复核，文化题中没有 scored 的多文化腿问题，默认 retrieval-only run 也不产生 claim-level 引用判断。因此本轮程序化 compare 只能验字段回归和运行稳定性，后三项必须补充人工多文化／claim-level qrels 并运行 `--audit` 后才有资格判定。**这些是 cutover 门槛，不是当前已通过的结果。**

人工审核不会原地修改 `data/qa/retrieval_eval_v1/questions.jsonl` 或 `qrels.jsonl`。每次按钮判断都以审阅者为作用域写入忽略提交的 SQLite revision 日志；写入前再次核对 v1 问题、qrels 与 17,246 件馆藏文件的 SHA-256，候选对和证据 ID 也必须属于冻结语料。页面隐藏 pooled-silver 原始候选分数并采用稳定打乱顺序，减少系统排序对人工判断的诱导。完成一题的 12 个候选后，审阅者还需用按钮确定该题的 `supported / partially_supported / unsupported`；全部 137 题完成后，审核快照才能进入另一个版本化金标冻结流程。

### 2026-08-31 第一阶段本地验收结论

以下保留首轮历史结果；2026-09-06 的修复、完整对照、授权 AI 复核与未过门槛以 [优化验收记录](data/qa/RAG_优化验收_20260906.md) 为准，不将两批结果混算。

- Qwen 768 维派生索引已覆盖 17,246 件对象与 61,620 条证据；SQLite v4 派生库同时提供白名单硬过滤、中文二／三元切分的对象 FTS 与证据 FTS。真实反例“雲山”可召回 `cma:1933.220`《雲山圖》。
- 同一组 100 个程序化字段 gold 上，BM25 → Qwen+RRF+rerank 的 Recall@50 为 `0.2691 → 0.7321`，nDCG@10 为 `0.1696 → 0.5233`，Success@5 为 `0.2000 → 0.6300`。35 个精确题名／作者／机构问题的 Recall@50 为 `0.2000 → 0.9714`，没有发生精确检索回退。
- 总耗时 P50/P95 为 `12.08/14.80 s → 6.33/13.04 s`；Qwen 查询 embedding 的 P95 为 `7.48 s`，rerank 的 P95 为 `6.07 s`。100 题没有未捕获 API error，但有 4 题发生可观测 rerank fallback。
- 13 个 deterministic evidence-boundary 问题在真实 DeepSeek 审查下全部判为 `unsupported`，错误 `supported` 为 `0/13`；但总耗时 P95 达 `39.72 s`，4 题带超时／降级警告。
- 当时 scored 单文化腿覆盖为 `75.56%`，低于 `90%` 门槛；更重要的是 v1 没有 scored 多文化腿、人工开放主题或 claim-level 引用金标。该轮结论为：**第一阶段代码与评测基础设施完成，但 cutover 未获批准；不应直接全量切成 hybrid。** 此处不是当前服务器实际 mode 的探测结果。

## 已知问题与未验证项

- **2026-08-30 的真实语义审核探针**：用户原句“有没有各文化地区的猫的藏品”从 68 个召回候选中接受 12 件，候选池与最终五件展厅都覆盖 4 个规范化文化区域，判为 `supported`（9.37 秒）；“狗狗在各国文化是怎么存在”从 99 个候选中接受 7 件、覆盖 4 个区域，但因馆方记录不足以完整支持“怎么存在”这一关系命题而判为 `partially_supported`（23.28 秒）。这说明对象量不是唯一瓶颈；同一批对象可以支持“有哪些形象／是什么物件与材料”，却未必支持完整象征或因果解释。以上是定向实测，不是冻结环境 P95，也不代表所有开放问题都可生成。
- **3D 交互与响应式布局已经过真实浏览器运行验收**：覆盖桌面与 433 px 移动端、展签拖拽/键盘移动/缩放/收纳、横版海报与馆藏图回退、上下导航可见性。尚未做长时间帧率、显存占用与不同 GPU 下 WebGL 光照基准测试。
- 这版全球广度仍主要来自 Cleveland Museum of Art（15,200 / 17,246）；The Met 1,046 件、AIC 1,000 件是分层种子，因此不能把机构分布或文化包件数解释成世界馆藏分布。下一批应优先扩 The Met 全球 Public Domain 池，并引入第四个权利与供图边界清楚的机构。
- 「交流与流动」虽由 84 件增至 118 件，仍是当前最稀疏的证据域；大洋洲文化包 103 件，也低于每包 200 件的内部目标。两者是下一轮定向补库优先级，而不是用更多欧洲/制作类对象稀释问题。
- 517 件合格对象暂未命中十个证据域、234 件未稳定分配到文化包；它们被明确保留并计入 manifest，供后续本体规则改进，而不是静默丢弃。
- AIC 图片请求目前依赖其文档建议的项目识别头；Cloudflare 规则若再次变化，需向 AIC 获取受支持的接入方式。AIC 官方同时要求图片单线程、约一秒间隔，因此不适合未经分层筛选就全量逐图验证。
- **落地页与 3D 展厅均已完成真实浏览器视觉走查**：落地页覆盖 1707 px 桌面与 390 px 移动端，3D 展厅覆盖桌面与 433 px 移动端，均无横向溢出或主要控件裁切。尚未覆盖更多浏览器内核、长时间 WebGL 性能和不同 GPU 下的光照差异。
- 未完成：8 名用户形成性试用、遗产专业人员内容走查、移动端与键盘走查、冻结环境性能测试、公开部署的隐私文本与管理员认证。

## 目录

```text
inquiry-curator/
├─ src/app/                 访客路由、分享页、/dev/admin
├─ src/components/          落地页、访谈、流水线、2D 版本
│  └─ hall/                 3D 展厅：布局计算、场景、双模相机、覆盖层、语音导览
├─ src/lib/                 API 客户端、类型、客户端能力探测
├─ api/app/                 FastAPI：访谈状态机、策展、任务、图片代理、验证器
├─ contracts/               JSON Schema
├─ data/collections/        冻结馆藏、快照、问题卡、回归集、权利审计
└─ scripts/                 多机构导入器（sources/）、覆盖域规则、离线回归
```

## 研究与隐私边界

- 日志使用一次浏览会话的伪匿名 ID，不称为完全匿名。
- 不收集照片、精确位置或敏感身份信息。
- 参观完成率、章节进入、来源展开等只作为**产品行为**描述，不解释为学习成效。
- 文物主体只使用机构提供的 CC0 原图；不生成仿古文物，不改画机构图像，不为没有扫描数据的立体器物造 3D 模型（以展柜 + 照片呈现）。
- 公开 Demo 的动态生成结果不直接当作受控实验刺激；研究模式另行冻结。
