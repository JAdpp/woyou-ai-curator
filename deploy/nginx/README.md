# nginx 配置（部署服务器，8081）

这两个文件是线上实际生效的配置副本，放进仓库是为了让改动的**理由**跟着代码走。改线上之前先改这里。

| 文件 | 线上位置 |
|---|---|
| `demo-inquiry.conf` | `/etc/nginx/sites-available/demo-inquiry.conf` |
| `demo-ratelimit.conf` | `/etc/nginx/conf.d/demo-ratelimit.conf` ⚠️ **三个 demo 共用** |

`demo-ratelimit.conf` 定义的限流桶被 8081、8082、8083 一起引用，改它会同时影响另外两个 demo（`demo-myth.conf`、`demo-trip.conf`）。

## 为什么图片要单独一个桶

2026-08-13 修：落地页在别的设备和浏览器上**大面积破图**。

根因不在前端，也不在图片接口——是 `/api/images/` 和昂贵的生成接口共用同一个 `demoapi` 桶（`5r/s`，`burst=15`，`nodelay`）。落地页水合后一次性请求约 30–48 张藏品图，前 15 个通过，**其余全部当场 503**（`nodelay` 是直接拒绝，不排队）。

看起来"跟设备有关"，是因为浏览器并发连接数不同：HTTP/1.1 每域名 6 条、HTTP/2 全部并发，于是每台机器破的图不一样。

实测（40 并发）：

| | 200 | 503 |
|---|---|---|
| 修复前 | 8 | 25 |
| 修复后 | 40 | 0 |

图片本身是**只读、不可变、命中本地磁盘缓存**的，和一次 DeepSeek 调用完全不是一类负载，所以给它 `demoimg` 桶（`120r/s`，`burst=240`）。实测三个访客在同一个 NAT 出口后面同时开页（90 并发）也全部 200。

## 两条别踩的

1. `location ^~ /api/images/` 的 **`^~` 不能删**，位置也**必须在** `location ~ ^/(api|...)` **之前**。nginx 里带 `^~` 的前缀匹配优先于正则匹配；去掉 `^~` 就会掉回 `demoapi` 桶，破图立刻复发。
2. 应用侧已经自己发 `Cache-Control: public, max-age=604800, immutable`，nginx **不要**再 `add_header` 覆盖。

## RC11 当前线上状态（2026-09-07）

已按用户授权切换 `8081` 的 RC11 公开 Demo，实际 `RAG_MODE=hybrid`，Qwen 768 维对象／证据索引与 SQLite 索引匹配 17,246 件馆藏。真实生成、资源验收和证据边界见 [RC11 上线验收](../../data/qa/RC11_上线验收_20260907.md)。这次是收尾版本的 Demo 发布，不表示下面历史方案中的全量开放题质量门槛已全部达到。

`/generated/posters/` 现由 nginx `alias` 直接读取 `/opt/demos/inquiry-curator/api/runtime/media/posters/`。发布时将原 standalone 的海报目录原样迁入此持久目录，新旧 standalone 都链接到这里；已有文件不因更换 Next 构建而消失。Next 对构建外符号链接目录曾返回 404，因此不能只检查文件存在或 API 的 `ready`。新增海报与保留媒体的实际 HTTP GET 另行验收。该 location 沿用 `demoimg` 桶，未改三个 Demo 共用的 `demo-ratelimit.conf`。

## 第一阶段文本检索原始上线方案（历史）

以下是公开切流前的原始方案，保留作为决策历史。当时线上保持 `RAG_MODE=bm25`，不能因本地代码接入 Qwen 就认定完成切换。RC11 实际上线状态以本文件上节和上线验收记录为准。

1. 先确认服务器与构建机的 `global_open` 版本和 `objects.json` SHA-256 一致。
2. 在有密钥的环境执行 `npm.cmd run rag:index:qwen` 与 `npm.cmd run rag:index:filters`。只同步完成的指纹目录；不同步 `.tmp-*` 中间目录，不在 API 请求或启动期间重建。
3. 服务端 `.env` 中配置 `RAG_EMBEDDING_PROVIDER=aliyun`、`RAG_EMBEDDING_MODEL=qwen3.7-text-embedding`、`RAG_EMBEDDING_DIMENSION=768`、`RAG_RERANK_MODEL=qwen3-rerank`、结构化 filter/证据 BM25 索引目录和 trace 目录。`ALIYUN_TEXT_API_HOST` 使用 `https://llm-nwypztqdwtzyt9zd.cn-beijing.maas.aliyuncs.com`；真实 `DASHSCOPE_API_KEY` 只留在服务器密钥文件，不回写仓库。
4. 首次只使用 `RAG_MODE=shadow`：对外仍交付 BM25，同时记录新候选排名、延迟和 provider 错误。检索 trace 只含问题哈希与候选诊断，不含问题原文。
5. 用 250 问冻结数据集产生 BM25 与 Phase-1 run，再执行 `qa:retrieval:compare`。只有达到 README 列出的 cutover 门槛、冻结环境 P95 可接受，且人工审阅 pooled-silver 候选后，才可另行决定是否改为 `RAG_MODE=hybrid`。

应急回滚只需把 `RAG_MODE` 改回 `bm25` 并重启 API 服务；不需要删除 dense 或 SQLite 索引，也不需要重载 nginx。

验证脚本见提交历史里的并发复现方法：取 `/api/collection/highlights` 里 `items[].id` 与 `domains[].samples[].id`，全部并发打一遍 `/api/images/{id}?w=512`，应当 100% 返回 200。注意别把 `institutionSummaries[].id`（`aic`/`cma`/`met`）和 `domains[].id` 当成藏品 id，它们本来就该 404。
