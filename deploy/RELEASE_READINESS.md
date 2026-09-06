# 卧游：检索升级发布准备

当前聚焦版本：[RC11 模板回退与遗留修复](../data/qa/RC8_模板回退与遗留修复_20260906.md)。停止新题扩测；服务器三条既有流程均成展且无整篇模板回退，仍保留明确的局部降级。部署兼容通过，尚未公开切流。

2026-09-06 的 V6 候选实测记录见 [部署候选验收](../data/qa/RAG_部署候选验收_20260906.md)。工程兼容、内容质量和公开入口切流分别记录，不能互相替代。

更新：2026-09-06。本文件说明可重复的工程检查，不宣称语义质量已达标，也不是已经部署的证明。每次发布另保留实际运行结果、源码指纹和目标服务器检查时间。

## 当前部署形态与边界

项目级 `deploy/nginx/` 是 nginx 的维护副本；父目录 `ICHEC/deploy/nginx/` 是旧副本，不能覆盖它。历史部署文档记录：

| 层 | 位置／监听 |
| --- | --- |
| 公开入口 | nginx `8081` |
| 前端 | `/opt/demos/inquiry-curator/.next/standalone/server.js`，Node `127.0.0.1:3000` |
| API | `/opt/demos/inquiry-curator` 中的 FastAPI，`127.0.0.1:9001` |
| 服务 | `inquiry-api.service`、`inquiry-web.service` |

这些是部署约定，不代替服务器实时清点。当前代码仍把 `APP_ENV=production` 下的私有展览读取等接口绑定到 editor token；纯 HTTP 下还有 Secure cookie 条件。保持现有公开 Demo 访问方式时，不应未经验证直接改变 APP_ENV。当前 nginx 必须继续封闭 `/api/admin` 与 `/dev/admin`。

2026-09-06 17:17（UTC+8）本地只读检查：Python 3.10.7、Node 24.18.0、所需 Python 模块存在，运行中的 `127.0.0.1:8000/health` 为 BM25；海报、TTS、文本模型均报告已配置。健康字段 `retrieval.available` 表示 dense 是否可用，BM25 模式下 false 不等于 BM25 故障。仅在子进程指定 hybrid 后，离线的两类索引验证通过；没有因此改变共享 API 的实际模式。

该时间点的 `.next/standalone` **不能直接上传**：还没有内部 `.next/static` 与 `public`，并包含两份 `.env*` 文件。只读检查只报告文件数，没有显示其内容。必须从构建结果建立独立暂存发布包，在暂存包内补静态资源、去除所有 `.env*`，再检查。不要删除或改写工作区／服务器的真实配置。

## 不发模型请求的 preflight

```bash
python scripts/release_preflight.py --expect-mode hybrid --require-cloud-assets
```

检查 Python／Node、必要模块、实际 Settings、模型配置是否存在、馆藏源 SHA-256、只读 SQLite 数据库的 SHA-256 和版本、dense 的模型／文本配方／维度／矩阵形状，另输出索引文件 SHA-256 供构建机与服务器逐项比对。脚本不创建目录、不下载模型、不建索引、不调用 embedding／重排／生成服务，不输出 key、环境变量全集、HTTP 错误体或代理凭据。

要检查**已经准备好的** standalone 包：

```bash
python scripts/release_preflight.py --expect-mode hybrid --require-cloud-assets --standalone-dir /path/to/staged/.next/standalone
```

要核对目标服务器正在运行的 API：

```bash
python scripts/release_preflight.py --expect-mode hybrid --require-cloud-assets --health-url http://127.0.0.1:9001/health
```

此命令在服务器项目目录运行；`/health` 不一定经公开 nginx 暴露。`--expect-mode` 是断言，不会修改配置；`shadow` 只计算新候选、对外仍交付 BM25，不等于 hybrid cutover。脚本退出码 0 仅说明指定的工程门通过，不能替代后续模型、前端和质量验收。

## 真实访客与资源测试

固定问题在 `data/qa/visitor_release_holdout_20260906.json`，SHA-256：

```text
d7f809ca93bf0f9ad17b2f2568492d7655b8d24607f3e76a159c4a96c9c7cb19
```

12 题在首次检索前冻结；只为去重读过旧问题，没有为这些题查询馆藏，没有对象 ID／gold。6 个成展意图、2 个明确边界的比较意图、2 个需要澄清的访谈输入、2 个资料域外要求。期待是产品行为假设，不是假装已知馆藏足够。只把问题和 profile 交给产品，不把 `checks`、期望、分类或任何评价理由放进模型上下文。

首次执行后，这套题成为固定回归集，不能在下一轮仍称“模型未见 holdout”。所有尝试保留，不能删失败、换题或改期望。诚实说明证据不足与实际成功成展分别统计。09、10 应从初始访谈入口检查澄清交互，不能仅调用生成接口后把报错当作成功澄清。

完整生成烟测示例：

```bash
python scripts/smoke_full_generation.py --output-dir artifacts/qa/release-run/case-01 --question "从冻结文件原样读取的问题" --default-stage-budgets --optional-assets-wait-seconds 120
```

脚本使用隔离 store 和产物目录，走正常异步生成接口，写入初始计划／检索、最终审核 diagnostics、实际视觉输入、选品和展签、校验结果。`--default-stage-budgets` 动态读取代码的 Settings dataclass 默认值，不把本地旧 `.env` 的自定义预算当作新默认。显式 `--job-timeout-seconds` 优先于该默认，结果中保留两者实际值。

没有 `--relay` 时不改代理环境。只有确有批准的临时转接时才显式给 `--relay`，不默认依赖已停用的 60885 端口。资源等待只观察已有海报、序章 TTS 任务，不发起重试或额外付费生成；记录 ready 海报文件 GET 和实际 MP3 有效性，未完成尾任务与 isolated-client 收尾取消另记。章节／展品／尾章音频、浏览器播放和页面交互仍需另外验收。

## 发包与目标服务器的必查项

1. 在本地完成质量回归与前端构建；不要在这台历史约 1.6 GB 内存的共享服务器上跑 Next build 或并发 pytest。内存、磁盘与现存服务状态需实时只读核验，不动其他 Demo 或 Streamlit。
2. 建立独立暂存包，静态资源位于 standalone 内，公开 JS 不得烤入 localhost／127.0.0.1 API 地址，包内不能有 `.env*`。preflight 的坏 URL 对照测试确保扫描不是因没有文件而空通过。
3. 同步完成的 Qwen 和 filter 指纹目录，排除 `.tmp-*`。保留并对比 `objects.json` 与各索引 artifact SHA-256；不能只比目录名。请求和启动期间不能重建索引。
4. 保留服务器 `.env`、data、store、generated 和缓存；仅原子替换本次代码／构建版本，旧版留作回滚。不要把本地环境覆盖到服务器。
5. 只读检查两个服务 active、公开首页与其静态 JS/CSS、图片并发请求、3D／2D切换、展览读取、海报实际 URL、音频实际 URL。海报 `ready` 不是资源路由通过证明：当前 nginx 的 `/generated/posters/` 走 Next，生成目录须对应实际服务的 standalone `public/generated/posters` 或已验证的现有媒体路由。不应猜测新路由已生效。
6. 目标服务器执行真实生成，并记录阶段时延、回退、内存高水位。单个成功例、candidate 数、HTTP 200、只有引用 ID 所有权正确，都不等于内容受证据支持。

BM25 是保留的回退开关：出现部署后故障可按既定授权恢复原配置和 API 版本；不要为回退删除已有索引或覆盖馆藏。需要诚实报告“可部署包”“服务器已部署”“模型质量通过”“端到端资源与交互通过”四个不同状态。
