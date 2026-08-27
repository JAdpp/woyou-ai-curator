# global_open

卧游当前默认使用的全球开放馆藏 serving corpus。

## 冻结版本

- 版本：`20260826-17246`
- 对象：17,246 件
- 机构：Cleveland Museum of Art 15,200；The Metropolitan Museum of Art 1,046；Art Institute of Chicago 1,000
- 证据深度：`full` 6,126；`thin` 11,119
- 远程图片校验：17,246 / 17,246
- 文化包分配率：98.64%
- 全球证据域路由率：97.00%
- 字段许可模式：`rightsSchemaVersion = 1.0`
- `objects.json` SHA-256：`8a7cee291acc9e5beedd40cf12e30299049613cbb61b6954e1a4fb9a5758d55e`

这是一份为个性化策展 Demo 构建的分层语料，不是世界文化或各机构馆藏的代表性样本。文化包和证据域均为可审计的检索分面，不是对文物身份的唯一判断，也不是预设展览主题。

## 文件

- `objects.json`：规范化馆藏对象与逐条证据、字段级许可、来源和图片校验结果。
- `manifest.json`：版本、机构/文化包/证据域分布、权利边界、快照 ID 与对象哈希。
- `question_cards.json`：由实际语料覆盖生成的访谈入口。
- `regression_questions.json`：答复性门控的支持、部分支持与拒绝样例。
- `rights_audit.md`：按机构、对象字段与证据片段汇总的权利审计。
- `raw/snapshots/`：机构 API 原始页、访问时间、请求参数与校验哈希。

## 重建与验证

```powershell
npm.cmd run data:rebuild:global
npm.cmd run test
```

`data:rebuild:global` 会依次恢复基础快照、CMA 南亚／东南亚补充快照和 AIC 分层切片；不要只运行基础导入器后直接发布，否则会漏掉补充集。

AIC 补库从 1,080 个分层候选中保留了 1,000 个图片可达对象；`description` 记为 CC BY 4.0，`short_description` 与其余作品 API 数据保持 CC0 1.0。未完成或失败的追加快照不会参与离线重建。

旧的 `chinese_art_open` 馆藏独立保留，便于比较与回退；不要用它覆盖本目录。
