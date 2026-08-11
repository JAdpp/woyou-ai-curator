"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import type {
  AnalyticsSummary,
  CuratorialBriefClaim,
  DataAuditReport,
  Exhibition,
} from "@/lib/types";
import {
  approveExhibition,
  getAnalytics,
  getAnalyticsExport,
  getDataAudit,
  listAdminExhibitions,
  rejectExhibition,
  withdrawExhibition,
} from "@/lib/api";
import {
  buildCuratorialEvidenceIndex,
  COMMUNITY_REVIEW_STATUS_LABELS,
  CURATORIAL_CONFIDENCE_LABELS,
  CULTURAL_SENSITIVITY_STATUS_LABELS,
  EVALUATION_METHOD_LABELS,
  PROVENANCE_STATUS_LABELS,
  type CuratorialEvidenceReference,
} from "@/lib/curatorialBrief";
import { SiteHeader } from "./SiteHeader";
import styles from "./admin.module.css";

const STATUS_LABELS: Record<Exhibition["status"], string> = {
  draft: "草稿",
  // Visitor-facing states since the review chain left the public path.
  generating: "生成中",
  ready: "已生成",
  // Retained so previously stored records still render in this dev tool.
  auto_validated: "结构与引用完整性检查通过",
  review_pending: "待人工审核",
  published: "已发布",
  rejected: "审核退回",
  withdrawn: "已撤下",
};

const DISTRIBUTION_LABELS: Record<string, string> = {
  supported: "可生成",
  partially_supported: "部分覆盖",
  unsupported: "不可生成",
  pending_human_review: "待人工审核",
  reviewed: "已人工审核",
  "global:nature-place": "自然与地方",
  "global:belief-ritual": "信仰与仪式",
  "global:death-afterlife": "死亡与来世",
  "global:power-status": "权力与身份秩序",
  "global:body-identity": "身体与身份",
  "global:making-material": "材料与制作",
  "global:text-memory": "书写与记忆",
  "global:exchange-mobility": "交流与流动",
  "global:daily-life": "日常生活",
  "global:image-story": "图像、观看与故事",
};

const POSTER_STATUS_LABELS = {
  idle: "未生成",
  generating: "生成中",
  ready: "AI 视觉已就绪",
  failed: "生成失败",
} as const;

function distributionLabel(value: string) {
  return DISTRIBUTION_LABELS[value] ?? value.replaceAll("_", " ");
}

function distributionEntries(distribution: Record<string, number>, limit?: number) {
  const entries = Object.entries(distribution).sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]));
  return typeof limit === "number" ? entries.slice(0, limit) : entries;
}

function ReviewEvidenceLinks({
  evidenceIds,
  evidenceIndex,
  unresolvedLabel = "无法定位",
}: {
  evidenceIds: string[];
  evidenceIndex: Record<string, CuratorialEvidenceReference>;
  unresolvedLabel?: string;
}) {
  if (evidenceIds.length === 0) {
    return <span className={styles.unresolvedEvidence}>未绑定馆藏证据</span>;
  }

  return (
    <span className={styles.reviewEvidenceLinks}>
      {evidenceIds.map((evidenceId) => {
        const reference = evidenceIndex[evidenceId];
        return reference ? (
          <a key={evidenceId} href={reference.sourceUrl} target="_blank" rel="noreferrer">
            <code>{evidenceId}</code>
            <span>{reference.objectTitle} · {reference.sourceLocation} ↗</span>
          </a>
        ) : (
          <span className={styles.unresolvedEvidence} key={evidenceId}>
            <code>{evidenceId}</code> {unresolvedLabel}
          </span>
        );
      })}
    </span>
  );
}

function ReviewClaim({
  claim,
  evidenceIndex,
}: {
  claim: CuratorialBriefClaim;
  evidenceIndex: Record<string, CuratorialEvidenceReference>;
}) {
  return (
    <article className={styles.reviewClaim} data-confidence={claim.confidence}>
      <div>
        <span>{CURATORIAL_CONFIDENCE_LABELS[claim.confidence].zh}</span>
        <p>{claim.text}</p>
      </div>
      <ReviewEvidenceLinks evidenceIds={claim.evidenceIds} evidenceIndex={evidenceIndex} />
    </article>
  );
}

function CuratorialBriefReview({ exhibition }: { exhibition: Exhibition }) {
  const brief = exhibition.curatorialBrief;
  if (!brief) {
    return (
      <div className={styles.missingBrief} role="note">
        <strong>缺少策展任务书</strong>
        <span>这可能是旧版本展览；当前界面无法核对中心论点、排除理由与伦理状态。</span>
      </div>
    );
  }

  const evidenceIndex = buildCuratorialEvidenceIndex(exhibition.items);
  const provenanceNeedsReview = brief.ethics.provenanceStatus !== "documented";
  const sensitivityNeedsReview = brief.ethics.culturalSensitivityStatus !== "no_flags_after_review";
  const communityNeedsReview = !["not_required", "completed"].includes(brief.ethics.communityReviewStatus);
  const excludedCandidates = brief.excludedCandidates ?? [];

  return (
    <details className={styles.briefReview} open>
      <summary>
        <span>策展依据与风险</span>
        <small>
          {brief.objects.length} 件入选 · {excludedCandidates.length} 件排除 · {brief.keyMessages.length + 1} 条论点
        </small>
      </summary>
      <div className={styles.briefReviewBody}>
        <p className={styles.briefReviewBoundary}>
          以下论点、关系和风险状态由系统生成；引用只证明记录可定位，不代表语义支持已经由机构、专家或相关社群确认。
        </p>

        <section className={styles.claimReview} aria-label="论点与证据对应">
          <h4>论点 → 馆藏记录</h4>
          <ReviewClaim claim={brief.bigIdea} evidenceIndex={evidenceIndex} />
          {brief.keyMessages.map((claim) => (
            <ReviewClaim key={claim.id} claim={claim} evidenceIndex={evidenceIndex} />
          ))}
        </section>

        <div className={styles.briefReviewColumns}>
          <section>
            <h4>入选判断</h4>
            <ol className={styles.decisionList}>
              {brief.objects.map((decision) => {
                const item = exhibition.items.find(
                  (candidate) => candidate.id === decision.itemId || candidate.object.id === decision.objectId,
                );
                return (
                  <li key={`${decision.itemId}-${decision.objectId}`}>
                    <div>
                      <strong>{item?.displayTitle || item?.object.titleOriginal || item?.object.title || decision.objectId}</strong>
                      <span>{item?.roleLabel || decision.role}</span>
                    </div>
                    <p>{decision.selectionRationale}</p>
                    <small>动线关系：{decision.relation}</small>
                    <ReviewEvidenceLinks evidenceIds={decision.evidenceIds} evidenceIndex={evidenceIndex} />
                  </li>
                );
              })}
            </ol>

            <details className={styles.excludedReview}>
              <summary>查看全部排除候选（{excludedCandidates.length}）</summary>
              {excludedCandidates.length === 0 ? (
                <p>任务书没有记录排除候选。</p>
              ) : (
                <ol>
                  {excludedCandidates.map((candidate) => (
                    <li key={candidate.objectId}>
                      <strong>{candidate.title}</strong>
                      <p>{candidate.reason}</p>
                      <ReviewEvidenceLinks
                        evidenceIds={candidate.evidenceIds}
                        evidenceIndex={evidenceIndex}
                        unresolvedLabel="候选记录未随展览载入"
                      />
                    </li>
                  ))}
                </ol>
              )}
            </details>
          </section>

          <section className={styles.ethicsReview}>
            <h4>来源与伦理状态</h4>
            <dl>
              <div data-review-needed={provenanceNeedsReview ? "true" : "manual"}>
                <dt>来源史</dt>
                <dd>{PROVENANCE_STATUS_LABELS[brief.ethics.provenanceStatus].zh}</dd>
              </div>
              <div data-review-needed={sensitivityNeedsReview ? "true" : "manual"}>
                <dt>文化敏感性</dt>
                <dd>
                  {CULTURAL_SENSITIVITY_STATUS_LABELS[brief.ethics.culturalSensitivityStatus].zh}
                  {brief.ethics.culturalSensitivity.length > 0
                    ? `：${brief.ethics.culturalSensitivity.join("；")}`
                    : brief.ethics.culturalSensitivityStatus === "no_flags_after_review"
                      ? ""
                      : "；不能据此判断无风险"}
                </dd>
              </div>
              <div data-review-needed={communityNeedsReview ? "true" : "manual"}>
                <dt>相关社群审阅</dt>
                <dd>{COMMUNITY_REVIEW_STATUS_LABELS[brief.ethics.communityReviewStatus].zh}</dd>
              </div>
              <div>
                <dt>外部知识</dt>
                <dd>{brief.interpretationPolicy.externalKnowledgeAllowed ? "允许" : "未使用"}</dd>
              </div>
            </dl>

            {[...brief.ethics.provenanceNotes, ...brief.ethics.communityReviewNotes].length > 0 && (
              <ul className={styles.ethicsNotes}>
                {[...brief.ethics.provenanceNotes, ...brief.ethics.communityReviewNotes].map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
            )}

            <div className={styles.evaluationReview}>
              <h4>待验证目标</h4>
              {brief.evaluationTargets.length === 0 ? (
                <p>尚未记录访客或专家验证目标。</p>
              ) : (
                <ul>
                  {brief.evaluationTargets.map((target) => (
                    <li key={target.id}>
                      <span>{EVALUATION_METHOD_LABELS[target.method].zh}</span>
                      {target.statement}
                    </li>
                  ))}
                </ul>
              )}
              <small>这些是计划中的检查，不是已经完成的评估结果。</small>
            </div>
          </section>
        </div>

        <p className={styles.briefReviewVersion}>
          {brief.schemaVersion} · {brief.status} · 检索 {brief.retrieval.method}@{brief.retrieval.version} ·
          候选 {brief.retrieval.candidateCount} / 入选 {brief.retrieval.selectedCount} ·
          生成于 {new Date(brief.generatedAt).toLocaleString("zh-CN")}
        </p>
      </div>
    </details>
  );
}

async function loadDashboardData() {
  return Promise.all([getAnalytics(), listAdminExhibitions(), getDataAudit()]);
}

export function AdminDashboard() {
  const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null);
  const [exhibitions, setExhibitions] = useState<Exhibition[]>([]);
  const [dataAudit, setDataAudit] = useState<DataAuditReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [attested, setAttested] = useState<Record<string, boolean>>({});

  const load = useCallback(async () => {
    setError(null);
    try {
      const [nextAnalytics, nextExhibitions, nextDataAudit] = await loadDashboardData();
      setAnalytics(nextAnalytics);
      setExhibitions(nextExhibitions);
      setDataAudit(nextDataAudit);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "无法读取审阅数据。");
    }
  }, []);

  useEffect(() => {
    let active = true;

    loadDashboardData()
      .then(([nextAnalytics, nextExhibitions, nextDataAudit]) => {
        if (!active) return;
        setAnalytics(nextAnalytics);
        setExhibitions(nextExhibitions);
        setDataAudit(nextDataAudit);
      })
      .catch((loadError: unknown) => {
        if (!active) return;
        setError(loadError instanceof Error ? loadError.message : "无法读取审阅数据。");
      });

    return () => { active = false; };
  }, []);

  async function review(exhibition: Exhibition, action: "approve" | "reject") {
    setBusy(exhibition.id);
    setError(null);
    try {
      if (action === "approve") {
        await approveExhibition(exhibition.id, Boolean(attested[exhibition.id]));
      } else {
        await rejectExhibition(exhibition.id, "证据或策展连贯性需要人工修订");
      }
      await load();
    } catch (reviewError) {
      setError(reviewError instanceof Error ? reviewError.message : "审核操作失败。");
    } finally {
      setBusy(null);
    }
  }

  async function withdraw(exhibition: Exhibition) {
    const confirmed = window.confirm(
      `确定撤下“${exhibition.title}”吗？撤下后公开链接会失效；此操作只更新当前 Demo 的本地应用状态。`,
    );
    if (!confirmed) return;

    setBusy(exhibition.id);
    setError(null);
    try {
      await withdrawExhibition(exhibition.id);
      await load();
    } catch (withdrawError) {
      setError(withdrawError instanceof Error ? withdrawError.message : "撤下操作没有完成。");
    } finally {
      setBusy(null);
    }
  }

  async function downloadAnalyticsExport() {
    setExporting(true);
    setError(null);
    try {
      const payload = await getAnalyticsExport();
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "woyou-descriptive-statistics.json";
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch (exportError) {
      setError(exportError instanceof Error ? exportError.message : "描述性版本摘要导出失败。");
    } finally {
      setExporting(false);
    }
  }

  return (
    <div className={styles.adminPage}>
      <SiteHeader compact />
      <main className={styles.adminMain}>
        <header className={styles.adminHero}>
          <div>
            <h1>内部审阅</h1>
          </div>
          <p>公开前检查馆藏来源、AI 生成文本和发布状态。这里记录审阅结果，不判断参观者的学习成效。</p>
        </header>

        {error && <div className={styles.error} role="alert"><strong>后台连接失败</strong><span>{error}</span><button onClick={load}>重试</button></div>}

        <section className={styles.readiness} aria-label="数据就绪度">
          <article><span>候选对象</span><strong>{analytics?.collectionObjects ?? "—"}</strong><small>当前冻结馆藏</small></article>
          <article><span>已审对象</span><strong>{analytics?.reviewedObjects ?? "—"}</strong><small>已完成人工内容复核</small></article>
          <article><span>生成展览</span><strong>{analytics?.exhibitions ?? "—"}</strong><small>不等于学习成效</small></article>
          <article><span>已发布</span><strong>{analytics?.published ?? "—"}</strong><small>人工审核通过</small></article>
        </section>

        <section className={styles.auditSection} aria-labelledby="data-audit-title">
          <div className={styles.sectionTitle}>
            <h2 id="data-audit-title">冻结数据审计</h2>
            <button type="button" onClick={downloadAnalyticsExport} disabled={exporting}>
              {exporting ? "正在导出…" : "导出描述性版本摘要"}
            </button>
          </div>
          <div className={styles.auditBoundary} role="note">
            <strong>只读审计，不写回馆藏</strong>
            <p>这里仅汇总冻结数据中的已有字段与审核状态，不执行或记录人工证据核对；所有待审内容仍须由编辑逐项确认。</p>
            {dataAudit && <small>{dataAudit.schemaVersion} · 生成于 {new Date(dataAudit.generatedAt).toLocaleString("zh-CN")}</small>}
          </div>
          {!dataAudit ? (
            <div className={styles.auditLoading}>正在读取冻结数据审计…</div>
          ) : dataAudit.collections.length === 0 ? (
            <div className={styles.auditLoading}>当前没有可审计的冻结馆藏。</div>
          ) : (
            <div className={styles.auditGrid}>
              {dataAudit.collections.map((collection) => (
                <article className={styles.auditCard} key={collection.collectionId}>
                  <header className={styles.auditCardHeader}>
                    <div><span>{collection.institution}</span><h3>{collection.name}</h3></div>
                    <div><strong>{collection.version}</strong><small>{collection.collectionLicense}</small></div>
                  </header>
                  <div className={styles.auditMetrics}>
                    <div><span>冻结对象</span><strong>{collection.objectCount}</strong><small>运行时 {collection.runtimeObjectCount} · 排除 {collection.excludedRuntimeObjectCount}</small></div>
                    <div><span>证据覆盖域</span><strong>{Object.keys(collection.themeDistribution).length}</strong><small>内部检索约束，不是固定策展主题</small></div>
                    <div><span>证据待审</span><strong>{collection.evidence.pendingReviewChunks}</strong><small>{collection.evidence.objectsPendingReview} 件对象仍待人工核对</small></div>
                    <div><span>缺图</span><strong>{collection.assets.missingImageCount}</strong><small>冻结记录缺少图片链接</small></div>
                    <div><span>缺替代文本</span><strong>{collection.assets.missingAltTextCount}</strong><small>发布前必须补齐</small></div>
                    <div><span>问题卡</span><strong>{collection.questionCards.length}</strong><small>{distributionEntries(collection.questionCardCoverageStatusDistribution).map(([status, count]) => `${distributionLabel(status)} ${count}`).join(" · ")}</small></div>
                    <div><span>回归问题</span><strong>{collection.regression.questionCount}</strong><small>冻结预期状态分布见下方</small></div>
                  </div>
                  <div className={styles.auditDistributions}>
                    <div>
                      <h4>覆盖域分布（前 8 项）</h4>
                      <ul>
                        {distributionEntries(collection.themeDistribution, 8).map(([domain, count]) => (
                          <li key={domain}><span>{distributionLabel(domain)}</span><strong>{count}</strong></li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <h4>{collection.regression.questionCount} 题回归分布</h4>
                      <ul>
                        {distributionEntries(collection.regression.statusDistribution).map(([status, count]) => (
                          <li key={status}><span>{distributionLabel(status)}</span><strong>{count}</strong></li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>

        <section className={styles.reviewSection}>
          <div className={styles.sectionTitle}>
            <h2>展览审核队列</h2>
            <button onClick={load}>刷新</button>
          </div>
          {exhibitions.length === 0 ? (
            <div className={styles.emptyState}><strong>当前没有待处理展览</strong><p>访客完成生成、编辑和自动检查后，展览会出现在这里。</p><Link href="/">创建一场测试微展 →</Link></div>
          ) : (
            <div className={styles.reviewList}>
              {exhibitions.map((exhibition) => (
                <article key={exhibition.id}>
                  <div className={styles.statusRail} data-status={exhibition.status} />
                  <div className={styles.reviewContent}>
                    <div className={styles.reviewMeta}><span>{STATUS_LABELS[exhibition.status]}</span><span>{exhibition.updatedAt}</span></div>
                    <h3>{exhibition.title}</h3>
                    <p>{exhibition.curatorialThesis}</p>
                    <dl>
                      <div><dt>动态主题</dt><dd>{exhibition.exhibitionTheme || "由学习问题与证据链形成"}</dd></div>
                      <div><dt>角色</dt><dd>{exhibition.items.length}/5</dd></div>
                      <div><dt>首屏视觉</dt><dd>{POSTER_STATUS_LABELS[exhibition.poster?.status ?? "idle"]} · 不作为馆藏证据</dd></div>
                      <div><dt>自动检查</dt><dd>{exhibition.validation.passed ? `结构与引用完整性通过${exhibition.validation.warnings?.length ? `，${exhibition.validation.warnings.length} 项语义事实待人工核对` : "；语义事实仍以人工审核为准"}` : `${exhibition.validation.blockingIssues.length} 个阻断项`}</dd></div>
                      <div><dt>馆藏</dt><dd>{exhibition.versions.collection}</dd></div>
                    </dl>
                    {exhibition.validation.warnings && exhibition.validation.warnings.length > 0 && (
                      <ul className={styles.warningList} aria-label="待人工核对的检查警告">
                        {exhibition.validation.warnings.map((warning, index) => {
                          const objectTitle = warning.itemId
                            ? exhibition.items.find((item) => item.id === warning.itemId)?.object.title
                            : null;
                          return (
                            <li key={`${warning.code}-${warning.itemId ?? "exhibition"}-${index}`}>
                              <strong>{objectTitle ?? "整场展览"}</strong>
                              <span>{warning.message}</span>
                            </li>
                          );
                        })}
                      </ul>
                    )}
                    <CuratorialBriefReview exhibition={exhibition} />
                    {exhibition.review?.note && (
                      <div className={styles.reviewNote}>
                        <strong>{exhibition.review.decision === "rejected" ? "退回意见" : "审核备注"}</strong>
                        <p>{exhibition.review.note}</p>
                      </div>
                    )}
                    {exhibition.status === "review_pending" && (
                      <label className={styles.attestation}>
                        <input
                          type="checkbox"
                          checked={Boolean(attested[exhibition.id])}
                          onChange={(event) => setAttested((current) => ({ ...current, [exhibition.id]: event.target.checked }))}
                        />
                        <span>我已逐项打开本展所用来源，核对展签事实、证据对应关系、图像权利、AI 生成视觉披露与材料边界。</span>
                      </label>
                    )}
                  </div>
                  <div className={styles.reviewActions}>
                    <Link href={exhibition.status === "published" && exhibition.slug ? `/e/${exhibition.slug}` : `/exhibitions/${exhibition.id}`}>查看内容</Link>
                    {exhibition.status === "review_pending" && (
                      <>
                        <button onClick={() => review(exhibition, "reject")} disabled={busy === exhibition.id}>驳回</button>
                        <button className={styles.approve} onClick={() => review(exhibition, "approve")} disabled={busy === exhibition.id || !exhibition.validation.passed || !attested[exhibition.id]}>确认已核对并发布</button>
                      </>
                    )}
                    {exhibition.status === "published" && (
                      <button className={styles.withdraw} type="button" onClick={() => withdraw(exhibition)} disabled={busy === exhibition.id}>
                        {busy === exhibition.id ? "正在撤下…" : "撤下"}
                      </button>
                    )}
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>

        <section className={styles.analyticsBoundary}>
          <strong>分析边界</strong>
          <p>{analytics?.note ?? "只汇总生成、来源展开、编辑与分享事件；不推断个人偏好，也不把行为指标解释为学习成效。"}</p>
        </section>
      </main>
    </div>
  );
}
