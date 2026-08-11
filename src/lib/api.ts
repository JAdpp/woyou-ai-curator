import type {
  AgendaInput,
  AnalyticsSummary,
  AnswerabilityResult,
  DataAuditReport,
  DescriptiveStatisticsExport,
  Exhibition,
  GenerationJob,
  CollectionHighlights,
  InterviewAnswerInput,
  InterviewState,
  EpilogueChatRequest,
  EpilogueChatResponse,
  EpilogueChatCitation,
} from "./types";
import { readStoredLanguage } from "./i18n";

const _envApiBase = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
export const API_BASE_URL = _envApiBase
  ? _envApiBase
  : // With no explicit base, server-side rendering talks to the FastAPI backend
    // directly on loopback while the browser uses a relative path so it rides the
    // same nginx origin it was served from.  A single baked absolute URL cannot
    // satisfy both, and an empty env value must fall through to this split rather
    // than short-circuit to "".
    typeof window === "undefined"
    ? "http://127.0.0.1:9001"
    : "";

export type AudioGuideKind = "lobby" | "chapter" | "artwork" | "epilogue";

export interface AudioGuideSegmentRequest {
  kind: AudioGuideKind;
  ref?: string;
}

const audioGuideCache = new Map<string, Promise<Blob>>();
const AUDIO_GUIDE_CACHE_LIMIT = 8;

export function resolveApiAssetUrl(path: string) {
  if (/^(?:https?:|data:|blob:)/i.test(path)) return path;
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

/**
 * Texture/image URL for an object, served through our own proxy.
 *
 * Institution CDNs do not all send CORS headers (Cleveland sends none), and a
 * WebGL texture cannot sample a tainted image. The proxy also downscales to a
 * texture-sized WebP, which is what keeps a twelve-object hall small enough to
 * load quickly.
 */
export function resolveObjectImageUrl(objectId: string, width: 512 | 1024 | 1536 = 1024) {
  return `${API_BASE_URL}/api/images/${encodeURIComponent(objectId)}?w=${width}`;
}

/** Absolute URL for one server-rendered Qwen narration segment. */
export function resolveAudioGuideUrl(
  exhibitionId: string,
  segment: AudioGuideSegmentRequest,
) {
  const parameters = new URLSearchParams({ kind: segment.kind });
  if (
    (segment.kind === "chapter" || segment.kind === "artwork") &&
    segment.ref?.trim()
  ) {
    parameters.set("ref", segment.ref);
  }
  return `${API_BASE_URL}/api/exhibitions/${encodeURIComponent(exhibitionId)}/audio-guide?${parameters.toString()}`;
}

/** Absolute endpoint for the optional post-visit conversation. */
export function resolveEpilogueChatUrl(exhibitionId: string) {
  return `${API_BASE_URL}/api/exhibitions/${encodeURIComponent(exhibitionId)}/epilogue-chat`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function formatValidationIssue(value: unknown): string | null {
  if (!isRecord(value)) return typeof value === "string" ? value : null;
  const message = typeof value.msg === "string"
    ? value.msg
    : typeof value.message === "string"
      ? value.message
      : null;
  if (!message) return null;
  const location = Array.isArray(value.loc)
    ? value.loc.filter((part): part is string | number => typeof part === "string" || typeof part === "number").join(".")
    : "";
  return location ? `${location}: ${message}` : message;
}

/* This module is not a component, so it reads the stored language directly
   rather than through the hook. Same key, same source of truth. */
function sep(): string {
  return readStoredLanguage() === "en" ? "; " : "；";
}

function unreadableReply(): string {
  return readStoredLanguage() === "en"
    ? "Yanyuan did not return a readable reply"
    : "彦远暂时没有返回可读的回应";
}

function formatDetail(detail: unknown): string | null {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const messages = detail.map(formatValidationIssue).filter((message): message is string => Boolean(message));
    return messages.length > 0 ? messages.join(sep()) : null;
  }
  if (!isRecord(detail)) return null;

  const parts: string[] = [];
  if (typeof detail.message === "string") parts.push(detail.message);
  if (Array.isArray(detail.blockingIssues)) {
    const blocking = detail.blockingIssues.filter((issue): issue is string => typeof issue === "string");
    if (blocking.length > 0) {
      parts.push(
        readStoredLanguage() === "en"
          ? `Blocking: ${blocking.join(sep())}`
          : `阻断项：${blocking.join(sep())}`,
      );
    }
  }
  return parts.length > 0 ? parts.join(" ") : null;
}

function apiErrorMessage(payload: unknown, status: number): string {
  if (isRecord(payload)) {
    const detail = formatDetail(payload.detail);
    if (detail) return detail;
    if (typeof payload.message === "string") return payload.message;
    if (isRecord(payload.error)) {
      const nested = formatDetail(payload.error);
      if (nested) return nested;
    }
  }
  return readStoredLanguage() === "en"
    ? `Request failed (${status})`
    : `请求失败（${status}）`;
}

class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function normalizeCitation(value: unknown): EpilogueChatCitation | null {
  if (!isRecord(value)) return null;
  const evidenceValue = Array.isArray(value.evidenceIds)
    ? value.evidenceIds
    : Array.isArray(value.evidence_ids)
      ? value.evidence_ids
      : [];
  const evidenceIds = evidenceValue.length > 0
    ? evidenceValue.filter((id): id is string => typeof id === "string" && id.trim().length > 0)
    : undefined;
  return {
    itemId: typeof value.itemId === "string"
      ? value.itemId
      : typeof value.item_id === "string"
        ? value.item_id
        : undefined,
    objectId: typeof value.objectId === "string"
      ? value.objectId
      : typeof value.object_id === "string"
        ? value.object_id
        : undefined,
    evidenceIds,
    label: typeof value.label === "string" ? value.label : undefined,
  };
}

/**
 * Keep the UI tolerant of a narrow set of server response aliases while the
 * endpoint evolves. The public return shape remains stable for components.
 */
export function normalizeEpilogueChatResponse(payload: unknown): EpilogueChatResponse {
  if (!isRecord(payload)) {
    throw new ApiError(unreadableReply(), 502);
  }
  const messageCandidates = [payload.message, payload.answer, payload.reply, payload.assistantMessage];
  const message = messageCandidates.find(
    (candidate): candidate is string => typeof candidate === "string" && candidate.trim().length > 0,
  )?.trim();
  if (!message) throw new ApiError(unreadableReply(), 502);

  const citations = Array.isArray(payload.citations)
    ? payload.citations.map(normalizeCitation).filter((citation): citation is EpilogueChatCitation => citation !== null)
    : [];
  const suggestedValue = Array.isArray(payload.suggestedReplies)
    ? payload.suggestedReplies
    : Array.isArray(payload.suggestedPrompts)
      ? payload.suggestedPrompts
      : Array.isArray(payload.suggested_prompts)
        ? payload.suggested_prompts
        : Array.isArray(payload.suggestions)
          ? payload.suggestions
          : [];
  const suggestedReplies = suggestedValue.filter(
    (reply): reply is string => typeof reply === "string" && reply.trim().length > 0,
  ).map((reply) => reply.trim());

  return {
    message,
    citations,
    suggestedReplies,
    mode: typeof payload.mode === "string" ? payload.mode : undefined,
    notice: typeof payload.notice === "string" && payload.notice.trim()
      ? payload.notice.trim()
      : undefined,
  };
}

async function requestAudioGuideBlob(url: string, signal?: AbortSignal) {
  const response = await fetch(url, {
    cache: "force-cache",
    headers: { Accept: "audio/mpeg" },
    signal,
  });

  if (!response.ok) {
    const payload: unknown = await response.json().catch(() => null);
    throw new ApiError(apiErrorMessage(payload, response.status), response.status);
  }

  const contentType = response.headers.get("content-type")?.toLowerCase() ?? "";
  if (!contentType.startsWith("audio/")) {
    throw new ApiError(
      readStoredLanguage() === "en"
        ? "The speech service returned something that cannot be played"
        : "语音服务返回了无法播放的内容",
      502,
    );
  }
  return response.blob();
}

/**
 * Load one narration segment, sharing the in-flight request with prefetches.
 * The small cache avoids holding an entire exhibition's audio in memory.
 */
export function getAudioGuideBlob(
  exhibitionId: string,
  segment: AudioGuideSegmentRequest,
  signal?: AbortSignal,
) {
  const url = resolveAudioGuideUrl(exhibitionId, segment);
  const cached = audioGuideCache.get(url);
  if (cached) return cached;

  if (audioGuideCache.size >= AUDIO_GUIDE_CACHE_LIMIT) {
    const oldest = audioGuideCache.keys().next().value as string | undefined;
    if (oldest) audioGuideCache.delete(oldest);
  }

  const request = requestAudioGuideBlob(url, signal).catch((error: unknown) => {
    audioGuideCache.delete(url);
    throw error;
  });
  audioGuideCache.set(url, request);
  return request;
}

/** Warm the next stop without delaying the current narration. */
export function prefetchAudioGuide(
  exhibitionId: string,
  segment: AudioGuideSegmentRequest | null,
) {
  if (!segment) return;
  void getAudioGuideBlob(exhibitionId, segment).catch(() => {
    // A prefetch failure is deliberately silent. The foreground request will
    // retry and can then fall back to the device voice with visible feedback.
  });
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  if (!response.ok) {
    const payload: unknown = await response.json().catch(() => null);
    throw new ApiError(apiErrorMessage(payload, response.status), response.status);
  }

  return response.json() as Promise<T>;
}

export function checkAgenda(agenda: AgendaInput) {
  return request<AnswerabilityResult>("/api/agenda/check", {
    method: "POST",
    body: JSON.stringify(agenda),
  });
}

export function getCollectionHighlights(limit = 24, language?: "zh" | "en") {
  const params = new URLSearchParams({ limit: String(limit) });
  // Domain labels come from the collection manifest in Chinese; the server maps
  // them by id when asked for English.
  if (language) params.set("language", language);
  return request<CollectionHighlights>(`/api/collection/highlights?${params}`);
}


/** `language` is fixed for the whole interview and rides on the profile from
 *  there into the exhibition record, so the labels are written in it too. */
export function startInterview(language?: "zh" | "en", collectionId?: string) {
  const params = new URLSearchParams();
  if (collectionId) params.set("collectionId", collectionId);
  if (language) params.set("language", language);
  const query = params.size > 0 ? `?${params}` : "";
  return request<InterviewState>(`/api/interview/start${query}`, { method: "POST" });
}

export function answerInterview(interviewId: string, answer: InterviewAnswerInput) {
  return request<InterviewState>(`/api/interview/${interviewId}/answer`, {
    method: "POST",
    body: JSON.stringify(answer),
  });
}

export function startCuration(interviewId: string, collectionId?: string) {
  return request<GenerationJob>("/api/exhibitions/generate", {
    method: "POST",
    body: JSON.stringify({ interviewId, collectionId }),
  });
}

export function getJob(id: string) {
  return request<GenerationJob>(`/api/jobs/${id}`, { cache: "no-store" });
}

/**
 * Follow a curation job.
 *
 * Prefers SSE so each pipeline step lands as it happens. Some proxies buffer
 * event streams into one chunk at the end, which would defeat the visible
 * todo list entirely, so a polling fallback takes over if the stream errors or
 * never delivers a first message.
 */
export function followJob(
  jobId: string,
  onUpdate: (job: GenerationJob) => void,
): () => void {
  let stopped = false;
  let source: EventSource | null = null;
  let pollTimer: number | null = null;

  const stop = () => {
    stopped = true;
    source?.close();
    if (pollTimer !== null) window.clearTimeout(pollTimer);
  };

  const poll = async () => {
    if (stopped) return;
    try {
      const job = await getJob(jobId);
      onUpdate(job);
      if (job.status === "completed" || job.status === "failed") return;
    } catch {
      // Transient failures are expected while the job is starting up.
    }
    pollTimer = window.setTimeout(poll, 1200);
  };

  let receivedFirstMessage = false;
  try {
    source = new EventSource(`${API_BASE_URL}/api/jobs/${jobId}/stream`);
    source.onmessage = (event) => {
      receivedFirstMessage = true;
      try {
        const job = JSON.parse(event.data) as GenerationJob;
        onUpdate(job);
        if (job.status === "completed" || job.status === "failed") stop();
      } catch {
        // Ignore a malformed frame; the next one usually arrives fine.
      }
    };
    source.onerror = () => {
      source?.close();
      source = null;
      if (!stopped) void poll();
    };
    // If the stream produces nothing quickly, assume it is being buffered.
    window.setTimeout(() => {
      if (!receivedFirstMessage && !stopped) {
        source?.close();
        source = null;
        void poll();
      }
    }, 3000);
  } catch {
    void poll();
  }

  return stop;
}

export function getExhibition(id: string) {
  return request<Exhibition>(`/api/exhibitions/${id}`, { cache: "no-store" });
}

export function getPublishedExhibition(slug: string) {
  return request<Exhibition>(`/api/public/exhibitions/${slug}`, { cache: "no-store" });
}

export async function sendEpilogueChatMessage(
  exhibitionId: string,
  input: EpilogueChatRequest,
  signal?: AbortSignal,
) {
  const payload = await request<unknown>(
    `/api/exhibitions/${encodeURIComponent(exhibitionId)}/epilogue-chat`,
    {
      method: "POST",
      body: JSON.stringify(input),
      signal,
    },
  );
  return normalizeEpilogueChatResponse(payload);
}

export function replaceItem(exhibitionId: string, itemId: string, objectId: string) {
  return request<Exhibition>(`/api/exhibitions/${exhibitionId}/items`, {
    method: "PATCH",
    body: JSON.stringify({ action: "replace", itemId, objectId }),
  });
}

export function reorderItem(exhibitionId: string, itemId: string, direction: "up" | "down") {
  return request<Exhibition>(`/api/exhibitions/${exhibitionId}/items`, {
    method: "PATCH",
    body: JSON.stringify({ action: "reorder", itemId, direction }),
  });
}

export function updateFocus(exhibitionId: string, focus: string) {
  return request<Exhibition>(`/api/exhibitions/${exhibitionId}/focus`, {
    method: "PATCH",
    body: JSON.stringify({ focus }),
  });
}

export function validateExhibition(exhibitionId: string) {
  return request<Exhibition>(`/api/exhibitions/${exhibitionId}/validate`, { method: "POST" });
}

export function publishExhibition(exhibitionId: string) {
  return request<Exhibition>(`/api/exhibitions/${exhibitionId}/publish`, { method: "POST" });
}

export function generateExhibitionPoster(exhibitionId: string, force = false) {
  const query = force ? "?force=true" : "";
  return request<Exhibition>(`/api/exhibitions/${exhibitionId}/poster${query}`, { method: "POST" });
}

/**
 * `crypto.randomUUID` is gated to secure contexts, so it is missing when the
 * demo is served over plain HTTP from an IP address. `getRandomValues` has no
 * such gate, so build the v4 ourselves rather than throwing out of `logEvent`.
 */
function randomSessionId(): string {
  if (typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  bytes[6] = ((bytes[6] ?? 0) & 0x0f) | 0x40; // version 4
  bytes[8] = ((bytes[8] ?? 0) & 0x3f) | 0x80; // variant 10
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

export function logEvent(event: string, exhibitionId?: string, properties: Record<string, unknown> = {}) {
  let sessionId = "server-unavailable";
  if (typeof window !== "undefined") {
    sessionId = window.sessionStorage.getItem("inquiry-curator-session") ?? randomSessionId();
    window.sessionStorage.setItem("inquiry-curator-session", sessionId);
  }
  return request<{ accepted: boolean }>("/api/events", {
    method: "POST",
    body: JSON.stringify({ sessionId, event, exhibitionId, parameters: properties }),
  }).catch(() => ({ accepted: false }));
}

export function getAnalytics() {
  return request<AnalyticsSummary>("/api/admin/analytics", { cache: "no-store" });
}

export function getDataAudit() {
  return request<DataAuditReport>("/api/admin/data-audit", { cache: "no-store" });
}

export function getAnalyticsExport() {
  return request<DescriptiveStatisticsExport>("/api/admin/analytics/export", { cache: "no-store" });
}

export function listAdminExhibitions() {
  return request<Exhibition[] | { items: Exhibition[] }>("/api/admin/exhibitions", { cache: "no-store" })
    .then((payload) => Array.isArray(payload) ? payload : payload.items);
}

export function approveExhibition(exhibitionId: string, evidenceReviewConfirmed: boolean) {
  return request<Exhibition>(`/api/admin/exhibitions/${exhibitionId}/approve`, {
    method: "POST",
    body: JSON.stringify({
      reviewer: "demo-admin",
      note: "已在审阅面板逐项核对所选藏品、展签与来源。",
      evidenceReviewConfirmed,
    }),
  });
}

export function rejectExhibition(exhibitionId: string, reason: string) {
  return request<Exhibition>(`/api/admin/exhibitions/${exhibitionId}/reject`, {
    method: "POST",
    body: JSON.stringify({ reviewer: "demo-admin", note: reason }),
  });
}

export function withdrawExhibition(exhibitionId: string) {
  return request<Exhibition>(`/api/admin/exhibitions/${exhibitionId}/withdraw`, {
    method: "POST",
  });
}

export { ApiError };
