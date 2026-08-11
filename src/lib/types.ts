export type PriorKnowledge = "none" | "some" | "familiar";
export type AnswerabilityStatus = "supported" | "partially_supported" | "unsupported";
export type SentenceType = "institution_fact" | "system_inference" | "uncertain";
export type ExhibitionStatus =
  | "draft"
  | "generating"
  | "ready"
  | "auto_validated"
  | "review_pending"
  | "published"
  | "rejected"
  | "withdrawn";

/** Falk (2009) identity-related visit motivations, reduced to four. */
export type VisitorMotivation = "explorer" | "recharger" | "facilitator" | "professional";

/** Véron & Levasseur (1983) circulation styles, derived from chosen duration. */
export type VisitorPace = "grasshopper" | "butterfly" | "ant";

export type InterviewQuestionId =
  | "curiosity"
  | "motivation"
  | "prior_knowledge"
  | "duration"
  | "negotiation"
  | "open_question"
  | "exclusions";

export interface InterviewOption {
  value: string;
  label: string;
  hint?: string | null;
}

export interface InterviewQuestion {
  id: InterviewQuestionId;
  prompt: string;
  options: InterviewOption[];
  allowFreeText: boolean;
  freeTextPlaceholder?: string | null;
  skippable: boolean;
  multiSelect: boolean;
  step: number;
  totalSteps: number;
}

export interface InterviewTurn {
  questionId: InterviewQuestionId;
  prompt: string;
  answerValue?: string | null;
  answerLabel?: string | null;
  freeText?: string | null;
  skipped: boolean;
  answeredAt: string;
  /** What the curator said back before moving on. Absent if the model was unavailable. */
  curatorReply?: string | null;
}

export interface VisitorProfile {
  curiosityDomainId?: string | null;
  curiosityLabel: string;
  freeFormQuestion?: string | null;
  openQuestion?: string | null;
  /** The language the exhibition was curated and written in. */
  language?: "zh" | "en";
  motivation: VisitorMotivation;
  priorKnowledge: PriorKnowledge;
  durationMinutes: 5 | 10 | 15;
  excludedTopics: string[];
  companion?: string | null;
}

export interface InterviewState {
  id: string;
  collectionId?: string | null;
  complete: boolean;
  profile: VisitorProfile;
  transcript: InterviewTurn[];
  nextQuestion?: InterviewQuestion | null;
  negotiationNote?: string | null;
}

export interface InterviewAnswerInput {
  questionId: InterviewQuestionId;
  value?: string | null;
  freeText?: string | null;
  skipped?: boolean;
}

export interface CollectionHighlight {
  id: string;
  title: string;
  altText: string;
  institution: string;
}

export interface CollectionDomain {
  id: string;
  label: string;
  hint: string;
  objectCount: number;
  samples: Array<{ id: string; title: string }>;
}

export interface CollectionInstitutionSummary {
  id: string;
  name: string;
  objectCount: number;
  sourceUrl: string | null;
  imageLicenses: string[];
  metadataLicenses: string[];
  curatorialTextLicenses: string[];
}

/** Sample of the live collection, used to build the landing hero and overview. */
export interface CollectionHighlights {
  collectionId: string;
  collectionVersion: string;
  objectCount: number;
  institutions: string[];
  institutionSummaries: CollectionInstitutionSummary[];
  /** Coverage domains the curator will actually offer, richest first. */
  domains: CollectionDomain[];
  topTypes: Array<{ label: string; count: number }>;
  datedObjectCount: number;
  items: CollectionHighlight[];
}

export type JobStepStatus = "pending" | "running" | "done" | "failed" | "skipped";

export interface JobStep {
  key: string;
  title: string;
  detail: string;
  status: JobStepStatus;
  /** One concrete sentence about what this step found. */
  finding?: string | null;
  startedAt?: string | null;
  finishedAt?: string | null;
}

export interface Chapter {
  id: string;
  order: number;
  title: string;
  leadIn: string;
  itemIds: string[];
  spaceHint?: string | null;
}

export interface Epilogue {
  text: string;
  openQuestions: string[];
  materialBoundary: string[];
}

export type EpilogueChatRole = "user" | "assistant";

/** Only the conversation text required by the model is sent back to the API. */
export interface EpilogueChatTurn {
  role: EpilogueChatRole;
  content: string;
}

export interface EpilogueChatCitation {
  itemId?: string | null;
  objectId?: string | null;
  evidenceIds?: string[];
  label?: string | null;
}

export interface EpilogueChatRequest {
  message: string;
  history: EpilogueChatTurn[];
  openQuestion?: string;
}

export interface EpilogueChatResponse {
  message: string;
  citations: EpilogueChatCitation[];
  suggestedReplies: string[];
  mode?: string | null;
  notice?: string | null;
}

/** Model-authored render parameters. The model never emits code. */
export interface SpaceDesignSpec {
  spaceForm: "hall" | "cloister" | "corridor" | "pavilion";
  wallColor: string;
  floorColor: string;
  accentColor: string;
  ceilingColor: string;
  lightTemperature: number;
  lightIntensity: number;
  mood: "warm_dim" | "neutral" | "cool_bright" | "dramatic";
  frameStyle: "thin_dark" | "wide_gold" | "scroll_hanging" | "vitrine";
  pacing: VisitorPace;
  typography: "serif" | "song" | "sans";
}

export interface AgendaInput {
  question: string;
  priorKnowledge: PriorKnowledge;
  durationMinutes: 5 | 10 | 15;
  personalConnection?: string;
  excludedTopics: string[];
  /** The language this exhibition was written in, fixed at generation time. */
  language?: "zh" | "en";
}

export interface AnswerabilityResult {
  status: AnswerabilityStatus;
  canGenerate?: boolean;
  exhibitionTheme?: string | null;
  supportedAspects: string[];
  coverageGaps: string[];
  recommendedQuestions: string[];
  evidenceRoles: string[];
  collectionVersion: string;
  coverage?: {
    collectionId: string;
    collectionName: string;
    eligibleObjectCount: number;
    matchedObjectCount: number;
    evidenceDomainIds: string[];
    candidateObjectIds: string[];
  };
}

export interface EvidenceChunk {
  id: string;
  text: string;
  sourceTitle: string;
  sourceUrl: string;
  sourceLocation: string;
  supports: string;
  reviewStatus: "reviewed" | "pending";
  license?: string | null;
  rightsUri?: string | null;
  sourceKind?: string | null;
}

export interface MuseumObject {
  id: string;
  accessionNumber: string;
  title: string;
  titleOriginal?: string;
  date: string;
  maker?: string;
  culture?: string;
  medium: string;
  type: string;
  imageUrl: string;
  imageUrlLarge?: string | null;
  objectUrl: string;
  rights: string;
  rightsUri?: string | null;
  imageLicense?: string | null;
  imageRightsUri?: string | null;
  metadataLicense?: string | null;
  metadataRightsUri?: string | null;
  curatorialTextLicense?: string | null;
  curatorialTextRightsUri?: string | null;
  altText: string;
  altTextSource?: string;
  description?: string;
  themes: string[];
  tags?: string[];
  evidence: EvidenceChunk[];
  institution?: string;
  institutionId?: string;
  classification?: string;
  creditLine?: string;
  /** `thin` objects carry metadata restatement only and never hold core evidence. */
  evidenceDepth?: "full" | "thin";
}

export interface MuseumObjectSummary {
  id: string;
  title: string;
  imageUrl: string;
  objectUrl: string;
  rights: string;
  rightsUri?: string | null;
  imageLicense?: string | null;
  imageRightsUri?: string | null;
  metadataLicense?: string | null;
  metadataRightsUri?: string | null;
  curatorialTextLicense?: string | null;
  curatorialTextRightsUri?: string | null;
  altText?: string;
  date?: string;
  medium?: string;
}

export interface LabelSentence {
  id: string;
  text: string;
  type: SentenceType;
  evidenceIds: string[];
}

export interface ExhibitionItem {
  id: string;
  object: MuseumObject;
  role: "opening" | "context" | "core_evidence" | "contrast" | "synthesis";
  roleLabel: string;
  /** Chinese title shown in the hall; English original stays on `object`. */
  displayTitle?: string;
  subQuestion: string;
  whySelected: string;
  relation: string;
  labelSentences: LabelSentence[];
  alternatives: MuseumObjectSummary[];
}

export type CuratorialBriefStatus = "deterministic" | "model_refined";
export type CuratorialClaimConfidence = "supported" | "provisional" | "uncertain";
export type CuratorialProvenanceStatus = "not_reviewed" | "unknown" | "partial" | "documented";
export type CuratorialSensitivityStatus =
  | "not_reviewed"
  | "unknown"
  | "no_flags_after_review"
  | "flags_present";
export type CuratorialCommunityReviewStatus =
  | "not_assessed"
  | "not_required"
  | "required"
  | "completed";

/** A claim in the internal curatorial argument, not an institution-authored fact. */
export interface CuratorialBriefClaim {
  id: string;
  text: string;
  evidenceIds: string[];
  confidence: CuratorialClaimConfidence;
}

export interface CuratorialBriefAudience {
  motivation: VisitorMotivation;
  priorKnowledge: PriorKnowledge;
  durationMinutes: 5 | 10 | 15;
  excludedTopics: string[];
  voice: string;
  density: string;
}

export interface CuratorialObjectDecision {
  itemId: string;
  objectId: string;
  role: ExhibitionItem["role"];
  selectionRationale: string;
  relation: string;
  evidenceIds: string[];
}

export interface CuratorialExcludedCandidate {
  objectId: string;
  title: string;
  reason: string;
  evidenceIds: string[];
}

export interface CuratorialEthicsStatus {
  provenanceStatus: CuratorialProvenanceStatus;
  provenanceNotes: string[];
  culturalSensitivityStatus: CuratorialSensitivityStatus;
  culturalSensitivity: string[];
  /** null means the need for source-community review has not been assessed. */
  communityReviewRequired: boolean | null;
  communityReviewStatus: CuratorialCommunityReviewStatus;
  communityReviewNotes: string[];
}

export interface CuratorialInterpretationPolicy {
  factPolicy: string;
  inferencePolicy: string;
  uncertaintyPolicy: string;
  externalKnowledgeAllowed: false;
}

export interface CuratorialEvaluationTarget {
  id: string;
  statement: string;
  method: "visitor_prompt" | "comprehension_check" | "expert_review";
}

export interface CuratorialRetrievalRecord {
  method: string;
  version: string;
  candidateCount: number;
  selectedCount: number;
}

/**
 * Machine-readable curatorial task record. It records the system's argument,
 * selection decisions and unresolved review state; it is never museum copy.
 */
export interface CuratorialBrief {
  schemaVersion: "curatorial-brief/v1";
  status: CuratorialBriefStatus;
  generatedAt: string;
  visitorInquiry: string;
  /** Redacted from the privacy-safe published projection. */
  audience?: CuratorialBriefAudience;
  bigIdea: CuratorialBriefClaim;
  keyMessages: CuratorialBriefClaim[];
  criticalQuestions: string[];
  objects: CuratorialObjectDecision[];
  /** Redacted from the privacy-safe published projection. */
  excludedCandidates?: CuratorialExcludedCandidate[];
  ethics: CuratorialEthicsStatus;
  interpretationPolicy: CuratorialInterpretationPolicy;
  evaluationTargets: CuratorialEvaluationTarget[];
  retrieval: CuratorialRetrievalRecord;
}

export interface ValidationResult {
  passed: boolean;
  checks: Array<{ key: string; label: string; passed: boolean; detail: string }>;
  blockingIssues: string[];
  warnings?: Array<{ code: string; message: string; itemId?: string }>;
  checkedAt: string;
}

export interface ReviewRecord {
  decision: "approved" | "rejected";
  reviewer: string;
  note?: string | null;
  evidenceReviewConfirmed: boolean;
  reviewedAt: string;
}

export interface ExhibitionPoster {
  status: "idle" | "generating" | "ready" | "failed";
  backgroundUrl?: string | null;
  altText?: string | null;
  promptSummary?: string | null;
  provider?: "aliyun-model-studio" | string | null;
  model?: "qwen-image-3.0-pro" | string | null;
  generatedBy?: string | null;
  size?: "1536*864" | string | null;
  generatedAt?: string | null;
  isAiGenerated: true;
  errorCode?: string | null;
}

export interface Exhibition {
  id: string;
  slug?: string;
  title: string;
  subtitle?: string;
  exhibitionTheme?: string | null;
  question: string;
  curatorialThesis: string;
  coreAnswer: string;
  subQuestions: string[];
  items: ExhibitionItem[];
  chapters: Chapter[];
  epilogue: Epilogue;
  spaceDesign: SpaceDesignSpec;
  visitorProfile?: VisitorProfile | null;
  /** Absent on exhibitions generated before curatorial-brief/v1. */
  curatorialBrief?: CuratorialBrief | null;
  coverageLimits: string[];
  status: ExhibitionStatus;
  versions: {
    model: string;
    /** Actual text-generation route; absent only on legacy client fixtures. */
    provider?: string;
    prompt: string;
    collection: string;
    validator: string;
  };
  validation: ValidationResult;
  review?: ReviewRecord | null;
  publicationReview?: { status: "approved"; reviewedAt: string };
  poster?: ExhibitionPoster | null;
  updatedAt: string;
}

export interface GenerationJob {
  id: string;
  status: "queued" | "running" | "completed" | "failed";
  progress: number;
  stage: string;
  steps: JobStep[];
  exhibitionId?: string | null;
  error?: string | null;
}

export interface AnalyticsSummary {
  collectionObjects: number;
  reviewedObjects: number;
  exhibitions: number;
  published: number;
  events: Record<string, number>;
  note: string;
}

export type CountDistribution = Record<string, number>;

export interface DataAuditEvidence {
  totalChunks: number;
  reviewedChunks: number;
  pendingReviewChunks: number;
  objectsWithEvidence: number;
  objectsWithoutEvidence: number;
  objectsFullyReviewed: number;
  objectsPendingReview: number;
}

export interface DataAuditAssets {
  missingImageCount: number;
  missingAltTextCount: number;
}

export interface DataAuditQuestionCard {
  id?: string | null;
  title?: string | null;
  question: string;
  coverageStatus: string;
  reviewStatus: string;
  themeId?: string | null;
  starterObjectCount: number;
  coverageLimits: string[];
}

export interface DataAuditRegression {
  questionCount: number;
  statusDistribution: CountDistribution;
  reviewStatusDistribution: CountDistribution;
}

export interface CollectionDataAudit {
  collectionId: string;
  name: string;
  institution: string;
  version: string;
  sourceUrl?: string | null;
  collectionLicense: string;
  objectCount: number;
  runtimeObjectCount: number;
  excludedRuntimeObjectCount: number;
  themeDistribution: CountDistribution;
  objectRightsDistribution: CountDistribution;
  evidence: DataAuditEvidence;
  assets: DataAuditAssets;
  questionCards: DataAuditQuestionCard[];
  questionCardCoverageStatusDistribution: CountDistribution;
  regression: DataAuditRegression;
}

export interface DataAuditReport {
  schemaVersion: string;
  readOnly: boolean;
  humanReviewRequired: boolean;
  generatedAt: string;
  collections: CollectionDataAudit[];
  reviewBoundary: string;
}

export interface DescriptiveStatisticsExport {
  schemaVersion: string;
  apiVersion: string;
  exportedAt: string;
  collections: Array<{ collectionId: string; version: string; objectCount: number }>;
  runtimeVersionDistributions: {
    provider: CountDistribution;
    model: CountDistribution;
    prompt: CountDistribution;
    collection: CountDistribution;
    validator: CountDistribution;
  };
  statistics: {
    totalEvents: number;
    eventCounts: CountDistribution;
    totalExhibitions: number;
    exhibitionStatusCounts: CountDistribution;
    publishedExhibitions: number;
    collectionObjects: number;
    reviewedObjects: number;
  };
  privacy: {
    aggregateOnly: boolean;
    rawSessionIdsIncluded: boolean;
    eventParametersIncluded: boolean;
    agendaFieldsIncluded: boolean;
    personalConnectionIncluded: boolean;
    excludedTopicsIncluded: boolean;
  };
  note: string;
}
