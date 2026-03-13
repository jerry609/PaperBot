export interface PaperBotOpenClawConfig {
  baseUrl: string;
  authToken?: string;
  defaultUserId: string;
  requestTimeoutMs: number;
  contextTrackId?: number;
  defaultSearchSources: string[];
  defaultSearchBranches: string[];
  enableContextInjection: boolean;
  cronQueries: string[];
  cronScholarId?: string;
}

export interface CronJobDefinition {
  id: string;
  expression: string;
  description: string;
  task: "paper-monitor" | "weekly-digest" | "conference-deadlines" | "citation-monitor";
  defaultInput: Record<string, unknown>;
}

export interface PaperSearchInput {
  query: string;
  sources?: string[];
  branches?: string[];
  topKPerQuery?: number;
  showPerBranch?: number;
  minScore?: number;
}

export interface PaperAnalyzeInput {
  title: string;
  abstract?: string;
  doi?: string;
}

export interface PaperTrackInput {
  scholarId?: string;
  scholarName?: string;
  force?: boolean;
  dryRun?: boolean;
  maxNewPapers?: number;
}

export interface GenCodeInput {
  title: string;
  abstract: string;
  methodSection?: string;
  userId?: string;
}

export interface ReviewInput {
  title: string;
  abstract: string;
}

export interface ResearchInput {
  query: string;
  userId?: string;
  trackId?: number;
  personalized?: boolean;
  paperLimit?: number;
  memoryLimit?: number;
}

export interface SseCollectionResult<TData = unknown> {
  result: TData | null;
  progress: Array<Record<string, unknown>>;
}

export type FetchLike = typeof fetch;

export function normalizeConfig(rawConfig: unknown): PaperBotOpenClawConfig {
  const config = rawConfig && typeof rawConfig === "object" ? (rawConfig as Record<string, unknown>) : {};

  return {
    baseUrl: stripTrailingSlashes(String(config.baseUrl ?? "http://127.0.0.1:8000")),
    authToken: String(config.authToken ?? "").trim() || undefined,
    defaultUserId: String(config.defaultUserId ?? "default").trim() || "default",
    requestTimeoutMs: Math.max(1000, Number(config.requestTimeoutMs ?? 30000)),
    contextTrackId:
      config.contextTrackId === undefined || config.contextTrackId === null
        ? undefined
        : Number(config.contextTrackId),
    defaultSearchSources: toStringList(config.defaultSearchSources, ["papers_cool"]),
    defaultSearchBranches: toStringList(config.defaultSearchBranches, ["arxiv", "venue"]),
    enableContextInjection: config.enableContextInjection !== false,
    cronQueries: toStringList(config.cronQueries, ["llm agents"]),
    cronScholarId: String(config.cronScholarId ?? "").trim() || undefined
  };
}

function toStringList(value: unknown, fallback: string[]): string[] {
  if (!Array.isArray(value)) {
    return [...fallback];
  }
  const items = value.map((item) => String(item ?? "").trim()).filter(Boolean);
  return items.length > 0 ? items : [...fallback];
}

function stripTrailingSlashes(value: string): string {
  let end = value.length;
  while (end > 0 && value.charCodeAt(end - 1) === 47) {
    end -= 1;
  }
  return value.slice(0, end);
}
