import type {
  FetchLike,
  GenCodeInput,
  PaperAnalyzeInput,
  PaperBotOpenClawConfig,
  PaperSearchInput,
  PaperTrackInput,
  ResearchInput,
  ReviewInput,
  SseCollectionResult
} from "./types.js";
import { normalizeConfig } from "./types.js";

export { normalizeConfig } from "./types.js";

export class PaperBotClient {
  private readonly config: PaperBotOpenClawConfig;
  private readonly fetchImpl: FetchLike;

  constructor(config: PaperBotOpenClawConfig, fetchImpl: FetchLike = fetch) {
    this.config = config;
    this.fetchImpl = fetchImpl;
  }

  async searchPapers(input: PaperSearchInput): Promise<unknown> {
    return this.requestJson("/api/research/paperscool/search", {
      method: "POST",
      body: JSON.stringify({
        queries: [input.query],
        sources: input.sources ?? this.config.defaultSearchSources,
        branches: input.branches ?? this.config.defaultSearchBranches,
        top_k_per_query: input.topKPerQuery ?? 5,
        show_per_branch: input.showPerBranch ?? 25,
        min_score: input.minScore ?? 0
      })
    });
  }

  async analyzePaper(input: PaperAnalyzeInput): Promise<unknown> {
    const response = await this.requestSse("/api/analyze", {
      method: "POST",
      body: JSON.stringify({
        title: input.title,
        abstract: input.abstract ?? "",
        doi: input.doi ?? null
      })
    });
    return response.result ?? response.progress.at(-1) ?? {};
  }

  async trackScholar(input: PaperTrackInput): Promise<unknown> {
    const params = new URLSearchParams();
    if (input.scholarId) {
      params.set("scholar_id", input.scholarId);
    }
    if (input.scholarName) {
      params.set("scholar_name", input.scholarName);
    }
    if (input.force) {
      params.set("force", "true");
    }
    if (input.dryRun) {
      params.set("dry_run", "true");
    }
    params.set("max_new_papers", String(input.maxNewPapers ?? 5));
    const response = await this.requestSse(`/api/track?${params.toString()}`, {
      method: "GET"
    });
    return response.result ?? response.progress.at(-1) ?? {};
  }

  async generateCode(input: GenCodeInput): Promise<unknown> {
    const response = await this.requestSse("/api/gen-code", {
      method: "POST",
      body: JSON.stringify({
        user_id: input.userId ?? this.config.defaultUserId,
        title: input.title,
        abstract: input.abstract,
        method_section: input.methodSection ?? null
      })
    });
    return response.result ?? response.progress.at(-1) ?? {};
  }

  async reviewPaper(input: ReviewInput): Promise<unknown> {
    const response = await this.requestSse("/api/review", {
      method: "POST",
      body: JSON.stringify(input)
    });
    return response.result ?? response.progress.at(-1) ?? {};
  }

  async buildResearchContext(input: ResearchInput): Promise<unknown> {
    return this.requestJson("/api/research/context", {
      method: "POST",
      body: JSON.stringify({
        query: input.query,
        user_id: input.userId ?? this.config.defaultUserId,
        track_id: input.trackId ?? this.config.contextTrackId ?? null,
        personalized: input.personalized ?? true,
        paper_limit: input.paperLimit ?? 5,
        memory_limit: input.memoryLimit ?? 8
      })
    });
  }

  async buildWeeklyDigest(input: { queries: string[] }): Promise<unknown> {
    return this.requestJson("/api/research/paperscool/daily", {
      method: "POST",
      body: JSON.stringify({
        queries: input.queries,
        notify: false,
        save: false
      })
    });
  }

  async getDeadlineRadar(): Promise<unknown> {
    return this.requestJson("/api/research/deadlines/radar", {
      method: "GET"
    });
  }

  async monitorCitationMilestones(input: { queries: string[] }): Promise<unknown> {
    return this.requestJson("/api/papers/search", {
      method: "POST",
      body: JSON.stringify({
        query: input.queries.join(" "),
        limit: 10,
        sort_by: "citation_count",
        sort_order: "desc"
      })
    });
  }

  private async requestJson(path: string, init: RequestInit): Promise<unknown> {
    const response = await this.fetchWithTimeout(path, init);
    const text = await response.text();
    if (!response.ok) {
      throw new Error(text || `PaperBot request failed with ${response.status}`);
    }
    return text ? JSON.parse(text) : {};
  }

  private async requestSse(path: string, init: RequestInit): Promise<SseCollectionResult> {
    const response = await this.fetchWithTimeout(path, init);
    const text = await response.text();
    if (!response.ok) {
      throw new Error(text || `PaperBot SSE request failed with ${response.status}`);
    }
    return collectSseResult(text);
  }

  private async fetchWithTimeout(path: string, init: RequestInit): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.requestTimeoutMs);
    try {
      return await this.fetchImpl(`${this.config.baseUrl}${path}`, {
        ...init,
        headers: {
          "content-type": "application/json",
          ...authHeaders(this.config),
          ...(init.headers ?? {})
        },
        signal: controller.signal
      });
    } finally {
      clearTimeout(timeoutId);
    }
  }
}

export function createPaperBotClient(
  rawConfig: unknown,
  fetchImpl: FetchLike = fetch
): PaperBotClient {
  return new PaperBotClient(normalizeConfig(rawConfig), fetchImpl);
}

export function collectSseResult(text: string): SseCollectionResult {
  const progress: Array<Record<string, unknown>> = [];
  let result: unknown = null;

  for (const chunk of text.split(/\n\n+/)) {
    const trimmed = chunk.trim();
    if (!trimmed) {
      continue;
    }
    const dataLines = trimmed
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim());
    for (const dataLine of dataLines) {
      if (!dataLine || dataLine === "[DONE]") {
        continue;
      }
      const payload = JSON.parse(dataLine) as {
        type?: string;
        data?: Record<string, unknown>;
        message?: string;
      };
      if (payload.type === "error") {
        throw new Error(payload.message ?? "PaperBot SSE stream failed");
      }
      if (payload.type === "result") {
        result = payload.data ?? {};
      } else if (payload.data && typeof payload.data === "object") {
        progress.push(payload.data);
      }
    }
  }

  return {
    result: (result as Record<string, unknown> | null) ?? null,
    progress
  };
}

function authHeaders(config: PaperBotOpenClawConfig): HeadersInit {
  if (!config.authToken) {
    return {};
  }
  return {
    authorization: `Bearer ${config.authToken}`
  };
}
