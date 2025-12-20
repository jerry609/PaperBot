/**
 * API Client for PaperBot Python backend
 * Supports both REST and Server-Sent Events (SSE) for streaming
 */

export interface ApiConfig {
  baseUrl: string;
  timeout: number;
}

export interface ScholarTrackRequest {
  scholarId?: string;
  scholarName?: string;
  force?: boolean;
}

export interface PaperAnalyzeRequest {
  title: string;
  abstract?: string;
  doi?: string;
}

export interface GenCodeRequest {
  title: string;
  abstract: string;
  methodSection?: string;
  useOrchestrator?: boolean;
  useRag?: boolean;
}

export interface ReviewRequest {
  title: string;
  abstract: string;
}

export interface StreamEvent {
  type: 'progress' | 'result' | 'error' | 'done';
  data: unknown;
  message?: string;
}

const DEFAULT_CONFIG: ApiConfig = {
  baseUrl: process.env['PAPERBOT_API_URL'] || 'http://localhost:8000',
  timeout: 300000, // 5 minutes
};

export class PaperBotClient {
  private config: ApiConfig;

  constructor(config: Partial<ApiConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Health check
   */
  async health(): Promise<boolean> {
    try {
      const response = await fetch(`${this.config.baseUrl}/health`, {
        signal: AbortSignal.timeout(5000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Track scholars with streaming progress
   */
  async *trackScholar(request: ScholarTrackRequest): AsyncGenerator<StreamEvent> {
    const params = new URLSearchParams();
    if (request.scholarId) params.set('scholar_id', request.scholarId);
    if (request.scholarName) params.set('scholar_name', request.scholarName);
    if (request.force) params.set('force', 'true');

    yield* this.streamRequest(`/api/track?${params.toString()}`);
  }

  /**
   * Analyze paper with streaming
   */
  async *analyzePaper(request: PaperAnalyzeRequest): AsyncGenerator<StreamEvent> {
    yield* this.streamRequest('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
  }

  /**
   * Generate code from paper with streaming
   */
  async *generateCode(request: GenCodeRequest): AsyncGenerator<StreamEvent> {
    yield* this.streamRequest('/api/gen-code', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: request.title,
        abstract: request.abstract,
        method_section: request.methodSection,
        use_orchestrator: request.useOrchestrator ?? true,
        use_rag: request.useRag ?? true,
      }),
    });
  }

  /**
   * Deep review paper
   */
  async *reviewPaper(request: ReviewRequest): AsyncGenerator<StreamEvent> {
    yield* this.streamRequest('/api/review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
  }

  /**
   * Chat with AI about papers
   */
  async *chat(message: string, history: Array<{role: string; content: string}>): AsyncGenerator<StreamEvent> {
    yield* this.streamRequest('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, history }),
    });
  }

  /**
   * Generic streaming request using SSE
   */
  private async *streamRequest(
    path: string,
    init: RequestInit = {}
  ): AsyncGenerator<StreamEvent> {
    const url = `${this.config.baseUrl}${path}`;

    try {
      const response = await fetch(url, {
        ...init,
        headers: {
          ...init.headers,
          'Accept': 'text/event-stream',
        },
        signal: AbortSignal.timeout(this.config.timeout),
      });

      if (!response.ok) {
        yield {
          type: 'error',
          data: null,
          message: `HTTP ${response.status}: ${response.statusText}`,
        };
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        yield { type: 'error', data: null, message: 'No response body' };
        return;
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') {
              yield { type: 'done', data: null };
              return;
            }
            try {
              const parsed = JSON.parse(data);
              yield {
                type: parsed.type || 'progress',
                data: parsed.data || parsed,
                message: parsed.message,
              };
            } catch {
              // Skip invalid JSON
            }
          }
        }
      }

      yield { type: 'done', data: null };
    } catch (error) {
      yield {
        type: 'error',
        data: null,
        message: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Non-streaming request
   */
  async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const url = `${this.config.baseUrl}${path}`;
    const response = await fetch(url, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        ...init.headers,
      },
      signal: AbortSignal.timeout(this.config.timeout),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }
}

// Singleton instance
export const client = new PaperBotClient();
