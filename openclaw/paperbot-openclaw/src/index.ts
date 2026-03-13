import type {
  BeforePromptBuildEvent,
  BeforePromptBuildResult,
  OpenClawPluginApi,
  ToolDefinition,
  ToolResult
} from "openclaw/plugin-sdk/core";

import { DEFAULT_PAPERBOT_CRON_JOBS, resolveCronJobs } from "./cron.js";
import { detectPaperIntent, latestUserMessage } from "./intents.js";
import { createPaperBotClient } from "./paperbot-client.js";
import type {
  FetchLike,
  GenCodeInput,
  PaperAnalyzeInput,
  PaperSearchInput,
  PaperTrackInput,
  ResearchInput,
  ReviewInput
} from "./types.js";
import { normalizeConfig } from "./types.js";

function textResult(payload: unknown): ToolResult {
  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(payload, null, 2)
      }
    ]
  };
}

function toolErrorResult(error: unknown): ToolResult {
  return textResult({
    ok: false,
    error: error instanceof Error ? error.message : String(error)
  });
}

function createTool<TInput extends object>(
  name: string,
  description: string,
  parameters: Record<string, unknown>,
  execute: (input: TInput) => Promise<unknown>
): ToolDefinition<TInput> {
  return {
    name,
    description,
    parameters,
    async execute(_invocationId: string, input: TInput): Promise<ToolResult> {
      try {
        const payload = await execute(input);
        return textResult(payload);
      } catch (error) {
        return toolErrorResult(error);
      }
    }
  };
}

function summarizeResearchContext(payload: unknown): string {
  const context = payload && typeof payload === "object" ? (payload as Record<string, unknown>) : {};
  const memories = Array.isArray(context.memory_items) ? context.memory_items.length : 0;
  const papers = Array.isArray(context.paper_recommendations)
    ? context.paper_recommendations.length
    : 0;
  return [
    "PaperBot context bridge:",
    `- memories: ${memories}`,
    `- paper recommendations: ${papers}`,
    `- raw payload: ${JSON.stringify(context)}`
  ].join("\n");
}

export function createPaperBotOpenClawPlugin(options: { fetchImpl?: FetchLike } = {}) {
  return {
    id: "paperbot-openclaw",
    name: "paperbot-openclaw",
    version: "0.1.0",
    async register(api: OpenClawPluginApi): Promise<void> {
      const config = normalizeConfig(api.config);
      const client = createPaperBotClient(config, options.fetchImpl);

      api.registerTool(
        createTool<PaperSearchInput>(
          "paper_search",
          "Run multi-source paper discovery through PaperBot.",
          {
            type: "object",
            properties: {
              query: { type: "string", minLength: 1 },
              sources: { type: "array", items: { type: "string" } },
              branches: { type: "array", items: { type: "string" } },
              topKPerQuery: { type: "integer", minimum: 1, maximum: 50 },
              showPerBranch: { type: "integer", minimum: 1, maximum: 200 },
              minScore: { type: "number", minimum: 0 }
            },
            required: ["query"],
            additionalProperties: false
          },
          async (input: PaperSearchInput) => client.searchPapers(input)
        )
      );

      api.registerTool(
        createTool<PaperAnalyzeInput>(
          "paper_analyze",
          "Stream PaperBot paper analysis and return the final result event.",
          {
            type: "object",
            properties: {
              title: { type: "string", minLength: 1 },
              abstract: { type: "string" },
              doi: { type: "string" }
            },
            required: ["title"],
            additionalProperties: false
          },
          async (input: PaperAnalyzeInput) => client.analyzePaper(input)
        )
      );

      api.registerTool(
        createTool<PaperTrackInput>(
          "paper_track",
          "Track a scholar through PaperBot and return the latest tracking summary.",
          {
            type: "object",
            properties: {
              scholarId: { type: "string" },
              scholarName: { type: "string" },
              force: { type: "boolean" },
              dryRun: { type: "boolean" },
              maxNewPapers: { type: "integer", minimum: 1, maximum: 20 }
            },
            additionalProperties: false
          },
          async (input: PaperTrackInput) => client.trackScholar(input)
        )
      );

      api.registerTool(
        createTool<GenCodeInput>(
          "gen_code",
          "Run Paper2Code generation and return the final generation payload.",
          {
            type: "object",
            properties: {
              title: { type: "string", minLength: 1 },
              abstract: { type: "string", minLength: 1 },
              methodSection: { type: "string" },
              userId: { type: "string" }
            },
            required: ["title", "abstract"],
            additionalProperties: false
          },
          async (input: GenCodeInput) => client.generateCode(input)
        )
      );

      api.registerTool(
        createTool<ReviewInput>(
          "review",
          "Run PaperBot's deep review flow and return the review result.",
          {
            type: "object",
            properties: {
              title: { type: "string", minLength: 1 },
              abstract: { type: "string", minLength: 1 }
            },
            required: ["title", "abstract"],
            additionalProperties: false
          },
          async (input: ReviewInput) => client.reviewPaper(input)
        )
      );

      api.registerTool(
        createTool<ResearchInput>(
          "research",
          "Build a research context pack for the current query.",
          {
            type: "object",
            properties: {
              query: { type: "string", minLength: 1 },
              userId: { type: "string" },
              trackId: { type: "integer", minimum: 1 },
              personalized: { type: "boolean" },
              paperLimit: { type: "integer", minimum: 1, maximum: 20 },
              memoryLimit: { type: "integer", minimum: 1, maximum: 20 }
            },
            required: ["query"],
            additionalProperties: false
          },
          async (input: ResearchInput) => client.buildResearchContext(input)
        )
      );

      const messageHook = async (payload: unknown) => {
        const content = extractHookText(payload);
        const suggestedTool = detectPaperIntent(content);
        if (!suggestedTool) {
          return null;
        }
        return {
          suggestedTool,
          confidence: 0.7,
          reason: `matched PaperBot intent in message: ${content.slice(0, 80)}`
        };
      };

      api.registerHook("msg_recv", messageHook, {
        name: "paperbot-msg-router",
        description: "Map paper-related user messages to the right PaperBot tool."
      });
      api.registerHook("message_received", messageHook, {
        name: "paperbot-message-router",
        description: "OpenClaw runtime hook alias for PaperBot message routing."
      });

      const beforePromptHook = async (
        event: BeforePromptBuildEvent
      ): Promise<BeforePromptBuildResult> => {
        if (!config.enableContextInjection) {
          return {};
        }
        const query = latestUserMessage(event.messages);
        if (!query || !detectPaperIntent(query)) {
          return {};
        }
        try {
          const context = await client.buildResearchContext({
            query,
            userId: config.defaultUserId,
            trackId: config.contextTrackId,
            personalized: true,
            paperLimit: 5,
            memoryLimit: 8
          });
          return {
            prependSystemContext: summarizeResearchContext(context)
          };
        } catch (error) {
          api.logger.warn("paperbot-openclaw context injection skipped", error);
          return {};
        }
      };

      api.registerHook(
        "before_prompt",
        async (payload) =>
          beforePromptHook({
            messages:
              payload && typeof payload === "object" && "messages" in payload
                ? ((payload as { messages?: BeforePromptBuildEvent["messages"] }).messages ?? [])
                : []
          }),
        {
          name: "paperbot-before-prompt",
          description: "Inject compact research context from PaperBot before prompt construction."
        }
      );
      api.on("before_prompt_build", beforePromptHook, { priority: 20 });

      api.registerCli(
        ({ program }) => {
          const paper = program.command("paper").description("PaperBot bridge commands");

          paper
            .command("search")
            .description("Search papers through PaperBot")
            .option("--query <query>", "Search query")
            .action(async (options) => {
              const payload = await client.searchPapers({
                query: String(options.query ?? "")
              });
              console.log(JSON.stringify(payload, null, 2));
            });

          paper
            .command("analyze")
            .description("Analyze a paper through PaperBot")
            .option("--title <title>", "Paper title")
            .option("--abstract <abstract>", "Paper abstract")
            .action(async (options) => {
              const payload = await client.analyzePaper({
                title: String(options.title ?? ""),
                abstract: String(options.abstract ?? "")
              });
              console.log(JSON.stringify(payload, null, 2));
            });

          paper
            .command("track")
            .description("Track a scholar through PaperBot")
            .option("--scholar-id <scholarId>", "Semantic Scholar author id")
            .option("--scholar-name <scholarName>", "Scholar display name")
            .action(async (options) => {
              const payload = await client.trackScholar({
                scholarId: asOptionalString(options.scholarId),
                scholarName: asOptionalString(options.scholarName)
              });
              console.log(JSON.stringify(payload, null, 2));
            });

          paper
            .command("gen-code")
            .description("Run Paper2Code through PaperBot")
            .option("--title <title>", "Paper title")
            .option("--abstract <abstract>", "Paper abstract")
            .action(async (options) => {
              const payload = await client.generateCode({
                title: String(options.title ?? ""),
                abstract: String(options.abstract ?? ""),
                userId: config.defaultUserId
              });
              console.log(JSON.stringify(payload, null, 2));
            });
        },
        { commands: ["paper"] }
      );

      const cronJobs = resolveCronJobs(config);
      api.registerService({
        id: "paperbot-openclaw-cron",
        async start() {
          api.logger.info("paperbot-openclaw cron jobs ready", cronJobs);
          return { cronJobs };
        },
        async stop() {
          api.logger.info("paperbot-openclaw cron service stopped");
        }
      });
    }
  };
}

export const DEFAULT_PAPERBOT_PLUGIN = createPaperBotOpenClawPlugin();
export { DEFAULT_PAPERBOT_CRON_JOBS } from "./cron.js";
export { createPaperBotClient, normalizeConfig } from "./paperbot-client.js";
export default DEFAULT_PAPERBOT_PLUGIN;

function extractHookText(payload: unknown): string {
  if (!payload || typeof payload !== "object") {
    return "";
  }
  if ("text" in payload) {
    return String((payload as { text?: unknown }).text ?? "").trim();
  }
  if ("content" in payload) {
    return String((payload as { content?: unknown }).content ?? "").trim();
  }
  if ("messages" in payload) {
    const messages = (payload as { messages?: BeforePromptBuildEvent["messages"] }).messages;
    return latestUserMessage(messages ?? []);
  }
  return "";
}

function asOptionalString(value: unknown): string | undefined {
  const normalized = String(value ?? "").trim();
  return normalized || undefined;
}
