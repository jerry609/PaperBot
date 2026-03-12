declare module "openclaw/plugin-sdk/core" {
  export interface ToolTextChunk {
    type: "text";
    text: string;
  }

  export interface ToolResult {
    content: ToolTextChunk[];
  }

  export interface ToolDefinition<TInput = Record<string, unknown>> {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
    execute(invocationId: string, input: TInput): Promise<ToolResult>;
  }

  export interface HookRegistrationOptions {
    name: string;
    description?: string;
  }

  export interface PromptMessage {
    role?: string;
    content?: string;
  }

  export interface BeforePromptBuildEvent {
    messages?: PromptMessage[];
  }

  export interface BeforePromptBuildResult {
    prependSystemContext?: string;
  }

  export interface CommandBuilder {
    description(text: string): CommandBuilder;
    option(flags: string, description?: string): CommandBuilder;
    action(handler: (options: Record<string, unknown>) => unknown | Promise<unknown>): CommandBuilder;
    command(name: string): CommandBuilder;
  }

  export interface CommandProgram {
    command(name: string): CommandBuilder;
  }

  export interface LoggerLike {
    info(...args: unknown[]): void;
    warn(...args: unknown[]): void;
    error(...args: unknown[]): void;
  }

  export interface ServiceRegistration {
    id: string;
    start(): unknown | Promise<unknown>;
    stop(): unknown | Promise<unknown>;
  }

  export interface OpenClawPluginApi {
    config?: unknown;
    logger: LoggerLike;
    registerTool<TInput = Record<string, unknown>>(tool: ToolDefinition<TInput>): void;
    registerHook(
      name: string,
      handler: (payload: unknown, context?: unknown) => unknown | Promise<unknown>,
      options?: HookRegistrationOptions
    ): void;
    on(
      name: "before_prompt_build",
      handler: (
        event: BeforePromptBuildEvent,
        context?: unknown
      ) => BeforePromptBuildResult | Promise<BeforePromptBuildResult>,
      options?: { priority?: number }
    ): void;
    registerCli(
      register: (ctx: { program: CommandProgram }) => void,
      options?: { commands?: string[] }
    ): void;
    registerService(service: ServiceRegistration): void;
  }
}
