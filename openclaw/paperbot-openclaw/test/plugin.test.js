import assert from "node:assert/strict";
import test from "node:test";

import plugin, {
  DEFAULT_PAPERBOT_CRON_JOBS,
  createPaperBotOpenClawPlugin
} from "../dist/index.js";

function createMockProgram(registry) {
  function createCommand(path) {
    return {
      description() {
        return this;
      },
      option() {
        return this;
      },
      action(handler) {
        registry.push({ path, handler });
        return this;
      },
      command(name) {
        return createCommand(`${path} ${name}`);
      }
    };
  }

  return {
    command(name) {
      return createCommand(name);
    }
  };
}

function createMockApi(config = {}) {
  const tools = [];
  const hooks = [];
  const lifecycleHooks = [];
  const cliCommands = [];
  const services = [];
  const logs = [];

  return {
    api: {
      config,
      logger: {
        info(...args) {
          logs.push(["info", ...args]);
        },
        warn(...args) {
          logs.push(["warn", ...args]);
        },
        error(...args) {
          logs.push(["error", ...args]);
        }
      },
      registerTool(tool) {
        tools.push(tool);
      },
      registerHook(name, handler, options) {
        hooks.push({ name, handler, options });
      },
      on(name, handler, options) {
        lifecycleHooks.push({ name, handler, options });
      },
      registerCli(register, options) {
        const registry = [];
        register({ program: createMockProgram(registry) });
        cliCommands.push({ options, registry });
      },
      registerService(service) {
        services.push(service);
      }
    },
    tools,
    hooks,
    lifecycleHooks,
    cliCommands,
    services,
    logs
  };
}

test("default export exposes the paperbot-openclaw plugin descriptor", () => {
  assert.equal(plugin.id, "paperbot-openclaw");
  assert.equal(plugin.name, "paperbot-openclaw");
});

test("register wires tools, hooks, cli commands, and cron-backed service", async () => {
  const fetchCalls = [];
  const mockFetch = async (url, init) => {
    fetchCalls.push([url, init]);
    return new Response(JSON.stringify({ ok: true, items: [] }), {
      status: 200,
      headers: { "content-type": "application/json" }
    });
  };
  const mock = createMockApi({
    baseUrl: "http://paperbot.local",
    defaultUserId: "openclaw-user",
    cronQueries: ["llm agents", "prompt compression"]
  });

  await createPaperBotOpenClawPlugin({ fetchImpl: mockFetch }).register(mock.api);

  assert.equal(mock.tools.length, 6);
  assert.deepEqual(
    mock.tools.map((tool) => tool.name).sort(),
    ["gen_code", "paper_analyze", "paper_search", "paper_track", "research", "review"]
  );
  assert.deepEqual(
    mock.hooks.map((hook) => hook.name).sort(),
    ["before_prompt", "message_received", "msg_recv"]
  );
  assert.deepEqual(
    mock.lifecycleHooks.map((hook) => hook.name),
    ["before_prompt_build"]
  );
  assert.equal(mock.cliCommands.length, 1);
  assert.deepEqual(
    mock.cliCommands[0].registry.map((row) => row.path).sort(),
    ["paper analyze", "paper gen-code", "paper search", "paper track"]
  );
  assert.equal(mock.services.length, 1);
  const servicePayload = await mock.services[0].start();
  assert.equal(servicePayload.cronJobs.length, 4);
  assert.equal(fetchCalls.length, 0);
});

test("paper_search tool bridges to PaperBot paperscool search", async () => {
  const mock = createMockApi({ baseUrl: "http://paperbot.local" });
  const fetchCalls = [];
  const mockFetch = async (url, init) => {
    fetchCalls.push([url, init]);
    return new Response(
      JSON.stringify({
        source: "papers_cool",
        queries: [],
        items: [{ title: "UniICL" }],
        summary: { total: 1 }
      }),
      {
        status: 200,
        headers: { "content-type": "application/json" }
      }
    );
  };

  await createPaperBotOpenClawPlugin({ fetchImpl: mockFetch }).register(mock.api);
  const tool = mock.tools.find((item) => item.name === "paper_search");

  const result = await tool.execute("tool-1", { query: "llm agents" });

  assert.equal(fetchCalls[0][0], "http://paperbot.local/api/research/paperscool/search");
  assert.match(result.content[0].text, /UniICL/);
});

test("SSE-backed tools return the final result envelope", async () => {
  const mock = createMockApi({ baseUrl: "http://paperbot.local" });
  const mockFetch = async () =>
    new Response(
      [
        'data: {"type":"progress","data":{"phase":"Initializing"}}',
        "",
        'data: {"type":"result","data":{"summary":"analysis complete","score":0.9}}',
        "",
        "data: [DONE]",
        ""
      ].join("\n"),
      {
        status: 200,
        headers: { "content-type": "text/event-stream" }
      }
    );

  await createPaperBotOpenClawPlugin({ fetchImpl: mockFetch }).register(mock.api);
  const analyzeTool = mock.tools.find((item) => item.name === "paper_analyze");
  const reviewTool = mock.tools.find((item) => item.name === "review");

  const analyzeResult = await analyzeTool.execute("tool-2", { title: "UniICL" });
  const reviewResult = await reviewTool.execute("tool-3", {
    title: "UniICL",
    abstract: "A paper about in-context learning."
  });

  assert.match(analyzeResult.content[0].text, /analysis complete/);
  assert.match(reviewResult.content[0].text, /analysis complete/);
});

test("message and prompt hooks provide intent routing and context injection", async () => {
  const mock = createMockApi({ baseUrl: "http://paperbot.local" });
  const mockFetch = async (url) => {
    if (String(url).endsWith("/api/research/context")) {
      return new Response(
        JSON.stringify({
          memory_items: [{ content: "prefers ICLR papers" }],
          paper_recommendations: [{ title: "UniICL" }]
        }),
        {
          status: 200,
          headers: { "content-type": "application/json" }
        }
      );
    }
    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "content-type": "application/json" }
    });
  };

  await createPaperBotOpenClawPlugin({ fetchImpl: mockFetch }).register(mock.api);
  const msgHook = mock.hooks.find((hook) => hook.name === "msg_recv");
  const promptHook = mock.lifecycleHooks.find((hook) => hook.name === "before_prompt_build");

  const msgPayload = await msgHook.handler({
    text: "Can you search papers about prompt compression?"
  });
  const promptPayload = await promptHook.handler({
    messages: [{ role: "user", content: "Find papers on prompt compression" }]
  });

  assert.equal(msgPayload.suggestedTool, "paper_search");
  assert.match(promptPayload.prependSystemContext, /PaperBot context bridge/);
  assert.match(promptPayload.prependSystemContext, /paper recommendations: 1/);
});

test("cron descriptors stay aligned with the planned four OpenClaw jobs", () => {
  assert.deepEqual(
    DEFAULT_PAPERBOT_CRON_JOBS.map((job) => job.id),
    ["paper-monitor", "weekly-digest", "conference-deadlines", "citation-monitor"]
  );
});
