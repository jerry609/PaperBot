import { spawn } from "node:child_process";
import { setTimeout as sleep } from "node:timers/promises";

const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";
const npxCommand = process.platform === "win32" ? "npx.cmd" : "npx";
const e2ePort = process.env.E2E_PORT || "3100";
const e2eBaseUrl = process.env.E2E_BASE_URL || `http://127.0.0.1:${e2ePort}`;
const readinessUrl = new URL("/dashboard", e2eBaseUrl).toString();
const playwrightArgs = ["playwright", "test", ...process.argv.slice(2)];

function isReadyStatus(status) {
  return status >= 200 && status < 400;
}

async function waitForServer(url, timeoutMs, serverProcess) {
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    if (serverProcess.exitCode !== null) {
      throw new Error(`Next.js server exited early with code ${serverProcess.exitCode}`);
    }

    try {
      const response = await fetch(url, { redirect: "manual" });
      if (isReadyStatus(response.status)) return;
    } catch {
      // Ignore transient connection errors while the dev server boots.
    }

    await sleep(500);
  }

  throw new Error(`Timed out waiting for ${url}`);
}

async function run() {
  let serverProcess = null;

  if (!process.env.E2E_BASE_URL) {
    serverProcess = spawn(
      npmCommand,
      ["run", "dev", "--", "--hostname", "127.0.0.1", "--port", e2ePort],
      {
        stdio: "inherit",
        env: process.env,
      },
    );

    const terminateServer = () => {
      if (serverProcess && serverProcess.exitCode === null) {
        serverProcess.kill("SIGTERM");
      }
    };

    process.on("SIGINT", terminateServer);
    process.on("SIGTERM", terminateServer);
    process.on("exit", terminateServer);

    await waitForServer(readinessUrl, 120_000, serverProcess);
  }

  const testProcess = spawn(npxCommand, playwrightArgs, {
    stdio: "inherit",
    env: {
      ...process.env,
      E2E_BASE_URL: e2eBaseUrl,
    },
  });

  const exitCode = await new Promise((resolve, reject) => {
    testProcess.on("error", reject);
    testProcess.on("exit", (code, signal) => {
      if (signal) {
        reject(new Error(`Playwright exited with signal ${signal}`));
        return;
      }
      resolve(code ?? 1);
    });
  });

  if (serverProcess && serverProcess.exitCode === null) {
    serverProcess.kill("SIGTERM");
  }

  process.exit(exitCode);
}

run().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
