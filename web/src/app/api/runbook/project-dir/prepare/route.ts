export const runtime = "nodejs"

import fs from "node:fs/promises"
import os from "node:os"
import path from "node:path"

import { apiBaseUrl } from "@/app/api/_utils/backend-proxy"
import { withBackendAuth } from "@/app/api/_utils/auth-headers"

type PrepareProjectDirRequest = {
  project_dir?: string
  create_if_missing?: boolean
}

function normalizeDirectoryInput(rawPath: string): string {
  const raw = rawPath.trim()
  if (!raw) {
    throw new Error("project_dir cannot be empty")
  }
  if (raw.includes("\u0000")) {
    throw new Error("project_dir contains invalid character")
  }

  const homeDir = process.env.HOME || os.homedir()
  const expanded =
    raw === "~"
      ? homeDir
      : raw.startsWith("~/")
        ? path.join(homeDir, raw.slice(2))
        : raw

  return path.resolve(expanded)
}

function uniqueRoots(values: string[]): string[] {
  const seen = new Set<string>()
  const result: string[] = []
  for (const value of values) {
    const normalized = path.resolve(value)
    if (seen.has(normalized)) continue
    seen.add(normalized)
    result.push(normalized)
  }
  return result
}

function allowedWorkspaceRoots(): string[] {
  const homeDir = process.env.HOME || os.homedir()
  return uniqueRoots([
    os.tmpdir(),
    process.cwd(),
    path.join(homeDir, "Documents"),
  ])
}

function isUnderRoot(target: string, root: string): boolean {
  const relative = path.relative(root, target)
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative))
}

async function fallbackPrepareProjectDir(body: PrepareProjectDirRequest): Promise<Response> {
  try {
    const projectDir = normalizeDirectoryInput(body.project_dir || "")
    const createIfMissing = body.create_if_missing !== false
    const allowedRoots = allowedWorkspaceRoots()

    if (!allowedRoots.some((root) => isUnderRoot(projectDir, root))) {
      return Response.json({ detail: "project_dir is not allowed" }, { status: 403 })
    }

    let created = false

    try {
      const stat = await fs.stat(projectDir)
      if (!stat.isDirectory()) {
        return Response.json({ detail: "project_dir must be a directory" }, { status: 400 })
      }
    } catch {
      if (!createIfMissing) {
        return Response.json({ detail: "project_dir must be an existing directory" }, { status: 400 })
      }
      await fs.mkdir(projectDir, { recursive: true })
      created = true
    }

    return Response.json({
      project_dir: projectDir,
      created,
      allowed_prefixes: allowedRoots,
      source: "next_fallback",
    })
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Failed to prepare project_dir"
    return Response.json({ detail }, { status: 400 })
  }
}

function relayTextResponse(text: string, upstream: Response): Response {
  return new Response(text, {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("content-type") || "application/json",
      "Cache-Control": "no-cache",
    },
  })
}

export async function POST(req: Request) {
  const rawBody = await req.text()
  let parsedBody: PrepareProjectDirRequest = {}

  try {
    parsedBody = rawBody ? (JSON.parse(rawBody) as PrepareProjectDirRequest) : {}
  } catch {
    return Response.json({ detail: "Invalid JSON body" }, { status: 400 })
  }

  try {
    const upstream = await fetch(`${apiBaseUrl()}/api/runbook/project-dir/prepare`, {
      method: "POST",
      headers: await withBackendAuth(req, {
        Accept: "application/json",
        "Content-Type": req.headers.get("content-type") || "application/json",
      }),
      body: rawBody,
      cache: "no-store",
    })

    const text = await upstream.text()
    if (upstream.ok || upstream.status !== 403) {
      return relayTextResponse(text, upstream)
    }
  } catch {
    // Fall through to the local Node.js fallback below.
  }

  return fallbackPrepareProjectDir(parsedBody)
}
