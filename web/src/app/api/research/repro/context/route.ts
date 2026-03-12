import { apiBaseUrl, proxyJson } from "../../_base"
import { withBackendAuth } from "../../../_utils/auth-headers"

export const runtime = "nodejs"

export async function GET(req: Request) {
  const url = new URL(req.url)
  return proxyJson(req, `${apiBaseUrl()}/api/research/repro/context?${url.searchParams.toString()}`, "GET")
}

export async function POST(req: Request) {
  const body = await req.text()
  const upstream = await fetch(`${apiBaseUrl()}/api/research/repro/context/generate`, {
    method: "POST",
    headers: await withBackendAuth(req, {
      "Content-Type": req.headers.get("content-type") || "application/json",
      Accept: "text/event-stream",
    }),
    body,
  })

  const headers = new Headers()
  headers.set("Content-Type", upstream.headers.get("content-type") || "text/event-stream")
  headers.set("Cache-Control", "no-cache")
  headers.set("Connection", "keep-alive")

  return new Response(upstream.body, {
    status: upstream.status,
    headers,
  })
}
