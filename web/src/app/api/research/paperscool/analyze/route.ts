export const runtime = "nodejs"

import { apiBaseUrl } from "../../_base"

export async function POST(req: Request) {
  const body = await req.text()
  const upstream = await fetch(`${apiBaseUrl()}/api/research/paperscool/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": req.headers.get("content-type") || "application/json",
      Accept: "text/event-stream",
    },
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
