export const runtime = "nodejs"

function apiBaseUrl() {
  return process.env.PAPERBOT_API_BASE_URL || "http://127.0.0.1:8000"
}

export async function POST(req: Request) {
  const body = await req.text()
  const upstream = await fetch(`${apiBaseUrl()}/api/studio/chat`, {
    method: "POST",
    headers: {
      "Content-Type": req.headers.get("content-type") || "application/json",
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
