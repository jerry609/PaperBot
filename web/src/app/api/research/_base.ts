export function apiBaseUrl() {
  return process.env.PAPERBOT_API_BASE_URL || "http://127.0.0.1:8000"
}

export async function proxyJson(req: Request, upstreamUrl: string, method: string) {
  const body = method === "GET" ? undefined : await req.text()
  const upstream = await fetch(upstreamUrl, {
    method,
    headers: {
      Accept: "application/json",
      "Content-Type": req.headers.get("content-type") || "application/json",
    },
    body,
  })
  const text = await upstream.text()
  return new Response(text, {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("content-type") || "application/json",
      "Cache-Control": "no-cache",
    },
  })
}

