export function apiBaseUrl() {
  return process.env.PAPERBOT_API_BASE_URL || "http://127.0.0.1:8000"
}

export async function proxyJson(req: Request, upstreamUrl: string, method: string) {
  const body = method === "GET" ? undefined : await req.text()
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 120_000) // 2 min timeout

  try {
    const baseHeaders = {
      method,
      Accept: "application/json",
      "Content-Type": req.headers.get("content-type") || "application/json",
    } as Record<string, string>
    const { withBackendAuth } = await import("../_utils/auth-headers")
    const headers = await withBackendAuth(req, baseHeaders)
    const upstream = await fetch(upstreamUrl, {
      method,
      headers,
      body,
      signal: controller.signal,
    })
    const text = await upstream.text()
    return new Response(text, {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("content-type") || "application/json",
        "Cache-Control": "no-cache",
      },
    })
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error)
    const isTimeout = error instanceof Error && error.name === "AbortError"
    return Response.json(
      {
        detail: isTimeout
          ? `Upstream API timed out: ${upstreamUrl}`
          : `Upstream API unreachable: ${upstreamUrl}`,
        error: detail,
      },
      { status: 502 },
    )
  } finally {
    clearTimeout(timeout)
  }
}
