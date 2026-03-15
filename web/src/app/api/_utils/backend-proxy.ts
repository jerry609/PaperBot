import { withBackendAuth } from "./auth-headers"

type ProxyTextOptions = {
  accept?: string
  auth?: boolean
  responseContentType?: string
}

export function apiBaseUrl(): string {
  return process.env.PAPERBOT_API_BASE_URL || "http://127.0.0.1:8000"
}

export async function proxyText(
  req: Request,
  upstreamUrl: string,
  method: string,
  options: ProxyTextOptions = {},
): Promise<Response> {
  const normalizedMethod = method.toUpperCase()
  const headers: Record<string, string> = {
    Accept: options.accept || "application/json",
  }

  let body: string | undefined
  if (normalizedMethod !== "GET" && normalizedMethod !== "HEAD") {
    body = await req.text()
    headers["Content-Type"] = req.headers.get("content-type") || "application/json"
  }

  const upstreamHeaders = options.auth ? await withBackendAuth(req, headers) : headers
  const upstream = await fetch(upstreamUrl, {
    method: normalizedMethod,
    headers: upstreamHeaders,
    body,
  })
  const text = await upstream.text()

  return new Response(text, {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("content-type") || options.responseContentType || "application/json",
      "Cache-Control": "no-cache",
    },
  })
}
