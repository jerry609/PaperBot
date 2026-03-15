import { Agent } from "undici"

import { backendBaseUrl, withBackendAuth } from "./auth-headers"

export type ProxyMethod = "DELETE" | "GET" | "HEAD" | "PATCH" | "POST" | "PUT"

type ProxyErrorContext = {
  error: unknown
  isTimeout: boolean
  upstreamUrl: string
}

type ProxyOptions = {
  accept?: string
  auth?: boolean
  cache?: RequestCache
  contentType?: string
  onError?: (context: ProxyErrorContext) => Response
  timeoutMs?: number
}

type TextProxyOptions = ProxyOptions & {
  responseContentType?: string
  responseHeaders?: HeadersInit
}

type StreamProxyOptions = ProxyOptions & {
  dispatcher?: Agent
  passthroughNonStreamResponse?: boolean
  responseContentType?: string
}

const DEFAULT_TIMEOUT_MS = 120_000
const SSE_DISPATCHER = new Agent({
  bodyTimeout: 0,
  headersTimeout: 0,
})

export function apiBaseUrl(): string {
  return backendBaseUrl()
}

export async function proxyJson(
  req: Request,
  upstreamUrl: string,
  method: ProxyMethod,
  options: TextProxyOptions = {},
): Promise<Response> {
  return proxyText(req, upstreamUrl, method, {
    accept: options.accept ?? "application/json",
    responseContentType: options.responseContentType ?? "application/json",
    ...options,
  })
}

export async function proxyText(
  req: Request,
  upstreamUrl: string,
  method: ProxyMethod,
  options: TextProxyOptions = {},
): Promise<Response> {
  const requestOptions = {
    accept: "application/json",
    ...options,
  }

  try {
    const upstream = await fetchUpstream(req, upstreamUrl, method, requestOptions)
    const text = await upstream.text()

    return buildTextResponse(text, upstream, {
      responseContentType: requestOptions.responseContentType,
      responseHeaders: requestOptions.responseHeaders,
    })
  } catch (error) {
    return handleProxyError(error, upstreamUrl, requestOptions.onError)
  }
}

export async function proxyStream(
  req: Request,
  upstreamUrl: string,
  method: ProxyMethod,
  options: StreamProxyOptions = {},
): Promise<Response> {
  const requestOptions = {
    responseContentType: "text/event-stream",
    timeoutMs: 0,
    ...options,
  }

  try {
    const upstream = await fetchUpstream(req, upstreamUrl, method, requestOptions, {
      dispatcher: requestOptions.dispatcher ?? SSE_DISPATCHER,
    })
    const upstreamContentType = upstream.headers.get("content-type") || ""

    if (
      requestOptions.passthroughNonStreamResponse &&
      !upstreamContentType.includes("text/event-stream")
    ) {
      const text = await upstream.text()
      return buildTextResponse(text, upstream, {
        responseContentType: requestOptions.responseContentType ?? "application/json",
        responseHeaders: undefined,
      })
    }

    const headers = new Headers()
    headers.set(
      "Content-Type",
      upstreamContentType || requestOptions.responseContentType || "text/event-stream",
    )
    headers.set("Cache-Control", "no-cache")
    headers.set("Connection", "keep-alive")

    return new Response(upstream.body, {
      status: upstream.status,
      headers,
    })
  } catch (error) {
    return handleProxyError(error, upstreamUrl, requestOptions.onError)
  }
}

async function fetchUpstream(
  req: Request,
  upstreamUrl: string,
  method: ProxyMethod,
  options: ProxyOptions,
  init: RequestInit & { dispatcher?: Agent } = {},
): Promise<Response> {
  const controller = options.timeoutMs === 0 ? null : new AbortController()
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS
  const timeout = controller ? setTimeout(() => controller.abort(), timeoutMs) : null
  const body = await resolveBody(req, method)

  try {
    return await fetch(upstreamUrl, {
      ...init,
      method,
      headers: await resolveHeaders(req, body, options),
      body,
      cache: options.cache,
      signal: controller?.signal,
    } as RequestInit & { dispatcher?: Agent })
  } finally {
    if (timeout) {
      clearTimeout(timeout)
    }
  }
}

async function resolveHeaders(
  req: Request,
  body: string | undefined,
  options: ProxyOptions,
): Promise<HeadersInit> {
  const headers: Record<string, string> = {}

  if (options.accept) {
    headers.Accept = options.accept
  }

  if (body !== undefined) {
    headers["Content-Type"] =
      options.contentType || req.headers.get("content-type") || "application/json"
  }

  if (options.auth) {
    return withBackendAuth(req, headers)
  }

  return headers
}

async function resolveBody(
  req: Request,
  method: ProxyMethod,
): Promise<string | undefined> {
  if (method === "GET" || method === "HEAD" || req.body === null) {
    return undefined
  }

  return req.text()
}

function buildTextResponse(
  text: string,
  upstream: Response,
  options: Pick<TextProxyOptions, "responseContentType" | "responseHeaders">,
): Response {
  const headers = new Headers(options.responseHeaders)
  headers.set("Cache-Control", "no-cache")

  const contentType =
    upstream.headers.get("content-type") || options.responseContentType

  if (contentType && upstream.status !== 204 && text.length > 0) {
    headers.set("Content-Type", contentType)
  }

  if (upstream.status === 204 || text.length === 0) {
    return new Response(null, {
      status: upstream.status,
      headers,
    })
  }

  return new Response(text, {
    status: upstream.status,
    headers,
  })
}

function handleProxyError(
  error: unknown,
  upstreamUrl: string,
  onError?: (context: ProxyErrorContext) => Response,
): Response {
  const isTimeout = error instanceof Error && error.name === "AbortError"
  const context = {
    error,
    isTimeout,
    upstreamUrl,
  }

  if (onError) {
    return onError(context)
  }

  const detail = error instanceof Error ? error.message : String(error)

  return Response.json(
    {
      detail: isTimeout
        ? `Upstream API timed out: ${upstreamUrl}`
        : `Upstream API unreachable: ${upstreamUrl}`,
      error: detail,
    },
    { status: 502 },
  )
}
