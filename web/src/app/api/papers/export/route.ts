export const runtime = "nodejs"

import { withBackendAuth } from "@/app/api/_utils/auth-headers"
import { apiBaseUrl } from "@/app/api/research/_base"

export async function GET(req: Request) {
  const url = new URL(req.url)
  const upstream = `${apiBaseUrl()}/api/research/papers/export?${url.searchParams.toString()}`

  try {
    const res = await fetch(upstream, {
      method: "GET",
      headers: await withBackendAuth(req, { Accept: "*/*" }),
    })
    const body = await res.arrayBuffer()
    return new Response(body, {
      status: res.status,
      headers: {
        "Content-Type": res.headers.get("content-type") || "application/octet-stream",
        "Content-Disposition": res.headers.get("content-disposition") || "",
        "Cache-Control": "no-cache",
      },
    })
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error)
    return Response.json(
      { detail: `Upstream API unreachable`, error: detail },
      { status: 502 },
    )
  }
}
