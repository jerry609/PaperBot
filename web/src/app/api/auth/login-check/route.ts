export const runtime = "nodejs"

import { backendBaseUrl } from "../../_utils/auth-headers"

export async function POST(req: Request) {
  const body = await req.text()
  const res = await fetch(`${backendBaseUrl()}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  }).catch(() => null)

  if (!res) return Response.json({ detail: "Service unavailable" }, { status: 502 })
  const text = await res.text()
  return new Response(text, {
    status: res.status,
    headers: { "Content-Type": "application/json" },
  })
}
