import { NextRequest, NextResponse } from "next/server"
import { backendBaseUrl, withBackendAuth } from "@/app/api/_utils/auth-headers"

export async function POST(req: NextRequest) {
  const body = await req.json()
  const headers = await withBackendAuth(req, { "Content-Type": "application/json" })
  const res = await fetch(`${backendBaseUrl()}/api/auth/me/change-password`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  })
  const data = await res.json().catch(() => null)
  return NextResponse.json(data, { status: res.status })
}
