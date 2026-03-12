import { NextRequest, NextResponse } from "next/server"
import { backendBaseUrl, withBackendAuth } from "@/app/api/_utils/auth-headers"

export async function GET(req: NextRequest) {
  const headers = await withBackendAuth(req)
  const res = await fetch(`${backendBaseUrl()}/api/auth/me`, { headers })
  const data = await res.json().catch(() => null)
  return NextResponse.json(data, { status: res.status })
}

export async function PATCH(req: NextRequest) {
  const body = await req.json()
  const headers = await withBackendAuth(req, { "Content-Type": "application/json" })
  const res = await fetch(`${backendBaseUrl()}/api/auth/me`, {
    method: "PATCH",
    headers,
    body: JSON.stringify(body),
  })
  const data = await res.json().catch(() => null)
  return NextResponse.json(data, { status: res.status })
}

export async function DELETE(req: NextRequest) {
  const headers = await withBackendAuth(req)
  const res = await fetch(`${backendBaseUrl()}/api/auth/me`, { method: "DELETE", headers })
  return new NextResponse(null, { status: res.status })
}
