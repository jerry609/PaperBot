import { NextRequest, NextResponse } from "next/server"
import { backendBaseUrl } from "@/app/api/_utils/auth-headers"

export async function POST(req: NextRequest) {
  const body = await req.json()
  const res = await fetch(`${backendBaseUrl()}/api/auth/reset-password`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  const data = await res.json().catch(() => null)
  return NextResponse.json(data, { status: res.status })
}
