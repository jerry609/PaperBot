import { auth } from "@/auth"

export function backendBaseUrl(): string {
  return (
    process.env.BACKEND_BASE_URL ||
    process.env.PAPERBOT_API_BASE_URL ||
    "http://127.0.0.1:8000"
  )
}

export async function withBackendAuth(
  req: Request,
  base: HeadersInit = {}
): Promise<HeadersInit> {
  const headers = new Headers(base as any)

  // Prefer client-provided Authorization header if present
  const incoming = req.headers.get("authorization")
  if (incoming) {
    headers.set("authorization", incoming)
    return headers
  }

  // Otherwise pull token from NextAuth session (server-side)
  try {
    const session = await auth()
    const token = (session as any)?.accessToken as string | undefined
    if (token) {
      headers.set("authorization", `Bearer ${token}`)
    } else {
      console.warn("[auth-headers] session.accessToken is missing", {
        hasSession: !!session,
        userId: (session as any)?.userId,
        provider: (session as any)?.provider,
      })
    }
  } catch (e) {
    console.error("[auth-headers] auth() threw:", e)
  }
  return headers
}
