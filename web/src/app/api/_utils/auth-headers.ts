import { auth } from "@/auth"

type SessionMetadata = {
  accessToken?: string
  provider?: string
  userId?: string | number
}

function readSessionMetadata(session: unknown): SessionMetadata {
  if (!session || typeof session !== "object") {
    return {}
  }

  const record = session as Record<string, unknown>
  return {
    accessToken: typeof record.accessToken === "string" ? record.accessToken : undefined,
    provider: typeof record.provider === "string" ? record.provider : undefined,
    userId:
      typeof record.userId === "string" || typeof record.userId === "number"
        ? record.userId
        : undefined,
  }
}

export function backendBaseUrl(): string {
  return process.env.BACKEND_BASE_URL || process.env.PAPERBOT_API_BASE_URL || "http://127.0.0.1:8000"
}

export async function withBackendAuth(
  req: Request,
  base: HeadersInit = {},
): Promise<HeadersInit> {
  const headers = new Headers(base)

  // Prefer client-provided Authorization header if present
  const incoming = req.headers.get("authorization")
  if (incoming) {
    headers.set("authorization", incoming)
    return headers
  }

  // Otherwise pull token from NextAuth session (server-side)
  try {
    const session = await auth()
    const metadata = readSessionMetadata(session)
    if (metadata.accessToken) {
      headers.set("authorization", `Bearer ${metadata.accessToken}`)
    } else {
      console.warn("[auth-headers] session.accessToken is missing", {
        hasSession: !!session,
        userId: metadata.userId,
        provider: metadata.provider,
      })
    }
  } catch (e) {
    console.error("[auth-headers] auth() threw:", e)
  }
  return headers
}
