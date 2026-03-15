import { apiBaseUrl, proxyJson } from "@/app/api/_utils/auth-json-proxy"
import { proxyStream } from "@/app/api/_utils/backend-proxy"

export const runtime = "nodejs"

export async function GET(req: Request) {
  const url = new URL(req.url)
  return proxyJson(req, `${apiBaseUrl()}/api/research/repro/context?${url.searchParams.toString()}`, "GET")
}

export async function POST(req: Request) {
  return proxyStream(req, `${apiBaseUrl()}/api/research/repro/context/generate`, "POST", {
    accept: "text/event-stream",
    auth: true,
    responseContentType: "text/event-stream",
  })
}
