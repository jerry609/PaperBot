import { apiBaseUrl, proxyJson } from "@/app/api/_utils/auth-json-proxy"

export async function GET(req: Request) {
  const url = new URL(req.url)
  const upstream = `${apiBaseUrl()}/api/papers/library${url.search}`
  return proxyJson(req, upstream, "GET")
}
