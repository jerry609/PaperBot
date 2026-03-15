export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/auth-json-proxy"

export async function GET(req: Request) {
  const url = new URL(req.url)
  return proxyJson(req, `${apiBaseUrl()}/api/research/collections?${url.searchParams.toString()}`, "GET")
}

export async function POST(req: Request) {
  return proxyJson(req, `${apiBaseUrl()}/api/research/collections`, "POST")
}
