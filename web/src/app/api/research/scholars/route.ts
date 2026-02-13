export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "../_base"

export async function GET(req: Request) {
  const url = new URL(req.url)
  return proxyJson(req, `${apiBaseUrl()}/api/research/scholars?${url.searchParams.toString()}`, "GET")
}

export async function POST(req: Request) {
  return proxyJson(req, `${apiBaseUrl()}/api/research/scholars`, "POST")
}
