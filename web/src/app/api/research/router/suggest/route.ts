export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "../../_base"

export async function POST(req: Request) {
  return proxyJson(req, `${apiBaseUrl()}/api/research/router/suggest`, "POST")
}

