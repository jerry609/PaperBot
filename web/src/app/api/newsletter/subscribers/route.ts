export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/auth-json-proxy"

export async function GET(req: Request) {
  return proxyJson(req, `${apiBaseUrl()}/api/newsletter/subscribers`, "GET")
}
