export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "../research/_base"

export async function GET(req: Request) {
  return proxyJson(req, `${apiBaseUrl()}/api/embedding-settings`, "GET")
}

export async function PATCH(req: Request) {
  return proxyJson(req, `${apiBaseUrl()}/api/embedding-settings`, "PATCH")
}
