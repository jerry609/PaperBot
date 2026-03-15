import { apiBaseUrl, proxyJson } from "@/app/api/_utils/backend-proxy"

export async function GET(req: Request) {
  return proxyJson(req, `${apiBaseUrl()}/api/auth/me`, "GET", { auth: true })
}

export async function PATCH(req: Request) {
  return proxyJson(req, `${apiBaseUrl()}/api/auth/me`, "PATCH", { auth: true })
}

export async function DELETE(req: Request) {
  return proxyJson(req, `${apiBaseUrl()}/api/auth/me`, "DELETE", { auth: true })
}
