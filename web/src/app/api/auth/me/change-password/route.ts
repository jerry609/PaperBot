import { apiBaseUrl, proxyJson } from "@/app/api/_utils/backend-proxy"

export async function POST(req: Request) {
  return proxyJson(req, `${apiBaseUrl()}/api/auth/me/change-password`, "POST", {
    auth: true,
  })
}
