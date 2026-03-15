import { apiBaseUrl, proxyText } from "@/app/api/_utils/backend-proxy"

export const runtime = "nodejs"

export async function POST(req: Request) {
  return proxyText(req, `${apiBaseUrl()}/api/runbook/revert-project`, "POST", { auth: true })
}
