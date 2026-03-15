import { apiBaseUrl, proxyText } from "@/app/api/_utils/backend-proxy"

export const runtime = "nodejs"

export async function GET(req: Request) {
  const url = new URL(req.url)
  return proxyText(req, `${apiBaseUrl()}/api/runbook/file?${url.searchParams.toString()}`, "GET")
}

export async function POST(req: Request) {
  return proxyText(req, `${apiBaseUrl()}/api/runbook/file`, "POST")
}
