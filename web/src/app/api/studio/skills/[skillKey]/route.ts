export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/backend-proxy"

export async function GET(
  req: Request,
  context: { params: Promise<{ skillKey: string }> },
) {
  const { skillKey } = await context.params
  return proxyJson(req, `${apiBaseUrl()}/api/studio/skills/${encodeURIComponent(skillKey)}`, "GET")
}
