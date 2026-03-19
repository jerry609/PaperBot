export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/backend-proxy"

export async function POST(
  req: Request,
  context: { params: Promise<{ repoSlug: string }> },
) {
  const { repoSlug } = await context.params
  return proxyJson(
    req,
    `${apiBaseUrl()}/api/studio/skills/repos/${encodeURIComponent(repoSlug)}/update`,
    "POST",
  )
}
