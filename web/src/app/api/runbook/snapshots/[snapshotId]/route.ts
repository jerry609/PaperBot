export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/backend-proxy"

export async function GET(req: Request, ctx: { params: Promise<{ snapshotId: string }> }) {
  const { snapshotId } = await ctx.params
  return proxyJson(
    req,
    `${apiBaseUrl()}/api/runbook/snapshots/${encodeURIComponent(snapshotId)}`,
    "GET",
  )
}
