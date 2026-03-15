import { apiBaseUrl, proxyJson as sharedProxyJson } from "./backend-proxy"
import type { ProxyMethod } from "./backend-proxy"

export { apiBaseUrl }

export function proxyJson(
  req: Request,
  upstreamUrl: string,
  method: ProxyMethod,
): Promise<Response> {
  return sharedProxyJson(req, upstreamUrl, method, { auth: true })
}
