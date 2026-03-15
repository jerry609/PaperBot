import {
  apiBaseUrl as sharedApiBaseUrl,
  proxyJson as sharedProxyJson,
  type ProxyMethod,
} from "../_utils/backend-proxy"

export function apiBaseUrl(): string {
  return sharedApiBaseUrl()
}

export function proxyJson(
  req: Request,
  upstreamUrl: string,
  method: ProxyMethod,
): Promise<Response> {
  return sharedProxyJson(req, upstreamUrl, method, { auth: true })
}
