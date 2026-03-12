import "next-auth"

declare module "next-auth" {
  interface Session {
    accessToken?: string
    userId?: number
    provider?: string
  }
}
