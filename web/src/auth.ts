import NextAuth from "next-auth"
import GitHub from "next-auth/providers/github"
import Credentials from "next-auth/providers/credentials"

function backendBaseUrl() {
  return process.env.BACKEND_BASE_URL || process.env.PAPERBOT_API_BASE_URL || "http://127.0.0.1:8000"
}

export const { handlers, signIn, signOut, auth } = NextAuth({
  providers: [
    GitHub,
    Credentials({
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        try {
          const res = await fetch(`${backendBaseUrl()}/api/auth/login`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email: credentials?.email, password: credentials?.password }),
          })
          if (!res.ok) return null
          const data = await res.json() as { access_token: string; user_id: number; display_name?: string }
          return { id: String(data.user_id), name: data.display_name || "", accessToken: data.access_token, userId: data.user_id } as any
        } catch {
          return null
        }
      },
    }),
  ],
  callbacks: {
    async jwt({ token, user, account, profile, trigger, session: sessionData }) {
      // Handle updateSession() calls — persist updated fields into the token
      if (trigger === "update") {
        if (sessionData?.name !== undefined) token.name = sessionData.name
      }

      // Credentials: copy backend JWT from user
      if (user && (user as any).accessToken) {
        token.accessToken = (user as any).accessToken
        token.userId = (user as any).userId
        if (user.name) token.name = user.name
      }

      // Persist provider on first sign-in
      if (account?.provider) {
        token.provider = account.provider
      }

      // GitHub OAuth: exchange for backend JWT
      if (account?.provider === "github" && account.access_token && profile) {
        const res = await fetch(`${backendBaseUrl()}/api/auth/github/exchange`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            github_id: String((profile as any).id),
            login: (profile as any).login,
            name: (profile as any).name,
            avatar_url: (profile as any).avatar_url,
            email: (profile as any).email,
            access_token: account.access_token,
          }),
        })
        if (res.ok) {
          const data = await res.json() as { access_token: string; user_id: number; display_name?: string }
          token.accessToken = data.access_token
          token.userId = data.user_id
          if (data.display_name) token.name = data.display_name
        } else {
          console.error("[auth] github/exchange failed:", res.status, await res.text().catch(() => ""))
          // Ensure we do not keep a half-authenticated session without backend JWT
          delete token.accessToken
          delete token.userId
        }
      }
      return token
    },
    async session({ session, token }) {
      ;(session as any).accessToken = token.accessToken
      ;(session as any).userId = token.userId
      ;(session as any).provider = token.provider
      if (token.name !== undefined) session.user.name = token.name as string
      return session
    },
  },
  pages: {
    signIn: "/login",
    error: "/login",
  },
})

