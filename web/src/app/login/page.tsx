"use client"

import { Suspense, useMemo, useState } from "react"
import { signIn } from "next-auth/react"
import { useRouter, useSearchParams } from "next/navigation"
import Link from "next/link"
import { Eye, EyeOff, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import { AuthSplitLayout } from "@/components/auth/AuthSplitLayout"
import { buildAuthGateContext } from "@/lib/auth-flow"

export default function LoginPage() {
  return (
    <Suspense fallback={<LoginPageContent callbackUrl="/dashboard" />}>
      <LoginPageWithSearchParams />
    </Suspense>
  )
}

function LoginPageWithSearchParams() {
  const searchParams = useSearchParams()
  const callbackUrl = searchParams.get("callbackUrl") || "/dashboard"
  return <LoginPageContent callbackUrl={callbackUrl} />
}

function LoginPageContent({ callbackUrl }: { callbackUrl: string }) {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [githubLoading, setGithubLoading] = useState(false)
  const router = useRouter()
  const gate = useMemo(() => buildAuthGateContext(callbackUrl), [callbackUrl])
  const registerHref = `/register?callbackUrl=${encodeURIComponent(callbackUrl)}`
  const forgotHref = `/forgot-password?callbackUrl=${encodeURIComponent(callbackUrl)}`

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setLoading(true)

    try {
      const check = await fetch("/api/auth/login-check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      })
      if (!check.ok) {
        const data = (await check.json().catch(() => ({}))) as { detail?: string }
        setError(data.detail || "Invalid email or password.")
        setLoading(false)
        return
      }
    } catch {
      // Fall through to provider sign-in so transient network errors share the same path.
    }

    const res = await signIn("credentials", { email, password, redirect: false })
    setLoading(false)
    if (res?.error) {
      setError("Invalid email or password.")
      return
    }
    router.push(callbackUrl)
  }

  const onGithub = () => {
    setGithubLoading(true)
    signIn("github", { callbackUrl })
  }

  return (
    <AuthSplitLayout
      panelEyebrow={gate.eyebrow}
      panelTitle={gate.title}
      panelDescription={gate.description}
      panelSteps={gate.steps}
      destinationLabel={gate.destinationLabel}
      destination={gate.destination}
      cardEyebrow={gate.eyebrow}
      cardTitle={gate.title}
      cardDescription={gate.description}
    >
      <Button
        variant="outline"
        className="h-11 w-full rounded-full border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
        onClick={onGithub}
        disabled={loading || githubLoading}
      >
        {githubLoading ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <GitHubIcon />
        )}
        Continue with GitHub
      </Button>

      <div className="flex items-center gap-3 text-slate-400">
        <Separator className="flex-1 bg-slate-200" />
        <span className="text-[11px] uppercase tracking-[0.16em]">or</span>
        <Separator className="flex-1 bg-slate-200" />
      </div>

      <form onSubmit={onSubmit} className="space-y-4">
        <div className="space-y-1.5">
          <Label htmlFor="email">Email address</Label>
          <Input
            id="email"
            type="email"
            autoComplete="email"
            placeholder="name@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            disabled={loading}
            required
            className="h-11 rounded-2xl border-slate-200 bg-white"
          />
        </div>

        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <Label htmlFor="password">Password</Label>
            <Link
              href={forgotHref}
              className="text-xs text-slate-500 underline-offset-4 hover:text-slate-900 hover:underline"
            >
              Forgot password?
            </Link>
          </div>
          <div className="relative">
            <Input
              id="password"
              type={showPassword ? "text" : "password"}
              autoComplete="current-password"
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={loading}
              required
              className="h-11 rounded-2xl border-slate-200 bg-white pr-10"
            />
            <button
              type="button"
              tabIndex={-1}
              onClick={() => setShowPassword((value) => !value)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 transition-colors hover:text-slate-700"
            >
              {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </button>
          </div>
        </div>

        {error ? (
          <p className="rounded-2xl border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">
            {error}
          </p>
        ) : null}

        <Button
          type="submit"
          className="h-11 w-full rounded-full bg-slate-900 text-white hover:bg-slate-800"
          disabled={loading || githubLoading}
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
          {gate.primaryActionLabel}
        </Button>
      </form>

      <p className="text-center text-sm text-slate-500">
        Need access first?{" "}
        <Link
          href={registerHref}
          className="font-medium text-slate-900 underline-offset-4 hover:underline"
        >
          {gate.createAccountLabel}
        </Link>
      </p>
    </AuthSplitLayout>
  )
}

function GitHubIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor" aria-hidden="true">
      <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z" />
    </svg>
  )
}
