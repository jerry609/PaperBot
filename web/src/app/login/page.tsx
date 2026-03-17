"use client"

import { Suspense, useMemo, useState } from "react"
import { signIn } from "next-auth/react"
import { useRouter, useSearchParams } from "next/navigation"
import Link from "next/link"
import {
  ArrowRight,
  BookOpen,
  Eye,
  EyeOff,
  FolderOpen,
  Loader2,
  MessageSquare,
  ShieldCheck,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"

type AuthGateContext = {
  title: string
  eyebrow: string
  description: string
  destination: string
  destinationLabel: string
  primaryActionLabel: string
  createAccountLabel: string
  steps: Array<{
    title: string
    detail: string
    icon: "paper" | "workspace" | "chat"
  }>
}

function normalizeCallbackPath(callbackUrl: string): string {
  try {
    const parsed = new URL(callbackUrl, "http://localhost")
    return `${parsed.pathname}${parsed.search}${parsed.hash}` || "/dashboard"
  } catch {
    return callbackUrl.startsWith("/") ? callbackUrl : "/dashboard"
  }
}

function buildAuthGateContext(callbackUrl: string): AuthGateContext {
  const destination = normalizeCallbackPath(callbackUrl)

  if (destination.startsWith("/studio")) {
    return {
      title: "Continue to DeepCode Studio",
      eyebrow: "Secure Studio Gate",
      description:
        "Your paper setup, workspace review, and chat launch surface are ready. Sign in once and continue the Studio flow without losing context.",
      destination,
      destinationLabel: "Studio session",
      primaryActionLabel: "Continue to Studio",
      createAccountLabel: "Create an account to continue",
      steps: [
        {
          title: "Paper selected",
          detail: "The Studio thread will reopen on the same paper context after sign-in.",
          icon: "paper",
        },
        {
          title: "Workspace review",
          detail: "Directory validation and workspace confirmation stay attached to the Studio launch flow.",
          icon: "workspace",
        },
        {
          title: "Claude Code chat",
          detail: "The chat surface resumes first. Monitor remains available for full worker and tool detail.",
          icon: "chat",
        },
      ],
    }
  }

  if (destination.startsWith("/research")) {
    return {
      title: "Continue to Research",
      eyebrow: "Secure Access",
      description:
        "Sign in to reopen your research workspace with the current route and session context intact.",
      destination,
      destinationLabel: "Research workspace",
      primaryActionLabel: "Continue",
      createAccountLabel: "Create an account",
      steps: [
        {
          title: "Route preserved",
          detail: "Your destination is kept so you can return directly after authentication.",
          icon: "paper",
        },
        {
          title: "Session protected",
          detail: "Authentication runs before protected data and actions are reloaded.",
          icon: "workspace",
        },
        {
          title: "Back to work",
          detail: "You land on the requested surface instead of a generic dashboard.",
          icon: "chat",
        },
      ],
    }
  }

  return {
    title: "Welcome back",
    eyebrow: "Secure Access",
    description: "Sign in to your account to continue where you left off in PaperBot.",
    destination,
    destinationLabel: "Next destination",
    primaryActionLabel: "Sign in",
    createAccountLabel: "Create an account",
    steps: [
      {
        title: "Continue your workspace",
        detail: "Protected routes resume after sign-in without exposing the app shell first.",
        icon: "paper",
      },
      {
        title: "Keep your context",
        detail: "Paper, route, and session metadata stay attached to the callback destination.",
        icon: "workspace",
      },
      {
        title: "Return directly",
        detail: "Authentication sends you to the route you originally requested.",
        icon: "chat",
      },
    ],
  }
}

function stepIcon(step: AuthGateContext["steps"][number]["icon"]) {
  if (step === "workspace") return FolderOpen
  if (step === "chat") return MessageSquare
  return ShieldCheck
}

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
    <div className="min-h-screen bg-[#f5f6f1]">
      <div className="grid min-h-screen lg:grid-cols-[minmax(0,1.05fr)_minmax(420px,0.95fr)]">
        <div className="relative hidden overflow-hidden border-r border-slate-200 bg-[radial-gradient(circle_at_top_left,_rgba(232,237,227,0.95),_rgba(243,245,239,0.98)_58%,_rgba(250,250,247,1)_100%)] lg:flex">
          <div className="absolute inset-0 bg-[linear-gradient(135deg,rgba(255,255,255,0.65),transparent_52%)]" />
          <div className="relative flex w-full flex-col justify-between px-12 py-10">
            <div className="flex items-center gap-2.5 text-slate-800">
              <div className="flex h-9 w-9 items-center justify-center rounded-2xl border border-slate-200 bg-white">
                <BookOpen className="h-4 w-4" />
              </div>
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                  PaperBot
                </p>
                <p className="text-sm font-semibold text-slate-900">Authentication</p>
              </div>
            </div>

            <div className="max-w-[34rem] space-y-6">
              <div className="space-y-3">
                <span className="inline-flex rounded-full border border-slate-200 bg-white px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-600">
                  {gate.eyebrow}
                </span>
                <div className="space-y-2">
                  <h1 className="text-[34px] font-semibold leading-[1.05] tracking-[-0.03em] text-slate-950">
                    {gate.title}
                  </h1>
                  <p className="max-w-[30rem] text-[15px] leading-7 text-slate-600">
                    {gate.description}
                  </p>
                </div>
              </div>

              <div className="rounded-[28px] border border-slate-200 bg-white/90 p-5 shadow-[0_18px_40px_rgba(15,23,42,0.06)]">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                      {gate.destinationLabel}
                    </p>
                    <p className="mt-1 text-base font-semibold text-slate-900">{gate.title}</p>
                  </div>
                  <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.12em] text-emerald-700">
                    callback ready
                  </span>
                </div>
                <div className="mt-4 flex items-center gap-2 rounded-2xl border border-slate-200 bg-[#f7f8f4] px-3 py-2.5">
                  <ArrowRight className="h-4 w-4 text-slate-400" />
                  <span className="truncate font-mono text-[11px] text-slate-600">{gate.destination}</span>
                </div>
              </div>

              <div className="grid gap-3">
                {gate.steps.map((step, index) => {
                  const Icon = stepIcon(step.icon)
                  return (
                    <div
                      key={step.title}
                      className="flex items-start gap-3 rounded-[24px] border border-slate-200 bg-white/70 px-4 py-4"
                    >
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl border border-slate-200 bg-white text-slate-700">
                        <Icon className="h-4 w-4" />
                      </div>
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="rounded-full border border-slate-200 bg-white px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-[0.12em] text-slate-500">
                            {index + 1}
                          </span>
                          <p className="text-sm font-semibold text-slate-900">{step.title}</p>
                        </div>
                        <p className="mt-1 text-[13px] leading-6 text-slate-600">{step.detail}</p>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            <p className="text-xs text-slate-400">Secure authentication gate for protected PaperBot surfaces.</p>
          </div>
        </div>

        <div className="flex items-center justify-center px-6 py-10">
          <div className="w-full max-w-[430px] space-y-6">
            <div className="flex items-center gap-2 lg:hidden">
              <BookOpen className="h-5 w-5 text-slate-700" />
              <span className="font-semibold tracking-tight text-slate-900">PaperBot</span>
            </div>

            <div className="rounded-[30px] border border-slate-200 bg-white/95 p-6 shadow-[0_20px_44px_rgba(15,23,42,0.06)]">
              <div className="space-y-5">
                <div className="space-y-3">
                  <span className="inline-flex rounded-full border border-slate-200 bg-[#f7f8f4] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                    {gate.eyebrow}
                  </span>
                  <div className="space-y-1.5">
                    <h2 className="text-[28px] font-semibold tracking-[-0.03em] text-slate-950">
                      {gate.title}
                    </h2>
                    <p className="text-sm leading-6 text-slate-600">
                      {gate.description}
                    </p>
                  </div>
                </div>

                <div className="rounded-[22px] border border-slate-200 bg-[#f8faf5] p-4">
                  <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                    {gate.destinationLabel}
                  </p>
                  <p className="mt-1 break-all font-mono text-[11px] leading-5 text-slate-700">
                    {gate.destination}
                  </p>
                </div>

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
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function GitHubIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor" aria-hidden="true">
      <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z" />
    </svg>
  )
}
