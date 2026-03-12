"use client"

import { useState, useMemo } from "react"
import { signIn } from "next-auth/react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { Loader2, Eye, EyeOff, BookOpen } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"

// ─── Password strength ────────────────────────────────────────────────────────

type Strength = { score: 0 | 1 | 2 | 3 | 4; label: string; color: string }

function getStrength(pw: string): Strength {
  if (!pw) return { score: 0, label: "", color: "" }
  let score = 0
  if (pw.length >= 8)  score++
  if (pw.length >= 12) score++
  if (/[A-Z]/.test(pw) && /[a-z]/.test(pw)) score++
  if (/\d/.test(pw) && /[^A-Za-z0-9]/.test(pw)) score++

  const levels: Strength[] = [
    { score: 0, label: "", color: "" },
    { score: 1, label: "Weak",   color: "bg-red-500" },
    { score: 2, label: "Fair",   color: "bg-orange-400" },
    { score: 3, label: "Good",   color: "bg-yellow-400" },
    { score: 4, label: "Strong", color: "bg-green-500" },
  ]
  return levels[score as 0 | 1 | 2 | 3 | 4]
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function RegisterPage() {
  const [displayName, setDisplayName] = useState("")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [confirm, setConfirm] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [githubLoading, setGithubLoading] = useState(false)
  const router = useRouter()

  const strength = useMemo(() => getStrength(password), [password])
  const mismatch = confirm.length > 0 && confirm !== password

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (password !== confirm) {
      setError("Passwords do not match.")
      return
    }
    setError(null)
    setLoading(true)

    const res = await fetch("/api/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, display_name: displayName || undefined }),
    })

    if (!res.ok) {
      const body = await res.json().catch(() => null)
      setError(body?.detail ?? "Registration failed. Please try again.")
      setLoading(false)
      return
    }

    const r = await signIn("credentials", { email, password, redirect: false })
    setLoading(false)
    if (r?.error) {
      setError("Account created. Please sign in.")
      router.push("/login")
    } else {
      router.push("/dashboard")
    }
  }

  const onGithub = () => {
    setGithubLoading(true)
    signIn("github", { callbackUrl: "/dashboard" })
  }

  return (
    <div className="flex min-h-screen">
      {/* Left branding panel */}
      <div className="hidden lg:flex lg:w-1/2 flex-col justify-between bg-zinc-950 p-12 text-white">
        <div className="flex items-center gap-2.5">
          <BookOpen className="h-5 w-5" />
          <span className="text-base font-semibold tracking-tight">PaperBot</span>
        </div>

        <div className="space-y-5">
          <p className="text-2xl font-medium leading-snug text-white/90">
            &ldquo;Your personal AI research assistant,<br />
            built for serious readers.&rdquo;
          </p>
          <div className="space-y-2.5 text-sm text-white/50">
            <BulletPoint text="Free to use — no credit card required" />
            <BulletPoint text="Connect via email or GitHub in seconds" />
            <BulletPoint text="Your data stays local by default" />
          </div>
        </div>

        <p className="text-xs text-white/25">© 2026 PaperBot. All rights reserved.</p>
      </div>

      {/* Right form panel */}
      <div className="flex w-full lg:w-1/2 items-center justify-center px-6 py-16">
        <div className="w-full max-w-sm space-y-7">
          {/* Mobile logo */}
          <div className="flex items-center gap-2 lg:hidden">
            <BookOpen className="h-5 w-5" />
            <span className="font-semibold tracking-tight">PaperBot</span>
          </div>

          {/* Heading */}
          <div className="space-y-1">
            <h1 className="text-2xl font-semibold tracking-tight">Create an account</h1>
            <p className="text-sm text-muted-foreground">
              Get started for free in under a minute
            </p>
          </div>

          {/* GitHub */}
          <Button
            variant="outline"
            className="w-full"
            onClick={onGithub}
            disabled={loading || githubLoading}
          >
            {githubLoading
              ? <Loader2 className="h-4 w-4 animate-spin" />
              : <GitHubIcon />
            }
            Sign up with GitHub
          </Button>

          <div className="flex items-center gap-3 text-muted-foreground">
            <Separator className="flex-1" />
            <span className="text-xs">or</span>
            <Separator className="flex-1" />
          </div>

          {/* Form */}
          <form onSubmit={onSubmit} className="space-y-4">
            <div className="space-y-1.5">
              <Label htmlFor="displayName">
                Display name{" "}
                <span className="text-muted-foreground font-normal">(optional)</span>
              </Label>
              <Input
                id="displayName"
                type="text"
                autoComplete="name"
                placeholder="Alex Johnson"
                value={displayName}
                onChange={e => setDisplayName(e.target.value)}
                disabled={loading}
              />
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="email">Email address</Label>
              <Input
                id="email"
                type="email"
                autoComplete="email"
                placeholder="name@example.com"
                value={email}
                onChange={e => setEmail(e.target.value)}
                disabled={loading}
                required
              />
            </div>

            {/* Password + strength meter */}
            <div className="space-y-1.5">
              <Label htmlFor="password">Password</Label>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  autoComplete="new-password"
                  placeholder="Min. 8 characters"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  disabled={loading}
                  required
                  minLength={8}
                  className="pr-9"
                />
                <button
                  type="button"
                  tabIndex={-1}
                  onClick={() => setShowPassword(v => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>

              {/* Strength bar — only visible once the user starts typing */}
              {password.length > 0 && (
                <div className="space-y-1">
                  <div className="flex gap-1">
                    {([1, 2, 3, 4] as const).map(n => (
                      <div
                        key={n}
                        className={`h-1 flex-1 rounded-full transition-colors duration-300 ${
                          strength.score >= n ? strength.color : "bg-muted"
                        }`}
                      />
                    ))}
                  </div>
                  {strength.label && (
                    <p className="text-xs text-muted-foreground">
                      Strength:{" "}
                      <span className={`font-medium ${
                        strength.score <= 1 ? "text-red-500"
                        : strength.score === 2 ? "text-orange-400"
                        : strength.score === 3 ? "text-yellow-500"
                        : "text-green-500"
                      }`}>
                        {strength.label}
                      </span>
                      {strength.score < 3 && (
                        <span className="ml-1 text-muted-foreground">
                          — try adding uppercase, numbers, or symbols
                        </span>
                      )}
                    </p>
                  )}
                </div>
              )}
            </div>

            {/* Confirm password */}
            <div className="space-y-1.5">
              <Label htmlFor="confirm">Confirm password</Label>
              <div className="relative">
                <Input
                  id="confirm"
                  type={showConfirm ? "text" : "password"}
                  autoComplete="new-password"
                  placeholder="Re-enter your password"
                  value={confirm}
                  onChange={e => setConfirm(e.target.value)}
                  disabled={loading}
                  required
                  className={`pr-9 ${mismatch ? "border-destructive focus-visible:ring-destructive/30" : ""}`}
                />
                <button
                  type="button"
                  tabIndex={-1}
                  onClick={() => setShowConfirm(v => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showConfirm ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
              {mismatch && (
                <p className="text-xs text-destructive">Passwords do not match.</p>
              )}
            </div>

            {error && (
              <p className="text-sm text-destructive">{error}</p>
            )}

            <Button
              type="submit"
              className="w-full"
              disabled={loading || githubLoading || mismatch}
            >
              {loading && <Loader2 className="h-4 w-4 animate-spin" />}
              Create account
            </Button>
          </form>

          <p className="text-center text-sm text-muted-foreground">
            Already have an account?{" "}
            <Link
              href="/login"
              className="font-medium text-foreground underline-offset-4 hover:underline"
            >
              Sign in
            </Link>
          </p>
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

function BulletPoint({ text }: { text: string }) {
  return (
    <div className="flex items-start gap-2">
      <span className="mt-0.5 select-none text-white/30">–</span>
      <span>{text}</span>
    </div>
  )
}
