export type AuthGateStepIcon = "paper" | "workspace" | "chat"

export type AuthGateStep = {
  title: string
  detail: string
  icon: AuthGateStepIcon
}

export type AuthGateContext = {
  title: string
  eyebrow: string
  description: string
  destination: string
  destinationLabel: string
  primaryActionLabel: string
  createAccountLabel: string
  steps: AuthGateStep[]
}

export function normalizeCallbackPath(callbackUrl: string): string {
  try {
    const parsed = new URL(callbackUrl, "http://localhost")
    return `${parsed.pathname}${parsed.search}${parsed.hash}` || "/dashboard"
  } catch {
    return callbackUrl.startsWith("/") ? callbackUrl : "/dashboard"
  }
}

export function buildAuthGateContext(callbackUrl: string): AuthGateContext {
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

export function buildLoginHref(callbackUrl: string): string {
  return callbackUrl.startsWith("/login")
    ? callbackUrl
    : `/login?callbackUrl=${encodeURIComponent(callbackUrl)}`
}
