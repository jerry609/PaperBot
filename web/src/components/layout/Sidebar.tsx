"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { useSession, signOut } from "next-auth/react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  LayoutDashboard,
  Users,
  FileText,
  FlaskConical,
  Code2,
  Sparkles,
  Settings,
  BookOpen,
  Workflow,
  PanelLeftClose,
  PanelLeft,
  Rocket,
  User as UserIcon,
  LogOut,
  LogIn,
  ChevronUp,
} from "lucide-react"

type SidebarProps = React.HTMLAttributes<HTMLDivElement> & {
  collapsed?: boolean
  onToggle?: () => void
}

const routes = [
  { label: "Dashboard", icon: LayoutDashboard, href: "/dashboard" },
  { label: "Research", icon: FlaskConical, href: "/research" },
  { label: "Scholars", icon: Users, href: "/scholars" },
  { label: "Papers", icon: FileText, href: "/papers" },
  { label: "Workflows", icon: Workflow, href: "/workflows" },
  { label: "Skills", icon: Sparkles, href: "/skills" },
  { label: "DeepCode Studio", icon: Code2, href: "/studio" },
  { label: "Wiki", icon: BookOpen, href: "/wiki" },
  { label: "Settings", icon: Settings, href: "/settings" },
]

export function Sidebar({ className, collapsed, onToggle }: SidebarProps) {
  const pathname = usePathname()
  const demoUrl = process.env.NEXT_PUBLIC_DEMO_URL
  const { data: session } = useSession()

  const isAuthenticated = !!session
  const displayName =
    session?.user?.name ||
    session?.user?.email ||
    String(session?.userId ?? "") ||
    "Guest"

  return (
    <div className={cn("flex min-h-screen flex-col border-r bg-background pb-12", className)}>
      <div className="space-y-4 py-4">
        <div className={cn("px-3 py-2", collapsed && "px-2")}>
          {/* Header */}
          <div className={cn("mb-2 flex items-center justify-between", collapsed ? "px-1" : "px-4")}>
            {!collapsed && (
              <h2 className="text-lg font-bold tracking-tight bg-gradient-to-r from-blue-600 to-cyan-500 text-transparent bg-clip-text">
                PaperBot
              </h2>
            )}
            <Button
              variant="ghost"
              size="icon"
              className="size-7"
              onClick={onToggle}
            >
              {collapsed ? <PanelLeft className="size-4" /> : <PanelLeftClose className="size-4" />}
            </Button>
          </div>

          {/* Nav items */}
          <div className="space-y-1">
            {routes.map((route) => {
              const isActive = pathname === route.href || pathname.startsWith(`${route.href}/`)
              return (
                <Button
                  key={route.href}
                  variant={isActive ? "secondary" : "ghost"}
                  className={cn("w-full", collapsed ? "justify-center px-0" : "justify-start")}
                  asChild
                  title={collapsed ? route.label : undefined}
                >
                  <Link href={route.href}>
                    <route.icon className={cn("h-4 w-4", !collapsed && "mr-2")} />
                    {!collapsed && route.label}
                  </Link>
                </Button>
              )
            })}
          </div>
        </div>
      </div>

      <div className={cn("mt-auto px-3 pb-4 space-y-2", collapsed && "px-2")}>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              className={cn(
                "flex w-full items-center rounded-md border px-2 py-2 text-xs",
                "hover:bg-accent hover:text-accent-foreground transition-colors",
                collapsed ? "justify-center px-0" : "justify-between",
              )}
            >
              <div className="flex items-center gap-2 min-w-0">
                <UserIcon className="h-4 w-4 shrink-0" />
                {!collapsed && (
                  <div className="flex min-w-0 flex-col text-left">
                    <span className="truncate font-medium">{displayName}</span>
                    <span className="text-[10px] text-muted-foreground">
                      {isAuthenticated ? "Signed in" : "Not signed in"}
                    </span>
                  </div>
                )}
              </div>
              {!collapsed && <ChevronUp className="h-3 w-3 shrink-0 text-muted-foreground" />}
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent side="top" align="start" className="w-52">
            <DropdownMenuLabel className="font-normal">
              <div className="flex flex-col space-y-0.5">
                <span className="text-sm font-medium">{displayName}</span>
                {isAuthenticated && (
                  <span className="text-xs text-muted-foreground truncate">
                    {session?.user?.email}
                  </span>
                )}
              </div>
            </DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem asChild>
              <Link href="/settings" className="cursor-pointer">
                <Settings className="mr-2 h-4 w-4" />
                Settings
              </Link>
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            {isAuthenticated ? (
              <DropdownMenuItem
                className="text-destructive focus:text-destructive cursor-pointer"
                onClick={() => signOut({ callbackUrl: "/login" })}
              >
                <LogOut className="mr-2 h-4 w-4" />
                Sign out
              </DropdownMenuItem>
            ) : (
              <DropdownMenuItem asChild>
                <Link href="/login" className="cursor-pointer">
                  <LogIn className="mr-2 h-4 w-4" />
                  Sign in
                </Link>
              </DropdownMenuItem>
            )}
          </DropdownMenuContent>
        </DropdownMenu>

        {demoUrl ? (
          <Button
            asChild
            variant="outline"
            className={cn("w-full", collapsed ? "justify-center px-0" : "justify-start")}
            title={collapsed ? "Live Demo" : undefined}
          >
            <a href={demoUrl} target="_blank" rel="noreferrer">
              <Rocket className={cn("h-4 w-4", !collapsed && "mr-2")} />
              {!collapsed && "Live Demo"}
            </a>
          </Button>
        ) : null}
      </div>
    </div>
  )
}
