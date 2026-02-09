"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { useState } from "react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  LayoutDashboard,
  Users,
  FileText,
  FlaskConical,
  Code2,
  Settings,
  BookOpen,
  Workflow,
  PanelLeftClose,
  PanelLeft,
} from "lucide-react"

type SidebarProps = React.HTMLAttributes<HTMLDivElement> & {
  collapsed?: boolean
  onToggle?: () => void
}

const routes = [
  { label: "Dashboard", icon: LayoutDashboard, href: "/" },
  { label: "Scholars", icon: Users, href: "/scholars" },
  { label: "Papers", icon: FileText, href: "/papers" },
  { label: "Research", icon: FlaskConical, href: "/research" },
  { label: "Workflows", icon: Workflow, href: "/workflows" },
  { label: "DeepCode Studio", icon: Code2, href: "/studio" },
  { label: "Wiki", icon: BookOpen, href: "/wiki" },
  { label: "Settings", icon: Settings, href: "/settings" },
]

export function Sidebar({ className, collapsed, onToggle }: SidebarProps) {
  const pathname = usePathname()

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
              const isActive =
                route.href === "/" ? pathname === "/" : pathname.startsWith(route.href)
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
    </div>
  )
}
