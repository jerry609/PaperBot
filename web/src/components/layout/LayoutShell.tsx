"use client"

import { useState } from "react"
import { cn } from "@/lib/utils"
import { Sidebar } from "./Sidebar"

export function LayoutShell({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <div className="flex min-h-screen">
      <aside
        className={cn(
          "fixed inset-y-0 z-50 hidden flex-col border-r bg-background transition-all duration-200 md:flex",
          collapsed ? "w-14" : "w-56",
        )}
      >
        <Sidebar collapsed={collapsed} onToggle={() => setCollapsed(!collapsed)} />
      </aside>
      <main className={cn("flex-1 transition-all duration-200", collapsed ? "md:pl-14" : "md:pl-56")}>
        {children}
      </main>
    </div>
  )
}
