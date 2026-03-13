"use client"

import { useState } from "react"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { Sidebar } from "./Sidebar"

export function LayoutShell({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(true)
  const pathname = usePathname()
  const isAgentBoardFocusPage = pathname.startsWith("/studio/agent-board/")

  return (
    <div className="flex min-h-screen">
      {!isAgentBoardFocusPage && (
        <aside
          className={cn(
            "fixed inset-y-0 z-50 hidden flex-col border-r bg-background transition-all duration-200 md:flex",
            collapsed ? "w-14" : "w-56",
          )}
        >
          <Sidebar collapsed={collapsed} onToggle={() => setCollapsed(!collapsed)} />
        </aside>
      )}
      <main
        className={cn(
          "flex-1 transition-all duration-200",
          !isAgentBoardFocusPage && (collapsed ? "md:pl-14" : "md:pl-56"),
        )}
      >
        {children}
      </main>
    </div>
  )
}
