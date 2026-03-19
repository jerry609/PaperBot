"use client"

import Link from "next/link"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Library, Search, Settings, Sparkles } from "lucide-react"

export function QuickActions() {
    return (
        <Card>
            <CardHeader className="p-3 pb-2">
                <CardTitle className="text-xs font-medium">Quick Actions</CardTitle>
            </CardHeader>
            <CardContent className="p-3 pt-0 grid grid-cols-2 gap-1.5">
                <Button asChild variant="outline" size="sm" className="h-auto py-2 flex-col gap-0.5 text-xs">
                    <Link href="/research">
                        <Search className="h-3.5 w-3.5" />
                        <span className="text-[10px]">Research</span>
                    </Link>
                </Button>
                <Button asChild variant="outline" size="sm" className="h-auto py-2 flex-col gap-0.5 text-xs">
                    <Link href="/skills">
                        <Sparkles className="h-3.5 w-3.5" />
                        <span className="text-[10px]">Skills</span>
                    </Link>
                </Button>
                <Button asChild variant="outline" size="sm" className="h-auto py-2 flex-col gap-0.5 text-xs">
                    <Link href="/papers">
                        <Library className="h-3.5 w-3.5" />
                        <span className="text-[10px]">Papers</span>
                    </Link>
                </Button>
                <Button asChild variant="outline" size="sm" className="h-auto py-2 flex-col gap-0.5 text-xs">
                    <Link href="/settings">
                        <Settings className="h-3.5 w-3.5" />
                        <span className="text-[10px]">Settings</span>
                    </Link>
                </Button>
            </CardContent>
        </Card>
    )
}
