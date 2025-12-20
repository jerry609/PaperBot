"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { RefreshCw, Download, FileText, Sparkles } from "lucide-react"

export function QuickActions() {
    return (
        <Card>
            <CardHeader>
                <CardTitle className="text-sm font-medium">Quick Actions</CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-2">
                <Button variant="outline" className="h-auto py-3 flex-col gap-1">
                    <RefreshCw className="h-5 w-5" />
                    <span className="text-xs">Update Scholars</span>
                </Button>
                <Button variant="outline" className="h-auto py-3 flex-col gap-1">
                    <Download className="h-5 w-5" />
                    <span className="text-xs">Import List</span>
                </Button>
                <Button variant="outline" className="h-auto py-3 flex-col gap-1">
                    <FileText className="h-5 w-5" />
                    <span className="text-xs">Weekly Report</span>
                </Button>
                <Button variant="outline" className="h-auto py-3 flex-col gap-1">
                    <Sparkles className="h-5 w-5" />
                    <span className="text-xs">AI Summary</span>
                </Button>
            </CardContent>
        </Card>
    )
}
