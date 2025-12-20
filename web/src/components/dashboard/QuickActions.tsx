"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { RefreshCw, Download, FileText, Sparkles } from "lucide-react"

export function QuickActions() {
    return (
        <Card>
            <CardHeader className="p-3 pb-2">
                <CardTitle className="text-xs font-medium">Actions</CardTitle>
            </CardHeader>
            <CardContent className="p-3 pt-0 grid grid-cols-2 gap-1.5">
                <Button variant="outline" size="sm" className="h-auto py-2 flex-col gap-0.5 text-xs">
                    <RefreshCw className="h-3.5 w-3.5" />
                    <span className="text-[10px]">Update</span>
                </Button>
                <Button variant="outline" size="sm" className="h-auto py-2 flex-col gap-0.5 text-xs">
                    <Download className="h-3.5 w-3.5" />
                    <span className="text-[10px]">Import</span>
                </Button>
                <Button variant="outline" size="sm" className="h-auto py-2 flex-col gap-0.5 text-xs">
                    <FileText className="h-3.5 w-3.5" />
                    <span className="text-[10px]">Report</span>
                </Button>
                <Button variant="outline" size="sm" className="h-auto py-2 flex-col gap-0.5 text-xs">
                    <Sparkles className="h-3.5 w-3.5" />
                    <span className="text-[10px]">AI</span>
                </Button>
            </CardContent>
        </Card>
    )
}

