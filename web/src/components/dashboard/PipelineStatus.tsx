"use client"

import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { RefreshCw, AlertCircle, CheckCircle2, Loader2 } from "lucide-react"
import type { PipelineTask } from "@/lib/types"

interface PipelineStatusProps {
    tasks: PipelineTask[]
}

const statusIcons = {
    downloading: <Loader2 className="h-4 w-4 animate-spin text-blue-500" />,
    analyzing: <Loader2 className="h-4 w-4 animate-spin text-yellow-500" />,
    building: <Loader2 className="h-4 w-4 animate-spin text-orange-500" />,
    testing: <Loader2 className="h-4 w-4 animate-spin text-purple-500" />,
    success: <CheckCircle2 className="h-4 w-4 text-green-500" />,
    failed: <AlertCircle className="h-4 w-4 text-red-500" />
}

export function PipelineStatus({ tasks }: PipelineStatusProps) {
    return (
        <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Reproduction Pipeline</CardTitle>
                <Badge variant="outline">{tasks.length} Active</Badge>
            </CardHeader>
            <CardContent className="space-y-4">
                {tasks.map((task) => (
                    <div key={task.id} className="space-y-2">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                {statusIcons[task.status]}
                                <span className="text-sm font-medium truncate max-w-[200px]">{task.paper_title}</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="text-xs text-muted-foreground">{task.started_at}</span>
                                {task.status === "failed" && (
                                    <Button size="icon" variant="ghost" className="h-6 w-6">
                                        <RefreshCw className="h-3 w-3" />
                                    </Button>
                                )}
                            </div>
                        </div>
                        <Progress value={task.progress} className={task.status === "failed" ? "bg-red-100" : ""} />
                    </div>
                ))}
            </CardContent>
        </Card>
    )
}
