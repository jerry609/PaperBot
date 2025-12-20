"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { GripVertical, Clock, Sparkles } from "lucide-react"
import type { ReadingQueueItem } from "@/lib/types"
import Link from "next/link"

interface ReadingQueueProps {
    items: ReadingQueueItem[]
}

export function ReadingQueue({ items }: ReadingQueueProps) {
    return (
        <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Reading Queue</CardTitle>
                <Badge variant="secondary">{items.length} papers</Badge>
            </CardHeader>
            <CardContent className="space-y-3">
                {items.map((item) => (
                    <div key={item.id} className="flex items-center gap-3 p-2 rounded-md hover:bg-muted/50 transition-colors group">
                        <GripVertical className="h-4 w-4 text-muted-foreground cursor-grab" />
                        <div className="flex-1 min-w-0">
                            <Link href={`/papers/${item.paper_id}`} className="text-sm font-medium truncate block hover:underline">
                                {item.title}
                            </Link>
                            <div className="flex items-center gap-2 text-xs text-muted-foreground">
                                <Clock className="h-3 w-3" />
                                <span>{item.estimated_time}</span>
                            </div>
                        </div>
                        <Button size="sm" variant="ghost" className="opacity-0 group-hover:opacity-100 transition-opacity">
                            <Sparkles className="h-3 w-3 mr-1" /> Summarize
                        </Button>
                    </div>
                ))}
            </CardContent>
        </Card>
    )
}
