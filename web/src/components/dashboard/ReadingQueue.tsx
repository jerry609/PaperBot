"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Clock } from "lucide-react"
import type { ReadingQueueItem } from "@/lib/types"
import Link from "next/link"

interface ReadingQueueProps {
    items: ReadingQueueItem[]
}

export function ReadingQueue({ items }: ReadingQueueProps) {
    return (
        <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 p-3 pb-2">
                <CardTitle className="text-xs font-medium">Queue</CardTitle>
                <Badge variant="secondary" className="text-[10px] px-1.5 py-0">{items.length}</Badge>
            </CardHeader>
            <CardContent className="p-3 pt-0 space-y-1.5">
                {items.map((item) => (
                    <div key={item.id} className="flex items-center gap-2 p-1.5 rounded hover:bg-muted/50 text-xs">
                        <div className="flex-1 min-w-0">
                            <Link href={`/papers/${item.paper_id}`} className="font-medium truncate block hover:underline text-xs">
                                {item.title}
                            </Link>
                            <div className="flex items-center gap-1 text-[10px] text-muted-foreground">
                                <Clock className="h-2.5 w-2.5" />
                                <span>{item.estimated_time}</span>
                            </div>
                        </div>
                    </div>
                ))}
            </CardContent>
        </Card>
    )
}

