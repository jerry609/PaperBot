"use client"

import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Calendar, BarChart2, Code2, ArrowUpRight, X } from "lucide-react"
import { Activity } from "@/lib/types"

interface NewPaperCardProps {
    activity: Activity
}

export function NewPaperCard({ activity }: NewPaperCardProps) {
    if (!activity.paper || !activity.scholar) return null

    return (
        <Card className="hover:shadow-md transition-shadow">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <div className="flex items-center gap-3">
                    {activity.paper.is_influential && (
                        <Badge variant="default" className="bg-blue-500 hover:bg-blue-600">NEW</Badge>
                    )}
                    <span className="text-sm font-medium text-foreground">
                        {activity.scholar.name} published a new paper
                    </span>
                </div>
                <span className="text-xs text-muted-foreground">{activity.timestamp}</span>
            </CardHeader>
            <CardContent className="pt-2">
                <div className="flex flex-col gap-4">
                    {/* Paper Title & Info */}
                    <div>
                        <h3 className="text-lg font-bold leading-tight mb-2">
                            {activity.paper.title}
                        </h3>
                        <div className="flex flex-wrap gap-2 text-sm text-muted-foreground mb-3">
                            <span className="flex items-center gap-1">
                                <Calendar className="h-3 w-3" /> {activity.paper.venue} {activity.paper.year}
                            </span>
                            <span>•</span>
                            <span>{activity.paper.citations} citations</span>
                        </div>
                        <p className="text-sm text-muted-foreground line-clamp-2">
                            {activity.paper.abstract_snippet}
                        </p>
                    </div>

                    {/* Tags */}
                    <div className="flex flex-wrap gap-2">
                        {activity.paper.tags.map(tag => (
                            <Badge key={tag} variant="secondary" className="bg-blue-50 text-blue-700 hover:bg-blue-100 dark:bg-blue-900/20 dark:text-blue-300">
                                {tag}
                            </Badge>
                        ))}
                    </div>

                    {/* Footer Actions */}
                    <div className="flex items-center justify-between pt-2 border-t mt-1">
                        <div className="flex gap-4">
                            <div className="flex items-center gap-1 text-xs font-semibold text-green-600 dark:text-green-400">
                                <BarChart2 className="h-3 w-3" /> {activity.paper.citations} citations
                            </div>
                            <div className="flex items-center gap-1 text-xs font-semibold text-teal-600 dark:text-teal-400">
                                <Code2 className="h-3 w-3" /> Code available
                            </div>
                        </div>
                        <div className="flex gap-2 text-sm">
                            <Button variant="ghost" size="sm" className="h-8 text-muted-foreground hover:text-foreground">
                                <X className="h-3 w-3 mr-1" /> Ignore
                            </Button>
                            <Button variant="ghost" size="sm" className="h-8 text-primary hover:text-primary/90 font-medium">
                                View paper <ArrowUpRight className="ml-1 h-3 w-3" />
                            </Button>
                        </div>
                    </div>
                </div>
            </CardContent>
        </Card>
    )
}
