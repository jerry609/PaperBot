"use client"

import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { BarChart, Star, Laptop, ArrowRight } from "lucide-react"
import { Activity } from "@/lib/types"

interface NewPaperCardProps {
    activity: Activity
}

export function NewPaperCard({ activity }: NewPaperCardProps) {
    if (!activity.paper || !activity.scholar) return null

    return (
        <Card className="bg-white rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
            <CardContent className="p-6 flex flex-col gap-4">

                {/* Header Section */}
                <div>
                    <h3 className="text-lg font-bold text-slate-800 leading-tight">
                        {activity.paper.title}
                    </h3>
                    <div className="flex items-center gap-2 mt-1">
                        <span className="text-sm text-slate-500">
                            {activity.scholar.name}, {activity.paper.venue} · {activity.scholar.affiliation}
                        </span>
                        {/* Optional Affiliation Badge */}
                        <Badge variant="secondary" className="hidden sm:inline-flex h-5 rounded-full px-2 text-[10px] bg-blue-50 text-blue-700 font-normal hover:bg-blue-100">
                            {activity.scholar.affiliation}
                        </Badge>
                    </div>
                </div>

                {/* Abstract Preview */}
                <p className="text-sm text-slate-500 leading-relaxed line-clamp-2">
                    {activity.paper.abstract_snippet || "No abstract available for this paper."}...
                </p>

                {/* Tags Row */}
                <div className="flex flex-wrap gap-2">
                    <Badge variant="secondary" className="rounded-full bg-blue-100 text-blue-800 hover:bg-blue-200 border-none font-medium px-3">
                        {activity.paper.venue}
                    </Badge>
                    <Badge variant="secondary" className="rounded-full bg-slate-100 text-slate-600 hover:bg-slate-200 border-none font-medium px-3">
                        {activity.paper.year}
                    </Badge>
                    {/* Add a static field tag as example since it's in the design spec but not in mock data explicitly */}
                    <Badge variant="secondary" className="rounded-full bg-teal-100 text-teal-800 hover:bg-teal-200 border-none font-medium px-3">
                        Security
                    </Badge>
                </div>

                {/* Metrics Row */}
                <div className="flex items-center gap-6 py-1">
                    <div className="flex items-center gap-1.5 text-slate-600">
                        <BarChart className="h-4 w-4 text-teal-600" />
                        <span className="text-xs font-medium">{activity.paper.citations} citations</span>
                    </div>

                    {/* Mock influential count since it's not in Activity type yet, or derive it */}
                    <div className="flex items-center gap-1.5 text-slate-600">
                        <Star className="h-4 w-4 text-teal-600" />
                        <span className="text-xs font-medium">45 influential</span>
                    </div>

                    <div className="flex items-center gap-1.5 text-slate-600">
                        <Laptop className="h-4 w-4 text-teal-600" />
                        <span className="text-xs font-medium">Code available</span>
                    </div>
                </div>

                {/* Footer Section */}
                <div className="flex items-center justify-between pt-2">
                    <div className="text-xs text-slate-400">
                        Published: {activity.timestamp} · Indexed: Today
                    </div>
                    <Button
                        variant="ghost"
                        size="sm"
                        className="text-teal-600 hover:text-teal-700 hover:bg-teal-50 px-0 h-auto font-medium"
                    >
                        View details <ArrowRight className="ml-1 h-3.5 w-3.5" />
                    </Button>
                </div>

            </CardContent>
        </Card>
    )
}
