"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"

export function QuickFilters() {
    return (
        <Card className="h-fit">
            <CardHeader>
                <CardTitle className="text-sm font-medium">Quick Filters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
                <div className="space-y-4">
                    <div className="flex items-center justify-between space-x-2">
                        <Label htmlFor="top-conf" className="flex flex-col">
                            <span className="font-medium">Top Venues</span>
                        </Label>
                        <Switch id="top-conf" defaultChecked />
                    </div>
                    <div className="flex items-center justify-between space-x-2">
                        <Label htmlFor="highly-cited" className="flex flex-col">
                            <span className="font-medium">Highly Cited</span>
                        </Label>
                        <Switch id="highly-cited" defaultChecked />
                    </div>
                    <div className="flex items-center justify-between space-x-2">
                        <Label htmlFor="rising-stars" className="flex flex-col">
                            <span className="font-medium">Rising Stars</span>
                        </Label>
                        <Switch id="rising-stars" />
                    </div>
                </div>

                <div className="space-y-3 pt-2">
                    <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Affiliation Types</h4>
                    <div className="items-top flex space-x-2">
                        <Checkbox id="univ" />
                        <div className="grid gap-1.5 leading-none">
                            <Label htmlFor="univ" className="text-sm font-normal leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                                Universities
                            </Label>
                        </div>
                    </div>
                    <div className="items-top flex space-x-2">
                        <Checkbox id="tech" />
                        <div className="grid gap-1.5 leading-none">
                            <Label htmlFor="tech" className="text-sm font-normal leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                                Tech Companies
                            </Label>
                        </div>
                    </div>
                </div>
            </CardContent>
        </Card>
    )
}
