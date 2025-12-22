"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts"
import type { LLMUsageRecord } from "@/lib/types"

interface LLMUsageChartProps {
    data: LLMUsageRecord[]
}

export function LLMUsageChart({ data }: LLMUsageChartProps) {
    return (
        <Card>
            <CardHeader>
                <CardTitle className="text-sm font-medium">LLM Token Usage (7 Days)</CardTitle>
            </CardHeader>
            <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" fontSize={12} />
                        <YAxis fontSize={12} tickFormatter={(v) => `${v / 1000}k`} />
                        <Tooltip formatter={(value) => value !== undefined ? `${Number(value).toLocaleString()} tokens` : ''} />
                        <Legend />
                        <Bar dataKey="gpt4" name="GPT-4" fill="#10b981" stackId="a" />
                        <Bar dataKey="claude" name="Claude" fill="#8b5cf6" stackId="a" />
                        <Bar dataKey="ollama" name="Ollama" fill="#6b7280" stackId="a" />
                    </BarChart>
                </ResponsiveContainer>
            </CardContent>
        </Card>
    )
}
