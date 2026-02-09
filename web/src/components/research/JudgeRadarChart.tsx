"use client"

import { PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer } from "recharts"

type DimensionScore = {
  score?: number
}

type JudgeResult = {
  relevance?: DimensionScore
  novelty?: DimensionScore
  rigor?: DimensionScore
  impact?: DimensionScore
  clarity?: DimensionScore
}

type JudgeRadarChartProps = {
  judge?: JudgeResult
}

export default function JudgeRadarChart({ judge }: JudgeRadarChartProps) {
  const data = [
    { subject: "Rel", score: Number(judge?.relevance?.score || 0), fullMark: 5 },
    { subject: "Nov", score: Number(judge?.novelty?.score || 0), fullMark: 5 },
    { subject: "Rig", score: Number(judge?.rigor?.score || 0), fullMark: 5 },
    { subject: "Imp", score: Number(judge?.impact?.score || 0), fullMark: 5 },
    { subject: "Clr", score: Number(judge?.clarity?.score || 0), fullMark: 5 },
  ]

  return (
    <ResponsiveContainer width="100%" height={190}>
      <RadarChart data={data} outerRadius="72%">
        <PolarGrid />
        <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11 }} />
        <PolarRadiusAxis domain={[0, 5]} tick={false} axisLine={false} />
        <Radar name="Judge" dataKey="score" stroke="#2563eb" fill="#2563eb" fillOpacity={0.45} />
      </RadarChart>
    </ResponsiveContainer>
  )
}
