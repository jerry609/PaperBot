import type { Metadata } from "next"

import { SkillsStudio } from "@/components/skills/SkillsStudio"
import { getSkillsStudioData } from "@/lib/skills-studio"

export const metadata: Metadata = {
  title: "Skills Studio | PaperBot",
  description: "A live capability map of the PaperBot dev architecture.",
}

export const dynamic = "force-dynamic"

export default async function SkillsPage() {
  const data = await getSkillsStudioData()
  return <SkillsStudio data={data} />
}
