import { Suspense } from "react"

import { SkillsDirectoryPage } from "@/components/skills/SkillsExperience"

export default function SkillsPage() {
  return (
    <Suspense fallback={null}>
      <SkillsDirectoryPage />
    </Suspense>
  )
}
