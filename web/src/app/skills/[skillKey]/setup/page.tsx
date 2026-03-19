import { SkillSetupPage } from "@/components/skills/SkillsExperience"

export default async function SkillSetupRoute({
  params,
}: {
  params: Promise<{ skillKey: string }>
}) {
  const { skillKey } = await params
  return <SkillSetupPage skillKey={skillKey} />
}
