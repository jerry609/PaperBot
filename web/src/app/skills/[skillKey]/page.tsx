import { SkillDetailPage } from "@/components/skills/SkillsExperience"

export default async function SkillDetailRoute({
  params,
}: {
  params: Promise<{ skillKey: string }>
}) {
  const { skillKey } = await params
  return <SkillDetailPage skillKey={skillKey} />
}
