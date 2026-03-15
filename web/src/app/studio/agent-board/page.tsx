import { redirect } from "next/navigation"

type AgentBoardPageProps = {
  searchParams?: Promise<Record<string, string | string[] | undefined>>
}

export default async function AgentBoardPage({
  searchParams,
}: AgentBoardPageProps) {
  const params = searchParams ? await searchParams : {}
  const paperIdValue = Array.isArray(params?.paperId) ? params.paperId[0] : params?.paperId
  const paperIdLegacyValue = Array.isArray(params?.paper_id) ? params.paper_id[0] : params?.paper_id
  const paperId = paperIdValue || paperIdLegacyValue
  redirect(
    paperId
      ? `/studio?paperId=${encodeURIComponent(paperId)}&surface=board`
      : "/studio?surface=board",
  )
}
