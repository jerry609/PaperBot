import { StudioMonitorWorkspace } from "@/components/studio/StudioMonitorWorkspace"

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
  const workerRunIdValue = Array.isArray(params?.workerRunId) ? params.workerRunId[0] : params?.workerRunId
  const workerRunIdLegacyValue = Array.isArray(params?.worker_run_id) ? params.worker_run_id[0] : params?.worker_run_id
  const workerRunId = workerRunIdValue || workerRunIdLegacyValue

  return (
    <StudioMonitorWorkspace
      initialPaperId={paperId ?? null}
      initialWorkerRunId={workerRunId ?? null}
    />
  )
}
