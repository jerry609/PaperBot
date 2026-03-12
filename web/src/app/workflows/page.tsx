import TopicWorkflowDashboard from "@/components/research/TopicWorkflowDashboard"

type WorkflowsPageProps = {
  searchParams?: Promise<Record<string, string | string[] | undefined>>
}

export default async function WorkflowsPage({ searchParams }: WorkflowsPageProps) {
  const params = searchParams ? await searchParams : {}
  const queryValue = Array.isArray(params?.query) ? params.query[0] : params?.query
  const initialQueries =
    typeof queryValue === "string"
      ? queryValue
          .split(",")
          .map((value) => value.trim())
          .filter(Boolean)
      : undefined

  return (
    <div className="min-h-screen bg-stone-50/50 pb-12 text-slate-900">
      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <div className="space-y-4">
          <header className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
            <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-indigo-600">Workflows</p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
              完整工作台
            </h1>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600">
              Search、DailyPaper、Analyze 和交付都留在这一页。首页只保留一次运行快照与热点摘要，避免完整控制台继续挤占 dashboard。
            </p>
          </header>

          <TopicWorkflowDashboard initialQueries={initialQueries} />
        </div>
      </main>
    </div>
  )
}
