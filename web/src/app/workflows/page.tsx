import Link from "next/link"

import TopicWorkflowDashboard from "@/components/research/TopicWorkflowDashboard"

type WorkflowsPageProps = {
  searchParams?: Promise<Record<string, string | string[] | undefined>>
}

export default async function WorkflowsPage({ searchParams }: WorkflowsPageProps) {
  const params = searchParams ? await searchParams : {}
  const rawQuery = params?.query
  const queryValue = Array.isArray(rawQuery) ? rawQuery[0] : rawQuery
  const initialQueries = queryValue
    ? queryValue
        .split(",")
        .map((q) => q.trim())
        .filter(Boolean)
    : undefined

  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <header className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-3xl">
            <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-indigo-600">
              Workflows
            </p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-900">
              Workflow Workbench
            </h1>
            <p className="mt-3 text-sm leading-6 text-slate-600">
              这里承载完整的 Search、DailyPaper、Analyze 与 Daily Dispatch 配置。Dashboard 只保留状态快照和继续工作的入口，不再嵌入整块操作台。
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <Link
              href="/dashboard"
              className="inline-flex min-h-11 items-center justify-center rounded-full border border-slate-200 bg-white px-5 text-sm font-semibold text-slate-700 transition-colors hover:border-slate-300 hover:bg-slate-50"
            >
              返回 Dashboard
            </Link>
            <Link
              href="/research"
              className="inline-flex min-h-11 items-center justify-center rounded-full bg-slate-900 px-5 text-sm font-semibold text-white transition-colors hover:bg-slate-800"
            >
              打开 Research
            </Link>
          </div>
        </div>
      </header>

      <TopicWorkflowDashboard initialQueries={initialQueries} />
    </div>
  )
}
