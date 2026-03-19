import { redirect } from "next/navigation"

type WorkflowsPageProps = {
  searchParams?: Promise<Record<string, string | string[] | undefined>>
}

export default async function WorkflowsPage({ searchParams }: WorkflowsPageProps) {
  const params = searchParams ? await searchParams : {}
  const nextSearch = new URLSearchParams()

  for (const [key, rawValue] of Object.entries(params || {})) {
    if (Array.isArray(rawValue)) {
      rawValue.forEach((value) => {
        if (typeof value === "string" && value.trim()) {
          nextSearch.append(key, value)
        }
      })
      continue
    }

    if (typeof rawValue === "string" && rawValue.trim()) {
      nextSearch.set(key, rawValue)
    }
  }

  redirect(nextSearch.size > 0 ? `/research?${nextSearch.toString()}` : "/research")
}
