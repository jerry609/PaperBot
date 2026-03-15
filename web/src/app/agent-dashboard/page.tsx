import { redirect } from "next/navigation"

export default function AgentDashboardRedirectPage() {
  redirect("/studio?surface=board")
}
