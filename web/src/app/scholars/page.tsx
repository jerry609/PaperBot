import { ScholarsWatchlist } from "@/components/scholars/ScholarsWatchlist"
import { fetchScholars } from "@/lib/api"

export default async function ScholarsPage() {
  const scholars = await fetchScholars()
  return <ScholarsWatchlist scholars={scholars} />
}
