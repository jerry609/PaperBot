export interface Scholar {
    id: string
    name: string
    affiliation: string
    h_index: number
    papers_tracked: number
    recent_activity: string
    status: "active" | "idle"
}

export interface Paper {
    id: string
    title: string
    venue: string
    authors: string
    citations: string | number
    status: "pending" | "analyzing" | "Reproduced"
    tags: string[]
}

export interface Activity {
    author: string
    action: "published" | "alert" | "repro"
    paper: string
    venue?: string
    detail?: string
    time: string
    type: "paper" | "alert" | "repro"
}

export interface Stats {
    tracked_scholars: number
    new_papers: number
    llm_usage: string
    read_later: number
}

export interface ScholarDetails extends Scholar {
    bio: string
    location: string
    website: string
    expertise_radar: { subject: string; A: number; fullMark: number }[]
    publications: Paper[]
    co_authors: { name: string; avatar: string }[]
    stats: {
        total_citations: number
        papers_count: number
        h_index: number
    }
}

export interface WikiConcept {
    id: string
    name: string
    description: string
    related_papers: string[]
    category: "Method" | "Task" | "Metric"
}
