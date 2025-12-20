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
