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

export interface PaperDetails extends Paper {
    abstract: string
    tldr: string
    pis_score: number
    impact_radar: { subject: string; A: number; fullMark: number }[]
    sentiment_analysis: { name: string; value: number; fill: string }[]
    citation_velocity: { month: string; citations: number }[]
    reproduction: {
        status: string
        logs: string[]
        dockerfile: string
    }
}

export interface TrendingTopic {
    text: string
    value: number
}


export type ActivityType = "published" | "alert" | "repro" | "milestone" | "conference"

export interface Activity {
    id: string
    type: ActivityType
    timestamp: string
    // Type-specific fields
    paper?: {
        title: string
        venue: string
        year: string
        citations: number
        tags: string[]
        abstract_snippet: string
        is_influential?: boolean
    }
    scholar?: {
        name: string
        avatar: string
        affiliation: string
    }
    milestone?: {
        title: string
        description: string
        current_value: number
        target_value?: number
        trend: "up" | "down" | "flat"
    }
    conference?: {
        name: string
        location: string
        date: string
        deadline_countdown: string
    }
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
    definition: string
    related_papers: string[]
    related_concepts: string[]
    examples: string[]
    category: "Method" | "Task" | "Metric" | "Architecture" | "Dataset"
    icon: string // Lucide icon name or identifier
}

export interface PipelineTask {
    id: string
    paper_title: string
    status: "downloading" | "analyzing" | "building" | "testing" | "success" | "failed"
    progress: number
    started_at: string
}

export interface ReadingQueueItem {
    id: string
    paper_id: string
    title: string
    estimated_time: string
    priority: number
}

export interface LLMUsageRecord {
    date: string
    gpt4: number
    claude: number
    ollama: number
}
