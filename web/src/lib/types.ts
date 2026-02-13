export interface Scholar {
    id: string
    semantic_scholar_id?: string
    name: string
    affiliation: string
    h_index: number
    papers_tracked: number
    recent_activity: string
    status: "active" | "idle"
    keywords?: string[]
    cached_papers?: number
    last_updated?: string | null
}

export interface Paper {
    id: string
    title: string
    venue: string
    authors: string
    citations: string | number
    status: "pending" | "analyzing" | "Reproduced" | "Saved"
    tags: string[]
    url?: string
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


export type TimelineItemKind = "harvest" | "save" | "note"

export interface TimelineItem {
    id: string
    kind: TimelineItemKind
    title: string
    subtitle?: string
    timestamp: string
}

export interface SavedPaper {
    id: string
    paper_id: string
    title: string
    authors: string
    saved_at: string
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
    trend_summary?: {
        publication_trend: "up" | "down" | "flat" | string
        citation_trend: "up" | "down" | "flat" | string
        window?: number
    }
    publication_velocity?: Array<{ year: number; papers: number; citations: number }>
    top_topics?: Array<{ topic: string; count: number }>
    top_venues?: Array<{ venue: string; count: number }>
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

export interface LLMUsageDailyRecord {
    date: string
    total_tokens: number
    total_cost_usd: number
    providers: Record<string, number>
}

export interface LLMUsageProviderModelRecord {
    provider_name: string
    model_name: string
    calls: number
    total_tokens: number
    total_cost_usd: number
}

export interface LLMUsageSummary {
    window_days: number
    daily: LLMUsageDailyRecord[]
    provider_models: LLMUsageProviderModelRecord[]
    totals: {
        calls: number
        total_tokens: number
        total_cost_usd: number
    }
}

export interface Activity {
    id: string
    type: "published" | "milestone" | "conference"
    timestamp: string
    scholar?: {
        name: string
        avatar: string
        affiliation: string
    }
    paper?: {
        title: string
        venue: string
        year: string
        citations: number
        tags: string[]
        abstract_snippet?: string
        is_influential?: boolean
    }
    milestone?: {
        title: string
        description: string
        current_value?: number
        trend?: "up" | "down" | "flat"
    }
    conference?: {
        name: string
        location: string
        date: string
        deadline_countdown?: string
    }
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
    authors?: string
    saved_at?: string
    estimated_time?: string
    priority?: number
    status?: "unread" | "reading" | "done"
}

export interface DeadlineRadarItem {
    name: string
    ccf_level: string
    field: string
    deadline: string
    days_left: number
    url: string
    keywords: string[]
    workflow_query: string
    matched_tracks: Array<{
        track_id: number
        track_name: string
        matched_keywords: string[]
    }>
}

export interface ResearchTrackSummary {
    id: number
    name: string
    description?: string
    keywords?: string[]
    methods?: string[]
    venues?: string[]
    is_active?: boolean
}

export interface TrackFeedPaper {
    id: number | string
    title: string
    year?: number | null
    venue?: string | null
    citation_count?: number
}

export interface TrackFeedItem {
    paper: TrackFeedPaper
    feed_score: number
    matched_terms: string[]
    latest_feedback_action?: string | null
    latest_judge?: {
        overall?: number | null
        recommendation?: string | null
    } | null
}

export interface AnchorPreviewItem {
    author_id: number
    name: string
    anchor_score: number
    anchor_level?: string
    user_action?: "follow" | "ignore" | null
}
