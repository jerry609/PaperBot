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
    muted?: boolean
    last_seen_at?: string | null
    last_seen_cached_papers?: number
    digest_enabled?: boolean
    digest_frequency?: "daily" | "weekly" | "monthly"
    alert_enabled?: boolean
    alert_keywords?: string[]
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
    paper_count?: number
    track_count?: number
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

export interface IntelligenceMatchedTrack {
    track_id: number
    track_name: string
    matched_keywords: string[]
}

export interface IntelligenceFeedItem {
    id: string
    source: "reddit" | "github" | "huggingface" | "twitter_x" | string
    source_label: string
    kind: string
    title: string
    summary: string
    url?: string
    repo_full_name?: string
    author_name?: string
    keyword_hits: string[]
    author_matches: string[]
    repo_matches: string[]
    match_reasons: string[]
    score: number
    metric: {
        name: string
        value: number
        delta: number
    }
    published_at?: string | null
    detected_at?: string | null
    matched_tracks: IntelligenceMatchedTrack[]
    research_query?: string
    payload?: Record<string, unknown>
}

export interface IntelligenceFeedResponse {
    items: IntelligenceFeedItem[]
    refreshed_at?: string | null
    refresh_scheduled?: boolean
    keywords: string[]
    watch_repos: string[]
    subreddits: string[]
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
        matched_terms?: string[]
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

export interface ResearchTrackContextTask {
    id: number
    track_id: number
    title: string
    status?: string
    priority?: number
    paper_id?: string | null
    paper_url?: string | null
    metadata?: Record<string, unknown>
    created_at?: string | null
    updated_at?: string | null
    done_at?: string | null
}

export interface ResearchTrackContextMilestone {
    id: number
    track_id: number
    name: string
    status?: string
    notes?: string
    due_at?: string | null
    created_at?: string | null
    updated_at?: string | null
}

export interface ResearchTrackContextMemorySummary {
    total_items: number
    approved_items: number
    pending_items: number
    top_tags: string[]
    latest_memory_at?: string | null
}

export interface ResearchTrackContextFeedbackItem {
    id: number
    track_id: number
    paper_id: string
    action: string
    ts?: string | null
    metadata?: Record<string, unknown>
}

export interface ResearchTrackContextFeedbackSummary {
    total_items: number
    actions: Record<string, number>
    latest_feedback_at?: string | null
    recent_items: ResearchTrackContextFeedbackItem[]
}

export interface ResearchTrackSavedPaperPreview {
    paper?: TrackFeedPaper & {
        authors?: string[]
        abstract?: string | null
        url?: string | null
    }
    saved_at?: string | null
    latest_judge?: {
        overall?: number | null
        recommendation?: string | null
    } | null
}

export interface ResearchTrackContextSavedPapersSummary {
    total_items: number
    latest_saved_at?: string | null
    recent_items: ResearchTrackSavedPaperPreview[]
}

export interface ResearchTrackContextResponse {
    user_id: string
    track_id: number
    track: ResearchTrackSummary
    tasks: ResearchTrackContextTask[]
    milestones: ResearchTrackContextMilestone[]
    memory: ResearchTrackContextMemorySummary
    feedback: ResearchTrackContextFeedbackSummary
    saved_papers: ResearchTrackContextSavedPapersSummary
    eval_summary: Record<string, unknown>
}
