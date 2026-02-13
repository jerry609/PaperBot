import {
    Activity,
    Paper,
    PaperDetails,
    Scholar,
    ScholarDetails,
    Stats,
    WikiConcept,
    TrendingTopic,
    PipelineTask,
    ReadingQueueItem,
    LLMUsageSummary,
    DeadlineRadarItem,
} from "./types"

const API_BASE_URL = (process.env.PAPERBOT_API_BASE_URL || "http://127.0.0.1:8000") + "/api"

function slugToName(slug: string): string {
    return slug
        .split("-")
        .filter(Boolean)
        .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
        .join(" ")
}

async function postJson<T>(path: string, payload: Record<string, unknown>): Promise<T | null> {
    try {
        const res = await fetch(`${API_BASE_URL}${path}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
            cache: "no-store",
        })
        if (!res.ok) return null
        return await res.json() as T
    } catch {
        return null
    }
}

export async function fetchStats(): Promise<Stats> {
    try {
        const [usage, papers, scholars] = await Promise.allSettled([
            fetchLLMUsage(),
            fetchPapers(),
            fetchScholars(),
        ])
        const usageData = usage.status === "fulfilled" ? usage.value : null
        const papersData = papers.status === "fulfilled" ? papers.value : []
        const scholarsData = scholars.status === "fulfilled" ? scholars.value : []
        const tokenCount = usageData?.totals?.total_tokens ?? 0
        const prettyTokens = tokenCount >= 1000 ? `${Math.round(tokenCount / 1000)}k` : `${tokenCount}`
        return {
            tracked_scholars: scholarsData.length,
            new_papers: papersData.length,
            llm_usage: prettyTokens,
            read_later: papersData.filter((p) => p.status === "Saved").length,
        }
    } catch {
        return { tracked_scholars: 0, new_papers: 0, llm_usage: "0", read_later: 0 }
    }
}

export async function fetchActivities(): Promise<Activity[]> {
    const activities: Activity[] = []
    try {
        // Fetch recent harvest runs
        const runsRes = await fetch(`${API_BASE_URL}/harvest/runs?limit=3`, { cache: "no-store" })
        if (runsRes.ok) {
            const runsData = await runsRes.json() as { runs?: Array<{ run_id: number; source: string; status: string; total_found: number; created_at: string }> }
            for (const run of (runsData.runs || []).slice(0, 2)) {
                activities.push({
                    id: `harvest-${run.run_id}`,
                    type: "milestone",
                    timestamp: new Date(run.created_at).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }),
                    milestone: {
                        title: `Harvest: ${run.source}`,
                        description: `Found ${run.total_found} papers (${run.status})`,
                        current_value: run.total_found,
                        trend: run.status === "completed" ? "up" : "flat",
                    },
                })
            }
        }
    } catch { /* keep going */ }

    try {
        // Fetch recent saved papers
        const savedRes = await fetch(`${API_BASE_URL}/research/papers/saved?user_id=default&limit=3`, { cache: "no-store" })
        if (savedRes.ok) {
            const savedData = await savedRes.json() as { papers?: Array<{ paper_id: string; title: string; authors?: string[]; venue?: string; year?: number; saved_at?: string }> }
            for (const paper of (savedData.papers || []).slice(0, 3)) {
                activities.push({
                    id: `saved-${paper.paper_id}`,
                    type: "published",
                    timestamp: paper.saved_at ? new Date(paper.saved_at).toLocaleDateString("en-US", { month: "short", day: "numeric" }) : "Recently",
                    scholar: {
                        name: (paper.authors || [])[0] || "Unknown",
                        avatar: `https://avatar.vercel.sh/${encodeURIComponent((paper.authors || ["unknown"])[0])}.png`,
                        affiliation: "",
                    },
                    paper: {
                        title: paper.title,
                        venue: paper.venue || "",
                        year: String(paper.year || ""),
                        citations: 0,
                        tags: [],
                        abstract_snippet: "",
                        is_influential: false,
                    },
                })
            }
        }
    } catch { /* keep going */ }

    return activities
}

export async function fetchTrendingTopics(): Promise<TrendingTopic[]> {
    try {
        const res = await fetch(`${API_BASE_URL}/research/tracks?user_id=default`, { cache: "no-store" })
        if (!res.ok) return []
        const data = await res.json() as { tracks?: Array<{ keywords?: string[] }> }
        const keywordCounts = new Map<string, number>()
        for (const track of data.tracks || []) {
            for (const kw of track.keywords || []) {
                keywordCounts.set(kw, (keywordCounts.get(kw) || 0) + 1)
            }
        }
        return Array.from(keywordCounts.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, 8)
            .map(([text, value]) => ({ text, value: value * 30 }))
    } catch {
        return []
    }
}

export async function fetchPipelineTasks(): Promise<PipelineTask[]> {
    try {
        const res = await fetch(`${API_BASE_URL}/harvest/runs?limit=5`, { cache: "no-store" })
        if (!res.ok) return []
        const data = await res.json() as { runs?: Array<{ run_id: number; source: string; status: string; total_found: number; created_at: string }> }
        return (data.runs || []).slice(0, 5).map((run) => {
            const statusMap: Record<string, PipelineTask["status"]> = { completed: "success", running: "building", failed: "failed", pending: "building" }
            const progressMap: Record<string, number> = { completed: 100, running: 50, failed: 100, pending: 10 }
            return {
                id: String(run.run_id),
                paper_title: `${run.source} harvest (${run.total_found} papers)`,
                status: statusMap[run.status] || "building" as PipelineTask["status"],
                progress: progressMap[run.status] || 0,
                started_at: new Date(run.created_at).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
            }
        })
    } catch {
        return []
    }
}

export async function fetchReadingQueue(): Promise<ReadingQueueItem[]> {
    try {
        const res = await fetch(`${API_BASE_URL}/research/papers/saved?user_id=default&limit=5`, { cache: "no-store" })
        if (!res.ok) return []
        const data = await res.json() as { papers?: Array<{ paper_id: string; title: string }> }
        return (data.papers || []).slice(0, 5).map((p, i) => ({
            id: String(i + 1),
            paper_id: p.paper_id,
            title: p.title,
            estimated_time: "",
            priority: i + 1,
        }))
    } catch {
        return []
    }
}

export async function fetchLLMUsage(days: number = 7): Promise<LLMUsageSummary> {
    try {
        const qs = new URLSearchParams({ days: String(days) })
        const res = await fetch(`${API_BASE_URL}/model-endpoints/usage?${qs.toString()}`, {
            cache: "no-store",
        })
        if (!res.ok) throw new Error("usage endpoint unavailable")
        const payload = await res.json() as { summary?: LLMUsageSummary }
        if (payload.summary) {
            return payload.summary
        }
    } catch {
        // Return empty summary when backend is unavailable.
    }

    return {
        window_days: days,
        daily: [],
        provider_models: [],
        totals: { calls: 0, total_tokens: 0, total_cost_usd: 0 },
    }
}

export async function fetchDeadlineRadar(userId: string = "default"): Promise<DeadlineRadarItem[]> {
    try {
        const qs = new URLSearchParams({
            user_id: userId,
            days: "180",
            ccf_levels: "A,B,C",
            limit: "10",
        })
        const res = await fetch(`${API_BASE_URL}/research/deadlines/radar?${qs.toString()}`, {
            cache: "no-store",
        })
        if (!res.ok) return []
        const payload = await res.json() as { items?: DeadlineRadarItem[] }
        return payload.items || []
    } catch {
        return []
    }
}


export async function fetchScholars(): Promise<Scholar[]> {
    try {
        const res = await fetch(`${API_BASE_URL}/research/scholars?limit=200`, { cache: "no-store" })
        if (!res.ok) return []
        const data = await res.json() as {
            items?: Array<{
                id?: string
                scholar_id?: string
                semantic_scholar_id?: string | null
                name?: string
                affiliation?: string
                keywords?: string[]
                h_index?: number
                paper_count?: number
                recent_activity?: string
                status?: "active" | "idle"
                cached_papers?: number
                last_updated?: string | null
            }>
        }
        return (data.items || []).map((row) => ({
            id: String(row.id || row.semantic_scholar_id || row.scholar_id || row.name || "unknown"),
            semantic_scholar_id: row.semantic_scholar_id || undefined,
            name: String(row.name || "Unknown"),
            affiliation: String(row.affiliation || "Unknown affiliation"),
            h_index: Number(row.h_index || 0),
            papers_tracked: Number(row.paper_count || 0),
            recent_activity: String(row.recent_activity || "No tracking runs yet"),
            status: row.status === "active" ? "active" : "idle",
            keywords: row.keywords || [],
            cached_papers: Number(row.cached_papers || 0),
            last_updated: row.last_updated || null,
        }))
    } catch {
        return []
    }
}

export async function fetchPaperDetails(id: string): Promise<PaperDetails> {
    // Mock data
    return {
        id,
        title: "Attention Is All You Need",
        venue: "NeurIPS 2017",
        authors: "Vaswani et al.",
        citations: "100k+",
        status: "Reproduced",
        tags: ["Transformer", "NLP"],
        abstract: "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
        tldr: "PROPOSED the Transformer, a novel network architecture based solely on attention mechanisms, which achieves state-of-the-art results in machine translation tasks while being parallelizable and requiring significantly less training time.",
        pis_score: 98,
        impact_radar: [
            { subject: 'Novelty', A: 120, fullMark: 150 },
            { subject: 'Accessibility', A: 98, fullMark: 150 },
            { subject: 'Rigor', A: 86, fullMark: 150 },
            { subject: 'Reproducibility', A: 99, fullMark: 150 },
            { subject: 'Impact', A: 145, fullMark: 150 },
            { subject: 'Clarity', A: 110, fullMark: 150 },
        ],
        sentiment_analysis: [
            { name: 'Positive', value: 400, fill: '#4ade80' },
            { name: 'Neutral', value: 300, fill: '#94a3b8' },
            { name: 'Critical', value: 50, fill: '#f87171' },
        ],
        citation_velocity: [
            { month: 'Jan', citations: 400 },
            { month: 'Feb', citations: 800 },
            { month: 'Mar', citations: 1200 },
            { month: 'Apr', citations: 2000 },
            { month: 'May', citations: 3500 },
            { month: 'Jun', citations: 5000 },
        ],
        reproduction: {
            status: "Success",
            logs: [
                "[INFO] Environment inferred: PyTorch 2.1, CUDA 12.1",
                "[INFO] Installing dependencies...",
                "[SUCCESS] Dependencies installed",
                "[INFO] Starting training loop...",
                "[INFO] Epoch 1: Loss 2.45",
                "[SUCCESS] Reproduction verification passed (BLEU > 28.0)"
            ],
            dockerfile: "FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime\nRUN pip install transformers datasets\nCOPY . /app\nWORKDIR /app\nCMD [\"python\", \"train.py\"]"
        }
    }
}

// TODO: add unit tests for fetchScholarDetails — cover successful network+trends,
//  partial responses, and both-null fallback path.
export async function fetchScholarDetails(id: string): Promise<ScholarDetails> {
    const scholarId = decodeURIComponent(id)
    const roster = await fetchScholars()
    const listed = roster.find((item) => item.id === scholarId || item.semantic_scholar_id === scholarId)
    const scholarName = listed?.name || slugToName(scholarId)

    type ScholarNetworkResponse = {
        scholar?: { name?: string; affiliations?: string[]; citation_count?: number; paper_count?: number; h_index?: number }
        stats?: { papers_used?: number }
        nodes?: Array<{ name?: string; type?: string; collab_papers?: number }>
    }

    type ScholarTrendsResponse = {
        scholar?: { name?: string; affiliations?: string[]; citation_count?: number; paper_count?: number; h_index?: number }
        trend_summary?: { publication_trend?: "up" | "down" | "flat"; citation_trend?: "up" | "down" | "flat"; window?: number }
        topic_distribution?: Array<{ topic?: string; count?: number }>
        venue_distribution?: Array<{ venue?: string; count?: number }>
        publication_velocity?: Array<{ year?: number; papers?: number; citations?: number }>
        recent_papers?: Array<{ title?: string; year?: number; citation_count?: number; venue?: string; url?: string }>
    }

    const payloadBase = listed?.semantic_scholar_id || /^\d+$/.test(scholarId)
        ? { scholar_id: listed?.semantic_scholar_id || scholarId }
        : { scholar_name: scholarName }

    const [network, trends] = await Promise.all([
        postJson<ScholarNetworkResponse>("/research/scholar/network", {
            ...payloadBase,
            max_papers: 120,
            recent_years: 5,
            max_nodes: 30,
        }),
        postJson<ScholarTrendsResponse>("/research/scholar/trends", {
            ...payloadBase,
            max_papers: 200,
            year_window: 10,
        }),
    ])

    // Fallback to minimal profile when upstream scholar APIs are unavailable.
    if (!network && !trends) {
        return {
            id: scholarId,
            semantic_scholar_id: listed?.semantic_scholar_id || scholarId,
            name: scholarName,
            affiliation: listed?.affiliation || "Unknown affiliation",
            h_index: listed?.h_index || 0,
            papers_tracked: listed?.papers_tracked || 0,
            recent_activity: listed?.recent_activity || "No tracking runs yet",
            status: listed?.status || "idle",
            keywords: listed?.keywords || [],
            cached_papers: listed?.cached_papers || 0,
            bio: "Live scholar signals are unavailable right now. Check Semantic Scholar API configuration and retry.",
            location: "N/A",
            website: "",
            expertise_radar: [
                { subject: "Coverage", A: 0, fullMark: 100 },
                { subject: "Recency", A: 0, fullMark: 100 },
                { subject: "Citation", A: 0, fullMark: 100 },
            ],
            publications: [],
            co_authors: [],
            stats: {
                total_citations: 0,
                papers_count: listed?.papers_tracked || 0,
                h_index: listed?.h_index || 0,
            },
            trend_summary: {
                publication_trend: "flat",
                citation_trend: "flat",
                window: 10,
            },
            publication_velocity: [],
            top_topics: [],
            top_venues: [],
        }
    }

    const scholar = network?.scholar || trends?.scholar || {}
    const topicDist = (trends?.topic_distribution || []).slice(0, 6)
    const venueDist = (trends?.venue_distribution || []).slice(0, 6)
    const velocity = (trends?.publication_velocity || []).slice(-10)
    const maxTopicCount = Math.max(1, ...topicDist.map((t) => Number(t.count || 0)))

    const publications: Paper[] = (trends?.recent_papers || []).slice(0, 15).map((paper, idx) => ({
        id: `sch-${scholarId}-paper-${idx}`,
        title: String(paper.title || "Untitled"),
        venue: String(paper.venue || "Unknown venue"),
        authors: String(scholar.name || scholarName),
        citations: Number(paper.citation_count || 0),
        status: "analyzing",
        tags: topicDist.map((t) => String(t.topic || "")).filter(Boolean).slice(0, 3),
        url: paper.url || "",
    }))

    const coauthors = (network?.nodes || [])
        .filter((n) => n.type === "coauthor")
        .slice(0, 12)
        .map((n) => {
            const name = String(n.name || "Unknown")
            const collab = Number(n.collab_papers || 0)
            return {
                name: collab > 0 ? `${name} (${collab})` : name,
                avatar: `https://avatar.vercel.sh/${encodeURIComponent(name)}.png`,
            }
        })

    const publicationTrend = trends?.trend_summary?.publication_trend || "flat"
    const recentActivity = publicationTrend === "up"
        ? "Publication trend up"
        : publicationTrend === "down"
            ? "Publication trend down"
            : "Publication trend stable"

    return {
        id: scholarId,
        semantic_scholar_id: listed?.semantic_scholar_id || (payloadBase as { scholar_id?: string }).scholar_id,
        name: String(scholar.name || scholarName),
        affiliation: String((scholar.affiliations || [listed?.affiliation || "Unknown affiliation"])[0] || "Unknown affiliation"),
        h_index: Number(scholar.h_index || listed?.h_index || 0),
        papers_tracked: Number(scholar.paper_count || listed?.papers_tracked || 0),
        recent_activity: recentActivity,
        status: publicationTrend === "up" ? "active" : "idle",
        keywords: listed?.keywords || topicDist.map((t) => String(t.topic || "")).filter(Boolean).slice(0, 6),
        cached_papers: listed?.cached_papers || 0,
        bio: `Trend snapshot: ${trends?.trend_summary?.citation_trend || "flat"} citation trend over the recent analysis window.`,
        location: "N/A",
        website: "",
        expertise_radar: topicDist.map((t) => ({
            subject: String(t.topic || "Topic"),
            A: Math.round((Number(t.count || 0) / maxTopicCount) * 100),
            fullMark: 100,
        })),
        publications,
        co_authors: coauthors,
        stats: {
            total_citations: Number(scholar.citation_count || 0),
            papers_count: Number(scholar.paper_count || 0),
            h_index: Number(scholar.h_index || 0),
        },
        trend_summary: {
            publication_trend: trends?.trend_summary?.publication_trend || "flat",
            citation_trend: trends?.trend_summary?.citation_trend || "flat",
            window: Number(trends?.trend_summary?.window || 10),
        },
        publication_velocity: velocity.map((row) => ({
            year: Number(row.year || 0),
            papers: Number(row.papers || 0),
            citations: Number(row.citations || 0),
        })),
        top_topics: topicDist.map((row) => ({
            topic: String(row.topic || "Unknown"),
            count: Number(row.count || 0),
        })),
        top_venues: venueDist.map((row) => ({
            venue: String(row.venue || "Unknown"),
            count: Number(row.count || 0),
        })),
    }
}

export async function fetchWikiConcepts(): Promise<WikiConcept[]> {
    return [
        {
            id: "transformer",
            name: "Transformer",
            description: "A deep learning model architecture relying on self-attention mechanisms.",
            definition: "The Transformer architecture processes input sequences in parallel using self-attention, allowing it to capture long-range dependencies more effectively than RNNs. It consists of encoder and decoder stacks, each containing multi-head attention and feed-forward layers.",
            related_papers: ["Attention Is All You Need", "BERT", "GPT-3"],
            related_concepts: ["Self-Attention", "Positional Encoding", "Multi-Head Attention"],
            examples: ["GPT-4", "Claude", "LLaMA"],
            category: "Architecture",
            icon: "layers"
        },
        {
            id: "rlhf",
            name: "RLHF",
            description: "Reinforcement Learning from Human Feedback, used to align LLMs with human preferences.",
            definition: "RLHF trains a reward model on human preference data, then fine-tunes the language model using PPO to maximize the reward. This alignment technique helps reduce harmful outputs and improve helpfulness.",
            related_papers: ["InstructGPT", "Constitutional AI"],
            related_concepts: ["PPO", "Reward Model", "Alignment"],
            examples: ["ChatGPT alignment", "Claude training"],
            category: "Method",
            icon: "target"
        },
        {
            id: "bleu",
            name: "BLEU Score",
            description: "A metric for evaluating the quality of machine translated text.",
            definition: "BLEU (Bilingual Evaluation Understudy) compares n-gram overlaps between generated and reference translations. Scores range from 0 to 1, with higher scores indicating better translation quality.",
            related_papers: ["BLEU: a Method for Automatic Evaluation"],
            related_concepts: ["ROUGE", "METEOR", "BERTScore"],
            examples: ["MT evaluation", "Summarization scoring"],
            category: "Metric",
            icon: "bar-chart"
        },
        {
            id: "diffusion",
            name: "Diffusion Models",
            description: "Generative models that learn to reverse a gradual noising process.",
            definition: "Diffusion models add Gaussian noise to data over multiple steps, then learn to reverse this process. They achieve state-of-the-art image generation by iteratively denoising random noise into coherent samples.",
            related_papers: ["DDPM", "Stable Diffusion", "DALL-E 2"],
            related_concepts: ["Denoising", "Score Matching", "Latent Diffusion"],
            examples: ["Midjourney", "Stable Diffusion XL"],
            category: "Method",
            icon: "waves"
        },
        {
            id: "imagenet",
            name: "ImageNet",
            description: "Large-scale visual database for object recognition research.",
            definition: "ImageNet contains over 14 million images annotated with 20,000+ categories. The ILSVRC subset (1000 classes) became the standard benchmark for image classification, driving major advances in CNNs.",
            related_papers: ["ImageNet Classification with Deep CNNs"],
            related_concepts: ["Transfer Learning", "Fine-tuning", "Pretraining"],
            examples: ["ResNet-50 on ImageNet", "ViT benchmarks"],
            category: "Dataset",
            icon: "image"
        }
    ]
}

export async function fetchPapers(): Promise<Paper[]> {
    try {
        const res = await fetch(`${API_BASE_URL}/papers/library`)
        if (!res.ok) {
            return []
        }
        const data = await res.json()
        // Transform backend response to frontend Paper type
        return (data.papers || []).map((item: { paper: Record<string, unknown>; action: string }) => ({
            id: String(item.paper.id),
            title: item.paper.title || "Untitled",
            venue: item.paper.venue || "Unknown",
            authors: Array.isArray(item.paper.authors) ? item.paper.authors.join(", ") : "Unknown",
            citations: item.paper.citation_count ? `${item.paper.citation_count}` : "0",
            status: item.action === "save" ? "Saved" : "pending",
            tags: Array.isArray(item.paper.fields_of_study) ? item.paper.fields_of_study.slice(0, 3) : []
        }))
    } catch {
        return []
    }
}
