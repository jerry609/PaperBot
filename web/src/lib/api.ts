import { Activity, Paper, Scholar, Stats } from "./types"

const API_BASE_URL = "http://localhost:8000/api"

export async function fetchStats(): Promise<Stats> {
    // TODO: Replace with real API call
    // const res = await fetch(`${API_BASE_URL}/stats`)
    // return res.json()
    return {
        tracked_scholars: 128,
        new_papers: 12,
        llm_usage: "45k",
        read_later: 8
    }
}

export async function fetchActivities(): Promise<Activity[]> {
    // TODO: Replace with real API call
    // const res = await fetch(`${API_BASE_URL}/activities`)
    // return res.json()
    return [
        {
            author: "Dawn Song",
            action: "published",
            paper: "LLM Security: A Comprehensive Survey",
            venue: "S&P 2025",
            time: "2h ago",
            type: "paper"
        },
        {
            author: "PaperBot",
            action: "alert",
            paper: "Attention Is All You Need",
            detail: "Citation velocity increased by 50%",
            time: "4h ago",
            type: "alert"
        },
        {
            author: "System",
            action: "repro",
            paper: "FlashAttention V3",
            detail: "Reproduction failed (Exit Code 1)",
            time: "6h ago",
            type: "repro"
        }
    ]
}

export async function fetchScholars(): Promise<Scholar[]> {
    // Mock data for now
    return [
        {
            id: "dawn-song",
            name: "Dawn Song",
            affiliation: "UC Berkeley",
            h_index: 120,
            papers_tracked: 45,
            recent_activity: "Published 2 days ago",
            status: "active"
        },
        {
            id: "kaiming-he",
            name: "Kaiming He",
            affiliation: "MIT",
            h_index: 145,
            papers_tracked: 28,
            recent_activity: "Cited 500+ times this week",
            status: "active"
        },
        {
            id: "yann-lecun",
            name: "Yann LeCun",
            affiliation: "Meta AI / NYU",
            h_index: 180,
            papers_tracked: 15,
            recent_activity: "New interview",
            status: "idle"
        }
    ]
}

export async function fetchPapers(): Promise<Paper[]> {
    return [
        {
            id: "attention-is-all-you-need",
            title: "Attention Is All You Need",
            venue: "NeurIPS 2017",
            authors: "Vaswani et al.",
            citations: "100k+",
            status: "Reproduced",
            tags: ["Transformer", "NLP"]
        },
        {
            id: "bert-pretraining",
            title: "BERT: Pre-training of Deep Bidirectional Transformers",
            venue: "NAACL 2019",
            authors: "Devlin et al.",
            citations: "80k+",
            status: "analyzing",
            tags: ["NLP", "Language Model"]
        },
        {
            id: "resnet-deep-residual",
            title: "Deep Residual Learning for Image Recognition",
            venue: "CVPR 2016",
            authors: "He et al.",
            citations: "150k+",
            status: "pending",
            tags: ["CV", "ResNet"]
        }
    ]
}
