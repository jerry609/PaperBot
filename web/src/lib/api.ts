import { Activity, Paper, PaperDetails, Scholar, ScholarDetails, Stats, WikiConcept, TrendingTopic, PipelineTask, ReadingQueueItem, LLMUsageRecord } from "./types"

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
    return [
        {
            id: "act-1",
            type: "published",
            timestamp: "Dec 3, 2025 · 12:00 PM",
            scholar: {
                name: "Dawn Song",
                avatar: "https://avatar.vercel.sh/dawn.png",
                affiliation: "UC Berkeley"
            },
            paper: {
                title: "Large Language Models for Academic Research: A Comprehensive Review",
                venue: "NeurIPS",
                year: "2024",
                citations: 127,
                tags: ["Security", "LLM"],
                abstract_snippet: "This paper provides a comprehensive review of recent advancements in large language models (LLMs) specifically tailored for academic research applications.",
                is_influential: true
            }
        },
        {
            id: "act-2",
            type: "milestone",
            timestamp: "2h ago",
            milestone: {
                title: "Citation Milestone Reached: 1,000 Citations",
                description: "Your tracked scholar, Andrew Ng, has reached a total of 1,000 citations across all publications.",
                current_value: 1000,
                trend: "up"
            }
        },
        {
            id: "act-3",
            type: "conference",
            timestamp: "5h ago",
            conference: {
                name: "ICML 2025",
                location: "Vancouver, Canada",
                date: "July 2025",
                deadline_countdown: "5 days, 14 hours"
            }
        }
    ]
}

export async function fetchTrendingTopics(): Promise<TrendingTopic[]> {
    return [
        { text: "Large Language Models", value: 100 },
        { text: "Transformer", value: 80 },
        { text: "Reinforcement Learning", value: 60 },
        { text: "Generative AI", value: 90 },
        { text: "Computer Vision", value: 50 },
        { text: "Diffusion Models", value: 70 },
        { text: "Prompt Engineering", value: 40 },
        { text: "Ethics", value: 30 }
    ]
}

export async function fetchPipelineTasks(): Promise<PipelineTask[]> {
    return [
        { id: "1", paper_title: "Attention Is All You Need", status: "testing", progress: 80, started_at: "5m ago" },
        { id: "2", paper_title: "ResNet: Deep Residual Learning", status: "building", progress: 45, started_at: "12m ago" },
        { id: "3", paper_title: "BERT Pretraining", status: "failed", progress: 100, started_at: "1h ago" }
    ]
}

export async function fetchReadingQueue(): Promise<ReadingQueueItem[]> {
    return [
        { id: "1", paper_id: "attention-is-all-you-need", title: "Attention Is All You Need", estimated_time: "15 min", priority: 1 },
        { id: "2", paper_id: "bert-pretraining", title: "BERT Pretraining", estimated_time: "20 min", priority: 2 },
        { id: "3", paper_id: "resnet", title: "ResNet Paper", estimated_time: "10 min", priority: 3 }
    ]
}

export async function fetchLLMUsage(): Promise<LLMUsageRecord[]> {
    return [
        { date: "Mon", gpt4: 12000, claude: 8000, ollama: 3000 },
        { date: "Tue", gpt4: 15000, claude: 9500, ollama: 4000 },
        { date: "Wed", gpt4: 10000, claude: 7000, ollama: 5000 },
        { date: "Thu", gpt4: 18000, claude: 12000, ollama: 2000 },
        { date: "Fri", gpt4: 14000, claude: 10000, ollama: 6000 },
        { date: "Sat", gpt4: 8000, claude: 5000, ollama: 1000 },
        { date: "Sun", gpt4: 6000, claude: 4000, ollama: 500 }
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

export async function fetchScholarDetails(id: string): Promise<ScholarDetails> {
    const papers = await fetchPapers()

    return {
        id,
        name: id.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '),
        affiliation: "University of California, Berkeley",
        h_index: 120,
        papers_tracked: 45,
        recent_activity: "Published 2 days ago",
        status: "active",
        bio: "Dawn Song is a Professor in the Department of Electrical Engineering and Computer Science at UC Berkeley. Her research interest lies in deep learning, security, and blockchain.",
        location: "Berkeley, CA",
        website: "https://dawnsong.io",
        expertise_radar: [
            { subject: 'Security', A: 100, fullMark: 100 },
            { subject: 'Deep Learning', A: 90, fullMark: 100 },
            { subject: 'Blockchain', A: 80, fullMark: 100 },
            { subject: 'Systems', A: 85, fullMark: 100 },
            { subject: 'Privacy', A: 95, fullMark: 100 },
        ],
        publications: papers,
        co_authors: [
            { name: "Dan Hendrycks", avatar: "https://avatar.vercel.sh/dan.png" },
            { name: "Kevin Eykholt", avatar: "https://avatar.vercel.sh/kevin.png" }
        ],
        stats: {
            total_citations: 54321,
            papers_count: 230,
            h_index: 120
        }
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
