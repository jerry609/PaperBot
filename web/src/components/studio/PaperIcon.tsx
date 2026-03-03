"use client"

/**
 * Deterministic geometric icon for papers.
 * Hashes the paper ID (djb2) to derive a color and 4×4 grid pattern,
 * producing a blocky identicon-like SVG. Also extracts a 2-letter
 * abbreviation from the title.
 */

const PALETTE = [
    "#e63946", "#457b9d", "#2a9d8f", "#e9c46a",
    "#f4a261", "#264653", "#6a4c93", "#1982c4",
    "#8ac926", "#ff595e", "#6d6875", "#3a86a7",
]

function djb2(str: string): number {
    let hash = 5381
    for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5) + hash + str.charCodeAt(i)) >>> 0
    }
    return hash
}

const SKIP_WORDS = new Set([
    "a", "an", "the", "of", "in", "on", "at", "to", "for",
    "and", "or", "is", "by", "with", "from", "as", "via",
])

function getAbbreviation(title: string): string {
    const words = title
        .replace(/[^a-zA-Z0-9\s]/g, "")
        .split(/\s+/)
        .filter(w => w.length > 0 && !SKIP_WORDS.has(w.toLowerCase()))
    if (words.length === 0) return "??"
    if (words.length === 1) return words[0].slice(0, 2).toUpperCase()
    return (words[0][0] + words[1][0]).toUpperCase()
}

interface PaperIconProps {
    paperId: string
    title: string
    size?: number
}

export function PaperIcon({ paperId, title, size = 48 }: PaperIconProps) {
    const hash = djb2(paperId)
    const color = PALETTE[hash % PALETTE.length]
    const abbr = getAbbreviation(title)

    // Generate a 4×4 symmetric pattern (mirror left half → right half)
    // Use 8 bits from the hash for 2×4 cells, then mirror horizontally
    const bits = (hash >>> 8) & 0xffff
    const cellSize = size / 4

    const cells: { x: number; y: number }[] = []
    for (let row = 0; row < 4; row++) {
        for (let col = 0; col < 2; col++) {
            const bitIndex = row * 2 + col
            if ((bits >> bitIndex) & 1) {
                cells.push({ x: col, y: row })
                cells.push({ x: 3 - col, y: row }) // mirror
            }
        }
    }

    return (
        <svg
            width={size}
            height={size}
            viewBox={`0 0 ${size} ${size}`}
            xmlns="http://www.w3.org/2000/svg"
            role="img"
            aria-label={abbr}
        >
            <rect width={size} height={size} rx={size * 0.15} fill={color} opacity={0.15} />
            {cells.map((cell, i) => (
                <rect
                    key={i}
                    x={cell.x * cellSize + cellSize * 0.1}
                    y={cell.y * cellSize + cellSize * 0.1}
                    width={cellSize * 0.8}
                    height={cellSize * 0.8}
                    rx={cellSize * 0.15}
                    fill={color}
                />
            ))}
            <text
                x={size / 2}
                y={size / 2}
                textAnchor="middle"
                dominantBaseline="central"
                fill="white"
                fontSize={size * 0.3}
                fontWeight="bold"
                fontFamily="system-ui, sans-serif"
                style={{ textShadow: "0 1px 2px rgba(0,0,0,0.3)" }}
            >
                {abbr}
            </text>
        </svg>
    )
}

export { getAbbreviation }
