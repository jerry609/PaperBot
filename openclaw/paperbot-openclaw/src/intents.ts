const INTENT_PATTERNS: Array<[string, RegExp]> = [
  ["gen_code", /\b(code|implement|reproduce|replicate|prototype)\b/i],
  ["review", /\b(review|critique|weakness|accept|reject)\b/i],
  ["paper_track", /\b(track|scholar|author monitor|new papers)\b/i],
  ["paper_analyze", /\b(analyze|analysis|summarize|summary|contribution)\b/i],
  ["research", /\b(context|related work|research plan|literature)\b/i],
  ["paper_search", /\b(find|search|papers?|survey|retrieve)\b/i]
];

export function detectPaperIntent(text: string): string | null {
  const normalized = String(text ?? "").trim();
  if (!normalized) {
    return null;
  }
  for (const [toolName, pattern] of INTENT_PATTERNS) {
    if (pattern.test(normalized)) {
      return toolName;
    }
  }
  return null;
}

export function latestUserMessage(messages: Array<{ role?: string; content?: string }> | undefined): string {
  if (!Array.isArray(messages)) {
    return "";
  }
  const lastUser = [...messages].reverse().find((message) => message.role === "user");
  return String(lastUser?.content ?? "").trim();
}
