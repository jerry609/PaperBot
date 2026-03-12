export interface ObsidianCommandOptions {
  vaultPath: string
  rootDir?: string
  userId?: string
  trackId?: number | null
}

function shellQuote(value: string): string {
  return `'${String(value).replace(/'/g, `'\"'\"'`)}'`
}

export function buildObsidianExportCommand({
  vaultPath,
  rootDir = "PaperBot",
  userId = "default",
  trackId = null,
}: ObsidianCommandOptions): string {
  const trimmedVault = vaultPath.trim()
  const trimmedRootDir = rootDir.trim() || "PaperBot"
  const args = ["paperbot", "export", "obsidian", "--user-id", shellQuote(userId)]

  if (trackId != null) {
    args.push("--track-id", String(trackId))
  }

  if (trimmedRootDir && trimmedRootDir !== "PaperBot") {
    args.push("--root-dir", shellQuote(trimmedRootDir))
  }

  if (trimmedVault) {
    args.push("--vault", shellQuote(trimmedVault))
  } else {
    args.push("--vault", shellQuote("/path/to/your/vault"))
  }

  return args.join(" ")
}

export function describeObsidianScope(trackId: number | null, trackName?: string | null): string {
  if (trackId != null) {
    return trackName?.trim() ? `Track: ${trackName.trim()}` : `Track #${trackId}`
  }
  return "Global saved library"
}
