export type PaperPreferenceAction = "like" | "dislike"

export type PaperFeedbackAction = "save" | PaperPreferenceAction

export type PaperFeedbackRequestAction =
  | PaperFeedbackAction
  | "unsave"
  | "unlike"
  | "undislike"

function normalizeFeedbackKey(action: string | null | undefined): string {
  return String(action || "").trim().toLowerCase().replace(/\s+/g, "_")
}

export function normalizePaperFeedbackAction(
  action: string | null | undefined,
): PaperFeedbackAction | null {
  const normalized = normalizeFeedbackKey(action)
  if (normalized === "not_relevant" || normalized === "not-related") {
    return "dislike"
  }
  if (normalized === "save" || normalized === "like" || normalized === "dislike") {
    return normalized
  }
  return null
}

export function normalizePaperPreferenceAction(
  action: string | null | undefined,
): PaperPreferenceAction | null {
  const normalized = normalizePaperFeedbackAction(action)
  if (normalized === "like" || normalized === "dislike") {
    return normalized
  }
  return null
}

export function currentFeedbackFromRequestAction(
  action: PaperFeedbackRequestAction,
): PaperFeedbackAction | null {
  return normalizePaperFeedbackAction(action)
}

export function toggleSaveFeedbackAction(isSaved: boolean): PaperFeedbackRequestAction {
  return isSaved ? "unsave" : "save"
}

export function togglePaperPreferenceAction(
  currentAction: PaperPreferenceAction | null,
  targetAction: PaperPreferenceAction,
): PaperFeedbackRequestAction {
  if (currentAction !== targetAction) {
    return targetAction
  }
  return targetAction === "like" ? "unlike" : "undislike"
}
