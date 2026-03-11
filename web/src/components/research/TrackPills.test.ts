import { describe, expect, it } from "vitest"

import { getVisibleTracks } from "./TrackPills"
import type { Track } from "./TrackSelector"

function makeTrack(id: number, name: string): Track {
  return { id, name }
}

describe("getVisibleTracks", () => {
  it("keeps the active track visible when it falls outside the initial slice", () => {
    const tracks = [
      makeTrack(1, "Track 1"),
      makeTrack(2, "Track 2"),
      makeTrack(3, "Track 3"),
      makeTrack(4, "Track 4"),
      makeTrack(5, "Track 5"),
      makeTrack(6, "Track 6"),
    ]

    const visible = getVisibleTracks(tracks, 6, 5)

    expect(visible.map((track) => track.id)).toEqual([1, 2, 3, 4, 6])
  })

  it("does not reorder tracks when the active one is already visible", () => {
    const tracks = [
      makeTrack(1, "Track 1"),
      makeTrack(2, "Track 2"),
      makeTrack(3, "Track 3"),
    ]

    const visible = getVisibleTracks(tracks, 2, 5)

    expect(visible.map((track) => track.id)).toEqual([1, 2, 3])
  })
})
