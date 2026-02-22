# Research Page Redesign v4 - Further Refinements

> **Status**: Implemented
> **Date**: 2026-02-13
> **Based on**: v3 implementation feedback

---

## Overview

This document addresses two refinements:

1. **Research Page SearchBox** - Adjust to golden ratio proportions
2. **Papers Library** - Replace "Refresh" with "Select Tracks" filter

---

## 1. Research Page SearchBox - Golden Ratio

### Current State
- Width: `max-w-2xl` (672px)
- Height: `min-h-[100px] sm:min-h-[120px]`
- Feels "small" relative to the centered layout

### Golden Ratio Design

The golden ratio (φ ≈ 1.618) creates visually pleasing proportions.

**Approach**: Increase width to `max-w-3xl` (768px) and ensure height creates harmonious proportions.

| Dimension | Before | After |
|-----------|--------|-------|
| Width | 672px (max-w-2xl) | 576px (max-w-xl) |
| Min Height | 100-120px | 120-140px |
| Aspect Ratio | ~5.6:1 | ~4.1:1 (more balanced) |

**Note**: For a search box, we don't want exact 1.618:1 (that would be too square). Instead, we apply golden ratio principles by making it more substantial while maintaining usability.

### Visual Comparison

```
CURRENT (max-w-2xl, 100px height):
┌────────────────────────────────────────────────────┐
│  Search for papers...                               │
│                                                     │
├────────────────────────────────────────────────────┤
│  [Personalized]              [🧠] [Track▼] [🔍]    │
└────────────────────────────────────────────────────┘

GOLDEN RATIO (max-w-3xl, 140px height):
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  Search for papers...                                         │
│                                                               │
│                                                               │
├──────────────────────────────────────────────────────────────┤
│  [Personalized]                      [🧠] [Track▼] [🔍]      │
└──────────────────────────────────────────────────────────────┘
```

### Implementation

**ResearchPageNew.tsx:**
```tsx
// Change from max-w-2xl to max-w-3xl
<div className={cn(
  "w-full px-4 sm:px-6 transition-all duration-500 ease-out",
  hasSearched ? "max-w-5xl mx-auto" : "max-w-3xl"
)}>
```

**SearchBox.tsx:**
```tsx
<Textarea
  className={cn(
    "min-h-[140px] sm:min-h-[160px] max-h-[280px]",  // was 100/120px
    "px-6 sm:px-7 pt-6 sm:pt-7 pb-18",               // more padding
    ...
  )}
/>
```

---

## 2. Papers Library - Track Selector

### Current State
- Header: Sort | Related Work | Export | **Refresh**
- Shows all saved papers globally (no track filtering)
- Refresh reloads the paper list

### New Design
- Header: Sort | **Select Tracks** | Related Work | Export (when selected)
- "Select Tracks" dropdown allows filtering by track
- Default option: "All Tracks" (current behavior)
- Selecting a track filters papers to that track only

### Visual Design

```
┌─────────────────────────────────────────────────────────────────┐
│  Saved Papers                                                    │
│  View saved items, sort by score/time...                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Sort: [Saved Time ▼]  Track: [All Tracks ▼]  [Related Work]   │
│                                    │                            │
│                                    ├──────────────────┐         │
│                                    │ ✓ All Tracks     │         │
│                                    │   ML Research    │         │
│                                    │   Security       │         │
│                                    │   NLP            │         │
│                                    └──────────────────┘         │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  [☐] │ Title          │ Source │ Saved  │ Judge │ Status │ Act │
│  ... │ ...            │ ...    │ ...    │ ...   │ ...    │ ... │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

**SavedPapersList.tsx:**
```tsx
// Add track state
const [tracks, setTracks] = useState<Track[]>([])
const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null) // null = all

// Fetch tracks on mount
useEffect(() => {
  fetch('/api/research/tracks?user_id=default')
    .then(res => res.json())
    .then(data => setTracks(data.tracks || []))
}, [])

// Update loadSavedPapers to filter by track
const qs = new URLSearchParams({
  sort_by: sortBy,
  sort_order: "desc",
  limit: "500",
  user_id: "default",
})
if (selectedTrackId) {
  qs.set("track_id", String(selectedTrackId))
}

// Header - replace Refresh with Track selector
<DropdownMenu>
  <DropdownMenuTrigger asChild>
    <Button variant="outline" size="sm">
      <Filter className="mr-1 h-4 w-4" />
      {selectedTrackId
        ? tracks.find(t => t.id === selectedTrackId)?.name
        : "All Tracks"}
      <ChevronDown className="ml-1 h-4 w-4" />
    </Button>
  </DropdownMenuTrigger>
  <DropdownMenuContent align="end">
    <DropdownMenuItem onClick={() => setSelectedTrackId(null)}>
      {!selectedTrackId && <Check className="mr-2 h-4 w-4" />}
      All Tracks
    </DropdownMenuItem>
    <DropdownMenuSeparator />
    {tracks.map(track => (
      <DropdownMenuItem
        key={track.id}
        onClick={() => setSelectedTrackId(track.id)}
      >
        {selectedTrackId === track.id && <Check className="mr-2 h-4 w-4" />}
        {track.name}
      </DropdownMenuItem>
    ))}
  </DropdownMenuContent>
</DropdownMenu>
```

---

## 3. Summary of Changes

| Area | Change |
|------|--------|
| ResearchPageNew.tsx | Change `max-w-2xl` to `max-w-3xl` before search |
| SearchBox.tsx | Increase min-height to 140-160px, more padding |
| SavedPapersList.tsx | Add track state, fetch tracks, add Track dropdown |
| SavedPapersList.tsx | Remove Refresh button |
| SavedPapersList.tsx | Filter papers by selected track |

---

## 4. Questions for Review

1. **Golden ratio interpretation**: Is `max-w-3xl` (768px) width with 140-160px height satisfactory, or would you prefer different dimensions?

2. **Track selector position**: Should it be placed:
   - After Sort (recommended - logical grouping of filters)
   - Before Sort
   - At the end before Export

3. **Default track**: Should the default be "All Tracks" or the user's currently active track?

---

*Please review and confirm before implementation.*
