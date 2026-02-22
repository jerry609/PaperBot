# Research Page Redesign v3 - Refinements

> **Status**: Implemented
> **Date**: 2026-02-13
> **Based on**: v2 implementation feedback

---

## Overview

This document addresses three refinements to the v2 implementation:

1. **Research Page SearchBox** - Resize and center like Claude's homepage
2. **Dashboard TrackSpotlight** - Make "Select Tracks" a real selector, not a redirect
3. **Papers Library Export** - Show Export only after selecting papers

---

## 1. Research Page SearchBox Redesign

### Current State
- SearchBox is full-width (max-w-5xl)
- Positioned at top of content area
- Greeting above, track pills below
- Toolbar at bottom of search box

### Target State (Claude Homepage Style)
- **Centered vertically** in viewport (before search)
- **Narrower width** (~640px / max-w-2xl)
- **Larger input** with more padding
- **Minimal, clean aesthetic**
- After search: moves to top, results appear below

### Visual Comparison

```
BEFORE (Current):
┌─────────────────────────────────────────────────────────────────┐
│  [Nav]                                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Good morning                                                    │
│  What papers are you looking for?                               │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Search box (full width, max-w-5xl)                         │  │
│  │                                                            │  │
│  │ [Personalized]              [🧠] [Track ▼] [🔍]           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  [Track 1] [Track 2] [+ New]                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

AFTER (Claude Style):
┌─────────────────────────────────────────────────────────────────┐
│  [Nav]                                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                                                                  │
│                                                                  │
│                     Good morning                                 │
│             What papers are you looking for?                     │
│                                                                  │
│           ┌─────────────────────────────────────┐               │
│           │                                      │               │
│           │  Search for papers...                │               │
│           │                                      │               │
│           │                                      │               │
│           ├─────────────────────────────────────┤               │
│           │ [Personalized]    [🧠] [Track▼] [🔍]│               │
│           └─────────────────────────────────────┘               │
│                                                                  │
│              [Track 1] [Track 2] [+ New]                        │
│                                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### CSS/Layout Changes

**ResearchPageNew.tsx:**
```tsx
// Before search: center everything vertically
<div className={cn(
  "min-h-[calc(100vh-4rem)] transition-all duration-500",
  !hasSearched && "flex flex-col items-center justify-center"
)}>
  {/* Content wrapper - narrower */}
  <div className={cn(
    "w-full px-4 transition-all duration-500",
    hasSearched ? "max-w-5xl mx-auto pt-6" : "max-w-2xl"
  )}>
    {/* Greeting */}
    {!hasSearched && (
      <div className="text-center mb-8">
        <h1 className="text-4xl font-semibold mb-2">Good morning</h1>
        <p className="text-xl text-muted-foreground">
          What papers are you looking for?
        </p>
      </div>
    )}

    {/* SearchBox - no width change needed, parent controls it */}
    <SearchBox ... />

    {/* Track Pills - centered */}
    {!hasSearched && tracks.length > 0 && (
      <div className="flex justify-center mt-6">
        <TrackPills ... />
      </div>
    )}

    {/* Results - full width after search */}
    {hasSearched && <SearchResults ... />}
  </div>
</div>
```

**SearchBox.tsx:**
```tsx
// Increase textarea height
<Textarea
  className={cn(
    "min-h-[100px] sm:min-h-[120px]",  // was 70px/80px
    "px-5 pt-5 pb-16",                  // more padding
    "text-lg placeholder:text-muted-foreground/50"
  )}
/>
```

### Files to Modify
- `web/src/components/research/ResearchPageNew.tsx` - Layout centering
- `web/src/components/research/SearchBox.tsx` - Height/padding adjustments

---

## 2. Dashboard TrackSpotlight - Real Track Selector

### Current State
- "Select Tracks" button redirects to `/research?track_id=X`
- TrackSpotlight receives `track` prop from parent (DashboardPage)
- No way to switch tracks without leaving Dashboard

### Target State
- "Select Tracks" opens a **dropdown menu** listing all tracks
- Selecting a track updates TrackSpotlight content in place
- Parent component (DashboardPage) manages track state and fetches

### Component Architecture

```
DashboardPage
├── TrackSpotlight (controlled)
│   ├── props.tracks: Track[]           // all available tracks
│   ├── props.activeTrack: Track        // currently selected
│   ├── props.onSelectTrack(id)         // callback to change track
│   ├── props.feedItems: TrackFeedItem[]
│   ├── props.anchors: AnchorPreviewItem[]
│   └── "Select Tracks" dropdown
└── (other dashboard components)
```

### Visual Design

```
┌─────────────────────────────────────────────────────────────────┐
│  Track Spotlight · ML Research                                   │
│  Track-level feed blending keyword match...      [Select Tracks▼]│
│                                                   ┌─────────────┐│
│  #transformer #attention #llm                     │ ✓ ML Research││
│                                                   │   Security   ││
├──────────────────────────────────────────────────│   NLP        ││
│  Feed Candidates              │  Anchor Authors   └─────────────┘│
│  ┌──────────────────────────┐ │  ┌─────────────┐                 │
│  │ Paper Title 1            │ │  │ Author 1    │                 │
│  │ Paper Title 2            │ │  │ Author 2    │                 │
│  └──────────────────────────┘ │  └─────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

**TrackSpotlight.tsx:**
```tsx
interface TrackSpotlightProps {
  tracks: Track[]                        // NEW: all tracks
  activeTrack: Track | null
  onSelectTrack: (trackId: number) => void  // NEW: callback
  feedItems: TrackFeedItem[]
  feedTotal: number
  anchors: AnchorPreviewItem[]
  isLoading?: boolean                    // NEW: loading state
}

export function TrackSpotlight({
  tracks,
  activeTrack,
  onSelectTrack,
  feedItems,
  feedTotal,
  anchors,
  isLoading,
}: TrackSpotlightProps) {
  // Replace Link button with DropdownMenu
  <DropdownMenu>
    <DropdownMenuTrigger asChild>
      <Button variant="outline" size="sm">
        Select Tracks
        <ChevronDown className="ml-1 h-4 w-4" />
      </Button>
    </DropdownMenuTrigger>
    <DropdownMenuContent align="end">
      {tracks.map((track) => (
        <DropdownMenuItem
          key={track.id}
          onClick={() => onSelectTrack(track.id)}
        >
          {track.id === activeTrack?.id && <Check className="mr-2 h-4 w-4" />}
          {track.name}
        </DropdownMenuItem>
      ))}
    </DropdownMenuContent>
  </DropdownMenu>
}
```

**DashboardPage (parent):**
```tsx
// Add state for all tracks and active track
const [tracks, setTracks] = useState<Track[]>([])
const [activeTrackId, setActiveTrackId] = useState<number | null>(null)
const [feedItems, setFeedItems] = useState<TrackFeedItem[]>([])
const [anchors, setAnchors] = useState<AnchorPreviewItem[]>([])
const [spotlightLoading, setSpotlightLoading] = useState(false)

// Fetch tracks on mount
useEffect(() => {
  fetchTracks().then(data => {
    setTracks(data.tracks)
    const active = data.tracks.find(t => t.is_active)
    if (active) setActiveTrackId(active.id)
  })
}, [])

// Fetch feed/anchors when activeTrackId changes
useEffect(() => {
  if (!activeTrackId) return
  setSpotlightLoading(true)
  Promise.all([
    fetchFeed(activeTrackId),
    fetchAnchors(activeTrackId),
  ]).then(([feed, anchors]) => {
    setFeedItems(feed.items)
    setAnchors(anchors.items)
  }).finally(() => setSpotlightLoading(false))
}, [activeTrackId])

// Handler
const handleSelectTrack = async (trackId: number) => {
  setActiveTrackId(trackId)
  // Optionally activate on backend
  await fetch(`/api/research/tracks/${trackId}/activate`, { method: 'POST' })
}

// Render
<TrackSpotlight
  tracks={tracks}
  activeTrack={tracks.find(t => t.id === activeTrackId) || null}
  onSelectTrack={handleSelectTrack}
  feedItems={feedItems}
  feedTotal={feedTotal}
  anchors={anchors}
  isLoading={spotlightLoading}
/>
```

### Files to Modify
- `web/src/components/dashboard/TrackSpotlight.tsx` - Add dropdown, new props
- `web/src/app/dashboard/page.tsx` (or parent) - Manage track state, fetch data

---

## 3. Papers Library - Export on Selection

### Current State
- Export button always visible in header
- Exports all saved papers
- No paper selection mechanism

### Target State
- Add **checkbox column** to table for paper selection
- **Export button hidden** until papers are selected
- Export only selected papers
- Add "Select All" checkbox in header
- Show selection count badge

### Visual Design

```
┌─────────────────────────────────────────────────────────────────┐
│  Saved Papers                                                    │
│  View saved items, sort by score/time...                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Sort: [Saved Time ▼]  [Related Work]  [Refresh]                │
│                                                                  │
│  ↓ After selecting papers:                                      │
│  Sort: [Saved Time ▼]  [Related Work]  [Export ▼] (3)  [Refresh]│
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  [☐] │ Title          │ Source │ Saved  │ Judge │ Status │ Act │
│  ────┼────────────────┼────────┼────────┼───────┼────────┼─────│
│  [☑] │ Paper Title 1  │ arxiv  │ 2h ago │ 4.2   │ unread │ ... │
│  [☑] │ Paper Title 2  │ s2     │ 1d ago │ 3.8   │ read   │ ... │
│  [☐] │ Paper Title 3  │ arxiv  │ 3d ago │ 4.5   │ unread │ ... │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

**SavedPapersList.tsx:**
```tsx
// Add selection state
const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set())

const toggleSelect = (paperId: number) => {
  setSelectedIds(prev => {
    const next = new Set(prev)
    if (next.has(paperId)) next.delete(paperId)
    else next.add(paperId)
    return next
  })
}

const toggleSelectAll = () => {
  if (selectedIds.size === pagedItems.length) {
    setSelectedIds(new Set())
  } else {
    setSelectedIds(new Set(pagedItems.map(item => item.paper.id)))
  }
}

const hasSelection = selectedIds.size > 0

// Update handleExport to use selectedIds
const handleExport = async (format) => {
  const qs = new URLSearchParams({ format, user_id: "default" })
  // Add selected paper IDs
  selectedIds.forEach(id => qs.append("paper_id", String(id)))
  // ... rest of export logic
}

// Header buttons - Export only when selected
<div className="flex items-center gap-2">
  <label>Sort</label>
  <select ...>...</select>
  <Button onClick={openRelatedWork}>Related Work</Button>

  {hasSelection && (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm">
          <Download className="mr-1 h-4 w-4" />
          Export ({selectedIds.size})
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent>
        <DropdownMenuItem onClick={() => handleExport("bibtex")}>BibTeX</DropdownMenuItem>
        ...
      </DropdownMenuContent>
    </DropdownMenu>
  )}

  <Button onClick={refresh}>Refresh</Button>
</div>

// Table - add checkbox column
<TableHeader>
  <TableRow>
    <TableHead className="w-[50px]">
      <Checkbox
        checked={selectedIds.size === pagedItems.length && pagedItems.length > 0}
        onCheckedChange={toggleSelectAll}
      />
    </TableHead>
    <TableHead>Title</TableHead>
    ...
  </TableRow>
</TableHeader>

<TableBody>
  {pagedItems.map((item) => (
    <TableRow key={item.paper.id}>
      <TableCell>
        <Checkbox
          checked={selectedIds.has(item.paper.id)}
          onCheckedChange={() => toggleSelect(item.paper.id)}
        />
      </TableCell>
      <TableCell>...</TableCell>
      ...
    </TableRow>
  ))}
</TableBody>
```

### Files to Modify
- `web/src/components/research/SavedPapersList.tsx` - Add selection, conditional Export
- `web/src/components/ui/checkbox.tsx` - Ensure Checkbox component exists (shadcn)

---

## 4. Summary of Changes

| Area | Change | Priority |
|------|--------|----------|
| ResearchPageNew.tsx | Center layout vertically (before search) | High |
| ResearchPageNew.tsx | Narrow width (max-w-2xl) before search | High |
| SearchBox.tsx | Increase height/padding | Medium |
| TrackSpotlight.tsx | Replace Link with dropdown selector | High |
| DashboardPage | Add track state management | High |
| SavedPapersList.tsx | Add checkbox selection column | High |
| SavedPapersList.tsx | Show Export only when papers selected | High |

---

## 5. Implementation Order

### Phase 1: Research Page Layout
1. Update ResearchPageNew.tsx layout - center vertically, narrow width
2. Adjust SearchBox.tsx dimensions

### Phase 2: Dashboard Track Selection
1. Update TrackSpotlight.tsx - add dropdown, new props
2. Update DashboardPage - manage track state, data fetching
3. Connect callbacks

### Phase 3: Papers Library Selection
1. Add Checkbox component if missing
2. Add selection state to SavedPapersList.tsx
3. Add checkbox column to table
4. Make Export conditional on selection
5. Update export API call to use selected IDs

---

## 6. Questions for Review

1. **SearchBox dimensions**: Should the narrow width be `max-w-2xl` (672px) or `max-w-xl` (576px)?

2. **Track selection persistence**: When user selects a track in Dashboard, should it also activate that track globally (affecting Research page)?

3. **Export scope**: When papers are selected, should Related Work also use only selected papers, or remain using all saved papers?

4. **Selection persistence**: Should paper selection persist across page changes, or reset when navigating away?

---

*Please review and confirm the design direction before implementation.*
