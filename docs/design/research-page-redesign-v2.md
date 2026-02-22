# Research Page Redesign v2

> **Status**: Draft - Pending Review
> **Date**: 2026-02-13
> **Author**: Claude

---

## 1. Executive Summary

This document outlines the redesign of the Research page to reduce feature overlap, streamline the user experience, and consolidate paper management into the Papers Library.

### Key Changes Overview

| Feature | Current Location | New Location | Action |
|---------|------------------|--------------|--------|
| Anchor Authors | Research page (top) | Search box (bottom-left) | Move & simplify |
| Feed tab | Research page | Dashboard (Track Spotlight) | Remove from Research |
| Saved tab | Research page | Papers Library | Merge into Library |
| Memory tab | Research page | TBD (see Section 6) | Relocate |
| Search tab | Always visible | Conditional (after search) | Hide until results |

---

## 2. Detailed Changes

### 2.1 Anchor Authors Relocation

**Current State:**
- Full card displayed at top of Research page
- Shows anchor level, score, evidence papers
- Follow/Ignore action buttons
- Personalized toggle button (top-right of card)

**New Design:**
- Compact dropdown/popover at **bottom-left of SearchBox**
- Only two options: **Personalized** | **Global** (toggle button or segmented control)
- Remove detailed anchor list from main page
- Anchor details accessible via Dashboard's Track Spotlight

**UI Mockup:**
```
┌─────────────────────────────────────────────────────────────┐
│  Search for papers...                                       │
│                                                             │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  [Personalized ▼]                    [Track: ML Research ▼] │
└─────────────────────────────────────────────────────────────┘
```

**Files to Modify:**
- `web/src/components/research/SearchBox.tsx` - Add anchor mode toggle
- `web/src/components/research/ResearchPageNew.tsx` - Remove Anchor Authors card

---

### 2.2 Feed Tab Removal (Overlap with Track Spotlight)

**Analysis:**
- **FeedTab** (Research page): Shows track feed from DailyPaper + personalized ranking
- **TrackSpotlight** (Dashboard): Shows same feed candidates + anchor authors

**Conclusion:** These features are **duplicative**. The TrackSpotlight provides a better overview.

**Changes:**

#### 2.2.1 Remove Feed Tab from Research Page
- Delete `FeedTab` import and usage from `ResearchPageNew.tsx`
- Remove "Feed" from TabsList (reduce to 3 tabs, then conditionally 2)

#### 2.2.2 Update TrackSpotlight in Dashboard
- Change **"Open Track"** button to **"Select Tracks"**
  - Opens track selection modal/dropdown instead of navigating away
- Remove **"View full feed"** link at bottom-left
  - Feed is already visible in TrackSpotlight

**Files to Modify:**
- `web/src/components/research/ResearchPageNew.tsx` - Remove Feed tab
- `web/src/components/dashboard/TrackSpotlight.tsx` - Update button text, remove link

---

### 2.3 Saved Tab Migration to Papers Library

**Current State (SavedTab):**
- Track-scoped saved papers list
- "Related Work" button - generates Related Work section from saved papers
- "Export" dropdown - BibTeX, RIS, Markdown, CSL-JSON
- "Refresh" button

**Current State (Papers Library / SavedPapersList):**
- Global saved papers list (all tracks)
- "Sort" dropdown
- "Per page" dropdown
- "Refresh" button

**New Design (Papers Library):**

```
┌─────────────────────────────────────────────────────────────────────┐
│  Papers Library                                                      │
│  Saved papers from registry feedback and reading states.             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Sort: [Saved Time ▼]  [Related Work]  [Export ▼]  [Refresh]        │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Paper Title 1                                                │    │
│  │ Authors • Venue • Citations                                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│  ... (20 papers per page, no per-page selector)                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Changes:**
1. **Remove "Per page" selector** - Fixed at 20 items per page
2. **Add "Related Work" button** - Opens dialog to generate Related Work section
3. **Add "Export" dropdown** - BibTeX, RIS, Markdown, CSL-JSON formats
4. **Layout:** Sort | Related Work | Export | Refresh (aligned right)

**Files to Modify:**
- `web/src/components/research/SavedPapersList.tsx` - Add Related Work, Export; remove Per page
- `web/src/components/research/ResearchPageNew.tsx` - Remove Saved tab
- `web/src/components/research/SavedTab.tsx` - Can be deleted or kept for reference

---

### 2.4 Search Tab Conditional Display

**Current State:**
- Search tab always visible in TabsList
- Shows empty state message when no search performed

**New Design:**
- **Before search:** Only show greeting + search box + track pills
- **After search:** Show search results inline (no tabs needed for search)
- Search results appear **below the search box** dynamically

**Behavior Pattern (inspired by ai4scholar.net):**
1. User lands on page → Sees greeting, search box, track pills
2. User types query and presses Enter
3. Search results appear below search box (slides in)
4. Greeting fades out, layout compacts
5. No "Search" tab label needed - results are contextual

**UI Flow:**
```
BEFORE SEARCH:
┌─────────────────────────────────────────┐
│        Good morning                      │
│   What papers are you looking for?       │
│                                          │
│   ┌────────────────────────────────┐    │
│   │ Search box                     │    │
│   └────────────────────────────────┘    │
│                                          │
│   [Track 1] [Track 2] [+ New]           │
│                                          │
│   ┌────────────────────────────────┐    │
│   │ Anchor Authors (compact)       │    │
│   └────────────────────────────────┘    │
└─────────────────────────────────────────┘

AFTER SEARCH:
┌─────────────────────────────────────────┐
│   ┌────────────────────────────────┐    │
│   │ Search box (with query)        │    │
│   └────────────────────────────────┘    │
│                                          │
│   ┌────────────────────────────────┐    │
│   │ Paper Result 1                 │    │
│   │ Paper Result 2                 │    │
│   │ ...                            │    │
│   └────────────────────────────────┘    │
│                                          │
│   [Memory]  ← Only remaining tab        │
└─────────────────────────────────────────┘
```

**Files to Modify:**
- `web/src/components/research/ResearchPageNew.tsx` - Restructure layout

---

## 3. Memory Tab Relocation Decision

### Options Analysis

| Option | Pros | Cons |
|--------|------|------|
| **A. Keep in Research page** | Contextual to current track; quick access during research | Only remaining tab (odd UX) |
| **B. Move to Settings** | Clean Research page | Not contextually relevant |
| **C. Move to Dashboard sidebar** | Global visibility | May clutter dashboard |
| **D. Collapsible panel in Research** | Accessible but not prominent | Adds UI complexity |
| **E. Icon button that opens drawer/modal** | Clean main area; on-demand access | Extra click to access |

### Recommendation: **Option E - Icon Button + Drawer**

**Rationale:**
- Memory is track-scoped metadata (notes, tags, preferences)
- Users don't need constant visibility
- A small icon button (e.g., brain/memory icon) near track selector provides access
- Opens a slide-out drawer showing memory items

**UI Placement:**
```
┌─────────────────────────────────────────────────────────────┐
│  Search for papers...                                       │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  [Personalized ▼]        [🧠]  [Track: ML Research ▼] [🔍] │
└─────────────────────────────────────────────────────────────┘
                            ↑
                     Memory icon button
```

**Alternative:** If you prefer Memory to remain more visible, **Option A** (keep as the only tab below search results) is also viable.

---

## 4. Updated Page Structure

### Research Page (After Redesign)

```
┌─────────────────────────────────────────────────────────────────┐
│  [Header/Nav]                                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Good morning                          │    │
│  │           What papers are you looking for?               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                         ↑ Hidden after search                    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  [Search input area]                                     │    │
│  │                                                          │    │
│  │  [Personalized▼]    [🧠 Memory]  [Track▼]  [🔍 Search]  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  [Track Pills: Track1, Track2, Track3, + New]                   │
│                         ↑ Hidden after search                    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Search Results (appears after search)                   │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │ Paper 1                                          │    │    │
│  │  │ Paper 2                                          │    │    │
│  │  │ ...                                              │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Dashboard Page (After Redesign)

```
TrackSpotlight changes:
- "Open Track" → "Select Tracks" (opens track selector)
- Remove "View full feed" link
```

### Papers Library (After Redesign)

```
Header area:
- Sort: [Saved Time ▼]
- [Related Work] [Export ▼] [Refresh]
- NO "Per page" selector (fixed at 20)

Content:
- Paper list with pagination
- 20 items per page
```

---

## 5. Files to Create/Modify/Delete

### Create
- (None - reusing existing components)

### Modify
| File | Changes |
|------|---------|
| `ResearchPageNew.tsx` | Remove Anchor card, Feed tab, Saved tab; restructure Search tab display |
| `SearchBox.tsx` | Add Personalized/Global toggle (bottom-left), Memory icon button |
| `SavedPapersList.tsx` | Add Related Work & Export buttons; remove Per page selector; fix page size to 20 |
| `TrackSpotlight.tsx` | Change "Open Track" to "Select Tracks"; remove "View full feed" link |

### Delete (Optional)
| File | Reason |
|------|--------|
| `FeedTab.tsx` | Functionality moved to Dashboard TrackSpotlight |
| `SavedTab.tsx` | Functionality moved to Papers Library |

---

## 6. Implementation Phases

### Phase 1: Papers Library Enhancement
1. Add Related Work button to `SavedPapersList.tsx`
2. Add Export dropdown to `SavedPapersList.tsx`
3. Remove Per page selector, fix to 20 items

### Phase 2: Dashboard TrackSpotlight Updates
1. Change "Open Track" button text to "Select Tracks"
2. Remove "View full feed" link

### Phase 3: Research Page Restructure
1. Move Anchor Authors toggle to SearchBox (Personalized/Global only)
2. Remove Feed tab from tabs
3. Remove Saved tab from tabs
4. Implement conditional Search results display
5. Add Memory icon button to SearchBox (opens drawer)
6. Remove old Anchor Authors card

### Phase 4: Cleanup
1. Remove unused components (FeedTab, SavedTab) or mark as deprecated

---

## 7. Open Questions

1. **Memory drawer vs. tab**: Should Memory open as a drawer/modal, or remain as a tab below search results?

2. **Track selection in Dashboard**: Should "Select Tracks" in TrackSpotlight open a modal or a dropdown?

3. **Anchor Authors details**: With the Research page showing only Personalized/Global toggle, should users view full anchor details only in Dashboard's TrackSpotlight?

---

## 8. Approval Checklist

- [ ] Anchor Authors relocation approach approved
- [ ] Feed tab removal confirmed (kept in Dashboard only)
- [ ] Saved tab migration to Papers Library confirmed
- [ ] Memory location decision finalized
- [ ] Search conditional display pattern approved
- [ ] Implementation phases order approved

---

*Please review and provide feedback on the above design.*
