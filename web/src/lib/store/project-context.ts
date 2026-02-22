
import { create } from 'zustand'

export interface VirtualFile {
    name: string
    content: string
    language: string
}

interface ProjectState {
    files: Record<string, VirtualFile>
    activeFile: string | null

    // Actions
    addFile: (name: string, content: string, language?: string) => void
    updateFile: (name: string, content: string) => void
    setActiveFile: (name: string) => void
    removeFile: (name: string) => void
}

export const useProjectContext = create<ProjectState>((set) => ({
    files: DEFAULT_FILES,
    activeFile: null,  // No file selected by default - shows "Ready to reproduce"

    addFile: (name, content, language = "python") => set((state) => ({
        files: {
            ...state.files,
            [name]: { name, content, language }
        },
        activeFile: name // Switch to new file
    })),

    updateFile: (name, content) => set((state) => ({
        files: {
            ...state.files,
            [name]: { ...state.files[name], content }
        }
    })),

    setActiveFile: (name) => set({ activeFile: name }),

    removeFile: (name) => set((state) => {
        const newFiles = { ...state.files }
        delete newFiles[name]
        return {
            files: newFiles,
            activeFile: state.activeFile === name ? Object.keys(newFiles)[0] || null : state.activeFile
        }
    })
}))
