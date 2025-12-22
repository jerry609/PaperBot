
"use client"

import Editor from "@monaco-editor/react"
import { useTheme } from "next-themes"
import { useProjectContext } from "@/lib/store/project-context"
import { cn } from "@/lib/utils"
import { X } from "lucide-react"

export function DeepCodeEditor() {
    const { theme } = useTheme()
    const { files, activeFile, setActiveFile, updateFile, removeFile } = useProjectContext()

    const activeFileData = activeFile ? files[activeFile] : null

    return (
        <div className="h-full flex flex-col bg-background">
            {/* File Tabs */}
            <div className="flex items-center bg-muted/40 border-b overflow-x-auto no-scrollbar">
                {Object.values(files).map((file) => (
                    <div
                        key={file.name}
                        onClick={() => setActiveFile(file.name)}
                        className={cn(
                            "group flex items-center gap-2 px-3 py-2.5 text-xs font-medium cursor-pointer border-r border-transparent min-w-[100px] select-none hover:bg-muted/60 transition-colors",
                            activeFile === file.name && "bg-background border-r-border text-foreground border-t-2 border-t-primary"
                        )}
                    >
                        <span className="truncate">{file.name}</span>
                        {Object.keys(files).length > 1 && (
                            <button
                                onClick={(e) => {
                                    e.stopPropagation()
                                    removeFile(file.name)
                                }}
                                className="opacity-0 group-hover:opacity-100 p-0.5 hover:bg-muted-foreground/20 rounded-sm"
                            >
                                <X className="h-3 w-3" />
                            </button>
                        )}
                    </div>
                ))}
            </div>

            {/* Editor Area */}
            <div className="flex-1 relative">
                {activeFileData ? (
                    <Editor
                        height="100%"
                        language={activeFileData.language}
                        value={activeFileData.content}
                        theme={theme === "dark" ? "vs-dark" : "light"}
                        onChange={(value) => updateFile(activeFileData.name, value || "")}
                        options={{
                            minimap: { enabled: false },
                            fontSize: 14,
                            lineNumbers: "on",
                            scrollBeyondLastLine: false,
                            automaticLayout: true,
                            padding: { top: 16, bottom: 16 },
                            fontFamily: "'JetBrains Mono', 'Menlo', 'Monaco', 'Courier New', monospace",
                        }}
                    />
                ) : (
                    <div className="h-full flex items-center justify-center text-muted-foreground text-sm">
                        No file open
                    </div>
                )}
            </div>
        </div>
    )
}

