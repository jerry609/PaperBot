"use client"

import { useMemo } from "react"
import { cn } from "@/lib/utils"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Button } from "@/components/ui/button"
import { X, Check, ChevronDown, ChevronUp } from "lucide-react"

interface DiffLine {
    type: 'add' | 'remove' | 'unchanged';
    content: string;
    lineNumber: {
        old?: number;
        new?: number;
    };
}

interface DiffViewerProps {
    oldValue: string;
    newValue: string;
    filename?: string;
    onApply?: () => void;
    onReject?: () => void;
    onClose?: () => void;
    splitView?: boolean;
}

/**
 * Simple diff algorithm (Longest Common Subsequence based)
 */
function computeDiff(oldLines: string[], newLines: string[]): DiffLine[] {
    const result: DiffLine[] = [];

    // Simple LCS-based diff
    const lcs = computeLCS(oldLines, newLines);

    let oldIdx = 0;
    let newIdx = 0;
    let lcsIdx = 0;

    while (oldIdx < oldLines.length || newIdx < newLines.length) {
        if (lcsIdx < lcs.length && oldIdx < oldLines.length && oldLines[oldIdx] === lcs[lcsIdx]) {
            if (newIdx < newLines.length && newLines[newIdx] === lcs[lcsIdx]) {
                // Unchanged line
                result.push({
                    type: 'unchanged',
                    content: oldLines[oldIdx],
                    lineNumber: { old: oldIdx + 1, new: newIdx + 1 }
                });
                oldIdx++;
                newIdx++;
                lcsIdx++;
            } else {
                // Line was added
                result.push({
                    type: 'add',
                    content: newLines[newIdx],
                    lineNumber: { new: newIdx + 1 }
                });
                newIdx++;
            }
        } else if (oldIdx < oldLines.length) {
            // Line was removed
            result.push({
                type: 'remove',
                content: oldLines[oldIdx],
                lineNumber: { old: oldIdx + 1 }
            });
            oldIdx++;
        } else if (newIdx < newLines.length) {
            // Line was added
            result.push({
                type: 'add',
                content: newLines[newIdx],
                lineNumber: { new: newIdx + 1 }
            });
            newIdx++;
        }
    }

    return result;
}

function computeLCS(a: string[], b: string[]): string[] {
    const m = a.length;
    const n = b.length;
    const dp: number[][] = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));

    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            if (a[i - 1] === b[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }

    // Backtrack to find LCS
    const lcs: string[] = [];
    let i = m, j = n;
    while (i > 0 && j > 0) {
        if (a[i - 1] === b[j - 1]) {
            lcs.unshift(a[i - 1]);
            i--;
            j--;
        } else if (dp[i - 1][j] > dp[i][j - 1]) {
            i--;
        } else {
            j--;
        }
    }

    return lcs;
}

export function DiffViewer({
    oldValue,
    newValue,
    filename,
    onApply,
    onReject,
    onClose,
}: DiffViewerProps) {
    const diff = useMemo(() => {
        const oldLines = oldValue.split('\n');
        const newLines = newValue.split('\n');
        return computeDiff(oldLines, newLines);
    }, [oldValue, newValue]);

    const stats = useMemo(() => {
        let added = 0;
        let removed = 0;
        for (const line of diff) {
            if (line.type === 'add') added++;
            if (line.type === 'remove') removed++;
        }
        return { added, removed };
    }, [diff]);

    return (
        <div className="flex flex-col h-full bg-background border rounded-lg overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between p-3 border-b bg-muted/30">
                <div className="flex items-center gap-3">
                    {filename && (
                        <span className="font-mono text-sm">{filename}</span>
                    )}
                    <span className="text-xs text-muted-foreground">
                        <span className="text-green-600">+{stats.added}</span>
                        {" / "}
                        <span className="text-red-600">-{stats.removed}</span>
                    </span>
                </div>
                <div className="flex items-center gap-2">
                    {onApply && (
                        <Button size="sm" variant="default" onClick={onApply} className="h-7 text-xs">
                            <Check className="h-3 w-3 mr-1" /> Apply
                        </Button>
                    )}
                    {onReject && (
                        <Button size="sm" variant="outline" onClick={onReject} className="h-7 text-xs">
                            <X className="h-3 w-3 mr-1" /> Reject
                        </Button>
                    )}
                    {onClose && (
                        <Button size="icon" variant="ghost" onClick={onClose} className="h-7 w-7">
                            <X className="h-4 w-4" />
                        </Button>
                    )}
                </div>
            </div>

            {/* Diff Content */}
            <ScrollArea className="flex-1">
                <div className="font-mono text-xs">
                    {diff.map((line, index) => (
                        <div
                            key={index}
                            className={cn(
                                "flex",
                                line.type === 'add' && "bg-green-50 dark:bg-green-900/20",
                                line.type === 'remove' && "bg-red-50 dark:bg-red-900/20",
                            )}
                        >
                            {/* Line numbers */}
                            <div className="flex shrink-0 text-muted-foreground select-none border-r border-border/50">
                                <span className="w-10 text-right px-2 py-0.5 bg-muted/30">
                                    {line.lineNumber.old || ''}
                                </span>
                                <span className="w-10 text-right px-2 py-0.5 bg-muted/30">
                                    {line.lineNumber.new || ''}
                                </span>
                            </div>

                            {/* Change indicator */}
                            <span className={cn(
                                "w-5 text-center py-0.5 shrink-0",
                                line.type === 'add' && "text-green-600",
                                line.type === 'remove' && "text-red-600",
                            )}>
                                {line.type === 'add' && '+'}
                                {line.type === 'remove' && '-'}
                            </span>

                            {/* Content */}
                            <span className="py-0.5 px-2 whitespace-pre overflow-x-auto">
                                {line.content}
                            </span>
                        </div>
                    ))}
                </div>
            </ScrollArea>
        </div>
    )
}

// Modal wrapper for DiffViewer
interface DiffModalProps extends DiffViewerProps {
    isOpen: boolean;
}

export function DiffModal({ isOpen, ...props }: DiffModalProps) {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
            <div className="w-[90vw] h-[80vh] max-w-4xl">
                <DiffViewer {...props} />
            </div>
        </div>
    );
}
