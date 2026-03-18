"use client"

import { useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { cn } from "@/lib/utils"
import { ChevronDown, Lock, LockOpen } from "lucide-react"

export type StudioPermissionProfile = "default" | "full_access"

interface StudioPermissionSelectorProps {
    permissionProfile: StudioPermissionProfile
    onPermissionChange: (profile: StudioPermissionProfile) => void
    supportedProfiles?: string[]
}

function isSupportedProfile(
    value: string,
): value is StudioPermissionProfile {
    return value === "default" || value === "full_access"
}

export function StudioPermissionSelector({
    permissionProfile,
    onPermissionChange,
    supportedProfiles,
}: StudioPermissionSelectorProps) {
    const [confirmOpen, setConfirmOpen] = useState(false)

    const availableProfiles = useMemo(() => {
        const filtered = (supportedProfiles ?? ["default", "full_access"]).filter(isSupportedProfile)
        return filtered.length > 0 ? filtered : ["default", "full_access"]
    }, [supportedProfiles])

    const isFullAccess = permissionProfile === "full_access"

    const applyProfile = (profile: StudioPermissionProfile) => {
        if (!availableProfiles.includes(profile)) return
        onPermissionChange(profile)
    }

    const handleSelect = (profile: StudioPermissionProfile) => {
        if (profile === "full_access" && permissionProfile !== "full_access") {
            setConfirmOpen(true)
            return
        }
        applyProfile(profile)
    }

    return (
        <>
            <DropdownMenu>
                <DropdownMenuTrigger asChild>
                    <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className={cn(
                            "h-6.5 rounded-full border bg-[#f7f8f4] px-2 text-[9px] shadow-none",
                            isFullAccess
                                ? "border-rose-200 bg-rose-50 text-rose-700 hover:bg-rose-100"
                                : "border-slate-200 text-slate-600 hover:bg-white",
                        )}
                    >
                        {isFullAccess ? <LockOpen className="h-2.5 w-2.5" /> : <Lock className="h-2.5 w-2.5" />}
                        <span>{isFullAccess ? "Full" : "Default"}</span>
                        <ChevronDown className="h-2.5 w-2.5 opacity-60" />
                    </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="min-w-[180px] rounded-2xl border-slate-200 p-1.5">
                    {availableProfiles.includes("default") ? (
                        <DropdownMenuItem className="rounded-xl px-2 py-2" onClick={() => handleSelect("default")}>
                            <Lock className="h-3.5 w-3.5" />
                            <div className="flex flex-col">
                                <span>Default</span>
                                <span className="text-[11px] text-slate-500">Prompt before risky edits</span>
                            </div>
                        </DropdownMenuItem>
                    ) : null}
                    {availableProfiles.includes("full_access") ? (
                        <DropdownMenuItem className="rounded-xl px-2 py-2" onClick={() => handleSelect("full_access")}>
                            <LockOpen className="h-3.5 w-3.5 text-rose-600" />
                            <div className="flex flex-col">
                                <span>Full access</span>
                                <span className="text-[11px] text-slate-500">Bypass Claude permission checks</span>
                            </div>
                        </DropdownMenuItem>
                    ) : null}
                </DropdownMenuContent>
            </DropdownMenu>

            <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>Enable full access?</AlertDialogTitle>
                        <AlertDialogDescription>
                            Claude Code will run in bypass-permissions mode for code turns. Use this only in a trusted workspace.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                            className="bg-destructive text-white hover:bg-destructive/90"
                            onClick={() => {
                                setConfirmOpen(false)
                                applyProfile("full_access")
                            }}
                        >
                            Enable full access
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>
        </>
    )
}
