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
                            "h-8 rounded-full border px-2.5 text-xs",
                            isFullAccess
                                ? "border-rose-200 bg-rose-50 text-rose-700 hover:bg-rose-100"
                                : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50",
                        )}
                    >
                        {isFullAccess ? <LockOpen className="h-3.5 w-3.5" /> : <Lock className="h-3.5 w-3.5" />}
                        <span>{isFullAccess ? "Full access" : "Default"}</span>
                        <ChevronDown className="h-3 w-3 opacity-60" />
                    </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="min-w-[180px]">
                    {availableProfiles.includes("default") ? (
                        <DropdownMenuItem onClick={() => handleSelect("default")}>
                            <Lock className="h-3.5 w-3.5" />
                            <div className="flex flex-col">
                                <span>Default</span>
                                <span className="text-[11px] text-slate-500">Prompt before risky edits</span>
                            </div>
                        </DropdownMenuItem>
                    ) : null}
                    {availableProfiles.includes("full_access") ? (
                        <DropdownMenuItem onClick={() => handleSelect("full_access")}>
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
