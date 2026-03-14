"use client"

import { getSmoothStepPath, type EdgeProps } from "@xyflow/react"

export type AnimatedEdgeData = {
  variant?: "default" | "success" | "repair"
}

const STROKE_COLORS = {
  default: "#d4d4d8",
  success: "#22c55e",
  repair: "#ef4444",
}

export function AnimatedEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  markerEnd,
  data,
}: EdgeProps) {
  const variant = (data as AnimatedEdgeData | undefined)?.variant || "default"
  const strokeColor = STROKE_COLORS[variant] || STROKE_COLORS.default

  const [edgePath] = getSmoothStepPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    borderRadius: 12,
  })

  const isRepair = variant === "repair"

  return (
    <>
      <path
        id={id}
        d={edgePath}
        fill="none"
        stroke={strokeColor}
        strokeWidth={isRepair ? 2 : 1.5}
        strokeDasharray={isRepair ? "6 4" : undefined}
        markerEnd={markerEnd as string}
        className={isRepair ? "animate-dash" : ""}
        style={style}
      />
      {/* Animated flow dot */}
      {!isRepair && (
        <circle r="3" fill={strokeColor} className="animate-flow-dot">
          <animateMotion dur="2s" repeatCount="indefinite" path={edgePath} />
        </circle>
      )}
    </>
  )
}

export const edgeTypes = {
  animated: AnimatedEdge,
}

/**
 * Inject global CSS for edge animations.
 * Render <EdgeAnimationStyles /> once inside the AgentBoard component.
 */
export function EdgeAnimationStyles() {
  return (
    <style>{`
      @keyframes dash-flow {
        to {
          stroke-dashoffset: -20;
        }
      }
      .animate-dash {
        animation: dash-flow 0.8s linear infinite;
      }
      @keyframes flow-dot-fade {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
      }
      .animate-flow-dot {
        animation: flow-dot-fade 2s ease-in-out infinite;
      }
    `}</style>
  )
}
