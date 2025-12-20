import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

const activities = [
    {
        author: "Dawn Song",
        action: "published",
        paper: "LLM Security: A Comprehensive Survey",
        venue: "S&P 2025",
        time: "2h ago",
        type: "paper"
    },
    {
        author: "PaperBot",
        action: "alert",
        paper: "Attention Is All You Need",
        detail: "Citation velocity increased by 50%",
        time: "4h ago",
        type: "alert"
    },
    {
        author: "System",
        action: "repro",
        paper: "FlashAttention V3",
        detail: "Reproduction failed (Exit Code 1)",
        time: "6h ago",
        type: "repro"
    }
]

export function ActivityFeed() {
    return (
        <Card className="col-span-4 lg:col-span-3">
            <CardHeader>
                <CardTitle>Activity Feed</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="space-y-8">
                    {activities.map((activity, index) => (
                        <div key={index} className="flex items-start gap-4">
                            <Avatar className="h-9 w-9">
                                <AvatarImage src={`/avatars/${index}.png`} alt="Avatar" />
                                <AvatarFallback>{activity.author[0]}</AvatarFallback>
                            </Avatar>
                            <div className="space-y-1">
                                <p className="text-sm font-medium leading-none">
                                    <span className="font-semibold">{activity.author}</span> {activity.action === "published" ? "published new paper" : activity.action === "alert" ? "detected anomaly in" : "updated reproduction for"}
                                </p>
                                <p className="text-sm text-muted-foreground">
                                    {activity.paper}
                                </p>
                                {activity.detail && (
                                    <p className="text-xs text-red-500 mt-1">{activity.detail}</p>
                                )}
                                <div className="flex items-center pt-2">
                                    <Badge variant={activity.type === "alert" ? "destructive" : "secondary"} className="text-xs">
                                        {activity.venue || activity.type.toUpperCase()}
                                    </Badge>
                                    <span className="ml-2 text-xs text-muted-foreground">{activity.time}</span>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </CardContent>
        </Card>
    )
}
