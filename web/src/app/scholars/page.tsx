import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import Link from "next/link"

const scholars = [
    {
        id: "dawn-song",
        name: "Dawn Song",
        affiliation: "UC Berkeley",
        h_index: 120,
        papers_tracked: 45,
        recent_activity: "Published 2 days ago",
        status: "active"
    },
    {
        id: "kaiming-he",
        name: "Kaiming He",
        affiliation: "MIT",
        h_index: 145,
        papers_tracked: 28,
        recent_activity: "Cited 500+ times this week",
        status: "active"
    },
    {
        id: "yann-lecun",
        name: "Yann LeCun",
        affiliation: "Meta AI / NYU",
        h_index: 180,
        papers_tracked: 15,
        recent_activity: "New interview",
        status: "idle"
    }
]

export default function ScholarsPage() {
    return (
        <div className="flex-1 space-y-4 p-8 pt-6">
            <div className="flex items-center justify-between space-y-2">
                <h2 className="text-3xl font-bold tracking-tight">Tracked Scholars</h2>
                <Button>Add Scholar</Button>
            </div>

            <div className="rounded-md border">
                <Table>
                    <TableHeader>
                        <TableRow>
                            <TableHead className="w-[80px]">Avatar</TableHead>
                            <TableHead>Name</TableHead>
                            <TableHead>Affiliation</TableHead>
                            <TableHead>H-Index</TableHead>
                            <TableHead>Activity</TableHead>
                            <TableHead className="text-right">Actions</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        {scholars.map((scholar) => (
                            <TableRow key={scholar.id}>
                                <TableCell>
                                    <Avatar>
                                        <AvatarImage src={`/scholars/${scholar.id}.png`} />
                                        <AvatarFallback>{scholar.name[0]}</AvatarFallback>
                                    </Avatar>
                                </TableCell>
                                <TableCell className="font-medium">{scholar.name}</TableCell>
                                <TableCell>{scholar.affiliation}</TableCell>
                                <TableCell>{scholar.h_index}</TableCell>
                                <TableCell>
                                    <Badge variant={scholar.status === "active" ? "default" : "secondary"}>
                                        {scholar.recent_activity}
                                    </Badge>
                                </TableCell>
                                <TableCell className="text-right">
                                    <Button variant="ghost" size="sm" asChild>
                                        <Link href={`/scholars/${scholar.id}`}>View Profile</Link>
                                    </Button>
                                </TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </div>
        </div>
    )
}
