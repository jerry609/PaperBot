import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import { FileText, Download, GitBranch } from "lucide-react"
import { fetchPapers } from "@/lib/api"

export default async function PapersPage() {
    const papers = await fetchPapers()

    return (
        <div className="flex-1 space-y-4 p-8 pt-6">
            <div className="flex items-center justify-between space-y-2">
                <h2 className="text-3xl font-bold tracking-tight">Papers Library</h2>
                <Button>
                    <Download className="mr-2 h-4 w-4" /> Import New Paper
                </Button>
            </div>

            <div className="rounded-md border">
                <Table>
                    <TableHeader>
                        <TableRow>
                            <TableHead className="w-[300px]">Title</TableHead>
                            <TableHead>Venue</TableHead>
                            <TableHead>Authors</TableHead>
                            <TableHead>Citations</TableHead>
                            <TableHead>Status</TableHead>
                            <TableHead className="text-right">Actions</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        {papers.map((paper) => (
                            <TableRow key={paper.id}>
                                <TableCell className="font-medium">
                                    <div className="flex items-center space-x-2">
                                        <FileText className="h-4 w-4 text-blue-500" />
                                        <span>{paper.title}</span>
                                    </div>
                                    <div className="mt-1 flex gap-1">
                                        {paper.tags.map(tag => (
                                            <Badge key={tag} variant="outline" className="text-[10px] px-1 py-0 h-4">{tag}</Badge>
                                        ))}
                                    </div>
                                </TableCell>
                                <TableCell>{paper.venue}</TableCell>
                                <TableCell>{paper.authors}</TableCell>
                                <TableCell>{paper.citations}</TableCell>
                                <TableCell>
                                    <Badge variant={paper.status === "Reproduced" ? "default" : paper.status === "analyzing" ? "secondary" : "outline"}>
                                        {paper.status}
                                    </Badge>
                                </TableCell>
                                <TableCell className="text-right flex items-center justify-end gap-2">
                                    {paper.status === "Reproduced" && (
                                        <Button variant="ghost" size="icon" title="View Code">
                                            <GitBranch className="h-4 w-4" />
                                        </Button>
                                    )}
                                    <Button variant="ghost" size="sm" asChild>
                                        <Link href={`/papers/${paper.id}`}>Analyze</Link>
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
