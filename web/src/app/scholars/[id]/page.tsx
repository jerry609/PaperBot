import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default async function ScholarProfilePage({ params }: { params: Promise<{ id: string }> }) {
    const { id } = await params

    return (
        <div className="flex-1 space-y-4 p-8 pt-6">
            <div className="flex items-center space-x-4">
                <Avatar className="h-20 w-20">
                    <AvatarImage src={`/scholars/${id}.png`} />
                    <AvatarFallback>{id[0].toUpperCase()}</AvatarFallback>
                </Avatar>
                <div>
                    <h2 className="text-3xl font-bold tracking-tight">{id.replace('-', ' ').toUpperCase()}</h2>
                    <p className="text-muted-foreground">Professor, Computer Science</p>
                    <div className="flex gap-2 mt-2">
                        <Badge>Security</Badge>
                        <Badge variant="outline">AI Safety</Badge>
                    </div>
                </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 mt-6">
                <Card>
                    <CardHeader>
                        <CardTitle>H-Index</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="text-3xl font-bold">128</div>
                    </CardContent>
                </Card>
                <Card>
                    <CardHeader>
                        <CardTitle>Impact Factor</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="text-3xl font-bold">Top 1%</div>
                    </CardContent>
                </Card>
                <Card>
                    <CardHeader>
                        <CardTitle>PIS Score</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="text-3xl font-bold text-green-600">92/100</div>
                    </CardContent>
                </Card>
            </div>

            <div className="mt-8">
                <h3 className="text-lg font-medium mb-4">Expertise Radar (Coming Soon)</h3>
                <div className="h-64 w-full bg-muted/20 rounded-md border border-dashed flex items-center justify-center text-muted-foreground">
                    Radar Chart Integration Pending
                </div>
            </div>
        </div>
    )
}
