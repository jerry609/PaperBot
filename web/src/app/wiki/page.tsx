export default function WikiPage() {
    return (
        <div className="flex-1 space-y-4 p-8 pt-6">
            <h2 className="text-3xl font-bold tracking-tight">Wiki & Knowledge Base</h2>
            <p className="text-muted-foreground">
                Access your personal research knowledge graph and notes here.
            </p>

            <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-6 border rounded-lg bg-card text-card-foreground shadow-sm">
                    <h3 className="font-semibold mb-2">Concepts</h3>
                    <p className="text-sm text-muted-foreground">Browse 128 extracted research concepts.</p>
                </div>
                <div className="p-6 border rounded-lg bg-card text-card-foreground shadow-sm">
                    <h3 className="font-semibold mb-2">methods</h3>
                    <p className="text-sm text-muted-foreground">Compare 45 algorithmic methods.</p>
                </div>
                <div className="p-6 border rounded-lg bg-card text-card-foreground shadow-sm">
                    <h3 className="font-semibold mb-2">Saved Notes</h3>
                    <p className="text-sm text-muted-foreground">View your annotations and summaries.</p>
                </div>
            </div>
        </div>
    )
}
