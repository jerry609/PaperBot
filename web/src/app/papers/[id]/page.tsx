export default async function PaperPage({ params }: { params: Promise<{ id: string }> }) {
    const { id } = await params

    return (
        <div className="p-8">
            <h1 className="text-2xl font-bold mb-4">Paper Analysis</h1>
            <p className="text-muted-foreground">Paper ID: {id}</p>

            <div className="mt-8 rounded-md border border-dashed p-8 text-center text-muted-foreground">
                Paper analysis details and reading mode will be implemented here.
                (PDF Viewer + DeepCode Analysis Panel)
            </div>
        </div>
    )
}
