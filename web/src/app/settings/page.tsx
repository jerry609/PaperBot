import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export default function SettingsPage() {
    return (
        <div className="flex-1 space-y-4 p-8 pt-6">
            <h2 className="text-3xl font-bold tracking-tight">Settings</h2>

            <div className="grid gap-4">
                <Card>
                    <CardHeader>
                        <CardTitle>LLM Configuration</CardTitle>
                        <CardDescription>Configure your AI model providers and keys.</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="grid w-full max-w-sm items-center gap-1.5">
                            <label htmlFor="openai-key" className="text-sm font-medium">OpenAI API Key</label>
                            <Input type="password" id="openai-key" placeholder="sk-..." />
                        </div>
                        <div className="grid w-full max-w-sm items-center gap-1.5">
                            <label htmlFor="anthropic-key" className="text-sm font-medium">Anthropic API Key</label>
                            <Input type="password" id="anthropic-key" placeholder="sk-ant-..." />
                        </div>
                        <Button>Save Credentials</Button>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader>
                        <CardTitle>Notifications</CardTitle>
                        <CardDescription>Manage your alert preferences.</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="flex items-center space-x-2">
                            <input type="checkbox" id="email-alerts" className="rounded border-gray-300" />
                            <label htmlFor="email-alerts" className="text-sm">Email me about new papers from tracked scholars</label>
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    )
}
