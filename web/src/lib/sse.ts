export type SSEMessage = {
  type?: string
  data?: unknown
  message?: string | null
}

export async function* readSSE(stream: ReadableStream<Uint8Array>): AsyncGenerator<SSEMessage> {
  const reader = stream.getReader()
  const decoder = new TextDecoder("utf-8")
  let buffer = ""

  while (true) {
    const { value, done } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    while (true) {
      const delimiterIndex = buffer.indexOf("\n\n")
      if (delimiterIndex === -1) break

      const rawEvent = buffer.slice(0, delimiterIndex)
      buffer = buffer.slice(delimiterIndex + 2)

      const lines = rawEvent.split("\n")
      for (const line of lines) {
        if (!line.startsWith("data:")) continue
        const payload = line.slice(5).trim()
        if (payload === "[DONE]") return
        try {
          yield JSON.parse(payload) as SSEMessage
        } catch {
          yield { type: "error", message: "Invalid SSE payload" }
        }
      }
    }
  }
}

