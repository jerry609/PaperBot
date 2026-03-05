export async function deleteProjectFiles(projectDir: string): Promise<void> {
  const listRes = await fetch(
    `/api/runbook/files?project_dir=${encodeURIComponent(projectDir)}&recursive=true`
  );
  if (!listRes.ok) {
    throw new Error(`Failed to list project files (${listRes.status})`);
  }

  const payload = (await listRes.json()) as { files?: unknown };
  const files = Array.isArray(payload.files)
    ? payload.files.filter((v): v is string => typeof v === "string" && v.length > 0)
    : [];

  for (const path of files) {
    const delRes = await fetch("/api/runbook/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_dir: projectDir, path }),
    });
    if (!delRes.ok) {
      throw new Error(`Failed to delete file '${path}' (${delRes.status})`);
    }
  }
}
