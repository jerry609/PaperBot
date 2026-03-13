import { test, expect } from "@playwright/test";

test.describe("Research Page E2E", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/research");
    await page.waitForLoadState("networkidle");
  });

  test("research page has search input", async ({ page }) => {
    // Should have a search/query input
    const searchInput = page.locator(
      'input[type="text"], input[type="search"], textarea'
    );
    const count = await searchInput.count();
    expect(count).toBeGreaterThan(0);
  });

  test("research page has track/topic tabs or sections", async ({ page }) => {
    // The research page should show tracks or topic workflow sections
    const content = await page.textContent("body");
    const hasResearchContent =
      content?.includes("Track") ||
      content?.includes("Research") ||
      content?.includes("Discovery") ||
      content?.includes("Topic");
    expect(hasResearchContent).toBeTruthy();
  });

  test("post-search view renders without Track Snapshot panel", async ({ page }) => {
    // Navigate with a query to trigger post-search UI state
    await page.goto("/research?query=rag");

    // 1) Wait for a stable post-search marker (auto-retrying)
    await expect(page.getByRole("link", { name: "Open Workflows" })).toBeVisible();

    // 2) Assert the old panel label is absent (auto-retrying)
    await expect(page.locator("text=Track Snapshot")).toHaveCount(0);

    // 3) Sanity: a summary badge/label remains visible (one is enough)
    await expect(page.locator("text=Results:").first()).toBeVisible();
  });
});
