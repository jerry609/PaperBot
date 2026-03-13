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
    await page.waitForLoadState("networkidle");

    // The old panel label should not appear anymore
    const body = page.locator("body");
    await expect(body).not.toContainText("Track Snapshot");

    // Basic sanity: post-search summary badges/cta remain
    // We don't depend on backend success; the UI switches to post-search immediately
    const summaryHints = ["Results:", "Mode:", "Sources:"];
    const text = await body.textContent();
    const anyPresent = summaryHints.some((t) => (text || "").includes(t));
    expect(anyPresent).toBeTruthy();
  });
});
