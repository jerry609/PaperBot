import { test, expect } from "@playwright/test";

test.describe("Papers Page E2E", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/papers");
    await page.waitForLoadState("networkidle");
  });

  test("papers page has saved papers section", async ({ page }) => {
    const content = await page.textContent("body");
    const hasPapersContent =
      content?.includes("Paper") ||
      content?.includes("Saved") ||
      content?.includes("Collection") ||
      content?.includes("Library");
    expect(hasPapersContent).toBeTruthy();
  });

  test("papers page has library import actions", async ({ page }) => {
    await expect(page.getByRole("button", { name: /import bibtex/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /zotero sync/i })).toBeVisible();
  });
});
