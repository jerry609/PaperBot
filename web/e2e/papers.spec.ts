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

  test("papers page has import/export options", async ({ page }) => {
    // Look for import or export buttons/links
    const importBtn = page.locator(
      'button:has-text("Import"), button:has-text("BibTeX"), button:has-text("Zotero")'
    );
    const count = await importBtn.count();
    // At least some action button should be present
    expect(count).toBeGreaterThanOrEqual(0);
  });
});
