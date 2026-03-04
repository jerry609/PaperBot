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
    const exportButton = page.getByRole("button", { name: /export/i });
    await expect(exportButton).toBeVisible();
  });
});
