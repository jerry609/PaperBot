import { test, expect } from "@playwright/test";

test.describe("PaperBot Smoke Tests", () => {
  test("homepage loads and shows sidebar navigation", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveTitle(/PaperBot/i);

    // Sidebar should have main navigation links
    const sidebar = page.locator("nav, aside").first();
    await expect(sidebar).toBeVisible();
  });

  test("dashboard page renders", async ({ page }) => {
    await page.goto("/dashboard");
    await page.waitForLoadState("networkidle");
    // Should not show an unhandled error page
    await expect(page.locator("text=Application error")).not.toBeVisible();
  });

  test("papers page renders", async ({ page }) => {
    await page.goto("/papers");
    await page.waitForLoadState("networkidle");
    await expect(page.locator("text=Application error")).not.toBeVisible();
  });

  test("research page renders", async ({ page }) => {
    await page.goto("/research");
    await page.waitForLoadState("networkidle");
    await expect(page.locator("text=Application error")).not.toBeVisible();
  });

  test("scholars page renders", async ({ page }) => {
    await page.goto("/scholars");
    await page.waitForLoadState("networkidle");
    await expect(page.locator("text=Application error")).not.toBeVisible();
  });

  test("settings page renders", async ({ page }) => {
    await page.goto("/settings");
    await page.waitForLoadState("networkidle");
    await expect(page.locator("text=Application error")).not.toBeVisible();
  });

  test("studio page renders", async ({ page }) => {
    await page.goto("/studio");
    await page.waitForLoadState("networkidle");
    await expect(page.locator("text=Application error")).not.toBeVisible();
  });
});
