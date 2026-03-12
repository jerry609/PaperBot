import { expect, test } from "@playwright/test"

test.describe("Dashboard Daily Brief", () => {
  test("surfaces the daily brief homepage and links into the full workbench", async ({ page }) => {
    await page.goto("/dashboard")

    await expect(page.getByText("Daily Research Brief", { exact: true })).toBeVisible()
    await expect(page.getByRole("heading", { name: "今日热点推送" })).toBeVisible()
    await expect(page.getByRole("heading", { name: "趋势雷达" })).toBeVisible()
    await expect(page.getByRole("heading", { name: "今天先处理什么" })).toBeVisible()

    await page.getByRole("link", { name: "打开完整工作台" }).first().click()
    await expect(page).toHaveURL(/\/workflows/)
  })
})
