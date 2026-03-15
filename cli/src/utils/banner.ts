/**
 * Banner utility - Display PaperBot ASCII art logo
 */

import { renderFilled } from 'oh-my-logo';

export async function showBanner(): Promise<void> {
  try {
    // Render PaperBot logo with ocean gradient (filled mode)
    const logo = await renderFilled('PaperBot', {
      palette: 'ocean',
    }) as unknown as string | undefined;
    if (logo) {
      console.log(logo);
    }
  } catch {
    // Fallback to simple text banner if oh-my-logo fails
    console.log('\n  ╔═══════════════════════════════════╗');
    console.log('  ║         🔬 PaperBot CLI           ║');
    console.log('  ║   Scholar Tracking & Analysis     ║');
    console.log('  ╚═══════════════════════════════════╝\n');
  }
}
