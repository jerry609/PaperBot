/**
 * Banner utility - Display PaperBot ASCII art logo
 */

import { renderLogo } from 'oh-my-logo';

export async function showBanner(): Promise<void> {
  try {
    // Render PaperBot logo with ocean gradient
    const logo = renderLogo('PaperBot', {
      palette: 'ocean',
      filled: true,
      font: 'block',
    });
    console.log(logo);
    console.log();
  } catch {
    // Fallback to simple text banner if oh-my-logo fails
    console.log('\n  ╔═══════════════════════════════════╗');
    console.log('  ║         🔬 PaperBot CLI           ║');
    console.log('  ║   Scholar Tracking & Analysis     ║');
    console.log('  ╚═══════════════════════════════════╝\n');
  }
}

export function showVersion(): void {
  console.log('PaperBot CLI v0.1.0');
}
