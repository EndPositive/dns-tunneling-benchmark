import { browser } from "k6/browser";
import { check } from "k6";

const BASE_URL = __ENV.BASE_URL || "https://quickpizza.grafana.com";

export const options = {
  scenarios: {
    ui: {
      executor: "shared-iterations",
      vus: 1,
      iterations: 5,
      options: {
        browser: {
          type: "chromium",
        },
      },
    },
  },
};

export default async function () {
  const page = await browser.newPage();
  try {
    await page.goto(BASE_URL, { timeout: 120000, waitUntil: 'networkidle' });
  } finally {
    await page.close();
  }
}

export function handleSummary(data) {
  return {
    'summary.json': JSON.stringify(data),
  };
}