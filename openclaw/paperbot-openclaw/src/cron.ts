import type { CronJobDefinition, PaperBotOpenClawConfig } from "./types.js";

export const DEFAULT_PAPERBOT_CRON_JOBS: CronJobDefinition[] = [
  {
    id: "paper-monitor",
    expression: "0 6 * * *",
    description: "Run scholar tracking through PaperBot every morning.",
    task: "paper-monitor",
    defaultInput: { scholarId: "", maxNewPapers: 5 }
  },
  {
    id: "weekly-digest",
    expression: "0 9 * * 1",
    description: "Build a weekly PaperBot digest using the configured research queries.",
    task: "weekly-digest",
    defaultInput: { queries: [] }
  },
  {
    id: "conference-deadlines",
    expression: "0 8 * * *",
    description: "Fetch the latest deadline radar for configured tracks.",
    task: "conference-deadlines",
    defaultInput: {}
  },
  {
    id: "citation-monitor",
    expression: "0 * * * *",
    description: "Poll PaperBot for citation milestones based on stored cron queries.",
    task: "citation-monitor",
    defaultInput: { queries: [] }
  }
];

export function resolveCronJobs(config: PaperBotOpenClawConfig): CronJobDefinition[] {
  return DEFAULT_PAPERBOT_CRON_JOBS.map((job) => {
    if (job.task === "paper-monitor") {
      return {
        ...job,
        defaultInput: {
          scholarId: config.cronScholarId ?? "",
          maxNewPapers: 5
        }
      };
    }
    if (job.task === "weekly-digest" || job.task === "citation-monitor") {
      return {
        ...job,
        defaultInput: {
          queries: [...config.cronQueries]
        }
      };
    }
    return job;
  });
}
