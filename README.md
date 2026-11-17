# Policy Placement and Evidence Access in Multi-Agent EHR Workflows

Code for the BioAgents 2026 workshop paper of the same name. It runs Claude agents on
[MedAgentBench](https://github.com/stanfordmlgroup/MedAgentBench) write tasks, with and
without guards and in a four-agent pipeline, and grades them with the benchmark's own
grader. A replay script resends logged review prompts to GPT and Gemini reviewers.

This repository contains code and task generators only. It does not contain MedAgentBench
patient records, its reference solutions, generated task files or run records. The
generators rebuild the task files from your own MedAgentBench server; with the default
seeds they reproduce the files used in the paper byte for byte.

## Setup

1. Clone MedAgentBench next to this repository (or set `MEDAGENTBENCH_DIR`) and start its
   FHIR server as its README describes. The code expects the server at
   `http://localhost:8080/fhir/`.
2. Install [uv](https://docs.astral.sh/uv/), then run `uv sync`.
3. Copy `.env.example` to `.env` and add your API keys. Agents and auditors need
   `ANTHROPIC_API_KEY`; the cross-family replay also needs `OPENAI_API_KEY` and
   `GEMINI_API_KEY`.

Run every command from the repository root. Results go to `mab/runs/`, one JSON record per
task, written as each task finishes.

## Build the task sets

```bash
uv run python mab/build_duplicates.py                          # 60 duplicate-order tasks, with and without the policy
uv run python mab/build_policy_rb.py                           # the same tasks with the chart's orders in the task
uv run python mab/build_variants.py --out mab/variants_v1.json # 60 hazard variants, with and without the policy
uv run python mab/build_labdups.py                             # 60 duplicate HbA1c tasks (second policy, 90 days)
```

## Reproduce the paper

Each script runs three repetitions unless noted. Run a one-task check first, for example
`--ids duplicate_1` or `--per-type 1`, and check its cost before a full run.

| Paper | Run | Summarize |
|---|---|---|
| Tables I and II: pipeline on the duplicate study | `scripts/run_e1.sh` | `mab/summarize_e1.py`, `mab/paths_e1.py`, `mab/risk_e1.py` |
| Table III: second policy, duplicate HbA1c tests | `scripts/run_labs.sh` | `mab/summarize_labs.py` |
| Table IV: Sonnet 5.5 for every agent (one run) | `scripts/run_sonnet.sh` | `mab/summarize_e1.py` and `mab/paths_e1.py` on the `w_*` conditions, `mab/summarize_labs.py` on the `lab_*` conditions; `single_rb` is printed by the run |
| Table V: single agent on the duplicate study | `scripts/run_cheap.sh` (no guard, policy, gate, Haiku auditor); `scripts/run_opus_dups.sh full` (Opus auditor); `mab/run_variants.py --arm none --data mab/duplicates_v1_policy_rb.json` (policy and chart) | `mab/summarize_dups.py` |
| Table VI: pipeline rows, 150 tasks | `scripts/run_e1.sh` (run 1), `scripts/run_e1_150.sh` (runs 2 and 3); `scripts/run_opusver.sh` (Opus verifier row); `scripts/run_factorial.sh` (FINISH review, raw records to the verifier, both) | `mab/boot150.py`; `mab/writes150.py` (Writes column); the Opus-verifier runs print their totals |
| Table VI: single-agent rows and guards (Section IV-G) | `scripts/run_one_arm.sh` per arm (below); Haiku-auditor row in `scripts/run_cheap.sh` | `mab/summarize_arms.py`, `mab/free_checks.py` |
| 150 read-only tasks and the 300-task totals (Section IV-G) | `scripts/run_read.sh` | printed by each run |
| Cross-family reviewer replay (Section IV-G) | `mab/replay_mab.py` | `mab/replay_summary.py` |
| Sonnet check on the hazard variants | `mab/run_variants.py --data mab/variants_v1.json` and `mab/variants_v1_nopolicy.json` | printed by the run |

Single-agent arms in Table VI (no guard, prompt rules, gate, Opus 5.5 auditor):

```bash
for k in 1 2 3; do
  scripts/run_one_arm.sh claude-haiku-5-5 haiku55 none          $k --arm none
  scripts/run_one_arm.sh claude-haiku-5-5 haiku55 prompt        $k --arm prompt
  scripts/run_one_arm.sh claude-haiku-5-5 haiku55 gate_alone    $k --arm gate --no-rules
  scripts/run_one_arm.sh claude-haiku-5-5 haiku55 auditor_alone $k --arm auditor --no-rules
done
```

`MAB_AUDITOR_MODEL` selects the auditor and verifier model (default `claude-opus-5-5`).
`mab/summarize_arms.py` takes the agent's input and output price per million tokens.

## Notes

- `mab/run_mab.py` reproduces the benchmark's prompt, actions and 8-round limit without the
  AgentBench controller, and grades with MedAgentBench's grader loaded from your clone.
- The gate's checks come only from the published function definitions, not from the grader.
- Reviews are parsed from the last JSON verdict in the reply. Sonnet 5.5 sometimes wrote a verdict,
  then a corrected one; earlier runs read the first verdict (see Section IV-D of the paper).
  `mab/run_pipeline.py` saves each verifier's raw reply.
- `mab/run_pipeline.py --review-finish` lets the verifier also review the orchestrator's final
  answer; a blocked answer returns its reason to the orchestrator. `--handoff raw_verifier`
  gives the verifier the reader's raw records while the writer keeps the summary.
- The Writes column regrades each task without its final answer, so it is not the official
  score. Most potassium and many magnesium tasks fail only because the agent answers in a
  sentence where the grader expects the bare value.
- The replay resends the review prompts stored in your own run records; it needs those
  records and sends their contents to the OpenAI and Gemini APIs.
- Model API costs in the paper are measured from the run records' token counts and list
  prices.

## License

MIT. MedAgentBench and its data are under their own license.
