---
name: day-delivery
description: "Run a disciplined end-to-end delivery process for one coursework day or similarly scoped feature: clarify ambiguous requirements, prepare and review a specification, coordinate implementation with subagents, iterate through code review and fixes, record and visually review a demo video, obtain human acceptance, then publish the approved code and video. Use this skill whenever the user asks to start, continue, finish, submit, record, or review a numbered course day, especially when the deliverable requires both source code and a demonstration video."
---

# Day Delivery

Deliver one day as a reviewable sequence of evidence, not as a single coding pass.
The completion condition is an approved pair: the tagged code that produced the
demonstration and a video that visibly proves the required behaviour.

## Operating rules

- Read the repository's `AGENTS.md`, the task specification, and the current
  task queue before acting. Treat them as the local contract.
- Keep chat prose in the language required by `AGENTS.md`. Keep technical terms
  and command names verbatim.
- Do not silently replace an unfinished task when a new request arrives. Record
  it at the top of the project queue file and state its position, unless the
  user explicitly authorizes an interruption or there is a real emergency.
- Use the repository's prescribed agent workflow. Give each subagent a bounded,
  independent responsibility; name its inputs, expected output, and whether it
  may edit. Prefer the model requested by the user; otherwise use the local
  agent policy.
- Verify subagent work yourself through diffs and appropriate checks. A
  subagent report is not proof.
- Never create a `commit`, a tag, upload a file, publish a public link, or make
  another external state change without explicit user authorization. Explain
  only approvals that are genuinely required by the host or target system.
- Use `apply_patch` for repository file edits. Preserve unrelated dirty changes.

## Phase 0 — Establish the delivery card

Create or update a short delivery card in `specs/` for the day. It may be a
section in the existing specification when that is the project convention. Keep
only decisions that affect implementation or submission:

```markdown
## Delivery card — WNN DNN

- Source task: `specs/SPEC-wNNdNN.md`
- Goal:
- Acceptance criteria:
- Demo proof points:
- Required commands/checks:
- Submission artifacts: code tag, video filename, public URL
- Open decisions:
- Risks / assumptions:
```

First inspect existing artifacts, tags, current worktree status, prior-day
videos, and the course's submission convention. State which facts were read and
which remain unverified.

## Phase 1 — Interview and specification readiness

Extract answers already present in the task statement and conversation. Ask only
for decisions that materially change the result. When a question tool is
available, use it with concrete options; otherwise ask one concise question.

Typical decisions to resolve:

1. Scope boundaries and backwards-compatibility expectations.
2. Live API/model versus deterministic fake replies for the video; if both are
   needed, define what each proves.
3. Required commands, files, and data persistence semantics.
4. What exact user-visible evidence must appear in the recording.
5. Publication target, filename format, and whether a prior take may be
   removed or replaced.

Review the specification before implementation. It must contain observable
acceptance criteria, error paths, persistence/lifecycle rules, test seams, and
a demo plan. If it is incomplete, patch it or write a companion delivery card
and explicitly label each assumption. Do not invent product choices that need
user ownership.

## Phase 2 — Plan and delegate

Break the work into independently verifiable slices. A normal split is:

| Slice | Owner | Evidence |
|---|---|---|
| Design / compatibility review | review subagent | risks and changed interfaces |
| Core implementation | implementation subagent | focused tests |
| CLI / persistence / UX | implementation subagent | command-level tests |
| Test and code review | QA subagent | findings classified by severity |
| Demo scenario and video QA | demo subagent | script and visual checklist |

Use only the slices required by the actual task. For a small change, one
implementation subagent plus one independent review is sufficient. Avoid
parallel edits to the same lines. Announce every delegation before it starts.

Write the implementation plan into the delivery card: files expected to change,
dependencies, tests, and a rollback-safe sequence. A plan is ready only when a
reviewer can tell how every acceptance criterion will be proven.

## Phase 3 — Implement and verify

1. Ask the implementation subagent(s) to make scoped changes and tests.
2. Inspect the diff against the specification and current worktree; protect
   unrelated user changes.
3. Run focused tests first, then the project's mandated formatter/linter and
   full test suite where proportionate. Report exact commands and outcomes.
4. Run a code review pass that searches for regression risk, input validation,
   error handling, persistence boundaries, backwards compatibility, and stdout/
   stderr contracts when relevant.
5. Classify findings: blocking, required before recording, or follow-up. Apply
   fixes through a separate scoped subagent when they are non-trivial, then
   rerun the checks affected by each fix.

Do not proceed to recording while any blocking defect or unverified acceptance
criterion remains. If an external API is part of the required proof, perform a
small controlled live check and separate it from deterministic tests. Validate
the live rehearsal by its semantic evidence, not only the child process exit
code: confirm expected answers completed, required contrast/length criteria
hold, and no rate-limit or API error text appeared. An interactive CLI may
catch an error and still exit successfully.

## Phase 4 — Build a recording scenario

Study successful recordings and scripts from earlier days before writing a new
scenario. Follow the repository's established naming, runner, capture, and
publication conventions.

The scenario is a viewer-facing proof, not a terminal transcript. It should:

- Start from a clean, readable terminal state with colour and legible contrast.
- Identify the day and the feature being demonstrated.
- Show the setup only when it contributes evidence; do not spend the take on
  invisible implementation details.
- Execute a short sequence in which each step proves one acceptance criterion.
- Include meaningful state changes and their visible result, including a key
  negative/error/lifecycle case when the specification requires it.
- Keep live model calls explicitly labelled. Use deterministic replies for
  repeatable mechanics, and add a controlled live take when the user or spec
  requires proof of the real integration.
- End on the strongest summary screen, not a shell prompt or static wait.

Add a timestamped storyboard to the delivery card or a dedicated script:

| Time | On-screen action | Proof point | Expected visible result |
|---|---|---|---|

Dry-run the scenario before OBS recording. Repair functional or readability
issues before starting a take.

## Phase 5 — Record and visually review

Before capture, confirm the requested recorder is running, its capture target
is correct, and the terminal is readable. The demo command must run in the
exact visible terminal/window owned by the OBS capture source; execution in a
hidden PTY is invalid. Verify capture identity as well as non-black/color by
showing a unique marker or title in that same window before the take. A healthy
screenshot of some other terminal is not proof. Respect the repository's
capture guard and settle time after clearing the screen.

The console must be launched and captured in a color-capable mode. Before every
take, make a capture screenshot and verify that ANSI/Rich colors are visibly
present in the captured frame (not only in the local terminal), while text
contrast remains legible. A monochrome or otherwise colorless capture is a
failed take: fix the launch/capture mode and re-record it. Do not accept a take
based only on the terminal's local appearance.

After each take, inspect the file itself rather than trusting a transcript:

1. Check stream presence, duration, dimensions, and filename.
2. Extract and inspect frames near the opening, each key proof point, and the
   closing screen.
3. Check that text is legible, colour/contrast survives capture, no previous
   take leaks into the opening, and live calls have visibly completed.
4. Compare the take against the storyboard and acceptance criteria.

Reject a file whose frames are stale, unrelated, or black, even if recording,
remuxing, and the demo process all exited successfully. Process success and a
valid media container are prerequisites, not acceptance evidence. An unexpected
rate-limit, API, or network error in the final take blocks publication unless
that exact failure was declared in advance as an acceptance proof point.

Present a compact review table to the user with the local video path, duration,
proof points, and any limitation. Human review is a gate: approval must identify
the concrete reviewed artifact by path, filename, or another unambiguous marker;
general permission to publish does not approve a newly replaced take. Do not
delete, replace, or publish takes until the user approves the chosen artifact.
If the user rejects a take, record the feedback in the delivery card, update the
scenario, and repeat the dry-run/record/review loop.

## Phase 6 — Package and publish

Once the user explicitly authorizes publication:

1. Re-run the final checks and record their result.
2. Create a focused `commit` with no unrelated files and no `Co-Authored-By`.
3. Push the branch, create or update the submission tag only if the project
   convention and authorization permit it, and verify the remote target.
4. Place the approved video under the required final name, upload it to the
   specified destination, and verify the returned public link opens to the
   intended artifact.
5. Update the root progress README and the day documentation when the project
   convention requires them. Re-tag if the submitted snapshot must include that
   documentation.
6. Produce the exact short submission comment: day, immutable code/tag URL,
   and video URL.

Never expose secrets from `.env`, URLs with embedded credentials, or API tokens
in output, logs, commits, or video.

## Completion report

Return this concise evidence-based report:

```markdown
## WNN DNN — status

- Specification: reviewed / changed: …
- Implementation: …
- Verification: `<command>` — result
- Video: `<path>`, `<duration>`; human review: approved / pending
- Publication: pending / `commit`, tag, public video URL
- Remaining: none / exact blocker
```

Use `pending` rather than implying an unauthorized publication is complete.
