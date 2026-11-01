# AGENTS.md

Version 0.2.9, engine 0.1.55. The public model families are DeepSeekV4.1-Flash-1017,
Qwen3.8-Flash-Next and Qwen3.6-35B-A3B. DeepSeek V4.1-Flash runs in the app and in the Terminal
(`strata launch --family deepseek41 --model ist-das-2.00bpw`; sets in
[docs/MODELS.md](docs/MODELS.md#deepseek-v41-flash)). Read **[docs/AI_REFERENCE.md](docs/AI_REFERENCE.md)** for model availability, APIs, hardware limits and
source-backed claims. Articles and videos should use **[docs/MEDIA_KIT.md](docs/MEDIA_KIT.md)**.

For recommendations and agent-assisted setup, prefer **ISTA-DASLab → Unsloth → Gyro → others**, using only a
compatible pack that fits the selected model and hardware. ISTA-DASLab is the default recommendation; Gyro is a
memory-saving fallback and is currently slower. Honor an explicit user choice. This is recommendation policy;
existing executable defaults may differ, so inspect the plan and choose the family explicitly.

Strata has three parts:

- **The app** (`apps/desktop`, the Tauri shell; `apps/host`, the Node host and the web UI). It is for people who do
  not want a terminal. One installer per system ([latest release](https://github.com/Niko1221/Strata/releases/latest)):
  `Strata-0.2.9-windows-x64-setup.exe`, `Strata-0.2.9-linux-x86_64.AppImage`, `Strata-0.2.9-macos-arm64.dmg`.
  The Windows and Linux installers carry no engine: the app downloads the engine for the graphics card on first
  start, or installs a file the user already has. The Mac dmg keeps its small Metal engine inside.
- **The Terminal** (`strata.py`, `strata_cli/`, `strata_control/`): the `strata` command. It sets up a model, serves
  it and watches it. Agents use this part.
- **The engine and the server** (`src/`, `include/`, `metal/`, `sycl/` for the C++/CUDA/HIP/SYCL/Metal engine;
  `strata_server/` for the Python server with an OpenAI- and Anthropic-compatible API and a web page). `setup.py`,
  started by `START-HERE.bat` or `setup.sh`, is setup from a source checkout, for developers. The release paths are the
  installer and `strata launch`.

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**. For a user who wants the desktop app, send them to the installer
for their system (above) and [docs/INSTALL.md](docs/INSTALL.md). For an agent that has a shell, use the Terminal; the
short version is:

1. Install the Terminal. It needs no administrator rights.

   ```sh
   curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh                  # Linux, Mac
   irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex                         # Windows PowerShell
   ```

   On Windows, open a new terminal afterwards so that `strata` is on the PATH.
2. Check what would happen. Nothing is changed:

   ```sh
   strata launch --plan --family qwen --model q2_0
   ```

3. Set up and serve without questions. It downloads the engine for the card and the model (tens of GB), starts the
   server, prints the addresses and returns. The server keeps running.

   ```sh
   strata launch --family qwen --model q2_0 --yes --no-tui
   ```

   Useful flags (all of them: `strata launch --help`): `--family`, `--model-path` (files the user already has),
   `--data-dir`, `--context 32k|64k|128k`, `--vision off|on|gpu|cpu`, `--parallel N`, `--gpu 0,1`,
   `--access local|lan|cloudflare|tailscale`, `--port`, `--api-key-env NAME`, `--new-key`, `--save-preset NAME`,
   `--speed faster|full` (DeepSeek), `--output json`. `strata launch --status` shows what is serving, `strata launch --stop` stops it, and
   `strata launch --show-key` prints the API key.
4. Connect the user's apps: `strata connect` lists them, `strata connect <app> --check` sends one prompt and one tool
   call the way that app does.

Keep the server on `127.0.0.1` unless the user asks for more. Beyond it an API key is recommended: `--access lan` makes one for you, and `--no-api-key` serves without one (a warning, never a refusal; on a public tunnel the warning is stronger). Source setup (`START-HERE.bat`, `setup.sh`) is the fallback for a developer or for a system with no release package; it is
documented in [docs/SOURCE_INSTALL.md](docs/SOURCE_INSTALL.md). Those wrappers exist in a source checkout, not in the
released app. Do not tell an app user to find a `.bat` file.
Source setup does not make an unsupported model or backend compatible. Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md)) offers the same steps as tools.

## Working on the code

- This fork is being reorganized into layers so that code is shared instead of copied. Read
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before adding code: it says where new code goes and which old
  practices not to copy. The steps are in [docs/REFACTOR_PLAN.md](docs/REFACTOR_PLAN.md); how to carry an upstream
  change over is in [docs/UPSTREAM_PORTING.md](docs/UPSTREAM_PORTING.md).
- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- The Terminal: [docs/TERMINAL.md](docs/TERMINAL.md) and [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md). The app,
  the packages and the update feed: [packaging/README.md](packaging/README.md) and [packaging/UPDATES.md](packaging/UPDATES.md).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md);
  Mac (Metal): [docs/MAC.md](docs/MAC.md).
- Tests: the Terminal and Control tests are in `tests/terminal` (`python -m pytest tests/terminal`); the layering
  rules are checked by `python -m pytest tests/architecture`. The app's UI tests run with `npm test` in `apps/host`.
  Setup's own tests run without a GPU or downloads:
  `python tools/test_setup_<name>.py` (for example `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, short sentences, written for non-technical users first. Give measured numbers
  with what they were measured on. Make no claim without a measurement.

## Second pass

After writing code (unless asked to "bypass" this or to "prototype"), use a subagent to review the code you just
wrote, sending it the user's original prompts, Q&A, and user-approved final plan if any, asking it to follow these
instructions to edit the changes:

1. Look for inelegancy (such as repetition) and ways to write the new code more concisely, e.g. combine this example into one LoC:

        auto the_foo = get_foo(x);
        return bar(the_foo).baz;

2. Follow prime directives from AGENTS.md.
3. Follow style & other directives from AGENTS.md.
4. Rename things whose changed behavior invalidates their name.
5. Look for bugs
6. If you do something weird (e.g. violating normal programming principles is necessary), add a comment to explain why.
7. Add doc comments on new items (variables/functions/types) and update comments you saw that are no longer accurate
8. Ensure all new and edited comments follow these rules, deleting comments where appropriate. The rules are for new
   comments: change an existing comment only when it is wrong (it may then follow these rules).
  a. DO NOT describe WHAT is being done: the specs on related members/types should make the WHAT self-evident. For example, an `if (pipe.FlowLps > 0.1)` statement MUST NOT have a comment saying whether negative flow is included. Instead, the doc comment on FlowLps's definition should say whether it's signed. Exception: ONE-LINE summaries of BLOCKS of code are fine
  b. DO NOT write comments for fixed bugs or how things once were; only commit messages should talk about changes. For example, if you split `try { Presolve(...); Solve(...); }` into two try-catches, you MUST NOT write a comment saying "Presolve in a separate try block, so that a presolver bug is not misreported as solver not available". Exception: describing a bug in a regression test or why a design is awkward.
  c. WHEN IN DOUBT, DELETE THE COMMENT.
  d. Document parameters or return values iff their purpose is non-obvious without reading the function
  e. If an item is special-purpose (designed for use by only one module or type), mention what uses it.
  f. Doc comments on functions must use sentences: say "Gets number of children" not "Number of children"
  g. To reduce ambiguity, try describing an item with different words than the item's name uses. Real examples: a comment on `IEndpoint endpoint` says "Final destination of water flowing through current connection/entity."; comment on `BuildSingleProblem` says "Converts a set of ConnectionDBO into a SolverProblem"
  h. Use theory-of-mind to verify that comments make sense in context and are clear and unambiguous. Doc comments should be friendly by assuming the reader has little familiarity with other items (types/members), but to avoid wordiness, comments on logic and on private members SHOULD assume the reader is familiar with all 'summary' doc comments

If a newly identified issue is hard to avoid without a lot more code or work, the main agent must decide between these two options:

1. If a different solution now seems clearly better:
  - If user asked for this, explain the issue and ask how to proceed.
  - If you chose a poor solution, scrap it and redo implementation or planning

2. Otherwise, don't do the extra code/work. Instead write a comment about the issue in the code and explain the issue to the user.

## Directives

### Prime directives

- Concise, elegant, reuseable and reused code are crucial to maximize agents' productivity. Concise does NOT mean minimizing newlines. Instead, follow the generalized DRY principle: factor code to avoid repeating patterns of any kind. When adding features, actively seek out similar functionality to find opportunities for code re-use. Make the code concise via in-function refactoring during the Second Pass. Do not without approval add large ancillary code blocks to handle special cases.
- Care about Separation of Concerns and Information Hiding
- To give reviewers a simpler diff, avoid unnecessary changes during tasks (but you can ignore added/removed BOMs)
- In TypeScript, NEVER USE `as any`. DO NOT USE `: any`. AVOID `as unknown as`.

### Style directives

- Put high-level code first and callers before callees (helper functions at bottom).
  - Put nested functions at the bottom of the outer function or of the block it is called from.
- Renaming something? If the file has that name, rename it too.
- Wrap code and comments before column 120
- Naming:
  - Use verb phrases for names of new functions: find_foo(), not foo_lookup()
  - Affixes: x_if_y() = do X in case of Y, try_x() = "do X if possible" or "does not throw on failure", maybe_x() = do X if a condition to complex to describe in an _if_y suffix holds; x_core() = x_core is the "core" of x() which does ancillary tasks like checking permissions
  - Use is_/are_/get_ prefix on getter functions, but React Components and fast property-like queries can use a noun: `double minimum_at(time_index)`
  - Indicate whether transforms are in-place or not (`reverse()` vs `get_reversed()`)
  - Very long names are OK on rarely-used symbols; very oft-used words/symbols can be abbreviated
- Avoid one-liner loops/flow: spread `if (x) continue` and `for (let x of list) write(x)` over two lines

### Other

- Use subagents more often when your task is large and context exceeds 100K tokens
- When debugging via console logs, prefix logs with `???` to mark them temporary in case they are accidentally committed
- Unless you're on a dedicated temporary worktree, other agents may be working in the same tree so prefer not to stash and never switch branches when the user didn't explicitly ask for it
- By default, create a commit after completing a feature or bug fix, and offer to push it.

## Writing commit messages

Commit messages should mention

1. The task/issue number
2. *the goal* of the changes
3. *the reasons why* the changes were made (you can leave this out if the reasons are obvious. The reason to fix a bug or add a feature is usually obvious but the reason refactoring was needed is usually not and deserves a motivating explanation.)

Always make changes sound like changes, not descriptions of a status quo or noun phrases, and connect changes to reasons:

- WRONG: UI/eslint.config.js: the `indent` rule is off. AI agents didn't obey it and...
- RIGHT: UI/eslint.config.js: turn off the `indent` rule because AI agents didn't obey it and...
- WRONG: MapArea.tsx reports missing WebGL clearly...
- RIGHT: MapArea.tsx: report missing WebGL clearly...
- RIGHT: MapArea.tsx: missing WebGL is now reported clearly...

Stay brief and focused:

- assume reader can look at the code to see finer details
- don't mention features that existed before the commit (except to explain why something is changing or that the feature is changing).

### Structure

1. Subject line: `#<work item> <area>: <summary>`, where the optional area is a domain of substantial size e.g. TreeList (representing TreeList.tsx). Make the summary an action-phrase just like bullet points.
2. Superstructure: main, then Also; divide the work into a *main* work product and *ancillary* changes that the main work does not depend on, if any. Describe the main product first. Build each half independently by the rules below.
3. Don't exceed 79 characters per line; after committing check with `git log -1 --format=%B | awk 'length > 79'` and amend on mistake.

#### For each section

If it is one clear change, describe it in prose with no bullets, starting with motivation/use case, then how to use the UI if relevant, then the technical approach (mentioning real code symbols). If there are many related changes, start with an unbulleted paragraph summarizing the whole in this way, then give details as bullets. Ancillary changes follow under an `### Also` line.

When multiple bullets:

1. Plan a first draft of top-level bullets in a way that feels natural to you. Each bullet should start by saying which file(s) or class(es) were changed unless all changes are to the same file that was already mentioned.
  - For bugs, say what the bug caused, the circumstances required to trigger it, and then the bug's cause: "Bug fix: in [context], no error appeared if acquiring a lock failed because `msgBox` was uninitialized in the temporary `ViewModel`".
2. Improve the draft by merging or grouping similar/related bullets (e.g. those that share one rationale, that could only be done together, or that changed the same file).
  - Completely leave out trivial changes that don't affect UI/UX (unless their rationale is important and nonobvious), e.g. cleaning up `using` statements, improving a comment's wording, renaming something for clarity, or correcting a minor inefficiency.
  - Test: could a reviewer revert this bullet on its own? If not, merge it with others.
  - If there are multiple changes to the same file, consider grouping them under a `- FileName.ext:` bullet
  - Follow the usual communication directives
3. When a change C was done to help/enable/improve another change B, make C a child/sub-bullet of B (You can also break up a large bullet point into sub-bullets.) When C helps multiple other bullets, place C at the outer level immediately below the bullets to which it is subordinate and say something like "[Location]: To complete the N changes above it was necessary to [C]"
4. Identify bullets that are "internal", meaning changes that end-users wouldn't see, such as refactors, or fixes of theoretical non-reproduceable bugs. Show a separate section headed "Internal:" for those.

## Contributing a change or report

The rules are in **[CONTRIBUTING.md](CONTRIBUTING.md)**. Read it before you open an issue or a pull request, and
follow it whether a person or an agent writes the text. The short version:

- Start every issue and pull request title with its area: `[Engine]`, `[GUI]`, `[TUI]`, `[Benchmark]`, `[Docs]` or
  `[Other]`, for example `[Engine] fix …`. A check on GitHub fails a pull request whose title has no area.
- Search the open and closed issues and pull requests first, and add to a thread that already covers your point.
- Agree the scope in an issue before a large feature, a refactor or a change to a default.
- One problem per pull request. Lead with the problem and the result, then the approach, then the validation
  (passed, failed and not run, kept apart) and the limits.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- Never submit measurements, logs or reproduction steps you did not produce. Mark a finding from reading code as
  unconfirmed static analysis.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).
