# Claude Code Documentation


## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Core Concepts](#core-concepts)
- [Configuration & CLI](#configuration--cli)
- [Workflows & Best Practices](#workflows--best-practices)
- [Sub-Agents](#sub-agents)
- [MCP & Extensions](#mcp--extensions)
- [Hooks & Automation](#hooks--automation)
- [Integrations](#integrations)
- [Platforms](#platforms)
- [Troubleshooting](#troubleshooting)


---

# Overview


### Claude Code overview


Claude Code is an agentic coding tool that reads your codebase, edits files, runs commands, and integrates with your development tools. Available in your terminal, IDE, desktop app, and browser.


Claude Code is an AI-powered coding assistant that helps you build features, fix bugs, and automate development tasks. It understands your entire codebase and can work across multiple files and tools to get things done.

### Get started

Choose your environment to get started. Most surfaces require a [Claude subscription](https://claude.com/pricing?utm_source=claude_code&utm_medium=docs&utm_content=overview_pricing) or [Anthropic Console](https://console.anthropic.com/) account. The Terminal CLI and VS Code also support [third-party providers](/docs/en/third-party-integrations) .

- Terminal
- VS Code
- Desktop app
- Web
- JetBrains

The full-featured CLI for working with Claude Code directly in your terminal. Edit files, run commands, and manage your entire project from the command line. To install Claude Code, use one of the following methods:

- Native Install (Recommended)
- Homebrew
- WinGet

**macOS, Linux, WSL:**

```
curl -fsSL https://claude.ai/install.sh | bash
```

**Windows PowerShell:**

```
irm https: // claude.ai / install.ps1 | iex
```

**Windows CMD:**

```
curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd
```

If you see `The token '&&' is not a valid statement separator` , you're in PowerShell, not CMD. Use the PowerShell command above instead. Your prompt shows `PS C:\` when you're in PowerShell. **Windows requires** [**Git for Windows**](https://git-scm.com/downloads/win) **.** Install it first if you don't have it.

Native installations automatically update in the background to keep you on the latest version.

```
brew install --cask claude-code
```

Homebrew offers two casks. `claude-code` tracks the stable release channel, which is typically about a week behind and skips releases with major regressions. `claude-code@latest` tracks the latest channel and receives new versions as soon as they ship.

Homebrew installations do not auto-update. Run `brew upgrade claude-code` or `brew upgrade claude-code@latest` , depending on which cask you installed, to get the latest features and security fixes.

```
winget install Anthropic.ClaudeCode
```

WinGet installations do not auto-update. Run `winget upgrade Anthropic.ClaudeCode` periodically to get the latest features and security fixes.

Then start Claude Code in any project:

```
cd your-project
claude
```

You'll be prompted to log in on first use. That's it! [Continue with the Quickstart →](/docs/en/quickstart)

See [advanced setup](/docs/en/setup) for installation options, manual updates, or uninstallation instructions. Visit [troubleshooting](/docs/en/troubleshooting) if you hit issues.

The VS Code extension provides inline diffs, @-mentions, plan review, and conversation history directly in your editor.

- [Install for VS Code](vscode:extension/anthropic.claude-code)
- [Install for Cursor](cursor:extension/anthropic.claude-code)

Or search for "Claude Code" in the Extensions view ( `Cmd+Shift+X` on Mac, `Ctrl+Shift+X` on Windows/Linux). After installing, open the Command Palette ( `Cmd+Shift+P` / `Ctrl+Shift+P` ), type "Claude Code", and select **Open in New Tab** . [Get started with VS Code →](/docs/en/vs-code#get-started)

A standalone app for running Claude Code outside your IDE or terminal. Review diffs visually, run multiple sessions side by side, schedule recurring tasks, and kick off cloud sessions. Download and install:

- [macOS](https://claude.ai/api/desktop/darwin/universal/dmg/latest/redirect?utm_source=claude_code&utm_medium=docs) (Intel and Apple Silicon)
- [Windows](https://claude.ai/api/desktop/win32/x64/setup/latest/redirect?utm_source=claude_code&utm_medium=docs) (x64)
- [Windows ARM64](https://claude.ai/api/desktop/win32/arm64/setup/latest/redirect?utm_source=claude_code&utm_medium=docs)

After installing, launch Claude, sign in, and click the **Code** tab to start coding. A [paid subscription](https://claude.com/pricing?utm_source=claude_code&utm_medium=docs&utm_content=overview_desktop_pricing) is required. [Learn more about the desktop app →](/docs/en/desktop-quickstart)

Run Claude Code in your browser with no local setup. Kick off long-running tasks and check back when they're done, work on repos you don't have locally, or run multiple tasks in parallel. Available on desktop browsers and the Claude iOS app. Start coding at [claude.ai/code](https://claude.ai/code) . [Get started on the web →](/docs/en/web-quickstart)

A plugin for IntelliJ IDEA, PyCharm, WebStorm, and other JetBrains IDEs with interactive diff viewing and selection context sharing. Install the [Claude Code plugin](https://plugins.jetbrains.com/plugin/27310-claude-code-beta-) from the JetBrains Marketplace and restart your IDE. [Get started with JetBrains →](/docs/en/jetbrains)

### What you can do

Here are some of the ways you can use Claude Code:

Automate the work you keep putting off

Claude Code handles the tedious tasks that eat up your day: writing tests for untested code, fixing lint errors across a project, resolving merge conflicts, updating dependencies, and writing release notes.

```
claude "write tests for the auth module, run them, and fix any failures"
```

Build features and fix bugs

Describe what you want in plain language. Claude Code plans the approach, writes the code across multiple files, and verifies it works. For bugs, paste an error message or describe the symptom. Claude Code traces the issue through your codebase, identifies the root cause, and implements a fix. See [common workflows](/docs/en/common-workflows) for more examples.

Create commits and pull requests

Claude Code works directly with git. It stages changes, writes commit messages, creates branches, and opens pull requests.

```
claude "commit my changes with a descriptive message"
```

In CI, you can automate code review and issue triage with [GitHub Actions](/docs/en/github-actions) or [GitLab CI/CD](/docs/en/gitlab-ci-cd) .

Connect your tools with MCP

The [Model Context Protocol (MCP)](/docs/en/mcp) is an open standard for connecting AI tools to external data sources. With MCP, Claude Code can read your design docs in Google Drive, update tickets in Jira, pull data from Slack, or use your own custom tooling.

Customize with instructions, skills, and hooks

[`CLAUDE.md`](/docs/en/memory) is a markdown file you add to your project root that Claude Code reads at the start of every session. Use it to set coding standards, architecture decisions, preferred libraries, and review checklists. Claude also builds [auto memory](/docs/en/memory#auto-memory) as it works, saving learnings like build commands and debugging insights across sessions without you writing anything. Create [custom commands](/docs/en/skills) to package repeatable workflows your team can share, like `/review-pr` or `/deploy-staging` . [Hooks](/docs/en/hooks) let you run shell commands before or after Claude Code actions, like auto-formatting after every file edit or running lint before a commit.

Run agent teams and build custom agents

Spawn [multiple Claude Code agents](/docs/en/sub-agents) that work on different parts of a task simultaneously. A lead agent coordinates the work, assigns subtasks, and merges results. For fully custom workflows, the [Agent SDK](/docs/en/agent-sdk/overview) lets you build your own agents powered by Claude Code's tools and capabilities, with full control over orchestration, tool access, and permissions.

Pipe, script, and automate with the CLI

Claude Code is composable and follows the Unix philosophy. Pipe logs into it, run it in CI, or chain it with other tools:

```
### Analyze recent log output
tail -200 app.log | claude -p "Slack me if you see any anomalies"

### Automate translations in CI
claude -p "translate new strings into French and raise a PR for review"

### Bulk operations across files
git diff main --name-only | claude -p "review these changed files for security issues"
```

See the [CLI reference](/docs/en/cli-reference) for the full set of commands and flags.

Schedule recurring tasks

Run Claude on a schedule to automate work that repeats: morning PR reviews, overnight CI failure analysis, weekly dependency audits, or syncing docs after PRs merge.

- [Cloud scheduled tasks](/docs/en/web-scheduled-tasks) run on Anthropic-managed infrastructure, so they keep running even when your computer is off. Create them from the web, the Desktop app, or by running `/schedule` in the CLI.
- [Desktop scheduled tasks](/docs/en/desktop-scheduled-tasks) run on your machine, with direct access to your local files and tools
- [`/loop`](/docs/en/scheduled-tasks) repeats a prompt within a CLI session for quick polling

Work from anywhere

Sessions aren't tied to a single surface. Move work between environments as your context changes:

- Step away from your desk and keep working from your phone or any browser with [Remote Control](/docs/en/remote-control)
- Message [Dispatch](/docs/en/desktop#sessions-from-dispatch) a task from your phone and open the Desktop session it creates
- Kick off a long-running task on the [web](/docs/en/claude-code-on-the-web) or [iOS app](https://apps.apple.com/app/claude-by-anthropic/id6473753684) , then pull it into your terminal with `claude --teleport`
- Hand off a terminal session to the [Desktop app](/docs/en/desktop) with `/desktop` for visual diff review
- Route tasks from team chat: mention `@Claude` in [Slack](/docs/en/slack) with a bug report and get a pull request back

### Use Claude Code everywhere

Each surface connects to the same underlying Claude Code engine, so your CLAUDE.md files, settings, and MCP servers work across all of them. Beyond the [Terminal](/docs/en/quickstart) , [VS Code](/docs/en/vs-code) , [JetBrains](/docs/en/jetbrains) , [Desktop](/docs/en/desktop) , and [Web](/docs/en/claude-code-on-the-web) environments above, Claude Code integrates with CI/CD, chat, and browser workflows:

| I want to...                                                                    | Best option                                                                                                             |
|---------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| Continue a local session from my phone or another device                        | [Remote Control](/docs/en/remote-control)                                                                               |
| Push events from Telegram, Discord, iMessage, or my own webhooks into a session | [Channels](/docs/en/channels)                                                                                           |
| Start a task locally, continue on mobile                                        | [Web](/docs/en/claude-code-on-the-web) or [Claude iOS app](https://apps.apple.com/app/claude-by-anthropic/id6473753684) |
| Run Claude on a recurring schedule                                              | [Cloud scheduled tasks](/docs/en/web-scheduled-tasks) or [Desktop scheduled tasks](/docs/en/desktop-scheduled-tasks)    |
| Automate PR reviews and issue triage                                            | [GitHub Actions](/docs/en/github-actions) or [GitLab CI/CD](/docs/en/gitlab-ci-cd)                                      |
| Get automatic code review on every PR                                           | [GitHub Code Review](/docs/en/code-review)                                                                              |
| Route bug reports from Slack to pull requests                                   | [Slack](/docs/en/slack)                                                                                                 |
| Debug live web applications                                                     | [Chrome](/docs/en/chrome)                                                                                               |
| Build custom agents for your own workflows                                      | [Agent SDK](/docs/en/agent-sdk/overview)                                                                                |

### Next steps

Once you've installed Claude Code, these guides help you go deeper.

- [Quickstart](/docs/en/quickstart) : walk through your first real task, from exploring a codebase to committing a fix
- [Store instructions and memories](/docs/en/memory) : give Claude persistent instructions with CLAUDE.md files and auto memory
- [Common workflows](/docs/en/common-workflows) and [best practices](/docs/en/best-practices) : patterns for getting the most out of Claude Code
- [Settings](/docs/en/settings) : customize Claude Code for your workflow
- [Troubleshooting](/docs/en/troubleshooting) : solutions for common issues
- [code.claude.com](https://code.claude.com/) : demos, pricing, and product details

Was this page helpful?

Yes

No

[Quickstart](/docs/en/quickstart)

⌘ I


### How Claude Code works


Understand the agentic loop, built-in tools, and how Claude Code interacts with your project.


Claude Code is an agentic assistant that runs in your terminal. While it excels at coding, it can help with anything you can do from the command line: writing docs, running builds, searching files, researching topics, and more. This guide covers the core architecture, built-in capabilities, and [tips for working effectively](#work-effectively-with-claude-code) . For step-by-step walkthroughs, see [Common workflows](/docs/en/common-workflows) . For extensibility features like skills, MCP, and hooks, see [Extend Claude Code](/docs/en/features-overview) .

### The agentic loop

When you give Claude a task, it works through three phases: **gather context** , **take action** , and **verify results** . These phases blend together. Claude uses tools throughout, whether searching files to understand your code, editing to make changes, or running tests to check its work.

The agentic loop: Your prompt leads to Claude gathering context, taking action, verifying results, and repeating until task complete. You can interrupt at any point.


The loop adapts to what you ask. A question about your codebase might only need context gathering. A bug fix cycles through all three phases repeatedly. A refactor might involve extensive verification. Claude decides what each step requires based on what it learned from the previous step, chaining dozens of actions together and course-correcting along the way. You're part of this loop too. You can interrupt at any point to steer Claude in a different direction, provide additional context, or ask it to try a different approach. Claude works autonomously but stays responsive to your input. The agentic loop is powered by two components: [models](#models) that reason and [tools](#tools) that act. Claude Code serves as the **agentic harness** around Claude: it provides the tools, context management, and execution environment that turn a language model into a capable coding agent.

#### Models

Claude Code uses Claude models to understand your code and reason about tasks. Claude can read code in any language, understand how components connect, and figure out what needs to change to accomplish your goal. For complex tasks, it breaks work into steps, executes them, and adjusts based on what it learns. [Multiple models](/docs/en/model-config) are available with different tradeoffs. Sonnet handles most coding tasks well. Opus provides stronger reasoning for complex architectural decisions. Switch with `/model` during a session or start with `claude --model <name>` . When this guide says "Claude chooses" or "Claude decides," it's the model doing the reasoning.

#### Tools

Tools are what make Claude Code agentic. Without tools, Claude can only respond with text. With tools, Claude can act: read your code, edit files, run commands, search the web, and interact with external services. Each tool use returns information that feeds back into the loop, informing Claude's next decision. The built-in tools generally fall into five categories, each representing a different kind of agency.

| Category              | What Claude can do                                                                                                                                                  |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **File operations**   | Read files, edit code, create new files, rename and reorganize                                                                                                      |
| **Search**            | Find files by pattern, search content with regex, explore codebases                                                                                                 |
| **Execution**         | Run shell commands, start servers, run tests, use git                                                                                                               |
| **Web**               | Search the web, fetch documentation, look up error messages                                                                                                         |
| **Code intelligence** | See type errors and warnings after edits, jump to definitions, find references (requires [code intelligence plugins](/docs/en/discover-plugins#code-intelligence) ) |

These are the primary capabilities. Claude also has tools for spawning subagents, asking you questions, and other orchestration tasks. See [Tools available to Claude](/docs/en/tools-reference) for the complete list. Claude chooses which tools to use based on your prompt and what it learns along the way. When you say "fix the failing tests," Claude might:

1. Run the test suite to see what's failing
2. Read the error output
3. Search for the relevant source files
4. Read those files to understand the code
5. Edit the files to fix the issue
6. Run the tests again to verify

Each tool use gives Claude new information that informs the next step. This is the agentic loop in action. **Extending the base capabilities:** The built-in tools are the foundation. You can extend what Claude knows with [skills](/docs/en/skills) , connect to external services with [MCP](/docs/en/mcp) , automate workflows with [hooks](/docs/en/hooks) , and offload tasks to [subagents](/docs/en/sub-agents) . These extensions form a layer on top of the core agentic loop. See [Extend Claude Code](/docs/en/features-overview) for guidance on choosing the right extension for your needs.

### What Claude can access

This guide focuses on the terminal. Claude Code also runs in [VS Code](/docs/en/vs-code) , [JetBrains IDEs](/docs/en/jetbrains) , and other environments. When you run `claude` in a directory, Claude Code gains access to:

- **Your project.** Files in your directory and subdirectories, plus files elsewhere with your permission.
- **Your terminal.** Any command you could run: build tools, git, package managers, system utilities, scripts. If you can do it from the command line, Claude can too.
- **Your git state.** Current branch, uncommitted changes, and recent commit history.
- **Your** [**CLAUDE.md**](/docs/en/memory) **.** A markdown file where you store project-specific instructions, conventions, and context that Claude should know every session.
- [**Auto memory**](/docs/en/memory#auto-memory) **.** Learnings Claude saves automatically as you work, like project patterns and your preferences. The first 200 lines or 25KB of MEMORY.md, whichever comes first, load at the start of each session.
- **Extensions you configure.** [MCP servers](/docs/en/mcp) for external services, [skills](/docs/en/skills) for workflows, [subagents](/docs/en/sub-agents) for delegated work, and [Claude in Chrome](/docs/en/chrome) for browser interaction.

Because Claude sees your whole project, it can work across it. When you ask Claude to "fix the authentication bug," it searches for relevant files, reads multiple files to understand context, makes coordinated edits across them, runs tests to verify the fix, and commits the changes if you ask. This is different from inline code assistants that only see the current file.

### Environments and interfaces

The agentic loop, tools, and capabilities described above are the same everywhere you use Claude Code. What changes is where the code executes and how you interact with it.

#### Execution environments

Claude Code runs in three environments, each with different tradeoffs for where your code executes.

| Environment        | Where code runs                         | Use case                                                   |
|--------------------|-----------------------------------------|------------------------------------------------------------|
| **Local**          | Your machine                            | Default. Full access to your files, tools, and environment |
| **Cloud**          | Anthropic-managed VMs                   | Offload tasks, work on repos you don't have locally        |
| **Remote Control** | Your machine, controlled from a browser | Use the web UI while keeping everything local              |

#### Interfaces

You can access Claude Code through the terminal, the [desktop app](/docs/en/desktop) , [IDE extensions](/docs/en/vs-code) , [claude.ai/code](https://claude.ai/code) , [Remote Control](/docs/en/remote-control) , [Slack](/docs/en/slack) , and [CI/CD pipelines](/docs/en/github-actions) . The interface determines how you see and interact with Claude, but the underlying agentic loop is identical. See [Use Claude Code everywhere](/docs/en/overview#use-claude-code-everywhere) for the full list.

### Work with sessions

Claude Code saves your conversation locally as you work. Each message, tool use, and result is written to a plaintext JSONL file under `~/.claude/projects/` , which enables [rewinding](#undo-changes-with-checkpoints) , [resuming, and forking](#resume-or-fork-sessions) sessions. Before Claude makes code changes, it also snapshots the affected files so you can revert if needed. For paths, retention, and how to clear this data, see [application data in](/docs/en/claude-directory#application-data) [`~/.claude`](/docs/en/claude-directory#application-data) . **Sessions are independent.** Each new session starts with a fresh context window, without the conversation history from previous sessions. Claude can persist learnings across sessions using [auto memory](/docs/en/memory#auto-memory) , and you can add your own persistent instructions in [CLAUDE.md](/docs/en/memory) .

#### Work across branches

Each Claude Code conversation is a session tied to your current directory. When you resume, you only see sessions from that directory. Claude sees your current branch's files. When you switch branches, Claude sees the new branch's files, but your conversation history stays the same. Claude remembers what you discussed even after switching. Since sessions are tied to directories, you can run parallel Claude sessions by using [git worktrees](/docs/en/common-workflows#run-parallel-claude-code-sessions-with-git-worktrees) , which create separate directories for individual branches.

#### Resume or fork sessions

When you resume a session with `claude --continue` or `claude --resume` , you pick up where you left off using the same session ID. New messages append to the existing conversation. Your full conversation history is restored, but session-scoped permissions are not. You'll need to re-approve those.

Session continuity: resume continues the same session, fork creates a new branch with a new ID.


To branch off and try a different approach without affecting the original session, use the `--fork-session` flag:

```
claude --continue --fork-session
```

This creates a new session ID while preserving the conversation history up to that point. The original session remains unchanged. Like resume, forked sessions don't inherit session-scoped permissions. **Same session in multiple terminals** : If you resume the same session in multiple terminals, both terminals write to the same session file. Messages from both get interleaved, like two people writing in the same notebook. Nothing corrupts, but the conversation becomes jumbled. Each terminal only sees its own messages during the session, but if you resume that session later, you'll see everything interleaved. For parallel work from the same starting point, use `--fork-session` to give each terminal its own clean session.

#### The context window

Claude's context window holds your conversation history, file contents, command outputs, [CLAUDE.md](/docs/en/memory) , [auto memory](/docs/en/memory#auto-memory) , loaded skills, and system instructions. As you work, context fills up. Claude compacts automatically, but instructions from early in the conversation can get lost. Put persistent rules in CLAUDE.md, and run `/context` to see what's using space. For an interactive walkthrough of what loads and when, see [Explore the context window](/docs/en/context-window) .

##### When context fills up

Claude Code manages context automatically as you approach the limit. It clears older tool outputs first, then summarizes the conversation if needed. Your requests and key code snippets are preserved; detailed instructions from early in the conversation may be lost. Put persistent rules in CLAUDE.md rather than relying on conversation history. To control what's preserved during compaction, add a "Compact Instructions" section to CLAUDE.md or run `/compact` with a focus (like `/compact focus on the API changes` ). If a single file or tool output is so large that context refills immediately after each summary, Claude Code stops auto-compacting after a few attempts and shows an error instead of looping. See [Auto-compaction stops with a thrashing error](/docs/en/troubleshooting#auto-compaction-stops-with-a-thrashing-error) for recovery steps. Run `/context` to see what's using space. MCP tool definitions are deferred by default and loaded on demand via [tool search](/docs/en/mcp#scale-with-mcp-tool-search) , so only tool names consume context until Claude uses a specific tool. Run `/mcp` to check per-server costs.

##### Manage context with skills and subagents

Beyond compaction, you can use other features to control what loads into context. [Skills](/docs/en/skills) load on demand. Claude sees skill descriptions at session start, but the full content only loads when a skill is used. For skills you invoke manually, set `disable-model-invocation: true` to keep descriptions out of context until you need them. [Subagents](/docs/en/sub-agents) get their own fresh context, completely separate from your main conversation. Their work doesn't bloat your context. When done, they return a summary. This isolation is why subagents help with long sessions. See [context costs](/docs/en/features-overview#understand-context-costs) for what each feature costs, and [reduce token usage](/docs/en/costs#reduce-token-usage) for tips on managing context.

### Stay safe with checkpoints and permissions

Claude has two safety mechanisms: checkpoints let you undo file changes, and permissions control what Claude can do without asking.

#### Undo changes with checkpoints

**Every file edit is reversible.** Before Claude edits any file, it snapshots the current contents. If something goes wrong, press `Esc` twice to rewind to a previous state, or ask Claude to undo. Checkpoints are local to your session, separate from git. They only cover file changes. Actions that affect remote systems (databases, APIs, deployments) can't be checkpointed, which is why Claude asks before running commands with external side effects.

#### Control what Claude can do

Press `Shift+Tab` to cycle through permission modes:

- **Default** : Claude asks before file edits and shell commands
- **Auto-accept edits** : Claude edits files and runs common filesystem commands like `mkdir` and `mv` without asking, still asks for other commands
- **Plan mode** : Claude uses read-only tools only, creating a plan you can approve before execution
- **Auto mode** : Claude evaluates all actions with background safety checks. Currently a research preview

You can also allow specific commands in `.claude/settings.json` so Claude doesn't ask each time. This is useful for trusted commands like `npm test` or `git status` . Settings can be scoped from organization-wide policies down to personal preferences. See [Permissions](/docs/en/permissions) for details.

### Work effectively with Claude Code

These tips help you get better results from Claude Code.

#### Ask Claude Code for help

Claude Code can teach you how to use it. Ask questions like "how do I set up hooks?" or "what's the best way to structure my CLAUDE.md?" and Claude will explain. Built-in commands also guide you through setup:

- `/init` walks you through creating a CLAUDE.md for your project
- `/agents` helps you configure custom subagents
- `/doctor` diagnoses common issues with your installation

#### It's a conversation

Claude Code is conversational. You don't need perfect prompts. Start with what you want, then refine:

```
Fix the login bug
```

[Claude investigates, tries something]

```
That's not quite right. The issue is in the session handling.
```

[Claude adjusts approach] When the first attempt isn't right, you don't start over. You iterate.

##### Interrupt and steer

You can interrupt Claude at any point. If it's going down the wrong path, just type your correction and press Enter. Claude will stop what it's doing and adjust its approach based on your input. You don't have to wait for it to finish or start over.

#### Be specific upfront

The more precise your initial prompt, the fewer corrections you'll need. Reference specific files, mention constraints, and point to example patterns.

```
The checkout flow is broken for users with expired cards.
Check src/payments/ for the issue, especially token refresh.
Write a failing test first, then fix it.
```

Vague prompts work, but you'll spend more time steering. Specific prompts like the one above often succeed on the first attempt.

#### Give Claude something to verify against

Claude performs better when it can check its own work. Include test cases, paste screenshots of expected UI, or define the output you want.

`Implement validateEmail. Test cases: '` [`[email protected]`](/cdn-cgi/l/email-protection) `' → true,
'invalid' → false, '` [`[email protected]`](/cdn-cgi/l/email-protection) `' → false. Run the tests after.`

For visual work, paste a screenshot of the design and ask Claude to compare its implementation against it.

#### Explore before implementing

For complex problems, separate research from coding. Use plan mode ( `Shift+Tab` twice) to analyze the codebase first:

```
Read src/auth/ and understand how we handle sessions.
Then create a plan for adding OAuth support.
```

Review the plan, refine it through conversation, then let Claude implement. This two-phase approach produces better results than jumping straight to code.

#### Delegate, don't dictate

Think of delegating to a capable colleague. Give context and direction, then trust Claude to figure out the details:

```
The checkout flow is broken for users with expired cards.
The relevant code is in src/payments/. Can you investigate and fix it?
```

You don't need to specify which files to read or what commands to run. Claude figures that out.

### What's next

### Extend with features

Add Skills, MCP connections, and custom commands

### Common workflows

Step-by-step guides for typical tasks

Was this page helpful?

Yes

No

[Changelog](/docs/en/changelog) [Extend Claude Code](/docs/en/features-overview)

⌘ I


### Extend Claude Code


Understand when to use CLAUDE.md, Skills, subagents, hooks, MCP, and plugins.


Claude Code combines a model that reasons about your code with [built-in tools](/docs/en/how-claude-code-works#tools) for file operations, search, execution, and web access. The built-in tools cover most coding tasks. This guide covers the extension layer: features you add to customize what Claude knows, connect it to external services, and automate workflows.

For how the core agentic loop works, see [How Claude Code works](/docs/en/how-claude-code-works) .

**New to Claude Code?** Start with [CLAUDE.md](/docs/en/memory) for project conventions, then add other extensions [as specific triggers come up](#build-your-setup-over-time) .

### Overview

Extensions plug into different parts of the agentic loop:

- [**CLAUDE.md**](/docs/en/memory) adds persistent context Claude sees every session
- [**Skills**](/docs/en/skills) add reusable knowledge and invocable workflows
- [**MCP**](/docs/en/mcp) connects Claude to external services and tools
- [**Subagents**](/docs/en/sub-agents) run their own loops in isolated context, returning summaries
- [**Agent teams**](/docs/en/agent-teams) coordinate multiple independent sessions with shared tasks and peer-to-peer messaging
- [**Hooks**](/docs/en/hooks) run outside the loop entirely as deterministic scripts
- [**Plugins**](/docs/en/plugins) and [**marketplaces**](/docs/en/plugin-marketplaces) package and distribute these features

[Skills](/docs/en/skills) are the most flexible extension. A skill is a markdown file containing knowledge, workflows, or instructions. You can invoke skills with a command like `/deploy` , or Claude can load them automatically when relevant. Skills can run in your current conversation or in an isolated context via subagents.

### Match features to your goal

Features range from always-on context that Claude sees every session, to on-demand capabilities you or Claude can invoke, to background automation that runs on specific events. The table below shows what's available and when each one makes sense.

| Feature                                 | What it does                                               | When to use it                                                                  | Example                                                                         |
|-----------------------------------------|------------------------------------------------------------|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| **CLAUDE.md**                           | Persistent context loaded every conversation               | Project conventions, "always do X" rules                                        | "Use pnpm, not npm. Run tests before committing."                               |
| **Skill**                               | Instructions, knowledge, and workflows Claude can use      | Reusable content, reference docs, repeatable tasks                              | `/deploy` runs your deployment checklist; API docs skill with endpoint patterns |
| **Subagent**                            | Isolated execution context that returns summarized results | Context isolation, parallel tasks, specialized workers                          | Research task that reads many files but returns only key findings               |
| [**Agent teams**](/docs/en/agent-teams) | Coordinate multiple independent Claude Code sessions       | Parallel research, new feature development, debugging with competing hypotheses | Spawn reviewers to check security, performance, and tests simultaneously        |
| **MCP**                                 | Connect to external services                               | External data or actions                                                        | Query your database, post to Slack, control a browser                           |
| **Hook**                                | Deterministic script that runs on events                   | Predictable automation, no LLM involved                                         | Run ESLint after every file edit                                                |

[**Plugins**](/docs/en/plugins) are the packaging layer. A plugin bundles skills, hooks, subagents, and MCP servers into a single installable unit. Plugin skills are namespaced (like `/my-plugin:review` ) so multiple plugins can coexist. Use plugins when you want to reuse the same setup across multiple repositories or distribute to others via a [**marketplace**](/docs/en/plugin-marketplaces) .

#### Build your setup over time

You don't need to configure everything up front. Each feature has a recognizable trigger, and most teams add them in roughly this order:

| Trigger                                                                          | Add                                                  |
|----------------------------------------------------------------------------------|------------------------------------------------------|
| Claude gets a convention or command wrong twice                                  | Add it to [CLAUDE.md](/docs/en/memory)               |
| You keep typing the same prompt to start a task                                  | Save it as a user-invocable [skill](/docs/en/skills) |
| You paste the same playbook or multi-step procedure into chat for the third time | Capture it as a [skill](/docs/en/skills)             |
| You keep copying data from a browser tab Claude can't see                        | Connect that system as an [MCP server](/docs/en/mcp) |
| A side task floods your conversation with output you won't reference again       | Route it through a [subagent](/docs/en/sub-agents)   |
| You want something to happen every time without asking                           | Write a [hook](/docs/en/hooks-guide)                 |
| A second repository needs the same setup                                         | Package it as a [plugin](/docs/en/plugins)           |

The same triggers tell you when to update what you already have. A repeated mistake or a recurring review comment is a CLAUDE.md edit, not a one-off correction in chat. A workflow you keep tweaking by hand is a skill that needs another revision.

#### Compare similar features

Some features can seem similar. Here's how to tell them apart.

- Skill vs Subagent
- CLAUDE.md vs Skill
- CLAUDE.md vs Rules vs Skills
- Subagent vs Agent team
- MCP vs Skill

Skills and subagents solve different problems:

- **Skills** are reusable content you can load into any context
- **Subagents** are isolated workers that run separately from your main conversation

| Aspect          | Skill                                          | Subagent                                                         |
|-----------------|------------------------------------------------|------------------------------------------------------------------|
| **What it is**  | Reusable instructions, knowledge, or workflows | Isolated worker with its own context                             |
| **Key benefit** | Share content across contexts                  | Context isolation. Work happens separately, only summary returns |
| **Best for**    | Reference material, invocable workflows        | Tasks that read many files, parallel work, specialized workers   |

**Skills can be reference or action.** Reference skills provide knowledge Claude uses throughout your session (like your API style guide). Action skills tell Claude to do something specific (like `/deploy` that runs your deployment workflow). **Use a subagent** when you need context isolation or when your context window is getting full. The subagent might read dozens of files or run extensive searches, but your main conversation only receives a summary. Since subagent work doesn't consume your main context, this is also useful when you don't need the intermediate work to remain visible. Custom subagents can have their own instructions and can preload skills. **They can combine.** A subagent can preload specific skills ( `skills:` field). A skill can run in isolated context using `context: fork` . See [Skills](/docs/en/skills) for details.

Both store instructions, but they load differently and serve different purposes.

| Aspect                    | CLAUDE.md                    | Skill                                   |
|---------------------------|------------------------------|-----------------------------------------|
| **Loads**                 | Every session, automatically | On demand                               |
| **Can include files**     | Yes, with `@path` imports    | Yes, with `@path` imports               |
| **Can trigger workflows** | No                           | Yes, with `/<name>`                     |
| **Best for**              | "Always do X" rules          | Reference material, invocable workflows |

**Put it in CLAUDE.md** if Claude should always know it: coding conventions, build commands, project structure, "never do X" rules. **Put it in a skill** if it's reference material Claude needs sometimes (API docs, style guides) or a workflow you trigger with `/<name>` (deploy, review, release). **Rule of thumb:** Keep CLAUDE.md under 200 lines. If it's growing, move reference content to skills or split into [`.claude/rules/`](/docs/en/memory#organize-rules-with-claude/rules) files.

All three store instructions, but they load differently:

| Aspect       | CLAUDE.md                           | `.claude/rules/`                                   | Skill                                    |
|--------------|-------------------------------------|----------------------------------------------------|------------------------------------------|
| **Loads**    | Every session                       | Every session, or when matching files are opened   | On demand, when invoked or relevant      |
| **Scope**    | Whole project                       | Can be scoped to file paths                        | Task-specific                            |
| **Best for** | Core conventions and build commands | Language-specific or directory-specific guidelines | Reference material, repeatable workflows |

**Use CLAUDE.md** for instructions every session needs: build commands, test conventions, project architecture. **Use rules** to keep CLAUDE.md focused. Rules with [`paths`](/docs/en/memory#path-specific-rules) [frontmatter](/docs/en/memory#path-specific-rules) only load when Claude works with matching files, saving context. **Use skills** for content Claude only needs sometimes, like API documentation or a deployment checklist you trigger with `/<name>` .

Both parallelize work, but they're architecturally different:

- **Subagents** run inside your session and report results back to your main context
- **Agent teams** are independent Claude Code sessions that communicate with each other

| Aspect            | Subagent                                         | Agent team                                          |
|-------------------|--------------------------------------------------|-----------------------------------------------------|
| **Context**       | Own context window; results return to the caller | Own context window; fully independent               |
| **Communication** | Reports results back to the main agent only      | Teammates message each other directly               |
| **Coordination**  | Main agent manages all work                      | Shared task list with self-coordination             |
| **Best for**      | Focused tasks where only the result matters      | Complex work requiring discussion and collaboration |
| **Token cost**    | Lower: results summarized back to main context   | Higher: each teammate is a separate Claude instance |

**Use a subagent** when you need a quick, focused worker: research a question, verify a claim, review a file. The subagent does the work and returns a summary. Your main conversation stays clean. **Use an agent team** when teammates need to share findings, challenge each other, and coordinate independently. Agent teams are best for research with competing hypotheses, parallel code review, and new feature development where each teammate owns a separate piece. **Transition point:** If you're running parallel subagents but hitting context limits, or if your subagents need to communicate with each other, agent teams are the natural next step.

Agent teams are experimental and disabled by default. See [agent teams](/docs/en/agent-teams) for setup and current limitations.

MCP connects Claude to external services. Skills extend what Claude knows, including how to use those services effectively.

| Aspect         | MCP                                                  | Skill                                                   |
|----------------|------------------------------------------------------|---------------------------------------------------------|
| **What it is** | Protocol for connecting to external services         | Knowledge, workflows, and reference material            |
| **Provides**   | Tools and data access                                | Knowledge, workflows, reference material                |
| **Examples**   | Slack integration, database queries, browser control | Code review checklist, deploy workflow, API style guide |

These solve different problems and work well together: **MCP** gives Claude the ability to interact with external systems. Without MCP, Claude can't query your database or post to Slack. **Skills** give Claude knowledge about how to use those tools effectively, plus workflows you can trigger with `/<name>` . A skill might include your team's database schema and query patterns, or a `/post-to-slack` workflow with your team's message formatting rules. Example: An MCP server connects Claude to your database. A skill teaches Claude your data model, common query patterns, and which tables to use for different tasks.

#### Understand how features layer

Features can be defined at multiple levels: user-wide, per-project, via plugins, or through managed policies. You can also nest CLAUDE.md files in subdirectories or place skills in specific packages of a monorepo. When the same feature exists at multiple levels, here's how they layer:

- **CLAUDE.md files** are additive: all levels contribute content to Claude's context simultaneously. Files from your working directory and above load at launch; subdirectories load as you work in them. When instructions conflict, Claude uses judgment to reconcile them, with more specific instructions typically taking precedence. See [how CLAUDE.md files load](/docs/en/memory#how-claude-md-files-load) .
- **Skills and subagents** override by name: when the same name exists at multiple levels, one definition wins based on priority (managed > user > project for skills; managed > CLI flag > project > user > plugin for subagents). Plugin skills are [namespaced](/docs/en/plugins#add-skills-to-your-plugin) to avoid conflicts. See [skill discovery](/docs/en/skills#where-skills-live) and [subagent scope](/docs/en/sub-agents#choose-the-subagent-scope) .
- **MCP servers** override by name: local > project > user. See [MCP scope](/docs/en/mcp#scope-hierarchy-and-precedence) .
- **Hooks** merge: all registered hooks fire for their matching events regardless of source. See [hooks](/docs/en/hooks) .

#### Combine features

Each extension solves a different problem: CLAUDE.md handles always-on context, skills handle on-demand knowledge and workflows, MCP handles external connections, subagents handle isolation, and hooks handle automation. Real setups combine them based on your workflow. For example, you might use CLAUDE.md for project conventions, a skill for your deployment workflow, MCP to connect to your database, and a hook to run linting after every edit. Each feature handles what it's best at.

| Pattern                | How it works                                                                     | Example                                                                                           |
|------------------------|----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **Skill + MCP**        | MCP provides the connection; a skill teaches Claude how to use it well           | MCP connects to your database, a skill documents your schema and query patterns                   |
| **Skill + Subagent**   | A skill spawns subagents for parallel work                                       | `/audit` skill kicks off security, performance, and style subagents that work in isolated context |
| **CLAUDE.md + Skills** | CLAUDE.md holds always-on rules; skills hold reference material loaded on demand | CLAUDE.md says "follow our API conventions," a skill contains the full API style guide            |
| **Hook + MCP**         | A hook triggers external actions through MCP                                     | Post-edit hook sends a Slack notification when Claude modifies critical files                     |

### Understand context costs

Every feature you add consumes some of Claude's context. Too much can fill up your context window, but it can also add noise that makes Claude less effective; skills may not trigger correctly, or Claude may lose track of your conventions. Understanding these trade-offs helps you build an effective setup. For an interactive view of how these features combine in a running session, see [Explore the context window](/docs/en/context-window) .

#### Context cost by feature

Each feature has a different loading strategy and context cost:

| Feature         | When it loads             | What loads                                    | Context cost                                 |
|-----------------|---------------------------|-----------------------------------------------|----------------------------------------------|
| **CLAUDE.md**   | Session start             | Full content                                  | Every request                                |
| **Skills**      | Session start + when used | Descriptions at start, full content when used | Low (descriptions every request)*            |
| **MCP servers** | Session start             | Tool names; full schemas on demand            | Low until a tool is used                     |
| **Subagents**   | When spawned              | Fresh context with specified skills           | Isolated from main session                   |
| **Hooks**       | On trigger                | Nothing (runs externally)                     | Zero, unless hook returns additional context |

*By default, skill descriptions load at session start so Claude can decide when to use them. Set `disable-model-invocation: true` in a skill's frontmatter to hide it from Claude entirely until you invoke it manually. This reduces context cost to zero for skills you only trigger yourself.

#### Understand how features load

Each feature loads at different points in your session. The tabs below explain when each one loads and what goes into context.

Context loading: CLAUDE.md loads at session start and stays in every request. MCP tool names load at start with full schemas deferred until use. Skills load descriptions at start, full content on invocation. Subagents get isolated context. Hooks run externally.


- CLAUDE.md
- Skills
- MCP servers
- Subagents
- Hooks

**When:** Session start **What loads:** Full content of all CLAUDE.md files (managed, user, and project levels). **Inheritance:** Claude reads CLAUDE.md files from your working directory up to the root, and discovers nested ones in subdirectories as it accesses those files. See [How CLAUDE.md files load](/docs/en/memory#how-claude-md-files-load) for details.

Keep CLAUDE.md under 200 lines. Move reference material to skills, which load on-demand.

Skills are extra capabilities in Claude's toolkit. They can be reference material (like an API style guide) or invocable workflows you trigger with `/<name>` (like `/deploy` ). Claude Code includes [bundled skills](/docs/en/commands) like `/simplify` , `/batch` , and `/debug` that work out of the box. You can also create your own. Claude uses skills when appropriate, or you can invoke one directly. **When:** Depends on the skill's configuration. By default, descriptions load at session start and full content loads when used. For user-only skills ( `disable-model-invocation: true` ), nothing loads until you invoke them. **What loads:** For model-invocable skills, Claude sees names and descriptions in every request. When you invoke a skill with `/<name>` or Claude loads it automatically, the full content loads into your conversation. **How Claude chooses skills:** Claude matches your task against skill descriptions to decide which are relevant. If descriptions are vague or overlap, Claude may load the wrong skill or miss one that would help. To tell Claude to use a specific skill, invoke it with `/<name>` . Skills with `disable-model-invocation: true` are invisible to Claude until you invoke them. **Context cost:** Low until used. User-only skills have zero cost until invoked. **In subagents:** Skills work differently in subagents. Instead of on-demand loading, skills passed to a subagent are fully preloaded into its context at launch. Subagents don't inherit skills from the main session; you must specify them explicitly.

Use `disable-model-invocation: true` for skills with side effects. This saves context and ensures only you trigger them.

**When:** Session start. **What loads:** Tool names from connected servers. Full JSON schemas stay deferred until Claude needs a specific tool. **Context cost:** [Tool search](/docs/en/mcp#scale-with-mcp-tool-search) is on by default, so idle MCP tools consume minimal context. **Reliability note:** MCP connections can fail silently mid-session. If a server disconnects, its tools disappear without warning. Claude may try to use a tool that no longer exists. If you notice Claude failing to use an MCP tool it previously could access, check the connection with `/mcp` .

Run `/mcp` to see token costs per server. Disconnect servers you're not actively using.

**When:** On demand, when you or Claude spawns one for a task. **What loads:** Fresh, isolated context containing:

- The system prompt (shared with parent for cache efficiency)
- Full content of skills listed in the agent's `skills:` field
- CLAUDE.md and git status (inherited from parent)
- Whatever context the lead agent passes in the prompt

**Context cost:** Isolated from main session. Subagents don't inherit your conversation history or invoked skills.

Use subagents for work that doesn't need your full conversation context. Their isolation prevents bloating your main session.

**When:** On trigger. Hooks fire at specific lifecycle events like tool execution, session boundaries, prompt submission, permission requests, and compaction. See [Hooks](/docs/en/hooks) for the full list. **What loads:** Nothing by default. Hooks run as external scripts. **Context cost:** Zero, unless the hook returns output that gets added as messages to your conversation.

Hooks are ideal for side effects (linting, logging) that don't need to affect Claude's context.

### Learn more

Each feature has its own guide with setup instructions, examples, and configuration options.

### CLAUDE.md

Store project context, conventions, and instructions

### Skills

Give Claude domain expertise and reusable workflows

### Subagents

Offload work to isolated context

### Agent teams

Coordinate multiple sessions working in parallel

### MCP

Connect Claude to external services

### Hooks

Automate workflows with hooks

### Plugins

Bundle and share feature sets

### Marketplaces

Host and distribute plugin collections

Was this page helpful?

Yes

No

[How Claude Code works](/docs/en/how-claude-code-works) [Explore the .claude directory](/docs/en/claude-directory)

⌘ I


---

# Getting Started


### Advanced setup


System requirements, platform-specific installation, version management, and uninstallation for Claude Code.


This page covers system requirements, platform-specific installation details, updates, and uninstallation. For a guided walkthrough of your first session, see the [quickstart](/docs/en/quickstart) . If you've never used a terminal before, see the [terminal guide](/docs/en/terminal-guide) .

### System requirements

Claude Code runs on the following platforms and configurations:

- **Operating system** :
    - macOS 13.0+
    - Windows 10 1809+ or Windows Server 2019+
    - Ubuntu 20.04+
    - Debian 10+
    - Alpine Linux 3.19+
- **Hardware** : 4 GB+ RAM, x64 or ARM64 processor
- **Network** : internet connection required. See [network configuration](/docs/en/network-config#network-access-requirements) .
- **Shell** : Bash, Zsh, PowerShell, or CMD. On Windows, [Git for Windows](https://git-scm.com/downloads/win) is required.
- **Location** : [Anthropic supported countries](https://www.anthropic.com/supported-countries)

#### Additional dependencies

- **ripgrep** : usually included with Claude Code. If search fails, see [search troubleshooting](/docs/en/troubleshooting#search-and-discovery-issues) .

### Install Claude Code

Prefer a graphical interface? The [Desktop app](/docs/en/desktop-quickstart) lets you use Claude Code without the terminal. Download it for [macOS](https://claude.ai/api/desktop/darwin/universal/dmg/latest/redirect?utm_source=claude_code&utm_medium=docs) or [Windows](https://claude.com/download?utm_source=claude_code&utm_medium=docs) . New to the terminal? See the [terminal guide](/docs/en/terminal-guide) for step-by-step instructions.

To install Claude Code, use one of the following methods:

- Native Install (Recommended)
- Homebrew
- WinGet

**macOS, Linux, WSL:**

```
curl -fsSL https://claude.ai/install.sh | bash
```

**Windows PowerShell:**

```
irm https: // claude.ai / install.ps1 | iex
```

**Windows CMD:**

```
curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd
```

If you see `The token '&&' is not a valid statement separator` , you're in PowerShell, not CMD. Use the PowerShell command above instead. Your prompt shows `PS C:\` when you're in PowerShell. **Windows requires** [**Git for Windows**](https://git-scm.com/downloads/win) **.** Install it first if you don't have it.

Native installations automatically update in the background to keep you on the latest version.

```
brew install --cask claude-code
```

Homebrew offers two casks. `claude-code` tracks the stable release channel, which is typically about a week behind and skips releases with major regressions. `claude-code@latest` tracks the latest channel and receives new versions as soon as they ship.

Homebrew installations do not auto-update. Run `brew upgrade claude-code` or `brew upgrade claude-code@latest` , depending on which cask you installed, to get the latest features and security fixes.

```
winget install Anthropic.ClaudeCode
```

WinGet installations do not auto-update. Run `winget upgrade Anthropic.ClaudeCode` periodically to get the latest features and security fixes.

After installation completes, open a terminal in the project you want to work in and start Claude Code:

```
claude
```

If you encounter any issues during installation, see the [troubleshooting guide](/docs/en/troubleshooting) .

#### Set up on Windows

Claude Code on Windows requires [Git for Windows](https://git-scm.com/downloads/win) or WSL. You can launch `claude` from PowerShell, CMD, or Git Bash. Claude Code uses Git Bash internally to run commands. You do not need to run PowerShell as Administrator. **Option 1: Native Windows with Git Bash** Install [Git for Windows](https://git-scm.com/downloads/win) , then run the install command from PowerShell or CMD. If Claude Code can't find your Git Bash installation, set the path in your [settings.json file](/docs/en/settings) :

```
{
"env" : {
"CLAUDE_CODE_GIT_BASH_PATH" : "C: \\ Program Files \\ Git \\ bin \\ bash.exe"
}
}
```

Claude Code can also run PowerShell natively on Windows as an opt-in preview. See [PowerShell tool](/docs/en/tools-reference#powershell-tool) for setup and limitations. **Option 2: WSL** Both WSL 1 and WSL 2 are supported. WSL 2 supports [sandboxing](/docs/en/sandboxing) for enhanced security. WSL 1 does not support sandboxing.

#### Alpine Linux and musl-based distributions

The native installer on Alpine and other musl/uClibc-based distributions requires `libgcc` , `libstdc++` , and `ripgrep` . Install these using your distribution's package manager, then set `USE_BUILTIN_RIPGREP=0` . This example installs the required packages on Alpine:

```
apk add libgcc libstdc++ ripgrep
```

Then set `USE_BUILTIN_RIPGREP` to `0` in your [`settings.json`](/docs/en/settings#available-settings) file:

```
{
"env" : {
"USE_BUILTIN_RIPGREP" : "0"
}
}
```

### Verify your installation

After installing, confirm Claude Code is working:

```
claude --version
```

For a more detailed check of your installation and configuration, run [`claude doctor`](/docs/en/troubleshooting#get-more-help) :

```
claude doctor
```

### Authenticate

Claude Code requires a Pro, Max, Team, Enterprise, or Console account. The free Claude.ai plan does not include Claude Code access. You can also use Claude Code with a third-party API provider like [Amazon Bedrock](/docs/en/amazon-bedrock) , [Google Vertex AI](/docs/en/google-vertex-ai) , or [Microsoft Foundry](/docs/en/microsoft-foundry) . After installing, log in by running `claude` and following the browser prompts. See [Authentication](/docs/en/authentication) for all account types and team setup options.

### Update Claude Code

Native installations automatically update in the background. You can [configure the release channel](#configure-release-channel) to control whether you receive updates immediately or on a delayed stable schedule, or [disable auto-updates](#disable-auto-updates) entirely. Homebrew and WinGet installations require manual updates.

#### Auto-updates

Claude Code checks for updates on startup and periodically while running. Updates download and install in the background, then take effect the next time you start Claude Code.

Homebrew and WinGet installations do not auto-update. For Homebrew, run `brew upgrade claude-code` or `brew upgrade claude-code@latest` , depending on which cask you installed. For WinGet, run `winget upgrade Anthropic.ClaudeCode` . **Known issue:** Claude Code may notify you of updates before the new version is available in these package managers. If an upgrade fails, wait and try again later. Homebrew keeps old versions on disk after upgrades. Run `brew cleanup` periodically to reclaim disk space.

#### Configure release channel

Control which release channel Claude Code follows for auto-updates and `claude update` with the `autoUpdatesChannel` setting:

- `"latest"` , the default: receive new features as soon as they're released
- `"stable"` : use a version that is typically about one week old, skipping releases with major regressions

Configure this via `/config` → **Auto-update channel** , or add it to your [settings.json file](/docs/en/settings) :

```
{
"autoUpdatesChannel" : "stable"
}
```

For enterprise deployments, you can enforce a consistent release channel across your organization using [managed settings](/docs/en/permissions#managed-settings) . Homebrew installations choose a channel by cask name instead of this setting: `claude-code` tracks stable and `claude-code@latest` tracks latest.

#### Disable auto-updates

Set `DISABLE_AUTOUPDATER` to `"1"` in the `env` key of your [`settings.json`](/docs/en/settings#available-settings) file:

```
{
"env" : {
"DISABLE_AUTOUPDATER" : "1"
}
}
```

#### Update manually

To apply an update immediately without waiting for the next background check, run:

```
claude update
```

### Advanced installation options

These options are for version pinning, migrating from npm, and verifying binary integrity.

#### Install a specific version

The native installer accepts either a specific version number or a release channel ( `latest` or `stable` ). The channel you choose at install time becomes your default for auto-updates. See [configure release channel](#configure-release-channel) for more information. To install the latest version (default):

- macOS, Linux, WSL
- Windows PowerShell
- Windows CMD

```
curl -fsSL https://claude.ai/install.sh | bash
```

```
irm https: // claude.ai / install.ps1 | iex
```

```
curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd
```

To install the stable version:

- macOS, Linux, WSL
- Windows PowerShell
- Windows CMD

```
curl -fsSL https://claude.ai/install.sh | bash -s stable
```

```
& ([ scriptblock ]::Create((irm https: // claude.ai / install.ps1))) stable
```

```
curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd stable && del install.cmd
```

To install a specific version number:

- macOS, Linux, WSL
- Windows PowerShell
- Windows CMD

```
curl -fsSL https://claude.ai/install.sh | bash -s 2.1.89
```

```
& ([ scriptblock ]::Create((irm https: // claude.ai / install.ps1))) 2.1 . 89
```

```
curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd 2.1.89 && del install.cmd
```

#### Deprecated npm installation

npm installation is deprecated. The native installer is faster, requires no dependencies, and auto-updates in the background. Use the [native installation](#install-claude-code) method when possible.

##### Migrate from npm to native

If you previously installed Claude Code with npm, switch to the native installer:

```
### Install the native binary
curl -fsSL https://claude.ai/install.sh | bash

### Remove the old npm installation
npm uninstall -g @anthropic-ai/claude-code
```

You can also run `claude install` from an existing npm installation to install the native binary alongside it, then remove the npm version.

##### Install with npm

If you need npm installation for compatibility reasons, you must have [Node.js 18+](https://nodejs.org/en/download) installed. Install the package globally:

```
npm install -g @anthropic-ai/claude-code
```

Do NOT use `sudo npm install -g` as this can lead to permission issues and security risks. If you encounter permission errors, see [troubleshooting permission errors](/docs/en/troubleshooting#permission-errors-during-installation) .

#### Binary integrity and code signing

Each release publishes a `manifest.json` containing SHA256 checksums for every platform binary. The manifest is signed with an Anthropic GPG key, so verifying the signature on the manifest transitively verifies every binary it lists.

##### Verify the manifest signature

Steps 1-3 require a POSIX shell with `gpg` and `curl` . On Windows, run them in Git Bash or WSL. Step 4 includes a PowerShell option.

1

Download and import the public key

The release signing key is published at a fixed URL.

```
curl -fsSL https://downloads.claude.ai/keys/claude-code.asc | gpg --import
```

Display the fingerprint of the imported key.

`gpg --fingerprint` [`[email protected]`](/cdn-cgi/l/email-protection) ``

Confirm the output includes this fingerprint:

```
31DD DE24 DDFA B679 F42D  7BD2 BAA9 29FF 1A7E CACE
```

2

Download the manifest and signature

Set `VERSION` to the release you want to verify.

```
REPO = https://storage.googleapis.com/claude-code-dist-86c565f3-f756-42ad-8dfa-d59b1c096819/claude-code-releases
VERSION = 2.1.89
curl -fsSLO " $REPO / $VERSION /manifest.json"
curl -fsSLO " $REPO / $VERSION /manifest.json.sig"
```

3

Verify the signature

Verify the detached signature against the manifest.

```
gpg --verify manifest.json.sig manifest.json
```

A valid result reports `Good signature from "Anthropic Claude Code Release Signing <` [`[email protected]`](/cdn-cgi/l/email-protection) `>"` . `gpg` also prints `WARNING: This key is not certified with a trusted signature!` for any freshly imported key. This is expected. The `Good signature` line confirms the cryptographic check passed. The fingerprint comparison in Step 1 confirms the key itself is authentic.

4

Check the binary against the manifest

Compare the SHA256 checksum of your downloaded binary with the value listed under `platforms.<platform>.checksum` in `manifest.json` .

- Linux
- macOS
- Windows PowerShell

```
sha256sum claude
```

```
shasum -a 256 claude
```

```
( Get-FileHash claude.exe - Algorithm SHA256).Hash.ToLower()
```

Manifest signatures are available for releases from `2.1.89` onward. Earlier releases publish checksums in `manifest.json` without a detached signature.

##### Platform code signatures

In addition to the signed manifest, individual binaries carry platform-native code signatures where supported.

- **macOS** : signed by "Anthropic PBC" and notarized by Apple. Verify with `codesign --verify --verbose ./claude` .
- **Windows** : signed by "Anthropic, PBC". Verify with `Get-AuthenticodeSignature .\claude.exe` .
- **Linux** : use the manifest signature above to verify integrity. Linux binaries are not individually code-signed.

### Uninstall Claude Code

To remove Claude Code, follow the instructions for your installation method.

#### Native installation

Remove the Claude Code binary and version files:

- macOS, Linux, WSL
- Windows PowerShell

```
rm -f ~/.local/bin/claude
rm -rf ~/.local/share/claude
```

```
Remove-Item - Path " $ env: USERPROFILE \.local\bin\claude.exe" - Force
Remove-Item - Path " $ env: USERPROFILE \.local\share\claude" - Recurse - Force
```

#### Homebrew installation

Remove the Homebrew cask you installed. If you installed the stable cask:

```
brew uninstall --cask claude-code
```

If you installed the latest cask:

```
brew uninstall --cask claude-code@latest
```

#### WinGet installation

Remove the WinGet package:

```
winget uninstall Anthropic.ClaudeCode
```

#### npm

Remove the global npm package:

```
npm uninstall -g @anthropic-ai/claude-code
```

#### Remove configuration files

Removing configuration files will delete all your settings, allowed tools, MCP server configurations, and session history.

To remove Claude Code settings and cached data:

- macOS, Linux, WSL
- Windows PowerShell

```
### Remove user settings and state
rm -rf ~/.claude
rm ~/.claude.json

### Remove project-specific settings (run from your project directory)
rm -rf .claude
rm -f .mcp.json
```

```
### Remove user settings and state
Remove-Item - Path " $ env: USERPROFILE \.claude" - Recurse - Force
Remove-Item - Path " $ env: USERPROFILE \.claude.json" - Force

### Remove project-specific settings (run from your project directory)
Remove-Item - Path ".claude" - Recurse - Force
Remove-Item - Path ".mcp.json" - Force
```

Was this page helpful?

Yes

No

[Authentication](/docs/en/authentication)

⌘ I


### Quickstart


Welcome to Claude Code!


This quickstart guide will have you using AI-powered coding assistance in a few minutes. By the end, you'll understand how to use Claude Code for common development tasks.

### Before you begin

Make sure you have:

- A terminal or command prompt open
    - If you've never used the terminal before, check out the [terminal guide](/docs/en/terminal-guide)
- A code project to work with
- A [Claude subscription](https://claude.com/pricing?utm_source=claude_code&utm_medium=docs&utm_content=quickstart_prereq) (Pro, Max, Team, or Enterprise), [Claude Console](https://console.anthropic.com/) account, or access through a [supported cloud provider](/docs/en/third-party-integrations)

This guide covers the terminal CLI. Claude Code is also available on the [web](https://claude.ai/code) , as a [desktop app](/docs/en/desktop) , in [VS Code](/docs/en/vs-code) and [JetBrains IDEs](/docs/en/jetbrains) , in [Slack](/docs/en/slack) , and in CI/CD with [GitHub Actions](/docs/en/github-actions) and [GitLab](/docs/en/gitlab-ci-cd) . See [all interfaces](/docs/en/overview#use-claude-code-everywhere) .

### Step 1: Install Claude Code

To install Claude Code, use one of the following methods:

- Native Install (Recommended)
- Homebrew
- WinGet

**macOS, Linux, WSL:**

```
curl -fsSL https://claude.ai/install.sh | bash
```

**Windows PowerShell:**

```
irm https: // claude.ai / install.ps1 | iex
```

**Windows CMD:**

```
curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd
```

If you see `The token '&&' is not a valid statement separator` , you're in PowerShell, not CMD. Use the PowerShell command above instead. Your prompt shows `PS C:\` when you're in PowerShell. **Windows requires** [**Git for Windows**](https://git-scm.com/downloads/win) **.** Install it first if you don't have it.

Native installations automatically update in the background to keep you on the latest version.

```
brew install --cask claude-code
```

Homebrew offers two casks. `claude-code` tracks the stable release channel, which is typically about a week behind and skips releases with major regressions. `claude-code@latest` tracks the latest channel and receives new versions as soon as they ship.

Homebrew installations do not auto-update. Run `brew upgrade claude-code` or `brew upgrade claude-code@latest` , depending on which cask you installed, to get the latest features and security fixes.

```
winget install Anthropic.ClaudeCode
```

WinGet installations do not auto-update. Run `winget upgrade Anthropic.ClaudeCode` periodically to get the latest features and security fixes.

### Step 2: Log in to your account

Claude Code requires an account to use. When you start an interactive session with the `claude` command, you'll need to log in:

```
claude
### You'll be prompted to log in on first use
```

```
/login
### Follow the prompts to log in with your account
```

You can log in using any of these account types:

- [Claude Pro, Max, Team, or Enterprise](https://claude.com/pricing?utm_source=claude_code&utm_medium=docs&utm_content=quickstart_login) (recommended)
- [Claude Console](https://console.anthropic.com/) (API access with pre-paid credits). On first login, a "Claude Code" workspace is automatically created in the Console for centralized cost tracking.
- [Amazon Bedrock, Google Vertex AI, or Microsoft Foundry](/docs/en/third-party-integrations) (enterprise cloud providers)

Once logged in, your credentials are stored and you won't need to log in again. To switch accounts later, use the `/login` command.

### Step 3: Start your first session

Open your terminal in any project directory and start Claude Code:

```
cd /path/to/your/project
claude
```

You'll see the Claude Code welcome screen with your session information, recent conversations, and latest updates. Type `/help` for available commands or `/resume` to continue a previous conversation.

After logging in (Step 2), your credentials are stored on your system. Learn more in [Credential Management](/docs/en/authentication#credential-management) .

### Step 4: Ask your first question

Let's start with understanding your codebase. Try one of these commands:

```
what does this project do?
```

Claude will analyze your files and provide a summary. You can also ask more specific questions:

```
what technologies does this project use?
```

```
where is the main entry point?
```

```
explain the folder structure
```

You can also ask Claude about its own capabilities:

```
what can Claude Code do?
```

```
how do I create custom skills in Claude Code?
```

```
can Claude Code work with Docker?
```

Claude Code reads your project files as needed. You don't have to manually add context.

### Step 5: Make your first code change

Now let's make Claude Code do some actual coding. Try a simple task:

```
add a hello world function to the main file
```

Claude Code will:

1. Find the appropriate file
2. Show you the proposed changes
3. Ask for your approval
4. Make the edit

Claude Code always asks for permission before modifying files. You can approve individual changes or enable "Accept all" mode for a session.

### Step 6: Use Git with Claude Code

Claude Code makes Git operations conversational:

```
what files have I changed?
```

```
commit my changes with a descriptive message
```

You can also prompt for more complex Git operations:

```
create a new branch called feature/quickstart
```

```
show me the last 5 commits
```

```
help me resolve merge conflicts
```

### Step 7: Fix a bug or add a feature

Claude is proficient at debugging and feature implementation. Describe what you want in natural language:

```
add input validation to the user registration form
```

Or fix existing issues:

```
there's a bug where users can submit empty forms - fix it
```

Claude Code will:

- Locate the relevant code
- Understand the context
- Implement a solution
- Run tests if available

### Step 8: Test out other common workflows

There are a number of ways to work with Claude: **Refactor code**

```
refactor the authentication module to use async/await instead of callbacks
```

**Write tests**

```
write unit tests for the calculator functions
```

**Update documentation**

```
update the README with installation instructions
```

**Code review**

```
review my changes and suggest improvements
```

Talk to Claude like you would a helpful colleague. Describe what you want to achieve, and it will help you get there.

### Essential commands

Here are the most important commands for daily use:

| Command             | What it does                                           | Example                             |
|---------------------|--------------------------------------------------------|-------------------------------------|
| `claude`            | Start interactive mode                                 | `claude`                            |
| `claude "task"`     | Run a one-time task                                    | `claude "fix the build error"`      |
| `claude -p "query"` | Run one-off query, then exit                           | `claude -p "explain this function"` |
| `claude -c`         | Continue most recent conversation in current directory | `claude -c`                         |
| `claude -r`         | Resume a previous conversation                         | `claude -r`                         |
| `/clear`            | Clear conversation history                             | `/clear`                            |
| `/help`             | Show available commands                                | `/help`                             |
| `exit` or Ctrl+D    | Exit Claude Code                                       | `exit`                              |

See the [CLI reference](/docs/en/cli-reference) for a complete list of commands.

### Pro tips for beginners

For more, see [best practices](/docs/en/best-practices) and [common workflows](/docs/en/common-workflows) .

Be specific with your requests

Instead of: "fix the bug" Try: "fix the login bug where users see a blank screen after entering wrong credentials"

Use step-by-step instructions

Break complex tasks into steps:

```
1. create a new database table for user profiles
2. create an API endpoint to get and update user profiles
3. build a webpage that allows users to see and edit their information
```

Let Claude explore first

Before making changes, let Claude understand your code:

```
analyze the database schema
```

```
build a dashboard showing products that are most frequently returned by our UK customers
```

Save time with shortcuts

- Press `?` to see all available keyboard shortcuts
- Use Tab for command completion
- Press ↑ for command history
- Type `/` to see all commands and skills

### What's next?

Now that you've learned the basics, explore more advanced features:

### How Claude Code works

Understand the agentic loop, built-in tools, and how Claude Code interacts with your project

### Best practices

Get better results with effective prompting and project setup

### Common workflows

Step-by-step guides for common tasks

### Extend Claude Code

Customize with CLAUDE.md, skills, hooks, MCP, and more

### Getting help

- **In Claude Code** : Type `/help` or ask "how do I..."
- **Documentation** : You're here! Browse other guides
- **Community** : Join our [Discord](https://www.anthropic.com/discord) for tips and support

Was this page helpful?

Yes

No

[Overview](/docs/en/overview) [Changelog](/docs/en/changelog)

⌘ I


---

# Core Concepts


### Explore the .claude directory


Where Claude Code reads CLAUDE.md, settings.json, hooks, skills, commands, subagents, rules, and auto memory. Explore the .claude directory in your project and ~/.claude in your home directory.


Claude Code reads instructions, settings, skills, subagents, and memory from your project directory and from `~/.claude` in your home directory. Commit project files to git to share them with your team; files in `~/.claude` are personal configuration that applies across all your projects. If you set [`CLAUDE_CONFIG_DIR`](/docs/en/env-vars) , every `~/.claude` path on this page lives under that directory instead. Most users only edit `CLAUDE.md` and `settings.json` . The rest of the directory is optional: add skills, rules, or subagents as you need them. This page is an interactive explorer: click files in the tree to see what each one does, when it loads, and an example. For a quick reference, see the [file reference table](#file-reference) below.

### What's not shown

The explorer covers files you author and edit. A few related files live elsewhere:

| File                    | Location                   | Purpose                                                                                                                                                                                                                                                                  |
|-------------------------|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `managed-settings.json` | System-level, varies by OS | Enterprise-enforced settings that you can't override. See [server-managed settings](/docs/en/server-managed-settings) .                                                                                                                                                  |
| `CLAUDE.local.md`       | Project root               | Your private preferences for this project, loaded alongside CLAUDE.md. Create it manually and add it to `.gitignore` .                                                                                                                                                   |
| Installed plugins       | `~/.claude/plugins/`       | Cloned marketplaces, installed plugin versions, and per-plugin data, managed by `claude plugin` commands. Orphaned versions are deleted 7 days after a plugin update or uninstall. See [plugin caching](/docs/en/plugins-reference#plugin-caching-and-file-resolution) . |

`~/.claude` also holds data Claude Code writes as you work: transcripts, prompt history, file snapshots, caches, and logs. See [application data](#application-data) below.

### File reference

This table lists every file the explorer covers. Project-scope files live in your repo under `.claude/` (or at the root for `CLAUDE.md` , `.mcp.json` , and `.worktreeinclude` ). Global-scope files live in `~/.claude/` and apply across all projects.

Several things can override what you put in these files:

- [Managed settings](/docs/en/server-managed-settings) deployed by your organization take precedence over everything
- CLI flags like `--permission-mode` or `--settings` override `settings.json` for that session
- Some environment variables take precedence over their equivalent setting, but this varies: check the [environment variables reference](/docs/en/env-vars) for each one

See [settings precedence](/docs/en/settings#settings-precedence) for the full order.

Click a filename to open that node in the explorer above.

| File                                                | Scope              | Commit   | What it does                                          | Reference                                                                 |
|-----------------------------------------------------|--------------------|----------|-------------------------------------------------------|---------------------------------------------------------------------------|
| [`CLAUDE.md`](#ce-claude-md)                        | Project and global | ✓        | Instructions loaded every session                     | [Memory](/docs/en/memory)                                                 |
| [`rules/*.md`](#ce-rules)                           | Project and global | ✓        | Topic-scoped instructions, optionally path-gated      | [Rules](/docs/en/memory#organize-rules-with-claude/rules)                 |
| [`settings.json`](#ce-settings-json)                | Project and global | ✓        | Permissions, hooks, env vars, model defaults          | [Settings](/docs/en/settings)                                             |
| [`settings.local.json`](#ce-settings-local-json)    | Project only       |          | Your personal overrides, auto-gitignored              | [Settings scopes](/docs/en/settings#settings-files)                       |
| [`.mcp.json`](#ce-mcp-json)                         | Project only       | ✓        | Team-shared MCP servers                               | [MCP scopes](/docs/en/mcp#mcp-installation-scopes)                        |
| [`.worktreeinclude`](#ce-worktreeinclude)           | Project only       | ✓        | Gitignored files to copy into new worktrees           | [Worktrees](/docs/en/common-workflows#copy-gitignored-files-to-worktrees) |
| [`skills/<name>/SKILL.md`](#ce-skills)              | Project and global | ✓        | Reusable prompts invoked with `/name` or auto-invoked | [Skills](/docs/en/skills)                                                 |
| [`commands/*.md`](#ce-commands)                     | Project and global | ✓        | Single-file prompts; same mechanism as skills         | [Skills](/docs/en/skills)                                                 |
| [`output-styles/*.md`](#ce-output-styles)           | Project and global | ✓        | Custom system-prompt sections                         | [Output styles](/docs/en/output-styles)                                   |
| [`agents/*.md`](#ce-agents)                         | Project and global | ✓        | Subagent definitions with their own prompt and tools  | [Subagents](/docs/en/sub-agents)                                          |
| [`agent-memory/<name>/`](#ce-agent-memory)          | Project and global | ✓        | Persistent memory for subagents                       | [Persistent memory](/docs/en/sub-agents#enable-persistent-memory)         |
| [`~/.claude.json`](#ce-claude-json)                 | Global only        |          | App state, OAuth, UI toggles, personal MCP servers    | [Global config](/docs/en/settings#global-config-settings)                 |
| [`projects/<project>/memory/`](#ce-global-projects) | Global only        |          | Auto memory: Claude's notes to itself across sessions | [Auto memory](/docs/en/memory#auto-memory)                                |
| [`keybindings.json`](#ce-keybindings)               | Global only        |          | Custom keyboard shortcuts                             | [Keybindings](/docs/en/keybindings)                                       |

### Check what loaded

The explorer shows what files can exist. To see what actually loaded in your current session, use these commands:

| Command        | Shows                                                                                 |
|----------------|---------------------------------------------------------------------------------------|
| `/context`     | Token usage by category: system prompt, memory files, skills, MCP tools, and messages |
| `/memory`      | Which CLAUDE.md and rules files loaded, plus auto-memory entries                      |
| `/agents`      | Configured subagents and their settings                                               |
| `/hooks`       | Active hook configurations                                                            |
| `/mcp`         | Connected MCP servers and their status                                                |
| `/skills`      | Available skills from project, user, and plugin sources                               |
| `/permissions` | Current allow and deny rules                                                          |
| `/doctor`      | Installation and configuration diagnostics                                            |

Run `/context` first for the overview, then the specific command for the area you want to investigate.

### Application data

Beyond the config you author, `~/.claude` holds data Claude Code writes during sessions. These files are plaintext. Anything that passes through a tool lands in a transcript on disk: file contents, command output, pasted text.

#### Cleaned up automatically

Files in the paths below are deleted on startup once they're older than [`cleanupPeriodDays`](/docs/en/settings#available-settings) . The default is 30 days.

| Path under `~/.claude/`                      | Contents                                                                                                |
|----------------------------------------------|---------------------------------------------------------------------------------------------------------|
| `projects/<project>/<session>.jsonl`         | Full conversation transcript: every message, tool call, and tool result                                 |
| `projects/<project>/<session>/tool-results/` | Large tool outputs spilled to separate files                                                            |
| `file-history/<session>/`                    | Pre-edit snapshots of files Claude changed, used for [checkpoint restore](/docs/en/checkpointing)       |
| `plans/`                                     | Plan files written during [plan mode](/docs/en/permission-modes#analyze-before-you-edit-with-plan-mode) |
| `debug/`                                     | Per-session debug logs, written only when you start with `--debug` or run `/debug`                      |
| `paste-cache/` , `image-cache/`              | Contents of large pastes and attached images                                                            |
| `session-env/`                               | Per-session environment metadata                                                                        |

#### Kept until you delete them

The following paths are not covered by automatic cleanup and persist indefinitely.

| Path under `~/.claude/`   | Contents                                                                              |
|---------------------------|---------------------------------------------------------------------------------------|
| `history.jsonl`           | Every prompt you've typed, with timestamp and project path. Used for up-arrow recall. |
| `stats-cache.json`        | Aggregated token and cost counts shown by `/cost`                                     |
| `backups/`                | Timestamped copies of `~/.claude.json` taken before config migrations                 |
| `todos/`                  | Legacy per-session task lists. No longer written by current versions; safe to delete. |

`shell-snapshots/` holds runtime files removed when the session exits cleanly. Other small cache and lock files appear depending on which features you use and are safe to delete.

#### Plaintext storage

Transcripts and history are not encrypted at rest. OS file permissions are the only protection. If a tool reads a `.env` file or a command prints a credential, that value is written to `projects/<project>/<session>.jsonl` . To reduce exposure:

- Lower `cleanupPeriodDays` to shorten how long transcripts are kept
- In non-interactive mode, pass `--no-session-persistence` alongside `-p` to skip writing transcripts entirely. In the Agent SDK, set `persistSession: false` . There is no interactive-mode equivalent.
- Use [permission rules](/docs/en/permissions) to deny reads of credential files

#### Clear local data

You can delete any of the application-data paths above at any time. New sessions are unaffected. The table below shows what you lose for past sessions.

| Delete                                                                                                                   | You lose                                                        |
|--------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| `~/.claude/projects/`                                                                                                    | Resume, continue, and rewind for past sessions                  |
| `~/.claude/history.jsonl`                                                                                                | Up-arrow prompt recall                                          |
| `~/.claude/file-history/`                                                                                                | Checkpoint restore for past sessions                            |
| `~/.claude/stats-cache.json`                                                                                             | Historical totals shown by `/cost`                              |
| `~/.claude/backups/`                                                                                                     | Rollback copies of `~/.claude.json` from past config migrations |
| `~/.claude/debug/` , `~/.claude/plans/` , `~/.claude/paste-cache/` , `~/.claude/image-cache/` , `~/.claude/session-env/` | Nothing user-facing                                             |
| `~/.claude/todos/`                                                                                                       | Nothing. Legacy directory not written by current versions.      |

Don't delete `~/.claude.json` , `~/.claude/settings.json` , or `~/.claude/plugins/` : those hold your auth, preferences, and installed plugins.

### Related resources

- [Manage Claude's memory](/docs/en/memory) : write and organize CLAUDE.md, rules, and auto memory
- [Configure settings](/docs/en/settings) : set permissions, hooks, environment variables, and model defaults
- [Create skills](/docs/en/skills) : build reusable prompts and workflows
- [Configure subagents](/docs/en/sub-agents) : define specialized agents with their own context

Was this page helpful?

Yes

No

[Extend Claude Code](/docs/en/features-overview) [Explore the context window](/docs/en/context-window)

⌘ I


### Explore the context window


An interactive simulation of how Claude Code's context window fills during a session. See what loads automatically, what each file read costs, and when rules and hooks fire.


Claude Code's context window holds everything Claude knows about your session: your instructions, the files it reads, its own responses, and content that never appears in your terminal. The timeline below walks through what loads and when. See [the written breakdown](#what-the-timeline-shows) for the same content as a list.

### What the timeline shows

The session walks through a realistic flow with representative token counts:

- **Before you type anything** : CLAUDE.md, auto memory, MCP tool names, and skill descriptions all load into context. Your own setup may add more here, like an [output style](/docs/en/output-styles) or text from [`--append-system-prompt`](/docs/en/cli-reference) , which both go into the system prompt the same way.
- **As Claude works** : each file read adds to context, [path-scoped rules](/docs/en/memory#path-specific-rules) load automatically alongside matching files, and a [PostToolUse hook](/docs/en/hooks-guide) fires after each edit.
- **The follow-up prompt** : a [subagent](/docs/en/sub-agents) handles the research in its own separate context window, so the large file reads stay out of yours. Only the summary and a small metadata trailer come back.
- **At the end** : `/compact` replaces the conversation with a structured summary. Most startup content reloads automatically; the table below shows what happens to each mechanism.

### What survives compaction

When a long session compacts, Claude Code summarizes the conversation history to fit the context window. What happens to your instructions depends on how they were loaded:

| Mechanism                                 | After compaction                                                                            |
|-------------------------------------------|---------------------------------------------------------------------------------------------|
| System prompt and output style            | Unchanged; not part of message history                                                      |
| Project-root CLAUDE.md and unscoped rules | Re-injected from disk                                                                       |
| Auto memory                               | Re-injected from disk                                                                       |
| Rules with `paths:` frontmatter           | Lost until a matching file is read again                                                    |
| Nested CLAUDE.md in subdirectories        | Lost until a file in that subdirectory is read again                                        |
| Invoked skill bodies                      | Re-injected, capped at 5,000 tokens per skill and 25,000 tokens total; oldest dropped first |
| Hooks                                     | Not applicable; hooks run as code, not context                                              |

Path-scoped rules and nested CLAUDE.md files load into message history when their trigger file is read, so compaction summarizes them away with everything else. They reload the next time Claude reads a matching file. If a rule must persist across compaction, drop the `paths:` frontmatter or move it to the project-root CLAUDE.md. Skill bodies are re-injected after compaction, but large skills are truncated to fit the per-skill cap, and the oldest invoked skills are dropped once the total budget is exceeded. Truncation keeps the start of the file, so put the most important instructions near the top of `SKILL.md` .

### Check your own session

The visualization uses representative numbers. To see your actual context usage at any point, run `/context` for a live breakdown by category with optimization suggestions. Run `/memory` to check which CLAUDE.md and auto memory files loaded at startup.

### Related resources

For deeper coverage of the features shown in the timeline, see these pages:

- [Extend Claude Code](/docs/en/features-overview) : when to use CLAUDE.md vs skills vs rules vs hooks vs MCP
- [Store instructions and memories](/docs/en/memory) : CLAUDE.md hierarchy and auto memory
- [Subagents](/docs/en/sub-agents) : delegate research to a separate context window
- [Best practices](/docs/en/best-practices) : managing context as your primary constraint
- [Reduce token usage](/docs/en/costs#reduce-token-usage) : strategies for keeping context usage low

Was this page helpful?

Yes

No

[Explore the .claude directory](/docs/en/claude-directory) [Store instructions and memories](/docs/en/memory)

⌘ I


### How Claude remembers your project


Give Claude persistent instructions with CLAUDE.md files, and let Claude accumulate learnings automatically with auto memory.


Each Claude Code session begins with a fresh context window. Two mechanisms carry knowledge across sessions:

- **CLAUDE.md files** : instructions you write to give Claude persistent context
- **Auto memory** : notes Claude writes itself based on your corrections and preferences

This page covers how to:

- [Write and organize CLAUDE.md files](#claude-md-files)
- [Scope rules to specific file types](#organize-rules-with-claude/rules) with `.claude/rules/`
- [Configure auto memory](#auto-memory) so Claude takes notes automatically
- [Troubleshoot](#troubleshoot-memory-issues) when instructions aren't being followed

### CLAUDE.md vs auto memory

Claude Code has two complementary memory systems. Both are loaded at the start of every conversation. Claude treats them as context, not enforced configuration. The more specific and concise your instructions, the more consistently Claude follows them.

|                      | CLAUDE.md files                                   | Auto memory                                                      |
|----------------------|---------------------------------------------------|------------------------------------------------------------------|
| **Who writes it**    | You                                               | Claude                                                           |
| **What it contains** | Instructions and rules                            | Learnings and patterns                                           |
| **Scope**            | Project, user, or org                             | Per working tree                                                 |
| **Loaded into**      | Every session                                     | Every session (first 200 lines or 25KB)                          |
| **Use for**          | Coding standards, workflows, project architecture | Build commands, debugging insights, preferences Claude discovers |

Use CLAUDE.md files when you want to guide Claude's behavior. Auto memory lets Claude learn from your corrections without manual effort. Subagents can also maintain their own auto memory. See [subagent configuration](/docs/en/sub-agents#enable-persistent-memory) for details.

### CLAUDE.md files

CLAUDE.md files are markdown files that give Claude persistent instructions for a project, your personal workflow, or your entire organization. You write these files in plain text; Claude reads them at the start of every session.

#### When to add to CLAUDE.md

Treat CLAUDE.md as the place you write down what you'd otherwise re-explain. Add to it when:

- Claude makes the same mistake a second time
- A code review catches something Claude should have known about this codebase
- You type the same correction or clarification into chat that you typed last session
- A new teammate would need the same context to be productive

Keep it to facts Claude should hold in every session: build commands, conventions, project layout, "always do X" rules. If an entry is a multi-step procedure or only matters for one part of the codebase, move it to a [skill](/docs/en/skills) or a [path-scoped rule](#organize-rules-with-claude/rules) instead. The [extension overview](/docs/en/features-overview#build-your-setup-over-time) covers when to use each mechanism.

#### Choose where to put CLAUDE.md files

CLAUDE.md files can live in several locations, each with a different scope. More specific locations take precedence over broader ones.

| Scope                    | Location                                                                                                                                                        | Purpose                                                    | Use case examples                                                    | Shared with                     |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|----------------------------------------------------------------------|---------------------------------|
| **Managed policy**       | • macOS: `/Library/Application Support/ClaudeCode/CLAUDE.md`  • Linux and WSL: `/etc/claude-code/CLAUDE.md`  • Windows: `C:\Program Files\ClaudeCode\CLAUDE.md` | Organization-wide instructions managed by IT/DevOps        | Company coding standards, security policies, compliance requirements | All users in organization       |
| **Project instructions** | `./CLAUDE.md` or `./.claude/CLAUDE.md`                                                                                                                          | Team-shared instructions for the project                   | Project architecture, coding standards, common workflows             | Team members via source control |
| **User instructions**    | `~/.claude/CLAUDE.md`                                                                                                                                           | Personal preferences for all projects                      | Code styling preferences, personal tooling shortcuts                 | Just you (all projects)         |
| **Local instructions**   | `./CLAUDE.local.md`                                                                                                                                             | Personal project-specific preferences; add to `.gitignore` | Your sandbox URLs, preferred test data                               | Just you (current project)      |

CLAUDE.md and CLAUDE.local.md files in the directory hierarchy above the working directory are loaded in full at launch. Files in subdirectories load on demand when Claude reads files in those directories. See [How CLAUDE.md files load](#how-claude-md-files-load) for the full resolution order. For large projects, you can break instructions into topic-specific files using [project rules](#organize-rules-with-claude/rules) . Rules let you scope instructions to specific file types or subdirectories.

#### Set up a project CLAUDE.md

A project CLAUDE.md can be stored in either `./CLAUDE.md` or `./.claude/CLAUDE.md` . Create this file and add instructions that apply to anyone working on the project: build and test commands, coding standards, architectural decisions, naming conventions, and common workflows. These instructions are shared with your team through version control, so focus on project-level standards rather than personal preferences.

Run `/init` to generate a starting CLAUDE.md automatically. Claude analyzes your codebase and creates a file with build commands, test instructions, and project conventions it discovers. If a CLAUDE.md already exists, `/init` suggests improvements rather than overwriting it. Refine from there with instructions Claude wouldn't discover on its own. Set `CLAUDE_CODE_NEW_INIT=1` to enable an interactive multi-phase flow. `/init` asks which artifacts to set up: CLAUDE.md files, skills, and hooks. It then explores your codebase with a subagent, fills in gaps via follow-up questions, and presents a reviewable proposal before writing any files.

#### Write effective instructions

CLAUDE.md files are loaded into the context window at the start of every session, consuming tokens alongside your conversation. The [context window visualization](/docs/en/context-window) shows where CLAUDE.md loads relative to the rest of the startup context. Because they're context rather than enforced configuration, how you write instructions affects how reliably Claude follows them. Specific, concise, well-structured instructions work best. **Size** : target under 200 lines per CLAUDE.md file. Longer files consume more context and reduce adherence. If your instructions are growing large, split them using [imports](#import-additional-files) or [`.claude/rules/`](#organize-rules-with-claude/rules) files. **Structure** : use markdown headers and bullets to group related instructions. Claude scans structure the same way readers do: organized sections are easier to follow than dense paragraphs. **Specificity** : write instructions that are concrete enough to verify. For example:

- "Use 2-space indentation" instead of "Format code properly"
- "Run `npm test` before committing" instead of "Test your changes"
- "API handlers live in `src/api/handlers/` " instead of "Keep files organized"

**Consistency** : if two rules contradict each other, Claude may pick one arbitrarily. Review your CLAUDE.md files, nested CLAUDE.md files in subdirectories, and [`.claude/rules/`](#organize-rules-with-claude/rules) periodically to remove outdated or conflicting instructions. In monorepos, use [`claudeMdExcludes`](#exclude-specific-claude-md-files) to skip CLAUDE.md files from other teams that aren't relevant to your work.

#### Import additional files

CLAUDE.md files can import additional files using `@path/to/import` syntax. Imported files are expanded and loaded into context at launch alongside the CLAUDE.md that references them. Both relative and absolute paths are allowed. Relative paths resolve relative to the file containing the import, not the working directory. Imported files can recursively import other files, with a maximum depth of five hops. To pull in a README, package.json, and a workflow guide, reference them with `@` syntax anywhere in your CLAUDE.md:

```
See @README for project overview and @package.json for available npm commands for this project.

### Additional Instructions
- git workflow @docs/git-instructions.md
```

For private per-project preferences that shouldn't be checked into version control, create a `CLAUDE.local.md` at the project root. It loads alongside `CLAUDE.md` and is treated the same way. Add `CLAUDE.local.md` to your `.gitignore` so it isn't committed; running `/init` and choosing the personal option does this for you. If you work across multiple git worktrees of the same repository, a gitignored `CLAUDE.local.md` only exists in the worktree where you created it. To share personal instructions across worktrees, import a file from your home directory instead:

```
### Individual Preferences
- @~/.claude/my-project-instructions.md
```

The first time Claude Code encounters external imports in a project, it shows an approval dialog listing the files. If you decline, the imports stay disabled and the dialog does not appear again.

For a more structured approach to organizing instructions, see [`.claude/rules/`](#organize-rules-with-claude/rules) .

#### AGENTS.md

Claude Code reads `CLAUDE.md` , not `AGENTS.md` . If your repository already uses `AGENTS.md` for other coding agents, create a `CLAUDE.md` that imports it so both tools read the same instructions without duplicating them. You can also add Claude-specific instructions below the import. Claude loads the imported file at session start, then appends the rest:

CLAUDE.md

```
@AGENTS.md

### Claude Code

Use plan mode for changes under `src/billing/` .
```

#### How CLAUDE.md files load

Claude Code reads CLAUDE.md files by walking up the directory tree from your current working directory, checking each directory along the way for `CLAUDE.md` and `CLAUDE.local.md` files. This means if you run Claude Code in `foo/bar/` , it loads instructions from `foo/bar/CLAUDE.md` , `foo/CLAUDE.md` , and any `CLAUDE.local.md` files alongside them. All discovered files are concatenated into context rather than overriding each other. Within each directory, `CLAUDE.local.md` is appended after `CLAUDE.md` , so when instructions conflict, your personal notes are the last thing Claude reads at that level. Claude also discovers `CLAUDE.md` and `CLAUDE.local.md` files in subdirectories under your current working directory. Instead of loading them at launch, they are included when Claude reads files in those subdirectories. If you work in a large monorepo where other teams' CLAUDE.md files get picked up, use [`claudeMdExcludes`](#exclude-specific-claude-md-files) to skip them. Block-level HTML comments ( `` ) in CLAUDE.md files are stripped before the content is injected into Claude's context. Use them to leave notes for human maintainers without spending context tokens on them. Comments inside code blocks are preserved. When you open a CLAUDE.md file directly with the Read tool, comments remain visible.

##### Load from additional directories

The `--add-dir` flag gives Claude access to additional directories outside your main working directory. By default, CLAUDE.md files from these directories are not loaded. To also load CLAUDE.md files from additional directories, including `CLAUDE.md` , `.claude/CLAUDE.md` , and `.claude/rules/*.md` , set the `CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD` environment variable:

```
CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD = 1 claude --add-dir ../shared-config
```

`CLAUDE.local.md` files in additional directories are not loaded.

#### Organize rules with .claude/rules/

For larger projects, you can organize instructions into multiple files using the `.claude/rules/` directory. This keeps instructions modular and easier for teams to maintain. Rules can also be [scoped to specific file paths](#path-specific-rules) , so they only load into context when Claude works with matching files, reducing noise and saving context space.

Rules load into context every session or when matching files are opened. For task-specific instructions that don't need to be in context all the time, use [skills](/docs/en/skills) instead, which only load when you invoke them or when Claude determines they're relevant to your prompt.

##### Set up rules

Place markdown files in your project's `.claude/rules/` directory. Each file should cover one topic, with a descriptive filename like `testing.md` or `api-design.md` . All `.md` files are discovered recursively, so you can organize rules into subdirectories like `frontend/` or `backend/` :

```
your-project/
├── .claude/
│   ├── CLAUDE.md           # Main project instructions
│   └── rules/
│       ├── code-style.md   # Code style guidelines
│       ├── testing.md      # Testing conventions
│       └── security.md     # Security requirements
```

Rules without [`paths`](#path-specific-rules) [frontmatter](#path-specific-rules) are loaded at launch with the same priority as `.claude/CLAUDE.md` .

##### Path-specific rules

Rules can be scoped to specific files using YAML frontmatter with the `paths` field. These conditional rules only apply when Claude is working with files matching the specified patterns.

```
---
paths :
- "src/api/**/*.ts"

### API Development Rules

- All API endpoints must include input validation
- Use the standard error response format
- Include OpenAPI documentation comments
```

Rules without a `paths` field are loaded unconditionally and apply to all files. Path-scoped rules trigger when Claude reads files matching the pattern, not on every tool use. Use glob patterns in the `paths` field to match files by extension, directory, or any combination:

| Pattern                | Matches                                  |
|------------------------|------------------------------------------|
| `**/*.ts`              | All TypeScript files in any directory    |
| `src/**/*`             | All files under `src/` directory         |
| `*.md`                 | Markdown files in the project root       |
| `src/components/*.tsx` | React components in a specific directory |

You can specify multiple patterns and use brace expansion to match multiple extensions in one pattern:

```
---
paths :
- "src/**/*.{ts,tsx}"
- "lib/**/*.ts"
- "tests/**/*.test.ts"
---
```

##### Share rules across projects with symlinks

The `.claude/rules/` directory supports symlinks, so you can maintain a shared set of rules and link them into multiple projects. Symlinks are resolved and loaded normally, and circular symlinks are detected and handled gracefully. This example links both a shared directory and an individual file:

```
ln -s ~/shared-claude-rules .claude/rules/shared
ln -s ~/company-standards/security.md .claude/rules/security.md
```

##### User-level rules

Personal rules in `~/.claude/rules/` apply to every project on your machine. Use them for preferences that aren't project-specific:

```
~/.claude/rules/
├── preferences.md    # Your personal coding preferences
└── workflows.md      # Your preferred workflows
```

User-level rules are loaded before project rules, giving project rules higher priority.

#### Manage CLAUDE.md for large teams

For organizations deploying Claude Code across teams, you can centralize instructions and control which CLAUDE.md files are loaded.

##### Deploy organization-wide CLAUDE.md

Organizations can deploy a centrally managed CLAUDE.md that applies to all users on a machine. This file cannot be excluded by individual settings.

1

Create the file at the managed policy location

- macOS: `/Library/Application Support/ClaudeCode/CLAUDE.md`
- Linux and WSL: `/etc/claude-code/CLAUDE.md`
- Windows: `C:\Program Files\ClaudeCode\CLAUDE.md`

2

Deploy with your configuration management system

Use MDM, Group Policy, Ansible, or similar tools to distribute the file across developer machines. See [managed settings](/docs/en/permissions#managed-settings) for other organization-wide configuration options.

A managed CLAUDE.md and [managed settings](/docs/en/settings#settings-files) serve different purposes. Use settings for technical enforcement and CLAUDE.md for behavioral guidance:

| Concern                                        | Configure in                                               |
|------------------------------------------------|------------------------------------------------------------|
| Block specific tools, commands, or file paths  | Managed settings: `permissions.deny`                       |
| Enforce sandbox isolation                      | Managed settings: `sandbox.enabled`                        |
| Environment variables and API provider routing | Managed settings: `env`                                    |
| Authentication method and organization lock    | Managed settings: `forceLoginMethod` , `forceLoginOrgUUID` |
| Code style and quality guidelines              | Managed CLAUDE.md                                          |
| Data handling and compliance reminders         | Managed CLAUDE.md                                          |
| Behavioral instructions for Claude             | Managed CLAUDE.md                                          |

Settings rules are enforced by the client regardless of what Claude decides to do. CLAUDE.md instructions shape Claude's behavior but are not a hard enforcement layer.

##### Exclude specific CLAUDE.md files

In large monorepos, ancestor CLAUDE.md files may contain instructions that aren't relevant to your work. The `claudeMdExcludes` setting lets you skip specific files by path or glob pattern. This example excludes a top-level CLAUDE.md and a rules directory from a parent folder. Add it to `.claude/settings.local.json` so the exclusion stays local to your machine:

```
{
"claudeMdExcludes" : [
"**/monorepo/CLAUDE.md" ,
"/home/user/monorepo/other-team/.claude/rules/**"
]
}
```

Patterns are matched against absolute file paths using glob syntax. You can configure `claudeMdExcludes` at any [settings layer](/docs/en/settings#settings-files) : user, project, local, or managed policy. Arrays merge across layers. Managed policy CLAUDE.md files cannot be excluded. This ensures organization-wide instructions always apply regardless of individual settings.

### Auto memory

Auto memory lets Claude accumulate knowledge across sessions without you writing anything. Claude saves notes for itself as it works: build commands, debugging insights, architecture notes, code style preferences, and workflow habits. Claude doesn't save something every session. It decides what's worth remembering based on whether the information would be useful in a future conversation.

Auto memory requires Claude Code v2.1.59 or later. Check your version with `claude --version` .

#### Enable or disable auto memory

Auto memory is on by default. To toggle it, open `/memory` in a session and use the auto memory toggle, or set `autoMemoryEnabled` in your project settings:

```
{
"autoMemoryEnabled" : false
}
```

To disable auto memory via environment variable, set `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1` .

#### Storage location

Each project gets its own memory directory at `~/.claude/projects/<project>/memory/` . The `<project>` path is derived from the git repository, so all worktrees and subdirectories within the same repo share one auto memory directory. Outside a git repo, the project root is used instead. To store auto memory in a different location, set `autoMemoryDirectory` in your user or local settings:

```
{
"autoMemoryDirectory" : "~/my-custom-memory-dir"
}
```

This setting is accepted from policy, local, and user settings. It is not accepted from project settings ( `.claude/settings.json` ) to prevent a shared project from redirecting auto memory writes to sensitive locations. The directory contains a `MEMORY.md` entrypoint and optional topic files:

```
~/.claude/projects/<project>/memory/
├── MEMORY.md          # Concise index, loaded into every session
├── debugging.md       # Detailed notes on debugging patterns
├── api-conventions.md # API design decisions
└── ...                # Any other topic files Claude creates
```

`MEMORY.md` acts as an index of the memory directory. Claude reads and writes files in this directory throughout your session, using `MEMORY.md` to keep track of what's stored where. Auto memory is machine-local. All worktrees and subdirectories within the same git repository share one auto memory directory. Files are not shared across machines or cloud environments.

#### How it works

The first 200 lines of `MEMORY.md` , or the first 25KB, whichever comes first, are loaded at the start of every conversation. Content beyond that threshold is not loaded at session start. Claude keeps `MEMORY.md` concise by moving detailed notes into separate topic files. This limit applies only to `MEMORY.md` . CLAUDE.md files are loaded in full regardless of length, though shorter files produce better adherence. Topic files like `debugging.md` or `patterns.md` are not loaded at startup. Claude reads them on demand using its standard file tools when it needs the information. Claude reads and writes memory files during your session. When you see "Writing memory" or "Recalled memory" in the Claude Code interface, Claude is actively updating or reading from `~/.claude/projects/<project>/memory/` .

#### Audit and edit your memory

Auto memory files are plain markdown you can edit or delete at any time. Run [`/memory`](#view-and-edit-with-memory) to browse and open memory files from within a session.

### View and edit with /memory

The `/memory` command lists all CLAUDE.md, CLAUDE.local.md, and rules files loaded in your current session, lets you toggle auto memory on or off, and provides a link to open the auto memory folder. Select any file to open it in your editor. When you ask Claude to remember something, like "always use pnpm, not npm" or "remember that the API tests require a local Redis instance," Claude saves it to auto memory. To add instructions to CLAUDE.md instead, ask Claude directly, like "add this to CLAUDE.md," or edit the file yourself via `/memory` .

### Troubleshoot memory issues

These are the most common issues with CLAUDE.md and auto memory, along with steps to debug them.

#### Claude isn't following my CLAUDE.md

CLAUDE.md content is delivered as a user message after the system prompt, not as part of the system prompt itself. Claude reads it and tries to follow it, but there's no guarantee of strict compliance, especially for vague or conflicting instructions. To debug:

- Run `/memory` to verify your CLAUDE.md and CLAUDE.local.md files are being loaded. If a file isn't listed, Claude can't see it.
- Check that the relevant CLAUDE.md is in a location that gets loaded for your session (see [Choose where to put CLAUDE.md files](#choose-where-to-put-claude-md-files) ).
- Make instructions more specific. "Use 2-space indentation" works better than "format code nicely."
- Look for conflicting instructions across CLAUDE.md files. If two files give different guidance for the same behavior, Claude may pick one arbitrarily.

For instructions you want at the system prompt level, use [`--append-system-prompt`](/docs/en/cli-reference#system-prompt-flags) . This must be passed every invocation, so it's better suited to scripts and automation than interactive use.

Use the [`InstructionsLoaded`](/docs/en/hooks#instructionsloaded) [hook](/docs/en/hooks#instructionsloaded) to log exactly which instruction files are loaded, when they load, and why. This is useful for debugging path-specific rules or lazy-loaded files in subdirectories.

#### I don't know what auto memory saved

Run `/memory` and select the auto memory folder to browse what Claude has saved. Everything is plain markdown you can read, edit, or delete.

#### My CLAUDE.md is too large

Files over 200 lines consume more context and may reduce adherence. Move detailed content into separate files referenced with `@path` imports (see [Import additional files](#import-additional-files) ), or split your instructions across `.claude/rules/` files.

#### Instructions seem lost after /compact

Project-root CLAUDE.md survives compaction: after `/compact` , Claude re-reads it from disk and re-injects it into the session. Nested CLAUDE.md files in subdirectories are not re-injected automatically; they reload the next time Claude reads a file in that subdirectory. If an instruction disappeared after compaction, it was either given only in conversation or lives in a nested CLAUDE.md that hasn't reloaded yet. Add conversation-only instructions to CLAUDE.md to make them persist. See [What survives compaction](/docs/en/context-window#what-survives-compaction) for the full breakdown. See [Write effective instructions](#write-effective-instructions) for guidance on size, structure, and specificity.

### Related resources

- [Skills](/docs/en/skills) : package repeatable workflows that load on demand
- [Settings](/docs/en/settings) : configure Claude Code behavior with settings files
- [Subagent memory](/docs/en/sub-agents#enable-persistent-memory) : let subagents maintain their own auto memory

Was this page helpful?

Yes

No

[Explore the context window](/docs/en/context-window) [Permission modes](/docs/en/permission-modes)

⌘ I


### Choose a permission mode


Control whether Claude asks before editing files or running commands. Cycle modes with Shift+Tab in the CLI or use the mode selector in VS Code, Desktop, and claude.ai.


When Claude wants to edit a file, run a shell command, or make a network request, it pauses and asks you to approve the action. Permission modes control how often that pause happens. The mode you pick shapes the flow of a session: default mode has you review each action as it comes, while looser modes let Claude work in longer uninterrupted stretches and report back when done. Pick more oversight for sensitive work, or fewer interruptions when you trust the direction.

### Available modes

Each mode makes a different tradeoff between convenience and oversight. The table below shows what Claude can do without a permission prompt in each mode.

| Mode                                                                | What runs without asking                                                                    | Best for                                |
|---------------------------------------------------------------------|---------------------------------------------------------------------------------------------|-----------------------------------------|
| `default`                                                           | Reads only                                                                                  | Getting started, sensitive work         |
| [`acceptEdits`](#auto-approve-file-edits-with-acceptedits-mode)     | Reads, file edits, and common filesystem commands ( `mkdir` , `touch` , `mv` , `cp` , etc.) | Iterating on code you're reviewing      |
| [`plan`](#analyze-before-you-edit-with-plan-mode)                   | Reads only                                                                                  | Exploring a codebase before changing it |
| [`auto`](#eliminate-prompts-with-auto-mode)                         | Everything, with background safety checks                                                   | Long tasks, reducing prompt fatigue     |
| [`dontAsk`](#allow-only-pre-approved-tools-with-dontask-mode)       | Only pre-approved tools                                                                     | Locked-down CI and scripts              |
| [`bypassPermissions`](#skip-all-checks-with-bypasspermissions-mode) | Everything except protected paths                                                           | Isolated containers and VMs only        |

Regardless of mode, writes to [protected paths](#protected-paths) are never auto-approved, guarding repository state and Claude's own configuration against accidental corruption. Modes set the baseline. Layer [permission rules](/docs/en/permissions#manage-permissions) on top to pre-approve or block specific tools in any mode except `bypassPermissions` , which skips the permission layer entirely.

### Switch permission modes

You can switch modes mid-session, at startup, or as a persistent default. The mode is set through these controls, not by asking Claude in chat. Select your interface below to see how to change it.

- CLI
- VS Code
- JetBrains
- Desktop
- Web and mobile

**During a session** : press `Shift+Tab` to cycle `default` → `acceptEdits` → `plan` . The current mode appears in the status bar. Not every mode is in the default cycle:

- `auto` : appears after you opt in with `--enable-auto-mode` or the persisted equivalent in settings
- `bypassPermissions` : appears after you start with `--permission-mode bypassPermissions` , `--dangerously-skip-permissions` , or `--allow-dangerously-skip-permissions` ; the `--allow-` variant adds the mode to the cycle without activating it
- `dontAsk` : never appears in the cycle; set it with `--permission-mode dontAsk`

Enabled optional modes slot in after `plan` , with `bypassPermissions` first and `auto` last. If you have both enabled, you will cycle through `bypassPermissions` on the way to `auto` . **At startup** : pass the mode as a flag.

```
claude --permission-mode plan
```

**As a default** : set `defaultMode` in [settings](/docs/en/settings#settings-files) .

```
{
"permissions" : {
"defaultMode" : "acceptEdits"
}
}
```

The same `--permission-mode` flag works with `-p` for [non-interactive runs](/docs/en/headless) .

**During a session** : click the mode indicator at the bottom of the prompt box. **As a default** : set `claudeCode.initialPermissionMode` in VS Code settings, or use the Claude Code extension settings panel. The mode indicator shows these labels, mapped to the mode each one applies:

| UI label           | Mode                |
|--------------------|---------------------|
| Ask before edits   | `default`           |
| Edit automatically | `acceptEdits`       |
| Plan mode          | `plan`              |
| Auto mode          | `auto`              |
| Bypass permissions | `bypassPermissions` |

Auto mode appears in the mode indicator after you enable **Allow dangerously skip permissions** in the extension settings, but it stays unavailable until your account meets every requirement listed in the [auto mode section](#eliminate-prompts-with-auto-mode) . The `claudeCode.initialPermissionMode` setting does not accept `auto` ; to start in auto mode by default, set `defaultMode` in your Claude Code [`settings.json`](/docs/en/settings#settings-files) instead. Bypass permissions also requires the **Allow dangerously skip permissions** toggle before it appears in the mode indicator. See the [VS Code guide](/docs/en/vs-code) for extension-specific details.

The JetBrains plugin runs Claude Code in the IDE terminal, so switching modes works the same as in the CLI: press `Shift+Tab` to cycle, or pass `--permission-mode` when launching.

Use the mode selector next to the send button. Auto and Bypass permissions appear only after you enable them in Desktop settings. See the [Desktop guide](/docs/en/desktop#choose-a-permission-mode) .

Use the mode dropdown next to the prompt box on [claude.ai/code](https://claude.ai/code) or in the mobile app. Permission prompts appear in claude.ai for approval. Which modes appear depends on where the session runs:

- **Cloud sessions** on [Claude Code on the web](/docs/en/claude-code-on-the-web) : Auto accept edits and Plan mode. Ask permissions, Auto, and Bypass permissions are not available.
- [**Remote Control**](/docs/en/remote-control) **sessions** on your local machine: Ask permissions, Auto accept edits, and Plan mode. Auto and Bypass permissions are not available.

For Remote Control, you can also set the starting mode when launching the host:

```
claude remote-control --permission-mode acceptEdits
```

### Auto-approve file edits with acceptEdits mode

`acceptEdits` mode lets Claude create and edit files in your working directory without prompting. The status bar shows `⏵⏵ accept edits on` while this mode is active. In addition to file edits, `acceptEdits` mode auto-approves common filesystem Bash commands: `mkdir` , `touch` , `rm` , `rmdir` , `mv` , `cp` , and `sed` . These commands are also auto-approved when prefixed with safe environment variables such as `LANG=C` or `NO_COLOR=1` , or process wrappers such as `timeout` , `nice` , or `nohup` . Like file edits, auto-approval applies only to paths inside your working directory or `additionalDirectories` . Paths outside that scope, writes to [protected paths](#protected-paths) , and all other Bash commands still prompt. Use `acceptEdits` when you want to review changes in your editor or via `git diff` after the fact rather than approving each edit inline. Press `Shift+Tab` once from default mode to enter it, or start with it directly:

```
claude --permission-mode acceptEdits
```

### Analyze before you edit with plan mode

Plan mode tells Claude to research and propose changes without making them. Claude reads files, runs shell commands to explore, and writes a plan, but does not edit your source. Permission prompts still apply the same as default mode. Enter plan mode by pressing `Shift+Tab` or prefixing a single prompt with `/plan` . You can also start in plan mode from the CLI:

```
claude --permission-mode plan
```

Press `Shift+Tab` again to leave plan mode without approving a plan. When the plan is ready, Claude presents it and asks how to proceed. From that prompt you can:

- Approve and start in auto mode
- Approve and accept edits
- Approve and review each edit manually
- Keep planning with feedback
- Refine with [Ultraplan](/docs/en/ultraplan) for browser-based review

Each approve option also offers to clear the planning context first.

### Eliminate prompts with auto mode

Auto mode requires Claude Code v2.1.83 or later.

Auto mode lets Claude execute without permission prompts. A separate classifier model reviews actions before they run, blocking anything that escalates beyond your request, targets unrecognized infrastructure, or appears driven by hostile content Claude read.

Auto mode is a research preview. It reduces prompts but does not guarantee safety. Use it for tasks where you trust the general direction, not as a replacement for review on sensitive operations.

Auto mode is available only when your account meets all of these requirements:

- **Plan** : Team, Enterprise, or API. Not available on Pro or Max.
- **Admin** : on Team and Enterprise, an admin must enable it in [Claude Code admin settings](https://claude.ai/admin-settings/claude-code) before users can turn it on. Admins can also lock it off by setting `permissions.disableAutoMode` to `"disable"` in [managed settings](/docs/en/permissions#managed-settings) .
- **Model** : Claude Sonnet 4.6 or Opus 4.6. Not available on Haiku or claude-3 models.
- **Provider** : Anthropic API only. Not available on Bedrock, Vertex, or Foundry.

If Claude Code reports auto mode as unavailable, one of these requirements is unmet; this is not a transient outage. Once enabled, start with the flag and `auto` joins the `Shift+Tab` cycle:

```
claude --enable-auto-mode
```

#### What the classifier blocks by default

The classifier trusts your working directory and your repo's configured remotes. Everything else is treated as external until you [configure trusted infrastructure](/docs/en/permissions#configure-the-auto-mode-classifier) . **Blocked by default** :

- Downloading and executing code, like `curl | bash`
- Sending sensitive data to external endpoints
- Production deploys and migrations
- Mass deletion on cloud storage
- Granting IAM or repo permissions
- Modifying shared infrastructure
- Irreversibly destroying files that existed before the session
- Force push, or pushing directly to `main`

**Allowed by default** :

- Local file operations in your working directory
- Installing dependencies declared in your lock files or manifests
- Reading `.env` and sending credentials to their matching API
- Read-only HTTP requests
- Pushing to the branch you started on or one Claude created
- Sandbox network access requests

Run `claude auto-mode defaults` to see the full rule lists. If routine actions get blocked, an administrator can add trusted repos, buckets, and services via the `autoMode.environment` setting: see [Configure the auto mode classifier](/docs/en/permissions#configure-the-auto-mode-classifier) .

#### When auto mode falls back

Each denied action shows a notification and appears in `/permissions` under the Recently denied tab, where you can press `r` to retry it with a manual approval. If the classifier blocks an action 3 times in a row or 20 times total, auto mode pauses and Claude Code resumes prompting. Approving the prompted action resumes auto mode. These thresholds are not configurable. Any allowed action resets the consecutive counter, while the total counter persists for the session and resets only when its own limit triggers a fallback. In [non-interactive mode](/docs/en/headless) with the `-p` flag, repeated blocks abort the session since there is no user to prompt. Repeated blocks usually mean the classifier is missing context about your infrastructure. Use `/feedback` to report false positives, or have an administrator [configure trusted infrastructure](/docs/en/permissions#configure-the-auto-mode-classifier) .

How the classifier evaluates actions

Each action goes through a fixed decision order. The first matching step wins:

1. Actions matching your [allow or deny rules](/docs/en/permissions#manage-permissions) resolve immediately
2. Read-only actions and file edits in your working directory are auto-approved, except writes to [protected paths](#protected-paths)
3. Everything else goes to the classifier
4. If the classifier blocks, Claude receives the reason and tries an alternative

On entering auto mode, broad allow rules that grant arbitrary code execution are dropped:

- Blanket `Bash(*)`
- Wildcarded interpreters like `Bash(python*)`
- Package-manager run commands
- `Agent` allow rules

Narrow rules like `Bash(npm test)` carry over. Dropped rules are restored when you leave auto mode. The classifier sees user messages, tool calls, and your CLAUDE.md content. Tool results are stripped, so hostile content in a file or web page cannot manipulate it directly. A separate server-side probe scans incoming tool results and flags suspicious content before Claude reads it. For more on how these layers work together, see the [auto mode announcement](https://claude.com/blog/auto-mode) and the [engineering deep dive](https://www.anthropic.com/engineering/claude-code-auto-mode) .

How auto mode handles subagents

The classifier checks [subagent](/docs/en/sub-agents) work at three points:

1. Before a subagent starts, the delegated task description is evaluated, so a dangerous-looking task is blocked at spawn time.
2. While the subagent runs, each of its actions goes through the classifier with the same rules as the parent session, and any `permissionMode` in the subagent's frontmatter is ignored.
3. When the subagent finishes, the classifier reviews its full action history; if that return check flags a concern, a security warning is prepended to the subagent's results.

Cost and latency

The classifier currently runs on Claude Sonnet 4.6 regardless of your main session model. Classifier calls count toward your token usage. Each check sends a portion of the transcript plus the pending action, adding a round-trip before execution. Reads and working-directory edits outside protected paths skip the classifier, so the overhead comes mainly from shell commands and network operations.

### Allow only pre-approved tools with dontAsk mode

`dontAsk` mode auto-denies every tool that is not explicitly allowed. Only actions matching your `permissions.allow` rules can execute; explicit `ask` rules are also denied rather than prompting. This makes the mode fully non-interactive for CI pipelines or restricted environments where you pre-define exactly what Claude may do. Set it at startup with the flag:

```
claude --permission-mode dontAsk
```

### Skip all checks with bypassPermissions mode

`bypassPermissions` mode disables permission prompts and safety checks so tool calls execute immediately. Writes to [protected paths](#protected-paths) are the only actions that still prompt. Only use this mode in isolated environments like containers, VMs, or devcontainers without internet access, where Claude Code cannot damage your host system. You cannot enter `bypassPermissions` from a session that was started without one of the enabling flags; restart with one to enable it:

```
claude --permission-mode bypassPermissions
```

The `--dangerously-skip-permissions` flag is equivalent.

`bypassPermissions` offers no protection against prompt injection or unintended actions. For background safety checks without prompts, use [auto mode](#eliminate-prompts-with-auto-mode) instead. Administrators can block this mode by setting `permissions.disableBypassPermissionsMode` to `"disable"` in [managed settings](/docs/en/permissions#managed-settings) .

### Protected paths

Writes to a small set of paths are never auto-approved, in every mode. This prevents accidental corruption of repository state and Claude's own configuration. In `default` , `acceptEdits` , `plan` , and `bypassPermissions` these writes prompt; in `auto` they route to the classifier; in `dontAsk` they are denied. Protected directories:

- .git
- .vscode
- .idea
- .husky
- `.claude` , except for `.claude/commands` , `.claude/agents` , `.claude/skills` , and `.claude/worktrees` where Claude routinely creates content

Protected files:

- `.gitconfig` , `.gitmodules`
- `.bashrc` , `.bash_profile` , `.zshrc` , `.zprofile` , `.profile`
- .ripgreprc
- `.mcp.json` , `.claude.json`

### See also

- [Permissions](/docs/en/permissions) : allow, ask, and deny rules; auto mode classifier configuration; managed policies
- [Hooks](/docs/en/hooks) : custom permission logic via `PreToolUse` and `PermissionRequest` hooks
- [Ultraplan](/docs/en/ultraplan) : run plan mode in a Claude Code on the web session with browser-based review
- [Security](/docs/en/security) : safeguards and best practices
- [Sandboxing](/docs/en/sandboxing) : filesystem and network isolation for Bash commands
- [Non-interactive mode](/docs/en/headless) : run Claude Code with the `-p` flag

Was this page helpful?

Yes

No

[Store instructions and memories](/docs/en/memory) [Common workflows](/docs/en/common-workflows)

⌘ I


---

# Configuration & CLI


### Claude Code settings


Configure Claude Code with global and project-level settings, and environment variables.


Claude Code offers a variety of settings to configure its behavior to meet your needs. You can configure Claude Code by running the `/config` command when using the interactive REPL, which opens a tabbed Settings interface where you can view status information and modify configuration options.

### Configuration scopes

Claude Code uses a **scope system** to determine where configurations apply and who they're shared with. Understanding scopes helps you decide how to configure Claude Code for personal use, team collaboration, or enterprise deployment.

#### Available scopes

| Scope       | Location                                                                           | Who it affects                       | Shared with team?      |
|-------------|------------------------------------------------------------------------------------|--------------------------------------|------------------------|
| **Managed** | Server-managed settings, plist / registry, or system-level `managed-settings.json` | All users on the machine             | Yes (deployed by IT)   |
| **User**    | `~/.claude/` directory                                                             | You, across all projects             | No                     |
| **Project** | `.claude/` in repository                                                           | All collaborators on this repository | Yes (committed to git) |
| **Local**   | `.claude/settings.local.json`                                                      | You, in this repository only         | No (gitignored)        |

#### When to use each scope

**Managed scope** is for:

- Security policies that must be enforced organization-wide
- Compliance requirements that can't be overridden
- Standardized configurations deployed by IT/DevOps

**User scope** is best for:

- Personal preferences you want everywhere (themes, editor settings)
- Tools and plugins you use across all projects
- API keys and authentication (stored securely)

**Project scope** is best for:

- Team-shared settings (permissions, hooks, MCP servers)
- Plugins the whole team should have
- Standardizing tooling across collaborators

**Local scope** is best for:

- Personal overrides for a specific project
- Testing configurations before sharing with the team
- Machine-specific settings that won't work for others

#### How scopes interact

When the same setting is configured in multiple scopes, more specific scopes take precedence:

1. **Managed** (highest) - can't be overridden by anything
2. **Command line arguments** - temporary session overrides
3. **Local** - overrides project and user settings
4. **Project** - overrides user settings
5. **User** (lowest) - applies when nothing else specifies the setting

For example, if a permission is allowed in user settings but denied in project settings, the project setting takes precedence and the permission is blocked.

#### What uses scopes

Scopes apply to many Claude Code features:

| Feature         | User location             | Project location                   | Local location                 |
|-----------------|---------------------------|------------------------------------|--------------------------------|
| **Settings**    | `~/.claude/settings.json` | `.claude/settings.json`            | `.claude/settings.local.json`  |
| **Subagents**   | `~/.claude/agents/`       | `.claude/agents/`                  | None                           |
| **MCP servers** | `~/.claude.json`          | `.mcp.json`                        | `~/.claude.json` (per-project) |
| **Plugins**     | `~/.claude/settings.json` | `.claude/settings.json`            | `.claude/settings.local.json`  |
| **CLAUDE.md**   | `~/.claude/CLAUDE.md`     | `CLAUDE.md` or `.claude/CLAUDE.md` | `CLAUDE.local.md`              |

### Settings files

The `settings.json` file is the official mechanism for configuring Claude

Code through hierarchical settings:

- **User settings** are defined in `~/.claude/settings.json` and apply to all projects.
- **Project settings** are saved in your project directory:
    - `.claude/settings.json` for settings that are checked into source control and shared with your team
    - `.claude/settings.local.json` for settings that are not checked in, useful for personal preferences and experimentation. Claude Code will configure git to ignore `.claude/settings.local.json` when it is created.
- **Managed settings** : For organizations that need centralized control, Claude Code supports multiple delivery mechanisms for managed settings. All use the same JSON format and cannot be overridden by user or project settings: See [managed settings](/docs/en/permissions#managed-only-settings) and [Managed MCP configuration](/docs/en/mcp#managed-mcp-configuration) for details. This [repository](https://github.com/anthropics/claude-code/tree/main/examples/mdm) includes starter deployment templates for Jamf, Iru (Kandji), Intune, and Group Policy. Use these as starting points and adjust them to fit your needs. Managed deployments can also restrict **plugin marketplace additions** using `strictKnownMarketplaces` . For more information, see [Managed marketplace restrictions](/docs/en/plugin-marketplaces#managed-marketplace-restrictions) .
    - **Server-managed settings** : delivered from Anthropic's servers via the Claude.ai admin console. See [server-managed settings](/docs/en/server-managed-settings) .
    - **MDM/OS-level policies** : delivered through native device management on macOS and Windows:
        - macOS: `com.anthropic.claudecode` managed preferences domain (deployed via configuration profiles in Jamf, Iru (Kandji), or other MDM tools)
        - Windows: `HKLM\SOFTWARE\Policies\ClaudeCode` registry key with a `Settings` value (REG\_SZ or REG\_EXPAND\_SZ) containing JSON (deployed via Group Policy or Intune)
        - Windows (user-level): `HKCU\SOFTWARE\Policies\ClaudeCode` (lowest policy priority, only used when no admin-level source exists)
    - **File-based** : `managed-settings.json` and `managed-mcp.json` deployed to system directories: The legacy Windows path `C:\ProgramData\ClaudeCode\managed-settings.json` is no longer supported as of v2.1.75. Administrators who deployed settings to that location must migrate files to `C:\Program Files\ClaudeCode\managed-settings.json` . File-based managed settings also support a drop-in directory at `managed-settings.d/` in the same system directory alongside `managed-settings.json` . This lets separate teams deploy independent policy fragments without coordinating edits to a single file. Following the systemd convention, `managed-settings.json` is merged first as the base, then all `*.json` files in the drop-in directory are sorted alphabetically and merged on top. Later files override earlier ones for scalar values; arrays are concatenated and de-duplicated; objects are deep-merged. Hidden files starting with `.` are ignored. Use numeric prefixes to control merge order, for example `10-telemetry.json` and `20-security.json` .
        - macOS: `/Library/Application Support/ClaudeCode/`
        - Linux and WSL: `/etc/claude-code/`
        - Windows: `C:\Program Files\ClaudeCode\`
- **Other configuration** is stored in `~/.claude.json` . This file contains your preferences (theme, notification settings, editor mode), OAuth session, [MCP server](/docs/en/mcp) configurations for user and local scopes, per-project state (allowed tools, trust settings), and various caches. Project-scoped MCP servers are stored separately in `.mcp.json` .

Claude Code automatically creates timestamped backups of configuration files and retains the five most recent backups to prevent data loss.

Example settings.json

```
{
"$schema" : "https://json.schemastore.org/claude-code-settings.json" ,
"permissions" : {
"allow" : [
"Bash(npm run lint)" ,
"Bash(npm run test *)" ,
"Read(~/.zshrc)"
],
"deny" : [
"Bash(curl *)" ,
"Read(./.env)" ,
"Read(./.env.*)" ,
"Read(./secrets/**)"
]
},
"env" : {
"CLAUDE_CODE_ENABLE_TELEMETRY" : "1" ,
"OTEL_METRICS_EXPORTER" : "otlp"
},
"companyAnnouncements" : [
"Welcome to Acme Corp! Review our code guidelines at docs.acme.com" ,
"Reminder: Code reviews required for all PRs" ,
"New security policy in effect"
]
}
```

The `$schema` line in the example above points to the [official JSON schema](https://json.schemastore.org/claude-code-settings.json) for Claude Code settings. Adding it to your `settings.json` enables autocomplete and inline validation in VS Code, Cursor, and any other editor that supports JSON schema validation.

#### Available settings

`settings.json` supports a number of options:

| Key                               | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Example                                                                                                                        |
|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `agent`                           | Run the main thread as a named subagent. Applies that subagent's system prompt, tool restrictions, and model. See [Invoke subagents explicitly](/docs/en/sub-agents#invoke-subagents-explicitly)                                                                                                                                                                                                                                                                                                                                                                     | `"code-reviewer"`                                                                                                              |
| `allowedChannelPlugins`           | (Managed settings only) Allowlist of channel plugins that may push messages. Replaces the default Anthropic allowlist when set. Undefined = fall back to the default, empty array = block all channel plugins. Requires `channelsEnabled: true` . See [Restrict which channel plugins can run](/docs/en/channels#restrict-which-channel-plugins-can-run)                                                                                                                                                                                                             | `[{ "marketplace": "claude-plugins-official", "plugin": "telegram" }]`                                                         |
| `allowedHttpHookUrls`             | Allowlist of URL patterns that HTTP hooks may target. Supports `*` as a wildcard. When set, hooks with non-matching URLs are blocked. Undefined = no restriction, empty array = block all HTTP hooks. Arrays merge across settings sources. See [Hook configuration](#hook-configuration)                                                                                                                                                                                                                                                                            | `["https://hooks.example.com/*"]`                                                                                              |
| `allowedMcpServers`               | When set in managed-settings.json, allowlist of MCP servers users can configure. Undefined = no restrictions, empty array = lockdown. Applies to all scopes. Denylist takes precedence. See [Managed MCP configuration](/docs/en/mcp#managed-mcp-configuration)                                                                                                                                                                                                                                                                                                      | `[{ "serverName": "github" }]`                                                                                                 |
| `allowManagedHooksOnly`           | (Managed settings only) Only managed hooks, SDK hooks, and hooks from plugins force-enabled in managed settings `enabledPlugins` are loaded. User, project, and all other plugin hooks are blocked. See [Hook configuration](#hook-configuration)                                                                                                                                                                                                                                                                                                                    | `true`                                                                                                                         |
| `allowManagedMcpServersOnly`      | (Managed settings only) Only `allowedMcpServers` from managed settings are respected. `deniedMcpServers` still merges from all sources. Users can still add MCP servers, but only the admin-defined allowlist applies. See [Managed MCP configuration](/docs/en/mcp#managed-mcp-configuration)                                                                                                                                                                                                                                                                       | `true`                                                                                                                         |
| `allowManagedPermissionRulesOnly` | (Managed settings only) Prevent user and project settings from defining `allow` , `ask` , or `deny` permission rules. Only rules in managed settings apply. See [Managed-only settings](/docs/en/permissions#managed-only-settings)                                                                                                                                                                                                                                                                                                                                  | `true`                                                                                                                         |
| `alwaysThinkingEnabled`           | Enable [extended thinking](/docs/en/common-workflows#use-extended-thinking-thinking-mode) by default for all sessions. Typically configured via the `/config` command rather than editing directly                                                                                                                                                                                                                                                                                                                                                                   | `true`                                                                                                                         |
| `apiKeyHelper`                    | Custom script, to be executed in `/bin/sh` , to generate an auth value. This value will be sent as `X-Api-Key` and `Authorization: Bearer` headers for model requests                                                                                                                                                                                                                                                                                                                                                                                                | `/bin/generate_temp_api_key.sh`                                                                                                |
| `attribution`                     | Customize attribution for git commits and pull requests. See [Attribution settings](#attribution-settings)                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `{"commit": "🤖 Generated with Claude Code", "pr": ""}`                                                                         |
| `autoMemoryDirectory`             | Custom directory for [auto memory](/docs/en/memory#storage-location) storage. Accepts `~/` -expanded paths. Not accepted in project settings ( `.claude/settings.json` ) to prevent shared repos from redirecting memory writes to sensitive locations. Accepted from policy, local, and user settings                                                                                                                                                                                                                                                               | `"~/my-memory-dir"`                                                                                                            |
| `autoMode`                        | Customize what the [auto mode](/docs/en/permission-modes#eliminate-prompts-with-auto-mode) classifier blocks and allows. Contains `environment` , `allow` , and `soft_deny` arrays of prose rules. See [Configure the auto mode classifier](/docs/en/permissions#configure-the-auto-mode-classifier) . Not read from shared project settings                                                                                                                                                                                                                         | `{"environment": ["Trusted repo: github.example.com/acme"]}`                                                                   |
| `autoUpdatesChannel`              | Release channel to follow for updates. Use `"stable"` for a version that is typically about one week old and skips versions with major regressions, or `"latest"` (default) for the most recent release                                                                                                                                                                                                                                                                                                                                                              | `"stable"`                                                                                                                     |
| `availableModels`                 | Restrict which models users can select via `/model` , `--model` , Config tool, or `ANTHROPIC_MODEL` . Does not affect the Default option. See [Restrict model selection](/docs/en/model-config#restrict-model-selection)                                                                                                                                                                                                                                                                                                                                             | `["sonnet", "haiku"]`                                                                                                          |
| `awsAuthRefresh`                  | Custom script that modifies the `.aws` directory (see [advanced credential configuration](/docs/en/amazon-bedrock#advanced-credential-configuration) )                                                                                                                                                                                                                                                                                                                                                                                                               | `aws sso login --profile myprofile`                                                                                            |
| `awsCredentialExport`             | Custom script that outputs JSON with AWS credentials (see [advanced credential configuration](/docs/en/amazon-bedrock#advanced-credential-configuration) )                                                                                                                                                                                                                                                                                                                                                                                                           | `/bin/generate_aws_grant.sh`                                                                                                   |
| `blockedMarketplaces`             | (Managed settings only) Blocklist of marketplace sources. Blocked sources are checked before downloading, so they never touch the filesystem. See [Managed marketplace restrictions](/docs/en/plugin-marketplaces#managed-marketplace-restrictions)                                                                                                                                                                                                                                                                                                                  | `[{ "source": "github", "repo": "untrusted/plugins" }]`                                                                        |
| `channelsEnabled`                 | (Managed settings only) Allow [channels](/docs/en/channels) for Team and Enterprise users. Unset or `false` blocks channel message delivery regardless of what users pass to `--channels`                                                                                                                                                                                                                                                                                                                                                                            | `true`                                                                                                                         |
| `cleanupPeriodDays`               | Session files older than this period are deleted at startup (default: 30 days, minimum 1). Setting to `0` is rejected with a validation error. Also controls the age cutoff for automatic removal of [orphaned subagent worktrees](/docs/en/common-workflows#worktree-cleanup) at startup. To disable transcript writes entirely in non-interactive mode ( `-p` ), use the `--no-session-persistence` flag or the `persistSession: false` SDK option; there is no interactive-mode equivalent.                                                                       | `20`                                                                                                                           |
| `companyAnnouncements`            | Announcement to display to users at startup. If multiple announcements are provided, they will be cycled through at random.                                                                                                                                                                                                                                                                                                                                                                                                                                          | `["Welcome to Acme Corp! Review our code guidelines at docs.acme.com"]`                                                        |
| `defaultShell`                    | Default shell for input-box `!` commands. Accepts `"bash"` (default) or `"powershell"` . Setting `"powershell"` routes interactive `!` commands through PowerShell on Windows. Requires `CLAUDE_CODE_USE_POWERSHELL_TOOL=1` . See [PowerShell tool](/docs/en/tools-reference#powershell-tool)                                                                                                                                                                                                                                                                        | `"powershell"`                                                                                                                 |
| `deniedMcpServers`                | When set in managed-settings.json, denylist of MCP servers that are explicitly blocked. Applies to all scopes including managed servers. Denylist takes precedence over allowlist. See [Managed MCP configuration](/docs/en/mcp#managed-mcp-configuration)                                                                                                                                                                                                                                                                                                           | `[{ "serverName": "filesystem" }]`                                                                                             |
| `disableAllHooks`                 | Disable all [hooks](/docs/en/hooks) and any custom [status line](/docs/en/statusline)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | `true`                                                                                                                         |
| `disableAutoMode`                 | Set to `"disable"` to prevent [auto mode](/docs/en/permission-modes#eliminate-prompts-with-auto-mode) from being activated. Removes `auto` from the `Shift+Tab` cycle and rejects `--permission-mode auto` at startup. Most useful in [managed settings](/docs/en/permissions#managed-settings) where users cannot override it                                                                                                                                                                                                                                       | `"disable"`                                                                                                                    |
| `disableDeepLinkRegistration`     | Set to `"disable"` to prevent Claude Code from registering the `claude-cli://` protocol handler with the operating system on startup. Deep links let external tools open a Claude Code session with a pre-filled prompt via `claude-cli://open?q=...` . The `q` parameter supports multi-line prompts using URL-encoded newlines ( `%0A` ). Useful in environments where protocol handler registration is restricted or managed separately                                                                                                                           | `"disable"`                                                                                                                    |
| `disabledMcpjsonServers`          | List of specific MCP servers from `.mcp.json` files to reject                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | `["filesystem"]`                                                                                                               |
| `disableSkillShellExecution`      | Disable inline shell execution for `!`...`` and ````!` blocks in [skills](/docs/en/skills) and custom commands from user, project, plugin, or additional-directory sources. Commands are replaced with `[shell command execution disabled by policy]` instead of being run. Bundled and managed skills are not affected. Most useful in [managed settings](/docs/en/permissions#managed-settings) where users cannot override it                                                                                                                                     | `true`                                                                                                                         |
| `effortLevel`                     | Persist the [effort level](/docs/en/model-config#adjust-effort-level) across sessions. Accepts `"low"` , `"medium"` , or `"high"` . Written automatically when you run `/effort low` , `/effort medium` , or `/effort high` . Supported on Opus 4.6 and Sonnet 4.6                                                                                                                                                                                                                                                                                                   | `"medium"`                                                                                                                     |
| `enableAllProjectMcpServers`      | Automatically approve all MCP servers defined in project `.mcp.json` files                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `true`                                                                                                                         |
| `enabledMcpjsonServers`           | List of specific MCP servers from `.mcp.json` files to approve                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | `["memory", "github"]`                                                                                                         |
| `env`                             | Environment variables that will be applied to every session                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | `{"FOO": "bar"}`                                                                                                               |
| `fastModePerSessionOptIn`         | When `true` , fast mode does not persist across sessions. Each session starts with fast mode off, requiring users to enable it with `/fast` . The user's fast mode preference is still saved. See [Require per-session opt-in](/docs/en/fast-mode#require-per-session-opt-in)                                                                                                                                                                                                                                                                                        | `true`                                                                                                                         |
| `feedbackSurveyRate`              | Probability (0-1) that the [session quality survey](/docs/en/data-usage#session-quality-surveys) appears when eligible. Set to `0` to suppress entirely. Useful when using Bedrock, Vertex, or Foundry where the default sample rate does not apply                                                                                                                                                                                                                                                                                                                  | `0.05`                                                                                                                         |
| `fileSuggestion`                  | Configure a custom script for `@` file autocomplete. See [File suggestion settings](#file-suggestion-settings)                                                                                                                                                                                                                                                                                                                                                                                                                                                       | `{"type": "command", "command": "~/.claude/file-suggestion.sh"}`                                                               |
| `forceLoginMethod`                | Use `claudeai` to restrict login to Claude.ai accounts, `console` to restrict login to Claude Console (API usage billing) accounts                                                                                                                                                                                                                                                                                                                                                                                                                                   | `claudeai`                                                                                                                     |
| `forceLoginOrgUUID`               | Require login to belong to a specific organization. Accepts a single UUID string, which also pre-selects that organization during login, or an array of UUIDs where any listed organization is accepted without pre-selection. When set in managed settings, login fails if the authenticated account does not belong to a listed organization; an empty array fails closed and blocks login with a misconfiguration message                                                                                                                                         | `"xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"` or `["xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", "yyyyyyyy-yyyy-yyyy-yyyy-yyyyyyyyyyyy"]` |
| `forceRemoteSettingsRefresh`      | (Managed settings only) Block CLI startup until remote managed settings are freshly fetched from the server. If the fetch fails, the CLI exits rather than continuing with cached or no settings. When not set, startup continues without waiting for remote settings. See [fail-closed enforcement](/docs/en/server-managed-settings#enforce-fail-closed-startup)                                                                                                                                                                                                   | `true`                                                                                                                         |
| `hooks`                           | Configure custom commands to run at lifecycle events. See [hooks documentation](/docs/en/hooks) for format                                                                                                                                                                                                                                                                                                                                                                                                                                                           | See [hooks](/docs/en/hooks)                                                                                                    |
| `httpHookAllowedEnvVars`          | Allowlist of environment variable names HTTP hooks may interpolate into headers. When set, each hook's effective `allowedEnvVars` is the intersection with this list. Undefined = no restriction. Arrays merge across settings sources. See [Hook configuration](#hook-configuration)                                                                                                                                                                                                                                                                                | `["MY_TOKEN", "HOOK_SECRET"]`                                                                                                  |
| `includeCoAuthoredBy`             | **Deprecated** : Use `attribution` instead. Whether to include the `co-authored-by Claude` byline in git commits and pull requests (default: `true` )                                                                                                                                                                                                                                                                                                                                                                                                                | `false`                                                                                                                        |
| `includeGitInstructions`          | Include built-in commit and PR workflow instructions and the git status snapshot in Claude's system prompt (default: `true` ). Set to `false` to remove both, for example when using your own git workflow skills. The `CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS` environment variable takes precedence over this setting when set                                                                                                                                                                                                                                       | `false`                                                                                                                        |
| `language`                        | Configure Claude's preferred response language (e.g., `"japanese"` , `"spanish"` , `"french"` ). Claude will respond in this language by default. Also sets the [voice dictation](/docs/en/voice-dictation#change-the-dictation-language) language                                                                                                                                                                                                                                                                                                                   | `"japanese"`                                                                                                                   |
| `model`                           | Override the default model to use for Claude Code                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | `"claude-sonnet-4-6"`                                                                                                          |
| `modelOverrides`                  | Map Anthropic model IDs to provider-specific model IDs such as Bedrock inference profile ARNs. Each model picker entry uses its mapped value when calling the provider API. See [Override model IDs per version](/docs/en/model-config#override-model-ids-per-version)                                                                                                                                                                                                                                                                                               | `{"claude-opus-4-6": "arn:aws:bedrock:..."}`                                                                                   |
| `otelHeadersHelper`               | Script to generate dynamic OpenTelemetry headers. Runs at startup and periodically (see [Dynamic headers](/docs/en/monitoring-usage#dynamic-headers) )                                                                                                                                                                                                                                                                                                                                                                                                               | `/bin/generate_otel_headers.sh`                                                                                                |
| `outputStyle`                     | Configure an output style to adjust the system prompt. See [output styles documentation](/docs/en/output-styles)                                                                                                                                                                                                                                                                                                                                                                                                                                                     | `"Explanatory"`                                                                                                                |
| `permissions`                     | See table below for structure of permissions.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                |
| `plansDirectory`                  | Customize where plan files are stored. Path is relative to project root. Default: `~/.claude/plans`                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | `"./plans"`                                                                                                                    |
| `pluginTrustMessage`              | (Managed settings only) Custom message appended to the plugin trust warning shown before installation. Use this to add organization-specific context, for example to confirm that plugins from your internal marketplace are vetted.                                                                                                                                                                                                                                                                                                                                 | `"All plugins from our marketplace are approved by IT"`                                                                        |
| `prefersReducedMotion`            | Reduce or disable UI animations (spinners, shimmer, flash effects) for accessibility                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | `true`                                                                                                                         |
| `respectGitignore`                | Control whether the `@` file picker respects `.gitignore` patterns. When `true` (default), files matching `.gitignore` patterns are excluded from suggestions                                                                                                                                                                                                                                                                                                                                                                                                        | `false`                                                                                                                        |
| `showClearContextOnPlanAccept`    | Show the "clear context" option on the plan accept screen. Defaults to `false` . Set to `true` to restore the option                                                                                                                                                                                                                                                                                                                                                                                                                                                 | `true`                                                                                                                         |
| `showThinkingSummaries`           | Show [extended thinking](/docs/en/common-workflows#use-extended-thinking-thinking-mode) summaries in interactive sessions. When unset or `false` (default in interactive mode), thinking blocks are redacted by the API and shown as a collapsed stub. Redaction only changes what you see, not what the model generates: to reduce thinking spend, [lower the budget or disable thinking](/docs/en/common-workflows#use-extended-thinking-thinking-mode) instead. Non-interactive mode ( `-p` ) and SDK callers always receive summaries regardless of this setting | `true`                                                                                                                         |
| `spinnerTipsEnabled`              | Show tips in the spinner while Claude is working. Set to `false` to disable tips (default: `true` )                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | `false`                                                                                                                        |
| `spinnerTipsOverride`             | Override spinner tips with custom strings. `tips` : array of tip strings. `excludeDefault` : if `true` , only show custom tips; if `false` or absent, custom tips are merged with built-in tips                                                                                                                                                                                                                                                                                                                                                                      | `{ "excludeDefault": true, "tips": ["Use our internal tool X"] }`                                                              |
| `spinnerVerbs`                    | Customize the action verbs shown in the spinner and turn duration messages. Set `mode` to `"replace"` to use only your verbs, or `"append"` to add them to the defaults                                                                                                                                                                                                                                                                                                                                                                                              | `{"mode": "append", "verbs": ["Pondering", "Crafting"]}`                                                                       |
| `statusLine`                      | Configure a custom status line to display context. See [`statusLine`](/docs/en/statusline) [documentation](/docs/en/statusline)                                                                                                                                                                                                                                                                                                                                                                                                                                      | `{"type": "command", "command": "~/.claude/statusline.sh"}`                                                                    |
| `strictKnownMarketplaces`         | (Managed settings only) Allowlist of plugin marketplaces users can add. Undefined = no restrictions, empty array = lockdown. Applies to marketplace additions only. See [Managed marketplace restrictions](/docs/en/plugin-marketplaces#managed-marketplace-restrictions)                                                                                                                                                                                                                                                                                            | `[{ "source": "github", "repo": "acme-corp/plugins" }]`                                                                        |
| `useAutoModeDuringPlan`           | Whether plan mode uses auto mode semantics when auto mode is available. Default: `true` . Not read from shared project settings. Appears in `/config` as "Use auto mode during plan"                                                                                                                                                                                                                                                                                                                                                                                 | `false`                                                                                                                        |
| `voiceEnabled`                    | Enable push-to-talk [voice dictation](/docs/en/voice-dictation) . Written automatically when you run `/voice` . Requires a Claude.ai account                                                                                                                                                                                                                                                                                                                                                                                                                         | `true`                                                                                                                         |

#### Global config settings

These settings are stored in `~/.claude.json` rather than `settings.json` . Adding them to `settings.json` will trigger a schema validation error.

| Key                          | Description                                                                                                                                                                                                                                                                                                                | Example        |
|------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| `autoConnectIde`             | Automatically connect to a running IDE when Claude Code starts from an external terminal. Default: `false` . Appears in `/config` as **Auto-connect to IDE (external terminal)** when running outside a VS Code or JetBrains terminal                                                                                      | `true`         |
| `autoInstallIdeExtension`    | Automatically install the Claude Code IDE extension when running from a VS Code terminal. Default: `true` . Appears in `/config` as **Auto-install IDE extension** when running inside a VS Code or JetBrains terminal. You can also set the [`CLAUDE_CODE_IDE_SKIP_AUTO_INSTALL`](/docs/en/env-vars) environment variable | `false`        |
| `editorMode`                 | Key binding mode for the input prompt: `"normal"` or `"vim"` . Default: `"normal"` . Appears in `/config` as **Editor mode**                                                                                                                                                                                               | `"vim"`        |
| `showTurnDuration`           | Show turn duration messages after responses, e.g. "Cooked for 1m 6s". Default: `true` . Appears in `/config` as **Show turn duration**                                                                                                                                                                                     | `false`        |
| `terminalProgressBarEnabled` | Show the terminal progress bar in supported terminals: ConEmu, Ghostty 1.2.0+, and iTerm2 3.6.6+. Default: `true` . Appears in `/config` as **Terminal progress bar**                                                                                                                                                      | `false`        |
| `teammateMode`               | How [agent team](/docs/en/agent-teams) teammates display: `auto` (picks split panes in tmux or iTerm2, in-process otherwise), `in-process` , or `tmux` . See [choose a display mode](/docs/en/agent-teams#choose-a-display-mode)                                                                                           | `"in-process"` |

#### Worktree settings

Configure how `--worktree` creates and manages git worktrees. Use these settings to reduce disk usage and startup time in large monorepos.

| Key                           | Description                                                                                                                                                  | Example                               |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------|
| `worktree.symlinkDirectories` | Directories to symlink from the main repository into each worktree to avoid duplicating large directories on disk. No directories are symlinked by default   | `["node_modules", ".cache"]`          |
| `worktree.sparsePaths`        | Directories to check out in each worktree via git sparse-checkout (cone mode). Only the listed paths are written to disk, which is faster in large monorepos | `["packages/my-app", "shared/utils"]` |

To copy gitignored files like `.env` into new worktrees, use a [`.worktreeinclude`](/docs/en/common-workflows#copy-gitignored-files-to-worktrees) [file](/docs/en/common-workflows#copy-gitignored-files-to-worktrees) in your project root instead of a setting.

#### Permission settings

| Keys                                | Description                                                                                                                                                                                                                                                                                 | Example                                                                |
|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| `allow`                             | Array of permission rules to allow tool use. See [Permission rule syntax](#permission-rule-syntax) below for pattern matching details                                                                                                                                                       | `[ "Bash(git diff *)" ]`                                               |
| `ask`                               | Array of permission rules to ask for confirmation upon tool use. See [Permission rule syntax](#permission-rule-syntax) below                                                                                                                                                                | `[ "Bash(git push *)" ]`                                               |
| `deny`                              | Array of permission rules to deny tool use. Use this to exclude sensitive files from Claude Code access. See [Permission rule syntax](#permission-rule-syntax) and [Bash permission limitations](/docs/en/permissions#tool-specific-permission-rules)                                       | `[ "WebFetch", "Bash(curl *)", "Read(./.env)", "Read(./secrets/**)" ]` |
| `additionalDirectories`             | Additional [working directories](/docs/en/permissions#working-directories) for file access. Most `.claude/` configuration is [not discovered](/docs/en/permissions#additional-directories-grant-file-access-not-configuration) from these directories                                       | `[ "../docs/" ]`                                                       |
| `defaultMode`                       | Default [permission mode](/docs/en/permission-modes) when opening Claude Code. Valid values: `default` , `acceptEdits` , `plan` , `auto` , `dontAsk` , `bypassPermissions` . The `--permission-mode` CLI flag overrides this setting for a single session                                   | `"acceptEdits"`                                                        |
| `disableBypassPermissionsMode`      | Set to `"disable"` to prevent `bypassPermissions` mode from being activated. This disables the `--dangerously-skip-permissions` command-line flag. Typically placed in [managed settings](/docs/en/permissions#managed-settings) to enforce organizational policy, but works from any scope | `"disable"`                                                            |
| `skipDangerousModePermissionPrompt` | Skip the confirmation prompt shown before entering bypass permissions mode via `--dangerously-skip-permissions` or `defaultMode: "bypassPermissions"` . Ignored when set in project settings ( `.claude/settings.json` ) to prevent untrusted repositories from auto-bypassing the prompt   | `true`                                                                 |

#### Permission rule syntax

Permission rules follow the format `Tool` or `Tool(specifier)` . Rules are evaluated in order: deny rules first, then ask, then allow. The first matching rule wins. Quick examples:

| Rule                           | Effect                                   |
|--------------------------------|------------------------------------------|
| `Bash`                         | Matches all Bash commands                |
| `Bash(npm run *)`              | Matches commands starting with `npm run` |
| `Read(./.env)`                 | Matches reading the `.env` file          |
| `WebFetch(domain:example.com)` | Matches fetch requests to example.com    |

For the complete rule syntax reference, including wildcard behavior, tool-specific patterns for Read, Edit, WebFetch, MCP, and Agent rules, and security limitations of Bash patterns, see [Permission rule syntax](/docs/en/permissions#permission-rule-syntax) .

#### Sandbox settings

Configure advanced sandboxing behavior. Sandboxing isolates bash commands from your filesystem and network. See [Sandboxing](/docs/en/sandboxing) for details.

| Keys                                   | Description                                                                                                                                                                                                                                                                                                                                         | Example                         |
|----------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------|
| `enabled`                              | Enable bash sandboxing (macOS, Linux, and WSL2). Default: false                                                                                                                                                                                                                                                                                     | `true`                          |
| `failIfUnavailable`                    | Exit with an error at startup if `sandbox.enabled` is true but the sandbox cannot start (missing dependencies, unsupported platform, or platform restrictions). When false (default), a warning is shown and commands run unsandboxed. Intended for managed settings deployments that require sandboxing as a hard gate                             | `true`                          |
| `autoAllowBashIfSandboxed`             | Auto-approve bash commands when sandboxed. Default: true                                                                                                                                                                                                                                                                                            | `true`                          |
| `excludedCommands`                     | Commands that should run outside of the sandbox                                                                                                                                                                                                                                                                                                     | `["docker *"]`                  |
| `allowUnsandboxedCommands`             | Allow commands to run outside the sandbox via the `dangerouslyDisableSandbox` parameter. When set to `false` , the `dangerouslyDisableSandbox` escape hatch is completely disabled and all commands must run sandboxed (or be in `excludedCommands` ). Useful for enterprise policies that require strict sandboxing. Default: true                 | `false`                         |
| `filesystem.allowWrite`                | Additional paths where sandboxed commands can write. Arrays are merged across all settings scopes: user, project, and managed paths are combined, not replaced. Also merged with paths from `Edit(...)` allow permission rules. See [path prefixes](#sandbox-path-prefixes) below.                                                                  | `["/tmp/build", "~/.kube"]`     |
| `filesystem.denyWrite`                 | Paths where sandboxed commands cannot write. Arrays are merged across all settings scopes. Also merged with paths from `Edit(...)` deny permission rules.                                                                                                                                                                                           | `["/etc", "/usr/local/bin"]`    |
| `filesystem.denyRead`                  | Paths where sandboxed commands cannot read. Arrays are merged across all settings scopes. Also merged with paths from `Read(...)` deny permission rules.                                                                                                                                                                                            | `["~/.aws/credentials"]`        |
| `filesystem.allowRead`                 | Paths to re-allow reading within `denyRead` regions. Takes precedence over `denyRead` . Arrays are merged across all settings scopes. Use this to create workspace-only read access patterns.                                                                                                                                                       | `["."]`                         |
| `filesystem.allowManagedReadPathsOnly` | (Managed settings only) Only `filesystem.allowRead` paths from managed settings are respected. `denyRead` still merges from all sources. Default: false                                                                                                                                                                                             | `true`                          |
| `network.allowUnixSockets`             | Unix socket paths accessible in sandbox (for SSH agents, etc.)                                                                                                                                                                                                                                                                                      | `["~/.ssh/agent-socket"]`       |
| `network.allowAllUnixSockets`          | Allow all Unix socket connections in sandbox. Default: false                                                                                                                                                                                                                                                                                        | `true`                          |
| `network.allowLocalBinding`            | Allow binding to localhost ports (macOS only). Default: false                                                                                                                                                                                                                                                                                       | `true`                          |
| `network.allowMachLookup`              | Additional XPC/Mach service names the sandbox may look up (macOS only). Supports a single trailing `*` for prefix matching. Needed for tools that communicate via XPC such as the iOS Simulator or Playwright.                                                                                                                                      | `["com.apple.coresimulator.*"]` |
| `network.allowedDomains`               | Array of domains to allow for outbound network traffic. Supports wildcards (e.g., `*.example.com` ).                                                                                                                                                                                                                                                | `["github.com", "*.npmjs.org"]` |
| `network.allowManagedDomainsOnly`      | (Managed settings only) Only `allowedDomains` and `WebFetch(domain:...)` allow rules from managed settings are respected. Domains from user, project, and local settings are ignored. Non-allowed domains are blocked automatically without prompting the user. Denied domains are still respected from all sources. Default: false                 | `true`                          |
| `network.httpProxyPort`                | HTTP proxy port used if you wish to bring your own proxy. If not specified, Claude will run its own proxy.                                                                                                                                                                                                                                          | `8080`                          |
| `network.socksProxyPort`               | SOCKS5 proxy port used if you wish to bring your own proxy. If not specified, Claude will run its own proxy.                                                                                                                                                                                                                                        | `8081`                          |
| `enableWeakerNestedSandbox`            | Enable weaker sandbox for unprivileged Docker environments (Linux and WSL2 only). **Reduces security.** Default: false                                                                                                                                                                                                                              | `true`                          |
| `enableWeakerNetworkIsolation`         | (macOS only) Allow access to the system TLS trust service ( `com.apple.trustd.agent` ) in the sandbox. Required for Go-based tools like `gh` , `gcloud` , and `terraform` to verify TLS certificates when using `httpProxyPort` with a MITM proxy and custom CA. **Reduces security** by opening a potential data exfiltration path. Default: false | `true`                          |

##### Sandbox path prefixes

Paths in `filesystem.allowWrite` , `filesystem.denyWrite` , `filesystem.denyRead` , and `filesystem.allowRead` support these prefixes:

| Prefix            | Meaning                                                                                | Example                                                                   |
|-------------------|----------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| `/`               | Absolute path from filesystem root                                                     | `/tmp/build` stays `/tmp/build`                                           |
| `~/`              | Relative to home directory                                                             | `~/.kube` becomes `$HOME/.kube`                                           |
| `./` or no prefix | Relative to the project root for project settings, or to `~/.claude` for user settings | `./output` in `.claude/settings.json` resolves to `<project-root>/output` |

The older `//path` prefix for absolute paths still works. If you previously used single-slash `/path` expecting project-relative resolution, switch to `./path` . This syntax differs from [Read and Edit permission rules](/docs/en/permissions#read-and-edit) , which use `//path` for absolute and `/path` for project-relative. Sandbox filesystem paths use standard conventions: `/tmp/build` is an absolute path. **Configuration example:**

```
{
"sandbox" : {
"enabled" : true ,
"autoAllowBashIfSandboxed" : true ,
"excludedCommands" : [ "docker *" ],
"filesystem" : {
"allowWrite" : [ "/tmp/build" , "~/.kube" ],
"denyRead" : [ "~/.aws/credentials" ]
},
"network" : {
"allowedDomains" : [ "github.com" , "*.npmjs.org" , "registry.yarnpkg.com" ],
"allowUnixSockets" : [
"/var/run/docker.sock"
],
"allowLocalBinding" : true
}
}
}
```

**Filesystem and network restrictions** can be configured in two ways that are merged together:

- **`sandbox.filesystem`** **settings** (shown above): Control paths at the OS-level sandbox boundary. These restrictions apply to all subprocess commands (e.g., `kubectl` , `terraform` , `npm` ), not just Claude's file tools.
- **Permission rules** : Use `Edit` allow/deny rules to control Claude's file tool access, `Read` deny rules to block reads, and `WebFetch` allow/deny rules to control network domains. Paths from these rules are also merged into the sandbox configuration.

#### Attribution settings

Claude Code adds attribution to git commits and pull requests. These are configured separately:

- Commits use [git trailers](https://git-scm.com/docs/git-interpret-trailers) (like `Co-Authored-By` ) by default, which can be customized or disabled
- Pull request descriptions are plain text

| Keys     | Description                                                                                |
|----------|--------------------------------------------------------------------------------------------|
| `commit` | Attribution for git commits, including any trailers. Empty string hides commit attribution |
| `pr`     | Attribution for pull request descriptions. Empty string hides pull request attribution     |

**Default commit attribution:**

`🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.6 <` [`[email protected]`](/cdn-cgi/l/email-protection) `>`

**Default pull request attribution:**

```
🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

**Example:**

`{
"attribution" : {
"commit" : "Generated with AI    Co-Authored-By: AI <` [`[email protected]`](/cdn-cgi/l/email-protection) `>" ,
"pr" : ""
}
}`

The `attribution` setting takes precedence over the deprecated `includeCoAuthoredBy` setting. To hide all attribution, set `commit` and `pr` to empty strings.

#### File suggestion settings

Configure a custom command for `@` file path autocomplete. The built-in file suggestion uses fast filesystem traversal, but large monorepos may benefit from project-specific indexing such as a pre-built file index or custom tooling.

```
{
"fileSuggestion" : {
"type" : "command" ,
"command" : "~/.claude/file-suggestion.sh"
}
}
```

The command runs with the same environment variables as [hooks](/docs/en/hooks) , including `CLAUDE_PROJECT_DIR` . It receives JSON via stdin with a `query` field:

```
{ "query" : "src/comp" }
```

Output newline-separated file paths to stdout (currently limited to 15):

```
src/components/Button.tsx
src/components/Modal.tsx
src/components/Form.tsx
```

**Example:**

```
#!/bin/bash
query = $( cat | jq -r '.query' )
your-repo-file-index --query " $query " | head -20
```

#### Hook configuration

These settings control which hooks are allowed to run and what HTTP hooks can access. The `allowManagedHooksOnly` setting can only be configured in [managed settings](#settings-files) . The URL and env var allowlists can be set at any settings level and merge across sources. **Behavior when** **`allowManagedHooksOnly`** **is** **`true`** **:**

- Managed hooks and SDK hooks are loaded
- Hooks from plugins force-enabled in managed settings `enabledPlugins` are loaded. This lets administrators distribute vetted hooks through an organization marketplace while blocking everything else. Trust is granted by full `plugin@marketplace` ID, so a plugin with the same name from a different marketplace stays blocked
- User hooks, project hooks, and all other plugin hooks are blocked

**Restrict HTTP hook URLs:** Limit which URLs HTTP hooks can target. Supports `*` as a wildcard for matching. When the array is defined, HTTP hooks targeting non-matching URLs are silently blocked.

```
{
"allowedHttpHookUrls" : [ "https://hooks.example.com/*" , "http://localhost:*" ]
}
```

**Restrict HTTP hook environment variables:** Limit which environment variable names HTTP hooks can interpolate into header values. Each hook's effective `allowedEnvVars` is the intersection of its own list and this setting.

```
{
"httpHookAllowedEnvVars" : [ "MY_TOKEN" , "HOOK_SECRET" ]
}
```

#### Settings precedence

Settings apply in order of precedence. From highest to lowest:

1. **Managed settings** ( [server-managed](/docs/en/server-managed-settings) , [MDM/OS-level policies](#configuration-scopes) , or [managed settings](/docs/en/settings#settings-files) )
    - Policies deployed by IT through server delivery, MDM configuration profiles, registry policies, or managed settings files
    - Cannot be overridden by any other level, including command line arguments
    - Within the managed tier, precedence is: server-managed > MDM/OS-level policies > file-based ( `managed-settings.d/*.json` + `managed-settings.json` ) > HKCU registry (Windows only). Only one managed source is used; sources do not merge across tiers. Within the file-based tier, drop-in files and the base file are merged together.
2. **Command line arguments**
    - Temporary overrides for a specific session
3. **Local project settings** ( `.claude/settings.local.json` )
    - Personal project-specific settings
4. **Shared project settings** ( `.claude/settings.json` )
    - Team-shared project settings in source control
5. **User settings** ( `~/.claude/settings.json` )
    - Personal global settings

This hierarchy ensures that organizational policies are always enforced while still allowing teams and individuals to customize their experience. The same precedence applies whether you run Claude Code from the CLI, the [VS Code extension](/docs/en/vs-code) , or a [JetBrains IDE](/docs/en/jetbrains) . For example, if your user settings allow `Bash(npm run *)` but a project's shared settings deny it, the project setting takes precedence and the command is blocked.

**Array settings merge across scopes.** When the same array-valued setting (such as `sandbox.filesystem.allowWrite` or `permissions.allow` ) appears in multiple scopes, the arrays are **concatenated and deduplicated** , not replaced. This means lower-priority scopes can add entries without overriding those set by higher-priority scopes, and vice versa. For example, if managed settings set `allowWrite` to `["/opt/company-tools"]` and a user adds `["~/.kube"]` , both paths are included in the final configuration.

#### Verify active settings

Run `/status` inside Claude Code to see which settings sources are active and where they come from. The output shows each configuration layer (managed, user, project) along with its origin, such as `Enterprise managed settings (remote)` , `Enterprise managed settings (plist)` , `Enterprise managed settings (HKLM)` , or `Enterprise managed settings (file)` . If a settings file contains errors, `/status` reports the issue so you can fix it.

#### Key points about the configuration system

- **Memory files (** **`CLAUDE.md`** **)** : Contain instructions and context that Claude loads at startup
- **Settings files (JSON)** : Configure permissions, environment variables, and tool behavior
- **Skills** : Custom prompts that can be invoked with `/skill-name` or loaded by Claude automatically
- **MCP servers** : Extend Claude Code with additional tools and integrations
- **Precedence** : Higher-level configurations (Managed) override lower-level ones (User/Project)
- **Inheritance** : Settings are merged, with more specific settings adding to or overriding broader ones

#### System prompt

Claude Code's internal system prompt is not published. To add custom instructions, use `CLAUDE.md` files or the `--append-system-prompt` flag.

#### Excluding sensitive files

To prevent Claude Code from accessing files containing sensitive information like API keys, secrets, and environment files, use the `permissions.deny` setting in your `.claude/settings.json` file:

```
{
"permissions" : {
"deny" : [
"Read(./.env)" ,
"Read(./.env.*)" ,
"Read(./secrets/**)" ,
"Read(./config/credentials.json)" ,
"Read(./build)"
]
}
}
```

This replaces the deprecated `ignorePatterns` configuration. Files matching these patterns are excluded from file discovery and search results, and read operations on these files are denied.

### Subagent configuration

Claude Code supports custom AI subagents that can be configured at both user and project levels. These subagents are stored as Markdown files with YAML frontmatter:

- **User subagents** : `~/.claude/agents/` - Available across all your projects
- **Project subagents** : `.claude/agents/` - Specific to your project and can be shared with your team

Subagent files define specialized AI assistants with custom prompts and tool permissions. Learn more about creating and using subagents in the [subagents documentation](/docs/en/sub-agents) .

### Plugin configuration

Claude Code supports a plugin system that lets you extend functionality with skills, agents, hooks, and MCP servers. Plugins are distributed through marketplaces and can be configured at both user and repository levels.

#### Plugin settings

Plugin-related settings in `settings.json` :

```
{
"enabledPlugins" : {
"formatter@acme-tools" : true ,
"deployer@acme-tools" : true ,
"analyzer@security-plugins" : false
},
"extraKnownMarketplaces" : {
"acme-tools" : {
"source" : "github" ,
"repo" : "acme-corp/claude-plugins"
}
}
}
```

##### enabledPlugins

Controls which plugins are enabled. Format: `"plugin-name@marketplace-name": true/false` **Scopes** :

- **User settings** ( `~/.claude/settings.json` ): Personal plugin preferences
- **Project settings** ( `.claude/settings.json` ): Project-specific plugins shared with team
- **Local settings** ( `.claude/settings.local.json` ): Per-machine overrides (not committed)
- **Managed settings** ( `managed-settings.json` ): Organization-wide policy overrides that block installation at all scopes and hide the plugin from the marketplace

**Example** :

```
{
"enabledPlugins" : {
"code-formatter@team-tools" : true ,
"deployment-tools@team-tools" : true ,
"experimental-features@personal" : false
}
}
```

##### extraKnownMarketplaces

Defines additional marketplaces that should be made available for the repository. Typically used in repository-level settings to ensure team members have access to required plugin sources. **When a repository includes** **`extraKnownMarketplaces`** :

1. Team members are prompted to install the marketplace when they trust the folder
2. Team members are then prompted to install plugins from that marketplace
3. Users can skip unwanted marketplaces or plugins (stored in user settings)
4. Installation respects trust boundaries and requires explicit consent

**Example** :

```
{
"extraKnownMarketplaces" : {
"acme-tools" : {
"source" : {
"source" : "github" ,
"repo" : "acme-corp/claude-plugins"
}
},
"security-plugins" : {
"source" : {
"source" : "git" ,
"url" : "https://git.example.com/security/plugins.git"
}
}
}
}
```

**Marketplace source types** :

- `github` : GitHub repository (uses `repo` )
- `git` : Any git URL (uses `url` )
- `directory` : Local filesystem path (uses `path` , for development only)
- `hostPattern` : regex pattern to match marketplace hosts (uses `hostPattern` )
- `settings` : inline marketplace declared directly in settings.json without a separate hosted repository (uses `name` and `plugins` )

Use `source: 'settings'` to declare a small set of plugins inline without setting up a hosted marketplace repository. Plugins listed here must reference external sources such as GitHub or npm. You still need to enable each plugin separately in `enabledPlugins` .

```
{
"extraKnownMarketplaces" : {
"team-tools" : {
"source" : {
"source" : "settings" ,
"name" : "team-tools" ,
"plugins" : [
{
"name" : "code-formatter" ,
"source" : {
"source" : "github" ,
"repo" : "acme-corp/code-formatter"
}
}
]
}
}
}
}
```

##### strictKnownMarketplaces

**Managed settings only** : Controls which plugin marketplaces users are allowed to add. This setting can only be configured in [managed settings](/docs/en/settings#settings-files) and provides administrators with strict control over marketplace sources. **Managed settings file locations** :

- **macOS** : `/Library/Application Support/ClaudeCode/managed-settings.json`
- **Linux and WSL** : `/etc/claude-code/managed-settings.json`
- **Windows** : `C:\Program Files\ClaudeCode\managed-settings.json`

**Key characteristics** :

- Only available in managed settings ( `managed-settings.json` )
- Cannot be overridden by user or project settings (highest precedence)
- Enforced BEFORE network/filesystem operations (blocked sources never execute)
- Uses exact matching for source specifications (including `ref` , `path` for git sources), except `hostPattern` , which uses regex matching

**Allowlist behavior** :

- `undefined` (default): No restrictions - users can add any marketplace
- Empty array `[]` : Complete lockdown - users cannot add any new marketplaces
- List of sources: Users can only add marketplaces that match exactly

**All supported source types** : The allowlist supports multiple marketplace source types. Most sources use exact matching, while `hostPattern` uses regex matching against the marketplace host.

1. **GitHub repositories** :

```
{ "source" : "github" , "repo" : "acme-corp/approved-plugins" }
{ "source" : "github" , "repo" : "acme-corp/security-tools" , "ref" : "v2.0" }
{ "source" : "github" , "repo" : "acme-corp/plugins" , "ref" : "main" , "path" : "marketplace" }
```

Fields: `repo` (required), `ref` (optional: branch/tag/SHA), `path` (optional: subdirectory)

2. **Git repositories** :

`{ "source" : "git" , "url" : "https://gitlab.example.com/tools/plugins.git" }
{ "source" : "git" , "url" : "https://bitbucket.org/acme-corp/plugins.git" , "ref" : "production" }
{ "source" : "git" , "url" : "ssh://` [`[email protected]`](/cdn-cgi/l/email-protection) `/plugins.git" , "ref" : "v3.1" , "path" : "approved" }`

Fields: `url` (required), `ref` (optional: branch/tag/SHA), `path` (optional: subdirectory)

3. **URL-based marketplaces** :

```
{ "source" : "url" , "url" : "https://plugins.example.com/marketplace.json" }
{ "source" : "url" , "url" : "https://cdn.example.com/marketplace.json" , "headers" : { "Authorization" : "Bearer ${TOKEN}" } }
```

Fields: `url` (required), `headers` (optional: HTTP headers for authenticated access)

URL-based marketplaces only download the `marketplace.json` file. They do not download plugin files from the server. Plugins in URL-based marketplaces must use external sources (GitHub, npm, or git URLs) rather than relative paths. For plugins with relative paths, use a Git-based marketplace instead. See [Troubleshooting](/docs/en/plugin-marketplaces#plugins-with-relative-paths-fail-in-url-based-marketplaces) for details.

4. **NPM packages** :

```
{ "source" : "npm" , "package" : "@acme-corp/claude-plugins" }
{ "source" : "npm" , "package" : "@acme-corp/approved-marketplace" }
```

Fields: `package` (required, supports scoped packages)

5. **File paths** :

```
{ "source" : "file" , "path" : "/usr/local/share/claude/acme-marketplace.json" }
{ "source" : "file" , "path" : "/opt/acme-corp/plugins/marketplace.json" }
```

Fields: `path` (required: absolute path to marketplace.json file)

6. **Directory paths** :

```
{ "source" : "directory" , "path" : "/usr/local/share/claude/acme-plugins" }
{ "source" : "directory" , "path" : "/opt/acme-corp/approved-marketplaces" }
```

Fields: `path` (required: absolute path to directory containing `.claude-plugin/marketplace.json` )

7. **Host pattern matching** :

```
{ "source" : "hostPattern" , "hostPattern" : "^github \\ .example \\ .com$" }
{ "source" : "hostPattern" , "hostPattern" : "^gitlab \\ .internal \\ .example \\ .com$" }
```

Fields: `hostPattern` (required: regex pattern to match against the marketplace host) Use host pattern matching when you want to allow all marketplaces from a specific host without enumerating each repository individually. This is useful for organizations with internal GitHub Enterprise or GitLab servers where developers create their own marketplaces. Host extraction by source type:

- `github` : always matches against `github.com`
- `git` : extracts hostname from the URL (supports both HTTPS and SSH formats)
- `url` : extracts hostname from the URL
- `npm` , `file` , `directory` : not supported for host pattern matching

**Configuration examples** : Example: allow specific marketplaces only:

```
{
"strictKnownMarketplaces" : [
{
"source" : "github" ,
"repo" : "acme-corp/approved-plugins"
},
{
"source" : "github" ,
"repo" : "acme-corp/security-tools" ,
"ref" : "v2.0"
},
{
"source" : "url" ,
"url" : "https://plugins.example.com/marketplace.json"
},
{
"source" : "npm" ,
"package" : "@acme-corp/compliance-plugins"
}
]
}
```

Example - Disable all marketplace additions:

```
{
"strictKnownMarketplaces" : []
}
```

Example: allow all marketplaces from an internal git server:

```
{
"strictKnownMarketplaces" : [
{
"source" : "hostPattern" ,
"hostPattern" : "^github \\ .example \\ .com$"
}
]
}
```

**Exact matching requirements** : Marketplace sources must match **exactly** for a user's addition to be allowed. For git-based sources ( `github` and `git` ), this includes all optional fields:

- The `repo` or `url` must match exactly
- The `ref` field must match exactly (or both be undefined)
- The `path` field must match exactly (or both be undefined)

Examples of sources that **do NOT match** :

```
// These are DIFFERENT sources:
{ "source" : "github" , "repo" : "acme-corp/plugins" }
{ "source" : "github" , "repo" : "acme-corp/plugins" , "ref" : "main" }

// These are also DIFFERENT:
{ "source" : "github" , "repo" : "acme-corp/plugins" , "path" : "marketplace" }
{ "source" : "github" , "repo" : "acme-corp/plugins" }
```

**Comparison with** **`extraKnownMarketplaces`** :

| Aspect                | `strictKnownMarketplaces`            | `extraKnownMarketplaces`             |
|-----------------------|--------------------------------------|--------------------------------------|
| **Purpose**           | Organizational policy enforcement    | Team convenience                     |
| **Settings file**     | `managed-settings.json` only         | Any settings file                    |
| **Behavior**          | Blocks non-allowlisted additions     | Auto-installs missing marketplaces   |
| **When enforced**     | Before network/filesystem operations | After user trust prompt              |
| **Can be overridden** | No (highest precedence)              | Yes (by higher precedence settings)  |
| **Source format**     | Direct source object                 | Named marketplace with nested source |
| **Use case**          | Compliance, security restrictions    | Onboarding, standardization          |

**Format difference** : `strictKnownMarketplaces` uses direct source objects:

```
{
"strictKnownMarketplaces" : [
{ "source" : "github" , "repo" : "acme-corp/plugins" }
]
}
```

`extraKnownMarketplaces` requires named marketplaces:

```
{
"extraKnownMarketplaces" : {
"acme-tools" : {
"source" : { "source" : "github" , "repo" : "acme-corp/plugins" }
}
}
}
```

**Using both together** : `strictKnownMarketplaces` is a policy gate: it controls what users may add but does not register any marketplaces. To both restrict and pre-register a marketplace for all users, set both in `managed-settings.json` :

```
{
"strictKnownMarketplaces" : [
{ "source" : "github" , "repo" : "acme-corp/plugins" }
],
"extraKnownMarketplaces" : {
"acme-tools" : {
"source" : { "source" : "github" , "repo" : "acme-corp/plugins" }
}
}
}
```

With only `strictKnownMarketplaces` set, users can still add the allowed marketplace manually via `/plugin marketplace add` , but it is not available automatically. **Important notes** :

- Restrictions are checked BEFORE any network requests or filesystem operations
- When blocked, users see clear error messages indicating the source is blocked by managed policy
- The restriction applies only to adding NEW marketplaces; previously installed marketplaces remain accessible
- Managed settings have the highest precedence and cannot be overridden

See [Managed marketplace restrictions](/docs/en/plugin-marketplaces#managed-marketplace-restrictions) for user-facing documentation.

#### Managing plugins

Use the `/plugin` command to manage plugins interactively:

- Browse available plugins from marketplaces
- Install/uninstall plugins
- Enable/disable plugins
- View plugin details (skills, agents, hooks provided)
- Add/remove marketplaces

Learn more about the plugin system in the [plugins documentation](/docs/en/plugins) .

### Environment variables

Environment variables let you control Claude Code behavior without editing settings files. Any variable can also be configured in [`settings.json`](#available-settings) under the `env` key to apply it to every session or roll it out to your team. See the [environment variables reference](/docs/en/env-vars) for the full list.

### Tools available to Claude

Claude Code has access to a set of tools for reading, editing, searching, running commands, and orchestrating subagents. Tool names are the exact strings you use in permission rules and hook matchers. See the [tools reference](/docs/en/tools-reference) for the full list and Bash tool behavior details.

### See also

- [Permissions](/docs/en/permissions) : permission system, rule syntax, tool-specific patterns, and managed policies
- [Authentication](/docs/en/authentication) : set up user access to Claude Code
- [Troubleshooting](/docs/en/troubleshooting) : solutions for common configuration issues

Was this page helpful?

Yes

No

[Permissions](/docs/en/permissions)

⌘ I


### CLI reference


Complete reference for Claude Code command-line interface, including commands and flags.


### CLI commands

You can start sessions, pipe content, resume conversations, and manage updates with these commands:

| Command                             | Description                                                                                                                                                                                                                                           | Example                                                     |
|-------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| `claude`                            | Start interactive session                                                                                                                                                                                                                             | `claude`                                                    |
| `claude "query"`                    | Start interactive session with initial prompt                                                                                                                                                                                                         | `claude "explain this project"`                             |
| `claude -p "query"`                 | Query via SDK, then exit                                                                                                                                                                                                                              | `claude -p "explain this function"`                         |
| `cat file | claude -p "query"` | Process piped content                                                                                                                                                                                                                                 | `cat logs.txt | claude -p "explain"`                   |
| `claude -c`                         | Continue most recent conversation in current directory                                                                                                                                                                                                | `claude -c`                                                 |
| `claude -c -p "query"`              | Continue via SDK                                                                                                                                                                                                                                      | `claude -c -p "Check for type errors"`                      |
| `claude -r "<session>" "query"`     | Resume session by ID or name                                                                                                                                                                                                                          | `claude -r "auth-refactor" "Finish this PR"`                |
| `claude update`                     | Update to latest version                                                                                                                                                                                                                              | `claude update`                                             |
| `claude auth login`                 | Sign in to your Anthropic account. Use `--email` to pre-fill your email address, `--sso` to force SSO authentication, and `--console` to sign in with Anthropic Console for API usage billing instead of a Claude subscription                        | `claude auth login --console`                               |
| `claude auth logout`                | Log out from your Anthropic account                                                                                                                                                                                                                   | `claude auth logout`                                        |
| `claude auth status`                | Show authentication status as JSON. Use `--text` for human-readable output. Exits with code 0 if logged in, 1 if not                                                                                                                                  | `claude auth status`                                        |
| `claude agents`                     | List all configured [subagents](/docs/en/sub-agents) , grouped by source                                                                                                                                                                              | `claude agents`                                             |
| `claude auto-mode defaults`         | Print the built-in [auto mode](/docs/en/permission-modes#eliminate-prompts-with-auto-mode) classifier rules as JSON. Use `claude auto-mode config` to see your effective config with settings applied                                                 | `claude auto-mode defaults > rules.json`                    |
| `claude mcp`                        | Configure Model Context Protocol (MCP) servers                                                                                                                                                                                                        | See the [Claude Code MCP documentation](/docs/en/mcp) .     |
| `claude plugin`                     | Manage Claude Code [plugins](/docs/en/plugins) . Alias: `claude plugins` . See [plugin reference](/docs/en/plugins-reference#cli-commands-reference) for subcommands                                                                                  | `claude plugin install code-review@claude-plugins-official` |
| `claude remote-control`             | Start a [Remote Control](/docs/en/remote-control) server to control Claude Code from Claude.ai or the Claude app. Runs in server mode (no local interactive session). See [Server mode flags](/docs/en/remote-control#start-a-remote-control-session) | `claude remote-control --name "My Project"`                 |
| `claude setup-token`                | Generate a long-lived OAuth token for CI and scripts. Prints the token to the terminal without saving it. Requires a Claude subscription. See [Generate a long-lived token](/docs/en/authentication#generate-a-long-lived-token)                      | `claude setup-token`                                        |

### CLI flags

Customize Claude Code's behavior with these command-line flags. `claude --help` does not list every flag, so a flag's absence from `--help` does not mean it is unavailable.

| Flag                                            | Description                                                                                                                                                                                                                                                                                                                                                                                         | Example                                                                                            |
|-------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| `--add-dir`                                     | Add additional working directories for Claude to read and edit files. Grants file access; most `.claude/` configuration is [not discovered](/docs/en/permissions#additional-directories-grant-file-access-not-configuration) from these directories. Validates each path exists as a directory                                                                                                      | `claude --add-dir ../apps ../lib`                                                                  |
| `--agent`                                       | Specify an agent for the current session (overrides the `agent` setting)                                                                                                                                                                                                                                                                                                                            | `claude --agent my-custom-agent`                                                                   |
| `--agents`                                      | Define custom subagents dynamically via JSON. Uses the same field names as subagent [frontmatter](/docs/en/sub-agents#supported-frontmatter-fields) , plus a `prompt` field for the agent's instructions                                                                                                                                                                                            | `claude --agents '{"reviewer":{"description":"Reviews code","prompt":"You are a code reviewer"}}'` |
| `--allow-dangerously-skip-permissions`          | Add `bypassPermissions` to the `Shift+Tab` mode cycle without starting in it. Lets you begin in a different mode like `plan` and switch to `bypassPermissions` later. See [permission modes](/docs/en/permission-modes#skip-all-checks-with-bypasspermissions-mode)                                                                                                                                 | `claude --permission-mode plan --allow-dangerously-skip-permissions`                               |
| `--allowedTools`                                | Tools that execute without prompting for permission. See [permission rule syntax](/docs/en/settings#permission-rule-syntax) for pattern matching. To restrict which tools are available, use `--tools` instead                                                                                                                                                                                      | `"Bash(git log *)" "Bash(git diff *)" "Read"`                                                      |
| `--append-system-prompt`                        | Append custom text to the end of the default system prompt                                                                                                                                                                                                                                                                                                                                          | `claude --append-system-prompt "Always use TypeScript"`                                            |
| `--append-system-prompt-file`                   | Load additional system prompt text from a file and append to the default prompt                                                                                                                                                                                                                                                                                                                     | `claude --append-system-prompt-file ./extra-rules.txt`                                             |
| `--bare`                                        | Minimal mode: skip auto-discovery of hooks, skills, plugins, MCP servers, auto memory, and CLAUDE.md so scripted calls start faster. Claude has access to Bash, file read, and file edit tools. Sets [`CLAUDE_CODE_SIMPLE`](/docs/en/env-vars) . See [bare mode](/docs/en/headless#start-faster-with-bare-mode)                                                                                     | `claude --bare -p "query"`                                                                         |
| `--betas`                                       | Beta headers to include in API requests (API key users only)                                                                                                                                                                                                                                                                                                                                        | `claude --betas interleaved-thinking`                                                              |
| `--channels`                                    | (Research preview) MCP servers whose [channel](/docs/en/channels) notifications Claude should listen for in this session. Space-separated list of `plugin:<name>@<marketplace>` entries. Requires Claude.ai authentication                                                                                                                                                                          | `claude --channels plugin:my-notifier@my-marketplace`                                              |
| `--chrome`                                      | Enable [Chrome browser integration](/docs/en/chrome) for web automation and testing                                                                                                                                                                                                                                                                                                                 | `claude --chrome`                                                                                  |
| `--continue` , `-c`                             | Load the most recent conversation in the current directory                                                                                                                                                                                                                                                                                                                                          | `claude --continue`                                                                                |
| `--dangerously-load-development-channels`       | Enable [channels](/docs/en/channels-reference#test-during-the-research-preview) that are not on the approved allowlist, for local development. Accepts `plugin:<name>@<marketplace>` and `server:<name>` entries. Prompts for confirmation                                                                                                                                                          | `claude --dangerously-load-development-channels server:webhook`                                    |
| `--dangerously-skip-permissions`                | Skip permission prompts. Equivalent to `--permission-mode bypassPermissions` . See [permission modes](/docs/en/permission-modes#skip-all-checks-with-bypasspermissions-mode) for what this does and does not skip                                                                                                                                                                                   | `claude --dangerously-skip-permissions`                                                            |
| `--debug`                                       | Enable debug mode with optional category filtering (for example, `"api,hooks"` or `"!statsig,!file"` )                                                                                                                                                                                                                                                                                              | `claude --debug "api,mcp"`                                                                         |
| `--debug-file <path>`                           | Write debug logs to a specific file path. Implicitly enables debug mode. Takes precedence over `CLAUDE_CODE_DEBUG_LOGS_DIR`                                                                                                                                                                                                                                                                         | `claude --debug-file /tmp/claude-debug.log`                                                        |
| `--disable-slash-commands`                      | Disable all skills and commands for this session                                                                                                                                                                                                                                                                                                                                                    | `claude --disable-slash-commands`                                                                  |
| `--disallowedTools`                             | Tools that are removed from the model's context and cannot be used                                                                                                                                                                                                                                                                                                                                  | `"Bash(git log *)" "Bash(git diff *)" "Edit"`                                                      |
| `--effort`                                      | Set the [effort level](/docs/en/model-config#adjust-effort-level) for the current session. Options: `low` , `medium` , `high` , `max` (Opus 4.6 only). Session-scoped and does not persist to settings                                                                                                                                                                                              | `claude --effort high`                                                                             |
| `--exclude-dynamic-system-prompt-sections`      | Move per-machine sections from the system prompt (working directory, environment info, memory paths, git status) into the first user message. Improves prompt-cache reuse across different users and machines running the same task. Only applies with the default system prompt; ignored when `--system-prompt` or `--system-prompt-file` is set. Use with `-p` for scripted, multi-user workloads | `claude -p --exclude-dynamic-system-prompt-sections "query"`                                       |
| `--fallback-model`                              | Enable automatic fallback to specified model when default model is overloaded (print mode only)                                                                                                                                                                                                                                                                                                     | `claude -p --fallback-model sonnet "query"`                                                        |
| `--fork-session`                                | When resuming, create a new session ID instead of reusing the original (use with `--resume` or `--continue` )                                                                                                                                                                                                                                                                                       | `claude --resume abc123 --fork-session`                                                            |
| `--from-pr`                                     | Resume sessions linked to a specific GitHub PR. Accepts a PR number or URL. Sessions are automatically linked when created via `gh pr create`                                                                                                                                                                                                                                                       | `claude --from-pr 123`                                                                             |
| `--ide`                                         | Automatically connect to IDE on startup if exactly one valid IDE is available                                                                                                                                                                                                                                                                                                                       | `claude --ide`                                                                                     |
| `--init`                                        | Run initialization hooks and start interactive mode                                                                                                                                                                                                                                                                                                                                                 | `claude --init`                                                                                    |
| `--init-only`                                   | Run initialization hooks and exit (no interactive session)                                                                                                                                                                                                                                                                                                                                          | `claude --init-only`                                                                               |
| `--include-hook-events`                         | Include all hook lifecycle events in the output stream. Requires `--output-format stream-json`                                                                                                                                                                                                                                                                                                      | `claude -p --output-format stream-json --include-hook-events "query"`                              |
| `--include-partial-messages`                    | Include partial streaming events in output. Requires `--print` and `--output-format stream-json`                                                                                                                                                                                                                                                                                                    | `claude -p --output-format stream-json --include-partial-messages "query"`                         |
| `--input-format`                                | Specify input format for print mode (options: `text` , `stream-json` )                                                                                                                                                                                                                                                                                                                              | `claude -p --output-format json --input-format stream-json`                                        |
| `--json-schema`                                 | Get validated JSON output matching a JSON Schema after agent completes its workflow (print mode only, see [structured outputs](/docs/en/agent-sdk/structured-outputs) )                                                                                                                                                                                                                             | `claude -p --json-schema '{"type":"object","properties":{...}}' "query"`                           |
| `--maintenance`                                 | Run maintenance hooks and start interactive mode                                                                                                                                                                                                                                                                                                                                                    | `claude --maintenance`                                                                             |
| `--max-budget-usd`                              | Maximum dollar amount to spend on API calls before stopping (print mode only)                                                                                                                                                                                                                                                                                                                       | `claude -p --max-budget-usd 5.00 "query"`                                                          |
| `--max-turns`                                   | Limit the number of agentic turns (print mode only). Exits with an error when the limit is reached. No limit by default                                                                                                                                                                                                                                                                             | `claude -p --max-turns 3 "query"`                                                                  |
| `--mcp-config`                                  | Load MCP servers from JSON files or strings (space-separated)                                                                                                                                                                                                                                                                                                                                       | `claude --mcp-config ./mcp.json`                                                                   |
| `--model`                                       | Sets the model for the current session with an alias for the latest model ( `sonnet` or `opus` ) or a model's full name                                                                                                                                                                                                                                                                             | `claude --model claude-sonnet-4-6`                                                                 |
| `--name` , `-n`                                 | Set a display name for the session, shown in `/resume` and the terminal title. You can resume a named session with `claude --resume <name>` .  [`/rename`](/docs/en/commands) changes the name mid-session and also shows it on the prompt bar                                                                                                                                                      | `claude -n "my-feature-work"`                                                                      |
| `--no-chrome`                                   | Disable [Chrome browser integration](/docs/en/chrome) for this session                                                                                                                                                                                                                                                                                                                              | `claude --no-chrome`                                                                               |
| `--no-session-persistence`                      | Disable session persistence so sessions are not saved to disk and cannot be resumed (print mode only)                                                                                                                                                                                                                                                                                               | `claude -p --no-session-persistence "query"`                                                       |
| `--output-format`                               | Specify output format for print mode (options: `text` , `json` , `stream-json` )                                                                                                                                                                                                                                                                                                                    | `claude -p "query" --output-format json`                                                           |
| `--enable-auto-mode`                            | Unlock [auto mode](/docs/en/permission-modes#eliminate-prompts-with-auto-mode) in the `Shift+Tab` cycle. Requires a Team, Enterprise, or API plan and Claude Sonnet 4.6 or Opus 4.6                                                                                                                                                                                                                 | `claude --enable-auto-mode`                                                                        |
| `--permission-mode`                             | Begin in a specified [permission mode](/docs/en/permission-modes) . Accepts `default` , `acceptEdits` , `plan` , `auto` , `dontAsk` , or `bypassPermissions` . Overrides `defaultMode` from settings files                                                                                                                                                                                          | `claude --permission-mode plan`                                                                    |
| `--permission-prompt-tool`                      | Specify an MCP tool to handle permission prompts in non-interactive mode                                                                                                                                                                                                                                                                                                                            | `claude -p --permission-prompt-tool mcp_auth_tool "query"`                                         |
| `--plugin-dir`                                  | Load plugins from a directory for this session only. Each flag takes one path. Repeat the flag for multiple directories: `--plugin-dir A --plugin-dir B`                                                                                                                                                                                                                                            | `claude --plugin-dir ./my-plugins`                                                                 |
| `--print` , `-p`                                | Print response without interactive mode (see [Agent SDK documentation](/docs/en/agent-sdk/overview) for programmatic usage details)                                                                                                                                                                                                                                                                 | `claude -p "query"`                                                                                |
| `--remote`                                      | Create a new [web session](/docs/en/claude-code-on-the-web) on claude.ai with the provided task description                                                                                                                                                                                                                                                                                         | `claude --remote "Fix the login bug"`                                                              |
| `--remote-control` , `--rc`                     | Start an interactive session with [Remote Control](/docs/en/remote-control#start-a-remote-control-session) enabled so you can also control it from claude.ai or the Claude app. Optionally pass a name for the session                                                                                                                                                                              | `claude --remote-control "My Project"`                                                             |
| `--remote-control-session-name-prefix <prefix>` | Prefix for auto-generated [Remote Control](/docs/en/remote-control) session names when no explicit name is set. Defaults to your machine's hostname, producing names like `myhost-graceful-unicorn` . Set `CLAUDE_REMOTE_CONTROL_SESSION_NAME_PREFIX` for the same effect                                                                                                                           | `claude remote-control --remote-control-session-name-prefix dev-box`                               |
| `--replay-user-messages`                        | Re-emit user messages from stdin back on stdout for acknowledgment. Requires `--input-format stream-json` and `--output-format stream-json`                                                                                                                                                                                                                                                         | `claude -p --input-format stream-json --output-format stream-json --replay-user-messages`          |
| `--resume` , `-r`                               | Resume a specific session by ID or name, or show an interactive picker to choose a session                                                                                                                                                                                                                                                                                                          | `claude --resume auth-refactor`                                                                    |
| `--session-id`                                  | Use a specific session ID for the conversation (must be a valid UUID)                                                                                                                                                                                                                                                                                                                               | `claude --session-id "550e8400-e29b-41d4-a716-446655440000"`                                       |
| `--setting-sources`                             | Comma-separated list of setting sources to load ( `user` , `project` , `local` )                                                                                                                                                                                                                                                                                                                    | `claude --setting-sources user,project`                                                            |
| `--settings`                                    | Path to a settings JSON file or a JSON string to load additional settings from                                                                                                                                                                                                                                                                                                                      | `claude --settings ./settings.json`                                                                |
| `--strict-mcp-config`                           | Only use MCP servers from `--mcp-config` , ignoring all other MCP configurations                                                                                                                                                                                                                                                                                                                    | `claude --strict-mcp-config --mcp-config ./mcp.json`                                               |
| `--system-prompt`                               | Replace the entire system prompt with custom text                                                                                                                                                                                                                                                                                                                                                   | `claude --system-prompt "You are a Python expert"`                                                 |
| `--system-prompt-file`                          | Load system prompt from a file, replacing the default prompt                                                                                                                                                                                                                                                                                                                                        | `claude --system-prompt-file ./custom-prompt.txt`                                                  |
| `--teleport`                                    | Resume a [web session](/docs/en/claude-code-on-the-web) in your local terminal                                                                                                                                                                                                                                                                                                                      | `claude --teleport`                                                                                |
| `--teammate-mode`                               | Set how [agent team](/docs/en/agent-teams) teammates display: `auto` (default), `in-process` , or `tmux` . See [Choose a display mode](/docs/en/agent-teams#choose-a-display-mode)                                                                                                                                                                                                                  | `claude --teammate-mode in-process`                                                                |
| `--tmux`                                        | Create a tmux session for the worktree. Requires `--worktree` . Uses iTerm2 native panes when available; pass `--tmux=classic` for traditional tmux                                                                                                                                                                                                                                                 | `claude -w feature-auth --tmux`                                                                    |
| `--tools`                                       | Restrict which built-in tools Claude can use. Use `""` to disable all, `"default"` for all, or tool names like `"Bash,Edit,Read"`                                                                                                                                                                                                                                                                   | `claude --tools "Bash,Edit,Read"`                                                                  |
| `--verbose`                                     | Enable verbose logging, shows full turn-by-turn output                                                                                                                                                                                                                                                                                                                                              | `claude --verbose`                                                                                 |
| `--version` , `-v`                              | Output the version number                                                                                                                                                                                                                                                                                                                                                                           | `claude -v`                                                                                        |
| `--worktree` , `-w`                             | Start Claude in an isolated [git worktree](/docs/en/common-workflows#run-parallel-claude-code-sessions-with-git-worktrees) at `<repo>/.claude/worktrees/<name>` . If no name is given, one is auto-generated                                                                                                                                                                                        | `claude -w feature-auth`                                                                           |

#### System prompt flags

Claude Code provides four flags for customizing the system prompt. All four work in both interactive and non-interactive modes.

| Flag                          | Behavior                                    | Example                                                 |
|-------------------------------|---------------------------------------------|---------------------------------------------------------|
| `--system-prompt`             | Replaces the entire default prompt          | `claude --system-prompt "You are a Python expert"`      |
| `--system-prompt-file`        | Replaces with file contents                 | `claude --system-prompt-file ./prompts/review.txt`      |
| `--append-system-prompt`      | Appends to the default prompt               | `claude --append-system-prompt "Always use TypeScript"` |
| `--append-system-prompt-file` | Appends file contents to the default prompt | `claude --append-system-prompt-file ./style-rules.txt`  |

`--system-prompt` and `--system-prompt-file` are mutually exclusive. The append flags can be combined with either replacement flag. For most use cases, use an append flag. Appending preserves Claude Code's built-in capabilities while adding your requirements. Use a replacement flag only when you need complete control over the system prompt.

### See also

- [Chrome extension](/docs/en/chrome) - Browser automation and web testing
- [Interactive mode](/docs/en/interactive-mode) - Shortcuts, input modes, and interactive features
- [Quickstart guide](/docs/en/quickstart) - Getting started with Claude Code
- [Common workflows](/docs/en/common-workflows) - Advanced workflows and patterns
- [Settings](/docs/en/settings) - Configuration options
- [Agent SDK documentation](/docs/en/agent-sdk/overview) - Programmatic usage and integrations

Was this page helpful?

Yes

No

[Commands](/docs/en/commands)

⌘ I


---

# Workflows & Best Practices


### Common workflows


Step-by-step guides for exploring codebases, fixing bugs, refactoring, testing, and other everyday tasks with Claude Code.


This page covers practical workflows for everyday development: exploring unfamiliar code, debugging, refactoring, writing tests, creating PRs, and managing sessions. Each section includes example prompts you can adapt to your own projects. For higher-level patterns and tips, see [Best practices](/docs/en/best-practices) .

### Understand new codebases

#### Get a quick codebase overview

Suppose you've just joined a new project and need to understand its structure quickly.

1

Navigate to the project root directory

```
cd /path/to/project
```

2

Start Claude Code

```
claude
```

3

Ask for a high-level overview

```
give me an overview of this codebase
```

4

Dive deeper into specific components

```
explain the main architecture patterns used here
```

```
what are the key data models?
```

```
how is authentication handled?
```

Tips:

- Start with broad questions, then narrow down to specific areas
- Ask about coding conventions and patterns used in the project
- Request a glossary of project-specific terms

#### Find relevant code

Suppose you need to locate code related to a specific feature or functionality.

1

Ask Claude to find relevant files

```
find the files that handle user authentication
```

2

Get context on how components interact

```
how do these authentication files work together?
```

3

Understand the execution flow

```
trace the login process from front-end to database
```

Tips:

- Be specific about what you're looking for
- Use domain language from the project
- Install a [code intelligence plugin](/docs/en/discover-plugins#code-intelligence) for your language to give Claude precise "go to definition" and "find references" navigation

### Fix bugs efficiently

Suppose you've encountered an error message and need to find and fix its source.

1

Share the error with Claude

```
I'm seeing an error when I run npm test
```

2

Ask for fix recommendations

```
suggest a few ways to fix the @ts-ignore in user.ts
```

3

Apply the fix

```
update user.ts to add the null check you suggested
```

Tips:

- Tell Claude the command to reproduce the issue and get a stack trace
- Mention any steps to reproduce the error
- Let Claude know if the error is intermittent or consistent

### Refactor code

Suppose you need to update old code to use modern patterns and practices.

1

Identify legacy code for refactoring

```
find deprecated API usage in our codebase
```

2

Get refactoring recommendations

```
suggest how to refactor utils.js to use modern JavaScript features
```

3

Apply the changes safely

```
refactor utils.js to use ES2024 features while maintaining the same behavior
```

4

Verify the refactoring

```
run tests for the refactored code
```

Tips:

- Ask Claude to explain the benefits of the modern approach
- Request that changes maintain backward compatibility when needed
- Do refactoring in small, testable increments

### Use specialized subagents

Suppose you want to use specialized AI subagents to handle specific tasks more effectively.

1

View available subagents

```
/agents
```

This shows all available subagents and lets you create new ones.

2

Use subagents automatically

Claude Code automatically delegates appropriate tasks to specialized subagents:

```
review my recent code changes for security issues
```

```
run all tests and fix any failures
```

3

Explicitly request specific subagents

```
use the code-reviewer subagent to check the auth module
```

```
have the debugger subagent investigate why users can't log in
```

4

Create custom subagents for your workflow

```
/agents
```

Then select "Create New subagent" and follow the prompts to define:

- A unique identifier that describes the subagent's purpose (for example, `code-reviewer` , `api-designer` ).
- When Claude should use this agent
- Which tools it can access
- A system prompt describing the agent's role and behavior

Tips:

- Create project-specific subagents in `.claude/agents/` for team sharing
- Use descriptive `description` fields to enable automatic delegation
- Limit tool access to what each subagent actually needs
- Check the [subagents documentation](/docs/en/sub-agents) for detailed examples

### Use Plan Mode for safe code analysis

Plan Mode instructs Claude to create a plan by analyzing the codebase with read-only operations, perfect for exploring codebases, planning complex changes, or reviewing code safely. In Plan Mode, Claude uses [`AskUserQuestion`](/docs/en/tools-reference) to gather requirements and clarify your goals before proposing a plan.

#### When to use Plan Mode

- **Multi-step implementation** : When your feature requires making edits to many files
- **Code exploration** : When you want to research the codebase thoroughly before changing anything
- **Interactive development** : When you want to iterate on the direction with Claude

#### How to use Plan Mode

**Turn on Plan Mode during a session** You can switch into Plan Mode during a session using **Shift+Tab** to cycle through permission modes. If you are in Normal Mode, **Shift+Tab** first switches into Auto-Accept Mode, indicated by `⏵⏵ accept edits on` at the bottom of the terminal. A subsequent **Shift+Tab** will switch into Plan Mode, indicated by `⏸ plan mode on` . **Start a new session in Plan Mode** To start a new session in Plan Mode, use the `--permission-mode plan` flag:

```
claude --permission-mode plan
```

**Run "headless" queries in Plan Mode** You can also run a query in Plan Mode directly with `-p` (that is, in ["headless mode"](/docs/en/headless) ):

```
claude --permission-mode plan -p "Analyze the authentication system and suggest improvements"
```

#### Example: Planning a complex refactor

```
claude --permission-mode plan
```

```
I need to refactor our authentication system to use OAuth2. Create a detailed migration plan.
```

Claude analyzes the current implementation and create a comprehensive plan. Refine with follow-ups:

```
What about backward compatibility?
```

```
How should we handle database migration?
```

Press `Ctrl+G` to open the plan in your default text editor, where you can edit it directly before Claude proceeds.

When you accept a plan, Claude automatically names the session from the plan content. The name appears on the prompt bar and in the session picker. If you've already set a name with `--name` or `/rename` , accepting a plan won't overwrite it.

#### Configure Plan Mode as default

```
// .claude/settings.json
{
"permissions" : {
"defaultMode" : "plan"
}
}
```

See [settings documentation](/docs/en/settings#available-settings) for more configuration options.

### Work with tests

Suppose you need to add tests for uncovered code.

1

Identify untested code

```
find functions in NotificationsService.swift that are not covered by tests
```

2

Generate test scaffolding

```
add tests for the notification service
```

3

Add meaningful test cases

```
add test cases for edge conditions in the notification service
```

4

Run and verify tests

```
run the new tests and fix any failures
```

Claude can generate tests that follow your project's existing patterns and conventions. When asking for tests, be specific about what behavior you want to verify. Claude examines your existing test files to match the style, frameworks, and assertion patterns already in use. For comprehensive coverage, ask Claude to identify edge cases you might have missed. Claude can analyze your code paths and suggest tests for error conditions, boundary values, and unexpected inputs that are easy to overlook.

### Create pull requests

You can create pull requests by asking Claude directly ("create a pr for my changes"), or guide Claude through it step-by-step:

1

Summarize your changes

```
summarize the changes I've made to the authentication module
```

2

Generate a pull request

```
create a pr
```

3

Review and refine

```
enhance the PR description with more context about the security improvements
```

When you create a PR using `gh pr create` , the session is automatically linked to that PR. You can resume it later with `claude --from-pr <number>` .

Review Claude's generated PR before submitting and ask Claude to highlight potential risks or considerations.

### Handle documentation

Suppose you need to add or update documentation for your code.

1

Identify undocumented code

```
find functions without proper JSDoc comments in the auth module
```

2

Generate documentation

```
add JSDoc comments to the undocumented functions in auth.js
```

3

Review and enhance

```
improve the generated documentation with more context and examples
```

4

Verify documentation

```
check if the documentation follows our project standards
```

Tips:

- Specify the documentation style you want (JSDoc, docstrings, etc.)
- Ask for examples in the documentation
- Request documentation for public APIs, interfaces, and complex logic

### Work with images

Suppose you need to work with images in your codebase, and you want Claude's help analyzing image content.

1

Add an image to the conversation

You can use any of these methods:

1. Drag and drop an image into the Claude Code window
2. Copy an image and paste it into the CLI with ctrl+v (Do not use cmd+v)
3. Provide an image path to Claude. E.g., "Analyze this image: /path/to/your/image.png"

2

Ask Claude to analyze the image

```
What does this image show?
```

```
Describe the UI elements in this screenshot
```

```
Are there any problematic elements in this diagram?
```

3

Use images for context

```
Here's a screenshot of the error. What's causing it?
```

```
This is our current database schema. How should we modify it for the new feature?
```

4

Get code suggestions from visual content

```
Generate CSS to match this design mockup
```

```
What HTML structure would recreate this component?
```

Tips:

- Use images when text descriptions would be unclear or cumbersome
- Include screenshots of errors, UI designs, or diagrams for better context
- You can work with multiple images in a conversation
- Image analysis works with diagrams, screenshots, mockups, and more
- When Claude references images (for example, `[Image #1]` ), `Cmd+Click` (Mac) or `Ctrl+Click` (Windows/Linux) the link to open the image in your default viewer

### Reference files and directories

Use @ to quickly include files or directories without waiting for Claude to read them.

1

Reference a single file

```
Explain the logic in @src/utils/auth.js
```

This includes the full content of the file in the conversation.

2

Reference a directory

```
What's the structure of @src/components?
```

This provides a directory listing with file information.

3

Reference MCP resources

```
Show me the data from @github:repos/owner/repo/issues
```

This fetches data from connected MCP servers using the format @server:resource. See [MCP resources](/docs/en/mcp#use-mcp-resources) for details.

Tips:

- File paths can be relative or absolute
- @ file references add `CLAUDE.md` in the file's directory and parent directories to context
- Directory references show file listings, not contents
- You can reference multiple files in a single message (for example, "@file1.js and @file2.js")

### Use extended thinking (thinking mode)

[Extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking) is enabled by default, giving Claude space to reason through complex problems step-by-step before responding. This reasoning is visible in verbose mode, which you can toggle on with `Ctrl+O` . Additionally, Opus 4.6 and Sonnet 4.6 support adaptive reasoning: instead of a fixed thinking token budget, the model dynamically allocates thinking based on your [effort level](/docs/en/model-config#adjust-effort-level) setting. Extended thinking and adaptive reasoning work together to give you control over how deeply Claude reasons before responding. Extended thinking is particularly valuable for complex architectural decisions, challenging bugs, multi-step implementation planning, and evaluating tradeoffs between different approaches.

Phrases like "think", "think hard", and "think more" are interpreted as regular prompt instructions and don't allocate thinking tokens.

#### Configure thinking mode

Thinking is enabled by default, but you can adjust or disable it.

| Scope                        | How to configure                                                                            | Details                                                                                                                                                                                   |
|------------------------------|---------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Effort level**             | Run `/effort` , adjust in `/model` , or set [`CLAUDE_CODE_EFFORT_LEVEL`](/docs/en/env-vars) | Control thinking depth for Opus 4.6 and Sonnet 4.6. See [Adjust effort level](/docs/en/model-config#adjust-effort-level)                                                                  |
| **`ultrathink`** **keyword** | Include "ultrathink" anywhere in your prompt                                                | Sets effort to high for that turn on Opus 4.6 and Sonnet 4.6. Useful for one-off tasks requiring deep reasoning without permanently changing your effort setting                          |
| **Toggle shortcut**          | Press `Option+T` (macOS) or `Alt+T` (Windows/Linux)                                         | Toggle thinking on/off for the current session (all models). May require [terminal configuration](/docs/en/terminal-config) to enable Option key shortcuts                                |
| **Global default**           | Use `/config` to toggle thinking mode                                                       | Sets your default across all projects (all models).  Saved as `alwaysThinkingEnabled` in `~/.claude/settings.json`                                                                        |
| **Limit token budget**       | Set [`MAX_THINKING_TOKENS`](/docs/en/env-vars) environment variable                         | Limit the thinking budget to a specific number of tokens. On Opus 4.6 and Sonnet 4.6, only `0` applies unless adaptive reasoning is disabled. Example: `export MAX_THINKING_TOKENS=10000` |

To view Claude's thinking process, press `Ctrl+O` to toggle verbose mode and see the internal reasoning displayed as gray italic text.

#### How extended thinking works

Extended thinking controls how much internal reasoning Claude performs before responding. More thinking provides more space to explore solutions, analyze edge cases, and self-correct mistakes. **With Opus 4.6 and Sonnet 4.6** , thinking uses adaptive reasoning: the model dynamically allocates thinking tokens based on the [effort level](/docs/en/model-config#adjust-effort-level) you select. This is the recommended way to tune the tradeoff between speed and reasoning depth. **With older models** , thinking uses a fixed token budget drawn from your output allocation. The budget varies by model; see [`MAX_THINKING_TOKENS`](/docs/en/env-vars) for per-model ceilings. You can limit the budget with that environment variable, or disable thinking entirely via `/config` or the `Option+T` / `Alt+T` toggle. On Opus 4.6 and Sonnet 4.6, [adaptive reasoning](/docs/en/model-config#adjust-effort-level) controls thinking depth, so `MAX_THINKING_TOKENS` only applies when set to `0` to disable thinking, or when `CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1` reverts these models to the fixed budget. See [environment variables](/docs/en/env-vars) .

You're charged for all thinking tokens used even when thinking summaries are redacted. In interactive mode, thinking appears as a collapsed stub by default. Set `showThinkingSummaries: true` in `settings.json` to show full summaries.

### Resume previous conversations

When starting Claude Code, you can resume a previous session:

- `claude --continue` continues the most recent conversation in the current directory
- `claude --resume` opens a conversation picker or resumes by name
- `claude --from-pr 123` resumes sessions linked to a specific pull request

From inside an active session, use `/resume` to switch to a different conversation. Sessions are stored per project directory. The `/resume` picker shows interactive sessions from the same git repository, including worktrees. When you select a session from another worktree of the same repository, Claude Code resumes it directly without requiring you to switch directories first. Sessions created by `claude -p` or SDK invocations do not appear in the picker, but you can still resume one by passing its session ID directly to `claude --resume <session-id>` .

#### Name your sessions

Give sessions descriptive names to find them later. This is a best practice when working on multiple tasks or features.

1

Name the session

Name a session at startup with `-n` :

```
claude -n auth-refactor
```

Or use `/rename` during a session, which also shows the name on the prompt bar:

```
/rename auth-refactor
```

You can also rename any session from the picker: run `/resume` , navigate to a session, and press `R` .

2

Resume by name later

From the command line:

```
claude --resume auth-refactor
```

Or from inside an active session:

```
/resume auth-refactor
```

#### Use the session picker

The `/resume` command (or `claude --resume` without arguments) opens an interactive session picker with these features: **Keyboard shortcuts in the picker:**

| Shortcut   | Action                                            |
|------------|---------------------------------------------------|
| `↑` / `↓`  | Navigate between sessions                         |
| `→` / `←`  | Expand or collapse grouped sessions               |
| `Enter`    | Select and resume the highlighted session         |
| `P`        | Preview the session content                       |
| `R`        | Rename the highlighted session                    |
| `/`        | Search to filter sessions                         |
| `A`        | Toggle between current directory and all projects |
| `B`        | Filter to sessions from your current git branch   |
| `Esc`      | Exit the picker or search mode                    |

**Session organization:** The picker displays sessions with helpful metadata:

- Session name or initial prompt
- Time elapsed since last activity
- Message count
- Git branch (if applicable)

Forked sessions (created with `/branch` , `/rewind` , or `--fork-session` ) are grouped together under their root session, making it easier to find related conversations.

Tips:

- **Name sessions early** : Use `/rename` when starting work on a distinct task: it's much easier to find "payment-integration" than "explain this function" later
- Use `--continue` for quick access to your most recent conversation in the current directory
- Use `--resume session-name` when you know which session you need
- Use `--resume` (without a name) when you need to browse and select
- For scripts, use `claude --continue --print "prompt"` to resume in non-interactive mode
- Press `P` in the picker to preview a session before resuming it
- The resumed conversation starts with the same model and configuration as the original

How it works:

1. **Conversation Storage** : All conversations are automatically saved locally with their full message history
2. **Message Deserialization** : When resuming, the entire message history is restored to maintain context
3. **Tool State** : Tool usage and results from the previous conversation are preserved
4. **Context Restoration** : The conversation resumes with all previous context intact

### Run parallel Claude Code sessions with Git worktrees

When working on multiple tasks at once, you need each Claude session to have its own copy of the codebase so changes don't collide. Git worktrees solve this by creating separate working directories that each have their own files and branch, while sharing the same repository history and remote connections. This means you can have Claude working on a feature in one worktree while fixing a bug in another, without either session interfering with the other. Use the `--worktree` ( `-w` ) flag to create an isolated worktree and start Claude in it. The value you pass becomes the worktree directory name and branch name:

```
### Start Claude in a worktree named "feature-auth"
### Creates .claude/worktrees/feature-auth/ with a new branch
claude --worktree feature-auth

### Start another session in a separate worktree
claude --worktree bugfix-123
```

If you omit the name, Claude generates a random one automatically:

```
### Auto-generates a name like "bright-running-fox"
claude --worktree
```

Worktrees are created at `<repo>/.claude/worktrees/<name>` and branch from the default remote branch, which is where `origin/HEAD` points. The worktree branch is named `worktree-<name>` . The base branch is not configurable through a Claude Code flag or setting. `origin/HEAD` is a reference stored in your local `.git` directory that Git set once when you cloned. If the repository's default branch later changes on GitHub or GitLab, your local `origin/HEAD` keeps pointing at the old one, and worktrees will branch from there. To re-sync your local reference with whatever the remote currently considers its default:

```
git remote set-head origin -a
```

This is a standard Git command that only updates your local `.git` directory. Nothing on the remote server changes. If you want worktrees to base off a specific branch rather than the remote's default, set it explicitly with `git remote set-head origin your-branch-name` . For full control over how worktrees are created, including choosing a different base per invocation, configure a [WorktreeCreate hook](/docs/en/hooks#worktreecreate) . The hook replaces Claude Code's default `git worktree` logic entirely, so you can fetch and branch from whatever ref you need. You can also ask Claude to "work in a worktree" or "start a worktree" during a session, and it will create one automatically.

#### Subagent worktrees

Subagents can also use worktree isolation to work in parallel without conflicts. Ask Claude to "use worktrees for your agents" or configure it in a [custom subagent](/docs/en/sub-agents#supported-frontmatter-fields) by adding `isolation: worktree` to the agent's frontmatter. Each subagent gets its own worktree that is automatically cleaned up when the subagent finishes without changes.

#### Worktree cleanup

When you exit a worktree session, Claude handles cleanup based on whether you made changes:

- **No changes** : the worktree and its branch are removed automatically
- **Changes or commits exist** : Claude prompts you to keep or remove the worktree. Keeping preserves the directory and branch so you can return later. Removing deletes the worktree directory and its branch, discarding all uncommitted changes and commits

Subagent worktrees orphaned by a crash or an interrupted parallel run are removed automatically at startup once they are older than your [`cleanupPeriodDays`](/docs/en/settings#available-settings) setting, provided they have no uncommitted changes, no untracked files, and no unpushed commits. Worktrees you create with `--worktree` are never removed by this sweep. To clean up worktrees outside of a Claude session, use [manual worktree management](#manage-worktrees-manually) .

Add `.claude/worktrees/` to your `.gitignore` to prevent worktree contents from appearing as untracked files in your main repository.

#### Copy gitignored files to worktrees

Git worktrees are fresh checkouts, so they don't include untracked files like `.env` or `.env.local` from your main repository. To automatically copy these files when Claude creates a worktree, add a `.worktreeinclude` file to your project root. The file uses `.gitignore` syntax to list which files to copy. Only files that match a pattern and are also gitignored get copied, so tracked files are never duplicated.

.worktreeinclude

```
.env
.env.local
config/secrets.json
```

This applies to worktrees created with `--worktree` , subagent worktrees, and parallel sessions in the [desktop app](/docs/en/desktop#work-in-parallel-with-sessions) .

#### Manage worktrees manually

For more control over worktree location and branch configuration, create worktrees with Git directly. This is useful when you need to check out a specific existing branch or place the worktree outside the repository.

```
### Create a worktree with a new branch
git worktree add ../project-feature-a -b feature-a

### Create a worktree with an existing branch
git worktree add ../project-bugfix bugfix-123

### Start Claude in the worktree
cd ../project-feature-a && claude

### Clean up when done
git worktree list
git worktree remove ../project-feature-a
```

Learn more in the [official Git worktree documentation](https://git-scm.com/docs/git-worktree) .

Remember to initialize your development environment in each new worktree according to your project's setup. Depending on your stack, this might include running dependency installation ( `npm install` , `yarn` ), setting up virtual environments, or following your project's standard setup process.

#### Non-git version control

Worktree isolation works with git by default. For other version control systems like SVN, Perforce, or Mercurial, configure [WorktreeCreate and WorktreeRemove hooks](/docs/en/hooks#worktreecreate) to provide custom worktree creation and cleanup logic. When configured, these hooks replace the default git behavior when you use `--worktree` , so [`.worktreeinclude`](#copy-gitignored-files-to-worktrees) is not processed. Copy any local configuration files inside your hook script instead. For automated coordination of parallel sessions with shared tasks and messaging, see [agent teams](/docs/en/agent-teams) .

### Get notified when Claude needs your attention

When you kick off a long-running task and switch to another window, you can set up desktop notifications so you know when Claude finishes or needs your input. This uses the `Notification` [hook event](/docs/en/hooks-guide#get-notified-when-claude-needs-input) , which fires whenever Claude is waiting for permission, idle and ready for a new prompt, or completing authentication.

1

Add the hook to your settings

Open `~/.claude/settings.json` and add a `Notification` hook that calls your platform's native notification command:

- macOS
- Linux
- Windows

```
{
"hooks" : {
"Notification" : [
{
"matcher" : "" ,
"hooks" : [
{
"type" : "command" ,
"command" : "osascript -e 'display notification \" Claude Code needs your attention \" with title \" Claude Code \" '"
}
]
}
]
}
}
```

```
{
"hooks" : {
"Notification" : [
{
"matcher" : "" ,
"hooks" : [
{
"type" : "command" ,
"command" : "notify-send 'Claude Code' 'Claude Code needs your attention'"
}
]
}
]
}
}
```

```
{
"hooks" : {
"Notification" : [
{
"matcher" : "" ,
"hooks" : [
{
"type" : "command" ,
"command" : "powershell.exe -Command \" [System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms'); [System.Windows.Forms.MessageBox]::Show('Claude Code needs your attention', 'Claude Code') \" "
}
]
}
]
}
}
```

If your settings file already has a `hooks` key, merge the `Notification` entry into it rather than overwriting. You can also ask Claude to write the hook for you by describing what you want in the CLI.

2

Optionally narrow the matcher

By default the hook fires on all notification types. To fire only for specific events, set the `matcher` field to one of these values:

| Matcher              | Fires when                                      |
|----------------------|-------------------------------------------------|
| `permission_prompt`  | Claude needs you to approve a tool use          |
| `idle_prompt`        | Claude is done and waiting for your next prompt |
| `auth_success`       | Authentication completes                        |
| `elicitation_dialog` | Claude is asking you a question                 |

3

Verify the hook

Type `/hooks` and select `Notification` to confirm the hook appears. Selecting it shows the command that will run. To test it end-to-end, ask Claude to run a command that requires permission and switch away from the terminal, or ask Claude to trigger a notification directly.

For the complete event schema and notification types, see the [Notification reference](/docs/en/hooks#notification) .

### Use Claude as a unix-style utility

#### Add Claude to your verification process

Suppose you want to use Claude Code as a linter or code reviewer. **Add Claude to your build script:**

```
// package.json
{
...
"scripts" : {
...
"lint:claude" : "claude -p 'you are a linter. please look at the changes vs. main and report any issues related to typos. report the filename and line number on one line, and a description of the issue on the second line. do not return any other text.'"
}
}
```

Tips:

- Use Claude for automated code review in your CI/CD pipeline
- Customize the prompt to check for specific issues relevant to your project
- Consider creating multiple scripts for different types of verification

#### Pipe in, pipe out

Suppose you want to pipe data into Claude, and get back data in a structured format. **Pipe data through Claude:**

```
cat build-error.txt | claude -p 'concisely explain the root cause of this build error' > output.txt
```

Tips:

- Use pipes to integrate Claude into existing shell scripts
- Combine with other Unix tools for powerful workflows
- Consider using `--output-format` for structured output

#### Control output format

Suppose you need Claude's output in a specific format, especially when integrating Claude Code into scripts or other tools.

1

Use text format (default)

```
cat data.txt | claude -p 'summarize this data' --output-format text > summary.txt
```

This outputs just Claude's plain text response (default behavior).

2

Use JSON format

```
cat code.py | claude -p 'analyze this code for bugs' --output-format json > analysis.json
```

This outputs a JSON array of messages with metadata including cost and duration.

3

Use streaming JSON format

```
cat log.txt | claude -p 'parse this log file for errors' --output-format stream-json
```

This outputs a series of JSON objects in real-time as Claude processes the request. Each message is a valid JSON object, but the entire output is not valid JSON if concatenated.

Tips:

- Use `--output-format text` for simple integrations where you just need Claude's response
- Use `--output-format json` when you need the full conversation log
- Use `--output-format stream-json` for real-time output of each conversation turn

### Run Claude on a schedule

Suppose you want Claude to handle a task automatically on a recurring basis, like reviewing open PRs every morning, auditing dependencies weekly, or checking for CI failures overnight. Pick a scheduling option based on where you want the task to run:

| Option                                                      | Where it runs                     | Best for                                                                                                      |
|-------------------------------------------------------------|-----------------------------------|---------------------------------------------------------------------------------------------------------------|
| [Cloud scheduled tasks](/docs/en/web-scheduled-tasks)       | Anthropic-managed infrastructure  | Tasks that should run even when your computer is off. Configure at [claude.ai/code](https://claude.ai/code) . |
| [Desktop scheduled tasks](/docs/en/desktop-scheduled-tasks) | Your machine, via the desktop app | Tasks that need direct access to local files, tools, or uncommitted changes.                                  |
| [GitHub Actions](/docs/en/github-actions)                   | Your CI pipeline                  | Tasks tied to repo events like opened PRs, or cron schedules that should live alongside your workflow config. |
| [`/loop`](/docs/en/scheduled-tasks)                         | The current CLI session           | Quick polling while a session is open. Tasks are cancelled when you exit.                                     |

When writing prompts for scheduled tasks, be explicit about what success looks like and what to do with results. The task runs autonomously, so it can't ask clarifying questions. For example: "Review open PRs labeled `needs-review` , leave inline comments on any issues, and post a summary in the `#eng-reviews` Slack channel."

### Ask Claude about its capabilities

Claude has built-in access to its documentation and can answer questions about its own features and limitations.

#### Example questions

```
can Claude Code create pull requests?
```

```
how does Claude Code handle permissions?
```

```
what skills are available?
```

```
how do I use MCP with Claude Code?
```

```
how do I configure Claude Code for Amazon Bedrock?
```

```
what are the limitations of Claude Code?
```

Claude provides documentation-based answers to these questions. For hands-on demonstrations, run `/powerup` for interactive lessons with animated demos, or refer to the specific workflow sections above.

Tips:

- Claude always has access to the latest Claude Code documentation, regardless of the version you're using
- Ask specific questions to get detailed answers
- Claude can explain complex features like MCP integration, enterprise configurations, and advanced workflows

### Next steps

### Best practices

Patterns for getting the most out of Claude Code

### How Claude Code works

Understand the agentic loop and context management

### Extend Claude Code

Add skills, hooks, MCP, subagents, and plugins

### Reference implementation

Clone the development container reference implementation

Was this page helpful?

Yes

No

[Permission modes](/docs/en/permission-modes) [Best practices](/docs/en/best-practices)

⌘ I


### Best Practices for Claude Code


Tips and patterns for getting the most out of Claude Code, from configuring your environment to scaling across parallel sessions.


Claude Code is an agentic coding environment. Unlike a chatbot that answers questions and waits, Claude Code can read your files, run commands, make changes, and autonomously work through problems while you watch, redirect, or step away entirely. This changes how you work. Instead of writing code yourself and asking Claude to review it, you describe what you want and Claude figures out how to build it. Claude explores, plans, and implements. But this autonomy still comes with a learning curve. Claude works within certain constraints you need to understand. This guide covers patterns that have proven effective across Anthropic's internal teams and for engineers using Claude Code across various codebases, languages, and environments. For how the agentic loop works under the hood, see [How Claude Code works](/docs/en/how-claude-code-works) .

Most best practices are based on one constraint: Claude's context window fills up fast, and performance degrades as it fills. Claude's context window holds your entire conversation, including every message, every file Claude reads, and every command output. However, this can fill up fast. A single debugging session or codebase exploration might generate and consume tens of thousands of tokens. This matters since LLM performance degrades as context fills. When the context window is getting full, Claude may start "forgetting" earlier instructions or making more mistakes. The context window is the most important resource to manage. To see how a session fills up in practice, [watch an interactive walkthrough](/docs/en/context-window) of what loads at startup and what each file read costs. Track context usage continuously with a [custom status line](/docs/en/statusline) , and see [Reduce token usage](/docs/en/costs#reduce-token-usage) for strategies on reducing token usage.

### Give Claude a way to verify its work

Include tests, screenshots, or expected outputs so Claude can check itself. This is the single highest-leverage thing you can do.

Claude performs dramatically better when it can verify its own work, like run tests, compare screenshots, and validate outputs. Without clear success criteria, it might produce something that looks right but actually doesn't work. You become the only feedback loop, and every mistake requires your attention.

| Strategy                              | Before                                                  | After                                                                                                                                                                                                                                                                                           |
|---------------------------------------|---------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Provide verification criteria**     | *"implement a function that validates email addresses"* | *"write a validateEmail function. example test cases:* [*[email protected]*](/cdn-cgi/l/email-protection#83f6f0e6f1c3e6fbe2eef3efe6ade0ecee) *is true, invalid is false,* [*[email protected]*](/cdn-cgi/l/email-protection#2d585e485f6d034e4240) *is false. run the tests after implementing"* |
| **Verify UI changes visually**        | *"make the dashboard look better"*                      | *"[paste screenshot] implement this design. take a screenshot of the result and compare it to the original. list differences and fix them"*                                                                                                                                                     |
| **Address root causes, not symptoms** | *"the build is failing"*                                | *"the build fails with this error: [paste error]. fix it and verify the build succeeds. address the root cause, don't suppress the error"*                                                                                                                                                      |

UI changes can be verified using the [Claude in Chrome extension](/docs/en/chrome) . It opens new tabs in your browser, tests the UI, and iterates until the code works. Your verification can also be a test suite, a linter, or a Bash command that checks output. Invest in making your verification rock-solid.

### Explore first, then plan, then code

Separate research and planning from implementation to avoid solving the wrong problem.

Letting Claude jump straight to coding can produce code that solves the wrong problem. Use [Plan Mode](/docs/en/common-workflows#use-plan-mode-for-safe-code-analysis) to separate exploration from execution. The recommended workflow has four phases:

1

Explore

Enter Plan Mode. Claude reads files and answers questions without making changes.

claude (Plan Mode)

```
read /src/auth and understand how we handle sessions and login.
also look at how we manage environment variables for secrets.
```

2

Plan

Ask Claude to create a detailed implementation plan.

claude (Plan Mode)

```
I want to add Google OAuth. What files need to change?
What's the session flow? Create a plan.
```

Press `Ctrl+G` to open the plan in your text editor for direct editing before Claude proceeds.

3

Implement

Switch back to Normal Mode and let Claude code, verifying against its plan.

claude (Normal Mode)

```
implement the OAuth flow from your plan. write tests for the
callback handler, run the test suite and fix any failures.
```

4

Commit

Ask Claude to commit with a descriptive message and create a PR.

claude (Normal Mode)

```
commit with a descriptive message and open a PR
```

Plan Mode is useful, but also adds overhead. For tasks where the scope is clear and the fix is small (like fixing a typo, adding a log line, or renaming a variable) ask Claude to do it directly. Planning is most useful when you're uncertain about the approach, when the change modifies multiple files, or when you're unfamiliar with the code being modified. If you could describe the diff in one sentence, skip the plan.

### Provide specific context in your prompts

The more precise your instructions, the fewer corrections you'll need.

Claude can infer intent, but it can't read your mind. Reference specific files, mention constraints, and point to example patterns.

| Strategy                                                                                         | Before                                               | After                                                                                                                                                                                                                                                                                                                                                            |
|--------------------------------------------------------------------------------------------------|------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Scope the task.** Specify which file, what scenario, and testing preferences.                  | *"add tests for foo.py"*                             | *"write a test for foo.py covering the edge case where the user is logged out. avoid mocks."*                                                                                                                                                                                                                                                                    |
| **Point to sources.** Direct Claude to the source that can answer a question.                    | *"why does ExecutionFactory have such a weird api?"* | *"look through ExecutionFactory's git history and summarize how its api came to be"*                                                                                                                                                                                                                                                                             |
| **Reference existing patterns.** Point Claude to patterns in your codebase.                      | *"add a calendar widget"*                            | *"look at how existing widgets are implemented on the home page to understand the patterns. HotDogWidget.php is a good example. follow the pattern to implement a new calendar widget that lets the user select a month and paginate forwards/backwards to pick a year. build from scratch without libraries other than the ones already used in the codebase."* |
| **Describe the symptom.** Provide the symptom, the likely location, and what "fixed" looks like. | *"fix the login bug"*                                | *"users report that login fails after session timeout. check the auth flow in src/auth/, especially token refresh. write a failing test that reproduces the issue, then fix it"*                                                                                                                                                                                 |

Vague prompts can be useful when you're exploring and can afford to course-correct. A prompt like `"what would you improve in this file?"` can surface things you wouldn't have thought to ask about.

#### Provide rich content

Use `@` to reference files, paste screenshots/images, or pipe data directly.

You can provide rich data to Claude in several ways:

- **Reference files with** **`@`** instead of describing where code lives. Claude reads the file before responding.
- **Paste images directly** . Copy/paste or drag and drop images into the prompt.
- **Give URLs** for documentation and API references. Use `/permissions` to allowlist frequently-used domains.
- **Pipe in data** by running `cat error.log | claude` to send file contents directly.
- **Let Claude fetch what it needs** . Tell Claude to pull context itself using Bash commands, MCP tools, or by reading files.

### Configure your environment

A few setup steps make Claude Code significantly more effective across all your sessions. For a full overview of extension features and when to use each one, see [Extend Claude Code](/docs/en/features-overview) .

#### Write an effective CLAUDE.md

Run `/init` to generate a starter CLAUDE.md file based on your current project structure, then refine over time.

CLAUDE.md is a special file that Claude reads at the start of every conversation. Include Bash commands, code style, and workflow rules. This gives Claude persistent context it can't infer from code alone. The `/init` command analyzes your codebase to detect build systems, test frameworks, and code patterns, giving you a solid foundation to refine. There's no required format for CLAUDE.md files, but keep it short and human-readable. For example:

CLAUDE.md

```
### Code style
- Use ES modules (import/export) syntax, not CommonJS (require)
- Destructure imports when possible (eg. import { foo } from 'bar')

### Workflow
- Be sure to typecheck when you're done making a series of code changes
- Prefer running single tests, and not the whole test suite, for performance
```

CLAUDE.md is loaded every session, so only include things that apply broadly. For domain knowledge or workflows that are only relevant sometimes, use [skills](/docs/en/skills) instead. Claude loads them on demand without bloating every conversation. Keep it concise. For each line, ask: *"Would removing this cause Claude to make mistakes?"* If not, cut it. Bloated CLAUDE.md files cause Claude to ignore your actual instructions!

| ✅ Include                                            | ❌ Exclude                                          |
|------------------------------------------------------|----------------------------------------------------|
| Bash commands Claude can't guess                     | Anything Claude can figure out by reading code     |
| Code style rules that differ from defaults           | Standard language conventions Claude already knows |
| Testing instructions and preferred test runners      | Detailed API documentation (link to docs instead)  |
| Repository etiquette (branch naming, PR conventions) | Information that changes frequently                |
| Architectural decisions specific to your project     | Long explanations or tutorials                     |
| Developer environment quirks (required env vars)     | File-by-file descriptions of the codebase          |
| Common gotchas or non-obvious behaviors              | Self-evident practices like "write clean code"     |

If Claude keeps doing something you don't want despite having a rule against it, the file is probably too long and the rule is getting lost. If Claude asks you questions that are answered in CLAUDE.md, the phrasing might be ambiguous. Treat CLAUDE.md like code: review it when things go wrong, prune it regularly, and test changes by observing whether Claude's behavior actually shifts. You can tune instructions by adding emphasis (e.g., "IMPORTANT" or "YOU MUST") to improve adherence. Check CLAUDE.md into git so your team can contribute. The file compounds in value over time. CLAUDE.md files can import additional files using `@path/to/import` syntax:

CLAUDE.md

```
See @README.md for project overview and @package.json for available npm commands.

### Additional Instructions
- Git workflow: @docs/git-instructions.md
- Personal overrides: @~/.claude/my-project-instructions.md
```

You can place CLAUDE.md files in several locations:

- **Home folder (** **`~/.claude/CLAUDE.md`** **)** : applies to all Claude sessions
- **Project root (** **`./CLAUDE.md`** **)** : check into git to share with your team
- **Project root (** **`./CLAUDE.local.md`** **)** : personal project-specific notes; add this file to your `.gitignore` so it isn't shared with your team
- **Parent directories** : useful for monorepos where both `root/CLAUDE.md` and `root/foo/CLAUDE.md` are pulled in automatically
- **Child directories** : Claude pulls in child CLAUDE.md files on demand when working with files in those directories

#### Configure permissions

Use [auto mode](/docs/en/permission-modes#eliminate-prompts-with-auto-mode) to let a classifier handle approvals, `/permissions` to allowlist specific commands, or `/sandbox` for OS-level isolation. Each reduces interruptions while keeping you in control.

By default, Claude Code requests permission for actions that might modify your system: file writes, Bash commands, MCP tools, etc. This is safe but tedious. After the tenth approval you're not really reviewing anymore, you're just clicking through. There are three ways to reduce these interruptions:

- **Auto mode** : a separate classifier model reviews commands and blocks only what looks risky: scope escalation, unknown infrastructure, or hostile-content-driven actions. Best when you trust the general direction of a task but don't want to click through every step
- **Permission allowlists** : permit specific tools you know are safe, like `npm run lint` or `git commit`
- **Sandboxing** : enable OS-level isolation that restricts filesystem and network access, allowing Claude to work more freely within defined boundaries

Read more about [permission modes](/docs/en/permission-modes) , [permission rules](/docs/en/permissions) , and [sandboxing](/docs/en/sandboxing) .

#### Use CLI tools

Tell Claude Code to use CLI tools like `gh` , `aws` , `gcloud` , and `sentry-cli` when interacting with external services.

CLI tools are the most context-efficient way to interact with external services. If you use GitHub, install the `gh` CLI. Claude knows how to use it for creating issues, opening pull requests, and reading comments. Without `gh` , Claude can still use the GitHub API, but unauthenticated requests often hit rate limits. Claude is also effective at learning CLI tools it doesn't already know. Try prompts like `Use 'foo-cli-tool --help' to learn about foo tool, then use it to solve A, B, C.`

#### Connect MCP servers

Run `claude mcp add` to connect external tools like Notion, Figma, or your database.

With [MCP servers](/docs/en/mcp) , you can ask Claude to implement features from issue trackers, query databases, analyze monitoring data, integrate designs from Figma, and automate workflows.

#### Set up hooks

Use hooks for actions that must happen every time with zero exceptions.

[Hooks](/docs/en/hooks-guide) run scripts automatically at specific points in Claude's workflow. Unlike CLAUDE.md instructions which are advisory, hooks are deterministic and guarantee the action happens. Claude can write hooks for you. Try prompts like *"Write a hook that runs eslint after every file edit"* or *"Write a hook that blocks writes to the migrations folder."* Edit `.claude/settings.json` directly to configure hooks by hand, and run `/hooks` to browse what's configured.

#### Create skills

Create `SKILL.md` files in `.claude/skills/` to give Claude domain knowledge and reusable workflows.

[Skills](/docs/en/skills) extend Claude's knowledge with information specific to your project, team, or domain. Claude applies them automatically when relevant, or you can invoke them directly with `/skill-name` . Create a skill by adding a directory with a `SKILL.md` to `.claude/skills/` :

.claude/skills/api-conventions/SKILL.md

```
---
name : api-conventions
description : REST API design conventions for our services
---
### API Conventions
- Use kebab-case for URL paths
- Use camelCase for JSON properties
- Always include pagination for list endpoints
- Version APIs in the URL path (/v1/, /v2/)
```

Skills can also define repeatable workflows you invoke directly:

.claude/skills/fix-issue/SKILL.md

```
---
name : fix-issue
description : Fix a GitHub issue
disable-model-invocation : true
---
Analyze and fix the GitHub issue: $ARGUMENTS.

1. Use `gh issue view` to get the issue details
2. Understand the problem described in the issue
3. Search the codebase for relevant files
4. Implement the necessary changes to fix the issue
5. Write and run tests to verify the fix
6. Ensure code passes linting and type checking
7. Create a descriptive commit message
8. Push and create a PR
```

Run `/fix-issue 1234` to invoke it. Use `disable-model-invocation: true` for workflows with side effects that you want to trigger manually.

#### Create custom subagents

Define specialized assistants in `.claude/agents/` that Claude can delegate to for isolated tasks.

[Subagents](/docs/en/sub-agents) run in their own context with their own set of allowed tools. They're useful for tasks that read many files or need specialized focus without cluttering your main conversation.

.claude/agents/security-reviewer.md

```
---
name : security-reviewer
description : Reviews code for security vulnerabilities
tools : Read, Grep, Glob, Bash
model : opus
---
You are a senior security engineer. Review code for:
- Injection vulnerabilities (SQL, XSS, command injection)
- Authentication and authorization flaws
- Secrets or credentials in code
- Insecure data handling

Provide specific line references and suggested fixes.
```

Tell Claude to use subagents explicitly: *"Use a subagent to review this code for security issues."*

#### Install plugins

Run `/plugin` to browse the marketplace. Plugins add skills, tools, and integrations without configuration.

[Plugins](/docs/en/plugins) bundle skills, hooks, subagents, and MCP servers into a single installable unit from the community and Anthropic. If you work with a typed language, install a [code intelligence plugin](/docs/en/discover-plugins#code-intelligence) to give Claude precise symbol navigation and automatic error detection after edits. For guidance on choosing between skills, subagents, hooks, and MCP, see [Extend Claude Code](/docs/en/features-overview#match-features-to-your-goal) .

### Communicate effectively

The way you communicate with Claude Code significantly impacts the quality of results.

#### Ask codebase questions

Ask Claude questions you'd ask a senior engineer.

When onboarding to a new codebase, use Claude Code for learning and exploration. You can ask Claude the same sorts of questions you would ask another engineer:

- How does logging work?
- How do I make a new API endpoint?
- What does `async move { ... }` do on line 134 of `foo.rs` ?
- What edge cases does `CustomerOnboardingFlowImpl` handle?
- Why does this code call `foo()` instead of `bar()` on line 333?

Using Claude Code this way is an effective onboarding workflow, improving ramp-up time and reducing load on other engineers. No special prompting required: ask questions directly.

#### Let Claude interview you

For larger features, have Claude interview you first. Start with a minimal prompt and ask Claude to interview you using the `AskUserQuestion` tool.

Claude asks about things you might not have considered yet, including technical implementation, UI/UX, edge cases, and tradeoffs.

```
I want to build [brief description]. Interview me in detail using the AskUserQuestion tool.

Ask about technical implementation, UI/UX, edge cases, concerns, and tradeoffs. Don't ask obvious questions, dig into the hard parts I might not have considered.

Keep interviewing until we've covered everything, then write a complete spec to SPEC.md.
```

Once the spec is complete, start a fresh session to execute it. The new session has clean context focused entirely on implementation, and you have a written spec to reference.

### Manage your session

Conversations are persistent and reversible. Use this to your advantage!

#### Course-correct early and often

Correct Claude as soon as you notice it going off track.

The best results come from tight feedback loops. Though Claude occasionally solves problems perfectly on the first attempt, correcting it quickly generally produces better solutions faster.

- **`Esc`** : stop Claude mid-action with the `Esc` key. Context is preserved, so you can redirect.
- **`Esc + Esc`** **or** **`/rewind`** : press `Esc` twice or run `/rewind` to open the rewind menu and restore previous conversation and code state, or summarize from a selected message.
- **`"Undo that"`** : have Claude revert its changes.
- **`/clear`** : reset context between unrelated tasks. Long sessions with irrelevant context can reduce performance.

If you've corrected Claude more than twice on the same issue in one session, the context is cluttered with failed approaches. Run `/clear` and start fresh with a more specific prompt that incorporates what you learned. A clean session with a better prompt almost always outperforms a long session with accumulated corrections.

#### Manage context aggressively

Run `/clear` between unrelated tasks to reset context.

Claude Code automatically compacts conversation history when you approach context limits, which preserves important code and decisions while freeing space. During long sessions, Claude's context window can fill with irrelevant conversation, file contents, and commands. This can reduce performance and sometimes distract Claude.

- Use `/clear` frequently between tasks to reset the context window entirely
- When auto compaction triggers, Claude summarizes what matters most, including code patterns, file states, and key decisions
- For more control, run `/compact <instructions>` , like `/compact Focus on the API changes`
- To compact only part of the conversation, use `Esc + Esc` or `/rewind` , select a message checkpoint, and choose **Summarize from here** . This condenses messages from that point forward while keeping earlier context intact.
- Customize compaction behavior in CLAUDE.md with instructions like `"When compacting, always preserve the full list of modified files and any test commands"` to ensure critical context survives summarization
- For quick questions that don't need to stay in context, use [`/btw`](/docs/en/interactive-mode#side-questions-with-btw) . The answer appears in a dismissible overlay and never enters conversation history, so you can check a detail without growing context.

#### Use subagents for investigation

Delegate research with `"use subagents to investigate X"` . They explore in a separate context, keeping your main conversation clean for implementation.

Since context is your fundamental constraint, subagents are one of the most powerful tools available. When Claude researches a codebase it reads lots of files, all of which consume your context. Subagents run in separate context windows and report back summaries:

```
Use subagents to investigate how our authentication system handles token
refresh, and whether we have any existing OAuth utilities I should reuse.
```

The subagent explores the codebase, reads relevant files, and reports back with findings, all without cluttering your main conversation. You can also use subagents for verification after Claude implements something:

```
use a subagent to review this code for edge cases
```

#### Rewind with checkpoints

Every action Claude makes creates a checkpoint. You can restore conversation, code, or both to any previous checkpoint.

Claude automatically checkpoints before changes. Double-tap `Escape` or run `/rewind` to open the rewind menu. You can restore conversation only, restore code only, restore both, or summarize from a selected message. See [Checkpointing](/docs/en/checkpointing) for details. Instead of carefully planning every move, you can tell Claude to try something risky. If it doesn't work, rewind and try a different approach. Checkpoints persist across sessions, so you can close your terminal and still rewind later.

Checkpoints only track changes made *by Claude* , not external processes. This isn't a replacement for git.

#### Resume conversations

Run `claude --continue` to pick up where you left off, or `--resume` to choose from recent sessions.

Claude Code saves conversations locally. When a task spans multiple sessions, you don't have to re-explain the context:

```
claude --continue # Resume the most recent conversation
claude --resume # Select from recent conversations
```

Use `/rename` to give sessions descriptive names like `"oauth-migration"` or `"debugging-memory-leak"` so you can find them later. Treat sessions like branches: different workstreams can have separate, persistent contexts.

### Automate and scale

Once you're effective with one Claude, multiply your output with parallel sessions, non-interactive mode, and fan-out patterns. Everything so far assumes one human, one Claude, and one conversation. But Claude Code scales horizontally. The techniques in this section show how you can get more done.

#### Run non-interactive mode

Use `claude -p "prompt"` in CI, pre-commit hooks, or scripts. Add `--output-format stream-json` for streaming JSON output.

With `claude -p "your prompt"` , you can run Claude non-interactively, without a session. Non-interactive mode is how you integrate Claude into CI pipelines, pre-commit hooks, or any automated workflow. The output formats let you parse results programmatically: plain text, JSON, or streaming JSON.

```
### One-off queries
claude -p "Explain what this project does"

### Structured output for scripts
claude -p "List all API endpoints" --output-format json

### Streaming for real-time processing
claude -p "Analyze this log file" --output-format stream-json
```

#### Run multiple Claude sessions

Run multiple Claude sessions in parallel to speed up development, run isolated experiments, or start complex workflows.

There are three main ways to run parallel sessions:

- [Claude Code desktop app](/docs/en/desktop#work-in-parallel-with-sessions) : Manage multiple local sessions visually. Each session gets its own isolated worktree.
- [Claude Code on the web](/docs/en/claude-code-on-the-web) : Run on Anthropic's secure cloud infrastructure in isolated VMs.
- [Agent teams](/docs/en/agent-teams) : Automated coordination of multiple sessions with shared tasks, messaging, and a team lead.

Beyond parallelizing work, multiple sessions enable quality-focused workflows. A fresh context improves code review since Claude won't be biased toward code it just wrote. For example, use a Writer/Reviewer pattern:

| Session A (Writer)                                                      | Session B (Reviewer)                                                                                                                                                     |
|-------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Implement a rate limiter for our API endpoints`                        |                                                                                                                                                                          |
|                                                                         | `Review the rate limiter implementation in @src/middleware/rateLimiter.ts. Look for edge cases, race conditions, and consistency with our existing middleware patterns.` |
| `Here's the review feedback: [Session B output]. Address these issues.` |                                                                                                                                                                          |

You can do something similar with tests: have one Claude write tests, then another write code to pass them.

#### Fan out across files

Loop through tasks calling `claude -p` for each. Use `--allowedTools` to scope permissions for batch operations.

For large migrations or analyses, you can distribute work across many parallel Claude invocations:

1

Generate a task list

Have Claude list all files that need migrating (e.g., `list all 2,000 Python files that need migrating` )

2

Write a script to loop through the list

```
for file in $( cat files.txt ); do
claude -p "Migrate $file from React to Vue. Return OK or FAIL." \
--allowedTools "Edit,Bash(git commit *)"
done
```

3

Test on a few files, then run at scale

Refine your prompt based on what goes wrong with the first 2-3 files, then run on the full set. The `--allowedTools` flag restricts what Claude can do, which matters when you're running unattended.

You can also integrate Claude into existing data/processing pipelines:

```
claude -p "<your prompt>" --output-format json | your_command
```

Use `--verbose` for debugging during development, and turn it off in production.

#### Run autonomously with auto mode

For uninterrupted execution with background safety checks, use [auto mode](/docs/en/permission-modes#eliminate-prompts-with-auto-mode) . A classifier model reviews commands before they run, blocking scope escalation, unknown infrastructure, and hostile-content-driven actions while letting routine work proceed without prompts.

```
claude --permission-mode auto -p "fix all lint errors"
```

For non-interactive runs with the `-p` flag, auto mode aborts if the classifier repeatedly blocks actions, since there is no user to fall back to. See [when auto mode falls back](/docs/en/permission-modes#when-auto-mode-falls-back) for thresholds.

### Avoid common failure patterns

These are common mistakes. Recognizing them early saves time:

- **The kitchen sink session.** You start with one task, then ask Claude something unrelated, then go back to the first task. Context is full of irrelevant information. **Fix** : `/clear` between unrelated tasks.
- **Correcting over and over.** Claude does something wrong, you correct it, it's still wrong, you correct again. Context is polluted with failed approaches. **Fix** : After two failed corrections, `/clear` and write a better initial prompt incorporating what you learned.
- **The over-specified CLAUDE.md.** If your CLAUDE.md is too long, Claude ignores half of it because important rules get lost in the noise. **Fix** : Ruthlessly prune. If Claude already does something correctly without the instruction, delete it or convert it to a hook.
- **The trust-then-verify gap.** Claude produces a plausible-looking implementation that doesn't handle edge cases. **Fix** : Always provide verification (tests, scripts, screenshots). If you can't verify it, don't ship it.
- **The infinite exploration.** You ask Claude to "investigate" something without scoping it. Claude reads hundreds of files, filling the context. **Fix** : Scope investigations narrowly or use subagents so the exploration doesn't consume your main context.

### Develop your intuition

The patterns in this guide aren't set in stone. They're starting points that work well in general, but might not be optimal for every situation. Sometimes you *should* let context accumulate because you're deep in one complex problem and the history is valuable. Sometimes you should skip planning and let Claude figure it out because the task is exploratory. Sometimes a vague prompt is exactly right because you want to see how Claude interprets the problem before constraining it. Pay attention to what works. When Claude produces great output, notice what you did: the prompt structure, the context you provided, the mode you were in. When Claude struggles, ask why. Was the context too noisy? The prompt too vague? The task too big for one pass? Over time, you'll develop intuition that no guide can capture. You'll know when to be specific and when to be open-ended, when to plan and when to explore, when to clear context and when to let it accumulate.

### Related resources

- [How Claude Code works](/docs/en/how-claude-code-works) : the agentic loop, tools, and context management
- [Extend Claude Code](/docs/en/features-overview) : skills, hooks, MCP, subagents, and plugins
- [Common workflows](/docs/en/common-workflows) : step-by-step recipes for debugging, testing, PRs, and more
- [CLAUDE.md](/docs/en/memory) : store project conventions and persistent context

Was this page helpful?

Yes

No

[Common workflows](/docs/en/common-workflows) [Overview](/docs/en/platforms)

⌘ I


---

# Sub-Agents


### Create custom subagents


Create and use specialized AI subagents in Claude Code for task-specific workflows and improved context management.


Subagents are specialized AI assistants that handle specific types of tasks. Use one when a side task would flood your main conversation with search results, logs, or file contents you won't reference again: the subagent does that work in its own context and returns only the summary. Define a custom subagent when you keep spawning the same kind of worker with the same instructions. Each subagent runs in its own context window with a custom system prompt, specific tool access, and independent permissions. When Claude encounters a task that matches a subagent's description, it delegates to that subagent, which works independently and returns results. To see the context savings in practice, the [context window visualization](/docs/en/context-window) walks through a session where a subagent handles research in its own separate window.

If you need multiple agents working in parallel and communicating with each other, see [agent teams](/docs/en/agent-teams) instead. Subagents work within a single session; agent teams coordinate across separate sessions.

Subagents help you:

- **Preserve context** by keeping exploration and implementation out of your main conversation
- **Enforce constraints** by limiting which tools a subagent can use
- **Reuse configurations** across projects with user-level subagents
- **Specialize behavior** with focused system prompts for specific domains
- **Control costs** by routing tasks to faster, cheaper models like Haiku

Claude uses each subagent's description to decide when to delegate tasks. When you create a subagent, write a clear description so Claude knows when to use it. Claude Code includes several built-in subagents like **Explore** , **Plan** , and **general-purpose** . You can also create custom subagents to handle specific tasks. This page covers the [built-in subagents](#built-in-subagents) , [how to create your own](#quickstart-create-your-first-subagent) , [full configuration options](#configure-subagents) , [patterns for working with subagents](#work-with-subagents) , and [example subagents](#example-subagents) .

### Built-in subagents

Claude Code includes built-in subagents that Claude automatically uses when appropriate. Each inherits the parent conversation's permissions with additional tool restrictions.

- Explore
- Plan
- General-purpose
- Other

A fast, read-only agent optimized for searching and analyzing codebases.

- **Model** : Haiku (fast, low-latency)
- **Tools** : Read-only tools (denied access to Write and Edit tools)
- **Purpose** : File discovery, code search, codebase exploration

Claude delegates to Explore when it needs to search or understand a codebase without making changes. This keeps exploration results out of your main conversation context. When invoking Explore, Claude specifies a thoroughness level: **quick** for targeted lookups, **medium** for balanced exploration, or **very thorough** for comprehensive analysis.

A research agent used during [plan mode](/docs/en/common-workflows#use-plan-mode-for-safe-code-analysis) to gather context before presenting a plan.

- **Model** : Inherits from main conversation
- **Tools** : Read-only tools (denied access to Write and Edit tools)
- **Purpose** : Codebase research for planning

When you're in plan mode and Claude needs to understand your codebase, it delegates research to the Plan subagent. This prevents infinite nesting (subagents cannot spawn other subagents) while still gathering necessary context.

A capable agent for complex, multi-step tasks that require both exploration and action.

- **Model** : Inherits from main conversation
- **Tools** : All tools
- **Purpose** : Complex research, multi-step operations, code modifications

Claude delegates to general-purpose when the task requires both exploration and modification, complex reasoning to interpret results, or multiple dependent steps.

Claude Code includes additional helper agents for specific tasks. These are typically invoked automatically, so you don't need to use them directly.

| Agent             | Model   | When Claude uses it                                      |
|-------------------|---------|----------------------------------------------------------|
| statusline-setup  | Sonnet  | When you run `/statusline` to configure your status line |
| Claude Code Guide | Haiku   | When you ask questions about Claude Code features        |

Beyond these built-in subagents, you can create your own with custom prompts, tool restrictions, permission modes, hooks, and skills. The following sections show how to get started and customize subagents.

### Quickstart: create your first subagent

Subagents are defined in Markdown files with YAML frontmatter. You can [create them manually](#write-subagent-files) or use the `/agents` command. This walkthrough guides you through creating a user-level subagent with the `/agents` command. The subagent reviews code and suggests improvements for the codebase.

1

Open the subagents interface

In Claude Code, run:

```
/agents
```

2

Choose a location

Switch to the **Library** tab, select **Create new agent** , then choose **Personal** . This saves the subagent to `~/.claude/agents/` so it's available in all your projects.

3

Generate with Claude

Select **Generate with Claude** . When prompted, describe the subagent:

```
A code improvement agent that scans files and suggests improvements
for readability, performance, and best practices. It should explain
each issue, show the current code, and provide an improved version.
```

Claude generates the identifier, description, and system prompt for you.

4

Select tools

For a read-only reviewer, deselect everything except **Read-only tools** . If you keep all tools selected, the subagent inherits all tools available to the main conversation.

5

Select model

Choose which model the subagent uses. For this example agent, select **Sonnet** , which balances capability and speed for analyzing code patterns.

6

Choose a color

Pick a background color for the subagent. This helps you identify which subagent is running in the UI.

7

Configure memory

Select **User scope** to give the subagent a [persistent memory directory](#enable-persistent-memory) at `~/.claude/agent-memory/` . The subagent uses this to accumulate insights across conversations, such as codebase patterns and recurring issues. Select **None** if you don't want the subagent to persist learnings.

8

Save and try it out

Review the configuration summary. Press `s` or `Enter` to save, or press `e` to save and edit the file in your editor. The subagent is available immediately. Try it:

```
Use the code-improver agent to suggest improvements in this project
```

Claude delegates to your new subagent, which scans the codebase and returns improvement suggestions.

You now have a subagent you can use in any project on your machine to analyze codebases and suggest improvements. You can also create subagents manually as Markdown files, define them via CLI flags, or distribute them through plugins. The following sections cover all configuration options.

### Configure subagents

#### Use the /agents command

The `/agents` command opens a tabbed interface for managing subagents. The **Running** tab shows live subagents and lets you open or stop them. The **Library** tab lets you:

- View all available subagents (built-in, user, project, and plugin)
- Create new subagents with guided setup or Claude generation
- Edit existing subagent configuration and tool access
- Delete custom subagents
- See which subagents are active when duplicates exist

This is the recommended way to create and manage subagents. For manual creation or automation, you can also add subagent files directly. To list all configured subagents from the command line without starting an interactive session, run `claude agents` . This shows agents grouped by source and indicates which are overridden by higher-priority definitions.

#### Choose the subagent scope

Subagents are Markdown files with YAML frontmatter. Store them in different locations depending on scope. When multiple subagents share the same name, the higher-priority location wins.

| Location                     | Scope                   | Priority    | How to create                                      |
|------------------------------|-------------------------|-------------|----------------------------------------------------|
| Managed settings             | Organization-wide       | 1 (highest) | Deployed via [managed settings](/docs/en/settings) |
| `--agents` CLI flag          | Current session         | 2           | Pass JSON when launching Claude Code               |
| `.claude/agents/`            | Current project         | 3           | Interactive or manual                              |
| `~/.claude/agents/`          | All your projects       | 4           | Interactive or manual                              |
| Plugin's `agents/` directory | Where plugin is enabled | 5 (lowest)  | Installed with [plugins](/docs/en/plugins)         |

**Project subagents** ( `.claude/agents/` ) are ideal for subagents specific to a codebase. Check them into version control so your team can use and improve them collaboratively. Project subagents are discovered by walking up from the current working directory. Directories added with `--add-dir` [grant file access only](/docs/en/permissions#additional-directories-grant-file-access-not-configuration) and are not scanned for subagents. To share subagents across projects, use `~/.claude/agents/` or a [plugin](/docs/en/plugins) . **User subagents** ( `~/.claude/agents/` ) are personal subagents available in all your projects. **CLI-defined subagents** are passed as JSON when launching Claude Code. They exist only for that session and aren't saved to disk, making them useful for quick testing or automation scripts. You can define multiple subagents in a single `--agents` call:

```
claude --agents '{
"code-reviewer": {
"description": "Expert code reviewer. Use proactively after code changes.",
"prompt": "You are a senior code reviewer. Focus on code quality, security, and best practices.",
"tools": ["Read", "Grep", "Glob", "Bash"],
"model": "sonnet"
},
"debugger": {
"description": "Debugging specialist for errors and test failures.",
"prompt": "You are an expert debugger. Analyze errors, identify root causes, and provide fixes."
}
}'
```

The `--agents` flag accepts JSON with the same [frontmatter](#supported-frontmatter-fields) fields as file-based subagents: `description` , `prompt` , `tools` , `disallowedTools` , `model` , `permissionMode` , `mcpServers` , `hooks` , `maxTurns` , `skills` , `initialPrompt` , `memory` , `effort` , `background` , `isolation` , and `color` . Use `prompt` for the system prompt, equivalent to the markdown body in file-based subagents. **Managed subagents** are deployed by organization administrators. Place markdown files in `.claude/agents/` inside the [managed settings directory](/docs/en/settings#settings-files) , using the same frontmatter format as project and user subagents. Managed definitions take precedence over project and user subagents with the same name. **Plugin subagents** come from [plugins](/docs/en/plugins) you've installed. They appear in `/agents` alongside your custom subagents. See the [plugin components reference](/docs/en/plugins-reference#agents) for details on creating plugin subagents.

For security reasons, plugin subagents do not support the `hooks` , `mcpServers` , or `permissionMode` frontmatter fields. These fields are ignored when loading agents from a plugin. If you need them, copy the agent file into `.claude/agents/` or `~/.claude/agents/` . You can also add rules to [`permissions.allow`](/docs/en/settings#permission-settings) in `settings.json` or `settings.local.json` , but these rules apply to the entire session, not just the plugin subagent.

Subagent definitions from any of these scopes are also available to [agent teams](/docs/en/agent-teams#use-subagent-definitions-for-teammates) : when spawning a teammate, you can reference a subagent type and the teammate uses its `tools` and `model` , with the definition's body appended to the teammate's system prompt as additional instructions. See [agent teams](/docs/en/agent-teams#use-subagent-definitions-for-teammates) for which frontmatter fields apply on that path.

#### Write subagent files

Subagent files use YAML frontmatter for configuration, followed by the system prompt in Markdown:

Subagents are loaded at session start. If you create a subagent by manually adding a file, restart your session or use `/agents` to load it immediately.

```
---
name : code-reviewer
description : Reviews code for quality and best practices
tools : Read, Glob, Grep
model : sonnet

You are a code reviewer. When invoked, analyze the code and provide
specific, actionable feedback on quality, security, and best practices.
```

The frontmatter defines the subagent's metadata and configuration. The body becomes the system prompt that guides the subagent's behavior. Subagents receive only this system prompt (plus basic environment details like working directory), not the full Claude Code system prompt. A subagent starts in the main conversation's current working directory. Within a subagent, `cd` commands do not persist between Bash or PowerShell tool calls and do not affect the main conversation's working directory. To give the subagent an isolated copy of the repository instead, set [`isolation: worktree`](#supported-frontmatter-fields) .

##### Supported frontmatter fields

The following fields can be used in the YAML frontmatter. Only `name` and `description` are required.

| Field             | Required   | Description                                                                                                                                                                                                                                                                             |
|-------------------|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `name`            | Yes        | Unique identifier using lowercase letters and hyphens                                                                                                                                                                                                                                   |
| `description`     | Yes        | When Claude should delegate to this subagent                                                                                                                                                                                                                                            |
| `tools`           | No         | [Tools](#available-tools) the subagent can use. Inherits all tools if omitted                                                                                                                                                                                                           |
| `disallowedTools` | No         | Tools to deny, removed from inherited or specified list                                                                                                                                                                                                                                 |
| `model`           | No         | [Model](#choose-a-model) to use: `sonnet` , `opus` , `haiku` , a full model ID (for example, `claude-opus-4-6` ), or `inherit` . Defaults to `inherit`                                                                                                                                  |
| `permissionMode`  | No         | [Permission mode](#permission-modes) : `default` , `acceptEdits` , `auto` , `dontAsk` , `bypassPermissions` , or `plan`                                                                                                                                                                 |
| `maxTurns`        | No         | Maximum number of agentic turns before the subagent stops                                                                                                                                                                                                                               |
| `skills`          | No         | [Skills](/docs/en/skills) to load into the subagent's context at startup. The full skill content is injected, not just made available for invocation. Subagents don't inherit skills from the parent conversation                                                                       |
| `mcpServers`      | No         | [MCP servers](/docs/en/mcp) available to this subagent. Each entry is either a server name referencing an already-configured server (e.g., `"slack"` ) or an inline definition with the server name as key and a full [MCP server config](/docs/en/mcp#installing-mcp-servers) as value |
| `hooks`           | No         | [Lifecycle hooks](#define-hooks-for-subagents) scoped to this subagent                                                                                                                                                                                                                  |
| `memory`          | No         | [Persistent memory scope](#enable-persistent-memory) : `user` , `project` , or `local` . Enables cross-session learning                                                                                                                                                                 |
| `background`      | No         | Set to `true` to always run this subagent as a [background task](#run-subagents-in-foreground-or-background) . Default: `false`                                                                                                                                                         |
| `effort`          | No         | Effort level when this subagent is active. Overrides the session effort level. Default: inherits from session. Options: `low` , `medium` , `high` , `max` (Opus 4.6 only)                                                                                                               |
| `isolation`       | No         | Set to `worktree` to run the subagent in a temporary [git worktree](/docs/en/common-workflows#run-parallel-claude-code-sessions-with-git-worktrees) , giving it an isolated copy of the repository. The worktree is automatically cleaned up if the subagent makes no changes           |
| `color`           | No         | Display color for the subagent in the task list and transcript. Accepts `red` , `blue` , `green` , `yellow` , `purple` , `orange` , `pink` , or `cyan`                                                                                                                                  |
| `initialPrompt`   | No         | Auto-submitted as the first user turn when this agent runs as the main session agent (via `--agent` or the `agent` setting). [Commands](/docs/en/commands) and [skills](/docs/en/skills) are processed. Prepended to any user-provided prompt                                           |

#### Choose a model

The `model` field controls which [AI model](/docs/en/model-config) the subagent uses:

- **Model alias** : Use one of the available aliases: `sonnet` , `opus` , or `haiku`
- **Full model ID** : Use a full model ID such as `claude-opus-4-6` or `claude-sonnet-4-6` . Accepts the same values as the `--model` flag
- **inherit** : Use the same model as the main conversation
- **Omitted** : If not specified, defaults to `inherit` (uses the same model as the main conversation)

When Claude invokes a subagent, it can also pass a `model` parameter for that specific invocation. Claude Code resolves the subagent's model in this order:

1. The [`CLAUDE_CODE_SUBAGENT_MODEL`](/docs/en/model-config#environment-variables) environment variable, if set
2. The per-invocation `model` parameter
3. The subagent definition's `model` frontmatter
4. The main conversation's model

#### Control subagent capabilities

You can control what subagents can do through tool access, permission modes, and conditional rules.

##### Available tools

Subagents can use any of Claude Code's [internal tools](/docs/en/tools-reference) . By default, subagents inherit all tools from the main conversation, including MCP tools. To restrict tools, use either the `tools` field (allowlist) or the `disallowedTools` field (denylist). This example uses `tools` to exclusively allow Read, Grep, Glob, and Bash. The subagent can't edit files, write files, or use any MCP tools:

```
---
name : safe-researcher
description : Research agent with restricted capabilities
tools : Read, Grep, Glob, Bash
---
```

This example uses `disallowedTools` to inherit every tool from the main conversation except Write and Edit. The subagent keeps Bash, MCP tools, and everything else:

```
---
name : no-writes
description : Inherits every tool except file writes
disallowedTools : Write, Edit
---
```

If both are set, `disallowedTools` is applied first, then `tools` is resolved against the remaining pool. A tool listed in both is removed.

##### Restrict which subagents can be spawned

When an agent runs as the main thread with `claude --agent` , it can spawn subagents using the Agent tool. To restrict which subagent types it can spawn, use `Agent(agent_type)` syntax in the `tools` field.

In version 2.1.63, the Task tool was renamed to Agent. Existing `Task(...)` references in settings and agent definitions still work as aliases.

```
---
name : coordinator
description : Coordinates work across specialized agents
tools : Agent(worker, researcher), Read, Bash
---
```

This is an allowlist: only the `worker` and `researcher` subagents can be spawned. If the agent tries to spawn any other type, the request fails and the agent sees only the allowed types in its prompt. To block specific agents while allowing all others, use [`permissions.deny`](#disable-specific-subagents) instead. To allow spawning any subagent without restrictions, use `Agent` without parentheses:

```
tools : Agent, Read, Bash
```

If `Agent` is omitted from the `tools` list entirely, the agent cannot spawn any subagents. This restriction only applies to agents running as the main thread with `claude --agent` . Subagents cannot spawn other subagents, so `Agent(agent_type)` has no effect in subagent definitions.

##### Scope MCP servers to a subagent

Use the `mcpServers` field to give a subagent access to [MCP](/docs/en/mcp) servers that aren't available in the main conversation. Inline servers defined here are connected when the subagent starts and disconnected when it finishes. String references share the parent session's connection. Each entry in the list is either an inline server definition or a string referencing an MCP server already configured in your session:

```
---
name : browser-tester
description : Tests features in a real browser using Playwright
mcpServers :
### Inline definition: scoped to this subagent only
- playwright :
type : stdio
command : npx
args : [ "-y" , "@playwright/mcp@latest" ]
### Reference by name: reuses an already-configured server
- github

Use the Playwright tools to navigate, screenshot, and interact with pages.
```

Inline definitions use the same schema as `.mcp.json` server entries ( `stdio` , `http` , `sse` , `ws` ), keyed by the server name. To keep an MCP server out of the main conversation entirely and avoid its tool descriptions consuming context there, define it inline here rather than in `.mcp.json` . The subagent gets the tools; the parent conversation does not.

##### Permission modes

The `permissionMode` field controls how the subagent handles permission prompts. Subagents inherit the permission context from the main conversation and can override the mode, except when the parent mode takes precedence as described below.

| Mode                | Behavior                                                                                                                                          |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `default`           | Standard permission checking with prompts                                                                                                         |
| `acceptEdits`       | Auto-accept file edits and common filesystem commands for paths in the working directory or `additionalDirectories`                               |
| `auto`              | [Auto mode](/docs/en/permission-modes#eliminate-prompts-with-auto-mode) : a background classifier reviews commands and protected-directory writes |
| `dontAsk`           | Auto-deny permission prompts (explicitly allowed tools still work)                                                                                |
| `bypassPermissions` | Skip permission prompts                                                                                                                           |
| `plan`              | Plan mode (read-only exploration)                                                                                                                 |

Use `bypassPermissions` with caution. It skips permission prompts, allowing the subagent to execute operations without approval. Writes to `.git` , `.claude` , `.vscode` , `.idea` , and `.husky` directories still prompt for confirmation, except for `.claude/commands` , `.claude/agents` , and `.claude/skills` . See [permission modes](/docs/en/permission-modes#skip-all-checks-with-bypasspermissions-mode) for details.

If the parent uses `bypassPermissions` , this takes precedence and cannot be overridden. If the parent uses [auto mode](/docs/en/permission-modes#eliminate-prompts-with-auto-mode) , the subagent inherits auto mode and any `permissionMode` in its frontmatter is ignored: the classifier evaluates the subagent's tool calls with the same block and allow rules as the parent session.

##### Preload skills into subagents

Use the `skills` field to inject skill content into a subagent's context at startup. This gives the subagent domain knowledge without requiring it to discover and load skills during execution.

```
---
name : api-developer
description : Implement API endpoints following team conventions
skills :
- api-conventions
- error-handling-patterns

Implement API endpoints. Follow the conventions and patterns from the preloaded skills.
```

The full content of each skill is injected into the subagent's context, not just made available for invocation. Subagents don't inherit skills from the parent conversation; you must list them explicitly.

This is the inverse of [running a skill in a subagent](/docs/en/skills#run-skills-in-a-subagent) . With `skills` in a subagent, the subagent controls the system prompt and loads skill content. With `context: fork` in a skill, the skill content is injected into the agent you specify. Both use the same underlying system.

##### Enable persistent memory

The `memory` field gives the subagent a persistent directory that survives across conversations. The subagent uses this directory to build up knowledge over time, such as codebase patterns, debugging insights, and architectural decisions.

```
---
name : code-reviewer
description : Reviews code for quality and best practices
memory : user

You are a code reviewer. As you review code, update your agent memory with
patterns, conventions, and recurring issues you discover.
```

Choose a scope based on how broadly the memory should apply:

| Scope     | Location                                      | Use when                                                                                    |
|-----------|-----------------------------------------------|---------------------------------------------------------------------------------------------|
| `user`    | `~/.claude/agent-memory/<name-of-agent>/`     | the subagent should remember learnings across all projects                                  |
| `project` | `.claude/agent-memory/<name-of-agent>/`       | the subagent's knowledge is project-specific and shareable via version control              |
| `local`   | `.claude/agent-memory-local/<name-of-agent>/` | the subagent's knowledge is project-specific but should not be checked into version control |

When memory is enabled:

- The subagent's system prompt includes instructions for reading and writing to the memory directory.
- The subagent's system prompt also includes the first 200 lines or 25KB of `MEMORY.md` in the memory directory, whichever comes first, with instructions to curate `MEMORY.md` if it exceeds that limit.
- Read, Write, and Edit tools are automatically enabled so the subagent can manage its memory files.

##### Persistent memory tips

- `project` is the recommended default scope. It makes subagent knowledge shareable via version control. Use `user` when the subagent's knowledge is broadly applicable across projects, or `local` when the knowledge should not be checked into version control.
- Ask the subagent to consult its memory before starting work: "Review this PR, and check your memory for patterns you've seen before."
- Ask the subagent to update its memory after completing a task: "Now that you're done, save what you learned to your memory." Over time, this builds a knowledge base that makes the subagent more effective.
- Include memory instructions directly in the subagent's markdown file so it proactively maintains its own knowledge base: `Update your agent memory as you discover codepaths, patterns, library locations, and key architectural decisions. This builds up institutional knowledge across conversations. Write concise notes about what you found and where.`

##### Conditional rules with hooks

For more dynamic control over tool usage, use `PreToolUse` hooks to validate operations before they execute. This is useful when you need to allow some operations of a tool while blocking others. This example creates a subagent that only allows read-only database queries. The `PreToolUse` hook runs the script specified in `command` before each Bash command executes:

```
---
name : db-reader
description : Execute read-only database queries
tools : Bash
hooks :
PreToolUse :
- matcher : "Bash"
hooks :
- type : command
command : "./scripts/validate-readonly-query.sh"
---
```

Claude Code [passes hook input as JSON](/docs/en/hooks#pretooluse-input) via stdin to hook commands. The validation script reads this JSON, extracts the Bash command, and [exits with code 2](/docs/en/hooks#exit-code-2-behavior-per-event) to block write operations:

```
#!/bin/bash
### ./scripts/validate-readonly-query.sh

INPUT = $( cat )
COMMAND = $( echo " $INPUT " | jq -r '.tool_input.command // empty' )

### Block SQL write operations (case-insensitive)
if echo " $COMMAND " | grep -iE '\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE)\b' > /dev/null ; then
echo "Blocked: Only SELECT queries are allowed" >&2
exit 2
fi

exit 0
```

See [Hook input](/docs/en/hooks#pretooluse-input) for the complete input schema and [exit codes](/docs/en/hooks#exit-code-output) for how exit codes affect behavior.

##### Disable specific subagents

You can prevent Claude from using specific subagents by adding them to the `deny` array in your [settings](/docs/en/settings#permission-settings) . Use the format `Agent(subagent-name)` where `subagent-name` matches the subagent's name field.

```
{
"permissions" : {
"deny" : [ "Agent(Explore)" , "Agent(my-custom-agent)" ]
}
}
```

This works for both built-in and custom subagents. You can also use the `--disallowedTools` CLI flag:

```
claude --disallowedTools "Agent(Explore)"
```

See [Permissions documentation](/docs/en/permissions#tool-specific-permission-rules) for more details on permission rules.

#### Define hooks for subagents

Subagents can define [hooks](/docs/en/hooks) that run during the subagent's lifecycle. There are two ways to configure hooks:

1. **In the subagent's frontmatter** : Define hooks that run only while that subagent is active
2. **In** **`settings.json`** : Define hooks that run in the main session when subagents start or stop

##### Hooks in subagent frontmatter

Define hooks directly in the subagent's markdown file. These hooks only run while that specific subagent is active and are cleaned up when it finishes. All [hook events](/docs/en/hooks#hook-events) are supported. The most common events for subagents are:

| Event         | Matcher input   | When it fires                                                       |
|---------------|-----------------|---------------------------------------------------------------------|
| `PreToolUse`  | Tool name       | Before the subagent uses a tool                                     |
| `PostToolUse` | Tool name       | After the subagent uses a tool                                      |
| `Stop`        | (none)          | When the subagent finishes (converted to `SubagentStop` at runtime) |

This example validates Bash commands with the `PreToolUse` hook and runs a linter after file edits with `PostToolUse` :

```
---
name : code-reviewer
description : Review code changes with automatic linting
hooks :
PreToolUse :
- matcher : "Bash"
hooks :
- type : command
command : "./scripts/validate-command.sh $TOOL_INPUT"
PostToolUse :
- matcher : "Edit|Write"
hooks :
- type : command
command : "./scripts/run-linter.sh"
---
```

`Stop` hooks in frontmatter are automatically converted to `SubagentStop` events.

##### Project-level hooks for subagent events

Configure hooks in `settings.json` that respond to subagent lifecycle events in the main session.

| Event           | Matcher input   | When it fires                    |
|-----------------|-----------------|----------------------------------|
| `SubagentStart` | Agent type name | When a subagent begins execution |
| `SubagentStop`  | Agent type name | When a subagent completes        |

Both events support matchers to target specific agent types by name. This example runs a setup script only when the `db-agent` subagent starts, and a cleanup script when any subagent stops:

```
{
"hooks" : {
"SubagentStart" : [
{
"matcher" : "db-agent" ,
"hooks" : [
{ "type" : "command" , "command" : "./scripts/setup-db-connection.sh" }
]
}
],
"SubagentStop" : [
{
"hooks" : [
{ "type" : "command" , "command" : "./scripts/cleanup-db-connection.sh" }
]
}
]
}
}
```

See [Hooks](/docs/en/hooks) for the complete hook configuration format.

### Work with subagents

#### Understand automatic delegation

Claude automatically delegates tasks based on the task description in your request, the `description` field in subagent configurations, and current context. To encourage proactive delegation, include phrases like "use proactively" in your subagent's description field.

#### Invoke subagents explicitly

When automatic delegation isn't enough, you can request a subagent yourself. Three patterns escalate from a one-off suggestion to a session-wide default:

- **Natural language** : name the subagent in your prompt; Claude decides whether to delegate
- **@-mention** : guarantees the subagent runs for one task
- **Session-wide** : the whole session uses that subagent's system prompt, tool restrictions, and model via the `--agent` flag or the `agent` setting

For natural language, there's no special syntax. Name the subagent and Claude typically delegates:

```
Use the test-runner subagent to fix failing tests
Have the code-reviewer subagent look at my recent changes
```

**@-mention the subagent.** Type `@` and pick the subagent from the typeahead, the same way you @-mention files. This ensures that specific subagent runs rather than leaving the choice to Claude:

```
@"code-reviewer (agent)" look at the auth changes
```

Your full message still goes to Claude, which writes the subagent's task prompt based on what you asked. The @-mention controls which subagent Claude invokes, not what prompt it receives. Subagents provided by an enabled [plugin](/docs/en/plugins) appear in the typeahead as `<plugin-name>:<agent-name>` . Named background subagents currently running in the session also appear in the typeahead, showing their status next to the name. You can also type the mention manually without using the picker: `@agent-<name>` for local subagents, or `@agent-<plugin-name>:<agent-name>` for plugin subagents. **Run the whole session as a subagent.** Pass [`--agent <name>`](/docs/en/cli-reference) to start a session where the main thread itself takes on that subagent's system prompt, tool restrictions, and model:

```
claude --agent code-reviewer
```

The subagent's system prompt replaces the default Claude Code system prompt entirely, the same way [`--system-prompt`](/docs/en/cli-reference) does. `CLAUDE.md` files and project memory still load through the normal message flow. The agent name appears as `@<name>` in the startup header so you can confirm it's active. This works with built-in and custom subagents, and the choice persists when you resume the session. For a plugin-provided subagent, pass the scoped name: `claude --agent <plugin-name>:<agent-name>` . To make it the default for every session in a project, set `agent` in `.claude/settings.json` :

```
{
"agent" : "code-reviewer"
}
```

The CLI flag overrides the setting if both are present.

#### Run subagents in foreground or background

Subagents can run in the foreground (blocking) or background (concurrent):

- **Foreground subagents** block the main conversation until complete. Permission prompts and clarifying questions (like [`AskUserQuestion`](/docs/en/tools-reference) ) are passed through to you.
- **Background subagents** run concurrently while you continue working. Before launching, Claude Code prompts for any tool permissions the subagent will need, ensuring it has the necessary approvals upfront. Once running, the subagent inherits these permissions and auto-denies anything not pre-approved. If a background subagent needs to ask clarifying questions, that tool call fails but the subagent continues.

If a background subagent fails due to missing permissions, you can start a new foreground subagent with the same task to retry with interactive prompts. Claude decides whether to run subagents in the foreground or background based on the task. You can also:

- Ask Claude to "run this in the background"
- Press **Ctrl+B** to background a running task

To disable all background task functionality, set the `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS` environment variable to `1` . See [Environment variables](/docs/en/env-vars) .

#### Common patterns

##### Isolate high-volume operations

One of the most effective uses for subagents is isolating operations that produce large amounts of output. Running tests, fetching documentation, or processing log files can consume significant context. By delegating these to a subagent, the verbose output stays in the subagent's context while only the relevant summary returns to your main conversation.

```
Use a subagent to run the test suite and report only the failing tests with their error messages
```

##### Run parallel research

For independent investigations, spawn multiple subagents to work simultaneously:

```
Research the authentication, database, and API modules in parallel using separate subagents
```

Each subagent explores its area independently, then Claude synthesizes the findings. This works best when the research paths don't depend on each other.

When subagents complete, their results return to your main conversation. Running many subagents that each return detailed results can consume significant context.

For tasks that need sustained parallelism or exceed your context window, [agent teams](/docs/en/agent-teams) give each worker its own independent context.

##### Chain subagents

For multi-step workflows, ask Claude to use subagents in sequence. Each subagent completes its task and returns results to Claude, which then passes relevant context to the next subagent.

```
Use the code-reviewer subagent to find performance issues, then use the optimizer subagent to fix them
```

#### Choose between subagents and main conversation

Use the **main conversation** when:

- The task needs frequent back-and-forth or iterative refinement
- Multiple phases share significant context (planning → implementation → testing)
- You're making a quick, targeted change
- Latency matters. Subagents start fresh and may need time to gather context

Use **subagents** when:

- The task produces verbose output you don't need in your main context
- You want to enforce specific tool restrictions or permissions
- The work is self-contained and can return a summary

Consider [Skills](/docs/en/skills) instead when you want reusable prompts or workflows that run in the main conversation context rather than isolated subagent context. For a quick question about something already in your conversation, use [`/btw`](/docs/en/interactive-mode#side-questions-with-btw) instead of a subagent. It sees your full context but has no tool access, and the answer is discarded rather than added to history.

Subagents cannot spawn other subagents. If your workflow requires nested delegation, use [Skills](/docs/en/skills) or [chain subagents](#chain-subagents) from the main conversation.

#### Manage subagent context

##### Resume subagents

Each subagent invocation creates a new instance with fresh context. To continue an existing subagent's work instead of starting over, ask Claude to resume it. Resumed subagents retain their full conversation history, including all previous tool calls, results, and reasoning. The subagent picks up exactly where it stopped rather than starting fresh. When a subagent completes, Claude receives its agent ID. Claude uses the `SendMessage` tool with the agent's ID as the `to` field to resume it. The `SendMessage` tool is only available when [agent teams](/docs/en/agent-teams) are enabled via `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` . To resume a subagent, ask Claude to continue the previous work:

```
Use the code-reviewer subagent to review the authentication module
[Agent completes]

Continue that code review and now analyze the authorization logic
[Claude resumes the subagent with full context from previous conversation]
```

If a stopped subagent receives a `SendMessage` , it auto-resumes in the background without requiring a new `Agent` invocation. You can also ask Claude for the agent ID if you want to reference it explicitly, or find IDs in the transcript files at `~/.claude/projects/{project}/{sessionId}/subagents/` . Each transcript is stored as `agent-{agentId}.jsonl` . Subagent transcripts persist independently of the main conversation:

- **Main conversation compaction** : When the main conversation compacts, subagent transcripts are unaffected. They're stored in separate files.
- **Session persistence** : Subagent transcripts persist within their session. You can [resume a subagent](#resume-subagents) after restarting Claude Code by resuming the same session.
- **Automatic cleanup** : Transcripts are cleaned up based on the `cleanupPeriodDays` setting (default: 30 days).

##### Auto-compaction

Subagents support automatic compaction using the same logic as the main conversation. By default, auto-compaction triggers at approximately 95% capacity. To trigger compaction earlier, set `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` to a lower percentage (for example, `50` ). See [environment variables](/docs/en/env-vars) for details. Compaction events are logged in subagent transcript files:

```
{
"type" : "system" ,
"subtype" : "compact_boundary" ,
"compactMetadata" : {
"trigger" : "auto" ,
"preTokens" : 167189
}
}
```

The `preTokens` value shows how many tokens were used before compaction occurred.

### Example subagents

These examples demonstrate effective patterns for building subagents. Use them as starting points, or generate a customized version with Claude.

**Best practices:**

- **Design focused subagents:** each subagent should excel at one specific task
- **Write detailed descriptions:** Claude uses the description to decide when to delegate
- **Limit tool access:** grant only necessary permissions for security and focus
- **Check into version control:** share project subagents with your team

#### Code reviewer

A read-only subagent that reviews code without modifying it. This example shows how to design a focused subagent with limited tool access (no Edit or Write) and a detailed prompt that specifies exactly what to look for and how to format output.

```
---
name : code-reviewer
description : Expert code review specialist. Proactively reviews code for quality, security, and maintainability. Use immediately after writing or modifying code.
tools : Read, Grep, Glob, Bash
model : inherit

You are a senior code reviewer ensuring high standards of code quality and security.

When invoked:
1. Run git diff to see recent changes
2. Focus on modified files
3. Begin review immediately

Review checklist:
- Code is clear and readable
- Functions and variables are well-named
- No duplicated code
- Proper error handling
- No exposed secrets or API keys
- Input validation implemented
- Good test coverage
- Performance considerations addressed

Provide feedback organized by priority:
- Critical issues (must fix)
- Warnings (should fix)
- Suggestions (consider improving)

Include specific examples of how to fix issues.
```

#### Debugger

A subagent that can both analyze and fix issues. Unlike the code reviewer, this one includes Edit because fixing bugs requires modifying code. The prompt provides a clear workflow from diagnosis to verification.

```
---
name : debugger
description : Debugging specialist for errors, test failures, and unexpected behavior. Use proactively when encountering any issues.
tools : Read, Edit, Bash, Grep, Glob

You are an expert debugger specializing in root cause analysis.

When invoked:
1. Capture error message and stack trace
2. Identify reproduction steps
3. Isolate the failure location
4. Implement minimal fix
5. Verify solution works

Debugging process:
- Analyze error messages and logs
- Check recent code changes
- Form and test hypotheses
- Add strategic debug logging
- Inspect variable states

For each issue, provide:
- Root cause explanation
- Evidence supporting the diagnosis
- Specific code fix
- Testing approach
- Prevention recommendations

Focus on fixing the underlying issue, not the symptoms.
```

#### Data scientist

A domain-specific subagent for data analysis work. This example shows how to create subagents for specialized workflows outside of typical coding tasks. It explicitly sets `model: sonnet` for more capable analysis.

```
---
name : data-scientist
description : Data analysis expert for SQL queries, BigQuery operations, and data insights. Use proactively for data analysis tasks and queries.
tools : Bash, Read, Write
model : sonnet

You are a data scientist specializing in SQL and BigQuery analysis.

When invoked:
1. Understand the data analysis requirement
2. Write efficient SQL queries
3. Use BigQuery command line tools (bq) when appropriate
4. Analyze and summarize results
5. Present findings clearly

Key practices:
- Write optimized SQL queries with proper filters
- Use appropriate aggregations and joins
- Include comments explaining complex logic
- Format results for readability
- Provide data-driven recommendations

For each analysis:
- Explain the query approach
- Document any assumptions
- Highlight key findings
- Suggest next steps based on data

Always ensure queries are efficient and cost-effective.
```

#### Database query validator

A subagent that allows Bash access but validates commands to permit only read-only SQL queries. This example shows how to use `PreToolUse` hooks for conditional validation when you need finer control than the `tools` field provides.

```
---
name : db-reader
description : Execute read-only database queries. Use when analyzing data or generating reports.
tools : Bash
hooks :
PreToolUse :
- matcher : "Bash"
hooks :
- type : command
command : "./scripts/validate-readonly-query.sh"

You are a database analyst with read-only access. Execute SELECT queries to answer questions about the data.

When asked to analyze data:
1. Identify which tables contain the relevant data
2. Write efficient SELECT queries with appropriate filters
3. Present results clearly with context

You cannot modify data. If asked to INSERT, UPDATE, DELETE, or modify schema, explain that you only have read access.
```

Claude Code [passes hook input as JSON](/docs/en/hooks#pretooluse-input) via stdin to hook commands. The validation script reads this JSON, extracts the command being executed, and checks it against a list of SQL write operations. If a write operation is detected, the script [exits with code 2](/docs/en/hooks#exit-code-2-behavior-per-event) to block execution and returns an error message to Claude via stderr. Create the validation script anywhere in your project. The path must match the `command` field in your hook configuration:

```
#!/bin/bash
### Blocks SQL write operations, allows SELECT queries

### Read JSON input from stdin
INPUT = $( cat )

### Extract the command field from tool_input using jq
COMMAND = $( echo " $INPUT " | jq -r '.tool_input.command // empty' )

if [ -z " $COMMAND " ]; then
exit 0
fi

### Block write operations (case-insensitive)
if echo " $COMMAND " | grep -iE '\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|MERGE)\b' > /dev/null ; then
echo "Blocked: Write operations not allowed. Use SELECT queries only." >&2
exit 2
fi

exit 0
```

Make the script executable:

```
chmod +x ./scripts/validate-readonly-query.sh
```

The hook receives JSON via stdin with the Bash command in `tool_input.command` . Exit code 2 blocks the operation and feeds the error message back to Claude. See [Hooks](/docs/en/hooks#exit-code-output) for details on exit codes and [Hook input](/docs/en/hooks#pretooluse-input) for the complete input schema.

### Next steps

Now that you understand subagents, explore these related features:

- [Distribute subagents with plugins](/docs/en/plugins) to share subagents across teams or projects
- [Run Claude Code programmatically](/docs/en/headless) with the Agent SDK for CI/CD and automation
- [Use MCP servers](/docs/en/mcp) to give subagents access to external tools and data

Was this page helpful?

Yes

No

[Run agent teams](/docs/en/agent-teams)

⌘ I


---

# MCP & Extensions


### Connect Claude Code to tools via MCP


Learn how to connect Claude Code to your tools with the Model Context Protocol.


Claude Code can connect to hundreds of external tools and data sources through the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) , an open source standard for AI-tool integrations. MCP servers give Claude Code access to your tools, databases, and APIs. Connect a server when you find yourself copying data into chat from another tool, like an issue tracker or a monitoring dashboard. Once connected, Claude can read and act on that system directly instead of working from what you paste.

### What you can do with MCP

With MCP servers connected, you can ask Claude Code to:

- **Implement features from issue trackers** : "Add the feature described in JIRA issue ENG-4521 and create a PR on GitHub."
- **Analyze monitoring data** : "Check Sentry and Statsig to check the usage of the feature described in ENG-4521."
- **Query databases** : "Find emails of 10 random users who used feature ENG-4521, based on our PostgreSQL database."
- **Integrate designs** : "Update our standard email template based on the new Figma designs that were posted in Slack"
- **Automate workflows** : "Create Gmail drafts inviting these 10 users to a feedback session about the new feature."
- **React to external events** : An MCP server can also act as a [channel](/docs/en/channels) that pushes messages into your session, so Claude reacts to Telegram messages, Discord chats, or webhook events while you're away.

### Popular MCP servers

Here are some commonly used MCP servers you can connect to Claude Code:

Use third party MCP servers at your own risk - Anthropic has not verified

the correctness or security of all these servers.

Make sure you trust MCP servers you are installing.

Be especially careful when using MCP servers that could fetch untrusted

content, as these can expose you to prompt injection risk.

**Need a specific integration?** [Find hundreds more MCP servers on GitHub](https://github.com/modelcontextprotocol/servers) , or build your own using the [MCP SDK](https://modelcontextprotocol.io/quickstart/server) .

### Installing MCP servers

MCP servers can be configured in three different ways depending on your needs:

#### Option 1: Add a remote HTTP server

HTTP servers are the recommended option for connecting to remote MCP servers. This is the most widely supported transport for cloud-based services.

```
### Basic syntax
claude mcp add --transport http < nam e > < ur l >

### Real example: Connect to Notion
claude mcp add --transport http notion https://mcp.notion.com/mcp

### Example with Bearer token
claude mcp add --transport http secure-api https://api.example.com/mcp \
--header "Authorization: Bearer your-token"
```

#### Option 2: Add a remote SSE server

The SSE (Server-Sent Events) transport is deprecated. Use HTTP servers instead, where available.

```
### Basic syntax
claude mcp add --transport sse < nam e > < ur l >

### Real example: Connect to Asana
claude mcp add --transport sse asana https://mcp.asana.com/sse

### Example with authentication header
claude mcp add --transport sse private-api https://api.company.com/sse \
--header "X-API-Key: your-key-here"
```

#### Option 3: Add a local stdio server

Stdio servers run as local processes on your machine. They're ideal for tools that need direct system access or custom scripts.

```
### Basic syntax
claude mcp add [options] < name > -- < command > [args...]

### Real example: Add Airtable server
claude mcp add --transport stdio --env AIRTABLE_API_KEY=YOUR_KEY airtable \
-- npx -y airtable-mcp-server
```

**Important: Option ordering** All options ( `--transport` , `--env` , `--scope` , `--header` ) must come **before** the server name. The `--` (double dash) then separates the server name from the command and arguments that get passed to the MCP server. For example:

- `claude mcp add --transport stdio myserver -- npx server` → runs `npx server`
- `claude mcp add --transport stdio --env KEY=value myserver -- python server.py --port 8080` → runs `python server.py --port 8080` with `KEY=value` in environment

This prevents conflicts between Claude's flags and the server's flags.

#### Managing your servers

Once configured, you can manage your MCP servers with these commands:

```
### List all configured servers
claude mcp list

### Get details for a specific server
claude mcp get github

### Remove a server
claude mcp remove github

### (within Claude Code) Check server status
/mcp
```

#### Dynamic tool updates

Claude Code supports MCP `list_changed` notifications, allowing MCP servers to dynamically update their available tools, prompts, and resources without requiring you to disconnect and reconnect. When an MCP server sends a `list_changed` notification, Claude Code automatically refreshes the available capabilities from that server.

#### Push messages with channels

An MCP server can also push messages directly into your session so Claude can react to external events like CI results, monitoring alerts, or chat messages. To enable this, your server declares the `claude/channel` capability and you opt it in with the `--channels` flag at startup. See [Channels](/docs/en/channels) to use an officially supported channel, or [Channels reference](/docs/en/channels-reference) to build your own.

Tips:

- Use the `--scope` flag to specify where the configuration is stored:
    - `local` (default): Available only to you in the current project (was called `project` in older versions)
    - `project` : Shared with everyone in the project via `.mcp.json` file
    - `user` : Available to you across all projects (was called `global` in older versions)
- Set environment variables with `--env` flags (for example, `--env KEY=value` )
- Configure MCP server startup timeout using the MCP\_TIMEOUT environment variable (for example, `MCP_TIMEOUT=10000 claude` sets a 10-second timeout)
- Claude Code will display a warning when MCP tool output exceeds 10,000 tokens. To increase this limit, set the `MAX_MCP_OUTPUT_TOKENS` environment variable (for example, `MAX_MCP_OUTPUT_TOKENS=50000` )
- Use `/mcp` to authenticate with remote servers that require OAuth 2.0 authentication

**Windows Users** : On native Windows (not WSL), local MCP servers that use `npx` require the `cmd /c` wrapper to ensure proper execution.

```
### This creates command="cmd" which Windows can execute
claude mcp add --transport stdio my-server -- cmd /c npx -y @some/package
```

Without the `cmd /c` wrapper, you'll encounter "Connection closed" errors because Windows cannot directly execute `npx` . (See the note above for an explanation of the `--` parameter.)

#### Plugin-provided MCP servers

[Plugins](/docs/en/plugins) can bundle MCP servers, automatically providing tools and integrations when the plugin is enabled. Plugin MCP servers work identically to user-configured servers. **How plugin MCP servers work** :

- Plugins define MCP servers in `.mcp.json` at the plugin root or inline in `plugin.json`
- When a plugin is enabled, its MCP servers start automatically
- Plugin MCP tools appear alongside manually configured MCP tools
- Plugin servers are managed through plugin installation (not `/mcp` commands)

**Example plugin MCP configuration** : In `.mcp.json` at plugin root:

```
{
"mcpServers" : {
"database-tools" : {
"command" : "${CLAUDE_PLUGIN_ROOT}/servers/db-server" ,
"args" : [ "--config" , "${CLAUDE_PLUGIN_ROOT}/config.json" ],
"env" : {
"DB_URL" : "${DB_URL}"
}
}
}
}
```

Or inline in `plugin.json` :

```
{
"name" : "my-plugin" ,
"mcpServers" : {
"plugin-api" : {
"command" : "${CLAUDE_PLUGIN_ROOT}/servers/api-server" ,
"args" : [ "--port" , "8080" ]
}
}
}
```

**Plugin MCP features** :

- **Automatic lifecycle** : At session startup, servers for enabled plugins connect automatically. If you enable or disable a plugin during a session, run `/reload-plugins` to connect or disconnect its MCP servers
- **Environment variables** : use `${CLAUDE_PLUGIN_ROOT}` for bundled plugin files and `${CLAUDE_PLUGIN_DATA}` for [persistent state](/docs/en/plugins-reference#persistent-data-directory) that survives plugin updates
- **User environment access** : Access to same environment variables as manually configured servers
- **Multiple transport types** : Support stdio, SSE, and HTTP transports (transport support may vary by server)

**Viewing plugin MCP servers** :

```
### Within Claude Code, see all MCP servers including plugin ones
/mcp
```

Plugin servers appear in the list with indicators showing they come from plugins. **Benefits of plugin MCP servers** :

- **Bundled distribution** : Tools and servers packaged together
- **Automatic setup** : No manual MCP configuration needed
- **Team consistency** : Everyone gets the same tools when plugin is installed

See the [plugin components reference](/docs/en/plugins-reference#mcp-servers) for details on bundling MCP servers with plugins.

### MCP installation scopes

MCP servers can be configured at three scopes. The scope you choose controls which projects the server loads in and whether the configuration is shared with your team.

| Scope                     | Loads in             | Shared with team         | Stored in                   |
|---------------------------|----------------------|--------------------------|-----------------------------|
| [Local](#local-scope)     | Current project only | No                       | `~/.claude.json`            |
| [Project](#project-scope) | Current project only | Yes, via version control | `.mcp.json` in project root |
| [User](#user-scope)       | All your projects    | No                       | `~/.claude.json`            |

#### Local scope

Local scope is the default. A local-scoped server loads only in the project where you added it and stays private to you. Claude Code stores it in `~/.claude.json` under that project's path, so the same server won't appear in your other projects. Use local scope for personal development servers, experimental configurations, or servers with credentials you don't want in version control.

The term "local scope" for MCP servers differs from general local settings. MCP local-scoped servers are stored in `~/.claude.json` (your home directory), while general local settings use `.claude/settings.local.json` (in the project directory). See [Settings](/docs/en/settings#settings-files) for details on settings file locations.

```
### Add a local-scoped server (default)
claude mcp add --transport http stripe https://mcp.stripe.com

### Explicitly specify local scope
claude mcp add --transport http stripe --scope local https://mcp.stripe.com
```

The command writes the server into the entry for your current project inside `~/.claude.json` . The example below shows the result when you run it from `/path/to/your/project` :

```
{
"projects" : {
"/path/to/your/project" : {
"mcpServers" : {
"stripe" : {
"type" : "http" ,
"url" : "https://mcp.stripe.com"
}
}
}
}
}
```

#### Project scope

Project-scoped servers enable team collaboration by storing configurations in a `.mcp.json` file at your project's root directory. This file is designed to be checked into version control, ensuring all team members have access to the same MCP tools and services. When you add a project-scoped server, Claude Code automatically creates or updates this file with the appropriate configuration structure.

```
### Add a project-scoped server
claude mcp add --transport http paypal --scope project https://mcp.paypal.com/mcp
```

The resulting `.mcp.json` file follows a standardized format:

```
{
"mcpServers" : {
"shared-server" : {
"command" : "/path/to/server" ,
"args" : [],
"env" : {}
}
}
}
```

For security reasons, Claude Code prompts for approval before using project-scoped servers from `.mcp.json` files. If you need to reset these approval choices, use the `claude mcp reset-project-choices` command.

#### User scope

User-scoped servers are stored in `~/.claude.json` and provide cross-project accessibility, making them available across all projects on your machine while remaining private to your user account. This scope works well for personal utility servers, development tools, or services you frequently use across different projects.

```
### Add a user server
claude mcp add --transport http hubspot --scope user https://mcp.hubspot.com/anthropic
```

#### Scope hierarchy and precedence

When the same server is defined in more than one place, Claude Code connects to it once, using the definition from the highest-precedence source:

1. Local scope
2. Project scope
3. User scope
4. [Plugin-provided servers](/docs/en/plugins)
5. [claude.ai connectors](#use-mcp-servers-from-claude-ai)

The three scopes match duplicates by name. Plugins and connectors match by endpoint, so one that points at the same URL or command as a server above is treated as a duplicate.

#### Environment variable expansion in .mcp.json

Claude Code supports environment variable expansion in `.mcp.json` files, allowing teams to share configurations while maintaining flexibility for machine-specific paths and sensitive values like API keys. **Supported syntax:**

- `${VAR}` - Expands to the value of environment variable `VAR`
- `${VAR:-default}` - Expands to `VAR` if set, otherwise uses `default`

**Expansion locations:** Environment variables can be expanded in:

- `command` - The server executable path
- `args` - Command-line arguments
- `env` - Environment variables passed to the server
- `url` - For HTTP server types
- `headers` - For HTTP server authentication

**Example with variable expansion:**

```
{
"mcpServers" : {
"api-server" : {
"type" : "http" ,
"url" : "${API_BASE_URL:-https://api.example.com}/mcp" ,
"headers" : {
"Authorization" : "Bearer ${API_KEY}"
}
}
}
}
```

If a required environment variable is not set and has no default value, Claude Code will fail to parse the config.

### Practical examples

#### Example: Monitor errors with Sentry

```
claude mcp add --transport http sentry https://mcp.sentry.dev/mcp
```

Authenticate with your Sentry account:

```
/mcp
```

Then debug production issues:

```
What are the most common errors in the last 24 hours?
```

```
Show me the stack trace for error ID abc123
```

```
Which deployment introduced these new errors?
```

#### Example: Connect to GitHub for code reviews

```
claude mcp add --transport http github https://api.githubcopilot.com/mcp/
```

Authenticate if needed by selecting "Authenticate" for GitHub:

```
/mcp
```

Then work with GitHub:

```
Review PR #456 and suggest improvements
```

```
Create a new issue for the bug we just found
```

```
Show me all open PRs assigned to me
```

#### Example: Query your PostgreSQL database

`claude mcp add --transport stdio db -- npx -y @bytebase/dbhub \
--dsn "postgresql://readonly:` [`[email protected]`](/cdn-cgi/l/email-protection) `:5432/analytics"`

Then query your database naturally:

```
What's our total revenue this month?
```

```
Show me the schema for the orders table
```

```
Find customers who haven't made a purchase in 90 days
```

### Authenticate with remote MCP servers

Many cloud-based MCP servers require authentication. Claude Code supports OAuth 2.0 for secure connections.

1

Add the server that requires authentication

For example:

```
claude mcp add --transport http sentry https://mcp.sentry.dev/mcp
```

2

Use the /mcp command within Claude Code

In Claude code, use the command:

```
/mcp
```

Then follow the steps in your browser to login.

Tips:

- Authentication tokens are stored securely and refreshed automatically
- Use "Clear authentication" in the `/mcp` menu to revoke access
- If your browser doesn't open automatically, copy the provided URL and open it manually
- If the browser redirect fails with a connection error after authenticating, paste the full callback URL from your browser's address bar into the URL prompt that appears in Claude Code
- OAuth authentication works with HTTP servers

#### Use a fixed OAuth callback port

Some MCP servers require a specific redirect URI registered in advance. By default, Claude Code picks a random available port for the OAuth callback. Use `--callback-port` to fix the port so it matches a pre-registered redirect URI of the form `http://localhost:PORT/callback` . You can use `--callback-port` on its own (with dynamic client registration) or together with `--client-id` (with pre-configured credentials).

```
### Fixed callback port with dynamic client registration
claude mcp add --transport http \
--callback-port 8080 \
my-server https://mcp.example.com/mcp
```

#### Use pre-configured OAuth credentials

Some MCP servers don't support automatic OAuth setup via Dynamic Client Registration. If you see an error like "Incompatible auth server: does not support dynamic client registration," the server requires pre-configured credentials. Claude Code also supports servers that use a Client ID Metadata Document (CIMD) instead of Dynamic Client Registration, and discovers these automatically. If automatic discovery fails, register an OAuth app through the server's developer portal first, then provide the credentials when adding the server.

1

Register an OAuth app with the server

Create an app through the server's developer portal and note your client ID and client secret. Many servers also require a redirect URI. If so, choose a port and register a redirect URI in the format `http://localhost:PORT/callback` . Use that same port with `--callback-port` in the next step.

2

Add the server with your credentials

Choose one of the following methods. The port used for `--callback-port` can be any available port. It just needs to match the redirect URI you registered in the previous step.

- claude mcp add
- claude mcp add-json
- claude mcp add-json (callback port only)
- CI / env var

Use `--client-id` to pass your app's client ID. The `--client-secret` flag prompts for the secret with masked input:

```
claude mcp add --transport http \
--client-id your-client-id --client-secret --callback-port 8080 \
my-server https://mcp.example.com/mcp
```

Include the `oauth` object in the JSON config and pass `--client-secret` as a separate flag:

```
claude mcp add-json my-server \
'{"type":"http","url":"https://mcp.example.com/mcp","oauth":{"clientId":"your-client-id","callbackPort":8080}}' \
--client-secret
```

Use `--callback-port` without a client ID to fix the port while using dynamic client registration:

```
claude mcp add-json my-server \
'{"type":"http","url":"https://mcp.example.com/mcp","oauth":{"callbackPort":8080}}'
```

Set the secret via environment variable to skip the interactive prompt:

```
MCP_CLIENT_SECRET = your-secret claude mcp add --transport http \
--client-id your-client-id --client-secret --callback-port 8080 \
my-server https://mcp.example.com/mcp
```

3

Authenticate in Claude Code

Run `/mcp` in Claude Code and follow the browser login flow.

Tips:

- The client secret is stored securely in your system keychain (macOS) or a credentials file, not in your config
- If the server uses a public OAuth client with no secret, use only `--client-id` without `--client-secret`
- `--callback-port` can be used with or without `--client-id`
- These flags only apply to HTTP and SSE transports. They have no effect on stdio servers
- Use `claude mcp get <name>` to verify that OAuth credentials are configured for a server

#### Override OAuth metadata discovery

If your MCP server's standard OAuth metadata endpoints return errors but the server exposes a working OIDC endpoint, you can point Claude Code at a specific metadata URL to bypass the default discovery chain. By default, Claude Code first checks RFC 9728 Protected Resource Metadata at `/.well-known/oauth-protected-resource` , then falls back to RFC 8414 authorization server metadata at `/.well-known/oauth-authorization-server` . Set `authServerMetadataUrl` in the `oauth` object of your server's config in `.mcp.json` :

```
{
"mcpServers" : {
"my-server" : {
"type" : "http" ,
"url" : "https://mcp.example.com/mcp" ,
"oauth" : {
"authServerMetadataUrl" : "https://auth.example.com/.well-known/openid-configuration"
}
}
}
}
```

The URL must use `https://` . This option requires Claude Code v2.1.64 or later.

#### Use dynamic headers for custom authentication

If your MCP server uses an authentication scheme other than OAuth (such as Kerberos, short-lived tokens, or an internal SSO), use `headersHelper` to generate request headers at connection time. Claude Code runs the command and merges its output into the connection headers.

```
{
"mcpServers" : {
"internal-api" : {
"type" : "http" ,
"url" : "https://mcp.internal.example.com" ,
"headersHelper" : "/opt/bin/get-mcp-auth-headers.sh"
}
}
}
```

The command can also be inline:

```
{
"mcpServers" : {
"internal-api" : {
"type" : "http" ,
"url" : "https://mcp.internal.example.com" ,
"headersHelper" : "echo '{ \" Authorization \" : \" Bearer ' \" $(get-token) \" ' \" }'"
}
}
}
```

**Requirements:**

- The command must write a JSON object of string key-value pairs to stdout
- The command runs in a shell with a 10-second timeout
- Dynamic headers override any static `headers` with the same name

The helper runs fresh on each connection (at session start and on reconnect). There is no caching, so your script is responsible for any token reuse. Claude Code sets these environment variables when executing the helper:

| Variable                      | Value                      |
|-------------------------------|----------------------------|
| `CLAUDE_CODE_MCP_SERVER_NAME` | the name of the MCP server |
| `CLAUDE_CODE_MCP_SERVER_URL`  | the URL of the MCP server  |

Use these to write a single helper script that serves multiple MCP servers.

`headersHelper` executes arbitrary shell commands. When defined at project or local scope, it only runs after you accept the workspace trust dialog.

### Add MCP servers from JSON configuration

If you have a JSON configuration for an MCP server, you can add it directly:

1

Add an MCP server from JSON

```
### Basic syntax
claude mcp add-json < nam e > '<json>'

### Example: Adding an HTTP server with JSON configuration
claude mcp add-json weather-api '{"type":"http","url":"https://api.weather.com/mcp","headers":{"Authorization":"Bearer token"}}'

### Example: Adding a stdio server with JSON configuration
claude mcp add-json local-weather '{"type":"stdio","command":"/path/to/weather-cli","args":["--api-key","abc123"],"env":{"CACHE_DIR":"/tmp"}}'

### Example: Adding an HTTP server with pre-configured OAuth credentials
claude mcp add-json my-server '{"type":"http","url":"https://mcp.example.com/mcp","oauth":{"clientId":"your-client-id","callbackPort":8080}}' --client-secret
```

2

Verify the server was added

```
claude mcp get weather-api
```

Tips:

- Make sure the JSON is properly escaped in your shell
- The JSON must conform to the MCP server configuration schema
- You can use `--scope user` to add the server to your user configuration instead of the project-specific one

### Import MCP servers from Claude Desktop

If you've already configured MCP servers in Claude Desktop, you can import them:

1

Import servers from Claude Desktop

```
### Basic syntax
claude mcp add-from-claude-desktop
```

2

Select which servers to import

After running the command, you'll see an interactive dialog that allows you to select which servers you want to import.

3

Verify the servers were imported

```
claude mcp list
```

Tips:

- This feature only works on macOS and Windows Subsystem for Linux (WSL)
- It reads the Claude Desktop configuration file from its standard location on those platforms
- Use the `--scope user` flag to add servers to your user configuration
- Imported servers will have the same names as in Claude Desktop
- If servers with the same names already exist, they will get a numerical suffix (for example, `server_1` )

### Use MCP servers from Claude.ai

If you've logged into Claude Code with a [Claude.ai](https://claude.ai/) account, MCP servers you've added in Claude.ai are automatically available in Claude Code:

1

Configure MCP servers in Claude.ai

Add servers at [claude.ai/settings/connectors](https://claude.ai/settings/connectors) . On Team and Enterprise plans, only admins can add servers.

2

Authenticate the MCP server

Complete any required authentication steps in Claude.ai.

3

View and manage servers in Claude Code

In Claude Code, use the command:

```
/mcp
```

Claude.ai servers appear in the list with indicators showing they come from Claude.ai.

To disable claude.ai MCP servers in Claude Code, set the `ENABLE_CLAUDEAI_MCP_SERVERS` environment variable to `false` :

```
ENABLE_CLAUDEAI_MCP_SERVERS = false claude
```

### Use Claude Code as an MCP server

You can use Claude Code itself as an MCP server that other applications can connect to:

```
### Start Claude as a stdio MCP server
claude mcp serve
```

You can use this in Claude Desktop by adding this configuration to claude\_desktop\_config.json:

```
{
"mcpServers" : {
"claude-code" : {
"type" : "stdio" ,
"command" : "claude" ,
"args" : [ "mcp" , "serve" ],
"env" : {}
}
}
}
```

**Configuring the executable path** : The `command` field must reference the Claude Code executable. If the `claude` command is not in your system's PATH, you'll need to specify the full path to the executable. To find the full path:

```
which claude
```

Then use the full path in your configuration:

```
{
"mcpServers" : {
"claude-code" : {
"type" : "stdio" ,
"command" : "/full/path/to/claude" ,
"args" : [ "mcp" , "serve" ],
"env" : {}
}
}
}
```

Without the correct executable path, you'll encounter errors like `spawn claude ENOENT` .

Tips:

- The server provides access to Claude's tools like View, Edit, LS, etc.
- In Claude Desktop, try asking Claude to read files in a directory, make edits, and more.
- Note that this MCP server is only exposing Claude Code's tools to your MCP client, so your own client is responsible for implementing user confirmation for individual tool calls.

### MCP output limits and warnings

When MCP tools produce large outputs, Claude Code helps manage the token usage to prevent overwhelming your conversation context:

- **Output warning threshold** : Claude Code displays a warning when any MCP tool output exceeds 10,000 tokens
- **Configurable limit** : you can adjust the maximum allowed MCP output tokens using the `MAX_MCP_OUTPUT_TOKENS` environment variable
- **Default limit** : the default maximum is 25,000 tokens
- **Scope** : the environment variable applies to tools that don't declare their own limit. Tools that set [`anthropic/maxResultSizeChars`](#raise-the-limit-for-a-specific-tool) use that value instead for text content, regardless of what `MAX_MCP_OUTPUT_TOKENS` is set to. Tools that return image data are still subject to `MAX_MCP_OUTPUT_TOKENS`

To increase the limit for tools that produce large outputs:

```
export MAX_MCP_OUTPUT_TOKENS = 50000
claude
```

This is particularly useful when working with MCP servers that:

- Query large datasets or databases
- Generate detailed reports or documentation
- Process extensive log files or debugging information

#### Raise the limit for a specific tool

If you're building an MCP server, you can allow individual tools to return results larger than the default persist-to-disk threshold by setting `_meta["anthropic/maxResultSizeChars"]` in the tool's `tools/list` response entry. Claude Code raises that tool's threshold to the annotated value, up to a hard ceiling of 500,000 characters. This is useful for tools that return inherently large but necessary outputs, such as database schemas or full file trees. Without the annotation, results that exceed the default threshold are persisted to disk and replaced with a file reference in the conversation.

```
{
"name" : "get_schema" ,
"description" : "Returns the full database schema" ,
"_meta" : {
"anthropic/maxResultSizeChars" : 200000
}
}
```

The annotation applies independently of `MAX_MCP_OUTPUT_TOKENS` for text content, so users don't need to raise the environment variable for tools that declare it. Tools that return image data are still subject to the token limit.

If you frequently encounter output warnings with specific MCP servers you don't control, consider increasing the `MAX_MCP_OUTPUT_TOKENS` limit. You can also ask the server author to add the `anthropic/maxResultSizeChars` annotation or to paginate their responses. The annotation has no effect on tools that return image content; for those, raising `MAX_MCP_OUTPUT_TOKENS` is the only option.

### Respond to MCP elicitation requests

MCP servers can request structured input from you mid-task using elicitation. When a server needs information it can't get on its own, Claude Code displays an interactive dialog and passes your response back to the server. No configuration is required on your side: elicitation dialogs appear automatically when a server requests them. Servers can request input in two ways:

- **Form mode** : Claude Code shows a dialog with form fields defined by the server (for example, a username and password prompt). Fill in the fields and submit.
- **URL mode** : Claude Code opens a browser URL for authentication or approval. Complete the flow in the browser, then confirm in the CLI.

To auto-respond to elicitation requests without showing a dialog, use the [`Elicitation`](/docs/en/hooks#elicitation) [hook](/docs/en/hooks#elicitation) . If you're building an MCP server that uses elicitation, see the [MCP elicitation specification](https://modelcontextprotocol.io/docs/learn/client-concepts#elicitation) for protocol details and schema examples.

### Use MCP resources

MCP servers can expose resources that you can reference using @ mentions, similar to how you reference files.

#### Reference MCP resources

1

List available resources

Type `@` in your prompt to see available resources from all connected MCP servers. Resources appear alongside files in the autocomplete menu.

2

Reference a specific resource

Use the format `@server:protocol://resource/path` to reference a resource:

```
Can you analyze @github:issue://123 and suggest a fix?
```

```
Please review the API documentation at @docs:file://api/authentication
```

3

Multiple resource references

You can reference multiple resources in a single prompt:

```
Compare @postgres:schema://users with @docs:file://database/user-model
```

Tips:

- Resources are automatically fetched and included as attachments when referenced
- Resource paths are fuzzy-searchable in the @ mention autocomplete
- Claude Code automatically provides tools to list and read MCP resources when servers support them
- Resources can contain any type of content that the MCP server provides (text, JSON, structured data, etc.)

### Scale with MCP Tool Search

Tool search keeps MCP context usage low by deferring tool definitions until Claude needs them. Only tool names load at session start, so adding more MCP servers has minimal impact on your context window.

#### How it works

Tool search is enabled by default. MCP tools are deferred rather than loaded into context upfront, and Claude uses a search tool to discover relevant ones when a task needs them. Only the tools Claude actually uses enter context. From your perspective, MCP tools work exactly as before. If you prefer threshold-based loading, set `ENABLE_TOOL_SEARCH=auto` to load schemas upfront when they fit within 10% of the context window and defer only the overflow. See [Configure tool search](#configure-tool-search) for all options.

#### For MCP server authors

If you're building an MCP server, the server instructions field becomes more useful with Tool Search enabled. Server instructions help Claude understand when to search for your tools, similar to how [skills](/docs/en/skills) work. Add clear, descriptive server instructions that explain:

- What category of tasks your tools handle
- When Claude should search for your tools
- Key capabilities your server provides

Claude Code truncates tool descriptions and server instructions at 2KB each. Keep them concise to avoid truncation, and put critical details near the start.

#### Configure tool search

Tool search is enabled by default: MCP tools are deferred and discovered on demand. When `ANTHROPIC_BASE_URL` points to a non-first-party host, tool search is disabled by default because most proxies do not forward `tool_reference` blocks. Set `ENABLE_TOOL_SEARCH` explicitly if your proxy does. This feature requires models that support `tool_reference` blocks: Sonnet 4 and later, or Opus 4 and later. Haiku models do not support tool search. Control tool search behavior with the `ENABLE_TOOL_SEARCH` environment variable:

| Value      | Behavior                                                                                                                       |
|------------|--------------------------------------------------------------------------------------------------------------------------------|
| (unset)    | All MCP tools deferred and loaded on demand. Falls back to loading upfront when `ANTHROPIC_BASE_URL` is a non-first-party host |
| `true`     | All MCP tools deferred, including for non-first-party `ANTHROPIC_BASE_URL`                                                     |
| `auto`     | Threshold mode: tools load upfront if they fit within 10% of the context window, deferred otherwise                            |
| `auto:<N>` | Threshold mode with a custom percentage, where `<N>` is 0-100 (e.g., `auto:5` for 5%)                                          |
| `false`    | All MCP tools loaded upfront, no deferral                                                                                      |

```
### Use a custom 5% threshold
ENABLE_TOOL_SEARCH = auto:5 claude

### Disable tool search entirely
ENABLE_TOOL_SEARCH = false claude
```

Or set the value in your [settings.json](/docs/en/settings#available-settings) [`env`](/docs/en/settings#available-settings) [field](/docs/en/settings#available-settings) . You can also disable the `ToolSearch` tool specifically:

```
{
"permissions" : {
"deny" : [ "ToolSearch" ]
}
}
```

### Use MCP prompts as commands

MCP servers can expose prompts that become available as commands in Claude Code.

#### Execute MCP prompts

1

Discover available prompts

Type `/` to see all available commands, including those from MCP servers. MCP prompts appear with the format `/mcp__servername__promptname` .

2

Execute a prompt without arguments

```
/mcp__github__list_prs
```

3

Execute a prompt with arguments

Many prompts accept arguments. Pass them space-separated after the command:

```
/mcp__github__pr_review 456
```

```
/mcp__jira__create_issue "Bug in login flow" high
```

Tips:

- MCP prompts are dynamically discovered from connected servers
- Arguments are parsed based on the prompt's defined parameters
- Prompt results are injected directly into the conversation
- Server and prompt names are normalized (spaces become underscores)

### Managed MCP configuration

For organizations that need centralized control over MCP servers, Claude Code supports two configuration options:

1. **Exclusive control with** **`managed-mcp.json`** : Deploy a fixed set of MCP servers that users cannot modify or extend
2. **Policy-based control with allowlists/denylists** : Allow users to add their own servers, but restrict which ones are permitted

These options allow IT administrators to:

- **Control which MCP servers employees can access** : Deploy a standardized set of approved MCP servers across the organization
- **Prevent unauthorized MCP servers** : Restrict users from adding unapproved MCP servers
- **Disable MCP entirely** : Remove MCP functionality completely if needed

#### Option 1: Exclusive control with managed-mcp.json

When you deploy a `managed-mcp.json` file, it takes **exclusive control** over all MCP servers. Users cannot add, modify, or use any MCP servers other than those defined in this file. This is the simplest approach for organizations that want complete control. System administrators deploy the configuration file to a system-wide directory:

- macOS: `/Library/Application Support/ClaudeCode/managed-mcp.json`
- Linux and WSL: `/etc/claude-code/managed-mcp.json`
- Windows: `C:\Program Files\ClaudeCode\managed-mcp.json`

These are system-wide paths (not user home directories like `~/Library/...` ) that require administrator privileges. They are designed to be deployed by IT administrators.

The `managed-mcp.json` file uses the same format as a standard `.mcp.json` file:

```
{
"mcpServers" : {
"github" : {
"type" : "http" ,
"url" : "https://api.githubcopilot.com/mcp/"
},
"sentry" : {
"type" : "http" ,
"url" : "https://mcp.sentry.dev/mcp"
},
"company-internal" : {
"type" : "stdio" ,
"command" : "/usr/local/bin/company-mcp-server" ,
"args" : [ "--config" , "/etc/company/mcp-config.json" ],
"env" : {
"COMPANY_API_URL" : "https://internal.company.com"
}
}
}
}
```

#### Option 2: Policy-based control with allowlists and denylists

Instead of taking exclusive control, administrators can allow users to configure their own MCP servers while enforcing restrictions on which servers are permitted. This approach uses `allowedMcpServers` and `deniedMcpServers` in the [managed settings file](/docs/en/settings#settings-files) .

**Choosing between options** : Use Option 1 ( `managed-mcp.json` ) when you want to deploy a fixed set of servers with no user customization. Use Option 2 (allowlists/denylists) when you want to allow users to add their own servers within policy constraints.

##### Restriction options

Each entry in the allowlist or denylist can restrict servers in three ways:

1. **By server name** ( `serverName` ): Matches the configured name of the server
2. **By command** ( `serverCommand` ): Matches the exact command and arguments used to start stdio servers
3. **By URL pattern** ( `serverUrl` ): Matches remote server URLs with wildcard support

**Important** : Each entry must have exactly one of `serverName` , `serverCommand` , or `serverUrl` .

##### Example configuration

```
{
"allowedMcpServers" : [
// Allow by server name
{ "serverName" : "github" },
{ "serverName" : "sentry" },

// Allow by exact command (for stdio servers)
{ "serverCommand" : [ "npx" , "-y" , "@modelcontextprotocol/server-filesystem" ] },
{ "serverCommand" : [ "python" , "/usr/local/bin/approved-server.py" ] },

// Allow by URL pattern (for remote servers)
{ "serverUrl" : "https://mcp.company.com/*" },
{ "serverUrl" : "https://*.internal.corp/*" }
],
"deniedMcpServers" : [
// Block by server name
{ "serverName" : "dangerous-server" },

// Block by exact command (for stdio servers)
{ "serverCommand" : [ "npx" , "-y" , "unapproved-package" ] },

// Block by URL pattern (for remote servers)
{ "serverUrl" : "https://*.untrusted.com/*" }
]
}
```

##### How command-based restrictions work

**Exact matching** :

- Command arrays must match **exactly** - both the command and all arguments in the correct order
- Example: `["npx", "-y", "server"]` will NOT match `["npx", "server"]` or `["npx", "-y", "server", "--flag"]`

**Stdio server behavior** :

- When the allowlist contains **any** `serverCommand` entries, stdio servers **must** match one of those commands
- Stdio servers cannot pass by name alone when command restrictions are present
- This ensures administrators can enforce which commands are allowed to run

**Non-stdio server behavior** :

- Remote servers (HTTP, SSE, WebSocket) use URL-based matching when `serverUrl` entries exist in the allowlist
- If no URL entries exist, remote servers fall back to name-based matching
- Command restrictions do not apply to remote servers

##### How URL-based restrictions work

URL patterns support wildcards using `*` to match any sequence of characters. This is useful for allowing entire domains or subdomains. **Wildcard examples** :

- `https://mcp.company.com/*` - Allow all paths on a specific domain
- `https://*.example.com/*` - Allow any subdomain of example.com
- `http://localhost:*/*` - Allow any port on localhost

**Remote server behavior** :

- When the allowlist contains **any** `serverUrl` entries, remote servers **must** match one of those URL patterns
- Remote servers cannot pass by name alone when URL restrictions are present
- This ensures administrators can enforce which remote endpoints are allowed

Example: URL-only allowlist

```
{
"allowedMcpServers" : [
{ "serverUrl" : "https://mcp.company.com/*" },
{ "serverUrl" : "https://*.internal.corp/*" }
]
}
```

**Result** :

- HTTP server at `https://mcp.company.com/api` : ✅ Allowed (matches URL pattern)
- HTTP server at `https://api.internal.corp/mcp` : ✅ Allowed (matches wildcard subdomain)
- HTTP server at `https://external.com/mcp` : ❌ Blocked (doesn't match any URL pattern)
- Stdio server with any command: ❌ Blocked (no name or command entries to match)

Example: Command-only allowlist

```
{
"allowedMcpServers" : [
{ "serverCommand" : [ "npx" , "-y" , "approved-package" ] }
]
}
```

**Result** :

- Stdio server with `["npx", "-y", "approved-package"]` : ✅ Allowed (matches command)
- Stdio server with `["node", "server.js"]` : ❌ Blocked (doesn't match command)
- HTTP server named "my-api": ❌ Blocked (no name entries to match)

Example: Mixed name and command allowlist

```
{
"allowedMcpServers" : [
{ "serverName" : "github" },
{ "serverCommand" : [ "npx" , "-y" , "approved-package" ] }
]
}
```

**Result** :

- Stdio server named "local-tool" with `["npx", "-y", "approved-package"]` : ✅ Allowed (matches command)
- Stdio server named "local-tool" with `["node", "server.js"]` : ❌ Blocked (command entries exist but doesn't match)
- Stdio server named "github" with `["node", "server.js"]` : ❌ Blocked (stdio servers must match commands when command entries exist)
- HTTP server named "github": ✅ Allowed (matches name)
- HTTP server named "other-api": ❌ Blocked (name doesn't match)

Example: Name-only allowlist

```
{
"allowedMcpServers" : [
{ "serverName" : "github" },
{ "serverName" : "internal-tool" }
]
}
```

**Result** :

- Stdio server named "github" with any command: ✅ Allowed (no command restrictions)
- Stdio server named "internal-tool" with any command: ✅ Allowed (no command restrictions)
- HTTP server named "github": ✅ Allowed (matches name)
- Any server named "other": ❌ Blocked (name doesn't match)

##### Allowlist behavior ( allowedMcpServers )

- `undefined` (default): No restrictions - users can configure any MCP server
- Empty array `[]` : Complete lockdown - users cannot configure any MCP servers
- List of entries: Users can only configure servers that match by name, command, or URL pattern

##### Denylist behavior ( deniedMcpServers )

- `undefined` (default): No servers are blocked
- Empty array `[]` : No servers are blocked
- List of entries: Specified servers are explicitly blocked across all scopes

##### Important notes

- **Option 1 and Option 2 can be combined** : If `managed-mcp.json` exists, it has exclusive control and users cannot add servers. Allowlists/denylists still apply to the managed servers themselves.
- **Denylist takes absolute precedence** : If a server matches a denylist entry (by name, command, or URL), it will be blocked even if it's on the allowlist
- Name-based, command-based, and URL-based restrictions work together: a server passes if it matches **either** a name entry, a command entry, or a URL pattern (unless blocked by denylist)

**When using** **`managed-mcp.json`** : Users cannot add MCP servers through `claude mcp add` or configuration files. The `allowedMcpServers` and `deniedMcpServers` settings still apply to filter which managed servers are actually loaded.

Was this page helpful?

Yes

No

[Run agent teams](/docs/en/agent-teams) [Discover and install prebuilt plugins](/docs/en/discover-plugins)

⌘ I


### Extend Claude with skills


Create, manage, and share skills to extend Claude's capabilities in Claude Code. Includes custom commands and bundled skills.


Skills extend what Claude can do. Create a `SKILL.md` file with instructions, and Claude adds it to its toolkit. Claude uses skills when relevant, or you can invoke one directly with `/skill-name` . Create a skill when you keep pasting the same playbook, checklist, or multi-step procedure into chat, or when a section of CLAUDE.md has grown into a procedure rather than a fact. Unlike CLAUDE.md content, a skill's body loads only when it's used, so long reference material costs almost nothing until you need it.

For built-in commands like `/help` and `/compact` , and bundled skills like `/debug` and `/simplify` , see the [commands reference](/docs/en/commands) . **Custom commands have been merged into skills.** A file at `.claude/commands/deploy.md` and a skill at `.claude/skills/deploy/SKILL.md` both create `/deploy` and work the same way. Your existing `.claude/commands/` files keep working. Skills add optional features: a directory for supporting files, frontmatter to [control whether you or Claude invokes them](#control-who-invokes-a-skill) , and the ability for Claude to load them automatically when relevant.

Claude Code skills follow the [Agent Skills](https://agentskills.io/) open standard, which works across multiple AI tools. Claude Code extends the standard with additional features like [invocation control](#control-who-invokes-a-skill) , [subagent execution](#run-skills-in-a-subagent) , and [dynamic context injection](#inject-dynamic-context) .

### Bundled skills

Claude Code includes a set of bundled skills that are available in every session, including `/simplify` , `/batch` , `/debug` , `/loop` , and `/claude-api` . Unlike built-in commands, which execute fixed logic directly, bundled skills are prompt-based: they give Claude a detailed playbook and let it orchestrate the work using its tools. You invoke them the same way as any other skill, by typing `/` followed by the skill name. Bundled skills are listed alongside built-in commands in the [commands reference](/docs/en/commands) , marked **Skill** in the Purpose column.

### Getting started

#### Create your first skill

This example creates a skill that teaches Claude to explain code using visual diagrams and analogies. Since it uses default frontmatter, Claude can load it automatically when you ask how something works, or you can invoke it directly with `/explain-code` .

1

Create the skill directory

Create a directory for the skill in your personal skills folder. Personal skills are available across all your projects.

```
mkdir -p ~/.claude/skills/explain-code
```

2

Write SKILL.md

Every skill needs a `SKILL.md` file with two parts: YAML frontmatter (between `---` markers) that tells Claude when to use the skill, and markdown content with instructions Claude follows when the skill is invoked. The `name` field becomes the `/slash-command` , and the `description` helps Claude decide when to load it automatically. Create `~/.claude/skills/explain-code/SKILL.md` :

```
---
name : explain-code
description : Explains code with visual diagrams and analogies. Use when explaining how code works, teaching about a codebase, or when the user asks "how does this work?"

When explaining code, always include :

1. **Start with an analogy** : Compare the code to something from everyday life
2. **Draw a diagram** : Use ASCII art to show the flow, structure, or relationships
3. **Walk through the code** : Explain step-by-step what happens
4. **Highlight a gotcha** : What's a common mistake or misconception?

Keep explanations conversational. For complex concepts, use multiple analogies.
```

3

Test the skill

You can test it two ways: **Let Claude invoke it automatically** by asking something that matches the description:

```
How does this code work?
```

**Or invoke it directly** with the skill name:

```
/explain-code src/auth/login.ts
```

Either way, Claude should include an analogy and ASCII diagram in its explanation.

#### Where skills live

Where you store a skill determines who can use it:

| Location   | Path                                                     | Applies to                     |
|------------|----------------------------------------------------------|--------------------------------|
| Enterprise | See [managed settings](/docs/en/settings#settings-files) | All users in your organization |
| Personal   | `~/.claude/skills/<skill-name>/SKILL.md`                 | All your projects              |
| Project    | `.claude/skills/<skill-name>/SKILL.md`                   | This project only              |
| Plugin     | `<plugin>/skills/<skill-name>/SKILL.md`                  | Where plugin is enabled        |

When skills share the same name across levels, higher-priority locations win: enterprise > personal > project. Plugin skills use a `plugin-name:skill-name` namespace, so they cannot conflict with other levels. If you have files in `.claude/commands/` , those work the same way, but if a skill and a command share the same name, the skill takes precedence.

##### Automatic discovery from nested directories

When you work with files in subdirectories, Claude Code automatically discovers skills from nested `.claude/skills/` directories. For example, if you're editing a file in `packages/frontend/` , Claude Code also looks for skills in `packages/frontend/.claude/skills/` . This supports monorepo setups where packages have their own skills. Each skill is a directory with `SKILL.md` as the entrypoint:

```
my-skill/
├── SKILL.md           # Main instructions (required)
├── template.md        # Template for Claude to fill in
├── examples/
│   └── sample.md      # Example output showing expected format
└── scripts/
└── validate.sh    # Script Claude can execute
```

The `SKILL.md` contains the main instructions and is required. Other files are optional and let you build more powerful skills: templates for Claude to fill in, example outputs showing the expected format, scripts Claude can execute, or detailed reference documentation. Reference these files from your `SKILL.md` so Claude knows what they contain and when to load them. See [Add supporting files](#add-supporting-files) for more details.

Files in `.claude/commands/` still work and support the same [frontmatter](#frontmatter-reference) . Skills are recommended since they support additional features like supporting files.

##### Skills from additional directories

The `--add-dir` flag [grants file access](/docs/en/permissions#additional-directories-grant-file-access-not-configuration) rather than configuration discovery, but skills are an exception: `.claude/skills/` within an added directory is loaded automatically and picked up by live change detection, so you can edit those skills during a session without restarting. Other `.claude/` configuration such as subagents, commands, and output styles is not loaded from additional directories. See the [exceptions table](/docs/en/permissions#additional-directories-grant-file-access-not-configuration) for the complete list of what is and isn't loaded, and the recommended ways to share configuration across projects.

CLAUDE.md files from `--add-dir` directories are not loaded by default. To load them, set `CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD=1` . See [Load from additional directories](/docs/en/memory#load-from-additional-directories) .

### Configure skills

Skills are configured through YAML frontmatter at the top of `SKILL.md` and the markdown content that follows.

#### Types of skill content

Skill files can contain any instructions, but thinking about how you want to invoke them helps guide what to include: **Reference content** adds knowledge Claude applies to your current work. Conventions, patterns, style guides, domain knowledge. This content runs inline so Claude can use it alongside your conversation context.

```
---
name : api-conventions
description : API design patterns for this codebase

When writing API endpoints :
- Use RESTful naming conventions
- Return consistent error formats
- Include request validation
```

**Task content** gives Claude step-by-step instructions for a specific action, like deployments, commits, or code generation. These are often actions you want to invoke directly with `/skill-name` rather than letting Claude decide when to run them. Add `disable-model-invocation: true` to prevent Claude from triggering it automatically.

```
---
name : deploy
description : Deploy the application to production
context : fork
disable-model-invocation : true

Deploy the application :
1. Run the test suite
2. Build the application
3. Push to the deployment target
```

Your `SKILL.md` can contain anything, but thinking through how you want the skill invoked (by you, by Claude, or both) and where you want it to run (inline or in a subagent) helps guide what to include. For complex skills, you can also [add supporting files](#add-supporting-files) to keep the main skill focused.

#### Frontmatter reference

Beyond the markdown content, you can configure skill behavior using YAML frontmatter fields between `---` markers at the top of your `SKILL.md` file:

```
---
name : my-skill
description : What this skill does
disable-model-invocation : true
allowed-tools : Read Grep

Your skill instructions here...
```

All fields are optional. Only `description` is recommended so Claude knows when to use the skill.

| Field                      | Required    | Description                                                                                                                                                                                                                                                                                     |
|----------------------------|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `name`                     | No          | Display name for the skill. If omitted, uses the directory name. Lowercase letters, numbers, and hyphens only (max 64 characters).                                                                                                                                                              |
| `description`              | Recommended | What the skill does and when to use it. Claude uses this to decide when to apply the skill. If omitted, uses the first paragraph of markdown content. Front-load the key use case: descriptions longer than 250 characters are truncated in the skill listing to reduce context usage.          |
| `argument-hint`            | No          | Hint shown during autocomplete to indicate expected arguments. Example: `[issue-number]` or `[filename] [format]` .                                                                                                                                                                             |
| `disable-model-invocation` | No          | Set to `true` to prevent Claude from automatically loading this skill. Use for workflows you want to trigger manually with `/name` . Default: `false` .                                                                                                                                         |
| `user-invocable`           | No          | Set to `false` to hide from the `/` menu. Use for background knowledge users shouldn't invoke directly. Default: `true` .                                                                                                                                                                       |
| `allowed-tools`            | No          | Tools Claude can use without asking permission when this skill is active. Accepts a space-separated string or a YAML list.                                                                                                                                                                      |
| `model`                    | No          | Model to use when this skill is active.                                                                                                                                                                                                                                                         |
| `effort`                   | No          | [Effort level](/docs/en/model-config#adjust-effort-level) when this skill is active. Overrides the session effort level. Default: inherits from session. Options: `low` , `medium` , `high` , `max` (Opus 4.6 only).                                                                            |
| `context`                  | No          | Set to `fork` to run in a forked subagent context.                                                                                                                                                                                                                                              |
| `agent`                    | No          | Which subagent type to use when `context: fork` is set.                                                                                                                                                                                                                                         |
| `hooks`                    | No          | Hooks scoped to this skill's lifecycle. See [Hooks in skills and agents](/docs/en/hooks#hooks-in-skills-and-agents) for configuration format.                                                                                                                                                   |
| `paths`                    | No          | Glob patterns that limit when this skill is activated. Accepts a comma-separated string or a YAML list. When set, Claude loads the skill automatically only when working with files matching the patterns. Uses the same format as [path-specific rules](/docs/en/memory#path-specific-rules) . |
| `shell`                    | No          | Shell to use for `!`command`` and ````!` blocks in this skill. Accepts `bash` (default) or `powershell` . Setting `powershell` runs inline shell commands via PowerShell on Windows. Requires `CLAUDE_CODE_USE_POWERSHELL_TOOL=1` .                                                             |

##### Available string substitutions

Skills support string substitution for dynamic values in the skill content:

| Variable               | Description                                                                                                                                                                                                                                                                              |
|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `$ARGUMENTS`           | All arguments passed when invoking the skill. If `$ARGUMENTS` is not present in the content, arguments are appended as `ARGUMENTS: <value>` .                                                                                                                                            |
| `$ARGUMENTS[N]`        | Access a specific argument by 0-based index, such as `$ARGUMENTS[0]` for the first argument.                                                                                                                                                                                             |
| `$N`                   | Shorthand for `$ARGUMENTS[N]` , such as `$0` for the first argument or `$1` for the second.                                                                                                                                                                                              |
| `${CLAUDE_SESSION_ID}` | The current session ID. Useful for logging, creating session-specific files, or correlating skill output with sessions.                                                                                                                                                                  |
| `${CLAUDE_SKILL_DIR}`  | The directory containing the skill's `SKILL.md` file. For plugin skills, this is the skill's subdirectory within the plugin, not the plugin root. Use this in bash injection commands to reference scripts or files bundled with the skill, regardless of the current working directory. |

Indexed arguments use shell-style quoting, so wrap multi-word values in quotes to pass them as a single argument. For example, `/my-skill "hello world" second` makes `$0` expand to `hello world` and `$1` to `second` . The `$ARGUMENTS` placeholder always expands to the full argument string as typed. **Example using substitutions:**

```
---
name : session-logger
description : Log activity for this session

Log the following to logs/${CLAUDE_SESSION_ID}.log :

$ARGUMENTS
```

#### Add supporting files

Skills can include multiple files in their directory. This keeps `SKILL.md` focused on the essentials while letting Claude access detailed reference material only when needed. Large reference docs, API specifications, or example collections don't need to load into context every time the skill runs.

```
my-skill/
├── SKILL.md (required - overview and navigation)
├── reference.md (detailed API docs - loaded when needed)
├── examples.md (usage examples - loaded when needed)
└── scripts/
└── helper.py (utility script - executed, not loaded)
```

Reference supporting files from `SKILL.md` so Claude knows what each file contains and when to load it:

```
### Additional resources

- For complete API details, see [ reference.md ]( reference.md )
- For usage examples, see [ examples.md ]( examples.md )
```

Keep `SKILL.md` under 500 lines. Move detailed reference material to separate files.

#### Control who invokes a skill

By default, both you and Claude can invoke any skill. You can type `/skill-name` to invoke it directly, and Claude can load it automatically when relevant to your conversation. Two frontmatter fields let you restrict this:

- **`disable-model-invocation: true`** : Only you can invoke the skill. Use this for workflows with side effects or that you want to control timing, like `/commit` , `/deploy` , or `/send-slack-message` . You don't want Claude deciding to deploy because your code looks ready.
- **`user-invocable: false`** : Only Claude can invoke the skill. Use this for background knowledge that isn't actionable as a command. A `legacy-system-context` skill explains how an old system works. Claude should know this when relevant, but `/legacy-system-context` isn't a meaningful action for users to take.

This example creates a deploy skill that only you can trigger. The `disable-model-invocation: true` field prevents Claude from running it automatically:

```
---
name : deploy
description : Deploy the application to production
disable-model-invocation : true

Deploy $ARGUMENTS to production :

1. Run the test suite
2. Build the application
3. Push to the deployment target
4. Verify the deployment succeeded
```

Here's how the two fields affect invocation and context loading:

| Frontmatter                      | You can invoke   | Claude can invoke   | When loaded into context                                     |
|----------------------------------|------------------|---------------------|--------------------------------------------------------------|
| (default)                        | Yes              | Yes                 | Description always in context, full skill loads when invoked |
| `disable-model-invocation: true` | Yes              | No                  | Description not in context, full skill loads when you invoke |
| `user-invocable: false`          | No               | Yes                 | Description always in context, full skill loads when invoked |

In a regular session, skill descriptions are loaded into context so Claude knows what's available, but full skill content only loads when invoked. [Subagents with preloaded skills](/docs/en/sub-agents#preload-skills-into-subagents) work differently: the full skill content is injected at startup.

#### Skill content lifecycle

When you or Claude invoke a skill, the rendered `SKILL.md` content enters the conversation as a single message and stays there for the rest of the session. Claude Code does not re-read the skill file on later turns, so write guidance that should apply throughout a task as standing instructions rather than one-time steps. [Auto-compaction](/docs/en/how-claude-code-works#when-context-fills-up) carries invoked skills forward within a token budget. When the conversation is summarized to free context, Claude Code re-attaches the most recent invocation of each skill after the summary, keeping the first 5,000 tokens of each. Re-attached skills share a combined budget of 25,000 tokens. Claude Code fills this budget starting from the most recently invoked skill, so older skills can be dropped entirely after compaction if you have invoked many in one session. If a skill seems to stop influencing behavior after the first response, the content is usually still present and the model is choosing other tools or approaches. Strengthen the skill's `description` and instructions so the model keeps preferring it, or use [hooks](/docs/en/hooks) to enforce behavior deterministically. If the skill is large or you invoked several others after it, re-invoke it after compaction to restore the full content.

#### Pre-approve tools for a skill

The `allowed-tools` field grants permission for the listed tools while the skill is active, so Claude can use them without prompting you for approval. It does not restrict which tools are available: every tool remains callable, and your [permission settings](/docs/en/permissions) still govern tools that are not listed. This skill lets Claude run git commands without per-use approval whenever you invoke it:

```
---
name : commit
description : Stage and commit the current changes
disable-model-invocation : true
allowed-tools : Bash(git add *) Bash(git commit *) Bash(git status *)
---
```

To block a skill from using certain tools, add deny rules in your [permission settings](/docs/en/permissions) instead.

#### Pass arguments to skills

Both you and Claude can pass arguments when invoking a skill. Arguments are available via the `$ARGUMENTS` placeholder. This skill fixes a GitHub issue by number. The `$ARGUMENTS` placeholder gets replaced with whatever follows the skill name:

```
---
name : fix-issue
description : Fix a GitHub issue
disable-model-invocation : true

Fix GitHub issue $ARGUMENTS following our coding standards.

1. Read the issue description
2. Understand the requirements
3. Implement the fix
4. Write tests
5. Create a commit
```

When you run `/fix-issue 123` , Claude receives "Fix GitHub issue 123 following our coding standards..." If you invoke a skill with arguments but the skill doesn't include `$ARGUMENTS` , Claude Code appends `ARGUMENTS: <your input>` to the end of the skill content so Claude still sees what you typed. To access individual arguments by position, use `$ARGUMENTS[N]` or the shorter `$N` :

```
---
name : migrate-component
description : Migrate a component from one framework to another

Migrate the $ARGUMENTS[0] component from $ARGUMENTS[1] to $ARGUMENTS[2].
Preserve all existing behavior and tests.
```

Running `/migrate-component SearchBar React Vue` replaces `$ARGUMENTS[0]` with `SearchBar` , `$ARGUMENTS[1]` with `React` , and `$ARGUMENTS[2]` with `Vue` . The same skill using the `$N` shorthand:

```
---
name : migrate-component
description : Migrate a component from one framework to another

Migrate the $0 component from $1 to $2.
Preserve all existing behavior and tests.
```

### Advanced patterns

#### Inject dynamic context

The `!`<command>`` syntax runs shell commands before the skill content is sent to Claude. The command output replaces the placeholder, so Claude receives actual data, not the command itself. This skill summarizes a pull request by fetching live PR data with the GitHub CLI. The `!`gh pr diff`` and other commands run first, and their output gets inserted into the prompt:

```
---
name : pr-summary
description : Summarize changes in a pull request
context : fork
agent : Explore
allowed-tools : Bash(gh *)

### Pull request context
- PR diff : !`gh pr diff`
- PR comments : !`gh pr view --comments`
- Changed files : !`gh pr diff --name-only`

### Your task
Summarize this pull request...
```

When this skill runs:

1. Each `!`<command>`` executes immediately (before Claude sees anything)
2. The output replaces the placeholder in the skill content
3. Claude receives the fully-rendered prompt with actual PR data

This is preprocessing, not something Claude executes. Claude only sees the final result. For multi-line commands, use a fenced code block opened with ````!` instead of the inline form:

```
### Environment
```!
node --version
npm --version
git status --short
```
```

To disable this behavior for skills and custom commands from user, project, plugin, or [additional-directory](#skills-from-additional-directories) sources, set `"disableSkillShellExecution": true` in [settings](/docs/en/settings) . Each command is replaced with `[shell command execution disabled by policy]` instead of being run. Bundled and managed skills are not affected. This setting is most useful in [managed settings](/docs/en/permissions#managed-settings) , where users cannot override it.

To enable [extended thinking](/docs/en/common-workflows#use-extended-thinking-thinking-mode) in a skill, include the word "ultrathink" anywhere in your skill content.

#### Run skills in a subagent

Add `context: fork` to your frontmatter when you want a skill to run in isolation. The skill content becomes the prompt that drives the subagent. It won't have access to your conversation history.

`context: fork` only makes sense for skills with explicit instructions. If your skill contains guidelines like "use these API conventions" without a task, the subagent receives the guidelines but no actionable prompt, and returns without meaningful output.

Skills and [subagents](/docs/en/sub-agents) work together in two directions:

| Approach                     | System prompt                                | Task                        | Also loads                   |
|------------------------------|----------------------------------------------|-----------------------------|------------------------------|
| Skill with `context: fork`   | From agent type ( `Explore` , `Plan` , etc.) | SKILL.md content            | CLAUDE.md                    |
| Subagent with `skills` field | Subagent's markdown body                     | Claude's delegation message | Preloaded skills + CLAUDE.md |

With `context: fork` , you write the task in your skill and pick an agent type to execute it. For the inverse (defining a custom subagent that uses skills as reference material), see [Subagents](/docs/en/sub-agents#preload-skills-into-subagents) .

##### Example: Research skill using Explore agent

This skill runs research in a forked Explore agent. The skill content becomes the task, and the agent provides read-only tools optimized for codebase exploration:

```
---
name : deep-research
description : Research a topic thoroughly
context : fork
agent : Explore

Research $ARGUMENTS thoroughly :

1. Find relevant files using Glob and Grep
2. Read and analyze the code
3. Summarize findings with specific file references
```

When this skill runs:

1. A new isolated context is created
2. The subagent receives the skill content as its prompt ("Research $ARGUMENTS thoroughly...")
3. The `agent` field determines the execution environment (model, tools, and permissions)
4. Results are summarized and returned to your main conversation

The `agent` field specifies which subagent configuration to use. Options include built-in agents ( `Explore` , `Plan` , `general-purpose` ) or any custom subagent from `.claude/agents/` . If omitted, uses `general-purpose` .

#### Restrict Claude's skill access

By default, Claude can invoke any skill that doesn't have `disable-model-invocation: true` set. Skills that define `allowed-tools` grant Claude access to those tools without per-use approval when the skill is active. Your [permission settings](/docs/en/permissions) still govern baseline approval behavior for all other tools. Built-in commands like `/compact` and `/init` are not available through the Skill tool. Three ways to control which skills Claude can invoke: **Disable all skills** by denying the Skill tool in `/permissions` :

```
### Add to deny rules:
Skill
```

**Allow or deny specific skills** using [permission rules](/docs/en/permissions) :

```
### Allow only specific skills
Skill(commit)
Skill(review-pr *)

### Deny specific skills
Skill(deploy *)
```

Permission syntax: `Skill(name)` for exact match, `Skill(name *)` for prefix match with any arguments. **Hide individual skills** by adding `disable-model-invocation: true` to their frontmatter. This removes the skill from Claude's context entirely.

The `user-invocable` field only controls menu visibility, not Skill tool access. Use `disable-model-invocation: true` to block programmatic invocation.

### Share skills

Skills can be distributed at different scopes depending on your audience:

- **Project skills** : Commit `.claude/skills/` to version control
- **Plugins** : Create a `skills/` directory in your [plugin](/docs/en/plugins)
- **Managed** : Deploy organization-wide through [managed settings](/docs/en/settings#settings-files)

#### Generate visual output

Skills can bundle and run scripts in any language, giving Claude capabilities beyond what's possible in a single prompt. One powerful pattern is generating visual output: interactive HTML files that open in your browser for exploring data, debugging, or creating reports. This example creates a codebase explorer: an interactive tree view where you can expand and collapse directories, see file sizes at a glance, and identify file types by color. Create the Skill directory:

```
mkdir -p ~/.claude/skills/codebase-visualizer/scripts
```

Create `~/.claude/skills/codebase-visualizer/SKILL.md` . The description tells Claude when to activate this Skill, and the instructions tell Claude to run the bundled script:

```
---
name : codebase-visualizer
description : Generate an interactive collapsible tree visualization of your codebase. Use when exploring a new repo, understanding project structure, or identifying large files.
allowed-tools : Bash(python *)

### Codebase Visualizer

Generate an interactive HTML tree view that shows your project's file structure with collapsible directories.

### Usage

Run the visualization script from your project root :

``` bash
python ~/.claude/skills/codebase-visualizer/scripts/visualize.py .
```

This creates `codebase-map.html` in the current directory and opens it in your default browser.

### What the visualization shows

- * *Collapsible directories** : Click folders to expand/collapse
- * *File sizes** : Displayed next to each file
- * *Colors**: Different colors for different file types
- * *Directory totals** : Shows aggregate size of each folder
```

Create `~/.claude/skills/codebase-visualizer/scripts/visualize.py` . This script scans a directory tree and generates a self-contained HTML file with:

- A **summary sidebar** showing file count, directory count, total size, and number of file types
- A **bar chart** breaking down the codebase by file type (top 8 by size)
- A **collapsible tree** where you can expand and collapse directories, with color-coded file type indicators

The script requires Python but uses only built-in libraries, so there are no packages to install:

```
#!/usr/bin/env python3
"""Generate an interactive collapsible tree visualization of a codebase."""

import json
import sys
import webbrowser
from pathlib import Path
from collections import Counter

IGNORE = { '.git' , 'node_modules' , '__pycache__' , '.venv' , 'venv' , 'dist' , 'build' }

def scan ( path : Path, stats : dict ) -> dict :
result = { "name" : path.name, "children" : [], "size" : 0 }
try :
for item in sorted (path.iterdir()):
if item.name in IGNORE or item.name.startswith( '.' ):
continue
if item.is_file():
size = item.stat().st_size
ext = item.suffix.lower() or '(no ext)'
result[ "children" ].append({ "name" : item.name, "size" : size, "ext" : ext})
result[ "size" ] += size
stats[ "files" ] += 1
stats[ "extensions" ][ext] += 1
stats[ "ext_sizes" ][ext] += size
elif item.is_dir():
stats[ "dirs" ] += 1
child = scan(item, stats)
if child[ "children" ]:
result[ "children" ].append(child)
result[ "size" ] += child[ "size" ]
except PermissionError :
pass
return result

def generate_html ( data : dict , stats : dict , output : Path) -> None :
ext_sizes = stats[ "ext_sizes" ]
total_size = sum (ext_sizes.values()) or 1
sorted_exts = sorted (ext_sizes.items(), key = lambda x : - x[ 1 ])[: 8 ]
colors = {
'.js' : '#f7df1e' , '.ts' : '#3178c6' , '.py' : '#3776ab' , '.go' : '#00add8' ,
'.rs' : '#dea584' , '.rb' : '#cc342d' , '.css' : '#264de4' , '.html' : '#e34c26' ,
'.json' : '#6b7280' , '.md' : '#083fa1' , '.yaml' : '#cb171e' , '.yml' : '#cb171e' ,
'.mdx' : '#083fa1' , '.tsx' : '#3178c6' , '.jsx' : '#61dafb' , '.sh' : '#4eaa25' ,
}
lang_bars = "" .join(
f '<div class="bar-row"><span class="bar-label"> { ext } </span>'
f '<div class="bar" style="width: { (size / total_size) * 100 } %;background: { colors.get(ext, "#6b7280" ) } "></div>'
f '<span class="bar-pct"> { (size / total_size) * 100 :.1f} %</span></div>'
for ext, size in sorted_exts
)
def fmt ( b ):
if b < 1024 : return f " { b } B"
if b < 1048576 : return f " { b / 1024 :.1f} KB"
return f " { b / 1048576 :.1f} MB"

html = f '''<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><title>Codebase Explorer</title>
<style>
body {{ font: 14px/1.5 system-ui, sans-serif; margin: 0; background: #1a1a2e; color: #eee; }}
.container {{ display: flex; height: 100vh; }}
.sidebar {{ width: 280px; background: #252542; padding: 20px; border-right: 1px solid #3d3d5c; overflow-y: auto; flex-shrink: 0; }}
.main {{ flex: 1; padding: 20px; overflow-y: auto; }}
h1 {{ margin: 0 0 10px 0; font-size: 18px; }}
h2 {{ margin: 20px 0 10px 0; font-size: 14px; color: #888; text-transform: uppercase; }}
.stat {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #3d3d5c; }}
.stat-value {{ font-weight: bold; }}
.bar-row {{ display: flex; align-items: center; margin: 6px 0; }}
.bar-label {{ width: 55px; font-size: 12px; color: #aaa; }}
.bar {{ height: 18px; border-radius: 3px; }}
.bar-pct {{ margin-left: 8px; font-size: 12px; color: #666; }}
.tree {{ list-style: none; padding-left: 20px; }}
details {{ cursor: pointer; }}
summary {{ padding: 4px 8px; border-radius: 4px; }}
summary:hover {{ background: #2d2d44; }}
.folder {{ color: #ffd700; }}
.file {{ display: flex; align-items: center; padding: 4px 8px; border-radius: 4px; }}
.file:hover {{ background: #2d2d44; }}
.size {{ color: #888; margin-left: auto; font-size: 12px; }}
.dot {{ width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; }}
</style>
</head><body>
<div class="container">
<div class="sidebar">
<h1>📊 Summary</h1>
<div class="stat"><span>Files</span><span class="stat-value"> { stats[ "files" ] :,} </span></div>
<div class="stat"><span>Directories</span><span class="stat-value"> { stats[ "dirs" ] :,} </span></div>
<div class="stat"><span>Total size</span><span class="stat-value"> { fmt(data[ "size" ]) } </span></div>
<div class="stat"><span>File types</span><span class="stat-value"> { len (stats[ "extensions" ]) } </span></div>
<h2>By file type</h2>
{ lang_bars }
</div>
<div class="main">
<h1>📁 { data[ "name" ] } </h1>
<ul class="tree" id="root"></ul>
</div>
</div>
<script>
const data = { json.dumps(data) } ;
const colors = { json.dumps(colors) } ;
function fmt(b) {{ if (b < 1024) return b + ' B'; if (b < 1048576) return (b/1024).toFixed(1) + ' KB'; return (b/1048576).toFixed(1) + ' MB'; }}
function render(node, parent) {{
if (node.children) {{
const det = document.createElement('details');
det.open = parent === document.getElementById('root');
det.innerHTML = `<summary><span class="folder">📁 $ {{ node.name }} </span><span class="size">$ {{ fmt(node.size) }} </span></summary>`;
const ul = document.createElement('ul'); ul.className = 'tree';
node.children.sort((a,b) => (b.children?1:0)-(a.children?1:0) || a.name.localeCompare(b.name));
node.children.forEach(c => render(c, ul));
det.appendChild(ul);
const li = document.createElement('li'); li.appendChild(det); parent.appendChild(li);
}} else {{
const li = document.createElement('li'); li.className = 'file';
li.innerHTML = `<span class="dot" style="background:$ {{ colors[node.ext]||'#6b7280' }} "></span>$ {{ node.name }} <span class="size">$ {{ fmt(node.size) }} </span>`;
parent.appendChild(li);
}}
}}
data.children.forEach(c => render(c, document.getElementById('root')));
</script>
</body></html>'''
output.write_text(html)

if __name__ == '__main__' :
target = Path(sys.argv[ 1 ] if len (sys.argv) > 1 else '.' ).resolve()
stats = { "files" : 0 , "dirs" : 0 , "extensions" : Counter(), "ext_sizes" : Counter()}
data = scan(target, stats)
out = Path( 'codebase-map.html' )
generate_html(data, stats, out)
print ( f 'Generated { out.absolute() } ' )
webbrowser.open( f 'file:// { out.absolute() } ' )
```

See all 131 lines

To test, open Claude Code in any project and ask "Visualize this codebase." Claude runs the script, generates `codebase-map.html` , and opens it in your browser. This pattern works for any visual output: dependency graphs, test coverage reports, API documentation, or database schema visualizations. The bundled script does the heavy lifting while Claude handles orchestration.

### Troubleshooting

#### Skill not triggering

If Claude doesn't use your skill when expected:

1. Check the description includes keywords users would naturally say
2. Verify the skill appears in `What skills are available?`
3. Try rephrasing your request to match the description more closely
4. Invoke it directly with `/skill-name` if the skill is user-invocable

#### Skill triggers too often

If Claude uses your skill when you don't want it:

1. Make the description more specific
2. Add `disable-model-invocation: true` if you only want manual invocation

#### Skill descriptions are cut short

Skill descriptions are loaded into context so Claude knows what's available. All skill names are always included, but if you have many skills, descriptions are shortened to fit the character budget, which can strip the keywords Claude needs to match your request. The budget scales dynamically at 1% of the context window, with a fallback of 8,000 characters. To raise the limit, set the `SLASH_COMMAND_TOOL_CHAR_BUDGET` environment variable. Or trim descriptions at the source: front-load the key use case, since each entry is capped at 250 characters regardless of budget.

### Related resources

- [**Subagents**](/docs/en/sub-agents) : delegate tasks to specialized agents
- [**Plugins**](/docs/en/plugins) : package and distribute skills with other extensions
- [**Hooks**](/docs/en/hooks) : automate workflows around tool events
- [**Memory**](/docs/en/memory) : manage CLAUDE.md files for persistent context
- [**Commands**](/docs/en/commands) : reference for built-in commands and bundled skills
- [**Permissions**](/docs/en/permissions) : control tool and skill access

Was this page helpful?

Yes

No

[Create plugins](/docs/en/plugins) [Automate with hooks](/docs/en/hooks-guide)

⌘ I


---

# Hooks & Automation


### Hooks reference


Reference for Claude Code hook events, configuration schema, JSON input/output formats, exit codes, async hooks, HTTP hooks, prompt hooks, and MCP tool hooks.


For a quickstart guide with examples, see [Automate workflows with hooks](/docs/en/hooks-guide) .

Hooks are user-defined shell commands, HTTP endpoints, or LLM prompts that execute automatically at specific points in Claude Code's lifecycle. Use this reference to look up event schemas, configuration options, JSON input/output formats, and advanced features like async hooks, HTTP hooks, and MCP tool hooks. If you're setting up hooks for the first time, start with the [guide](/docs/en/hooks-guide) instead.

### Hook lifecycle

Hooks fire at specific points during a Claude Code session. When an event fires and a matcher matches, Claude Code passes JSON context about the event to your hook handler. For command hooks, input arrives on stdin. For HTTP hooks, it arrives as the POST request body. Your handler can then inspect the input, take action, and optionally return a decision. Events fall into three cadences: once per session ( `SessionStart` , `SessionEnd` ), once per turn ( `UserPromptSubmit` , `Stop` , `StopFailure` ), and on every tool call inside the agentic loop ( `PreToolUse` , `PostToolUse` ):

Hook lifecycle diagram showing SessionStart, then a per-turn loop containing UserPromptSubmit, the nested agentic loop (PreToolUse, PermissionRequest, PostToolUse, SubagentStart/Stop, TaskCreated, TaskCompleted), and Stop or StopFailure, followed by TeammateIdle, PreCompact, PostCompact, and SessionEnd, with Elicitation and ElicitationResult nested inside MCP tool execution, PermissionDenied as a side branch from PermissionRequest for auto-mode denials, and WorktreeCreate, WorktreeRemove, Notification, ConfigChange, InstructionsLoaded, CwdChanged, and FileChanged as standalone async events


The table below summarizes when each event fires. The [Hook events](#hook-events) section documents the full input schema and decision control options for each one.

| Event                | When it fires                                                                                                                                          |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `SessionStart`       | When a session begins or resumes                                                                                                                       |
| `UserPromptSubmit`   | When you submit a prompt, before Claude processes it                                                                                                   |
| `PreToolUse`         | Before a tool call executes. Can block it                                                                                                              |
| `PermissionRequest`  | When a permission dialog appears                                                                                                                       |
| `PermissionDenied`   | When a tool call is denied by the auto mode classifier. Return `{retry: true}` to tell the model it may retry the denied tool call                     |
| `PostToolUse`        | After a tool call succeeds                                                                                                                             |
| `PostToolUseFailure` | After a tool call fails                                                                                                                                |
| `Notification`       | When Claude Code sends a notification                                                                                                                  |
| `SubagentStart`      | When a subagent is spawned                                                                                                                             |
| `SubagentStop`       | When a subagent finishes                                                                                                                               |
| `TaskCreated`        | When a task is being created via `TaskCreate`                                                                                                          |
| `TaskCompleted`      | When a task is being marked as completed                                                                                                               |
| `Stop`               | When Claude finishes responding                                                                                                                        |
| `StopFailure`        | When the turn ends due to an API error. Output and exit code are ignored                                                                               |
| `TeammateIdle`       | When an [agent team](/docs/en/agent-teams) teammate is about to go idle                                                                                |
| `InstructionsLoaded` | When a CLAUDE.md or `.claude/rules/*.md` file is loaded into context. Fires at session start and when files are lazily loaded during a session         |
| `ConfigChange`       | When a configuration file changes during a session                                                                                                     |
| `CwdChanged`         | When the working directory changes, for example when Claude executes a `cd` command. Useful for reactive environment management with tools like direnv |
| `FileChanged`        | When a watched file changes on disk. The `matcher` field specifies which filenames to watch                                                            |
| `WorktreeCreate`     | When a worktree is being created via `--worktree` or `isolation: "worktree"` . Replaces default git behavior                                           |
| `WorktreeRemove`     | When a worktree is being removed, either at session exit or when a subagent finishes                                                                   |
| `PreCompact`         | Before context compaction                                                                                                                              |
| `PostCompact`        | After context compaction completes                                                                                                                     |
| `Elicitation`        | When an MCP server requests user input during a tool call                                                                                              |
| `ElicitationResult`  | After a user responds to an MCP elicitation, before the response is sent back to the server                                                            |
| `SessionEnd`         | When a session terminates                                                                                                                              |

#### How a hook resolves

To see how these pieces fit together, consider this `PreToolUse` hook that blocks destructive shell commands. The `matcher` narrows to Bash tool calls and the `if` condition narrows further to commands starting with `rm` , so `block-rm.sh` only spawns when both filters match:

```
{
"hooks" : {
"PreToolUse" : [
{
"matcher" : "Bash" ,
"hooks" : [
{
"type" : "command" ,
"if" : "Bash(rm *)" ,
"command" : " \" $CLAUDE_PROJECT_DIR \" /.claude/hooks/block-rm.sh"
}
]
}
]
}
}
```

The script reads the JSON input from stdin, extracts the command, and returns a `permissionDecision` of `"deny"` if it contains `rm -rf` :

```
#!/bin/bash
### .claude/hooks/block-rm.sh
COMMAND = $( jq -r '.tool_input.command' )

if echo " $COMMAND " | grep -q 'rm -rf' ; then
jq -n '{
hookSpecificOutput: {
hookEventName: "PreToolUse",
permissionDecision: "deny",
permissionDecisionReason: "Destructive command blocked by hook"
}
}'
else
exit 0 # allow the command
fi
```

Now suppose Claude Code decides to run `Bash "rm -rf /tmp/build"` . Here's what happens:

Hook resolution flow: PreToolUse event fires, matcher checks for Bash match, if condition checks for Bash(rm *) match, hook handler runs, result returns to Claude Code


1

Event fires

The `PreToolUse` event fires. Claude Code sends the tool input as JSON on stdin to the hook:

```
{ "tool_name" : "Bash" , "tool_input" : { "command" : "rm -rf /tmp/build" }, ... }
```

2

Matcher checks

The matcher `"Bash"` matches the tool name, so this hook group activates. If you omit the matcher or use `"*"` , the group activates on every occurrence of the event.

3

If condition checks

The `if` condition `"Bash(rm *)"` matches because the command starts with `rm` , so this handler spawns. If the command had been `npm test` , the `if` check would fail and `block-rm.sh` would never run, avoiding the process spawn overhead. The `if` field is optional; without it, every handler in the matched group runs.

4

Hook handler runs

The script inspects the full command and finds `rm -rf` , so it prints a decision to stdout:

```
{
"hookSpecificOutput" : {
"hookEventName" : "PreToolUse" ,
"permissionDecision" : "deny" ,
"permissionDecisionReason" : "Destructive command blocked by hook"
}
}
```

If the command had been a safer `rm` variant like `rm file.txt` , the script would hit `exit 0` instead, which tells Claude Code to allow the tool call with no further action.

5

Claude Code acts on the result

Claude Code reads the JSON decision, blocks the tool call, and shows Claude the reason.

The [Configuration](#configuration) section below documents the full schema, and each [hook event](#hook-events) section documents what input your command receives and what output it can return.

### Configuration

Hooks are defined in JSON settings files. The configuration has three levels of nesting:

1. Choose a [hook event](#hook-events) to respond to, like `PreToolUse` or `Stop`
2. Add a [matcher group](#matcher-patterns) to filter when it fires, like "only for the Bash tool"
3. Define one or more [hook handlers](#hook-handler-fields) to run when matched

See [How a hook resolves](#how-a-hook-resolves) above for a complete walkthrough with an annotated example.

This page uses specific terms for each level: **hook event** for the lifecycle point, **matcher group** for the filter, and **hook handler** for the shell command, HTTP endpoint, prompt, or agent that runs. "Hook" on its own refers to the general feature.

#### Hook locations

Where you define a hook determines its scope:

| Location                                                             | Scope                         | Shareable                          |
|----------------------------------------------------------------------|-------------------------------|------------------------------------|
| `~/.claude/settings.json`                                            | All your projects             | No, local to your machine          |
| `.claude/settings.json`                                              | Single project                | Yes, can be committed to the repo  |
| `.claude/settings.local.json`                                        | Single project                | No, gitignored                     |
| Managed policy settings                                              | Organization-wide             | Yes, admin-controlled              |
| [Plugin](/docs/en/plugins) `hooks/hooks.json`                        | When plugin is enabled        | Yes, bundled with the plugin       |
| [Skill](/docs/en/skills) or [agent](/docs/en/sub-agents) frontmatter | While the component is active | Yes, defined in the component file |

For details on settings file resolution, see [settings](/docs/en/settings) . Enterprise administrators can use `allowManagedHooksOnly` to block user, project, and plugin hooks. Hooks from plugins force-enabled in managed settings `enabledPlugins` are exempt, so administrators can distribute vetted hooks through an organization marketplace. See [Hook configuration](/docs/en/settings#hook-configuration) .

#### Matcher patterns

The `matcher` field filters when hooks fire. How a matcher is evaluated depends on the characters it contains:

| Matcher value                            | Evaluated as                                               | Example                                                                                                            |
|------------------------------------------|------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `"*"` , `""` , or omitted                | Match all                                                  | fires on every occurrence of the event                                                                             |
| Only letters, digits, `_` , and `|` | Exact string, or `|` -separated list of exact strings | `Bash` matches only the Bash tool; `Edit|Write` matches either tool exactly                                   |
| Contains any other character             | JavaScript regular expression                              | `^Notebook` matches any tool starting with Notebook; `mcp__memory__.*` matches every tool from the `memory` server |

The `FileChanged` event does not follow these rules when building its watch list. See [FileChanged](#filechanged) . Each event type matches on a different field:

| Event                                                                                                                | What the matcher filters                                      | Example matcher values                                                                                                          |
|----------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| `PreToolUse` , `PostToolUse` , `PostToolUseFailure` , `PermissionRequest` , `PermissionDenied`                       | tool name                                                     | `Bash` , `Edit|Write` , `mcp__.*`                                                                                          |
| `SessionStart`                                                                                                       | how the session started                                       | `startup` , `resume` , `clear` , `compact`                                                                                      |
| `SessionEnd`                                                                                                         | why the session ended                                         | `clear` , `resume` , `logout` , `prompt_input_exit` , `bypass_permissions_disabled` , `other`                                   |
| `Notification`                                                                                                       | notification type                                             | `permission_prompt` , `idle_prompt` , `auth_success` , `elicitation_dialog`                                                     |
| `SubagentStart`                                                                                                      | agent type                                                    | `Bash` , `Explore` , `Plan` , or custom agent names                                                                             |
| `PreCompact` , `PostCompact`                                                                                         | what triggered compaction                                     | `manual` , `auto`                                                                                                               |
| `SubagentStop`                                                                                                       | agent type                                                    | same values as `SubagentStart`                                                                                                  |
| `ConfigChange`                                                                                                       | configuration source                                          | `user_settings` , `project_settings` , `local_settings` , `policy_settings` , `skills`                                          |
| `CwdChanged`                                                                                                         | no matcher support                                            | always fires on every directory change                                                                                          |
| `FileChanged`                                                                                                        | literal filenames to watch (see [FileChanged](#filechanged) ) | `.envrc|.env`                                                                                                              |
| `StopFailure`                                                                                                        | error type                                                    | `rate_limit` , `authentication_failed` , `billing_error` , `invalid_request` , `server_error` , `max_output_tokens` , `unknown` |
| `InstructionsLoaded`                                                                                                 | load reason                                                   | `session_start` , `nested_traversal` , `path_glob_match` , `include` , `compact`                                                |
| `Elicitation`                                                                                                        | MCP server name                                               | your configured MCP server names                                                                                                |
| `ElicitationResult`                                                                                                  | MCP server name                                               | same values as `Elicitation`                                                                                                    |
| `UserPromptSubmit` , `Stop` , `TeammateIdle` , `TaskCreated` , `TaskCompleted` , `WorktreeCreate` , `WorktreeRemove` | no matcher support                                            | always fires on every occurrence                                                                                                |

The matcher runs against a field from the [JSON input](#hook-input-and-output) that Claude Code sends to your hook on stdin. For tool events, that field is `tool_name` . Each [hook event](#hook-events) section lists the full set of matcher values and the input schema for that event. This example runs a linting script only when Claude writes or edits a file:

```
{
"hooks" : {
"PostToolUse" : [
{
"matcher" : "Edit|Write" ,
"hooks" : [
{
"type" : "command" ,
"command" : "/path/to/lint-check.sh"
}
]
}
]
}
}
```

`UserPromptSubmit` , `Stop` , `TeammateIdle` , `TaskCreated` , `TaskCompleted` , `WorktreeCreate` , `WorktreeRemove` , and `CwdChanged` don't support matchers and always fire on every occurrence. If you add a `matcher` field to these events, it is silently ignored. For tool events, you can filter more narrowly by setting the [`if`](#common-fields) [field](#common-fields) on individual hook handlers. `if` uses [permission rule syntax](/docs/en/permissions) to match against the tool name and arguments together, so `"Bash(git *)"` runs only for `git` commands and `"Edit(*.ts)"` runs only for TypeScript files.

##### Match MCP tools

[MCP](/docs/en/mcp) server tools appear as regular tools in tool events ( `PreToolUse` , `PostToolUse` , `PostToolUseFailure` , `PermissionRequest` , `PermissionDenied` ), so you can match them the same way you match any other tool name. MCP tools follow the naming pattern `mcp__<server>__<tool>` , for example:

- `mcp__memory__create_entities` : Memory server's create entities tool
- `mcp__filesystem__read_file` : Filesystem server's read file tool
- `mcp__github__search_repositories` : GitHub server's search tool

To match every tool from a server, append `.*` to the server prefix. The `.*` is required: a matcher like `mcp__memory` contains only letters and underscores, so it is compared as an exact string and matches no tool.

- `mcp__memory__.*` matches all tools from the `memory` server
- `mcp__.*__write.*` matches any tool whose name starts with `write` from any server

This example logs all memory server operations and validates write operations from any MCP server:

```
{
"hooks" : {
"PreToolUse" : [
{
"matcher" : "mcp__memory__.*" ,
"hooks" : [
{
"type" : "command" ,
"command" : "echo 'Memory operation initiated' >> ~/mcp-operations.log"
}
]
},
{
"matcher" : "mcp__.*__write.*" ,
"hooks" : [
{
"type" : "command" ,
"command" : "/home/user/scripts/validate-mcp-write.py"
}
]
}
]
}
}
```

#### Hook handler fields

Each object in the inner `hooks` array is a hook handler: the shell command, HTTP endpoint, LLM prompt, or agent that runs when the matcher matches. There are four types:

- [**Command hooks**](#command-hook-fields) ( `type: "command"` ): run a shell command. Your script receives the event's [JSON input](#hook-input-and-output) on stdin and communicates results back through exit codes and stdout.
- [**HTTP hooks**](#http-hook-fields) ( `type: "http"` ): send the event's JSON input as an HTTP POST request to a URL. The endpoint communicates results back through the response body using the same [JSON output format](#json-output) as command hooks.
- [**Prompt hooks**](#prompt-and-agent-hook-fields) ( `type: "prompt"` ): send a prompt to a Claude model for single-turn evaluation. The model returns a yes/no decision as JSON. See [Prompt-based hooks](#prompt-based-hooks) .
- [**Agent hooks**](#prompt-and-agent-hook-fields) ( `type: "agent"` ): spawn a subagent that can use tools like Read, Grep, and Glob to verify conditions before returning a decision. See [Agent-based hooks](#agent-based-hooks) .

##### Common fields

These fields apply to all hook types:

| Field           | Required   | Description                                                                                                                                                                                                                                                                                                                                                                                                         |
|-----------------|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `type`          | yes        | `"command"` , `"http"` , `"prompt"` , or `"agent"`                                                                                                                                                                                                                                                                                                                                                                  |
| `if`            | no         | Permission rule syntax to filter when this hook runs, such as `"Bash(git *)"` or `"Edit(*.ts)"` . The hook only spawns if the tool call matches the pattern. Only evaluated on tool events: `PreToolUse` , `PostToolUse` , `PostToolUseFailure` , `PermissionRequest` , and `PermissionDenied` . On other events, a hook with `if` set never runs. Uses the same syntax as [permission rules](/docs/en/permissions) |
| `timeout`       | no         | Seconds before canceling. Defaults: 600 for command, 30 for prompt, 60 for agent                                                                                                                                                                                                                                                                                                                                    |
| `statusMessage` | no         | Custom spinner message displayed while the hook runs                                                                                                                                                                                                                                                                                                                                                                |
| `once`          | no         | If `true` , runs only once per session then is removed. Skills only, not agents. See [Hooks in skills and agents](#hooks-in-skills-and-agents)                                                                                                                                                                                                                                                                      |

##### Command hook fields

In addition to the [common fields](#common-fields) , command hooks accept these fields:

| Field     | Required   | Description                                                                                                                                                                                                                            |
|-----------|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `command` | yes        | Shell command to execute                                                                                                                                                                                                               |
| `async`   | no         | If `true` , runs in the background without blocking. See [Run hooks in the background](#run-hooks-in-the-background)                                                                                                                   |
| `shell`   | no         | Shell to use for this hook. Accepts `"bash"` (default) or `"powershell"` . Setting `"powershell"` runs the command via PowerShell on Windows. Does not require `CLAUDE_CODE_USE_POWERSHELL_TOOL` since hooks spawn PowerShell directly |

##### HTTP hook fields

In addition to the [common fields](#common-fields) , HTTP hooks accept these fields:

| Field            | Required   | Description                                                                                                                                                                                      |
|------------------|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `url`            | yes        | URL to send the POST request to                                                                                                                                                                  |
| `headers`        | no         | Additional HTTP headers as key-value pairs. Values support environment variable interpolation using `$VAR_NAME` or `${VAR_NAME}` syntax. Only variables listed in `allowedEnvVars` are resolved  |
| `allowedEnvVars` | no         | List of environment variable names that may be interpolated into header values. References to unlisted variables are replaced with empty strings. Required for any env var interpolation to work |

Claude Code sends the hook's [JSON input](#hook-input-and-output) as the POST request body with `Content-Type: application/json` . The response body uses the same [JSON output format](#json-output) as command hooks. Error handling differs from command hooks: non-2xx responses, connection failures, and timeouts all produce non-blocking errors that allow execution to continue. To block a tool call or deny a permission, return a 2xx response with a JSON body containing `decision: "block"` or a `hookSpecificOutput` with `permissionDecision: "deny"` . This example sends `PreToolUse` events to a local validation service, authenticating with a token from the `MY_TOKEN` environment variable:

```
{
"hooks" : {
"PreToolUse" : [
{
"matcher" : "Bash" ,
"hooks" : [
{
"type" : "http" ,
"url" : "http://localhost:8080/hooks/pre-tool-use" ,
"timeout" : 30 ,
"headers" : {
"Authorization" : "Bearer $MY_TOKEN"
},
"allowedEnvVars" : [ "MY_TOKEN" ]
}
]
}
]
}
}
```

##### Prompt and agent hook fields

In addition to the [common fields](#common-fields) , prompt and agent hooks accept these fields:

| Field    | Required   | Description                                                                                 |
|----------|------------|---------------------------------------------------------------------------------------------|
| `prompt` | yes        | Prompt text to send to the model. Use `$ARGUMENTS` as a placeholder for the hook input JSON |
| `model`  | no         | Model to use for evaluation. Defaults to a fast model                                       |

All matching hooks run in parallel, and identical handlers are deduplicated automatically. Command hooks are deduplicated by command string, and HTTP hooks are deduplicated by URL. Handlers run in the current directory with Claude Code's environment. The `$CLAUDE_CODE_REMOTE` environment variable is set to `"true"` in remote web environments and not set in the local CLI.

#### Reference scripts by path

Use environment variables to reference hook scripts relative to the project or plugin root, regardless of the working directory when the hook runs:

- `$CLAUDE_PROJECT_DIR` : the project root. Wrap in quotes to handle paths with spaces.
- `${CLAUDE_PLUGIN_ROOT}` : the plugin's installation directory, for scripts bundled with a [plugin](/docs/en/plugins) . Changes on each plugin update.
- `${CLAUDE_PLUGIN_DATA}` : the plugin's [persistent data directory](/docs/en/plugins-reference#persistent-data-directory) , for dependencies and state that should survive plugin updates.

- Project scripts
- Plugin scripts

This example uses `$CLAUDE_PROJECT_DIR` to run a style checker from the project's `.claude/hooks/` directory after any `Write` or `Edit` tool call:

```
{
"hooks" : {
"PostToolUse" : [
{
"matcher" : "Write|Edit" ,
"hooks" : [
{
"type" : "command" ,
"command" : " \" $CLAUDE_PROJECT_DIR \" /.claude/hooks/check-style.sh"
}
]
}
]
}
}
```

Define plugin hooks in `hooks/hooks.json` with an optional top-level `description` field. When a plugin is enabled, its hooks merge with your user and project hooks. This example runs a formatting script bundled with the plugin:

```
{
"description" : "Automatic code formatting" ,
"hooks" : {
"PostToolUse" : [
{
"matcher" : "Write|Edit" ,
"hooks" : [
{
"type" : "command" ,
"command" : "${CLAUDE_PLUGIN_ROOT}/scripts/format.sh" ,
"timeout" : 30
}
]
}
]
}
}
```

See the [plugin components reference](/docs/en/plugins-reference#hooks) for details on creating plugin hooks.

#### Hooks in skills and agents

In addition to settings files and plugins, hooks can be defined directly in [skills](/docs/en/skills) and [subagents](/docs/en/sub-agents) using frontmatter. These hooks are scoped to the component's lifecycle and only run when that component is active. All hook events are supported. For subagents, `Stop` hooks are automatically converted to `SubagentStop` since that is the event that fires when a subagent completes. Hooks use the same configuration format as settings-based hooks but are scoped to the component's lifetime and cleaned up when it finishes. This skill defines a `PreToolUse` hook that runs a security validation script before each `Bash` command:

```
---
name : secure-operations
description : Perform operations with security checks
hooks :
PreToolUse :
- matcher : "Bash"
hooks :
- type : command
command : "./scripts/security-check.sh"
---
```

Agents use the same format in their YAML frontmatter.

#### The /hooks menu

Type `/hooks` in Claude Code to open a read-only browser for your configured hooks. The menu shows every hook event with a count of configured hooks, lets you drill into matchers, and shows the full details of each hook handler. Use it to verify configuration, check which settings file a hook came from, or inspect a hook's command, prompt, or URL. The menu displays all four hook types: `command` , `prompt` , `agent` , and `http` . Each hook is labeled with a `[type]` prefix and a source indicating where it was defined:

- `User` : from `~/.claude/settings.json`
- `Project` : from `.claude/settings.json`
- `Local` : from `.claude/settings.local.json`
- `Plugin` : from a plugin's `hooks/hooks.json`
- `Session` : registered in memory for the current session
- `Built-in` : registered internally by Claude Code

Selecting a hook opens a detail view showing its event, matcher, type, source file, and the full command, prompt, or URL. The menu is read-only: to add, modify, or remove hooks, edit the settings JSON directly or ask Claude to make the change.

#### Disable or remove hooks

To remove a hook, delete its entry from the settings JSON file. To temporarily disable all hooks without removing them, set `"disableAllHooks": true` in your settings file. There is no way to disable an individual hook while keeping it in the configuration. The `disableAllHooks` setting respects the managed settings hierarchy. If an administrator has configured hooks through managed policy settings, `disableAllHooks` set in user, project, or local settings cannot disable those managed hooks. Only `disableAllHooks` set at the managed settings level can disable managed hooks. Direct edits to hooks in settings files are normally picked up automatically by the file watcher.

### Hook input and output

Command hooks receive JSON data via stdin and communicate results through exit codes, stdout, and stderr. HTTP hooks receive the same JSON as the POST request body and communicate results through the HTTP response body. This section covers fields and behavior common to all events. Each event's section under [Hook events](#hook-events) includes its specific input schema and decision control options.

#### Common input fields

Hook events receive these fields as JSON, in addition to event-specific fields documented in each [hook event](#hook-events) section. For command hooks, this JSON arrives via stdin. For HTTP hooks, it arrives as the POST request body.

| Field             | Description                                                                                                                                                                                                                                       |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `session_id`      | Current session identifier                                                                                                                                                                                                                        |
| `transcript_path` | Path to conversation JSON                                                                                                                                                                                                                         |
| `cwd`             | Current working directory when the hook is invoked                                                                                                                                                                                                |
| `permission_mode` | Current [permission mode](/docs/en/permissions#permission-modes) : `"default"` , `"plan"` , `"acceptEdits"` , `"auto"` , `"dontAsk"` , or `"bypassPermissions"` . Not all events receive this field: see each event's JSON example below to check |
| `hook_event_name` | Name of the event that fired                                                                                                                                                                                                                      |

When running with `--agent` or inside a subagent, two additional fields are included:

| Field        | Description                                                                                                                                                                                                                           |
|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `agent_id`   | Unique identifier for the subagent. Present only when the hook fires inside a subagent call. Use this to distinguish subagent hook calls from main-thread calls.                                                                      |
| `agent_type` | Agent name (for example, `"Explore"` or `"security-reviewer"` ). Present when the session uses `--agent` or the hook fires inside a subagent. For subagents, the subagent's type takes precedence over the session's `--agent` value. |

For example, a `PreToolUse` hook for a Bash command receives this on stdin:

```
{
"session_id" : "abc123" ,
"transcript_path" : "/home/user/.claude/projects/.../transcript.jsonl" ,
"cwd" : "/home/user/my-project" ,
"permission_mode" : "default" ,
"hook_event_name" : "PreToolUse" ,
"tool_name" : "Bash" ,
"tool_input" : {
"command" : "npm test"
}
}
```

The `tool_name` and `tool_input` fields are event-specific. Each [hook event](#hook-events) section documents the additional fields for that event.

#### Exit code output

The exit code from your hook command tells Claude Code whether the action should proceed, be blocked, or be ignored. **Exit 0** means success. Claude Code parses stdout for [JSON output fields](#json-output) . JSON output is only processed on exit 0. For most events, stdout is written to the debug log but not shown in the transcript. The exceptions are `UserPromptSubmit` and `SessionStart` , where stdout is added as context that Claude can see and act on. **Exit 2** means a blocking error. Claude Code ignores stdout and any JSON in it. Instead, stderr text is fed back to Claude as an error message. The effect depends on the event: `PreToolUse` blocks the tool call, `UserPromptSubmit` rejects the prompt, and so on. See [exit code 2 behavior](#exit-code-2-behavior-per-event) for the full list. **Any other exit code** is a non-blocking error for most hook events. The transcript shows a `<hook name> hook error` notice followed by the first line of stderr, so you can identify the cause without `--debug` . Execution continues and the full stderr is written to the debug log. For example, a hook command script that blocks dangerous Bash commands:

```
#!/bin/bash
### Reads JSON input from stdin, checks the command
command = $( jq -r '.tool_input.command' < /dev/stdin )

if [[ " $command " == rm * ]]; then
echo "Blocked: rm commands are not allowed" >&2
exit 2 # Blocking error: tool call is prevented
fi

exit 0 # Success: tool call proceeds
```

For most hook events, only exit code 2 blocks the action. Claude Code treats exit code 1 as a non-blocking error and proceeds with the action, even though 1 is the conventional Unix failure code. If your hook is meant to enforce a policy, use `exit 2` . The exception is `WorktreeCreate` , where any non-zero exit code aborts worktree creation.

##### Exit code 2 behavior per event

Exit code 2 is the way a hook signals "stop, don't do this." The effect depends on the event, because some events represent actions that can be blocked (like a tool call that hasn't happened yet) and others represent things that already happened or can't be prevented.

| Hook event           | Can block?   | What happens on exit 2                                                                                                               |
|----------------------|--------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `PreToolUse`         | Yes          | Blocks the tool call                                                                                                                 |
| `PermissionRequest`  | Yes          | Denies the permission                                                                                                                |
| `UserPromptSubmit`   | Yes          | Blocks prompt processing and erases the prompt                                                                                       |
| `Stop`               | Yes          | Prevents Claude from stopping, continues the conversation                                                                            |
| `SubagentStop`       | Yes          | Prevents the subagent from stopping                                                                                                  |
| `TeammateIdle`       | Yes          | Prevents the teammate from going idle (teammate continues working)                                                                   |
| `TaskCreated`        | Yes          | Rolls back the task creation                                                                                                         |
| `TaskCompleted`      | Yes          | Prevents the task from being marked as completed                                                                                     |
| `ConfigChange`       | Yes          | Blocks the configuration change from taking effect (except `policy_settings` )                                                       |
| `StopFailure`        | No           | Output and exit code are ignored                                                                                                     |
| `PostToolUse`        | No           | Shows stderr to Claude (tool already ran)                                                                                            |
| `PostToolUseFailure` | No           | Shows stderr to Claude (tool already failed)                                                                                         |
| `PermissionDenied`   | No           | Exit code and stderr are ignored (denial already occurred). Use JSON `hookSpecificOutput.retry: true` to tell the model it may retry |
| `Notification`       | No           | Shows stderr to user only                                                                                                            |
| `SubagentStart`      | No           | Shows stderr to user only                                                                                                            |
| `SessionStart`       | No           | Shows stderr to user only                                                                                                            |
| `SessionEnd`         | No           | Shows stderr to user only                                                                                                            |
| `CwdChanged`         | No           | Shows stderr to user only                                                                                                            |
| `FileChanged`        | No           | Shows stderr to user only                                                                                                            |
| `PreCompact`         | No           | Shows stderr to user only                                                                                                            |
| `PostCompact`        | No           | Shows stderr to user only                                                                                                            |
| `Elicitation`        | Yes          | Denies the elicitation                                                                                                               |
| `ElicitationResult`  | Yes          | Blocks the response (action becomes decline)                                                                                         |
| `WorktreeCreate`     | Yes          | Any non-zero exit code causes worktree creation to fail                                                                              |
| `WorktreeRemove`     | No           | Failures are logged in debug mode only                                                                                               |
| `InstructionsLoaded` | No           | Exit code is ignored                                                                                                                 |

#### HTTP response handling

HTTP hooks use HTTP status codes and response bodies instead of exit codes and stdout:

- **2xx with an empty body** : success, equivalent to exit code 0 with no output
- **2xx with a plain text body** : success, the text is added as context
- **2xx with a JSON body** : success, parsed using the same [JSON output](#json-output) schema as command hooks
- **Non-2xx status** : non-blocking error, execution continues
- **Connection failure or timeout** : non-blocking error, execution continues

Unlike command hooks, HTTP hooks cannot signal a blocking error through status codes alone. To block a tool call or deny a permission, return a 2xx response with a JSON body containing the appropriate decision fields.

#### JSON output

Exit codes let you allow or block, but JSON output gives you finer-grained control. Instead of exiting with code 2 to block, exit 0 and print a JSON object to stdout. Claude Code reads specific fields from that JSON to control behavior, including [decision control](#decision-control) for blocking, allowing, or escalating to the user.

You must choose one approach per hook, not both: either use exit codes alone for signaling, or exit 0 and print JSON for structured control. Claude Code only processes JSON on exit 0. If you exit 2, any JSON is ignored.

Your hook's stdout must contain only the JSON object. If your shell profile prints text on startup, it can interfere with JSON parsing. See [JSON validation failed](/docs/en/hooks-guide#json-validation-failed) in the troubleshooting guide. Hook output injected into context ( `additionalContext` , `systemMessage` , or plain stdout) is capped at 10,000 characters. Output that exceeds this limit is saved to a file and replaced with a preview and file path, the same way large tool results are handled. The JSON object supports three kinds of fields:

- **Universal fields** like `continue` work across all events. These are listed in the table below.
- **Top-level** **`decision`** **and** **`reason`** are used by some events to block or provide feedback.
- **`hookSpecificOutput`** is a nested object for events that need richer control. It requires a `hookEventName` field set to the event name.

| Field            | Default   | Description                                                                                                                 |
|------------------|-----------|-----------------------------------------------------------------------------------------------------------------------------|
| `continue`       | `true`    | If `false` , Claude stops processing entirely after the hook runs. Takes precedence over any event-specific decision fields |
| `stopReason`     | none      | Message shown to the user when `continue` is `false` . Not shown to Claude                                                  |
| `suppressOutput` | `false`   | If `true` , omits stdout from the debug log                                                                                 |
| `systemMessage`  | none      | Warning message shown to the user                                                                                           |

To stop Claude entirely regardless of event type:

```
{ "continue" : false , "stopReason" : "Build failed, fix errors before continuing" }
```

##### Decision control

Not every event supports blocking or controlling behavior through JSON. The events that do each use a different set of fields to express that decision. Use this table as a quick reference before writing a hook:

| Events                                                                                                                      | Decision pattern               | Key fields                                                                                                                                                          |
|-----------------------------------------------------------------------------------------------------------------------------|--------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| UserPromptSubmit, PostToolUse, PostToolUseFailure, Stop, SubagentStop, ConfigChange                                         | Top-level `decision`           | `decision: "block"` , `reason`                                                                                                                                      |
| TeammateIdle, TaskCreated, TaskCompleted                                                                                    | Exit code or `continue: false` | Exit code 2 blocks the action with stderr feedback. JSON `{"continue": false, "stopReason": "..."}` also stops the teammate entirely, matching `Stop` hook behavior |
| PreToolUse                                                                                                                  | `hookSpecificOutput`           | `permissionDecision` (allow/deny/ask/defer), `permissionDecisionReason`                                                                                             |
| PermissionRequest                                                                                                           | `hookSpecificOutput`           | `decision.behavior` (allow/deny)                                                                                                                                    |
| PermissionDenied                                                                                                            | `hookSpecificOutput`           | `retry: true` tells the model it may retry the denied tool call                                                                                                     |
| WorktreeCreate                                                                                                              | path return                    | Command hook prints path on stdout; HTTP hook returns `hookSpecificOutput.worktreePath` . Hook failure or missing path fails creation                               |
| Elicitation                                                                                                                 | `hookSpecificOutput`           | `action` (accept/decline/cancel), `content` (form field values for accept)                                                                                          |
| ElicitationResult                                                                                                           | `hookSpecificOutput`           | `action` (accept/decline/cancel), `content` (form field values override)                                                                                            |
| WorktreeRemove, Notification, SessionEnd, PreCompact, PostCompact, InstructionsLoaded, StopFailure, CwdChanged, FileChanged | None                           | No decision control. Used for side effects like logging or cleanup                                                                                                  |

Here are examples of each pattern in action:

- Top-level decision
- PreToolUse
- PermissionRequest

Used by `UserPromptSubmit` , `PostToolUse` , `PostToolUseFailure` , `Stop` , `SubagentStop` , and `ConfigChange` . The only value is `"block"` . To allow the action to proceed, omit `decision` from your JSON, or exit 0 without any JSON at all:

```
{
"decision" : "block" ,
"reason" : "Test suite must pass before proceeding"
}
```

Uses `hookSpecificOutput` for richer control: allow, deny, or escalate to the user. You can also modify tool input before it runs or inject additional context for Claude. See [PreToolUse decision control](#pretooluse-decision-control) for the full set of options.

```
{
"hookSpecificOutput" : {
"hookEventName" : "PreToolUse" ,
"permissionDecision" : "deny" ,
"permissionDecisionReason" : "Database writes are not allowed"
}
}
```

Uses `hookSpecificOutput` to allow or deny a permission request on behalf of the user. When allowing, you can also modify the tool's input or apply permission rules so the user isn't prompted again. See [PermissionRequest decision control](#permissionrequest-decision-control) for the full set of options.

```
{
"hookSpecificOutput" : {
"hookEventName" : "PermissionRequest" ,
"decision" : {
"behavior" : "allow" ,
"updatedInput" : {
"command" : "npm run lint"
}
}
}
}
```

For extended examples including Bash command validation, prompt filtering, and auto-approval scripts, see [What you can automate](/docs/en/hooks-guide#what-you-can-automate) in the guide and the [Bash command validator reference implementation](https://github.com/anthropics/claude-code/blob/main/examples/hooks/bash_command_validator_example.py) .

### Hook events

Each event corresponds to a point in Claude Code's lifecycle where hooks can run. The sections below are ordered to match the lifecycle: from session setup through the agentic loop to session end. Each section describes when the event fires, what matchers it supports, the JSON input it receives, and how to control behavior through output.

#### SessionStart

Runs when Claude Code starts a new session or resumes an existing session. Useful for loading development context like existing issues or recent changes to your codebase, or setting up environment variables. For static context that does not require a script, use [CLAUDE.md](/docs/en/memory) instead. SessionStart runs on every session, so keep these hooks fast. Only `type: "command"` hooks are supported. The matcher value corresponds to how the session was initiated:

| Matcher   | When it fires                            |
|-----------|------------------------------------------|
| `startup` | New session                              |
| `resume`  | `--resume` , `--continue` , or `/resume` |
| `clear`   | `/clear`                                 |
| `compact` | Auto or manual compaction                |

##### SessionStart input

In addition to the [common input fields](#common-input-fields) , SessionStart hooks receive `source` , `model` , and optionally `agent_type` . The `source` field indicates how the session started: `"startup"` for new sessions, `"resume"` for resumed sessions, `"clear"` after `/clear` , or `"compact"` after compaction. The `model` field contains the model identifier. If you start Claude Code with `claude --agent <name>` , an `agent_type` field contains the agent name.

```
{
"session_id" : "abc123" ,
"transcript_path" : "/Users/.../.claude/projects/.../00893aaf-19fa-41d2-8238-13269b9b3ca0.jsonl" ,
"cwd" : "/Users/..." ,
"hook_event_name" : "SessionStart" ,
"source" : "startup" ,
"model" : "claude-sonnet-4-6"
}
```

##### SessionStart decision control

Any text your hook script prints to stdout is added as context for Claude. In addition to the [JSON output fields](#json-output) available to all hooks, you can return these event-specific fields:

| Field               | Description                                                               |
|---------------------|---------------------------------------------------------------------------|
| `additionalContext` | String added to Claude's context. Multiple hooks' values are concatenated |

```
{
"hookSpecificOutput" : {
"hookEventName" : "SessionStart" ,
"additionalContext" : "My additional context here"
}
}
```

##### Persist environment variables

SessionStart hooks have access to the `CLAUDE_ENV_FILE` environment variable, which provides a file path where you can persist environment variables for subsequent Bash commands. To set individual environment variables, write `export` statements to `CLAUDE_ENV_FILE` . Use append ( `>>` ) to preserve variables set by other hooks:

```
#!/bin/bash

if [ -n " $CLAUDE_ENV_FILE " ]; then
echo 'export NODE_ENV=production' >> " $CLAUDE_ENV_FILE "
echo 'export DEBUG_LOG=true' >> " $CLAUDE_ENV_FILE "
echo 'export PATH="$PATH:./node_modules/.bin"' >> " $CLAUDE_ENV_FILE "
fi

exit 0
```

To capture all environment changes from setup commands, compare the exported variables before and after:

```
#!/bin/bash

ENV_BEFORE = $(e xport -p | sort )

### Run your setup commands that modify the environment
source ~/.nvm/nvm.sh
nvm use 20

if [ -n " $CLAUDE_ENV_FILE " ]; then
ENV_AFTER = $(e xport -p | sort )
comm -13 <( echo " $ENV_BEFORE ") <( echo " $ENV_AFTER ") >> " $CLAUDE_ENV_FILE "
fi

exit 0
```

Any variables written to this file will be available in all subsequent Bash commands that Claude Code executes during the session.

`CLAUDE_ENV_FILE` is available for SessionStart, [CwdChanged](#cwdchanged) , and [FileChanged](#filechanged) hooks. Other hook types do not have access to this variable.

#### InstructionsLoaded

Fires when a `CLAUDE.md` or `.claude/rules/*.md` file is loaded into context. This event fires at session start for eagerly-loaded files and again later when files are lazily loaded, for example when Claude accesses a subdirectory that contains a nested `CLAUDE.md` or when conditional rules with `paths:` frontmatter match. The hook does not support blocking or decision control. It runs asynchronously for observability purposes. The matcher runs against `load_reason` . For example, use `"matcher": "session_start"` to fire only for files loaded at session start, or `"matcher": "path_glob_match|nested_traversal"` to fire only for lazy loads.

##### InstructionsLoaded input

In addition to the [common input fields](#common-input-fields) , InstructionsLoaded hooks receive these fields:

| Field               | Description                                                                                                                                                                                                        |
|---------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `file_path`         | Absolute path to the instruction file that was loaded                                                                                                                                                              |
| `memory_type`       | Scope of the file: `"User"` , `"Project"` , `"Local"` , or `"Managed"`                                                                                                                                             |
| `load_reason`       | Why the file was loaded: `"session_start"` , `"nested_traversal"` , `"path_glob_match"` , `"include"` , or `"compact"` . The `"compact"` value fires when instruction files are re-loaded after a compaction event |
| `globs`             | Path glob patterns from the file's `paths:` frontmatter, if any. Present only for `path_glob_match` loads                                                                                                          |
| `trigger_file_path` | Path to the file whose access triggered this load, for lazy loads                                                                                                                                                  |
| `parent_file_path`  | Path to the parent instruction file that included this one, for `include` loads                                                                                                                                    |

```
{
"session_id" : "abc123" ,
"transcript_path" : "/Users/.../.claude/projects/.../transcript.jsonl" ,
"cwd" : "/Users/my-project" ,
"hook_event_name" : "InstructionsLoaded" ,
"file_path" : "/Users/my-project/CLAUDE.md" ,
"memory_type" : "Project" ,
"load_reason" : "session_start"
}
```

##### InstructionsLoaded decision control

InstructionsLoaded hooks have no decision control. They cannot block or modify instruction loading. Use this event for audit logging, compliance tracking, or observability.

#### UserPromptSubmit

Runs when the user submits a prompt, before Claude processes it. This allows you

to add additional context based on the prompt/conversation, validate prompts, or

block certain types of prompts.

##### UserPromptSubmit input

In addition to the [common input fields](#common-input-fields) , UserPromptSubmit hooks receive the `prompt` field containing the text the user submitted.

```
{
"session_id" : "abc123" ,
"transcript_path" : "/Users/.../.claude/projects/.../00893aaf-19fa-41d2-8238-13269b9b3ca0.jsonl" ,
"cwd" : "/Users/..." ,
"permission_mode" : "default" ,
"hook_event_name" : "UserPromptSubmit" ,
"prompt" : "Write a function to calculate the factorial of a number"
}
```

##### UserPromptSubmit decision control

`UserPromptSubmit` hooks can control whether a user prompt is processed and add context. All [JSON output fields](#json-output) are available. There are two ways to add context to the conversation on exit code 0:

- **Plain text stdout** : any non-JSON text written to stdout is added as context
- **JSON with** **`additionalContext`** : use the JSON format below for more control. The `additionalContext` field is added as context

Plain stdout is shown as hook output in the transcript. The `additionalContext` field is added more discretely. To block a prompt, return a JSON object with `decision` set to `"block"` :

| Field               | Description                                                                                                        |
|---------------------|--------------------------------------------------------------------------------------------------------------------|
| `decision`          | `"block"` prevents the prompt from being processed and erases it from context. Omit to allow the prompt to proceed |
| `reason`            | Shown to the user when `decision` is `"block"` . Not added to context                                              |
| `additionalContext` | String added to Claude's context                                                                                   |
| `sessionTitle`      | Sets the session title, same effect as `/rename` . Use to name sessions automatically based on the prompt content  |

```
{
"decision" : "block" ,
"reason" : "Explanation for decision" ,
"hookSpecificOutput" : {
"hookEventName" : "UserPromptSubmit" ,
"additionalContext" : "My additional context here" ,
"sessionTitle" : "My session title"
}
}
```

The JSON format isn't required for simple use cases. To add context, you can print plain text to stdout with exit code 0. Use JSON when you need to

block prompts or want more structured control.

#### PreToolUse

Runs after Claude creates tool parameters and before processing the tool call. Matches on tool name: `Bash` , `Edit` , `Write` , `Read` , `Glob` , `Grep` , `Agent` , `WebFetch` , `WebSearch` , `AskUserQuestion` , `ExitPlanMode` , and any [MCP tool names](#match-mcp-tools) . Use [PreToolUse decision control](#pretooluse-decision-control) to allow, deny, ask, or defer the tool call.

##### PreToolUse input

In addition to the [common input fields](#common-input-fields) , PreToolUse hooks receive `tool_name` , `tool_input` , and `tool_use_id` . The `tool_input` fields depend on the tool:

##### Bash

Executes shell commands.

| Field               | Type    | Example            | Description                                   |
|---------------------|---------|--------------------|-----------------------------------------------|
| `command`           | string  | `"npm test"`       | The shell command to execute                  |
| `description`       | string  | `"Run test suite"` | Optional description of what the command does |
| `timeout`           | number  | `120000`           | Optional timeout in milliseconds              |
| `run_in_background` | boolean | `false`            | Whether to run the command in background      |

##### Write

Creates or overwrites a file.

| Field       | Type   | Example               | Description                        |
|-------------|--------|-----------------------|------------------------------------|
| `file_path` | string | `"/path/to/file.txt"` | Absolute path to the file to write |
| `content`   | string | `"file content"`      | Content to write to the file       |

##### Edit

Replaces a string in an existing file.

| Field         | Type    | Example               | Description                        |
|---------------|---------|-----------------------|------------------------------------|
| `file_path`   | string  | `"/path/to/file.txt"` | Absolute path to the file to edit  |
| `old_string`  | string  | `"original text"`     | Text to find and replace           |
| `new_string`  | string  | `"replacement text"`  | Replacement text                   |
| `replace_all` | boolean | `false`               | Whether to replace all occurrences |

##### Read

Reads file contents.

| Field       | Type   | Example               | Description                                |
|-------------|--------|-----------------------|--------------------------------------------|
| `file_path` | string | `"/path/to/file.txt"` | Absolute path to the file to read          |
| `offset`    | number | `10`                  | Optional line number to start reading from |
| `limit`     | number | `50`                  | Optional number of lines to read           |

##### Glob

Finds files matching a glob pattern.

| Field     | Type   | Example          | Description                                                            |
|-----------|--------|------------------|------------------------------------------------------------------------|
| `pattern` | string | `"**/*.ts"`      | Glob pattern to match files against                                    |
| `path`    | string | `"/path/to/dir"` | Optional directory to search in. Defaults to current working directory |

##### Grep

Searches file contents with regular expressions.

| Field         | Type    | Example          | Description                                                                              |
|---------------|---------|------------------|------------------------------------------------------------------------------------------|
| `pattern`     | string  | `"TODO.*fix"`    | Regular expression pattern to search for                                                 |
| `path`        | string  | `"/path/to/dir"` | Optional file or directory to search in                                                  |
| `glob`        | string  | `"*.ts"`         | Optional glob pattern to filter files                                                    |
| `output_mode` | string  | `"content"`      | `"content"` , `"files_with_matches"` , or `"count"` . Defaults to `"files_with_matches"` |
| `-i`          | boolean | `true`           | Case insensitive search                                                                  |
| `multiline`   | boolean | `false`          | Enable multiline matching                                                                |

##### WebFetch

Fetches and processes web content.

| Field    | Type   | Example                       | Description                          |
|----------|--------|-------------------------------|--------------------------------------|
| `url`    | string | `"https://example.com/api"`   | URL to fetch content from            |
| `prompt` | string | `"Extract the API endpoints"` | Prompt to run on the fetched content |

##### WebSearch

Searches the web.

| Field             | Type   | Example                        | Description                                       |
|-------------------|--------|--------------------------------|---------------------------------------------------|
| `query`           | string | `"react hooks best practices"` | Search query                                      |
| `allowed_domains` | array  | `["docs.example.com"]`         | Optional: only include results from these domains |
| `blocked_domains` | array  | `["spam.example.com"]`         | Optional: exclude results from these domains      |

##### Agent

Spawns a [subagent](/docs/en/sub-agents) .

| Field           | Type   | Example                    | Description                                  |
|-----------------|--------|----------------------------|----------------------------------------------|
| `prompt`        | string | `"Find all API endpoints"` | The task for the agent to perform            |
| `description`   | string | `"Find API endpoints"`     | Short description of the task                |
| `subagent_type` | string | `"Explore"`                | Type of specialized agent to use             |
| `model`         | string | `"sonnet"`                 | Optional model alias to override the default |

##### AskUserQuestion

Asks the user one to four multiple-choice questions.

| Field       | Type   | Example                                                                                                            | Description                                                                                                                                                                                      |
|-------------|--------|--------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `questions` | array  | `[{"question": "Which framework?", "header": "Framework", "options": [{"label": "React"}], "multiSelect": false}]` | Questions to present, each with a `question` string, short `header` , `options` array, and optional `multiSelect` flag                                                                           |
| `answers`   | object | `{"Which framework?": "React"}`                                                                                    | Optional. Maps question text to the selected option label. Multi-select answers join labels with commas. Claude does not set this field; supply it via `updatedInput` to answer programmatically |

##### PreToolUse decision control

`PreToolUse` hooks can control whether a tool call proceeds. Unlike other hooks that use a top-level `decision` field, PreToolUse returns its decision inside a `hookSpecificOutput` object. This gives it richer control: four outcomes (allow, deny, ask, or defer) plus the ability to modify tool input before execution.

| Field                      | Description                                                                                                                                                                                                                                                                       |
|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `permissionDecision`       | `"allow"` skips the permission prompt. `"deny"` prevents the tool call. `"ask"` prompts the user to confirm. `"defer"` exits gracefully so the tool can be resumed later. [Deny and ask rules](/docs/en/permissions#manage-permissions) still apply when a hook returns `"allow"` |
| `permissionDecisionReason` | For `"allow"` and `"ask"` , shown to the user but not Claude. For `"deny"` , shown to Claude. For `"defer"` , ignored                                                                                                                                                             |
| `updatedInput`             | Modifies the tool's input parameters before execution. Replaces the entire input object, so include unchanged fields alongside modified ones. Combine with `"allow"` to auto-approve, or `"ask"` to show the modified input to the user. For `"defer"` , ignored                  |
| `additionalContext`        | String added to Claude's context before the tool executes. For `"defer"` , ignored                                                                                                                                                                                                |

When multiple PreToolUse hooks return different decisions, precedence is `deny` > `defer` > `ask` > `allow` . When a hook returns `"ask"` , the permission prompt displayed to the user includes a label identifying where the hook came from: for example, `[User]` , `[Project]` , `[Plugin]` , or `[Local]` . This helps users understand which configuration source is requesting confirmation.

```
{
"hookSpecificOutput" : {
"hookEventName" : "PreToolUse" ,
"permissionDecision" : "allow" ,
"permissionDecisionReason" : "My reason here" ,
"updatedInput" : {
"field_to_modify" : "new value"
},
"additionalContext" : "Current environment: production. Proceed with caution."
}
}
```

`AskUserQuestion` and `ExitPlanMode` require user interaction and normally block in [non-interactive mode](/docs/en/headless) with the `-p` flag. Returning `permissionDecision: "allow"` together with `updatedInput` satisfies that requirement: the hook reads the tool's input from stdin, collects the answer through your own UI, and returns it in `updatedInput` so the tool runs without prompting. Returning `"allow"` alone is not sufficient for these tools. For `AskUserQuestion` , echo back the original `questions` array and add an [`answers`](#askuserquestion) object mapping each question's text to the chosen answer.

PreToolUse previously used top-level `decision` and `reason` fields, but these are deprecated for this event. Use `hookSpecificOutput.permissionDecision` and `hookSpecificOutput.permissionDecisionReason` instead. The deprecated values `"approve"` and `"block"` map to `"allow"` and `"deny"` respectively. Other events like PostToolUse and Stop continue to use top-level `decision` and `reason` as their current format.

##### Defer a tool call for later

`"defer"` is for integrations that run `claude -p` as a subprocess and read its JSON output, such as an Agent SDK app or a custom UI built on top of Claude Code. It lets that calling process pause Claude at a tool call, collect input through its own interface, and resume where it left off. Claude Code honors this value only in [non-interactive mode](/docs/en/headless) with the `-p` flag. In interactive sessions it logs a warning and ignores the hook result.

The `defer` value requires Claude Code v2.1.89 or later. Earlier versions do not recognize it and the tool proceeds through the normal permission flow.

The `AskUserQuestion` tool is the typical case: Claude wants to ask the user something, but there is no terminal to answer in. The round trip works like this:

1. Claude calls `AskUserQuestion` . The `PreToolUse` hook fires.
2. The hook returns `permissionDecision: "defer"` . The tool does not execute. The process exits with `stop_reason: "tool_deferred"` and the pending tool call preserved in the transcript.
3. The calling process reads `deferred_tool_use` from the SDK result, surfaces the question in its own UI, and waits for an answer.
4. The calling process runs `claude -p --resume <session-id>` . The same tool call fires `PreToolUse` again.
5. The hook returns `permissionDecision: "allow"` with the answer in `updatedInput` . The tool executes and Claude continues.

The `deferred_tool_use` field carries the tool's `id` , `name` , and `input` . The `input` is the parameters Claude generated for the tool call, captured before execution:

```
{
"type" : "result" ,
"subtype" : "success" ,
"stop_reason" : "tool_deferred" ,
"session_id" : "abc123" ,
"deferred_tool_use" : {
"id" : "toolu_01abc" ,
"name" : "AskUserQuestion" ,
"input" : { "questions" : [{ "question" : "Which framework?" , "header" : "Framework" , "options" : [{ "label" : "React" }, { "label" : "Vue" }], "multiSelect" : false }] }
}
}
```

There is no timeout or retry limit. The session remains on disk until you resume it. If the answer is not ready when you resume, the hook can return `"defer"` again and the process exits the same way. The calling process controls when to break the loop by eventually returning `"allow"` or `"deny"` from the hook. `"defer"` only works when Claude makes a single tool call in the turn. If Claude makes several tool calls at once, `"defer"` is ignored with a warning and the tool proceeds through the normal permission flow. The constraint exists because resume can only re-run one tool: there is no way to defer one call from a batch without leaving the others unresolved. If the deferred tool is no longer available when you resume, the process exits with `stop_reason: "tool_deferred_unavailable"` and `is_error: true` before the hook fires. This happens when an MCP server that provided the tool is not connected for the resumed session. The `deferred_tool_use` payload is still included so you can identify which tool went missing.

`--resume` does not restore the permission mode from the prior session. Pass the same `--permission-mode` flag on resume that was active when the tool was deferred. Claude Code logs a warning if the modes differ.

#### PermissionRequest

Runs when the user is shown a permission dialog.

Use

[PermissionRequest decision control](#permissionrequest-decision-control) to allow or deny on behalf of the user. Matches on tool name, same values as PreToolUse.

##### PermissionRequest input

PermissionRequest hooks receive `tool_name` and `tool_input` fields like PreToolUse hooks, but without `tool_use_id` . An optional `permission_suggestions` array contains the "always allow" options the user would normally see in the permission dialog. The difference is when the hook fires: PermissionRequest hooks run when a permission dialog is about to be shown to the user, while PreToolUse hooks run before tool execution regardless of permission status.

```
{
"session_id" : "abc123" ,
"transcript_path" : "/Users/.../.claude/projects/.../00893aaf-19fa-41d2-8238-13269b9b3ca0.jsonl" ,
"cwd" : "/Users/..." ,
"permission_mode" : "default" ,
"hook_event_name" : "PermissionRequest" ,
"tool_name" : "Bash" ,
"tool_input" : {
"command" : "rm -rf node_modules" ,
"description" : "Remove node_modules directory"
},
"permission_suggestions" : [
{
"type" : "addRules" ,
"rules" : [{ "toolName" : "Bash" , "ruleContent" : "rm -rf node_modules" }],
"behavior" : "allow" ,
"destination" : "localSettings"
}
]
}
```

##### PermissionRequest decision control

`PermissionRequest` hooks can allow or deny permission requests. In addition to the [JSON output fields](#json-output) available to all hooks, your hook script can return a `decision` object with these event-specific fields:

| Field                | Description                                                                                                                                                         |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `behavior`           | `"allow"` grants the permission, `"deny"` denies it                                                                                                                 |
| `updatedInput`       | For `"allow"` only: modifies the tool's input parameters before execution. Replaces the entire input object, so include unchanged fields alongside modified ones    |
| `updatedPermissions` | For `"allow"` only: array of [permission update entries](#permission-update-entries) to apply, such as adding an allow rule or changing the session permission mode |
| `message`            | For `"deny"` only: tells Claude why the permission was denied                                                                                                       |
| `interrupt`          | For `"deny"` only: if `true` , stops Claude                                                                                                                         |

```
{
"hookSpecificOutput" : {
"hookEventName" : "PermissionRequest" ,
"decision" : {
"behavior" : "allow" ,
"updatedInput" : {
"command" : "npm run lint"
}
}
}
}
```

##### Permission update entries

The `updatedPermissions` output field and the [`permission_suggestions`](#permissionrequest-input) [input field](#permissionrequest-input) both use the same array of entry objects. Each entry has a `type` that determines its other fields, and a `destination` that controls where the change is written.

| `type`              | Fields                               | Effect                                                                                                                                                                        |
|---------------------|--------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `addRules`          | `rules` , `behavior` , `destination` | Adds permission rules. `rules` is an array of `{toolName, ruleContent?}` objects. Omit `ruleContent` to match the whole tool. `behavior` is `"allow"` , `"deny"` , or `"ask"` |
| `replaceRules`      | `rules` , `behavior` , `destination` | Replaces all rules of the given `behavior` at the `destination` with the provided `rules`                                                                                     |
| `removeRules`       | `rules` , `behavior` , `destination` | Removes matching rules of the given `behavior`                                                                                                                                |
| `setMode`           | `mode` , `destination`               | Changes the permission mode. Valid modes are `default` , `acceptEdits` , `dontAsk` , `bypassPermissions` , and `plan`                                                         |
| `addDirectories`    | `directories` , `destination`        | Adds working directories. `directories` is an array of path strings                                                                                                           |
| `removeDirectories` | `directories` , `destination`        | Removes working directories                                                                                                                                                   |

The `destination` field on every entry determines whether the change stays in memory or persists to a settings file.

| `destination`     | Writes to                                       |
|-------------------|-------------------------------------------------|
| `session`         | in-memory only, discarded when the session ends |
| `localSettings`   | `.claude/settings.local.json`                   |
| `projectSettings` | `.claude/settings.json`                         |
| `userSettings`    | `~/.claude/settings.json`                       |

A hook can echo one of the `permission_suggestions` it received as its own `updatedPermissions` output, which is equivalent to the user selecting that "always allow" option in the dialog.

#### PostToolUse

Runs immediately after a tool completes successfully. Matches on tool name, same values as PreToolUse.

##### PostToolUse input

`PostToolUse` hooks fire after a tool has already executed successfully. The input includes both `tool_input` , the arguments sent to the tool, and `tool_response` , the result it returned. The exact schema for both depends on the tool.

```
{
"session_id" : "abc123" ,
"transcript_path" : "/Users/.../.claude/projects/.../00893aaf-19fa-41d2-8238-13269b9b3ca0.jsonl" ,
"cwd" : "/Users/..." ,
"permission_mode" : "default" ,
"hook_event_name" : "PostToolUse" ,
"tool_name" : "Write" ,
"tool_input" : {
"file_path" : "/path/to/file.txt" ,
"content" : "file content"
},
"tool_response" : {
"filePath" : "/path/to/file.txt" ,
"success" : true
},
"tool_use_id" : "toolu_01ABC123..."
}
```

##### PostToolUse decision control

`PostToolUse` hooks can provide feedback to Claude after tool execution. In addition to the [JSON output fields](#json-output) available to all hooks, your hook script can return these event-specific fields:

| Field                  | Description                                                                                |
|------------------------|--------------------------------------------------------------------------------------------|
| `decision`             | `"block"` prompts Claude with the `reason` . Omit to allow the action to proceed           |
| `reason`               | Explanation shown to Claude when `decision` is `"block"`                                   |
| `additionalContext`    | Additional context for Claude to consider                                                  |
| `updatedMCPToolOutput` | For [MCP tools](#match-mcp-tools) only: replaces the tool's output with the provided value |

```
{
"decision" : "block" ,
"reason" : "Explanation for decision" ,
"hookSpecificOutput" : {
"hookEventName" : "PostToolUse" ,
"additionalContext" : "Additional information for Claude"
}
}
```

#### PostToolUseFailure

Runs when a tool execution fails. This event fires for tool calls that throw errors or return failure results. Use this to log failures, send alerts, or provide corrective feedback to Claude. Matches on tool name, same values as PreToolUse.

##### PostToolUseFailure input

PostToolUseFailure hooks receive the same `tool_name` and `tool_input` fields as PostToolUse, along with error information as top-level fields:

```
{
"session_id" : "abc123" ,
"transcript_path" : "/Users/.../.claude/projects/.../00893aaf-19fa-41d2-8238-13269b9b3ca0.jsonl" ,
"cwd" : "/Users/..." ,
"permission_mode" : "default" ,
"hook_event_name" : "PostToolUseFailure" ,
"tool_name" : "Bash" ,
"tool_input" : {
"command" : "npm test" ,
"description" : "Run test suite"
},
"tool_use_id" : "toolu_01ABC123..." ,
"error" : "Command exited with non-zero status code 1" ,
"is_interrupt" : false
}
```

| Field          | Description                                                                     |
|----------------|---------------------------------------------------------------------------------|
| `error`        | String describing what went wrong                                               |
| `is_interrupt` | Optional boolean indicating whether the failure was caused by user interruption |

##### PostToolUseFailure decision control

`PostToolUseFailure` hooks can provide context to Claude after a tool failure. In addition to the [JSON output fields](#json-output) available to all hooks, your hook script can return these event-specific fields:

| Field               | Description                                                   |
|---------------------|---------------------------------------------------------------|
| `additionalContext` | Additional context for Claude to consider alongside the error |

```
{
"hookSpecificOutput" : {
"hookEventName" : "PostToolUseFailure" ,
"additionalContext" : "Additional information about the failure for Claude"
}
}
```

#### PermissionDenied

Runs when the [auto mode](/docs/en/permission-modes#eliminate-prompts-with-auto-mode) classifier denies a tool call. This hook only fires in auto mode: it does not run when you manually deny a permission dialog, when a `PreToolUse` hook blocks a call, or when a `deny` rule matches. Use it to log classifier denials, adjust configuration, or tell the model it may retry the tool call. Matches on tool name, same values as PreToolUse.

##### PermissionDenied input

In addition to the [common input fields](#common-input-fields) , PermissionDenied hooks receive `tool_name` , `tool_input` , `tool_use_id` , and `reason` .

```
{
"session_id" : "abc123" ,
"transcript_path" : "/Users/.../.claude/projects/.../00893aaf-19fa-41d2-8238-13269b9b3ca0.jsonl" ,
"cwd" : "/Users/..." ,
"permission_mode" : "auto" ,
"hook_event_name" : "PermissionDenied" ,
"tool_name" : "Bash" ,
"tool_input" : {
"command" : "rm -rf /tmp/build" ,
"description" : "Clean build directory"
},
"tool_use_id" : "toolu_01ABC123..." ,
"reason" : "Auto mode denied: command targets a path outside the project"
}
```

| Field    | Description                                                   |
|----------|---------------------------------------------------------------|
| `reason` | The classifier's explanation for why the tool call was denied |

##### PermissionDenied decision control

PermissionDenied hooks can tell the model it may retry the denied tool call. Return a JSON object with `hookSpecificOutput.retry` set to `true` :

```
{
"hookSpecificOutput" : {
"hookEventName" : "PermissionDenied" ,
"retry" : true
}
}
```

When `retry` is `true` , Claude Code adds a message to the conversation telling the model it may retry the tool call. The denial itself is not reversed. If your hook does not return JSON, or returns `retry: false` , the denial stands and the model receives the original rejection message.

#### Notification

Runs when Claude Code sends notifications. Matches on notification type: `permission_prompt` , `idle_prompt` , `auth_success` , `elicitation_dialog` . Omit the matcher to run hooks for all notification types. Use separate matchers to run different handlers depending on the notification type. This configuration triggers a permission-specific alert script when Claude needs permission approval and a different notification when Claude has been idle:

```
{
"hooks" : {
"Notification" : [
{
"matcher" : "permission_prompt" ,
"hooks" : [
{
"type" : "command" ,
"command" : "/path/to/permission-alert.sh"
}
]
},
{
"matcher" : "idle_prompt" ,
"hooks" : [
{
"type" : "command" ,
"command" : "/path/to/idle-notification.sh"
}
]
}
]
}
}
```

##### Notification input

In addition to the [common input fields](#common-input-fields) , Notification hooks receive `message` with the notification text, an optional `title` , and `notification_type` indicating which type fired.

```
{
"session_id" : "abc123" ,
"transcript_path" : "/Users/.../.claude/projects/.../00893aaf-19fa-41d2-8238-13269b9b3ca0.jsonl" ,
"cwd" : "/Users/..." ,
"hook_event_name" : "Notification" ,
"message" : "Claude needs your permission to use Bash" ,
"title" : "Permission needed" ,
"notification_type" : "permission_prompt"
}
```

Notification hooks cannot block or modify notifications. In addition to the [JSON output fields](#json-output) available to all hooks, you can return `additionalContext` to add context to the conversation:

| Field               | Description                      |
|---------------------|----------------------------------|
| `additionalContext` | String added to Claude's context |

#### SubagentStart

Runs when a Claude Code subagent is spawned via the Agent tool. Supports matchers to filter by agent type name (built-in agents like `Bash` , `Explore` , `Plan` , or custom agent names from `.claude/agents/` ).

##### SubagentStart input

In addition to the [common input fields](#common-input-fields) , SubagentStart hooks receive `agent_id` with the unique identifier for the subagent and `agent_type` with the agent name (built-in agents like `"Bash"` , `"Explore"` , `"Plan"` , or custom agent names).

```
{
"session_id" : "abc123" ,
"transcript_path" : "/Users/.../.claude/projects/.../00893aaf-19fa-41d2-8238-13269b9b3ca0.jsonl" ,
"cwd" : "/Users/..." ,
"hook_event_name" : "SubagentStart" ,
"agent_id" : "agent-abc123" ,
"agent_type" : "Explore"
}
```

SubagentStart hooks cannot block subagent creation, but they can inject context into the subagent. In addition to the [JSON output fields](#json-output) available to all hooks, you can return:

| Field               | Description                            |
|---------------------|----------------------------------------|
| `additionalContext` | String added to the subagent's context |

```
{
"hookSpecificOutput" : {
"hookEventName" : "SubagentStart" ,
"additionalContext" : "Follow security guidelines for this task"
}
}
```

#### SubagentStop

Runs when a Claude Code subagent has finished responding. Matches on agent type, same values as SubagentStart.

##### SubagentStop input

In addition to the [common input fields](#common-input-fields) , SubagentStop hooks receive `stop_hook_active` , `agent_id` , `agent_type` , `agent_transcript_path` , and `last_assistant_message` . The `agent_type` field is the value used for matcher filtering. The `transcript_path` is the main session's transcript, while `agent_transcript_path` is the subagent's own transcript stored in a nested `subagents/` folder. The `last_assistant_message` field contains the text content of the subagent's final response, so hooks can access it without parsing the transcript file.

```
{
"session_id" : "abc123" ,
"transcript_path" : "~/.claude/projects/.../abc123.jsonl" ,
"cwd" : "/Users/..." ,
"permission_mode" : "default" ,
"hook_event_name" : "SubagentStop" ,
"stop_hook_active" : false ,
"agent_id" : "def456" ,
"agent_type" : "Explore" ,
"agent_transcript_path" : "~/.claude/projects/.../abc123/subagents/agent-def456.jsonl" ,
"last_assistant_message" : "Analysis complete. Found 3 potential issues..."
}
```

SubagentStop hooks use the same decision control format as [Stop hooks](#stop-decision-control) .

#### TaskCreated

Runs when a task is being created via the `TaskCreate` tool. Use this to enforce naming conventions, require task descriptions, or prevent certain tasks from being created. When a `TaskCreated` hook exits with code 2, the task is not created and the stderr message is fed back to the model as feedback. To stop the teammate entirely instead of re-running it, return JSON with `{"continue": false, "stopReason": "..."}` . TaskCreated hooks do not support matchers and fire on every occurrence.

##### TaskCreated input

In addition to the [common input fields](#common-input-fields) , TaskCreated hooks receive `task_id` , `task_subject` , and optionally `task_description` , `teammate_name` , and `team_name` .

```
{
"session_id" : "abc123" ,
"transcript_path" : "/Users/.../.claude/projects/.../00893aaf-19fa-41d2-8238-13269b9b3ca0.jsonl" ,
"cwd" : "/Users/..." ,
"permission_mode" : "default" ,
"hook_event_name" : "TaskCreated" ,
"task_id" : "task-001" ,
"task_subject" : "Implement user authentication" ,
"task_description" : "Add login and signup endpoints" ,
"teammate_name" : "implementer" ,
"team_name" : "my-project"
}
```

| Field              | Description                                           |
|--------------------|-------------------------------------------------------|
| `task_id`          | Identifier of the task being created                  |
| `task_subject`     | Title of the task                                     |
| `task_description` | Detailed description of the task. May be absent       |
| `teammate_name`    | Name of the teammate creating the task. May be absent |
| `team_name`        | Name of the team. May be absent                       |

##### TaskCreated decision control

TaskCreated hooks support two ways to control task creation:

- **Exit code 2** : the task is not created and the stderr message is fed back to the model as feedback.
- **JSON** **`{"continue": false, "stopReason": "..."}`** : stops the teammate entirely, matching `Stop` hook behavior. The `stopReason` is shown to the user.

This example blocks tasks whose subjects don't follow the required format:

```
#!/bin/bash
INPUT = $( cat )
TASK_SUBJECT = $( echo " $INPUT " | jq -r '.task_subject' )

if [[ ! " $TASK_SUBJECT " =~ ^ \[ TICKET-[0-9]+ \] ]]; then
echo "Task subject must start with a ticket number, e.g. '[TICKET-123] Add feature'" >&2
exit 2
fi

exit 0
```

#### TaskCompleted

Runs when a task is being marked as completed. This fires in two situations: when any agent explicitly marks a task as completed through the TaskUpdate tool, or when an [agent team](/docs/en/agent-teams) teammate finishes its turn with in-progress tasks. Use this to enforce completion criteria like passing tests or lint checks before a task can close. When a `TaskCompleted` hook exits with code 2, the task is not marked as completed and the stderr message is fed back to the model as feedback. To stop the teammate entirely instead of re-running it, return JSON with `{"continue": false, "stopReason": "..."}` . TaskCompleted hooks do not support matchers and fire on every occurrence.

##### TaskCompleted input

In addition to the [common input fields](#common-input-fields) , TaskCompleted hooks receive `task_id` , `task_subject` , and optionally `task_description` , `teammate_name` , and `team_name` .

```
{
"session_id" : "abc123" ,
"transcript_path" : "/Users/.../.claude/projects/.../00893aaf-19fa-41d2-8238-13269b9b3ca0.jsonl" ,
"cwd" : "/Users/..." ,
"permission_mode" : "default" ,
"hook_event_name" : "TaskCompleted" ,
"task_id" : "task-001" ,
"task_subject" : "Implement user authentication" ,
"task_description" : "Add login and signup endpoints" ,
"teammate_name" : "implementer" ,
"team_name" : "my-project"
}
```

| Field              | Description                                             |
|--------------------|---------------------------------------------------------|
| `task_id`          | Identifier of the task being completed                  |
| `task_subject`     | Title of the task                                       |
| `task_description` | Detailed description of the task. May be absent         |
| `teammate_name`    | Name of the teammate completing the task. May be absent |
| `team_name`        | Name of the team. May be absent                         |

##### TaskCompleted decision control

TaskCompleted hooks support two ways to control task completion:

- **Exit code 2** : the task is not marked as completed and the stderr message is fed back to the model as feedback.
- **JSON** **`{"continue": false, "stopReason": "..."}`** : stops the teammate entirely, matching `Stop` hook behavior. The `stopReason` is shown to the user.

This example runs tests and blocks task completion if they fail:

```
#!/bin/bash
INPUT = $( cat )
TASK_SUBJECT = $( echo " $INPUT " | jq -r '.task_subject' )

### Run the test suite
if ! npm test 2>&1 ; then
echo "Tests not passing. Fix failing tests before completing: $TASK_SUBJECT " >&2
exit 2
fi

exit 0
```

#### Stop

Runs when the main Claude Code agent has finished responding. Does not run if

the stoppage occurred due to a user interrupt. API errors fire

[StopFailure](#stopfailure) instead.

##### Stop input

In addition to the [common input fields](#common-input-fields) , Stop hooks receive `stop_hook_active` and `last_assistant_message` . The `stop_hook_active` field is `true` when Claude Code is already continuing as a result of a stop hook. Check this value or process the transcript to prevent Claude Code from running indefinitely. The `last_assistant_message` field contains the text content of Claude's final response, so hooks can access it without parsing the transcript file.

```
{
"session_id" : "abc123" ,
"transcript_path" : "~/.claude/projects/.../00893aaf-19fa-41d2-8238-13269b9b3ca0.jsonl" ,
"cwd" : "/Users/..." ,
"permission_mode" : "default" ,
"hook_event_name" : "Stop" ,
"stop_hook_active" : true ,
"last_assistant_message" : "I've completed the refactoring. Here's a summary..."
}
```

##### Stop decision control

`Stop` and `SubagentStop` hooks can control whether Claude continues. In addition to the [JSON output fields](#json-output) available to all hooks, your hook script can return these event-specific fields:

| Field      | Description                                                                 |
|------------|-----------------------------------------------------------------------------|
| `decision` | `"block"` prevents Claude from stopping. Omit to allow Claude to stop       |
| `reason`   | Required when `decision` is `"block"` . Tells Claude why it should continue |

```
{
"decision" : "block" ,
"reason" : "Must be provided when Claude is blocked from stopping"
}
```

#### StopFailure

Runs instead of [Stop](#stop) when the turn ends due to an API error. Output and exit code are ignored. Use this to log failures, send alerts, or take recovery actions when Claude cannot complete a response due to rate limits, authentication problems, or other API errors.

##### StopFailure input

In addition to the [common input fields](#common-input-fields) , StopFailure hooks receive `error` , optional `error_details` , and optional `last_assistant_message` . The `error` field identifies the error type and is used for matcher filtering.

| Field                    | Description                                                                                                                                                                                                                                       |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `error`                  | Error type: `rate_limit` , `authentication_failed` , `billing_error` , `invalid_request` , `server_error` , `max_output_tokens` , or `unknown`                                                                                                    |
| `error_details`          | Additional details about the error, when available                                                                                                                                                                                                |
| `last_assistant_message` | The rendered error text shown in the conversation. Unlike `Stop` and `SubagentStop` , where this field holds Claude's conversational output, for `StopFailure` it contains the API error string itself, such as `"API Error: Rate limit reached"` |

```
{
"session_id" : "abc123" ,
"transcript_path" : "/Users/.../.claude/projects/.../00893aaf-19fa-41d2-8238-13269b9b3ca0.jsonl" ,
"cwd" : "/Users/..." ,
"hook_event_name" : "StopFailure" ,
"error" : "rate_limit" ,
"error_details" : "429 Too Many Requests" ,
"last_assistant_message" : "API Error: Rate limit reached"
}
```

StopFailure hooks have no decision control. They run for notification and logging purposes only.

#### TeammateIdle

Runs when an [agent team](/docs/en/agent-teams) teammate is about to go idle after finishing its turn. Use this to enforce quality gates before a teammate stops working, such as requiring passing lint checks or verifying that output files exist. When a `TeammateIdle` hook exits with code 2, the teammate receives the stderr message as feedback and continues working instead of going idle. To stop the teammate entirely instead of re-running it, return JSON with `{"continue": false, "stopReason": "..."}` . TeammateIdle hooks do not support matchers and fire on every occurrence.

##### TeammateIdle input

In addition to the [common input fields](#common-input-fields) , TeammateIdle hooks receive `teammate_name` and `team_name` .

```
{
"session_id" : "abc123" ,
"transcript_path" : "/Users/.../.claude/projects/.../00893aaf-19fa-41d2-8238-13269b9b3ca0.jsonl" ,
"cwd" : "/Users/..." ,
"permission_mode" : "default" ,
"hook_event_name" : "TeammateIdle" ,
"teammate_name" : "researcher" ,
"team_name" : "my-project"
}
```

| Field           | Description                                   |
|-----------------|-----------------------------------------------|
| `teammate_name` | Name of the teammate that is about to go idle |
| `team_name`     | Name of the team                              |

##### TeammateIdle decision control

TeammateIdle hooks support two ways to control teammate behavior:

- **Exit code 2** : the teammate receives the stderr message as feedback and continues working instead of going idle.
- **JSON** **`{"continue": false, "stopReason": "..."}`** : stops the teammate entirely, matching `Stop` hook behavior. The `stopReason` is shown to the user.

This example checks that a build artifact exists before allowing a teammate to go idle:

```
#!/bin/bash

if [ ! -f "./dist/output.js" ]; then
echo "Build artifact missing. Run the build before stopping." >&2
exit 2
fi

exit 0
```

#### ConfigChange

Runs when a configuration file changes during a session. Use this to audit settings changes, enforce security policies, or block unauthorized modifications to configuration files. ConfigChange hooks fire for changes to settings files, managed policy settings, and skill files. The `source` field in the input tells you which type of configuration changed, and the optional `file_path` field provides the path to the changed file. The matcher filters on the configuration source:

| Matcher            | When it fires                             |
|--------------------|-------------------------------------------|
| `user_settings`    | `~/.claude/settings.json` changes         |
| `project_settings` | `.claude/settings.json` changes           |
| `local_settings`   | `.claude/settings.local.json` changes     |
| `policy_settings`  | Managed policy settings change            |
| `skills`           | A skill file in `.claude/skills/` changes |

This example logs all configuration changes for security auditing:

```
{
"hooks" : {
"ConfigChange" : [
{
"hooks" : [
{
"type" : "command" ,
"command" : " \" $CLAUDE_PROJECT_DIR \" /.claude/hooks/audit-config-change.sh"
}
]
}
]
}
}
```

##### ConfigChange input

In addition to the [common input fields](#common-input-fields) , ConfigChange hooks receive `source` and optionally `file_path` . The `source` field indicates which configuration type changed, and `file_path` provides the path to the specific file that was modified.

```
{
"session_id" : "abc123" ,
"transcript_path" : "/Users/.../.claude/projects/.../00893aaf-19fa-41d2-8238-13269b9b3ca0.jsonl" ,
"cwd" : "/Users/..." ,
"hook_event_name" : "ConfigChange" ,
"source" : "project_settings" ,
"file_path" : "/Users/.../my-project/.claude/settings.json"
}
```

##### ConfigChange decision control

ConfigChange hooks can block configuration changes from taking effect. Use exit code 2 or a JSON `decision` to prevent the change. When blocked, the new settings are not applied to the running session.

| Field      | Description                                                                              |
|------------|------------------------------------------------------------------------------------------|
| `decision` | `"block"` prevents the configuration change from being applied. Omit to allow the change |
| `reason`   | Explanation shown to the user when `decision` is `"block"`                               |

```
{
"decision" : "block" ,
"reason" : "Configuration changes to project settings require admin approval"
}
```

`policy_settings` changes cannot be blocked. Hooks still fire for `policy_settings` sources, so you can use them for audit logging, but any blocking decision is ignored. This ensures enterprise-managed settings always take effect.

#### CwdChanged

Runs when the working directory changes during a session, for example when Claude executes a `cd` command. Use this to react to directory changes: reload environment variables, activate project-specific toolchains, or run setup scripts automatically. Pairs with [FileChanged](#filechanged) for tools like [direnv](https://direnv.net/) that manage per-directory environment. CwdChanged hooks have access to `CLAUDE_ENV_FILE` . Variables written to that file persist into subsequent Bash commands for the session, just as in [SessionStart hooks](#persist-environment-variables) . Only `type: "command"` hooks are supported. CwdChanged does not support matchers and fires on every directory change.

##### CwdChanged input

In addition to the [common input fields](#common-input-fields) , CwdChanged hooks receive `old_cwd` and `new_cwd` .

```
{
"session_id" : "abc123" ,
"transcript_path" : "/Users/.../.claude/projects/.../transcript.jsonl" ,
"cwd" : "/Users/my-project/src" ,
"hook_event_name" : "CwdChanged" ,
"old_cwd" : "/Users/my-project" ,
"new_cwd" : "/Users/my-project/src"
}
```

##### CwdChanged output

In addition to the [JSON output fields](#json-output) available to all hooks, CwdChanged hooks can return `watchPaths` to dynamically set which file paths [FileChanged](#filechanged) watches:

| Field        | Description                                                                                                                                                                                                                     |
|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `watchPaths` | Array of absolute paths. Replaces the current dynamic watch list (paths from your `matcher` configuration are always watched). Returning an empty array clears the dynamic list, which is typical when entering a new directory |

CwdChanged hooks have no decision control. They cannot block the directory change.

#### FileChanged

Runs when a watched file changes on disk. Useful for reloading environment variables when project configuration files are modified. The `matcher` for this event serves two roles:

- **Build the watch list** : the value is split on `|` and each segment is registered as a literal filename in the working directory, so `".envrc|.env"` watches exactly those two files. Regex patterns are not useful here: a value like `^\.env` would watch a file literally named `^\.env` .
- **Filter which hooks run** : when a watched file changes, the same value filters which hook groups run using the standard [matcher rules](#matcher-patterns) against the changed file's basename.

FileChanged hooks have access to `CLAUDE_ENV_FILE` . Variables written to that file persist into subsequent Bash commands for the session, just as in [SessionStart hooks](#persist-environment-variables) . Only `type: "command"` hooks are supported.

##### FileChanged input

In addition to the [common input fields](#common-input-fields) , FileChanged hooks receive `file_path` and `event` .

| Field       | Description                                                                                     |
|-------------|-------------------------------------------------------------------------------------------------|
| `file_path` | Absolute path to the file that changed                                                          |
| `event`     | What happened: `"change"` (file modified), `"add"` (file created), or `"unlink"` (file deleted) |

```
{
"session_id" : "abc123" ,
"transcript_path" : "/Users/.../.claude/projects/.../transcript.jsonl" ,
"cwd" : "/Users/my-project" ,
"hook_event_name" : "FileChanged" ,
"file_path" : "/Users/my-project/.envrc" ,
"event" : "change"
}
```

##### FileChanged output

In addition to the [JSON output fields](#json-output) available to all hooks, FileChanged hooks can return `watchPaths` to dynamically update which file paths are watched:

| Field        | Description                                                                                                                                                                                                                 |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `watchPaths` | Array of absolute paths. Replaces the current dynamic watch list (paths from your `matcher` configuration are always watched). Use this when your hook script discovers additional files to watch based on the changed file |

FileChanged hooks have no decision control. They cannot block the file change from occurring.

#### WorktreeCreate

When you run `claude --worktree` or a [subagent uses](/docs/en/sub-agents#choose-the-subagent-scope) [`isolation: "worktree"`](/docs/en/sub-agents#choose-the-subagent-scope) , Claude Code creates an isolated working copy using `git worktree` . If you configure a WorktreeCreate hook, it replaces the default git behavior, letting you use a different version control system like SVN, Perforce, or Mercurial. Because the hook replaces the default behavior entirely, [`.worktreeinclude`](/docs/en/common-workflows#copy-gitignored-files-to-worktrees) is not processed. If you need to copy local configuration files like `.env` into the new worktree, do it inside your hook script. The hook must return the absolute path to the created worktree directory. Claude Code uses this path as the working directory for the isolated session. Command hooks print it on stdout; HTTP hooks return it via `hookSpecificOutput.worktreePath` . This example creates an SVN working copy and prints the path for Claude Code to use. Replace the repository URL with your own:

```
{
"hooks" : {
"WorktreeCreate" : [
{
"hooks" : [
{
"type" : "command" ,
"command" : "bash -c 'NAME=$(jq -r .name); DIR= \" $HOME/.claude/worktrees/$NAME \" ; svn checkout https://svn.example.com/repo/trunk \" $DIR \" >&2 && echo \" $DIR \" '"
}
]
}
]
}
}
```

The hook reads the worktree `name` from the JSON input on stdin, checks out a fresh copy into a new directory, and prints the directory path. The `echo` on the last line is what Claude Code reads as the worktree path. Redirect any other output to stderr so it doesn't interfere with the path.

##### WorktreeCreate input

In addition to the [common input fields](#common-input-fields) , WorktreeCreate hooks receive the `name` field. This is a slug identifier for the new worktree, either specified by the user or auto-generated (for example, `bold-oak-a3f2` ).

```
{
"session_id" : "abc123" ,
"transcript_path" : "/Users/.../.claude/projects/.../00893aaf-19fa-41d2-8238-13269b9b3ca0.jsonl" ,
"cwd" : "/Users/..." ,
"hook_event_name" : "WorktreeCreate" ,
"name" : "feature-auth"
}
```

##### WorktreeCreate output

WorktreeCreate hooks do not use the standard allow/block decision model. Instead, the hook's success or failure determines the outcome. The hook must return the absolute path to the created worktree directory:

- **Command hooks** ( `type: "command"` ): print the path on stdout.
- **HTTP hooks** ( `type: "http"` ): return `{ "hookSpecificOutput": { "hookEventName": "WorktreeCreate", "worktreePath": "/absolute/path" } }` in the response body.

If the hook fails or produces no path, worktree creation fails with an error.

#### WorktreeRemove

The cleanup counterpart to [WorktreeCreate](#worktreecreate) . This hook fires when a worktree is being removed, either when you exit a `--worktree` session and choose to remove it, or when a subagent with `isolation: "worktree"` finishes. For git-based worktrees, Claude handles cleanup automatically with `git worktree remove` . If you configured a WorktreeCreate hook for a non-git version control system, pair it with a WorktreeRemove hook to handle cleanup. Without one, the worktree directory is left on disk. Claude Code passes the path returned by WorktreeCreate as `worktree_path` in the hook input. This example reads that path and removes the directory:

```
{
"hooks" : {
"WorktreeRemove" : [
{
"hooks" : [
{
"type" : "command" ,
"command" : "bash -c 'jq -r .worktree_path | xargs rm -rf'"
}
]
}
]
}
}
```

##### WorktreeRemove input

In addition to the [common input fields](#common-input-fields) , WorktreeRemove hooks receive the `worktree_path` field, which is the absolute path to the worktree being removed.

```
{
"session_id" : "abc123" ,
"transcript_path" : "/Users/.../.claude/projects/.../00893aaf-19fa-41d2-8238-13269b9b3ca0.jsonl" ,
"cwd" : "/Users/..." ,
"hook_event_name" : "WorktreeRemove" ,
"worktree_path" : "/Users/.../my-project/.claude/worktrees/feature-auth"
}
```

WorktreeRemove hooks have no decision control. They cannot block worktree removal but can perform cleanup tasks like removing version control state or archiving changes. Hook failures are logged in debug mode only.

#### PreCompact

Runs before Claude Code is about to run a compact operation. The matcher value indicates whether compaction was triggered manually or automatically:

| Matcher   | When it fires                                |
|-----------|----------------------------------------------|
| `manual`  | `/compact`                                   |
| `auto`    | Auto-compact when the context window is full |

##### PreCompact input

In addition to the [common input fields](#common-input-fields) , PreCompact hooks receive `trigger` and `custom_instructions` . For `manual` , `custom_instructions` contains what the user passes into `/compact` . For `auto` , `custom_instructions` is empty.

```
{
"session_id" : "abc123" ,
"transcript_path" : "/Users/.../.claude/projects/.../00893aaf-19fa-41d2-8238-13269b9b3ca0.jsonl" ,
"cwd" : "/Users/..." ,
"hook_event_name" : "PreCompact" ,
"trigger" : "manual" ,
"custom_instructions" : ""
}
```

#### PostCompact

Runs after Claude Code completes a compact operation. Use this event to react to the new compacted state, for example to log the generated summary or update external state. The same matcher values apply as for `PreCompact` :

| Matcher   | When it fires                                      |
|-----------|----------------------------------------------------|
| `manual`  | After `/compact`                                   |
| `auto`    | After auto-compact when the context window is full |

##### PostCompact input

In addition to the [common input fields](#common-input-fields) , PostCompact hooks receive `trigger` and `compact_summary` . The `compact_summary` field contains the conversation summary generated by the compact operation.

```
{
"session_id" : "abc123" ,
"transcript_path" : "/Users/.../.claude/projects/.../00893aaf-19fa-41d2-8238-13269b9b3ca0.jsonl" ,
"cwd" : "/Users/..." ,
"hook_event_name" : "PostCompact" ,
"trigger" : "manual" ,
"compact_summary" : "Summary of the compacted conversation..."
}
```

PostCompact hooks have no decision control. They cannot affect the compaction result but can perform follow-up tasks.

#### SessionEnd

Runs when a Claude Code session ends. Useful for cleanup tasks, logging session

statistics, or saving session state. Supports matchers to filter by exit reason. The

`reason` field in the hook input indicates why the session ended:

| Reason                        | Description                                |
|-------------------------------|--------------------------------------------|
| `clear`                       | Session cleared with `/clear` command      |
| `resume`                      | Session switched via interactive `/resume` |
| `logout`                      | User logged out                            |
| `prompt_input_exit`           | User exited while prompt input was visible |
| `bypass_permissions_disabled` | Bypass permissions mode was disabled       |
| `other`                       | Other exit reasons                         |

##### SessionEnd input

In addition to the [common input fields](#common-input-fields) , SessionEnd hooks receive a `reason` field indicating why the session ended. See the [reason table](#sessionend) above for all values.

```
{
"session_id" : "abc123" ,
"transcript_path" : "/Users/.../.claude/projects/.../00893aaf-19fa-41d2-8238-13269b9b3ca0.jsonl" ,
"cwd" : "/Users/..." ,
"hook_event_name" : "SessionEnd" ,
"reason" : "other"
}
```

SessionEnd hooks have no decision control. They cannot block session termination but can perform cleanup tasks. SessionEnd hooks have a default timeout of 1.5 seconds. This applies to session exit, `/clear` , and switching sessions via interactive `/resume` . If your hooks need more time, set the `CLAUDE_CODE_SESSIONEND_HOOKS_TIMEOUT_MS` environment variable to a higher value in milliseconds. Any per-hook `timeout` setting is also capped by this value.

```
CLAUDE_CODE_SESSIONEND_HOOKS_TIMEOUT_MS = 5000 claude
```

#### Elicitation

Runs when an MCP server requests user input mid-task. By default, Claude Code shows an interactive dialog for the user to respond. Hooks can intercept this request and respond programmatically, skipping the dialog entirely. The matcher field matches against the MCP server name.

##### Elicitation input

In addition to the [common input fields](#common-input-fields) , Elicitation hooks receive `mcp_server_name` , `message` , and optional `mode` , `url` , `elicitation_id` , and `requested_schema` fields. For form-mode elicitation (the most common case):

```
{
"session_id" : "abc123" ,
"transcript_path" : "/Users/.../.claude/projects/.../00893aaf-19fa-41d2-8238-13269b9b3ca0.jsonl" ,
"cwd" : "/Users/..." ,
"permission_mode" : "default" ,
"hook_event_name" : "Elicitation" ,
"mcp_server_name" : "my-mcp-server" ,
"message" : "Please provide your credentials" ,
"mode" : "form" ,
"requested_schema" : {
"type" : "object" ,
"properties" : {
"username" : { "type" : "string" , "title" : "Username" }
}
}
}
```

For URL-mode elicitation (browser-based authentication):

```
{
"session_id" : "abc123" ,
"transcript_path" : "/Users/.../.claude/projects/.../00893aaf-19fa-41d2-8238-13269b9b3ca0.jsonl" ,
"cwd" : "/Users/..." ,
"permission_mode" : "default" ,
"hook_event_name" : "Elicitation" ,
"mcp_server_name" : "my-mcp-server" ,
"message" : "Please authenticate" ,
"mode" : "url" ,
"url" : "https://auth.example.com/login"
}
```

##### Elicitation output

To respond programmatically without showing the dialog, return a JSON object with `hookSpecificOutput` :

```
{
"hookSpecificOutput" : {
"hookEventName" : "Elicitation" ,
"action" : "accept" ,
"content" : {
"username" : "alice"
}
}
}
```

| Field     | Values                          | Description                                                      |
|-----------|---------------------------------|------------------------------------------------------------------|
| `action`  | `accept` , `decline` , `cancel` | Whether to accept, decline, or cancel the request                |
| `content` | object                          | Form field values to submit. Only used when `action` is `accept` |

Exit code 2 denies the elicitation and shows stderr to the user.

#### ElicitationResult

Runs after a user responds to an MCP elicitation. Hooks can observe, modify, or block the response before it is sent back to the MCP server. The matcher field matches against the MCP server name.

##### ElicitationResult input

In addition to the [common input fields](#common-input-fields) , ElicitationResult hooks receive `mcp_server_name` , `action` , and optional `mode` , `elicitation_id` , and `content` fields.

```
{
"session_id" : "abc123" ,
"transcript_path" : "/Users/.../.claude/projects/.../00893aaf-19fa-41d2-8238-13269b9b3ca0.jsonl" ,
"cwd" : "/Users/..." ,
"permission_mode" : "default" ,
"hook_event_name" : "ElicitationResult" ,
"mcp_server_name" : "my-mcp-server" ,
"action" : "accept" ,
"content" : { "username" : "alice" },
"mode" : "form" ,
"elicitation_id" : "elicit-123"
}
```

##### ElicitationResult output

To override the user's response, return a JSON object with `hookSpecificOutput` :

```
{
"hookSpecificOutput" : {
"hookEventName" : "ElicitationResult" ,
"action" : "decline" ,
"content" : {}
}
}
```

| Field     | Values                          | Description                                                            |
|-----------|---------------------------------|------------------------------------------------------------------------|
| `action`  | `accept` , `decline` , `cancel` | Overrides the user's action                                            |
| `content` | object                          | Overrides form field values. Only meaningful when `action` is `accept` |

Exit code 2 blocks the response, changing the effective action to `decline` .

### Prompt-based hooks

In addition to command and HTTP hooks, Claude Code supports prompt-based hooks ( `type: "prompt"` ) that use an LLM to evaluate whether to allow or block an action, and agent hooks ( `type: "agent"` ) that spawn an agentic verifier with tool access. Not all events support every hook type. Events that support all four hook types ( `command` , `http` , `prompt` , and `agent` ):

- PermissionRequest
- PostToolUse
- PostToolUseFailure
- PreToolUse
- Stop
- SubagentStop
- TaskCompleted
- TaskCreated
- UserPromptSubmit

Events that support `command` and `http` hooks but not `prompt` or `agent` :

- ConfigChange
- CwdChanged
- Elicitation
- ElicitationResult
- FileChanged
- InstructionsLoaded
- Notification
- PermissionDenied
- PostCompact
- PreCompact
- SessionEnd
- StopFailure
- SubagentStart
- TeammateIdle
- WorktreeCreate
- WorktreeRemove

`SessionStart` supports only `command` hooks.

#### How prompt-based hooks work

Instead of executing a Bash command, prompt-based hooks:

1. Send the hook input and your prompt to a Claude model, Haiku by default
2. The LLM responds with structured JSON containing a decision
3. Claude Code processes the decision automatically

#### Prompt hook configuration

Set `type` to `"prompt"` and provide a `prompt` string instead of a `command` . Use the `$ARGUMENTS` placeholder to inject the hook's JSON input data into your prompt text. Claude Code sends the combined prompt and input to a fast Claude model, which returns a JSON decision. This `Stop` hook asks the LLM to evaluate whether all tasks are complete before allowing Claude to finish:

```
{
"hooks" : {
"Stop" : [
{
"hooks" : [
{
"type" : "prompt" ,
"prompt" : "Evaluate if Claude should stop: $ARGUMENTS. Check if all tasks are complete."
}
]
}
]
}
}
```

| Field     | Required   | Description                                                                                                                                                         |
|-----------|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `type`    | yes        | Must be `"prompt"`                                                                                                                                                  |
| `prompt`  | yes        | The prompt text to send to the LLM. Use `$ARGUMENTS` as a placeholder for the hook input JSON. If `$ARGUMENTS` is not present, input JSON is appended to the prompt |
| `model`   | no         | Model to use for evaluation. Defaults to a fast model                                                                                                               |
| `timeout` | no         | Timeout in seconds. Default: 30                                                                                                                                     |

#### Response schema

The LLM must respond with JSON containing:

```
{
"ok" : true | false ,
"reason" : "Explanation for the decision"
}
```

| Field    | Description                                                 |
|----------|-------------------------------------------------------------|
| `ok`     | `true` allows the action, `false` prevents it               |
| `reason` | Required when `ok` is `false` . Explanation shown to Claude |

#### Example: Multi-criteria Stop hook

This `Stop` hook uses a detailed prompt to check three conditions before allowing Claude to stop. If `"ok"` is `false` , Claude continues working with the provided reason as its next instruction. `SubagentStop` hooks use the same format to evaluate whether a [subagent](/docs/en/sub-agents) should stop:

```
{
"hooks" : {
"Stop" : [
{
"hooks" : [
{
"type" : "prompt" ,
"prompt" : "You are evaluating whether Claude should stop working. Context: $ARGUMENTS \n\n Analyze the conversation and determine if: \n 1. All user-requested tasks are complete \n 2. Any errors need to be addressed \n 3. Follow-up work is needed \n\n Respond with JSON: { \" ok \" : true} to allow stopping, or { \" ok \" : false, \" reason \" : \" your explanation \" } to continue working." ,
"timeout" : 30
}
]
}
]
}
}
```

### Agent-based hooks

Agent-based hooks ( `type: "agent"` ) are like prompt-based hooks but with multi-turn tool access. Instead of a single LLM call, an agent hook spawns a subagent that can read files, search code, and inspect the codebase to verify conditions. Agent hooks support the same events as prompt-based hooks.

#### How agent hooks work

When an agent hook fires:

1. Claude Code spawns a subagent with your prompt and the hook's JSON input
2. The subagent can use tools like Read, Grep, and Glob to investigate
3. After up to 50 turns, the subagent returns a structured `{ "ok": true/false }` decision
4. Claude Code processes the decision the same way as a prompt hook

Agent hooks are useful when verification requires inspecting actual files or test output, not just evaluating the hook input data alone.

#### Agent hook configuration

Set `type` to `"agent"` and provide a `prompt` string. The configuration fields are the same as [prompt hooks](#prompt-hook-configuration) , with a longer default timeout:

| Field     | Required   | Description                                                                                 |
|-----------|------------|---------------------------------------------------------------------------------------------|
| `type`    | yes        | Must be `"agent"`                                                                           |
| `prompt`  | yes        | Prompt describing what to verify. Use `$ARGUMENTS` as a placeholder for the hook input JSON |
| `model`   | no         | Model to use. Defaults to a fast model                                                      |
| `timeout` | no         | Timeout in seconds. Default: 60                                                             |

The response schema is the same as prompt hooks: `{ "ok": true }` to allow or `{ "ok": false, "reason": "..." }` to block. This `Stop` hook verifies that all unit tests pass before allowing Claude to finish:

```
{
"hooks" : {
"Stop" : [
{
"hooks" : [
{
"type" : "agent" ,
"prompt" : "Verify that all unit tests pass. Run the test suite and check the results. $ARGUMENTS" ,
"timeout" : 120
}
]
}
]
}
}
```

### Run hooks in the background

By default, hooks block Claude's execution until they complete. For long-running tasks like deployments, test suites, or external API calls, set `"async": true` to run the hook in the background while Claude continues working. Async hooks cannot block or control Claude's behavior: response fields like `decision` , `permissionDecision` , and `continue` have no effect, because the action they would have controlled has already completed.

#### Configure an async hook

Add `"async": true` to a command hook's configuration to run it in the background without blocking Claude. This field is only available on `type: "command"` hooks. This hook runs a test script after every `Write` tool call. Claude continues working immediately while `run-tests.sh` executes for up to 120 seconds. When the script finishes, its output is delivered on the next conversation turn:

```
{
"hooks" : {
"PostToolUse" : [
{
"matcher" : "Write" ,
"hooks" : [
{
"type" : "command" ,
"command" : "/path/to/run-tests.sh" ,
"async" : true ,
"timeout" : 120
}
]
}
]
}
}
```

The `timeout` field sets the maximum time in seconds for the background process. If not specified, async hooks use the same 10-minute default as sync hooks.

#### How async hooks execute

When an async hook fires, Claude Code starts the hook process and immediately continues without waiting for it to finish. The hook receives the same JSON input via stdin as a synchronous hook. After the background process exits, if the hook produced a JSON response with a `systemMessage` or `additionalContext` field, that content is delivered to Claude as context on the next conversation turn. Async hook completion notifications are suppressed by default. To see them, enable verbose mode with `Ctrl+O` or start Claude Code with `--verbose` .

#### Example: run tests after file changes

This hook starts a test suite in the background whenever Claude writes a file, then reports the results back to Claude when the tests finish. Save this script to `.claude/hooks/run-tests-async.sh` in your project and make it executable with `chmod +x` :

```
#!/bin/bash
### run-tests-async.sh

### Read hook input from stdin
INPUT = $( cat )
FILE_PATH = $( echo " $INPUT " | jq -r '.tool_input.file_path // empty' )

### Only run tests for source files
if [[ " $FILE_PATH " != * .ts && " $FILE_PATH " != * .js ]]; then
exit 0
fi

### Run tests and report results via systemMessage
RESULT = $( npm test 2>&1 )
EXIT_CODE = $?

if [ $EXIT_CODE -eq 0 ]; then
echo "{ \" systemMessage \" : \" Tests passed after editing $FILE_PATH \" }"
else
echo "{ \" systemMessage \" : \" Tests failed after editing $FILE_PATH : $RESULT \" }"
fi
```

Then add this configuration to `.claude/settings.json` in your project root. The `async: true` flag lets Claude keep working while tests run:

```
{
"hooks" : {
"PostToolUse" : [
{
"matcher" : "Write|Edit" ,
"hooks" : [
{
"type" : "command" ,
"command" : " \" $CLAUDE_PROJECT_DIR \" /.claude/hooks/run-tests-async.sh" ,
"async" : true ,
"timeout" : 300
}
]
}
]
}
}
```

#### Limitations

Async hooks have several constraints compared to synchronous hooks:

- Only `type: "command"` hooks support `async` . Prompt-based hooks cannot run asynchronously.
- Async hooks cannot block tool calls or return decisions. By the time the hook completes, the triggering action has already proceeded.
- Hook output is delivered on the next conversation turn. If the session is idle, the response waits until the next user interaction.
- Each execution creates a separate background process. There is no deduplication across multiple firings of the same async hook.

### Security considerations

#### Disclaimer

Command hooks run with your system user's full permissions.

Command hooks execute shell commands with your full user permissions. They can modify, delete, or access any files your user account can access. Review and test all hook commands before adding them to your configuration.

#### Security best practices

Keep these practices in mind when writing hooks:

- **Validate and sanitize inputs** : never trust input data blindly
- **Always quote shell variables** : use `"$VAR"` not `$VAR`
- **Block path traversal** : check for `..` in file paths
- **Use absolute paths** : specify full paths for scripts, using `"$CLAUDE_PROJECT_DIR"` for the project root
- **Skip sensitive files** : avoid `.env` , `.git/` , keys, etc.

### Windows PowerShell tool

On Windows, you can run individual hooks in PowerShell by setting `"shell": "powershell"` on a command hook. Hooks spawn PowerShell directly, so this works regardless of whether `CLAUDE_CODE_USE_POWERSHELL_TOOL` is set. Claude Code auto-detects `pwsh.exe` (PowerShell 7+) with a fallback to `powershell.exe` (5.1).

```
{
"hooks" : {
"PostToolUse" : [
{
"matcher" : "Write" ,
"hooks" : [
{
"type" : "command" ,
"shell" : "powershell" ,
"command" : "Write-Host 'File written'"
}
]
}
]
}
}
```

### Debug hooks

Hook execution details, including which hooks matched, their exit codes, and full stdout and stderr, are written to the debug log file. Start Claude Code with `claude --debug-file <path>` to write the log to a known location, or run `claude --debug` and read the log at `~/.claude/debug/<session-id>.txt` . The `--debug` flag does not print to the terminal.

```
[DEBUG] Executing hooks for PostToolUse:Write
[DEBUG] Found 1 hook commands to execute
[DEBUG] Executing hook command: <Your command> with timeout 600000ms
[DEBUG] Hook command completed with status 0: <Your stdout>
```

For more granular hook matching details, set `CLAUDE_CODE_DEBUG_LOG_LEVEL=verbose` to see additional log lines such as hook matcher counts and query matching. For troubleshooting common issues like hooks not firing, infinite Stop hook loops, or configuration errors, see [Limitations and troubleshooting](/docs/en/hooks-guide#limitations-and-troubleshooting) in the guide.

Was this page helpful?

Yes

No

[Checkpointing](/docs/en/checkpointing) [Plugins reference](/docs/en/plugins-reference)

⌘ I


### Schedule tasks on the web


Schedule recurring Claude Code tasks on a cron-like interval. Automate PR reviews, dependency audits, and CI triage in cloud sessions.


A scheduled task runs a prompt on a recurring cadence using Anthropic-managed infrastructure. Tasks keep working even when your computer is off. A few examples of recurring work you can automate:

- Reviewing open pull requests each morning
- Analyzing CI failures overnight and surfacing summaries
- Syncing documentation after PRs merge
- Running dependency audits every week

Scheduled tasks are available to all Claude Code on the web users, including Pro, Max, Team, and Enterprise.

### Compare scheduling options

Claude Code offers three ways to schedule recurring work:

|                            | [Cloud](/docs/en/web-scheduled-tasks)   | [Desktop](/docs/en/desktop-scheduled-tasks)   | [`/loop`](/docs/en/scheduled-tasks)   |
|----------------------------|-----------------------------------------|-----------------------------------------------|---------------------------------------|
| Runs on                    | Anthropic cloud                         | Your machine                                  | Your machine                          |
| Requires machine on        | No                                      | Yes                                           | Yes                                   |
| Requires open session      | No                                      | No                                            | Yes                                   |
| Persistent across restarts | Yes                                     | Yes                                           | No (session-scoped)                   |
| Access to local files      | No (fresh clone)                        | Yes                                           | Yes                                   |
| MCP servers                | Connectors configured per task          | [Config files](/docs/en/mcp) and connectors   | Inherits from session                 |
| Permission prompts         | No (runs autonomously)                  | Configurable per task                         | Inherits from session                 |
| Customizable schedule      | Via `/schedule` in the CLI              | Yes                                           | Yes                                   |
| Minimum interval           | 1 hour                                  | 1 minute                                      | 1 minute                              |

Use **cloud tasks** for work that should run reliably without your machine. Use **Desktop tasks** when you need access to local files and tools. Use **`/loop`** for quick polling during a session.

### Create a scheduled task

You can create a scheduled task from three places:

- **Web** : visit [claude.ai/code/scheduled](https://claude.ai/code/scheduled) and click **New scheduled task**
- **Desktop app** : open the **Schedule** page, click **New task** , and choose **New remote task** . See [Desktop scheduled tasks](/docs/en/desktop-scheduled-tasks) for details.
- **CLI** : run `/schedule` in any session. Claude walks you through the setup conversationally. You can also pass a description directly, like `/schedule daily PR review at 9am` .

The web and Desktop entry points open a form. The CLI collects the same information through a guided conversation. The steps below walk through the web interface.

1

Open the creation form

Visit [claude.ai/code/scheduled](https://claude.ai/code/scheduled) and click **New scheduled task** .

2

Name the task and write the prompt

Give the task a descriptive name and write the prompt Claude runs each time. The prompt is the most important part: the task runs autonomously, so the prompt must be self-contained and explicit about what to do and what success looks like. The prompt input includes a model selector. Claude uses this model for each run of the task.

3

Select repositories

Add one or more GitHub repositories for Claude to work in. Each repository is cloned at the start of a run, starting from the default branch. Claude creates `claude/` -prefixed branches for its changes. To allow pushes to any branch, enable **Allow unrestricted branch pushes** for that repository.

4

Select an environment

Select a [cloud environment](/docs/en/claude-code-on-the-web#the-cloud-environment) for the task. Environments control what the cloud session has access to:

- **Network access** : set the level of internet access available during each run
- **Environment variables** : provide API keys, tokens, or other secrets Claude can use
- **Setup script** : run install commands before each session starts, like installing dependencies or configuring tools

A **Default** environment is available out of the box. To use a custom environment, [create one](/docs/en/claude-code-on-the-web#the-cloud-environment) before creating the task.

5

Choose a schedule

Pick how often the task runs from the [frequency options](#frequency-options) . The default is daily at 9:00 AM in your local time zone. Tasks may run a few minutes after their scheduled time due to stagger. If the preset options don't fit your needs, pick the closest one and update the schedule from the CLI with `/schedule update` to set a specific schedule.

6

Review connectors

All of your connected [MCP connectors](/docs/en/mcp) are included by default. Remove any that the task doesn't need. Connectors give Claude access to external services like Slack, Linear, or Google Drive during each run.

7

Create the task

Click **Create** . The task appears in the scheduled tasks list and runs automatically at the next scheduled time. Each run creates a new session alongside your other sessions, where you can see what Claude did, review changes, and create a pull request. To trigger a run immediately, click **Run now** from the task's detail page.

#### Frequency options

The schedule picker offers preset frequencies that handle time zone conversion for you. Pick a time in your local zone and the task runs at that wall-clock time regardless of where the cloud infrastructure is located.

Tasks may run a few minutes after their scheduled time. The offset is consistent for each task.

| Frequency   | Description                                                                |
|-------------|----------------------------------------------------------------------------|
| Hourly      | Runs every hour.                                                           |
| Daily       | Runs once per day at the time you specify. Defaults to 9:00 AM local time. |
| Weekdays    | Same as Daily but skips Saturday and Sunday.                               |
| Weekly      | Runs once per week on the day and time you specify.                        |

For custom intervals like every 2 hours or first of each month, pick the closest preset and update the schedule from the CLI with `/schedule update` to set a specific cron expression. The minimum interval is 1 hour. Expressions that fire more frequently, such as `*/30 * * * *` , are rejected.

#### Repositories and branch permissions

Scheduled tasks need GitHub access to clone repositories. When you create a task from the CLI with `/schedule` , Claude checks whether your account has GitHub connected and prompts you to run `/web-setup` if it doesn't. See [GitHub authentication options](/docs/en/claude-code-on-the-web#github-authentication-options) for the two ways to grant access. Each repository you add is cloned on every run. Claude starts from the repository's default branch unless your prompt specifies otherwise. By default, Claude can only push to branches prefixed with `claude/` . This prevents scheduled tasks from accidentally modifying protected or long-lived branches. To remove this restriction for a specific repository, enable **Allow unrestricted branch pushes** for that repository when creating or editing the task.

#### Connectors

Scheduled tasks can use your connected MCP connectors to read from and write to external services during each run. For example, a task that triages support requests might read from a Slack channel and create issues in Linear. When you create a task, all of your currently connected connectors are included by default. Remove any that aren't needed to limit which tools Claude has access to during the run. You can also add connectors directly from the task form. To manage or add connectors outside of the task form, visit **Settings > Connectors** on claude.ai or use `/schedule update` in the CLI.

#### Environments

Each task runs in a [cloud environment](/docs/en/claude-code-on-the-web#the-cloud-environment) that controls network access, environment variables, and setup scripts. Configure environments before creating a task to give Claude access to APIs, install dependencies, or restrict network scope. See [cloud environment](/docs/en/claude-code-on-the-web#the-cloud-environment) for the full setup guide.

### Manage scheduled tasks

Click a task in the **Scheduled** list to open its detail page. The detail page shows the task's repositories, connectors, prompt, schedule, and a list of past runs.

#### View and interact with runs

Click any run to open it as a full session. From there you can see what Claude did, review changes, create a pull request, or continue the conversation. Each run session works like any other session: use the dropdown menu next to the session title to rename, archive, or delete it.

#### Edit and control tasks

From the task detail page you can:

- Click **Run now** to start a run immediately without waiting for the next scheduled time.
- Use the toggle in the **Repeats** section to pause or resume the schedule. Paused tasks keep their configuration but don't run until you re-enable them.
- Click the edit icon to change the name, prompt, schedule, repositories, environment, or connectors.
- Click the delete icon to remove the task. Past sessions created by the task remain in your session list.

You can also manage tasks from the CLI with `/schedule` . Run `/schedule list` to see all tasks, `/schedule update` to change a task, or `/schedule run` to trigger one immediately.

### Related resources

- [Desktop scheduled tasks](/docs/en/desktop-scheduled-tasks) : schedule tasks that run on your machine with access to local files. The Desktop app's **Schedule** page shows both local and remote tasks in the same grid.
- [`/loop`](/docs/en/scheduled-tasks) [and CLI scheduled tasks](/docs/en/scheduled-tasks) : lightweight scheduling within a CLI session
- [Cloud environment](/docs/en/claude-code-on-the-web#the-cloud-environment) : configure the runtime environment for cloud tasks
- [MCP connectors](/docs/en/mcp) : connect external services like Slack, Linear, and Google Drive
- [GitHub Actions](/docs/en/github-actions) : run Claude in your CI pipeline on repo events

Was this page helpful?

Yes

No

[Plan in the cloud](/docs/en/ultraplan) [Get started](/docs/en/desktop-quickstart)

⌘ I


### Schedule recurring tasks in Claude Code Desktop


Set up scheduled tasks in Claude Code Desktop to run Claude automatically on a recurring basis for daily code reviews, dependency audits, or morning briefings.


By default, scheduled tasks start a new session automatically at a time and frequency you choose. Use them for recurring work like daily code reviews, dependency update checks, or morning briefings that pull from your calendar and inbox.

### Compare scheduling options

Claude Code offers three ways to schedule recurring work:

|                            | [Cloud](/docs/en/web-scheduled-tasks)   | [Desktop](/docs/en/desktop-scheduled-tasks)   | [`/loop`](/docs/en/scheduled-tasks)   |
|----------------------------|-----------------------------------------|-----------------------------------------------|---------------------------------------|
| Runs on                    | Anthropic cloud                         | Your machine                                  | Your machine                          |
| Requires machine on        | No                                      | Yes                                           | Yes                                   |
| Requires open session      | No                                      | No                                            | Yes                                   |
| Persistent across restarts | Yes                                     | Yes                                           | No (session-scoped)                   |
| Access to local files      | No (fresh clone)                        | Yes                                           | Yes                                   |
| MCP servers                | Connectors configured per task          | [Config files](/docs/en/mcp) and connectors   | Inherits from session                 |
| Permission prompts         | No (runs autonomously)                  | Configurable per task                         | Inherits from session                 |
| Customizable schedule      | Via `/schedule` in the CLI              | Yes                                           | Yes                                   |
| Minimum interval           | 1 hour                                  | 1 minute                                      | 1 minute                              |

Use **cloud tasks** for work that should run reliably without your machine. Use **Desktop tasks** when you need access to local files and tools. Use **`/loop`** for quick polling during a session.

The Schedule page supports two kinds of tasks:

- **Local tasks** : run on your machine. They have direct access to your local files and tools, but the desktop app must be open and your computer awake for them to run.
- **Remote tasks** : run on Anthropic-managed cloud infrastructure. They keep running even when your computer is off, but work against a fresh clone of your repository rather than your local checkout.

Both kinds appear in the same task grid. Click **New task** to pick which kind to create. The rest of this page covers local tasks; for remote tasks, see [Cloud scheduled tasks](/docs/en/web-scheduled-tasks) . See [How scheduled tasks run](#how-scheduled-tasks-run) for details on missed runs and catch-up behavior for local tasks.

By default, local scheduled tasks run against whatever state your working directory is in, including uncommitted changes. Enable the worktree toggle in the prompt input to give each run its own isolated Git worktree, the same way [parallel sessions](/docs/en/desktop#work-in-parallel-with-sessions) work.

### Create a scheduled task

To create a local scheduled task, click **Schedule** in the sidebar, click **New task** , and choose **New local task** . Configure these fields:

| Field       | Description                                                                                                                                                                                                              |
|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Name        | Identifier for the task. Converted to lowercase kebab-case and used as the folder name on disk. Must be unique across your tasks.                                                                                        |
| Description | Short summary shown in the task list.                                                                                                                                                                                    |
| Prompt      | The instructions sent to Claude when the task runs. Write this the same way you'd write any message in the prompt box. The prompt input also includes controls for model, permission mode, working folder, and worktree. |
| Frequency   | How often the task runs. See [frequency options](#frequency-options) below.                                                                                                                                              |

You can also create a task by describing what you want in any session. For example, "set up a daily code review that runs every morning at 9am."

### Frequency options

Pick a preset from the frequency dropdown, or ask Claude for anything the picker doesn't cover:

- **Manual** : no schedule, only runs when you click **Run now** . Useful for saving a prompt you trigger on demand
- **Hourly** : runs every hour. Each task gets a fixed offset of up to 10 minutes from the top of the hour to stagger API traffic
- **Daily** : shows a time picker, defaults to 9:00 AM local time
- **Weekdays** : same as Daily but skips Saturday and Sunday
- **Weekly** : shows a time picker and a day picker

For intervals the picker doesn't offer (every 15 minutes, first of each month, etc.), ask Claude in any Desktop session to set the schedule. Use plain language; for example, "schedule a task to run all the tests every 6 hours."

### How scheduled tasks run

Local scheduled tasks run on your machine. Desktop checks the schedule every minute while the app is open and starts a fresh session when a task is due, independent of any manual sessions you have open. Each task gets a fixed delay of up to 10 minutes after the scheduled time to stagger API traffic. The delay is deterministic: the same task always starts at the same offset. When a task fires, you get a desktop notification and a new session appears under a **Scheduled** section in the sidebar. Open it to see what Claude did, review changes, or respond to permission prompts. The session works like any other: Claude can edit files, run commands, create commits, and open pull requests. Tasks only run while the desktop app is running and your computer is awake. If your computer sleeps through a scheduled time, the run is skipped. To prevent idle-sleep, enable **Keep computer awake** in Settings under **Desktop app → General** . Closing the laptop lid still puts it to sleep. For tasks that need to run even when your computer is off, use a [remote task](/docs/en/web-scheduled-tasks) instead.

### Missed runs

When the app starts or your computer wakes, Desktop checks whether each task missed any runs in the last seven days. If it did, Desktop starts exactly one catch-up run for the most recently missed time and discards anything older. A daily task that missed six days runs once on wake. Desktop shows a notification when a catch-up run starts. Keep this in mind when writing prompts. A task scheduled for 9am might run at 11pm if your computer was asleep all day. If timing matters, add guardrails to the prompt itself, for example: "Only review today's commits. If it's after 5pm, skip the review and just post a summary of what was missed."

### Permissions for scheduled tasks

Each task has its own permission mode, which you set when creating or editing the task. Allow rules from `~/.claude/settings.json` also apply to scheduled task sessions. If a task runs in Ask mode and needs to run a tool it doesn't have permission for, the run stalls until you approve it. The session stays open in the sidebar so you can answer later. To avoid stalls, click **Run now** after creating a task, watch for permission prompts, and select "always allow" for each one. Future runs of that task auto-approve the same tools without prompting. You can review and revoke these approvals from the task's detail page.

### Manage scheduled tasks

Click a task in the **Schedule** list to open its detail page. From here you can:

- **Run now** : start the task immediately without waiting for the next scheduled time
- **Toggle repeats** : pause or resume scheduled runs without deleting the task
- **Edit** : change the prompt, frequency, folder, or other settings
- **Review history** : see every past run, including ones that were skipped because your computer was asleep
- **Review allowed permissions** : see and revoke saved tool approvals for this task from the **Always allowed** panel
- **Delete** : remove the task and archive all sessions it created

You can also manage tasks by asking Claude in any Desktop session. For example, "pause my dependency-audit task", "delete the standup-prep task", or "show me my scheduled tasks." To edit a task's prompt on disk, open `~/.claude/scheduled-tasks/<task-name>/SKILL.md` (or under [`CLAUDE_CONFIG_DIR`](/docs/en/env-vars) if set). The file uses YAML frontmatter for `name` and `description` , with the prompt as the body. Changes take effect on the next run. Schedule, folder, model, and enabled state are not in this file: change them through the Edit form or ask Claude.

### Related resources

- [Cloud scheduled tasks](/docs/en/web-scheduled-tasks) : schedule tasks that run on Anthropic-managed infrastructure even when your computer is off
- [Run prompts on a schedule](/docs/en/scheduled-tasks) : session-scoped scheduling with `/loop` in the CLI
- [Claude Code GitHub Actions](/docs/en/github-actions) : run Claude on a schedule in CI instead of on your machine
- [Use Claude Code Desktop](/docs/en/desktop) : the full Desktop app guide

Was this page helpful?

Yes

No

[Reference](/docs/en/desktop) [Chrome extension (beta)](/docs/en/chrome)

⌘ I


### Run prompts on a schedule


Use /loop and the cron scheduling tools to run prompts repeatedly, poll for status, or set one-time reminders within a Claude Code session.


Scheduled tasks require Claude Code v2.1.72 or later. Check your version with `claude --version` .

Scheduled tasks let Claude re-run a prompt automatically on an interval. Use them to poll a deployment, babysit a PR, check back on a long-running build, or remind yourself to do something later in the session. To react to events as they happen instead of polling, see [Channels](/docs/en/channels) : your CI can push the failure into the session directly. Tasks are session-scoped: they live in the current Claude Code process and are gone when you exit. For durable scheduling that survives restarts, use [Cloud](/docs/en/web-scheduled-tasks) or [Desktop](/docs/en/desktop-scheduled-tasks) scheduled tasks, or [GitHub Actions](/docs/en/github-actions) .

### Compare scheduling options

Claude Code offers three ways to schedule recurring work:

|                            | [Cloud](/docs/en/web-scheduled-tasks)   | [Desktop](/docs/en/desktop-scheduled-tasks)   | [`/loop`](/docs/en/scheduled-tasks)   |
|----------------------------|-----------------------------------------|-----------------------------------------------|---------------------------------------|
| Runs on                    | Anthropic cloud                         | Your machine                                  | Your machine                          |
| Requires machine on        | No                                      | Yes                                           | Yes                                   |
| Requires open session      | No                                      | No                                            | Yes                                   |
| Persistent across restarts | Yes                                     | Yes                                           | No (session-scoped)                   |
| Access to local files      | No (fresh clone)                        | Yes                                           | Yes                                   |
| MCP servers                | Connectors configured per task          | [Config files](/docs/en/mcp) and connectors   | Inherits from session                 |
| Permission prompts         | No (runs autonomously)                  | Configurable per task                         | Inherits from session                 |
| Customizable schedule      | Via `/schedule` in the CLI              | Yes                                           | Yes                                   |
| Minimum interval           | 1 hour                                  | 1 minute                                      | 1 minute                              |

Use **cloud tasks** for work that should run reliably without your machine. Use **Desktop tasks** when you need access to local files and tools. Use **`/loop`** for quick polling during a session.

### Run a prompt repeatedly with /loop

The `/loop` [bundled skill](/docs/en/commands) is the quickest way to run a prompt on repeat while the session stays open. Both the interval and the prompt are optional, and what you provide determines how the loop behaves.

| What you provide          | Example                     | What happens                                                                                                  |
|---------------------------|-----------------------------|---------------------------------------------------------------------------------------------------------------|
| Interval and prompt       | `/loop 5m check the deploy` | Your prompt runs on a [fixed schedule](#run-on-a-fixed-interval)                                              |
| Prompt only               | `/loop check the deploy`    | Your prompt runs at an [interval Claude chooses](#let-claude-choose-the-interval) each iteration              |
| Interval only, or nothing | `/loop`                     | The [built-in maintenance prompt](#run-the-built-in-maintenance-prompt) runs, or your `loop.md` if one exists |

You can also pass another command as the prompt, for example `/loop 20m /review-pr 1234` , to re-run a packaged workflow each iteration.

#### Run on a fixed interval

When you supply an interval, Claude converts it to a cron expression, schedules the job, and confirms the cadence and job ID.

```
/loop 5m check if the deployment finished and tell me what happened
```

The interval can lead the prompt as a bare token like `30m` , or trail it as a clause like `every 2 hours` . Supported units are `s` for seconds, `m` for minutes, `h` for hours, and `d` for days. Seconds are rounded up to the nearest minute since cron has one-minute granularity. Intervals that don't map to a clean cron step, such as `7m` or `90m` , are rounded to the nearest interval that does and Claude tells you what it picked.

#### Let Claude choose the interval

When you omit the interval, Claude chooses one dynamically instead of running on a fixed cron schedule. After each iteration it picks a delay between one minute and one hour based on what it observed: short waits while a build is finishing or a PR is active, longer waits when nothing is pending. The chosen delay and the reason for it are printed at the end of each iteration. The example below checks CI and review comments, with Claude waiting longer between iterations once the PR goes quiet:

```
/loop check whether CI passed and address any review comments
```

When you ask for a dynamic `/loop` schedule, Claude may use the [Monitor tool](/docs/en/tools-reference#monitor-tool) directly. Monitor runs a background script and streams each output line back, which avoids polling altogether and is often more token-efficient and responsive than re-running a prompt on an interval. A dynamically scheduled loop appears in your [scheduled task list](#manage-scheduled-tasks) like any other task, so you can list or cancel it the same way. The [jitter rules](#jitter) don't apply to it, but the [seven-day expiry](#seven-day-expiry) does: the loop ends automatically seven days after you start it.

On Bedrock, Vertex AI, and Microsoft Foundry, a prompt with no interval runs on a fixed 10-minute schedule instead.

#### Run the built-in maintenance prompt

When you omit the prompt, Claude uses a built-in maintenance prompt instead of one you supply. On each iteration it works through the following, in order:

- continue any unfinished work from the conversation
- tend to the current branch's pull request: review comments, failed CI runs, merge conflicts
- run cleanup passes such as bug hunts or simplification when nothing else is pending

Claude does not start new initiatives outside that scope, and irreversible actions such as pushing or deleting only proceed when they continue something the transcript already authorized.

```
/loop
```

A bare `/loop` runs this prompt at a [dynamically chosen interval](#let-claude-choose-the-interval) . Add an interval, for example `/loop 15m` , to run it on a fixed schedule instead. To replace the built-in prompt with your own default, see [Customize the default prompt with loop.md](#customize-the-default-prompt-with-loop-md) .

On Bedrock, Vertex AI, and Microsoft Foundry, `/loop` with no prompt prints the usage message instead of starting the maintenance loop.

#### Customize the default prompt with loop.md

A `loop.md` file replaces the built-in maintenance prompt with your own instructions. It defines a single default prompt for bare `/loop` , not a list of separate scheduled tasks, and is ignored whenever you supply a prompt on the command line. To schedule additional prompts alongside it, use `/loop <prompt>` or [ask Claude directly](#manage-scheduled-tasks) . Claude looks for the file in two locations and uses the first one it finds.

| Path                | Scope                                                            |
|---------------------|------------------------------------------------------------------|
| `.claude/loop.md`   | Project-level. Takes precedence when both files exist.           |
| `~/.claude/loop.md` | User-level. Applies in any project that does not define its own. |

The file is plain Markdown with no required structure. Write it as if you were typing the `/loop` prompt directly. The following example keeps a release branch healthy:

.claude/loop.md

```
Check the `release/next` PR. If CI is red, pull the failing job log,
diagnose, and push a minimal fix. If new review comments have arrived,
address each one and resolve the thread. If everything is green and
quiet, say so in one line.
```

Edits to `loop.md` take effect on the next iteration, so you can refine the instructions while a loop is running. When no `loop.md` exists in either location, the loop falls back to the built-in maintenance prompt. Keep the file concise: content beyond 25,000 bytes is truncated.

### Set a one-time reminder

For one-shot reminders, describe what you want in natural language instead of using `/loop` . Claude schedules a single-fire task that deletes itself after running.

```
remind me at 3pm to push the release branch
```

```
in 45 minutes, check whether the integration tests passed
```

Claude pins the fire time to a specific minute and hour using a cron expression and confirms when it will fire.

### Manage scheduled tasks

Ask Claude in natural language to list or cancel tasks, or reference the underlying tools directly.

```
what scheduled tasks do I have?
```

```
cancel the deploy check job
```

Under the hood, Claude uses these tools:

| Tool         | Purpose                                                                                                         |
|--------------|-----------------------------------------------------------------------------------------------------------------|
| `CronCreate` | Schedule a new task. Accepts a 5-field cron expression, the prompt to run, and whether it recurs or fires once. |
| `CronList`   | List all scheduled tasks with their IDs, schedules, and prompts.                                                |
| `CronDelete` | Cancel a task by ID.                                                                                            |

Each scheduled task has an 8-character ID you can pass to `CronDelete` . A session can hold up to 50 scheduled tasks at once.

### How scheduled tasks run

The scheduler checks every second for due tasks and enqueues them at low priority. A scheduled prompt fires between your turns, not while Claude is mid-response. If Claude is busy when a task comes due, the prompt waits until the current turn ends. All times are interpreted in your local timezone. A cron expression like `0 9 * * *` means 9am wherever you're running Claude Code, not UTC.

#### Jitter

To avoid every session hitting the API at the same wall-clock moment, the scheduler adds a small deterministic offset to fire times:

- Recurring tasks fire up to 10% of their period late, capped at 15 minutes. An hourly job might fire anywhere from `:00` to `:06` .
- One-shot tasks scheduled for the top or bottom of the hour fire up to 90 seconds early.

The offset is derived from the task ID, so the same task always gets the same offset. If exact timing matters, pick a minute that is not `:00` or `:30` , for example `3 9 * * *` instead of `0 9 * * *` , and the one-shot jitter will not apply.

#### Seven-day expiry

Recurring tasks automatically expire 7 days after creation. The task fires one final time, then deletes itself. This bounds how long a forgotten loop can run. If you need a recurring task to last longer, cancel and recreate it before it expires, or use [Cloud scheduled tasks](/docs/en/web-scheduled-tasks) or [Desktop scheduled tasks](/docs/en/desktop-scheduled-tasks) for durable scheduling.

### Cron expression reference

`CronCreate` accepts standard 5-field cron expressions: `minute hour day-of-month month day-of-week` . All fields support wildcards ( `*` ), single values ( `5` ), steps ( `*/15` ), ranges ( `1-5` ), and comma-separated lists ( `1,15,30` ).

| Example        | Meaning                      |
|----------------|------------------------------|
| `*/5 * * * *`  | Every 5 minutes              |
| `0 * * * *`    | Every hour on the hour       |
| `7 * * * *`    | Every hour at 7 minutes past |
| `0 9 * * *`    | Every day at 9am local       |
| `0 9 * * 1-5`  | Weekdays at 9am local        |
| `30 14 15 3 *` | March 15 at 2:30pm local     |

Day-of-week uses `0` or `7` for Sunday through `6` for Saturday. Extended syntax like `L` , `W` , `?` , and name aliases such as `MON` or `JAN` is not supported. When both day-of-month and day-of-week are constrained, a date matches if either field matches. This follows standard vixie-cron semantics.

### Disable scheduled tasks

Set `CLAUDE_CODE_DISABLE_CRON=1` in your environment to disable the scheduler entirely. The cron tools and `/loop` become unavailable, and any already-scheduled tasks stop firing. See [Environment variables](/docs/en/env-vars) for the full list of disable flags.

### Limitations

Session-scoped scheduling has inherent constraints:

- Tasks only fire while Claude Code is running and idle. Closing the terminal or letting the session exit cancels everything.
- No catch-up for missed fires. If a task's scheduled time passes while Claude is busy on a long-running request, it fires once when Claude becomes idle, not once per missed interval.
- No persistence across restarts. Restarting Claude Code clears all session-scoped tasks.

For cron-driven automation that needs to run unattended:

- [Cloud scheduled tasks](/docs/en/web-scheduled-tasks) : run on Anthropic-managed infrastructure
- [GitHub Actions](/docs/en/github-actions) : use a `schedule` trigger in CI
- [Desktop scheduled tasks](/docs/en/desktop-scheduled-tasks) : run locally on your machine

Was this page helpful?

Yes

No

[Push external events to Claude](/docs/en/channels) [Programmatic usage](/docs/en/headless)

⌘ I


### Push events into a running session with channels


Use channels to push messages, alerts, and webhooks into your Claude Code session from an MCP server. Forward CI results, chat messages, and monitoring events so Claude can react while you're away.


Channels are in [research preview](#research-preview) and require Claude Code v2.1.80 or later. They require claude.ai login. Console and API key authentication is not supported. Team and Enterprise organizations must [explicitly enable them](#enterprise-controls) .

A channel is an MCP server that pushes events into your running Claude Code session, so Claude can react to things that happen while you're not at the terminal. Channels can be two-way: Claude reads the event and replies back through the same channel, like a chat bridge. Events only arrive while the session is open, so for an always-on setup you run Claude in a background process or persistent terminal. Unlike integrations that spawn a fresh cloud session or wait to be polled, the event arrives in the session you already have open: see [how channels compare](#how-channels-compare) . You install a channel as a plugin and configure it with your own credentials. Telegram, Discord, and iMessage are included in the research preview. When Claude replies through a channel, you see the inbound message in your terminal but not the reply text. The terminal shows the tool call and a confirmation (like "sent"), and the actual reply appears on the other platform. This page covers:

- [Supported channels](#supported-channels) : Telegram, Discord, and iMessage setup
- [Install and run a channel](#quickstart) with fakechat, a localhost demo
- [Who can push messages](#security) : sender allowlists and how you pair
- [Enable channels for your organization](#enterprise-controls) on Team and Enterprise
- [How channels compare](#how-channels-compare) to web sessions, Slack, MCP, and Remote Control

To build your own channel, see the [Channels reference](/docs/en/channels-reference) .

### Supported channels

Each supported channel is a plugin that requires [Bun](https://bun.sh/) . For a hands-on demo of the plugin flow before connecting a real platform, try the [fakechat quickstart](#quickstart) .

- Telegram
- Discord
- iMessage

View the full [Telegram plugin source](https://github.com/anthropics/claude-plugins-official/tree/main/external_plugins/telegram) .

1

Create a Telegram bot

Open [BotFather](https://t.me/BotFather) in Telegram and send `/newbot` . Give it a display name and a unique username ending in `bot` . Copy the token BotFather returns.

2

Install the plugin

In Claude Code, run:

```
/plugin install telegram@claude-plugins-official
```

If Claude Code reports that the plugin is not found in any marketplace, your marketplace is either missing or outdated. Run `/plugin marketplace update claude-plugins-official` to refresh it, or `/plugin marketplace add anthropics/claude-plugins-official` if you haven't added it before. Then retry the install. After installing, run `/reload-plugins` to activate the plugin's configure command.

3

Configure your token

Run the configure command with the token from BotFather:

```
/telegram:configure <token>
```

This saves it to `~/.claude/channels/telegram/.env` . You can also set `TELEGRAM_BOT_TOKEN` in your shell environment before launching Claude Code.

4

Restart with channels enabled

Exit Claude Code and restart with the channel flag. This starts the Telegram plugin, which begins polling for messages from your bot:

```
claude --channels plugin:telegram@claude-plugins-official
```

5

Pair your account

Open Telegram and send any message to your bot. The bot replies with a pairing code.

If your bot doesn't respond, make sure Claude Code is running with `--channels` from the previous step. The bot can only reply while the channel is active.

Back in Claude Code, run:

```
/telegram:access pair <code>
```

Then lock down access so only your account can send messages:

```
/telegram:access policy allowlist
```

View the full [Discord plugin source](https://github.com/anthropics/claude-plugins-official/tree/main/external_plugins/discord) .

1

Create a Discord bot

Go to the [Discord Developer Portal](https://discord.com/developers/applications) , click **New Application** , and name it. In the **Bot** section, create a username, then click **Reset Token** and copy the token.

2

Enable Message Content Intent

In your bot's settings, scroll to **Privileged Gateway Intents** and enable **Message Content Intent** .

3

Invite the bot to your server

Go to **OAuth2 > URL Generator** . Select the `bot` scope and enable these permissions:

- View Channels
- Send Messages
- Send Messages in Threads
- Read Message History
- Attach Files
- Add Reactions

Open the generated URL to add the bot to your server.

4

Install the plugin

In Claude Code, run:

```
/plugin install discord@claude-plugins-official
```

If Claude Code reports that the plugin is not found in any marketplace, your marketplace is either missing or outdated. Run `/plugin marketplace update claude-plugins-official` to refresh it, or `/plugin marketplace add anthropics/claude-plugins-official` if you haven't added it before. Then retry the install. After installing, run `/reload-plugins` to activate the plugin's configure command.

5

Configure your token

Run the configure command with the bot token you copied:

```
/discord:configure <token>
```

This saves it to `~/.claude/channels/discord/.env` . You can also set `DISCORD_BOT_TOKEN` in your shell environment before launching Claude Code.

6

Restart with channels enabled

Exit Claude Code and restart with the channel flag. This connects the Discord plugin so your bot can receive and respond to messages:

```
claude --channels plugin:discord@claude-plugins-official
```

7

Pair your account

DM your bot on Discord. The bot replies with a pairing code.

If your bot doesn't respond, make sure Claude Code is running with `--channels` from the previous step. The bot can only reply while the channel is active.

Back in Claude Code, run:

```
/discord:access pair <code>
```

Then lock down access so only your account can send messages:

```
/discord:access policy allowlist
```

View the full [iMessage plugin source](https://github.com/anthropics/claude-plugins-official/tree/main/external_plugins/imessage) . The iMessage channel reads your Messages database directly and sends replies through AppleScript. It requires macOS and needs no bot token or external service.

1

Grant Full Disk Access

The Messages database at `~/Library/Messages/chat.db` is protected by macOS. The first time the server reads it, macOS prompts for access: click **Allow** . The prompt names whichever app launched Bun, such as Terminal, iTerm, or your IDE. If the prompt doesn't appear or you clicked Don't Allow, grant access manually under **System Settings > Privacy & Security > Full Disk Access** and add your terminal. Without this, the server exits immediately with `authorization denied` .

2

Install the plugin

In Claude Code, run:

```
/plugin install imessage@claude-plugins-official
```

If Claude Code reports that the plugin is not found in any marketplace, your marketplace is either missing or outdated. Run `/plugin marketplace update claude-plugins-official` to refresh it, or `/plugin marketplace add anthropics/claude-plugins-official` if you haven't added it before. Then retry the install.

3

Restart with channels enabled

Exit Claude Code and restart with the channel flag:

```
claude --channels plugin:imessage@claude-plugins-official
```

4

Text yourself

Open Messages on any device signed into your Apple ID and send a message to yourself. It reaches Claude immediately: self-chat bypasses access control with no setup.

The first reply Claude sends triggers a macOS Automation prompt asking if your terminal can control Messages. Click **OK** .

5

Allow other senders

By default, only your own messages pass through. To let another contact reach Claude, add their handle:

```
/imessage:access allow +15551234567
```

Handles are phone numbers in `+country` format or Apple ID emails like [`[email protected]`](/cdn-cgi/l/email-protection) .

You can also [build your own channel](/docs/en/channels-reference) for systems that don't have a plugin yet.

### Quickstart

Fakechat is an officially supported demo channel that runs a chat UI on localhost, with nothing to authenticate and no external service to configure. Once you install and enable fakechat, you can type in the browser and the message arrives in your Claude Code session. Claude replies, and the reply shows up back in the browser. After you've tested the fakechat interface, try out [Telegram](https://github.com/anthropics/claude-plugins-official/tree/main/external_plugins/telegram) , [Discord](https://github.com/anthropics/claude-plugins-official/tree/main/external_plugins/discord) , or [iMessage](https://github.com/anthropics/claude-plugins-official/tree/main/external_plugins/imessage) . To try the fakechat demo, you'll need:

- Claude Code [installed and authenticated](/docs/en/quickstart#step-1-install-claude-code) with a claude.ai account
- [Bun](https://bun.sh/) installed. The pre-built channel plugins are Bun scripts. Check with `bun --version` ; if that fails, [install Bun](https://bun.sh/docs/installation) .
- **Team/Enterprise users** : your organization admin must [enable channels](#enterprise-controls) in managed settings

1

Install the fakechat channel plugin

Start a Claude Code session and run the install command:

```
/plugin install fakechat@claude-plugins-official
```

If Claude Code reports that the plugin is not found in any marketplace, your marketplace is either missing or outdated. Run `/plugin marketplace update claude-plugins-official` to refresh it, or `/plugin marketplace add anthropics/claude-plugins-official` if you haven't added it before. Then retry the install.

2

Restart with the channel enabled

Exit Claude Code, then restart with `--channels` and pass the fakechat plugin you installed:

```
claude --channels plugin:fakechat@claude-plugins-official
```

The fakechat server starts automatically.

You can pass several plugins to `--channels` , space-separated.

3

Push a message in

Open the fakechat UI at [http://localhost:8787](http://localhost:8787/) and type a message:

```
hey, what's in my working directory?
```

The message arrives in your Claude Code session as a `<channel source="fakechat">` event. Claude reads it, does the work, and calls fakechat's `reply` tool. The answer shows up in the chat UI.

If Claude hits a permission prompt while you're away from the terminal, the session pauses until you respond. Channel servers that declare the [permission relay capability](/docs/en/channels-reference#relay-permission-prompts) can forward these prompts to you so you can approve or deny remotely. For unattended use, [`--dangerously-skip-permissions`](/docs/en/permission-modes#skip-all-checks-with-bypasspermissions-mode) bypasses prompts entirely, but only use it in environments you trust.

### Security

Every approved channel plugin maintains a sender allowlist: only IDs you've added can push messages, and everyone else is silently dropped. Telegram and Discord bootstrap the list by pairing:

1. Find your bot in Telegram or Discord and send it any message
2. The bot replies with a pairing code
3. In your Claude Code session, approve the code when prompted
4. Your sender ID is added to the allowlist

iMessage works differently: texting yourself bypasses the gate automatically, and you add other contacts by handle with `/imessage:access allow` . On top of that, you control which servers are enabled each session with `--channels` , and on Team and Enterprise plans your organization controls availability with [`channelsEnabled`](#enterprise-controls) . Being in `.mcp.json` isn't enough to push messages: a server also has to be named in `--channels` . The allowlist also gates [permission relay](/docs/en/channels-reference#relay-permission-prompts) if the channel declares it. Anyone who can reply through the channel can approve or deny tool use in your session, so only allowlist senders you trust with that authority.

### Enterprise controls

On Team and Enterprise plans, channels are off by default. Admins control availability through two [managed settings](/docs/en/settings) that users cannot override:

| Setting                 | Purpose                                                                                                                                                                                                                                                     | When not configured            |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|
| `channelsEnabled`       | Master switch. Must be `true` for any channel to deliver messages. Set via the [claude.ai Admin console](https://claude.ai/admin-settings/claude-code) toggle or directly in managed settings. Blocks all channels including the development flag when off. | Channels blocked               |
| `allowedChannelPlugins` | Which plugins can register once channels are enabled. Replaces the Anthropic-maintained list when set. Only applies when `channelsEnabled` is `true` .                                                                                                      | Anthropic default list applies |

Pro and Max users without an organization skip these checks entirely: channels are available and users opt in per session with `--channels` .

#### Enable channels for your organization

Admins can enable channels from [**claude.ai → Admin settings → Claude Code → Channels**](https://claude.ai/admin-settings/claude-code) , or by setting `channelsEnabled` to `true` in managed settings. Once enabled, users in your organization can use `--channels` to opt channel servers into individual sessions. If the setting is disabled or unset, the MCP server still connects and its tools work, but channel messages won't arrive. A startup warning tells the user to have an admin enable the setting.

#### Restrict which channel plugins can run

By default, any plugin on the Anthropic-maintained allowlist can register as a channel. Admins on Team and Enterprise plans can replace that allowlist with their own by setting `allowedChannelPlugins` in managed settings. Use this to restrict which official plugins are allowed, approve channels from your own internal marketplace, or both. Each entry names a plugin and the marketplace it comes from:

```
{
"channelsEnabled" : true ,
"allowedChannelPlugins" : [
{ "marketplace" : "claude-plugins-official" , "plugin" : "telegram" },
{ "marketplace" : "claude-plugins-official" , "plugin" : "discord" },
{ "marketplace" : "acme-corp-plugins" , "plugin" : "internal-alerts" }
]
}
```

When `allowedChannelPlugins` is set, it replaces the Anthropic allowlist entirely: only the listed plugins can register. Leave it unset to fall back to the default Anthropic allowlist. An empty array blocks all channel plugins from the allowlist, but `--dangerously-load-development-channels` can still bypass it for local testing. To block channels entirely including the development flag, leave `channelsEnabled` unset instead. This setting requires `channelsEnabled: true` . If a user passes a plugin to `--channels` that isn't on your list, Claude Code starts normally but the channel doesn't register, and the startup notice explains that the plugin isn't on the organization's approved list.

### Research preview

Channels are a research preview feature. Availability is rolling out gradually, and the `--channels` flag syntax and protocol contract may change based on feedback. During the preview, `--channels` only accepts plugins from an Anthropic-maintained allowlist, or from your organization's allowlist if an admin has set [`allowedChannelPlugins`](#restrict-which-channel-plugins-can-run) . The channel plugins in [claude-plugins-official](https://github.com/anthropics/claude-plugins-official/tree/main/external_plugins) are the default approved set. If you pass something that isn't on the effective allowlist, Claude Code starts normally but the channel doesn't register, and the startup notice tells you why. To test a channel you're building, use `--dangerously-load-development-channels` . See [Test during the research preview](/docs/en/channels-reference#test-during-the-research-preview) for information about testing custom channels that you build. Report issues or feedback on the [Claude Code GitHub repository](https://github.com/anthropics/claude-code/issues) .

### How channels compare

Several Claude Code features connect to systems outside the terminal, each suited to a different kind of work:

| Feature                                                   | What it does                                                          | Good for                                                  |
|-----------------------------------------------------------|-----------------------------------------------------------------------|-----------------------------------------------------------|
| [Claude Code on the web](/docs/en/claude-code-on-the-web) | Runs tasks in a fresh cloud sandbox, cloned from GitHub               | Delegating self-contained async work you check on later   |
| [Claude in Slack](/docs/en/slack)                         | Spawns a web session from an `@Claude` mention in a channel or thread | Starting tasks directly from team conversation context    |
| Standard [MCP server](/docs/en/mcp)                       | Claude queries it during a task; nothing is pushed to the session     | Giving Claude on-demand access to read or query a system  |
| [Remote Control](/docs/en/remote-control)                 | You drive your local session from claude.ai or the Claude mobile app  | Steering an in-progress session while away from your desk |

Channels fill the gap in that list by pushing events from non-Claude sources into your already-running local session.

- **Chat bridge** : ask Claude something from your phone via Telegram, Discord, or iMessage, and the answer comes back in the same chat while the work runs on your machine against your real files.
- [**Webhook receiver**](/docs/en/channels-reference#example-build-a-webhook-receiver) : a webhook from CI, your error tracker, a deploy pipeline, or other external service arrives where Claude already has your files open and remembers what you were debugging.

### Next steps

Once you have a channel running, explore these related features:

- [Build your own channel](/docs/en/channels-reference) for systems that don't have plugins yet
- [Remote Control](/docs/en/remote-control) to drive a local session from your phone instead of forwarding events into it
- [Scheduled tasks](/docs/en/scheduled-tasks) to poll on a timer instead of reacting to pushed events

Was this page helpful?

Yes

No

[Automate with hooks](/docs/en/hooks-guide) [Run prompts on a schedule](/docs/en/scheduled-tasks)

⌘ I


---

# Integrations


### Enterprise deployment overview


Learn how Claude Code can integrate with various third-party services and infrastructure to meet enterprise deployment requirements.


Organizations can deploy Claude Code through Anthropic directly or through a cloud provider. This page helps you choose the right configuration.

### Compare deployment options

For most organizations, Claude for Teams or Claude for Enterprise provides the best experience. Team members get access to both Claude Code and Claude on the web with a single subscription, centralized billing, and no infrastructure setup required. **Claude for Teams** is self-service and includes collaboration features, admin tools, and billing management. Best for smaller teams that need to get started quickly. **Claude for Enterprise** adds SSO and domain capture, role-based permissions, compliance API access, and managed policy settings for deploying organization-wide Claude Code configurations. Best for larger organizations with security and compliance requirements. Learn more about [Team plans](https://support.claude.com/en/articles/9266767-what-is-the-team-plan) and [Enterprise plans](https://support.claude.com/en/articles/9797531-what-is-the-enterprise-plan) . If your organization has specific infrastructure requirements, compare the options below:

| Feature                | Claude for Teams/Enterprise                                                                                                                                                                     | Anthropic Console                                                    | Amazon Bedrock                                                                                   | Google Vertex AI                                                                              | Microsoft Foundry                                                                                             |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| Best for               | Most organizations (recommended)                                                                                                                                                                | Individual developers                                                | AWS-native deployments                                                                           | GCP-native deployments                                                                        | Azure-native deployments                                                                                      |
| Billing                | **Teams:** $150/seat (Premium) with PAYG available  **Enterprise:** [Contact Sales](https://claude.com/contact-sales?utm_source=claude_code&utm_medium=docs&utm_content=third_party_enterprise) | PAYG                                                                 | PAYG through AWS                                                                                 | PAYG through GCP                                                                              | PAYG through Azure                                                                                            |
| Regions                | Supported [countries](https://www.anthropic.com/supported-countries)                                                                                                                            | Supported [countries](https://www.anthropic.com/supported-countries) | Multiple AWS [regions](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html) | Multiple GCP [regions](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations) | Multiple Azure [regions](https://azure.microsoft.com/en-us/explore/global-infrastructure/products-by-region/) |
| Prompt caching         | Enabled by default                                                                                                                                                                              | Enabled by default                                                   | Enabled by default                                                                               | Enabled by default                                                                            | Enabled by default                                                                                            |
| Authentication         | Claude.ai SSO or email                                                                                                                                                                          | API key                                                              | API key or AWS credentials                                                                       | GCP credentials                                                                               | API key or Microsoft Entra ID                                                                                 |
| Cost tracking          | Usage dashboard                                                                                                                                                                                 | Usage dashboard                                                      | AWS Cost Explorer                                                                                | GCP Billing                                                                                   | Azure Cost Management                                                                                         |
| Includes Claude on web | Yes                                                                                                                                                                                             | No                                                                   | No                                                                                               | No                                                                                            | No                                                                                                            |
| Enterprise features    | Team management, SSO, usage monitoring                                                                                                                                                          | None                                                                 | IAM policies, CloudTrail                                                                         | IAM roles, Cloud Audit Logs                                                                   | RBAC policies, Azure Monitor                                                                                  |

Select a deployment option to view setup instructions:

- [Claude for Teams or Enterprise](/docs/en/authentication#claude-for-teams-or-enterprise)
- [Anthropic Console](/docs/en/authentication#claude-console-authentication)
- [Amazon Bedrock](/docs/en/amazon-bedrock)
- [Google Vertex AI](/docs/en/google-vertex-ai)
- [Microsoft Foundry](/docs/en/microsoft-foundry)

### Configure proxies and gateways

Most organizations can use a cloud provider directly without additional configuration. However, you may need to configure a corporate proxy or LLM gateway if your organization has specific network or management requirements. These are different configurations that can be used together:

- **Corporate proxy** : Routes traffic through an HTTP/HTTPS proxy. Use this if your organization requires all outbound traffic to pass through a proxy server for security monitoring, compliance, or network policy enforcement. Configure with the `HTTPS_PROXY` or `HTTP_PROXY` environment variables. Learn more in [Enterprise network configuration](/docs/en/network-config) .
- **LLM Gateway** : A service that sits between Claude Code and the cloud provider to handle authentication and routing. Use this if you need centralized usage tracking across teams, custom rate limiting or budgets, or centralized authentication management. Configure with the `ANTHROPIC_BASE_URL` , `ANTHROPIC_BEDROCK_BASE_URL` , or `ANTHROPIC_VERTEX_BASE_URL` environment variables. Learn more in [LLM gateway configuration](/docs/en/llm-gateway) .

The following examples show the environment variables to set in your shell or shell profile ( `.bashrc` , `.zshrc` ). See [Settings](/docs/en/settings) for other configuration methods.

#### Amazon Bedrock

- Corporate proxy
- LLM Gateway

Route Bedrock traffic through your corporate proxy by setting the following [environment variables](/docs/en/env-vars) :

```
### Enable Bedrock
export CLAUDE_CODE_USE_BEDROCK = 1
export AWS_REGION = us-east-1

### Configure corporate proxy
export HTTPS_PROXY = 'https://proxy.example.com:8080'
```

Route Bedrock traffic through your LLM gateway by setting the following [environment variables](/docs/en/env-vars) :

```
### Enable Bedrock
export CLAUDE_CODE_USE_BEDROCK = 1

### Configure LLM gateway
export ANTHROPIC_BEDROCK_BASE_URL = 'https://your-llm-gateway.com/bedrock'
export CLAUDE_CODE_SKIP_BEDROCK_AUTH = 1 # If gateway handles AWS auth
```

#### Microsoft Foundry

- Corporate proxy
- LLM Gateway

Route Foundry traffic through your corporate proxy by setting the following [environment variables](/docs/en/env-vars) :

```
### Enable Microsoft Foundry
export CLAUDE_CODE_USE_FOUNDRY = 1
export ANTHROPIC_FOUNDRY_RESOURCE = your-resource
export ANTHROPIC_FOUNDRY_API_KEY = your-api-key # Or omit for Entra ID auth

### Configure corporate proxy
export HTTPS_PROXY = 'https://proxy.example.com:8080'
```

Route Foundry traffic through your LLM gateway by setting the following [environment variables](/docs/en/env-vars) :

```
### Enable Microsoft Foundry
export CLAUDE_CODE_USE_FOUNDRY = 1

### Configure LLM gateway
export ANTHROPIC_FOUNDRY_BASE_URL = 'https://your-llm-gateway.com'
export CLAUDE_CODE_SKIP_FOUNDRY_AUTH = 1 # If gateway handles Azure auth
```

#### Google Vertex AI

- Corporate proxy
- LLM Gateway

Route Vertex AI traffic through your corporate proxy by setting the following [environment variables](/docs/en/env-vars) :

```
### Enable Vertex
export CLAUDE_CODE_USE_VERTEX = 1
export CLOUD_ML_REGION = us-east5
export ANTHROPIC_VERTEX_PROJECT_ID = your-project-id

### Configure corporate proxy
export HTTPS_PROXY = 'https://proxy.example.com:8080'
```

Route Vertex AI traffic through your LLM gateway by setting the following [environment variables](/docs/en/env-vars) :

```
### Enable Vertex
export CLAUDE_CODE_USE_VERTEX = 1

### Configure LLM gateway
export ANTHROPIC_VERTEX_BASE_URL = 'https://your-llm-gateway.com/vertex'
export CLAUDE_CODE_SKIP_VERTEX_AUTH = 1 # If gateway handles GCP auth
```

Use `/status` in Claude Code to verify your proxy and gateway configuration is applied correctly.

### Best practices for organizations

#### Invest in documentation and memory

We strongly recommend investing in documentation so that Claude Code understands your codebase. Organizations can deploy CLAUDE.md files at multiple levels:

- **Organization-wide** : Deploy to system directories like `/Library/Application Support/ClaudeCode/CLAUDE.md` (macOS) for company-wide standards
- **Repository-level** : Create `CLAUDE.md` files in repository roots containing project architecture, build commands, and contribution guidelines. Check these into source control so all users benefit

Learn more in [Memory and CLAUDE.md files](/docs/en/memory) .

#### Simplify deployment

If you have a custom development environment, we find that creating a "one click" way to install Claude Code is key to growing adoption across an organization.

#### Start with guided usage

Encourage new users to try Claude Code for codebase Q&A, or on smaller bug fixes or feature requests. Ask Claude Code to make a plan. Check Claude's suggestions and give feedback if it's off-track. Over time, as users understand this new paradigm better, then they'll be more effective at letting Claude Code run more agentically.

#### Pin model versions for cloud providers

If you deploy through [Bedrock](/docs/en/amazon-bedrock) , [Vertex AI](/docs/en/google-vertex-ai) , or [Foundry](/docs/en/microsoft-foundry) , pin specific model versions using `ANTHROPIC_DEFAULT_OPUS_MODEL` , `ANTHROPIC_DEFAULT_SONNET_MODEL` , and `ANTHROPIC_DEFAULT_HAIKU_MODEL` . Without pinning, model aliases resolve to the latest version, which may not yet be enabled in your account when Anthropic releases an update. Pinning lets you control when your users move to a new model. See [Model configuration](/docs/en/model-config#pin-models-for-third-party-deployments) for what each provider does when the latest version is unavailable.

#### Configure security policies

Security teams can configure managed permissions for what Claude Code is and is not allowed to do, which cannot be overwritten by local configuration. [Learn more](/docs/en/security) .

#### Leverage MCP for integrations

MCP is a great way to give Claude Code more information, such as connecting to ticket management systems or error logs. We recommend that one central team configures MCP servers and checks a `.mcp.json` configuration into the codebase so that all users benefit. [Learn more](/docs/en/mcp) . At Anthropic, we trust Claude Code to power development across every Anthropic codebase. We hope you enjoy using Claude Code as much as we do.

### Next steps

Once you've chosen a deployment option and configured access for your team:

1. **Roll out to your team** : Share installation instructions and have team members [install Claude Code](/docs/en/setup) and authenticate with their credentials.
2. **Set up shared configuration** : Create a [CLAUDE.md file](/docs/en/memory) in your repositories to help Claude Code understand your codebase and coding standards.
3. **Configure permissions** : Review [security settings](/docs/en/security) to define what Claude Code can and cannot do in your environment.

Was this page helpful?

Yes

No

[Amazon Bedrock](/docs/en/amazon-bedrock)

⌘ I


### Use Claude Code with Chrome (beta)


Connect Claude Code to your Chrome browser to test web apps, debug with console logs, automate form filling, and extract data from web pages.


Claude Code integrates with the Claude in Chrome browser extension to give you browser automation capabilities from the CLI or the [VS Code extension](/docs/en/vs-code#automate-browser-tasks-with-chrome) . Build your code, then test and debug in the browser without switching contexts. Claude opens new tabs for browser tasks and shares your browser's login state, so it can access any site you're already signed into. Browser actions run in a visible Chrome window in real time. When Claude encounters a login page or CAPTCHA, it pauses and asks you to handle it manually.

Chrome integration is in beta and currently works with Google Chrome and Microsoft Edge. It is not yet supported on Brave, Arc, or other Chromium-based browsers. WSL (Windows Subsystem for Linux) is also not supported.

### Capabilities

With Chrome connected, you can chain browser actions with coding tasks in a single workflow:

- **Live debugging** : read console errors and DOM state directly, then fix the code that caused them
- **Design verification** : build a UI from a Figma mock, then open it in the browser to verify it matches
- **Web app testing** : test form validation, check for visual regressions, or verify user flows
- **Authenticated web apps** : interact with Google Docs, Gmail, Notion, or any app you're logged into without API connectors
- **Data extraction** : pull structured information from web pages and save it locally
- **Task automation** : automate repetitive browser tasks like data entry, form filling, or multi-site workflows
- **Session recording** : record browser interactions as GIFs to document or share what happened

### Prerequisites

Before using Claude Code with Chrome, you need:

- [Google Chrome](https://www.google.com/chrome/) or [Microsoft Edge](https://www.microsoft.com/edge) browser
- [Claude in Chrome extension](https://chromewebstore.google.com/detail/claude/fcoeoabgfenejglbffodgkkbkcdhcgfn) version 1.0.36 or higher, available in the Chrome Web Store for both browsers
- [Claude Code](/docs/en/quickstart#step-1-install-claude-code) version 2.0.73 or higher
- A direct Anthropic plan (Pro, Max, Team, or Enterprise)

Chrome integration is not available through third-party providers like Amazon Bedrock, Google Cloud Vertex AI, or Microsoft Foundry. If you access Claude exclusively through a third-party provider, you need a separate claude.ai account to use this feature.

### Get started in the CLI

1

Launch Claude Code with Chrome

Start Claude Code with the `--chrome` flag:

```
claude --chrome
```

You can also enable Chrome from within an existing session by running `/chrome` .

2

Ask Claude to use the browser

This example navigates to a page, interacts with it, and reports what it finds, all from your terminal or editor:

```
Go to code.claude.com/docs, click on the search box,
type "hooks", and tell me what results appear
```

Run `/chrome` at any time to check the connection status, manage permissions, or reconnect the extension. For VS Code, see [browser automation in VS Code](/docs/en/vs-code#automate-browser-tasks-with-chrome) .

#### Enable Chrome by default

To avoid passing `--chrome` each session, run `/chrome` and select "Enabled by default". In the [VS Code extension](/docs/en/vs-code#automate-browser-tasks-with-chrome) , Chrome is available whenever the Chrome extension is installed. No additional flag is needed.

Enabling Chrome by default in the CLI increases context usage since browser tools are always loaded. If you notice increased context consumption, disable this setting and use `--chrome` only when needed.

#### Manage site permissions

Site-level permissions are inherited from the Chrome extension. Manage permissions in the Chrome extension settings to control which sites Claude can browse, click, and type on.

### Example workflows

These examples show common ways to combine browser actions with coding tasks. Run `/mcp` and select `claude-in-chrome` to see the full list of available browser tools.

#### Test a local web application

When developing a web app, ask Claude to verify your changes work correctly:

```
I just updated the login form validation. Can you open localhost:3000,
try submitting the form with invalid data, and check if the error
messages appear correctly?
```

Claude navigates to your local server, interacts with the form, and reports what it observes.

#### Debug with console logs

Claude can read console output to help diagnose problems. Tell Claude what patterns to look for rather than asking for all console output, since logs can be verbose:

```
Open the dashboard page and check the console for any errors when
the page loads.
```

Claude reads the console messages and can filter for specific patterns or error types.

#### Automate form filling

Speed up repetitive data entry tasks:

```
I have a spreadsheet of customer contacts in contacts.csv. For each row,
go to the CRM at crm.example.com, click "Add Contact", and fill in the
name, email, and phone fields.
```

Claude reads your local file, navigates the web interface, and enters the data for each record.

#### Draft content in Google Docs

Use Claude to write directly in your documents without API setup:

```
Draft a project update based on the recent commits and add it to my
Google Doc at docs.google.com/document/d/abc123
```

Claude opens the document, clicks into the editor, and types the content. This works with any web app you're logged into: Gmail, Notion, Sheets, and more.

#### Extract data from web pages

Pull structured information from websites:

```
Go to the product listings page and extract the name, price, and
availability for each item. Save the results as a CSV file.
```

Claude navigates to the page, reads the content, and compiles the data into a structured format.

#### Run multi-site workflows

Coordinate tasks across multiple websites:

```
Check my calendar for meetings tomorrow, then for each meeting with
an external attendee, look up their company website and add a note
about what they do.
```

Claude works across tabs to gather information and complete the workflow.

#### Record a demo GIF

Create shareable recordings of browser interactions:

```
Record a GIF showing how to complete the checkout flow, from adding
an item to the cart through to the confirmation page.
```

Claude records the interaction sequence and saves it as a GIF file.

### Troubleshooting

#### Extension not detected

If Claude Code shows "Chrome extension not detected":

1. Verify the Chrome extension is installed and enabled in `chrome://extensions`
2. Verify Claude Code is up to date by running `claude --version`
3. Check that Chrome is running
4. Run `/chrome` and select "Reconnect extension" to re-establish the connection
5. If the issue persists, restart both Claude Code and Chrome

The first time you enable Chrome integration, Claude Code installs a native messaging host configuration file. Chrome reads this file on startup, so if the extension isn't detected on your first attempt, restart Chrome to pick up the new configuration. If the connection still fails, verify the host configuration file exists at: For Chrome:

- **macOS** : `~/Library/Application Support/Google/Chrome/NativeMessagingHosts/com.anthropic.claude_code_browser_extension.json`
- **Linux** : `~/.config/google-chrome/NativeMessagingHosts/com.anthropic.claude_code_browser_extension.json`
- **Windows** : check `HKCU\Software\Google\Chrome\NativeMessagingHosts\` in the Windows Registry

For Edge:

- **macOS** : `~/Library/Application Support/Microsoft Edge/NativeMessagingHosts/com.anthropic.claude_code_browser_extension.json`
- **Linux** : `~/.config/microsoft-edge/NativeMessagingHosts/com.anthropic.claude_code_browser_extension.json`
- **Windows** : check `HKCU\Software\Microsoft\Edge\NativeMessagingHosts\` in the Windows Registry

#### Browser not responding

If Claude's browser commands stop working:

1. Check if a modal dialog (alert, confirm, prompt) is blocking the page. JavaScript dialogs block browser events and prevent Claude from receiving commands. Dismiss the dialog manually, then tell Claude to continue.
2. Ask Claude to create a new tab and try again
3. Restart the Chrome extension by disabling and re-enabling it in `chrome://extensions`

#### Connection drops during long sessions

The Chrome extension's service worker can go idle during extended sessions, which breaks the connection. If browser tools stop working after a period of inactivity, run `/chrome` and select "Reconnect extension".

#### Windows-specific issues

On Windows, you may encounter:

- **Named pipe conflicts (EADDRINUSE)** : if another process is using the same named pipe, restart Claude Code. Close any other Claude Code sessions that might be using Chrome.
- **Native messaging host errors** : if the native messaging host crashes on startup, try reinstalling Claude Code to regenerate the host configuration.

#### Common error messages

These are the most frequently encountered errors and how to resolve them:

| Error                                | Cause                                            | Fix                                                             |
|--------------------------------------|--------------------------------------------------|-----------------------------------------------------------------|
| "Browser extension is not connected" | Native messaging host cannot reach the extension | Restart Chrome and Claude Code, then run `/chrome` to reconnect |
| "Extension not detected"             | Chrome extension is not installed or is disabled | Install or enable the extension in `chrome://extensions`        |
| "No tab available"                   | Claude tried to act before a tab was ready       | Ask Claude to create a new tab and retry                        |
| "Receiving end does not exist"       | Extension service worker went idle               | Run `/chrome` and select "Reconnect extension"                  |

### See also

- [Computer use](/docs/en/computer-use) : control native macOS apps when a task can't be done in a browser
- [Use Claude Code in VS Code](/docs/en/vs-code#automate-browser-tasks-with-chrome) : browser automation in the VS Code extension
- [CLI reference](/docs/en/cli-reference) : command-line flags including `--chrome`
- [Common workflows](/docs/en/common-workflows) : more ways to use Claude Code
- [Data and privacy](/docs/en/data-usage) : how Claude Code handles your data
- [Getting started with Claude in Chrome](https://support.claude.com/en/articles/12012173-getting-started-with-claude-in-chrome) : full documentation for the Chrome extension, including shortcuts, scheduling, and permissions

Was this page helpful?

Yes

No

[Scheduled tasks](/docs/en/desktop-scheduled-tasks) [Computer use (preview)](/docs/en/computer-use)

⌘ I


### Claude Code in Slack


Delegate coding tasks directly from your Slack workspace


Claude Code in Slack brings the power of Claude Code directly into your Slack workspace. When you mention `@Claude` with a coding task, Claude automatically detects the intent and creates a Claude Code session on the web, allowing you to delegate development work without leaving your team conversations. This integration is built on the existing Claude for Slack app but adds intelligent routing to Claude Code on the web for coding-related requests.

### Use cases

- **Bug investigation and fixes** : Ask Claude to investigate and fix bugs as soon as they're reported in Slack channels.
- **Quick code reviews and modifications** : Have Claude implement small features or refactor code based on team feedback.
- **Collaborative debugging** : When team discussions provide crucial context (e.g., error reproductions or user reports), Claude can use that information to inform its debugging approach.
- **Parallel task execution** : Kick off coding tasks in Slack while you continue other work, receiving notifications when complete.

### Prerequisites

Before using Claude Code in Slack, ensure you have the following:

| Requirement            | Details                                                                                           |
|------------------------|---------------------------------------------------------------------------------------------------|
| Claude Plan            | Pro, Max, Team, or Enterprise with Claude Code access (premium seats or Chat + Claude Code seats) |
| Claude Code on the web | Access to [Claude Code on the web](/docs/en/claude-code-on-the-web) must be enabled               |
| GitHub Account         | Connected to Claude Code on the web with at least one repository authenticated                    |
| Slack Authentication   | Your Slack account linked to your Claude account via the Claude app                               |

### Setting up Claude Code in Slack

1

Install the Claude App in Slack

A workspace administrator must install the Claude app from the Slack App Marketplace. Visit the [Slack App Marketplace](https://slack.com/marketplace/A08SF47R6P4) and click "Add to Slack" to begin the installation process.

2

Connect your Claude account

After the app is installed, authenticate your individual Claude account:

1. Open the Claude app in Slack by clicking on "Claude" in your Apps section
2. Navigate to the App Home tab
3. Click "Connect" to link your Slack account with your Claude account
4. Complete the authentication flow in your browser

3

Configure Claude Code on the web

Ensure your Claude Code on the web is properly configured:

- Visit [claude.ai/code](https://claude.ai/code) and sign in with the same account you connected to Slack
- Connect your GitHub account if not already connected
- Authenticate at least one repository that you want Claude to work with

4

Choose your routing mode

After connecting your accounts, configure how Claude handles your messages in Slack. Navigate to the Claude App Home in Slack to find the **Routing Mode** setting.

| Mode            | Behavior                                                                                                                                                                                                                                 |
|-----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Code only**   | Claude routes all @mentions to Claude Code sessions. Best for teams using Claude in Slack exclusively for development tasks.                                                                                                             |
| **Code + Chat** | Claude analyzes each message and intelligently routes between Claude Code (for coding tasks) and Claude Chat (for writing, analysis, and general questions). Best for teams who want a single @Claude entry point for all types of work. |

In Code + Chat mode, if Claude routes a message to Chat but you wanted a coding session, you can click "Retry as Code" to create a Claude Code session instead. Similarly, if it's routed to Code but you wanted a Chat session, you can choose that option in that thread.

5

Add Claude to channels

Claude is not automatically added to any channels after installation. To use Claude in a channel, invite it by typing `/invite @Claude` in that channel. Claude can only respond to @mentions in channels where it has been added.

### How it works

#### Automatic detection

When you mention @Claude in a Slack channel or thread, Claude automatically analyzes your message to determine if it's a coding task. If Claude detects coding intent, it will route your request to Claude Code on the web instead of responding as a regular chat assistant. You can also explicitly tell Claude to handle a request as a coding task, even if it doesn't automatically detect it.

Claude Code in Slack only works in channels (public or private). It does not work in direct messages (DMs).

#### Context gathering

**From threads** : When you @mention Claude in a thread, it gathers context from all messages in that thread to understand the full conversation. **From channels** : When mentioned directly in a channel, Claude looks at recent channel messages for relevant context. This context helps Claude understand the problem, select the appropriate repository, and inform its approach to the task.

When @Claude is invoked in Slack, Claude is given access to the conversation context to better understand your request. Claude may follow directions from other messages in the context, so users should make sure to only use Claude in trusted Slack conversations.

#### Session flow

1. **Initiation** : You @mention Claude with a coding request
2. **Detection** : Claude analyzes your message and detects coding intent
3. **Session creation** : A new Claude Code session is created on claude.ai/code
4. **Progress updates** : Claude posts status updates to your Slack thread as work progresses
5. **Completion** : When finished, Claude @mentions you with a summary and action buttons
6. **Review** : Click "View Session" to see the full transcript, or "Create PR" to open a pull request

### User interface elements

#### App Home

The App Home tab shows your connection status and allows you to connect or disconnect your Claude account from Slack.

#### Message actions

- **View Session** : Opens the full Claude Code session in your browser where you can see all work performed, continue the session, or make additional requests.
- **Create PR** : Creates a pull request directly from the session's changes.
- **Retry as Code** : If Claude initially responds as a chat assistant but you wanted a coding session, click this button to retry the request as a Claude Code task.
- **Change Repo** : Allows you to select a different repository if Claude chose incorrectly.

#### Repository selection

Claude automatically selects a repository based on context from your Slack conversation. If multiple repositories could apply, Claude may display a dropdown allowing you to choose the correct one.

### Access and permissions

#### User-level access

| Access Type          | Requirement                                                     |
|----------------------|-----------------------------------------------------------------|
| Claude Code Sessions | Each user runs sessions under their own Claude account          |
| Usage & Rate Limits  | Sessions count against the individual user's plan limits        |
| Repository Access    | Users can only access repositories they've personally connected |
| Session History      | Sessions appear in your Claude Code history on claude.ai/code   |

#### Workspace-level access

Slack workspace administrators control whether the Claude app is available in their workspace:

| Control                      | Description                                                                                                       |
|------------------------------|-------------------------------------------------------------------------------------------------------------------|
| App installation             | Workspace admins decide whether to install the Claude app from the Slack App Marketplace                          |
| Enterprise Grid distribution | For Enterprise Grid organizations, organization admins can control which workspaces have access to the Claude app |
| App removal                  | Removing the app from a workspace immediately revokes access for all users in that workspace                      |

#### Channel-based access control

Claude is not automatically added to any channels after installation. Users must explicitly invite Claude to channels where they want to use it:

- **Invite required** : Type `/invite @Claude` in any channel to add Claude to that channel
- **Channel membership controls access** : Claude can only respond to @mentions in channels where it has been added
- **Access gating through channels** : Admins can control who uses Claude Code by managing which channels Claude is invited to and who has access to those channels
- **Private channel support** : Claude works in both public and private channels, giving teams flexibility in controlling visibility

This channel-based model allows teams to restrict Claude Code usage to specific channels, providing an additional layer of access control beyond workspace-level permissions.

### What's accessible where

**In Slack** : You'll see status updates, completion summaries, and action buttons. The full transcript is preserved and always accessible. **On the web** : The complete Claude Code session with full conversation history, all code changes, file operations, and the ability to continue the session or create pull requests. For Enterprise and Team accounts, sessions created from Claude in Slack are

automatically visible to the organization. See

[Claude Code on the Web sharing](/docs/en/claude-code-on-the-web#share-sessions) for more details.

### Best practices

#### Writing effective requests

- **Be specific** : Include file names, function names, or error messages when relevant.
- **Provide context** : Mention the repository or project if it's not clear from the conversation.
- **Define success** : Explain what "done" looks like-should Claude write tests? Update documentation? Create a PR?
- **Use threads** : Reply in threads when discussing bugs or features so Claude can gather the full context.

#### When to use Slack vs. web

**Use Slack when** : Context already exists in a Slack discussion, you want to kick off a task asynchronously, or you're collaborating with teammates who need visibility. **Use the web directly when** : You need to upload files, want real-time interaction during development, or are working on longer, more complex tasks.

### Troubleshooting

#### Sessions not starting

1. Verify your Claude account is connected in the Claude App Home
2. Check that you have Claude Code on the web access enabled
3. Ensure you have at least one GitHub repository connected to Claude Code

#### Repository not showing

1. Connect the repository in Claude Code on the web at [claude.ai/code](https://claude.ai/code)
2. Verify your GitHub permissions for that repository
3. Try disconnecting and reconnecting your GitHub account

#### Wrong repository selected

1. Click the "Change Repo" button to select a different repository
2. Include the repository name in your request for more accurate selection

#### Authentication errors

1. Disconnect and reconnect your Claude account in the App Home
2. Ensure you're signed into the correct Claude account in your browser
3. Check that your Claude plan includes Claude Code access

#### Session expiration

1. Sessions remain accessible in your Claude Code history on the web
2. You can continue or reference past sessions from [claude.ai/code](https://claude.ai/code)

### Current limitations

- **GitHub only** : Currently supports repositories on GitHub.
- **One PR at a time** : Each session can create one pull request.
- **Rate limits apply** : Sessions use your individual Claude plan's rate limits.
- **Web access required** : Users must have Claude Code on the web access; those without it will only get standard Claude chat responses.

### Related resources

### Claude Code on the web

Learn more about Claude Code on the web

### Claude for Slack

General Claude for Slack documentation

### Slack App Marketplace

Install the Claude app from the Slack Marketplace

### Claude Help Center

Get additional support

Was this page helpful?

Yes

No

[GitLab CI/CD](/docs/en/gitlab-ci-cd)

⌘ I


### Claude Code GitHub Actions


Learn about integrating Claude Code into your development workflow with Claude Code GitHub Actions


Claude Code GitHub Actions brings AI-powered automation to your GitHub workflow. With a simple `@claude` mention in any PR or issue, Claude can analyze your code, create pull requests, implement features, and fix bugs - all while following your project's standards. For automatic reviews posted on every PR without a trigger, see [GitHub Code Review](/docs/en/code-review) .

Claude Code GitHub Actions is built on top of the [Claude Agent SDK](/docs/en/agent-sdk/overview) , which enables programmatic integration of Claude Code into your applications. You can use the SDK to build custom automation workflows beyond GitHub Actions.

**Claude Opus 4.6 is now available.** Claude Code GitHub Actions default to Sonnet. To use Opus 4.6, configure the [model parameter](#breaking-changes-reference) to use `claude-opus-4-6` .

### Why use Claude Code GitHub Actions?

- **Instant PR creation** : Describe what you need, and Claude creates a complete PR with all necessary changes
- **Automated code implementation** : Turn issues into working code with a single command
- **Follows your standards** : Claude respects your `CLAUDE.md` guidelines and existing code patterns
- **Simple setup** : Get started in minutes with our installer and API key
- **Secure by default** : Your code stays on Github's runners

### What can Claude do?

Claude Code provides a powerful GitHub Action that transforms how you work with code:

#### Claude Code Action

This GitHub Action allows you to run Claude Code within your GitHub Actions workflows. You can use this to build any custom workflow on top of Claude Code. [View repository →](https://github.com/anthropics/claude-code-action)

### Setup

### Quick setup

The easiest way to set up this action is through Claude Code in the terminal. Just open claude and run `/install-github-app` . This command will guide you through setting up the GitHub app and required secrets.

- You must be a repository admin to install the GitHub app and add secrets
- The GitHub app will request read & write permissions for Contents, Issues, and Pull requests
- This quickstart method is only available for direct Claude API users. If you're using AWS Bedrock or Google Vertex AI, please see the [Using with AWS Bedrock & Google Vertex AI](#using-with-aws-bedrock-%26-google-vertex-ai) section.

### Manual setup

If the `/install-github-app` command fails or you prefer manual setup, please follow these manual setup instructions:

1. **Install the Claude GitHub app** to your repository: [https://github.com/apps/claude](https://github.com/apps/claude) The Claude GitHub app requires the following repository permissions: For more details on security and permissions, see the [security documentation](https://github.com/anthropics/claude-code-action/blob/main/docs/security.md) .
    - **Contents** : Read & write (to modify repository files)
    - **Issues** : Read & write (to respond to issues)
    - **Pull requests** : Read & write (to create PRs and push changes)
2. **Add ANTHROPIC\_API\_KEY** to your repository secrets ( [Learn how to use secrets in GitHub Actions](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions) )
3. **Copy the workflow file** from [examples/claude.yml](https://github.com/anthropics/claude-code-action/blob/main/examples/claude.yml) into your repository's `.github/workflows/`

After completing either the quickstart or manual setup, test the action by tagging `@claude` in an issue or PR comment.

### Upgrading from Beta

Claude Code GitHub Actions v1.0 introduces breaking changes that require updating your workflow files in order to upgrade to v1.0 from the beta version.

If you're currently using the beta version of Claude Code GitHub Actions, we recommend that you update your workflows to use the GA version. The new version simplifies configuration while adding powerful new features like automatic mode detection.

#### Essential changes

All beta users must make these changes to their workflow files in order to upgrade:

1. **Update the action version** : Change `@beta` to `@v1`
2. **Remove mode configuration** : Delete `mode: "tag"` or `mode: "agent"` (now auto-detected)
3. **Update prompt inputs** : Replace `direct_prompt` with `prompt`
4. **Move CLI options** : Convert `max_turns` , `model` , `custom_instructions` , etc. to `claude_args`

#### Breaking Changes Reference

| Old Beta Input        | New v1.0 Input                        |
|-----------------------|---------------------------------------|
| `mode`                | *(Removed - auto-detected)*           |
| `direct_prompt`       | `prompt`                              |
| `override_prompt`     | `prompt` with GitHub variables        |
| `custom_instructions` | `claude_args: --append-system-prompt` |
| `max_turns`           | `claude_args: --max-turns`            |
| `model`               | `claude_args: --model`                |
| `allowed_tools`       | `claude_args: --allowedTools`         |
| `disallowed_tools`    | `claude_args: --disallowedTools`      |
| `claude_env`          | `settings` JSON format                |

#### Before and After Example

**Beta version:**

```
- uses : anthropics/claude-code-action@beta
with :
mode : "tag"
direct_prompt : "Review this PR for security issues"
anthropic_api_key : ${{ secrets.ANTHROPIC_API_KEY }}
custom_instructions : "Follow our coding standards"
max_turns : "10"
model : "claude-sonnet-4-6"
```

**GA version (v1.0):**

```
- uses : anthropics/claude-code-action@v1
with :
prompt : "Review this PR for security issues"
anthropic_api_key : ${{ secrets.ANTHROPIC_API_KEY }}
claude_args : |
--append-system-prompt "Follow our coding standards"
--max-turns 10
--model claude-sonnet-4-6
```

The action now automatically detects whether to run in interactive mode (responds to `@claude` mentions) or automation mode (runs immediately with a prompt) based on your configuration.

### Example use cases

Claude Code GitHub Actions can help you with a variety of tasks. The [examples directory](https://github.com/anthropics/claude-code-action/tree/main/examples) contains ready-to-use workflows for different scenarios.

#### Basic workflow

```
name : Claude Code
on :
issue_comment :
types : [ created ]
pull_request_review_comment :
types : [ created ]
jobs :
claude :
runs-on : ubuntu-latest
steps :
- uses : anthropics/claude-code-action@v1
with :
anthropic_api_key : ${{ secrets.ANTHROPIC_API_KEY }}
### Responds to @claude mentions in comments
```

#### Using skills

```
name : Code Review
on :
pull_request :
types : [ opened , synchronize ]
jobs :
review :
runs-on : ubuntu-latest
steps :
- uses : anthropics/claude-code-action@v1
with :
anthropic_api_key : ${{ secrets.ANTHROPIC_API_KEY }}
prompt : "Review this pull request for code quality, correctness, and security. Analyze the diff, then post your findings as review comments."
claude_args : "--max-turns 5"
```

#### Custom automation with prompts

```
name : Daily Report
on :
schedule :
- cron : "0 9 * * *"
jobs :
report :
runs-on : ubuntu-latest
steps :
- uses : anthropics/claude-code-action@v1
with :
anthropic_api_key : ${{ secrets.ANTHROPIC_API_KEY }}
prompt : "Generate a summary of yesterday's commits and open issues"
claude_args : "--model opus"
```

#### Common use cases

In issue or PR comments:

```
@claude implement this feature based on the issue description
@claude how should I implement user authentication for this endpoint?
@claude fix the TypeError in the user dashboard component
```

Claude will automatically analyze the context and respond appropriately.

### Best practices

#### CLAUDE.md configuration

Create a `CLAUDE.md` file in your repository root to define code style guidelines, review criteria, project-specific rules, and preferred patterns. This file guides Claude's understanding of your project standards.

#### Security considerations

Never commit API keys directly to your repository.

For comprehensive security guidance including permissions, authentication, and best practices, see the [Claude Code Action security documentation](https://github.com/anthropics/claude-code-action/blob/main/docs/security.md) . Always use GitHub Secrets for API keys:

- Add your API key as a repository secret named `ANTHROPIC_API_KEY`
- Reference it in workflows: `anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}`
- Limit action permissions to only what's necessary
- Review Claude's suggestions before merging

Always use GitHub Secrets (for example, `${{ secrets.ANTHROPIC_API_KEY }}` ) rather than hardcoding API keys directly in your workflow files.

#### Optimizing performance

Use issue templates to provide context, keep your `CLAUDE.md` concise and focused, and configure appropriate timeouts for your workflows.

#### CI costs

When using Claude Code GitHub Actions, be aware of the associated costs: **GitHub Actions costs:**

- Claude Code runs on GitHub-hosted runners, which consume your GitHub Actions minutes
- See [GitHub's billing documentation](https://docs.github.com/en/billing/managing-billing-for-your-products/managing-billing-for-github-actions/about-billing-for-github-actions) for detailed pricing and minute limits

**API costs:**

- Each Claude interaction consumes API tokens based on the length of prompts and responses
- Token usage varies by task complexity and codebase size
- See [Claude's pricing page](https://claude.com/platform/api) for current token rates

**Cost optimization tips:**

- Use specific `@claude` commands to reduce unnecessary API calls
- Configure appropriate `--max-turns` in `claude_args` to prevent excessive iterations
- Set workflow-level timeouts to avoid runaway jobs
- Consider using GitHub's concurrency controls to limit parallel runs

### Configuration examples

The Claude Code Action v1 simplifies configuration with unified parameters:

```
- uses : anthropics/claude-code-action@v1
with :
anthropic_api_key : ${{ secrets.ANTHROPIC_API_KEY }}
prompt : "Your instructions here" # Optional
claude_args : "--max-turns 5" # Optional CLI arguments
```

Key features:

- **Unified prompt interface** - Use `prompt` for all instructions
- **Skills** - Invoke installed [skills](/docs/en/skills) directly from the prompt
- **CLI passthrough** - Any Claude Code CLI argument via `claude_args`
- **Flexible triggers** - Works with any GitHub event

Visit the [examples directory](https://github.com/anthropics/claude-code-action/tree/main/examples) for complete workflow files.

When responding to issue or PR comments, Claude automatically responds to @claude mentions. For other events, use the `prompt` parameter to provide instructions.

### Using with AWS Bedrock & Google Vertex AI

For enterprise environments, you can use Claude Code GitHub Actions with your own cloud infrastructure. This approach gives you control over data residency and billing while maintaining the same functionality.

#### Prerequisites

Before setting up Claude Code GitHub Actions with cloud providers, you need:

##### For Google Cloud Vertex AI:

1. A Google Cloud Project with Vertex AI enabled
2. Workload Identity Federation configured for GitHub Actions
3. A service account with the required permissions
4. A GitHub App (recommended) or use the default GITHUB\_TOKEN

##### For AWS Bedrock:

1. An AWS account with Amazon Bedrock enabled
2. GitHub OIDC Identity Provider configured in AWS
3. An IAM role with Bedrock permissions
4. A GitHub App (recommended) or use the default GITHUB\_TOKEN

1

Create a custom GitHub App (Recommended for 3P Providers)

For best control and security when using 3P providers like Vertex AI or Bedrock, we recommend creating your own GitHub App:

1. Go to [https://github.com/settings/apps/new](https://github.com/settings/apps/new)
2. Fill in the basic information:
    - **GitHub App name** : Choose a unique name (e.g., "YourOrg Claude Assistant")
    - **Homepage URL** : Your organization's website or the repository URL
3. Configure the app settings:
    - **Webhooks** : Uncheck "Active" (not needed for this integration)
4. Set the required permissions:
    - **Repository permissions** :
        - Contents: Read & Write
        - Issues: Read & Write
        - Pull requests: Read & Write
5. Click "Create GitHub App"
6. After creation, click "Generate a private key" and save the downloaded `.pem` file
7. Note your App ID from the app settings page
8. Install the app to your repository:
    - From your app's settings page, click "Install App" in the left sidebar
    - Select your account or organization
    - Choose "Only select repositories" and select the specific repository
    - Click "Install"
9. Add the private key as a secret to your repository:
    - Go to your repository's Settings → Secrets and variables → Actions
    - Create a new secret named `APP_PRIVATE_KEY` with the contents of the `.pem` file
10. Add the App ID as a secret:

- Create a new secret named `APP_ID` with your GitHub App's ID

This app will be used with the [actions/create-github-app-token](https://github.com/actions/create-github-app-token) action to generate authentication tokens in your workflows.

**Alternative for Claude API or if you don't want to setup your own Github app** : Use the official Anthropic app:

1. Install from: [https://github.com/apps/claude](https://github.com/apps/claude)
2. No additional configuration needed for authentication

2

Configure cloud provider authentication

Choose your cloud provider and set up secure authentication:

AWS Bedrock

**Configure AWS to allow GitHub Actions to authenticate securely without storing credentials.**

**Security Note** : Use repository-specific configurations and grant only the minimum required permissions.

**Required Setup** :

1. **Enable Amazon Bedrock** :
    - Request access to Claude models in Amazon Bedrock
    - For cross-region models, request access in all required regions
2. **Set up GitHub OIDC Identity Provider** :
    - Provider URL: `https://token.actions.githubusercontent.com`
    - Audience: `sts.amazonaws.com`
3. **Create IAM Role for GitHub Actions** :
    - Trusted entity type: Web identity
    - Identity provider: `token.actions.githubusercontent.com`
    - Permissions: `AmazonBedrockFullAccess` policy
    - Configure trust policy for your specific repository

**Required Values** : After setup, you'll need:

- **AWS\_ROLE\_TO\_ASSUME** : The ARN of the IAM role you created

OIDC is more secure than using static AWS access keys because credentials are temporary and automatically rotated.

See [AWS documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_providers_create_oidc.html) for detailed OIDC setup instructions.

Google Vertex AI

**Configure Google Cloud to allow GitHub Actions to authenticate securely without storing credentials.**

**Security Note** : Use repository-specific configurations and grant only the minimum required permissions.

**Required Setup** :

1. **Enable APIs** in your Google Cloud project:
    - IAM Credentials API
    - Security Token Service (STS) API
    - Vertex AI API
2. **Create Workload Identity Federation resources** :
    - Create a Workload Identity Pool
    - Add a GitHub OIDC provider with:
        - Issuer: `https://token.actions.githubusercontent.com`
        - Attribute mappings for repository and owner
        - **Security recommendation** : Use repository-specific attribute conditions
3. **Create a Service Account** :
    - Grant only `Vertex AI User` role
    - **Security recommendation** : Create a dedicated service account per repository
4. **Configure IAM bindings** :
    - Allow the Workload Identity Pool to impersonate the service account
    - **Security recommendation** : Use repository-specific principal sets

**Required Values** : After setup, you'll need:

- **GCP\_WORKLOAD\_IDENTITY\_PROVIDER** : The full provider resource name
- **GCP\_SERVICE\_ACCOUNT** : The service account email address

Workload Identity Federation eliminates the need for downloadable service account keys, improving security.

For detailed setup instructions, consult the [Google Cloud Workload Identity Federation documentation](https://cloud.google.com/iam/docs/workload-identity-federation) .

3

Add Required Secrets

Add the following secrets to your repository (Settings → Secrets and variables → Actions):

##### For Claude API (Direct):

1. **For API Authentication** :
    - `ANTHROPIC_API_KEY` : Your Claude API key from [console.anthropic.com](https://console.anthropic.com/)
2. **For GitHub App (if using your own app)** :
    - `APP_ID` : Your GitHub App's ID
    - `APP_PRIVATE_KEY` : The private key (.pem) content

##### For Google Cloud Vertex AI

1. **For GCP Authentication** :
    - GCP\_WORKLOAD\_IDENTITY\_PROVIDER
    - GCP\_SERVICE\_ACCOUNT
2. **For GitHub App (if using your own app)** :
    - `APP_ID` : Your GitHub App's ID
    - `APP_PRIVATE_KEY` : The private key (.pem) content

##### For AWS Bedrock

1. **For AWS Authentication** :
    - AWS\_ROLE\_TO\_ASSUME
2. **For GitHub App (if using your own app)** :
    - `APP_ID` : Your GitHub App's ID
    - `APP_PRIVATE_KEY` : The private key (.pem) content

4

Create workflow files

Create GitHub Actions workflow files that integrate with your cloud provider. The examples below show complete configurations for both AWS Bedrock and Google Vertex AI:

AWS Bedrock workflow

**Prerequisites:**

- AWS Bedrock access enabled with Claude model permissions
- GitHub configured as an OIDC identity provider in AWS
- IAM role with Bedrock permissions that trusts GitHub Actions

**Required GitHub secrets:**

| Secret Name          | Description                                       |
|----------------------|---------------------------------------------------|
| `AWS_ROLE_TO_ASSUME` | ARN of the IAM role for Bedrock access            |
| `APP_ID`             | Your GitHub App ID (from app settings)            |
| `APP_PRIVATE_KEY`    | The private key you generated for your GitHub App |

```
name : Claude PR Action

permissions :
contents : write
pull-requests : write
issues : write
id-token : write

on :
issue_comment :
types : [ created ]
pull_request_review_comment :
types : [ created ]
issues :
types : [ opened , assigned ]

jobs :
claude-pr :
if : |
(github.event_name == 'issue_comment' && contains(github.event.comment.body, '@claude')) ||
(github.event_name == 'pull_request_review_comment' && contains(github.event.comment.body, '@claude')) ||
(github.event_name == 'issues' && contains(github.event.issue.body, '@claude'))
runs-on : ubuntu-latest
env :
AWS_REGION : us-west-2
steps :
- name : Checkout repository
uses : actions/checkout@v4

- name : Generate GitHub App token
id : app-token
uses : actions/create-github-app-token@v2
with :
app-id : ${{ secrets.APP_ID }}
private-key : ${{ secrets.APP_PRIVATE_KEY }}

- name : Configure AWS Credentials (OIDC)
uses : aws-actions/configure-aws-credentials@v4
with :
role-to-assume : ${{ secrets.AWS_ROLE_TO_ASSUME }}
aws-region : us-west-2

- uses : anthropics/claude-code-action@v1
with :
github_token : ${{ steps.app-token.outputs.token }}
use_bedrock : "true"
claude_args : '--model us.anthropic.claude-sonnet-4-6 --max-turns 10'
```

The model ID format for Bedrock includes a region prefix (for example, `us.anthropic.claude-sonnet-4-6` ).

Google Vertex AI workflow

**Prerequisites:**

- Vertex AI API enabled in your GCP project
- Workload Identity Federation configured for GitHub
- Service account with Vertex AI permissions

**Required GitHub secrets:**

| Secret Name                      | Description                                       |
|----------------------------------|---------------------------------------------------|
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Workload identity provider resource name          |
| `GCP_SERVICE_ACCOUNT`            | Service account email with Vertex AI access       |
| `APP_ID`                         | Your GitHub App ID (from app settings)            |
| `APP_PRIVATE_KEY`                | The private key you generated for your GitHub App |

```
name : Claude PR Action

permissions :
contents : write
pull-requests : write
issues : write
id-token : write

on :
issue_comment :
types : [ created ]
pull_request_review_comment :
types : [ created ]
issues :
types : [ opened , assigned ]

jobs :
claude-pr :
if : |
(github.event_name == 'issue_comment' && contains(github.event.comment.body, '@claude')) ||
(github.event_name == 'pull_request_review_comment' && contains(github.event.comment.body, '@claude')) ||
(github.event_name == 'issues' && contains(github.event.issue.body, '@claude'))
runs-on : ubuntu-latest
steps :
- name : Checkout repository
uses : actions/checkout@v4

- name : Generate GitHub App token
id : app-token
uses : actions/create-github-app-token@v2
with :
app-id : ${{ secrets.APP_ID }}
private-key : ${{ secrets.APP_PRIVATE_KEY }}

- name : Authenticate to Google Cloud
id : auth
uses : google-github-actions/auth@v2
with :
workload_identity_provider : ${{ secrets.GCP_WORKLOAD_IDENTITY_PROVIDER }}
service_account : ${{ secrets.GCP_SERVICE_ACCOUNT }}

- uses : anthropics/claude-code-action@v1
with :
github_token : ${{ steps.app-token.outputs.token }}
trigger_phrase : "@claude"
use_vertex : "true"
claude_args : '--model claude-sonnet-4-5@20250929 --max-turns 10'
env :
ANTHROPIC_VERTEX_PROJECT_ID : ${{ steps.auth.outputs.project_id }}
CLOUD_ML_REGION : us-east5
VERTEX_REGION_CLAUDE_4_5_SONNET : us-east5
```

The project ID is automatically retrieved from the Google Cloud authentication step, so you don't need to hardcode it.

### Troubleshooting

#### Claude not responding to @claude commands

Verify the GitHub App is installed correctly, check that workflows are enabled, ensure API key is set in repository secrets, and confirm the comment contains `@claude` (not `/claude` ).

#### CI not running on Claude's commits

Ensure you're using the GitHub App or custom app (not Actions user), check workflow triggers include the necessary events, and verify app permissions include CI triggers.

#### Authentication errors

Confirm API key is valid and has sufficient permissions. For Bedrock/Vertex, check credentials configuration and ensure secrets are named correctly in workflows.

### Advanced configuration

#### Action parameters

The Claude Code Action v1 uses a simplified configuration:

| Parameter           | Description                                                             | Required   |
|---------------------|-------------------------------------------------------------------------|------------|
| `prompt`            | Instructions for Claude (plain text or a [skill](/docs/en/skills) name) | No*        |
| `claude_args`       | CLI arguments passed to Claude Code                                     | No         |
| `anthropic_api_key` | Claude API key                                                          | Yes**      |
| `github_token`      | GitHub token for API access                                             | No         |
| `trigger_phrase`    | Custom trigger phrase (default: "@claude")                              | No         |
| `use_bedrock`       | Use AWS Bedrock instead of Claude API                                   | No         |
| `use_vertex`        | Use Google Vertex AI instead of Claude API                              | No         |

*Prompt is optional - when omitted for issue/PR comments, Claude responds to trigger phrase **Required for direct Claude API, not for Bedrock/Vertex

##### Pass CLI arguments

The `claude_args` parameter accepts any Claude Code CLI arguments:

```
claude_args : "--max-turns 5 --model claude-sonnet-4-6 --mcp-config /path/to/config.json"
```

Common arguments:

- `--max-turns` : Maximum conversation turns (default: 10)
- `--model` : Model to use (for example, `claude-sonnet-4-6` )
- `--mcp-config` : Path to MCP configuration
- `--allowedTools` : Comma-separated list of allowed tools. The `--allowed-tools` alias also works.
- `--debug` : Enable debug output

#### Alternative integration methods

While the `/install-github-app` command is the recommended approach, you can also:

- **Custom GitHub App** : For organizations needing branded usernames or custom authentication flows. Create your own GitHub App with required permissions (contents, issues, pull requests) and use the actions/create-github-app-token action to generate tokens in your workflows.
- **Manual GitHub Actions** : Direct workflow configuration for maximum flexibility
- **MCP Configuration** : Dynamic loading of Model Context Protocol servers

See the [Claude Code Action documentation](https://github.com/anthropics/claude-code-action/blob/main/docs) for detailed guides on authentication, security, and advanced configuration.

#### Customizing Claude's behavior

You can configure Claude's behavior in two ways:

1. **CLAUDE.md** : Define coding standards, review criteria, and project-specific rules in a `CLAUDE.md` file at the root of your repository. Claude will follow these guidelines when creating PRs and responding to requests. Check out our [Memory documentation](/docs/en/memory) for more details.
2. **Custom prompts** : Use the `prompt` parameter in the workflow file to provide workflow-specific instructions. This allows you to customize Claude's behavior for different workflows or tasks.

Claude will follow these guidelines when creating PRs and responding to requests.

Was this page helpful?

Yes

No

[Code Review](/docs/en/code-review) [GitHub Enterprise Server](/docs/en/github-enterprise-server)

⌘ I


### Claude Code GitLab CI/CD


Learn about integrating Claude Code into your development workflow with GitLab CI/CD


Claude Code for GitLab CI/CD is currently in beta. Features and functionality may evolve as we refine the experience. This integration is maintained by GitLab. For support, see the following [GitLab issue](https://gitlab.com/gitlab-org/gitlab/-/issues/573776) .

This integration is built on top of the [Claude Code CLI and Agent SDK](/docs/en/agent-sdk/overview) , enabling programmatic use of Claude in your CI/CD jobs and custom automation workflows.

### Why use Claude Code with GitLab?

- **Instant MR creation** : Describe what you need, and Claude proposes a complete MR with changes and explanation
- **Automated implementation** : Turn issues into working code with a single command or mention
- **Project-aware** : Claude follows your `CLAUDE.md` guidelines and existing code patterns
- **Simple setup** : Add one job to `.gitlab-ci.yml` and a masked CI/CD variable
- **Enterprise-ready** : Choose Claude API, AWS Bedrock, or Google Vertex AI to meet data residency and procurement needs
- **Secure by default** : Runs in your GitLab runners with your branch protection and approvals

### How it works

Claude Code uses GitLab CI/CD to run AI tasks in isolated jobs and commit results back via MRs:

1. **Event-driven orchestration** : GitLab listens for your chosen triggers (for example, a comment that mentions `@claude` in an issue, MR, or review thread). The job collects context from the thread and repository, builds prompts from that input, and runs Claude Code.
2. **Provider abstraction** : Use the provider that fits your environment:
    - Claude API (SaaS)
    - AWS Bedrock (IAM-based access, cross-region options)
    - Google Vertex AI (GCP-native, Workload Identity Federation)
3. **Sandboxed execution** : Each interaction runs in a container with strict network and filesystem rules. Claude Code enforces workspace-scoped permissions to constrain writes. Every change flows through an MR so reviewers see the diff and approvals still apply.

Pick regional endpoints to reduce latency and meet data-sovereignty requirements while using existing cloud agreements.

### What can Claude do?

Claude Code enables powerful CI/CD workflows that transform how you work with code:

- Create and update MRs from issue descriptions or comments
- Analyze performance regressions and propose optimizations
- Implement features directly in a branch, then open an MR
- Fix bugs and regressions identified by tests or comments
- Respond to follow-up comments to iterate on requested changes

### Setup

#### Quick setup

The fastest way to get started is to add a minimal job to your `.gitlab-ci.yml` and set your API key as a masked variable.

1. **Add a masked CI/CD variable**
    - Go to **Settings** → **CI/CD** → **Variables**
    - Add `ANTHROPIC_API_KEY` (masked, protected as needed)
2. **Add a Claude job to** **`.gitlab-ci.yml`**

```
stages :
- ai

claude :
stage : ai
image : node:24-alpine3.21
### Adjust rules to fit how you want to trigger the job:
### - manual runs
### - merge request events
### - web/API triggers when a comment contains '@claude'
rules :
- if : '$CI_PIPELINE_SOURCE == "web"'
- if : '$CI_PIPELINE_SOURCE == "merge_request_event"'
variables :
GIT_STRATEGY : fetch
before_script :
- apk update
- apk add --no-cache git curl bash
- curl -fsSL https://claude.ai/install.sh | bash
script :
### Optional: start a GitLab MCP server if your setup provides one
- /bin/gitlab-mcp-server || true
### Use AI_FLOW_* variables when invoking via web/API triggers with context payloads
- echo "$AI_FLOW_INPUT for $AI_FLOW_CONTEXT on $AI_FLOW_EVENT"
- >
claude
-p "${AI_FLOW_INPUT:-'Review this MR and implement the requested changes'}"
--permission-mode acceptEdits
--allowedTools "Bash Read Edit Write mcp__gitlab"
--debug
```

After adding the job and your `ANTHROPIC_API_KEY` variable, test by running the job manually from **CI/CD** → **Pipelines** , or trigger it from an MR to let Claude propose updates in a branch and open an MR if needed.

To run on AWS Bedrock or Google Vertex AI instead of the Claude API, see the [Using with AWS Bedrock & Google Vertex AI](#using-with-aws-bedrock--google-vertex-ai) section below for authentication and environment setup.

#### Manual setup (recommended for production)

If you prefer a more controlled setup or need enterprise providers:

1. **Configure provider access** :
    - **Claude API** : Create and store `ANTHROPIC_API_KEY` as a masked CI/CD variable
    - **AWS Bedrock** : **Configure GitLab** → **AWS OIDC** and create an IAM role for Bedrock
    - **Google Vertex AI** : **Configure Workload Identity Federation for GitLab** → **GCP**
2. **Add project credentials for GitLab API operations** :
    - Use `CI_JOB_TOKEN` by default, or create a Project Access Token with `api` scope
    - Store as `GITLAB_ACCESS_TOKEN` (masked) if using a PAT
3. **Add the Claude job to** **`.gitlab-ci.yml`** (see examples below)
4. **(Optional) Enable mention-driven triggers** :
    - Add a project webhook for "Comments (notes)" to your event listener (if you use one)
    - Have the listener call the pipeline trigger API with variables like `AI_FLOW_INPUT` and `AI_FLOW_CONTEXT` when a comment contains `@claude`

### Example use cases

#### Turn issues into MRs

In an issue comment:

```
@claude implement this feature based on the issue description
```

Claude analyzes the issue and codebase, writes changes in a branch, and opens an MR for review.

#### Get implementation help

In an MR discussion:

```
@claude suggest a concrete approach to cache the results of this API call
```

Claude proposes changes, adds code with appropriate caching, and updates the MR.

#### Fix bugs quickly

In an issue or MR comment:

```
@claude fix the TypeError in the user dashboard component
```

Claude locates the bug, implements a fix, and updates the branch or opens a new MR.

### Using with AWS Bedrock & Google Vertex AI

For enterprise environments, you can run Claude Code entirely on your cloud infrastructure with the same developer experience.

- AWS Bedrock
- Google Vertex AI

#### Prerequisites

Before setting up Claude Code with AWS Bedrock, you need:

1. An AWS account with Amazon Bedrock access to the desired Claude models
2. GitLab configured as an OIDC identity provider in AWS IAM
3. An IAM role with Bedrock permissions and a trust policy restricted to your GitLab project/refs
4. GitLab CI/CD variables for role assumption:
    - `AWS_ROLE_TO_ASSUME` (role ARN)
    - `AWS_REGION` (Bedrock region)

#### Setup instructions

Configure AWS to allow GitLab CI jobs to assume an IAM role via OIDC (no static keys). **Required setup:**

1. Enable Amazon Bedrock and request access to your target Claude models
2. Create an IAM OIDC provider for GitLab if not already present
3. Create an IAM role trusted by the GitLab OIDC provider, restricted to your project and protected refs
4. Attach least-privilege permissions for Bedrock invoke APIs

**Required values to store in CI/CD variables:**

- AWS\_ROLE\_TO\_ASSUME
- AWS\_REGION

Add variables in Settings → CI/CD → Variables:

```
### For AWS Bedrock:
- AWS_ROLE_TO_ASSUME
- AWS_REGION
```

Use the AWS Bedrock job example above to exchange the GitLab job token for temporary AWS credentials at runtime.

#### Prerequisites

Before setting up Claude Code with Google Vertex AI, you need:

1. A Google Cloud project with:
    - Vertex AI API enabled
    - Workload Identity Federation configured to trust GitLab OIDC
2. A dedicated service account with only the required Vertex AI roles
3. GitLab CI/CD variables for WIF:
    - `GCP_WORKLOAD_IDENTITY_PROVIDER` (full resource name)
    - `GCP_SERVICE_ACCOUNT` (service account email)

#### Setup instructions

Configure Google Cloud to allow GitLab CI jobs to impersonate a service account via Workload Identity Federation. **Required setup:**

1. Enable IAM Credentials API, STS API, and Vertex AI API
2. Create a Workload Identity Pool and provider for GitLab OIDC
3. Create a dedicated service account with Vertex AI roles
4. Grant the WIF principal permission to impersonate the service account

**Required values to store in CI/CD variables:**

- GCP\_WORKLOAD\_IDENTITY\_PROVIDER
- GCP\_SERVICE\_ACCOUNT

Add variables in Settings → CI/CD → Variables:

```
### For Google Vertex AI:
- GCP_WORKLOAD_IDENTITY_PROVIDER
- GCP_SERVICE_ACCOUNT
- CLOUD_ML_REGION (for example, us-east5)
```

Use the Google Vertex AI job example above to authenticate without storing keys.

### Configuration examples

Below are ready-to-use snippets you can adapt to your pipeline.

#### Basic .gitlab-ci.yml (Claude API)

```
stages :
- ai

claude :
stage : ai
image : node:24-alpine3.21
rules :
- if : '$CI_PIPELINE_SOURCE == "web"'
- if : '$CI_PIPELINE_SOURCE == "merge_request_event"'
variables :
GIT_STRATEGY : fetch
before_script :
- apk update
- apk add --no-cache git curl bash
- curl -fsSL https://claude.ai/install.sh | bash
script :
- /bin/gitlab-mcp-server || true
- >
claude
-p "${AI_FLOW_INPUT:-'Summarize recent changes and suggest improvements'}"
--permission-mode acceptEdits
--allowedTools "Bash Read Edit Write mcp__gitlab"
--debug
### Claude Code will use ANTHROPIC_API_KEY from CI/CD variables
```

#### AWS Bedrock job example (OIDC)

**Prerequisites:**

- Amazon Bedrock enabled with access to your chosen Claude model(s)
- GitLab OIDC configured in AWS with a role that trusts your GitLab project and refs
- IAM role with Bedrock permissions (least privilege recommended)

**Required CI/CD variables:**

- `AWS_ROLE_TO_ASSUME` : ARN of the IAM role for Bedrock access
- `AWS_REGION` : Bedrock region (for example, `us-west-2` )

```
claude-bedrock :
stage : ai
image : node:24-alpine3.21
rules :
- if : '$CI_PIPELINE_SOURCE == "web"'
before_script :
- apk add --no-cache bash curl jq git python3 py3-pip
- pip install --no-cache-dir awscli
- curl -fsSL https://claude.ai/install.sh | bash
### Exchange GitLab OIDC token for AWS credentials
- export AWS_WEB_IDENTITY_TOKEN_FILE="${CI_JOB_JWT_FILE:-/tmp/oidc_token}"
- if [ -n "${CI_JOB_JWT_V2}" ]; then printf "%s" "$CI_JOB_JWT_V2" > "$AWS_WEB_IDENTITY_TOKEN_FILE"; fi
- >
aws sts assume-role-with-web-identity
--role-arn "$AWS_ROLE_TO_ASSUME"
--role-session-name "gitlab-claude-$(date +%s)"
--web-identity-token "file://$AWS_WEB_IDENTITY_TOKEN_FILE"
--duration-seconds 3600 > /tmp/aws_creds.json
- export AWS_ACCESS_KEY_ID="$(jq -r .Credentials.AccessKeyId /tmp/aws_creds.json)"
- export AWS_SECRET_ACCESS_KEY="$(jq -r .Credentials.SecretAccessKey /tmp/aws_creds.json)"
- export AWS_SESSION_TOKEN="$(jq -r .Credentials.SessionToken /tmp/aws_creds.json)"
script :
- /bin/gitlab-mcp-server || true
- >
claude
-p "${AI_FLOW_INPUT:-'Implement the requested changes and open an MR'}"
--permission-mode acceptEdits
--allowedTools "Bash Read Edit Write mcp__gitlab"
--debug
variables :
AWS_REGION : "us-west-2"
```

Model IDs for Bedrock include region-specific prefixes (for example, `us.anthropic.claude-sonnet-4-6` ). Pass the desired model via your job configuration or prompt if your workflow supports it.

#### Google Vertex AI job example (Workload Identity Federation)

**Prerequisites:**

- Vertex AI API enabled in your GCP project
- Workload Identity Federation configured to trust GitLab OIDC
- A service account with Vertex AI permissions

**Required CI/CD variables:**

- `GCP_WORKLOAD_IDENTITY_PROVIDER` : Full provider resource name
- `GCP_SERVICE_ACCOUNT` : Service account email
- `CLOUD_ML_REGION` : Vertex region (for example, `us-east5` )

```
claude-vertex :
stage : ai
image : gcr.io/google.com/cloudsdktool/google-cloud-cli:slim
rules :
- if : '$CI_PIPELINE_SOURCE == "web"'
before_script :
- apt-get update && apt-get install -y git && apt-get clean
- curl -fsSL https://claude.ai/install.sh | bash
### Authenticate to Google Cloud via WIF (no downloaded keys)
- >
gcloud auth login --cred-file=<(cat <<EOF
{
"type": "external_account",
"audience": "${GCP_WORKLOAD_IDENTITY_PROVIDER}",
"subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
"service_account_impersonation_url": "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/${GCP_SERVICE_ACCOUNT}:generateAccessToken",
"token_url": "https://sts.googleapis.com/v1/token"
}
EOF
)
- gcloud config set project "$(gcloud projects list --format='value(projectId)' --filter="name:${CI_PROJECT_NAMESPACE}" | head -n1)" || true
script :
- /bin/gitlab-mcp-server || true
- >
CLOUD_ML_REGION="${CLOUD_ML_REGION:-us-east5}"
claude
-p "${AI_FLOW_INPUT:-'Review and update code as requested'}"
--permission-mode acceptEdits
--allowedTools "Bash Read Edit Write mcp__gitlab"
--debug
variables :
CLOUD_ML_REGION : "us-east5"
```

With Workload Identity Federation, you do not need to store service account keys. Use repository-specific trust conditions and least-privilege service accounts.

### Best practices

#### CLAUDE.md configuration

Create a `CLAUDE.md` file at the repository root to define coding standards, review criteria, and project-specific rules. Claude reads this file during runs and follows your conventions when proposing changes.

#### Security considerations

**Never commit API keys or cloud credentials to your repository** . Always use GitLab CI/CD variables:

- Add `ANTHROPIC_API_KEY` as a masked variable (and protect it if needed)
- Use provider-specific OIDC where possible (no long-lived keys)
- Limit job permissions and network egress
- Review Claude's MRs like any other contributor

#### Optimizing performance

- Keep `CLAUDE.md` focused and concise
- Provide clear issue/MR descriptions to reduce iterations
- Configure sensible job timeouts to avoid runaway runs
- Cache npm and package installs in runners where possible

#### CI costs

When using Claude Code with GitLab CI/CD, be aware of associated costs:

- **GitLab Runner time** :
    - Claude runs on your GitLab runners and consumes compute minutes
    - See your GitLab plan's runner billing for details
- **API costs** :
    - Each Claude interaction consumes tokens based on prompt and response size
    - Token usage varies by task complexity and codebase size
    - See [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) for details
- **Cost optimization tips** :
    - Use specific `@claude` commands to reduce unnecessary turns
    - Set appropriate `max_turns` and job timeout values
    - Limit concurrency to control parallel runs

### Security and governance

- Each job runs in an isolated container with restricted network access
- Claude's changes flow through MRs so reviewers see every diff
- Branch protection and approval rules apply to AI-generated code
- Claude Code uses workspace-scoped permissions to constrain writes
- Costs remain under your control because you bring your own provider credentials

### Troubleshooting

#### Claude not responding to @claude commands

- Verify your pipeline is being triggered (manually, MR event, or via a note event listener/webhook)
- Ensure CI/CD variables ( `ANTHROPIC_API_KEY` or cloud provider settings) are present and unmasked
- Check that the comment contains `@claude` (not `/claude` ) and that your mention trigger is configured

#### Job can't write comments or open MRs

- Ensure `CI_JOB_TOKEN` has sufficient permissions for the project, or use a Project Access Token with `api` scope
- Check the `mcp__gitlab` tool is enabled in `--allowedTools`
- Confirm the job runs in the context of the MR or has enough context via `AI_FLOW_*` variables

#### Authentication errors

- **For Claude API** : Confirm `ANTHROPIC_API_KEY` is valid and unexpired
- **For Bedrock/Vertex** : Verify OIDC/WIF configuration, role impersonation, and secret names; confirm region and model availability

### Advanced configuration

#### Common parameters and variables

Claude Code supports these commonly used inputs:

- `prompt` / `prompt_file` : Provide instructions inline ( `-p` ) or via a file
- `max_turns` : Limit the number of back-and-forth iterations
- `timeout_minutes` : Limit total execution time
- `ANTHROPIC_API_KEY` : Required for the Claude API (not used for Bedrock/Vertex)
- Provider-specific environment: `AWS_REGION` , project/region vars for Vertex

Exact flags and parameters may vary by version of `@anthropic-ai/claude-code` . Run `claude --help` in your job to see supported options.

#### Customizing Claude's behavior

You can guide Claude in two primary ways:

1. **CLAUDE.md** : Define coding standards, security requirements, and project conventions. Claude reads this during runs and follows your rules.
2. **Custom prompts** : Pass task-specific instructions via `prompt` / `prompt_file` in the job. Use different prompts for different jobs (for example, review, implement, refactor).

Was this page helpful?

Yes

No

[GitHub Enterprise Server](/docs/en/github-enterprise-server) [Claude Code in Slack](/docs/en/slack)

⌘ I


### Code Review


Set up automated PR reviews that catch logic errors, security vulnerabilities, and regressions using multi-agent analysis of your full codebase


Code Review is in research preview, available for [Team and Enterprise](https://claude.ai/admin-settings/claude-code) subscriptions. It is not available for organizations with [Zero Data Retention](/docs/en/zero-data-retention) enabled.

Code Review analyzes your GitHub pull requests and posts findings as inline comments on the lines of code where it found issues. A fleet of specialized agents examine the code changes in the context of your full codebase, looking for logic errors, security vulnerabilities, broken edge cases, and subtle regressions. Findings are tagged by severity and don't approve or block your PR, so existing review workflows stay intact. You can tune what Claude flags by adding a `CLAUDE.md` or `REVIEW.md` file to your repository. To run Claude in your own CI infrastructure instead of this managed service, see [GitHub Actions](/docs/en/github-actions) or [GitLab CI/CD](/docs/en/gitlab-ci-cd) . For repositories on a self-hosted GitHub instance, see [GitHub Enterprise Server](/docs/en/github-enterprise-server) . This page covers:

- [How reviews work](#how-reviews-work)
- [Setup](#set-up-code-review)
- [Triggering reviews manually](#manually-trigger-reviews) with `@claude review` and `@claude review once`
- [Customizing reviews](#customize-reviews) with `CLAUDE.md` and `REVIEW.md`
- [Pricing](#pricing)
- [Troubleshooting](#troubleshooting) failed runs and missing comments

### How reviews work

Once an admin [enables Code Review](#set-up-code-review) for your organization, reviews trigger when a PR opens, on every push, or when manually requested, depending on the repository's configured behavior. Commenting `@claude review` [starts reviews on a PR](#manually-trigger-reviews) in any mode. When a review runs, multiple agents analyze the diff and surrounding code in parallel on Anthropic infrastructure. Each agent looks for a different class of issue, then a verification step checks candidates against actual code behavior to filter out false positives. The results are deduplicated, ranked by severity, and posted as inline comments on the specific lines where issues were found. If no issues are found, Claude posts a short confirmation comment on the PR. Reviews scale in cost with PR size and complexity, completing in 20 minutes on average. Admins can monitor review activity and spend via the [analytics dashboard](#view-usage) .

#### Severity levels

Each finding is tagged with a severity level:

| Marker   | Severity     | Meaning                                                             |
|----------|--------------|---------------------------------------------------------------------|
| 🔴        | Important    | A bug that should be fixed before merging                           |
| 🟡        | Nit          | A minor issue, worth fixing but not blocking                        |
| 🟣        | Pre-existing | A bug that exists in the codebase but was not introduced by this PR |

Findings include a collapsible extended reasoning section you can expand to understand why Claude flagged the issue and how it verified the problem.

#### Rate and reply to findings

Each review comment from Claude arrives with 👍 and 👎 already attached so both buttons appear in the GitHub UI for one-click rating. Click 👍 if the finding was useful or 👎 if it was wrong or noisy. Anthropic collects reaction counts after the PR merges and uses them to tune the reviewer. Reactions do not trigger a re-review or change anything on the PR. Replying to an inline comment does not prompt Claude to respond or update the PR. To act on a finding, fix the code and push. If the PR is subscribed to push-triggered reviews, the next run resolves the thread when the issue is fixed. To request a fresh review without pushing, comment `@claude review once` as a [top-level PR comment](#manually-trigger-reviews) .

#### Check run output

Beyond the inline review comments, each review populates the **Claude Code Review** check run that appears alongside your CI checks. Expand its **Details** link to see a summary of every finding in one place, sorted by severity:

| Severity    | File:Line                 | Issue                                                          |
|-------------|---------------------------|----------------------------------------------------------------|
| 🔴 Important | `src/auth/session.ts:142` | Token refresh races with logout, leaving stale sessions active |
| 🟡 Nit       | `src/auth/session.ts:88`  | `parseExpiry` silently returns 0 on malformed input            |

Each finding also appears as an annotation in the **Files changed** tab, marked directly on the relevant diff lines. Important findings render with a red marker, nits with a yellow warning, and pre-existing bugs with a gray notice. Annotations and the severity table are written to the check run independently of inline review comments, so they remain available even if GitHub rejects an inline comment on a line that moved. The check run always completes with a neutral conclusion so it never blocks merging through branch protection rules. If you want to gate merges on Code Review findings, read the severity breakdown from the check run output in your own CI. The last line of the Details text is a machine-readable comment your workflow can parse with `gh` and jq:

```
gh api repos/OWNER/REPO/check-runs/CHECK_RUN_ID \
--jq '.output.text | split("bughunter-severity: ")[1] | split(" -->")[0] | fromjson'
```

This returns a JSON object with counts per severity, for example `{"normal": 2, "nit": 1, "pre_existing": 0}` . The `normal` key holds the count of Important findings; a non-zero value means Claude found at least one bug worth fixing before merge.

#### What Code Review checks

By default, Code Review focuses on correctness: bugs that would break production, not formatting preferences or missing test coverage. You can expand what it checks by [adding guidance files](#customize-reviews) to your repository.

### Set up Code Review

An admin enables Code Review once for the organization and selects which repositories to include.

1

Open Claude Code admin settings

Go to [claude.ai/admin-settings/claude-code](https://claude.ai/admin-settings/claude-code) and find the Code Review section. You need admin access to your Claude organization and permission to install GitHub Apps in your GitHub organization.

2

Start setup

Click **Setup** . This begins the GitHub App installation flow.

3

Install the Claude GitHub App

Follow the prompts to install the Claude GitHub App to your GitHub organization. The app requests these repository permissions:

- **Contents** : read and write
- **Issues** : read and write
- **Pull requests** : read and write

Code Review uses read access to contents and write access to pull requests. The broader permission set also supports [GitHub Actions](/docs/en/github-actions) if you enable that later.

4

Select repositories

Choose which repositories to enable for Code Review. If you don't see a repository, make sure you gave the Claude GitHub App access to it during installation. You can add more repositories later.

5

Set review triggers per repo

After setup completes, the Code Review section shows your repositories in a table. For each repository, use the **Review Behavior** dropdown to choose when reviews run:

- **Once after PR creation** : review runs once when a PR is opened or marked ready for review
- **After every push** : review runs on every push to the PR branch, catching new issues as the PR evolves and auto-resolving threads when you fix flagged issues
- **Manual** : reviews start only when someone [comments](#manually-trigger-reviews) [`@claude review`](#manually-trigger-reviews) [or](#manually-trigger-reviews) [`@claude review once`](#manually-trigger-reviews) [on a PR](#manually-trigger-reviews) ; `@claude review` also subscribes the PR to reviews on subsequent pushes

Reviewing on every push runs the most reviews and costs the most. Manual mode is useful for high-traffic repos where you want to opt specific PRs into review, or to only start reviewing your PRs once they're ready.

The repositories table also shows the average cost per review for each repo based on recent activity. Use the row actions menu to turn Code Review on or off per repository, or to remove a repository entirely. To verify setup, open a test PR. If you chose an automatic trigger, a check run named **Claude Code Review** appears within a few minutes. If you chose Manual, comment `@claude review` on the PR to start the first review. If no check run appears, confirm the repository is listed in your admin settings and the Claude GitHub App has access to it.

### Manually trigger reviews

Two comment commands start a review on demand. Both work regardless of the repository's configured trigger, so you can use them to opt specific PRs into review in Manual mode or to get an immediate re-review in other modes.

| Command               | What it does                                                                  |
|-----------------------|-------------------------------------------------------------------------------|
| `@claude review`      | Starts a review and subscribes the PR to push-triggered reviews going forward |
| `@claude review once` | Starts a single review without subscribing the PR to future pushes            |

Use `@claude review once` when you want feedback on the current state of a PR but don't want every subsequent push to incur a review. This is useful for long-running PRs with frequent pushes, or when you want a one-off second opinion without changing the PR's review behavior. For either command to trigger a review:

- Post it as a top-level PR comment, not an inline comment on a diff line
- Put the command at the start of the comment, with `once` on the same line if you're using the one-shot form
- You must have owner, member, or collaborator access to the repository
- The PR must be open

Unlike automatic triggers, manual triggers run on draft PRs, since an explicit request signals you want the review now regardless of draft status. If a review is already running on that PR, the request is queued until the in-progress review completes. You can monitor progress via the check run on the PR.

### Customize reviews

Code Review reads two files from your repository to guide what it flags. Both are additive on top of the default correctness checks:

- **`CLAUDE.md`** : shared project instructions that Claude Code uses for all tasks, not just reviews. Use it when guidance also applies to interactive Claude Code sessions.
- **`REVIEW.md`** : review-only guidance, read exclusively during code reviews. Use it for rules that are strictly about what to flag or skip during review and would clutter your general `CLAUDE.md` .

#### CLAUDE.md

Code Review reads your repository's `CLAUDE.md` files and treats newly-introduced violations as nit-level findings. This works bidirectionally: if your PR changes code in a way that makes a `CLAUDE.md` statement outdated, Claude flags that the docs need updating too. Claude reads `CLAUDE.md` files at every level of your directory hierarchy, so rules in a subdirectory's `CLAUDE.md` apply only to files under that path. See the [memory documentation](/docs/en/memory) for more on how `CLAUDE.md` works. For review-specific guidance that you don't want applied to general Claude Code sessions, use [`REVIEW.md`](#review-md) instead.

#### REVIEW.md

Add a `REVIEW.md` file to your repository root for review-specific rules. Use it to encode:

- Company or team style guidelines: "prefer early returns over nested conditionals"
- Language- or framework-specific conventions not covered by linters
- Things Claude should always flag: "any new API route must have an integration test"
- Things Claude should skip: "don't comment on formatting in generated code under `/gen/` "

Example `REVIEW.md` :

```
### Code Review Guidelines

### Always check
- New API endpoints have corresponding integration tests
- Database migrations are backward-compatible
- Error messages don't leak internal details to users

### Style
- Prefer `match` statements over chained `isinstance` checks
- Use structured logging, not f-string interpolation in log calls

### Skip
- Generated files under `src/gen/`
- Formatting-only changes in `*.lock` files
```

Claude auto-discovers `REVIEW.md` at the repository root. No configuration needed.

### View usage

Go to [claude.ai/analytics/code-review](https://claude.ai/analytics/code-review) to see Code Review activity across your organization. The dashboard shows:

| Section              | What it shows                                                                            |
|----------------------|------------------------------------------------------------------------------------------|
| PRs reviewed         | Daily count of pull requests reviewed over the selected time range                       |
| Cost weekly          | Weekly spend on Code Review                                                              |
| Feedback             | Count of review comments that were auto-resolved because a developer addressed the issue |
| Repository breakdown | Per-repo counts of PRs reviewed and comments resolved                                    |

The repositories table in admin settings also shows average cost per review for each repo.

### Pricing

Code Review is billed based on token usage. Each review averages $15-25 in cost, scaling with PR size, codebase complexity, and how many issues require verification. Code Review usage is billed separately through [extra usage](https://support.claude.com/en/articles/12429409-extra-usage-for-paid-claude-plans) and does not count against your plan's included usage. The review trigger you choose affects total cost:

- **Once after PR creation** : runs once per PR
- **After every push** : runs on each push, multiplying cost by the number of pushes
- **Manual** : no reviews until someone comments `@claude review` on a PR

In any mode, commenting `@claude review` [opts the PR into push-triggered reviews](#manually-trigger-reviews) , so additional cost accrues per push after that comment. To run a single review without subscribing to future pushes, comment `@claude review once` instead. Costs appear on your Anthropic bill regardless of whether your organization uses AWS Bedrock or Google Vertex AI for other Claude Code features. To set a monthly spend cap for Code Review, go to [claude.ai/admin-settings/usage](https://claude.ai/admin-settings/usage) and configure the limit for the Claude Code Review service. Monitor spend via the weekly cost chart in [analytics](#view-usage) or the per-repo average cost column in admin settings.

### Troubleshooting

Review runs are best-effort. A failed run never blocks your PR, but it also doesn't retry on its own. This section covers how to recover from a failed run and where to look when the check run reports issues you can't find.

#### Retrigger a failed or timed-out review

When the review infrastructure hits an internal error or exceeds its time limit, the check run completes with a title of **Code review encountered an error** or **Code review timed out** . The conclusion is still neutral, so nothing blocks your merge, but no findings are posted. To run the review again, comment `@claude review once` on the PR. This starts a fresh review without subscribing the PR to future pushes. If the PR is already subscribed to push-triggered reviews, pushing a new commit also starts a new review. The **Re-run** button in GitHub's Checks tab does not retrigger Code Review. Use the comment command or a new push instead.

#### Find issues that aren't showing as inline comments

If the check run title says issues were found but you don't see inline review comments on the diff, look in these other locations where findings are surfaced:

- **Check run Details** : click **Details** next to the Claude Code Review check in the Checks tab. The severity table lists every finding with its file, line, and summary regardless of whether the inline comment was accepted.
- **Files changed annotations** : open the **Files changed** tab on the PR. Findings render as annotations attached directly to the diff lines, separate from review comments.
- **Review body** : if you pushed to the PR while a review was running, some findings may reference lines that no longer exist in the current diff. Those appear under an **Additional findings** heading in the review body text rather than as inline comments.

### Related resources

Code Review is designed to work alongside the rest of Claude Code. If you want to run reviews locally before opening a PR, need a self-hosted setup, or want to go deeper on how `CLAUDE.md` shapes Claude's behavior across tools, these pages are good next stops:

- [Plugins](/docs/en/discover-plugins) : browse the plugin marketplace, including a `code-review` plugin for running on-demand reviews locally before pushing
- [GitHub Actions](/docs/en/github-actions) : run Claude in your own GitHub Actions workflows for custom automation beyond code review
- [GitLab CI/CD](/docs/en/gitlab-ci-cd) : self-hosted Claude integration for GitLab pipelines
- [Memory](/docs/en/memory) : how `CLAUDE.md` files work across Claude Code
- [Analytics](/docs/en/analytics) : track Claude Code usage beyond code review

Was this page helpful?

Yes

No

[JetBrains IDEs](/docs/en/jetbrains) [GitHub Actions](/docs/en/github-actions)

⌘ I


---

# Platforms


### Platforms and integrations


Choose where to run Claude Code and what to connect it to. Compare the CLI, Desktop, VS Code, JetBrains, web, mobile, and integrations like Chrome, Slack, and CI/CD.


Claude Code runs the same underlying engine everywhere, but each surface is tuned for a different way of working. This page helps you pick the right platform for your workflow and connect the tools you already use.

### Where to run Claude Code

Choose a platform based on how you like to work and where your project lives.

| Platform                               | Best for                                                                                           | What you get                                                                                                                                                                                        |
|----------------------------------------|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [CLI](/docs/en/quickstart)             | Terminal workflows, scripting, remote servers                                                      | Full feature set, [Agent SDK](/docs/en/headless) , [computer use](/docs/en/computer-use) on macOS (Pro and Max), third-party providers                                                              |
| [Desktop](/docs/en/desktop)            | Visual review, parallel sessions, managed setup                                                    | Diff viewer, app preview, [computer use](/docs/en/desktop#let-claude-use-your-computer) and [Dispatch](/docs/en/desktop#sessions-from-dispatch) on Pro and Max                                      |
| [VS Code](/docs/en/vs-code)            | Working inside VS Code without switching to a terminal                                             | Inline diffs, integrated terminal, file context                                                                                                                                                     |
| [JetBrains](/docs/en/jetbrains)        | Working inside IntelliJ, PyCharm, WebStorm, or other JetBrains IDEs                                | Diff viewer, selection sharing, terminal session                                                                                                                                                    |
| [Web](/docs/en/claude-code-on-the-web) | Long-running tasks that don't need much steering, or work that should continue when you're offline | Anthropic-managed cloud, continues after you disconnect                                                                                                                                             |
| Mobile                                 | Starting and monitoring tasks while away from your computer                                        | Cloud sessions from the Claude app for iOS and Android, [Remote Control](/docs/en/remote-control) for local sessions, [Dispatch](/docs/en/desktop#sessions-from-dispatch) to Desktop on Pro and Max |

The CLI is the most complete surface for terminal-native work: scripting, third-party providers, and the Agent SDK are CLI-only. Desktop and the IDE extensions trade some CLI-only features for visual review and tighter editor integration. The web runs in Anthropic's cloud, so tasks keep going after you disconnect. Mobile is a thin client into those same cloud sessions or into a local session via Remote Control, and can send tasks to Desktop with Dispatch. You can mix surfaces on the same project. Configuration, project memory, and MCP servers are shared across the local surfaces.

### Connect your tools

Integrations let Claude work with services outside your codebase.

| Integration                               | What it does                                       | Use it for                                                       |
|-------------------------------------------|----------------------------------------------------|------------------------------------------------------------------|
| [Chrome](/docs/en/chrome)                 | Controls your browser with your logged-in sessions | Testing web apps, filling forms, automating sites without an API |
| [GitHub Actions](/docs/en/github-actions) | Runs Claude in your CI pipeline                    | Automated PR reviews, issue triage, scheduled maintenance        |
| [GitLab CI/CD](/docs/en/gitlab-ci-cd)     | Same as GitHub Actions for GitLab                  | CI-driven automation on GitLab                                   |
| [Code Review](/docs/en/code-review)       | Reviews every PR automatically                     | Catching bugs before human review                                |
| [Slack](/docs/en/slack)                   | Responds to `@Claude` mentions in your channels    | Turning bug reports into pull requests from team chat            |

For integrations not listed here, [MCP servers](/docs/en/mcp) and [connectors](/docs/en/desktop#connect-external-tools) let you connect almost anything: Linear, Notion, Google Drive, or your own internal APIs.

### Work when you are away from your terminal

Claude Code offers several ways to work when you're not at your terminal. They differ in what triggers the work, where Claude runs, and how much you need to set up.

|                                                     | Trigger                                                                                        | Claude runs on                                                                                                           | Setup                                                                                                                                          | Best for                                                      |
|-----------------------------------------------------|------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| [Dispatch](/docs/en/desktop#sessions-from-dispatch) | Message a task from the Claude mobile app                                                      | Your machine (Desktop)                                                                                                   | [Pair the mobile app with Desktop](https://support.claude.com/en/articles/13947068)                                                            | Delegating work while you're away, minimal setup              |
| [Remote Control](/docs/en/remote-control)           | Drive a running session from [claude.ai/code](https://claude.ai/code) or the Claude mobile app | Your machine (CLI or VS Code)                                                                                            | Run `claude remote-control`                                                                                                                    | Steering in-progress work from another device                 |
| [Channels](/docs/en/channels)                       | Push events from a chat app like Telegram or Discord, or your own server                       | Your machine (CLI)                                                                                                       | [Install a channel plugin](/docs/en/channels#quickstart) or [build your own](/docs/en/channels-reference)                                      | Reacting to external events like CI failures or chat messages |
| [Slack](/docs/en/slack)                             | Mention `@Claude` in a team channel                                                            | Anthropic cloud                                                                                                          | [Install the Slack app](/docs/en/slack#setting-up-claude-code-in-slack) with [Claude Code on the web](/docs/en/claude-code-on-the-web) enabled | PRs and reviews from team chat                                |
| [Scheduled tasks](/docs/en/scheduled-tasks)         | Set a schedule                                                                                 | [CLI](/docs/en/scheduled-tasks) , [Desktop](/docs/en/desktop-scheduled-tasks) , or [cloud](/docs/en/web-scheduled-tasks) | Pick a frequency                                                                                                                               | Recurring automation like daily reviews                       |

If you're not sure where to start, [install the CLI](/docs/en/quickstart) and run it in a project directory. If you'd rather not use a terminal, [Desktop](/docs/en/desktop-quickstart) gives you the same engine with a graphical interface.

### Related resources

#### Platforms

- [CLI quickstart](/docs/en/quickstart) : install and run your first command in the terminal
- [Desktop](/docs/en/desktop) : visual diff review, parallel sessions, computer use, and Dispatch
- [VS Code](/docs/en/vs-code) : the Claude Code extension inside your editor
- [JetBrains](/docs/en/jetbrains) : the extension for IntelliJ, PyCharm, and other JetBrains IDEs
- [Claude Code on the web](/docs/en/claude-code-on-the-web) : cloud sessions that keep running when you disconnect
- Mobile: the Claude app for [iOS](https://apps.apple.com/us/app/claude-by-anthropic/id6473753684) and [Android](https://play.google.com/store/apps/details?id=com.anthropic.claude) for starting and monitoring tasks while away from your computer

#### Integrations

- [Chrome](/docs/en/chrome) : automate browser tasks with your logged-in sessions
- [Computer use](/docs/en/computer-use) : let Claude open apps and control your screen on macOS
- [GitHub Actions](/docs/en/github-actions) : run Claude in your CI pipeline
- [GitLab CI/CD](/docs/en/gitlab-ci-cd) : the same for GitLab
- [Code Review](/docs/en/code-review) : automatic review on every pull request
- [Slack](/docs/en/slack) : send tasks from team chat, get PRs back

#### Remote access

- [Dispatch](/docs/en/desktop#sessions-from-dispatch) : message a task from your phone and it can spawn a Desktop session
- [Remote Control](/docs/en/remote-control) : drive a running session from your phone or browser
- [Channels](/docs/en/channels) : push events from chat apps or your own servers into a session
- [Scheduled tasks](/docs/en/scheduled-tasks) : run prompts on a recurring schedule

Was this page helpful?

Yes

No

[Best practices](/docs/en/best-practices) [Remote Control](/docs/en/remote-control)

⌘ I


### Continue local sessions from any device with Remote Control


Continue a local Claude Code session from your phone, tablet, or any browser using Remote Control. Works with claude.ai/code and the Claude mobile app.


Remote Control is available on all plans. On Team and Enterprise, it is off by default until an admin enables the Remote Control toggle in [Claude Code admin settings](https://claude.ai/admin-settings/claude-code) .

Remote Control connects [claude.ai/code](https://claude.ai/code) or the Claude app for [iOS](https://apps.apple.com/us/app/claude-by-anthropic/id6473753684) and [Android](https://play.google.com/store/apps/details?id=com.anthropic.claude) to a Claude Code session running on your machine. Start a task at your desk, then pick it up from your phone on the couch or a browser on another computer. When you start a Remote Control session on your machine, Claude keeps running locally the entire time, so nothing moves to the cloud. With Remote Control you can:

- **Use your full local environment remotely** : your filesystem, [MCP servers](/docs/en/mcp) , tools, and project configuration all stay available
- **Work from both surfaces at once** : the conversation stays in sync across all connected devices, so you can send messages from your terminal, browser, and phone interchangeably
- **Survive interruptions** : if your laptop sleeps or your network drops, the session reconnects automatically when your machine comes back online

Unlike [Claude Code on the web](/docs/en/claude-code-on-the-web) , which runs on cloud infrastructure, Remote Control sessions run directly on your machine and interact with your local filesystem. The web and mobile interfaces are just a window into that local session.

Remote Control requires Claude Code v2.1.51 or later. Check your version with `claude --version` .

This page covers setup, how to start and connect to sessions, and how Remote Control compares to Claude Code on the web.

### Requirements

Before using Remote Control, confirm that your environment meets these conditions:

- **Subscription** : available on Pro, Max, Team, and Enterprise plans. API keys are not supported. On Team and Enterprise, an admin must first enable the Remote Control toggle in [Claude Code admin settings](https://claude.ai/admin-settings/claude-code) .
- **Authentication** : run `claude` and use `/login` to sign in through claude.ai if you haven't already.
- **Workspace trust** : run `claude` in your project directory at least once to accept the workspace trust dialog.

### Start a Remote Control session

You can start a Remote Control session from the CLI or the VS Code extension. The CLI offers three invocation modes; VS Code uses the `/remote-control` command.

- Server mode
- Interactive session
- From an existing session
- VS Code

Navigate to your project directory and run:

```
claude remote-control
```

The process stays running in your terminal in server mode, waiting for remote connections. It displays a session URL you can use to [connect from another device](#connect-from-another-device) , and you can press spacebar to show a QR code for quick access from your phone. While a remote session is active, the terminal shows connection status and tool activity. Available flags:

| Flag                                            | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|-------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--name "My Project"`                           | Set a custom session title visible in the session list at claude.ai/code.                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `--remote-control-session-name-prefix <prefix>` | Prefix for auto-generated session names when no explicit name is set. Defaults to your machine's hostname, producing names like `myhost-graceful-unicorn` . Set `CLAUDE_REMOTE_CONTROL_SESSION_NAME_PREFIX` for the same effect.                                                                                                                                                                                                                                                                                                        |
| `--spawn <mode>`                                | How the server creates sessions.  • `same-dir` (default): all sessions share the current working directory, so they can conflict if editing the same files.  • `worktree` : each on-demand session gets its own [git worktree](/docs/en/common-workflows#run-parallel-claude-code-sessions-with-git-worktrees) . Requires a git repository.  • `session` : single-session mode. Serves exactly one session and rejects additional connections. Set at startup only.  Press `w` at runtime to toggle between `same-dir` and `worktree` . |
| `--capacity <N>`                                | Maximum number of concurrent sessions. Default is 32. Cannot be used with `--spawn=session` .                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `--verbose`                                     | Show detailed connection and session logs.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `--sandbox` / `--no-sandbox`                    | Enable or disable [sandboxing](/docs/en/sandboxing) for filesystem and network isolation. Off by default.                                                                                                                                                                                                                                                                                                                                                                                                                               |

To start a normal interactive Claude Code session with Remote Control enabled, use the `--remote-control` flag (or `--rc` ):

```
claude --remote-control
```

Optionally pass a name for the session:

```
claude --remote-control "My Project"
```

This gives you a full interactive session in your terminal that you can also control from claude.ai or the Claude app. Unlike `claude remote-control` (server mode), you can type messages locally while the session is also available remotely.

If you're already in a Claude Code session and want to continue it remotely, use the `/remote-control` (or `/rc` ) command:

```
/remote-control
```

Pass a name as an argument to set a custom session title:

```
/remote-control My Project
```

This starts a Remote Control session that carries over your current conversation history and displays a session URL and QR code you can use to [connect from another device](#connect-from-another-device) . The `--verbose` , `--sandbox` , and `--no-sandbox` flags are not available with this command.

In the [Claude Code VS Code extension](/docs/en/vs-code) , type `/remote-control` or `/rc` in the prompt box, or open the command menu with `/` and select it. Requires Claude Code v2.1.79 or later.

```
/remote-control
```

A banner appears above the prompt box showing connection status. Once connected, click **Open in browser** in the banner to go directly to the session, or find it in the session list at [claude.ai/code](https://claude.ai/code) . The session URL is also posted in the conversation. To disconnect, click the close icon on the banner or run `/remote-control` again. Unlike the CLI, the VS Code command does not accept a name argument or display a QR code. The session title is derived from your conversation history or first prompt.

#### Connect from another device

Once a Remote Control session is active, you have a few ways to connect from another device:

- **Open the session URL** in any browser to go directly to the session on [claude.ai/code](https://claude.ai/code) .
- **Scan the QR code** shown alongside the session URL to open it directly in the Claude app. With `claude remote-control` , press spacebar to toggle the QR code display.
- **Open** [**claude.ai/code**](https://claude.ai/code) **or the Claude app** and find the session by name in the session list. Remote Control sessions show a computer icon with a green status dot when online.

The remote session title is chosen in this order:

1. The name you passed to `--name` , `--remote-control` , or `/remote-control`
2. The title you set with `/rename`
3. The last meaningful message in existing conversation history
4. An auto-generated name like `myhost-graceful-unicorn` , where `myhost` is your machine's hostname or the prefix you set with `--remote-control-session-name-prefix`

If you didn't set an explicit name, the title updates to reflect your prompt once you send one. If the environment already has an active session, you'll be asked whether to continue it or start a new one. If you don't have the Claude app yet, use the `/mobile` command inside Claude Code to display a download QR code for [iOS](https://apps.apple.com/us/app/claude-by-anthropic/id6473753684) or [Android](https://play.google.com/store/apps/details?id=com.anthropic.claude) .

#### Enable Remote Control for all sessions

By default, Remote Control only activates when you explicitly run `claude remote-control` , `claude --remote-control` , or `/remote-control` . To enable it automatically for every interactive session, run `/config` inside Claude Code and set **Enable Remote Control for all sessions** to `true` . Set it back to `false` to disable. With this setting on, each interactive Claude Code process registers one remote session. If you run multiple instances, each one gets its own environment and session. To run multiple concurrent sessions from a single process, use [server mode](#start-a-remote-control-session) instead.

### Connection and security

Your local Claude Code session makes outbound HTTPS requests only and never opens inbound ports on your machine. When you start Remote Control, it registers with the Anthropic API and polls for work. When you connect from another device, the server routes messages between the web or mobile client and your local session over a streaming connection. All traffic travels through the Anthropic API over TLS, the same transport security as any Claude Code session. The connection uses multiple short-lived credentials, each scoped to a single purpose and expiring independently.

### Remote Control vs Claude Code on the web

Remote Control and [Claude Code on the web](/docs/en/claude-code-on-the-web) both use the claude.ai/code interface. The key difference is where the session runs: Remote Control executes on your machine, so your local MCP servers, tools, and project configuration stay available. Claude Code on the web executes in Anthropic-managed cloud infrastructure. Use Remote Control when you're in the middle of local work and want to keep going from another device. Use Claude Code on the web when you want to kick off a task without any local setup, work on a repo you don't have cloned, or run multiple tasks in parallel.

### Limitations

- **One remote session per interactive process** : outside of server mode, each Claude Code instance supports one remote session at a time. Use [server mode](#start-a-remote-control-session) to run multiple concurrent sessions from a single process.
- **Local process must keep running** : Remote Control runs as a local process. If you close the terminal, quit VS Code, or otherwise stop the `claude` process, the session ends.
- **Extended network outage** : if your machine is awake but unable to reach the network for more than roughly 10 minutes, the session times out and the process exits. Run `claude remote-control` again to start a new session.
- **Ultraplan disconnects Remote Control** : starting an [ultraplan](/docs/en/ultraplan) session disconnects any active Remote Control session because both features occupy the claude.ai/code interface and only one can be connected at a time.

### Troubleshooting

#### "Remote Control requires a claude.ai subscription"

You're not authenticated with a claude.ai account. Run `claude auth login` and choose the claude.ai option. If `ANTHROPIC_API_KEY` is set in your environment, unset it first.

#### "Remote Control requires a full-scope login token"

You're authenticated with a long-lived token from `claude setup-token` or the `CLAUDE_CODE_OAUTH_TOKEN` environment variable. These tokens are limited to inference-only and cannot establish Remote Control sessions. Run `claude auth login` to authenticate with a full-scope session token instead.

#### "Unable to determine your organization for Remote Control eligibility"

Your cached account information is stale or incomplete. Run `claude auth login` to refresh it.

#### "Remote Control is not yet enabled for your account"

The eligibility check can fail with certain environment variables present:

- `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` or `DISABLE_TELEMETRY` : unset them and try again.
- `CLAUDE_CODE_USE_BEDROCK` , `CLAUDE_CODE_USE_VERTEX` , or `CLAUDE_CODE_USE_FOUNDRY` : Remote Control requires claude.ai authentication and does not work with third-party providers.

If none of these are set, run `/logout` then `/login` to refresh.

#### "Remote Control is disabled by your organization's policy"

This error has three distinct causes. Run `/status` first to see which login method and subscription you're using.

- **You're authenticated with an API key or Console account** : Remote Control requires claude.ai OAuth. Run `/login` and choose the claude.ai option. If `ANTHROPIC_API_KEY` is set in your environment, unset it.
- **Your Team or Enterprise admin hasn't enabled it** : Remote Control is off by default on these plans. An admin can enable it at [claude.ai/admin-settings/claude-code](https://claude.ai/admin-settings/claude-code) by turning on the **Remote Control** toggle. This is a server-side organization setting, not a [managed settings](/docs/en/permissions#managed-only-settings) key.
- **The admin toggle is grayed out** : your organization has a data retention or compliance configuration that is incompatible with Remote Control. This cannot be changed from the admin panel. Contact Anthropic support to discuss options.

#### "Remote credentials fetch failed"

Claude Code could not obtain a short-lived credential from the Anthropic API to establish the connection. Re-run with `--verbose` to see the full error:

```
claude remote-control --verbose
```

Common causes:

- Not signed in: run `claude` and use `/login` to authenticate with your claude.ai account. API key authentication is not supported for Remote Control.
- Network or proxy issue: a firewall or proxy may be blocking the outbound HTTPS request. Remote Control requires access to the Anthropic API on port 443.
- Session creation failed: if you also see `Session creation failed - see debug log` , the failure happened earlier in setup. Check that your subscription is active.

### Choose the right approach

Claude Code offers several ways to work when you're not at your terminal. They differ in what triggers the work, where Claude runs, and how much you need to set up.

|                                                     | Trigger                                                                                        | Claude runs on                                                                                                           | Setup                                                                                                                                          | Best for                                                      |
|-----------------------------------------------------|------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| [Dispatch](/docs/en/desktop#sessions-from-dispatch) | Message a task from the Claude mobile app                                                      | Your machine (Desktop)                                                                                                   | [Pair the mobile app with Desktop](https://support.claude.com/en/articles/13947068)                                                            | Delegating work while you're away, minimal setup              |
| [Remote Control](/docs/en/remote-control)           | Drive a running session from [claude.ai/code](https://claude.ai/code) or the Claude mobile app | Your machine (CLI or VS Code)                                                                                            | Run `claude remote-control`                                                                                                                    | Steering in-progress work from another device                 |
| [Channels](/docs/en/channels)                       | Push events from a chat app like Telegram or Discord, or your own server                       | Your machine (CLI)                                                                                                       | [Install a channel plugin](/docs/en/channels#quickstart) or [build your own](/docs/en/channels-reference)                                      | Reacting to external events like CI failures or chat messages |
| [Slack](/docs/en/slack)                             | Mention `@Claude` in a team channel                                                            | Anthropic cloud                                                                                                          | [Install the Slack app](/docs/en/slack#setting-up-claude-code-in-slack) with [Claude Code on the web](/docs/en/claude-code-on-the-web) enabled | PRs and reviews from team chat                                |
| [Scheduled tasks](/docs/en/scheduled-tasks)         | Set a schedule                                                                                 | [CLI](/docs/en/scheduled-tasks) , [Desktop](/docs/en/desktop-scheduled-tasks) , or [cloud](/docs/en/web-scheduled-tasks) | Pick a frequency                                                                                                                               | Recurring automation like daily reviews                       |

### Related resources

- [Claude Code on the web](/docs/en/claude-code-on-the-web) : run sessions in Anthropic-managed cloud environments instead of on your machine
- [Ultraplan](/docs/en/ultraplan) : launch a cloud planning session from your terminal and review the plan in your browser
- [Channels](/docs/en/channels) : forward Telegram, Discord, or iMessage into a session so Claude reacts to messages while you're away
- [Dispatch](/docs/en/desktop#sessions-from-dispatch) : message a task from your phone and it can spawn a Desktop session to handle it
- [Authentication](/docs/en/authentication) : set up `/login` and manage credentials for claude.ai
- [CLI reference](/docs/en/cli-reference) : full list of flags and commands including `claude remote-control`
- [Security](/docs/en/security) : how Remote Control sessions fit into the Claude Code security model
- [Data usage](/docs/en/data-usage) : what data flows through the Anthropic API during local and remote sessions

Was this page helpful?

Yes

No

[Overview](/docs/en/platforms) [Get started](/docs/en/web-quickstart)

⌘ I


### Let Claude use your computer from the CLI


Enable computer use in the Claude Code CLI so Claude can open apps, click, type, and see your screen on macOS. Test native apps, debug visual issues, and automate GUI-only tools without leaving your terminal.


Computer use is a research preview on macOS that requires a Pro or Max plan. It is not available on Team or Enterprise plans. It requires Claude Code v2.1.85 or later and an interactive session, so it is not available in non-interactive mode with the `-p` flag.

Computer use lets Claude open apps, control your screen, and work on your machine the way you would. From the CLI, Claude can compile a Swift app, launch it, click through every button, and screenshot the result, all in the same conversation where it wrote the code. This page covers how computer use works in the CLI. For the Desktop app on macOS or Windows, see [computer use in Desktop](/docs/en/desktop#let-claude-use-your-computer) .

### What you can do with computer use

Computer use handles tasks that require a GUI: anything you'd normally have to leave the terminal and do by hand.

- **Build and validate native apps** : ask Claude to build a macOS menu bar app. Claude writes the Swift, compiles it, launches it, and clicks through every control to verify it works before you ever open it.
- **End-to-end UI testing** : point Claude at a local Electron app and say "test the onboarding flow." Claude opens the app, clicks through signup, and screenshots each step. No Playwright config, no test harness.
- **Debug visual and layout issues** : tell Claude "the modal is clipping on small windows." Claude resizes the window, reproduces the bug, screenshots it, patches the CSS, and verifies the fix. Claude sees what you see.
- **Drive GUI-only tools** : interact with design tools, hardware control panels, the iOS Simulator, or proprietary apps that have no CLI or API.

### When computer use applies

Claude has several ways to interact with an app or service. Computer use is the broadest and slowest, so Claude tries the most precise tool first:

- If you have an [MCP server](/docs/en/mcp) for the service, Claude uses that.
- If the task is a shell command, Claude uses Bash.
- If the task is browser work and you have [Claude in Chrome](/docs/en/chrome) set up, Claude uses that.
- If none of those apply, Claude uses computer use.

Screen control is reserved for things nothing else can reach: native apps, simulators, and tools without an API.

### Enable computer use

Computer use is available as a built-in MCP server called `computer-use` . It's off by default until you enable it.

1

Open the MCP menu

In an interactive Claude Code session, run:

```
/mcp
```

Find `computer-use` in the server list. It shows as disabled.

2

Enable the server

Select `computer-use` and choose **Enable** . The setting persists per project, so you only do this once for each project where you want computer use.

3

Grant macOS permissions

The first time Claude tries to use your computer, you'll see a prompt to grant two macOS permissions:

- **Accessibility** : lets Claude click, type, and scroll
- **Screen Recording** : lets Claude see what's on your screen

The prompt includes links to open the relevant System Settings pane. Grant both, then select **Try again** in the prompt. macOS may require you to restart Claude Code after granting Screen Recording.

After setup, ask Claude to do something that needs the GUI:

```
Build the app target, launch it, and click through each tab to make
sure nothing crashes. Screenshot any error states you find.
```

### Approve apps per session

Enabling the `computer-use` server doesn't grant Claude access to every app on your machine. The first time Claude needs a specific app in a session, a prompt appears in your terminal showing:

- Which apps Claude wants to control
- Any extra permissions requested, such as clipboard access
- How many other apps will be hidden while Claude works

Choose **Allow for this session** or **Deny** . Approvals last for the current session. You can approve multiple apps at once when Claude requests them together. Apps with broad reach show an extra warning in the prompt so you know what approving them grants:

| Warning                    | Applies to                                                   |
|----------------------------|--------------------------------------------------------------|
| Equivalent to shell access | Terminal, iTerm, VS Code, Warp, and other terminals and IDEs |
| Can read or write any file | Finder                                                       |
| Can change system settings | System Settings                                              |

These apps aren't blocked. The warning lets you decide whether the task warrants that level of access. Claude's level of control also varies by app category: browsers and trading platforms are view-only, terminals and IDEs are click-only, and everything else gets full control. See [app permissions in Desktop](/docs/en/desktop#app-permissions) for the complete tier breakdown.

### How Claude works on your screen

Understanding the flow helps you anticipate what Claude will do and how to intervene.

#### One session at a time

Computer use holds a machine-wide lock while active. If another Claude Code session is already using your computer, new attempts fail with a message telling you which session holds the lock. Finish or exit that session first.

#### Apps are hidden while Claude works

When Claude starts controlling your screen, other visible apps are hidden so Claude interacts with only the approved apps. Your terminal window stays visible and is excluded from screenshots, so you can watch the session and Claude never sees its own output. When Claude finishes the turn, hidden apps are restored automatically.

#### Screenshots are downscaled automatically

Claude Code downscales every screenshot before sending it to the model. You don't need to lower your display resolution or resize windows on Retina or other high-resolution displays. A 16-inch MacBook Pro at native Retina resolution captures at 3456×2234 and downscales to roughly 1372×887, preserving aspect ratio. There is no setting to change the target size. If on-screen text or controls are too small for Claude to read after downscaling, increase their size in the app rather than changing your display resolution.

#### Stop at any time

When Claude acquires the lock, a macOS notification appears: "Claude is using your computer · press Esc to stop." Press `Esc` anywhere to abort the current action immediately, or press `Ctrl+C` in the terminal. Either way, Claude releases the lock, unhides your apps, and returns control to you. A second notification appears when Claude is done.

### Safety and the trust boundary

Unlike the [sandboxed Bash tool](/docs/en/sandboxing) , computer use runs on your actual desktop with access to the apps you approve. Claude checks each action and flags potential prompt injection from on-screen content, but the trust boundary is different. See the [computer use safety guide](https://support.claude.com/en/articles/14128542) for best practices.

The built-in guardrails reduce risk without requiring configuration:

- **Per-app approval** : Claude can only control apps you've approved in the current session.
- **Sentinel warnings** : apps that grant shell, filesystem, or system settings access are flagged before you approve.
- **Terminal excluded from screenshots** : Claude never sees your terminal window, so on-screen prompts in your session can't feed back into the model.
- **Global escape** : the `Esc` key aborts computer use from anywhere, and the key press is consumed so prompt injection can't use it to dismiss dialogs.
- **Lock file** : only one session can control your machine at a time.

### Example workflows

These examples show common ways to combine computer use with coding tasks.

#### Validate a native build

After making changes to a macOS or iOS app, have Claude compile and verify in one pass:

```
Build the MenuBarStats target, launch it, open the preferences window,
and verify the interval slider updates the label. Screenshot the
preferences window when you're done.
```

Claude runs `xcodebuild` , launches the app, interacts with the UI, and reports what it finds.

#### Reproduce a layout bug

When a visual bug only appears at certain window sizes, let Claude find it:

```
The settings modal clips its footer on narrow windows. Resize the app
window down until you can reproduce it, screenshot the clipped state,
then check the CSS for the modal container.
```

Claude resizes the window, captures the broken state, and reads the relevant stylesheets.

#### Test a simulator flow

Drive the iOS Simulator without writing XCTest:

```
Open the iOS Simulator, launch the app, tap through the onboarding
screens, and tell me if any screen takes more than a second to load.
```

Claude controls the simulator the same way you would with a mouse.

### Differences from the Desktop app

The CLI and Desktop surfaces share the same computer use engine, with a few differences:

| Feature              | Desktop                                                      | CLI                             |
|----------------------|--------------------------------------------------------------|---------------------------------|
| Platforms            | macOS and Windows                                            | macOS only                      |
| Enable               | Toggle in **Settings > General** (under **Desktop app** ) | Enable `computer-use` in `/mcp` |
| Denied apps list     | Configurable in Settings                                     | Not yet available               |
| Auto-unhide toggle   | Optional                                                     | Always on                       |
| Dispatch integration | Dispatch-spawned sessions can use computer use               | Not applicable                  |

### Troubleshooting

#### "Computer use is in use by another Claude session"

Another Claude Code session holds the lock. Finish the task in that session or exit it. If the other session crashed, the lock is released automatically when Claude detects the process is no longer running.

#### macOS permissions prompt keeps reappearing

macOS sometimes requires a restart of the requesting process after you grant Screen Recording. Quit Claude Code completely and start a new session. If the prompt persists, open **System Settings > Privacy & Security > Screen Recording** and confirm your terminal app is listed and enabled.

#### computer-use doesn't appear in /mcp

The server only appears on eligible setups. Check that:

- You're on macOS. Computer use in the CLI is not available on Linux or Windows. On Windows, use [computer use in Desktop](/docs/en/desktop#let-claude-use-your-computer) instead.
- You're running Claude Code v2.1.85 or later. Run `claude --version` to check.
- You're on a Pro or Max plan. Run `/status` to confirm your subscription.
- You're authenticated through claude.ai. Computer use is not available with third-party providers like Amazon Bedrock, Google Cloud Vertex AI, or Microsoft Foundry. If you access Claude exclusively through a third-party provider, you need a separate claude.ai account to use this feature.
- You're in an interactive session. Computer use is not available in non-interactive mode with the `-p` flag.

### See also

- [Computer use in Desktop](/docs/en/desktop#let-claude-use-your-computer) : the same capability with a graphical settings page
- [Claude in Chrome](/docs/en/chrome) : browser automation for web-based tasks
- [MCP](/docs/en/mcp) : connect Claude to structured tools and APIs
- [Sandboxing](/docs/en/sandboxing) : how Claude's Bash tool isolates filesystem and network access
- [Computer use safety guide](https://support.claude.com/en/articles/14128542) : best practices for safe computer use

Was this page helpful?

Yes

No

[Chrome extension (beta)](/docs/en/chrome) [Visual Studio Code](/docs/en/vs-code)

⌘ I


### Use Claude Code in VS Code


Install and configure the Claude Code extension for VS Code. Get AI coding assistance with inline diffs, @-mentions, plan review, and keyboard shortcuts.


VS Code editor with the Claude Code extension panel open on the right side, showing a conversation with Claude


The VS Code extension provides a native graphical interface for Claude Code, integrated directly into your IDE. This is the recommended way to use Claude Code in VS Code. With the extension, you can review and edit Claude's plans before accepting them, auto-accept edits as they're made, @-mention files with specific line ranges from your selection, access conversation history, and open multiple conversations in separate tabs or windows.

### Prerequisites

Before installing, make sure you have:

- VS Code 1.98.0 or higher
- An Anthropic account (you'll sign in when you first open the extension). If you're using a third-party provider like Amazon Bedrock or Google Vertex AI, see [Use third-party providers](#use-third-party-providers) instead.

The extension includes the CLI (command-line interface), which you can access from VS Code's integrated terminal for advanced features. See [VS Code extension vs. Claude Code CLI](#vs-code-extension-vs-claude-code-cli) for details.

### Install the extension

Click the link for your IDE to install directly:

- [Install for VS Code](vscode:extension/anthropic.claude-code)
- [Install for Cursor](cursor:extension/anthropic.claude-code)

Or in VS Code, press `Cmd+Shift+X` (Mac) or `Ctrl+Shift+X` (Windows/Linux) to open the Extensions view, search for "Claude Code", and click **Install** .

If the extension doesn't appear after installation, restart VS Code or run "Developer: Reload Window" from the Command Palette.

### Get started

Once installed, you can start using Claude Code through the VS Code interface:

1

Open the Claude Code panel

Throughout VS Code, the Spark icon indicates Claude Code:

Spark icon


The quickest way to open Claude is to click the Spark icon in the **Editor Toolbar** (top-right corner of the editor). The icon only appears when you have a file open.

VS Code editor showing the Spark icon in the Editor Toolbar


Other ways to open Claude Code:

- **Activity Bar** : click the Spark icon in the left sidebar to open the sessions list. Click any session to open it as a full editor tab, or start a new one. This icon is always visible in the Activity Bar.
- **Command Palette** : `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux), type "Claude Code", and select an option like "Open in New Tab"
- **Status Bar** : click **✱ Claude Code** in the bottom-right corner of the window. This works even when no file is open.

You can drag the Claude panel to reposition it anywhere in VS Code. See [Customize your workflow](#customize-your-workflow) for details.

2

Sign in

The first time you open the panel, a sign-in screen appears. Click **Sign in** and complete authorization in your browser. If you see **Not logged in · Please run /login** later, the extension reopens the sign-in screen automatically. If it doesn't appear, reload the window from the Command Palette with **Developer: Reload Window** . If you have `ANTHROPIC_API_KEY` set in your shell but still see the sign-in prompt, VS Code may not have inherited your shell environment. Launch VS Code from a terminal with `code .` so it inherits your environment variables, or sign in with your Claude account instead. After you sign in, a **Learn Claude Code** checklist appears. Work through each item by clicking **Show me** , or dismiss it with the X. To reopen it later, uncheck **Hide Onboarding** in VS Code settings under Extensions → Claude Code.

3

Send a prompt

Ask Claude to help with your code or files, whether that's explaining how something works, debugging an issue, or making changes.

Claude automatically sees your selected text. Press `Option+K` (Mac) / `Alt+K` (Windows/Linux) to also insert an @-mention reference (like `@file.ts#5-10` ) into your prompt.

Here's an example of asking about a particular line in a file:

VS Code editor with lines 2-3 selected in a Python file, and the Claude Code panel showing a question about those lines with an @-mention reference


4

Review changes

When Claude wants to edit a file, it shows a side-by-side comparison of the original and proposed changes, then asks for permission. You can accept, reject, or tell Claude what to do instead.

VS Code showing a diff of Claude's proposed changes with a permission prompt asking whether to make the edit


For more ideas on what you can do with Claude Code, see [Common workflows](/docs/en/common-workflows) .

Run "Claude Code: Open Walkthrough" from the Command Palette for a guided tour of the basics.

### Use the prompt box

The prompt box supports several features:

- **Permission modes** : click the mode indicator at the bottom of the prompt box to switch modes. In normal mode, Claude asks permission before each action. In Plan mode, Claude describes what it will do and waits for approval before making changes. VS Code automatically opens the plan as a full markdown document where you can add inline comments to give feedback before Claude begins. In auto-accept mode, Claude makes edits without asking. Set the default in VS Code settings under `claudeCode.initialPermissionMode` .
- **Command menu** : click `/` or type `/` to open the command menu. Options include attaching files, switching models, toggling extended thinking, viewing plan usage ( `/usage` ), and starting a [Remote Control](/docs/en/remote-control) session ( `/remote-control` ). The Customize section provides access to MCP servers, hooks, memory, permissions, and plugins. Items with a terminal icon open in the integrated terminal.
- **Context indicator** : the prompt box shows how much of Claude's context window you're using. Claude automatically compacts when needed, or you can run `/compact` manually.
- **Extended thinking** : lets Claude spend more time reasoning through complex problems. Toggle it on via the command menu ( `/` ). See [Extended thinking](/docs/en/common-workflows#use-extended-thinking-thinking-mode) for details.
- **Multi-line input** : press `Shift+Enter` to add a new line without sending. This also works in the "Other" free-text input of question dialogs.

#### Reference files and folders

Use @-mentions to give Claude context about specific files or folders. When you type `@` followed by a file or folder name, Claude reads that content and can answer questions about it or make changes to it. Claude Code supports fuzzy matching, so you can type partial names to find what you need:

```
> Explain the logic in @auth (fuzzy matches auth.js, AuthService.ts, etc.)
> What's in @src/components/ (include a trailing slash for folders)
```

For large PDFs, you can ask Claude to read specific pages instead of the whole file: a single page, a range like pages 1-10, or an open-ended range like page 3 onward. When you select text in the editor, Claude can see your highlighted code automatically. The prompt box footer shows how many lines are selected. Press `Option+K` (Mac) / `Alt+K` (Windows/Linux) to insert an @-mention with the file path and line numbers (e.g., `@app.ts#5-10` ). Click the selection indicator to toggle whether Claude can see your highlighted text - the eye-slash icon means the selection is hidden from Claude. You can also hold `Shift` while dragging files into the prompt box to add them as attachments. Click the X on any attachment to remove it from context.

#### Resume past conversations

Click the dropdown at the top of the Claude Code panel to access your conversation history. You can search by keyword or browse by time (Today, Yesterday, Last 7 days, etc.). Click any conversation to resume it with the full message history. New sessions receive AI-generated titles based on your first message. Hover over a session to reveal rename and remove actions: rename to give it a descriptive title, or remove to delete it from the list. For more on resuming sessions, see [Common workflows](/docs/en/common-workflows#resume-previous-conversations) .

#### Resume remote sessions from Claude.ai

If you use [Claude Code on the web](/docs/en/claude-code-on-the-web) , you can resume those remote sessions directly in VS Code. This requires signing in with **Claude.ai Subscription** , not Anthropic Console.

1

Open Past Conversations

Click the **Past Conversations** dropdown at the top of the Claude Code panel.

2

Select the Remote tab

The dialog shows two tabs: Local and Remote. Click **Remote** to see sessions from claude.ai.

3

Select a session to resume

Browse or search your remote sessions. Click any session to download it and continue the conversation locally.

Only web sessions started with a GitHub repository appear in the Remote tab. Resuming loads the conversation history locally; changes are not synced back to claude.ai.

### Customize your workflow

Once you're up and running, you can reposition the Claude panel, run multiple sessions, or switch to terminal mode.

#### Choose where Claude lives

You can drag the Claude panel to reposition it anywhere in VS Code. Grab the panel's tab or title bar and drag it to:

- **Secondary sidebar** : the right side of the window. Keeps Claude visible while you code.
- **Primary sidebar** : the left sidebar with icons for Explorer, Search, etc.
- **Editor area** : opens Claude as a tab alongside your files. Useful for side tasks.

Use the sidebar for your main Claude session and open additional tabs for side tasks. Claude remembers your preferred location. The Activity Bar sessions list icon is separate from the Claude panel: the sessions list is always visible in the Activity Bar, while the Claude panel icon only appears there when the panel is docked to the left sidebar.

#### Run multiple conversations

Use **Open in New Tab** or **Open in New Window** from the Command Palette to start additional conversations. Each conversation maintains its own history and context, allowing you to work on different tasks in parallel. When using tabs, a small colored dot on the spark icon indicates status: blue means a permission request is pending, orange means Claude finished while the tab was hidden.

#### Switch to terminal mode

By default, the extension opens a graphical chat panel. If you prefer the CLI-style interface, open the [Use Terminal setting](vscode://settings/claudeCode.useTerminal) and check the box. You can also open VS Code settings ( `Cmd+,` on Mac or `Ctrl+,` on Windows/Linux), go to Extensions → Claude Code, and check **Use Terminal** .

### Manage plugins

The VS Code extension includes a graphical interface for installing and managing [plugins](/docs/en/plugins) . Type `/plugins` in the prompt box to open the **Manage plugins** interface.

#### Install plugins

The plugin dialog shows two tabs: **Plugins** and **Marketplaces** . In the Plugins tab:

- **Installed plugins** appear at the top with toggle switches to enable or disable them
- **Available plugins** from your configured marketplaces appear below
- Search to filter plugins by name or description
- Click **Install** on any available plugin

When you install a plugin, choose the installation scope:

- **Install for you** : available in all your projects (user scope)
- **Install for this project** : shared with project collaborators (project scope)
- **Install locally** : only for you, only in this repository (local scope)

#### Manage marketplaces

Switch to the **Marketplaces** tab to add or remove plugin sources:

- Enter a GitHub repo, URL, or local path to add a new marketplace
- Click the refresh icon to update a marketplace's plugin list
- Click the trash icon to remove a marketplace

After making changes, a banner prompts you to restart Claude Code to apply the updates.

Plugin management in VS Code uses the same CLI commands under the hood. Plugins and marketplaces you configure in the extension are also available in the CLI, and vice versa.

For more about the plugin system, see [Plugins](/docs/en/plugins) and [Plugin marketplaces](/docs/en/plugin-marketplaces) .

### Automate browser tasks with Chrome

Connect Claude to your Chrome browser to test web apps, debug with console logs, and automate browser workflows without leaving VS Code. This requires the [Claude in Chrome extension](https://chromewebstore.google.com/detail/claude/fcoeoabgfenejglbffodgkkbkcdhcgfn) version 1.0.36 or higher. Type `@browser` in the prompt box followed by what you want Claude to do:

```
@browser go to localhost:3000 and check the console for errors
```

You can also open the attachment menu to select specific browser tools like opening a new tab or reading page content. Claude opens new tabs for browser tasks and shares your browser's login state, so it can access any site you're already signed into. For setup instructions, the full list of capabilities, and troubleshooting, see [Use Claude Code with Chrome](/docs/en/chrome) .

### VS Code commands and shortcuts

Open the Command Palette ( `Cmd+Shift+P` on Mac or `Ctrl+Shift+P` on Windows/Linux) and type "Claude Code" to see all available VS Code commands for the Claude Code extension. Some shortcuts depend on which panel is "focused" (receiving keyboard input). When your cursor is in a code file, the editor is focused. When your cursor is in Claude's prompt box, Claude is focused. Use `Cmd+Esc` / `Ctrl+Esc` to toggle between them.

These are VS Code commands for controlling the extension. Not all built-in Claude Code commands are available in the extension. See [VS Code extension vs. Claude Code CLI](#vs-code-extension-vs-claude-code-cli) for details.

| Command                    | Shortcut                                                 | Description                                                                                               |
|----------------------------|----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| Focus Input                | `Cmd+Esc` (Mac) / `Ctrl+Esc` (Windows/Linux)             | Toggle focus between editor and Claude                                                                    |
| Open in Side Bar           | -                                                        | Open Claude in the left sidebar                                                                           |
| Open in Terminal           | -                                                        | Open Claude in terminal mode                                                                              |
| Open in New Tab            | `Cmd+Shift+Esc` (Mac) / `Ctrl+Shift+Esc` (Windows/Linux) | Open a new conversation as an editor tab                                                                  |
| Open in New Window         | -                                                        | Open a new conversation in a separate window                                                              |
| New Conversation           | `Cmd+N` (Mac) / `Ctrl+N` (Windows/Linux)                 | Start a new conversation. Requires Claude to be focused and `enableNewConversationShortcut` set to `true` |
| Insert @-Mention Reference | `Option+K` (Mac) / `Alt+K` (Windows/Linux)               | Insert a reference to the current file and selection (requires editor to be focused)                      |
| Show Logs                  | -                                                        | View extension debug logs                                                                                 |
| Logout                     | -                                                        | Sign out of your Anthropic account                                                                        |

#### Launch a VS Code tab from other tools

The extension registers a URI handler at `vscode://anthropic.claude-code/open` . Use it to open a new Claude Code tab from your own tooling: a shell alias, a browser bookmarklet, or any script that can open a URL. If VS Code isn't already running, opening the URL launches it first. If VS Code is already running, the URL opens in whichever window is currently focused. Invoke the handler with your operating system's URL opener. On macOS:

```
open "vscode://anthropic.claude-code/open"
```

Use `xdg-open` on Linux or `start` on Windows. The handler accepts two optional query parameters:

| Parameter   | Description                                                                                                                                                                                                                                                                                                                                                                          |
|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `prompt`    | Text to pre-fill in the prompt box. Must be URL-encoded. The prompt is pre-filled but not submitted automatically.                                                                                                                                                                                                                                                                   |
| `session`   | A session ID to resume instead of starting a new conversation. The session must belong to the workspace currently open in VS Code. If the session isn't found, a fresh conversation starts instead. If the session is already open in a tab, that tab is focused. To capture a session ID programmatically, see [Continue conversations](/docs/en/headless#continue-conversations) . |

For example, to open a tab pre-filled with "review my changes":

```
vscode://anthropic.claude-code/open?prompt=review%20my%20changes
```

### Configure settings

The extension has two types of settings:

- **Extension settings** in VS Code: control the extension's behavior within VS Code. Open with `Cmd+,` (Mac) or `Ctrl+,` (Windows/Linux), then go to Extensions → Claude Code. You can also type `/` and select **General Config** to open settings.
- **Claude Code settings** in `~/.claude/settings.json` : shared between the extension and CLI. Use for allowed commands, environment variables, hooks, and MCP servers. See [Settings](/docs/en/settings) for details.

Add `"$schema": "https://json.schemastore.org/claude-code-settings.json"` to your `settings.json` to get autocomplete and inline validation for all available settings directly in VS Code.

#### Extension settings

| Setting                           | Default   | Description                                                                                                                                                                                                                                                                                                                                                                    |
|-----------------------------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `useTerminal`                     | `false`   | Launch Claude in terminal mode instead of graphical panel                                                                                                                                                                                                                                                                                                                      |
| `initialPermissionMode`           | `default` | Controls approval prompts for new conversations: `default` , `plan` , `acceptEdits` , or `bypassPermissions` . See [permission modes](/docs/en/permission-modes) .                                                                                                                                                                                                             |
| `preferredLocation`               | `panel`   | Where Claude opens: `sidebar` (right) or `panel` (new tab)                                                                                                                                                                                                                                                                                                                     |
| `autosave`                        | `true`    | Auto-save files before Claude reads or writes them                                                                                                                                                                                                                                                                                                                             |
| `useCtrlEnterToSend`              | `false`   | Use Ctrl/Cmd+Enter instead of Enter to send prompts                                                                                                                                                                                                                                                                                                                            |
| `enableNewConversationShortcut`   | `false`   | Enable Cmd/Ctrl+N to start a new conversation                                                                                                                                                                                                                                                                                                                                  |
| `hideOnboarding`                  | `false`   | Hide the onboarding checklist (graduation cap icon)                                                                                                                                                                                                                                                                                                                            |
| `respectGitIgnore`                | `true`    | Exclude .gitignore patterns from file searches                                                                                                                                                                                                                                                                                                                                 |
| `usePythonEnvironment`            | `true`    | Activate the workspace's Python environment when running Claude. Requires the Python extension.                                                                                                                                                                                                                                                                                |
| `environmentVariables`            | `[]`      | Set environment variables for the Claude process. Use Claude Code settings instead for shared config.                                                                                                                                                                                                                                                                          |
| `disableLoginPrompt`              | `false`   | Skip authentication prompts (for third-party provider setups)                                                                                                                                                                                                                                                                                                                  |
| `allowDangerouslySkipPermissions` | `false`   | Adds [Auto mode](/docs/en/permission-modes#eliminate-prompts-with-auto-mode) and Bypass permissions to the mode selector. Auto mode has [plan, admin, model, and provider requirements](/docs/en/permission-modes#eliminate-prompts-with-auto-mode) , so it may remain unavailable even with this toggle on. Use Bypass permissions only in sandboxes with no internet access. |
| `claudeProcessWrapper`            | -         | Executable path used to launch the Claude process                                                                                                                                                                                                                                                                                                                              |

### VS Code extension vs. Claude Code CLI

Claude Code is available as both a VS Code extension (graphical panel) and a CLI (command-line interface in the terminal). Some features are only available in the CLI. If you need a CLI-only feature, run `claude` in VS Code's integrated terminal.

| Feature             | CLI                      | VS Code Extension                                                                    |
|---------------------|--------------------------|--------------------------------------------------------------------------------------|
| Commands and skills | [All](/docs/en/commands) | Subset (type `/` to see available)                                                   |
| MCP server config   | Yes                      | Partial (add servers via CLI; manage existing servers with `/mcp` in the chat panel) |
| Checkpoints         | Yes                      | Yes                                                                                  |
| `!` bash shortcut   | Yes                      | No                                                                                   |
| Tab completion      | Yes                      | No                                                                                   |

#### Rewind with checkpoints

The VS Code extension supports checkpoints, which track Claude's file edits and let you rewind to a previous state. Hover over any message to reveal the rewind button, then choose from three options:

- **Fork conversation from here** : start a new conversation branch from this message while keeping all code changes intact
- **Rewind code to here** : revert file changes back to this point in the conversation while keeping the full conversation history
- **Fork conversation and rewind code** : start a new conversation branch and revert file changes to this point

For full details on how checkpoints work and their limitations, see [Checkpointing](/docs/en/checkpointing) .

#### Run CLI in VS Code

To use the CLI while staying in VS Code, open the integrated terminal ( `Ctrl+`` on Windows/Linux or `Cmd+`` on Mac) and run `claude` . The CLI automatically integrates with your IDE for features like diff viewing and diagnostic sharing. If using an external terminal, run `/ide` inside Claude Code to connect it to VS Code.

#### Switch between extension and CLI

The extension and CLI share the same conversation history. To continue an extension conversation in the CLI, run `claude --resume` in the terminal. This opens an interactive picker where you can search for and select your conversation.

#### Include terminal output in prompts

Reference terminal output in your prompts using `@terminal:name` where `name` is the terminal's title. This lets Claude see command output, error messages, or logs without copy-pasting.

#### Monitor background processes

When Claude runs long-running commands, the extension shows progress in the status bar. However, visibility for background tasks is limited compared to the CLI. For better visibility, have Claude output the command so you can run it in VS Code's integrated terminal.

#### Connect to external tools with MCP

MCP (Model Context Protocol) servers give Claude access to external tools, databases, and APIs. To add an MCP server, open the integrated terminal ( `Ctrl+`` or `Cmd+`` ) and run:

```
claude mcp add --transport http github https://api.githubcopilot.com/mcp/
```

Once configured, ask Claude to use the tools (e.g., "Review PR #456"). To manage MCP servers without leaving VS Code, type `/mcp` in the chat panel. The MCP management dialog lets you enable or disable servers, reconnect to a server, and manage OAuth authentication. See the [MCP documentation](/docs/en/mcp) for available servers.

### Work with git

Claude Code integrates with git to help with version control workflows directly in VS Code. Ask Claude to commit changes, create pull requests, or work across branches.

#### Create commits and pull requests

Claude can stage changes, write commit messages, and create pull requests based on your work:

```
> commit my changes with a descriptive message
> create a pr for this feature
> summarize the changes I've made to the auth module
```

When creating pull requests, Claude generates descriptions based on the actual code changes and can add context about testing or implementation decisions.

#### Use git worktrees for parallel tasks

Use the `--worktree` ( `-w` ) flag to start Claude in an isolated worktree with its own files and branch:

```
claude --worktree feature-auth
```

Each worktree maintains independent file state while sharing git history. This prevents Claude instances from interfering with each other when working on different tasks. For more details, see [Run parallel sessions with Git worktrees](/docs/en/common-workflows#run-parallel-claude-code-sessions-with-git-worktrees) .

### Use third-party providers

By default, Claude Code connects directly to Anthropic's API. If your organization uses Amazon Bedrock, Google Vertex AI, or Microsoft Foundry to access Claude, configure the extension to use your provider instead:

1

Disable login prompt

Open the [Disable Login Prompt setting](vscode://settings/claudeCode.disableLoginPrompt) and check the box. You can also open VS Code settings ( `Cmd+,` on Mac or `Ctrl+,` on Windows/Linux), search for "Claude Code login", and check **Disable Login Prompt** .

2

Configure your provider

Follow the setup guide for your provider:

- [Claude Code on Amazon Bedrock](/docs/en/amazon-bedrock)
- [Claude Code on Google Vertex AI](/docs/en/google-vertex-ai)
- [Claude Code on Microsoft Foundry](/docs/en/microsoft-foundry)

These guides cover configuring your provider in `~/.claude/settings.json` , which ensures your settings are shared between the VS Code extension and the CLI.

### Security and privacy

Your code stays private. Claude Code processes your code to provide assistance but does not use it to train models. For details on data handling and how to opt out of logging, see [Data and privacy](/docs/en/data-usage) . With auto-edit permissions enabled, Claude Code can modify VS Code configuration files (like `settings.json` or `tasks.json` ) that VS Code may execute automatically. To reduce risk when working with untrusted code:

- Enable [VS Code Restricted Mode](https://code.visualstudio.com/docs/editor/workspace-trust#_restricted-mode) for untrusted workspaces
- Use manual approval mode instead of auto-accept for edits
- Review changes carefully before accepting them

#### The built-in IDE MCP server

When the extension is active, it runs a local MCP server that the CLI connects to automatically. This is how the CLI opens diffs in VS Code's native diff viewer, reads your current selection for `@` -mentions, and - when you're working in a Jupyter notebook - asks VS Code to execute cells. The server is named `ide` and is hidden from `/mcp` because there's nothing to configure. If your organization uses a `PreToolUse` hook to allowlist MCP tools, though, you'll need to know it exists. **Transport and authentication.** The server binds to `127.0.0.1` on a random high port and is not reachable from other machines. Each extension activation generates a fresh random auth token that the CLI must present to connect. The token is written to a lock file under `~/.claude/ide/` with `0600` permissions in a `0700` directory, so only the user running VS Code can read it. **Tools exposed to the model.** The server hosts a dozen tools, but only two are visible to the model. The rest are internal RPC the CLI uses for its own UI - opening diffs, reading selections, saving files - and are filtered out before the tool list reaches Claude.

| Tool name (as seen by hooks)   | What it does                                                                                                              | Writes?   |
|--------------------------------|---------------------------------------------------------------------------------------------------------------------------|-----------|
| `mcp__ide__getDiagnostics`     | Returns language-server diagnostics - the errors and warnings in VS Code's Problems panel. Optionally scoped to one file. | No        |
| `mcp__ide__executeCode`        | Runs Python code in the active Jupyter notebook's kernel. See confirmation flow below.                                    | Yes       |

**Jupyter execution always asks first.** `mcp__ide__executeCode` can't run anything silently. On each call, the code is inserted as a new cell at the end of the active notebook, VS Code scrolls it into view, and a native Quick Pick asks you to **Execute** or **Cancel** . Cancelling - or dismissing the picker with `Esc` - returns an error to Claude and nothing runs. The tool also refuses outright when there's no active notebook, when the Jupyter extension ( `ms-toolsai.jupyter` ) isn't installed, or when the kernel isn't Python.

The Quick Pick confirmation is separate from `PreToolUse` hooks. An allowlist entry for `mcp__ide__executeCode` lets Claude *propose* running a cell; the Quick Pick inside VS Code is what lets it *actually* run.

### Fix common issues

#### Extension won't install

- Ensure you have a compatible version of VS Code (1.98.0 or later)
- Check that VS Code has permission to install extensions
- Try installing directly from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=anthropic.claude-code)

#### Spark icon not visible

The Spark icon appears in the **Editor Toolbar** (top-right of editor) when you have a file open. If you don't see it:

1. **Open a file** : The icon requires a file to be open. Having just a folder open isn't enough.
2. **Check VS Code version** : Requires 1.98.0 or higher (Help → About)
3. **Restart VS Code** : Run "Developer: Reload Window" from the Command Palette
4. **Disable conflicting extensions** : Temporarily disable other AI extensions (Cline, Continue, etc.)
5. **Check workspace trust** : The extension doesn't work in Restricted Mode

Alternatively, click "✱ Claude Code" in the **Status Bar** (bottom-right corner). This works even without a file open. You can also use the **Command Palette** ( `Cmd+Shift+P` / `Ctrl+Shift+P` ) and type "Claude Code".

#### Claude Code never responds

If Claude Code isn't responding to your prompts:

1. **Check your internet connection** : Ensure you have a stable internet connection
2. **Start a new conversation** : Try starting a fresh conversation to see if the issue persists
3. **Try the CLI** : Run `claude` from the terminal to see if you get more detailed error messages

If problems persist, [file an issue on GitHub](https://github.com/anthropics/claude-code/issues) with details about the error.

### Uninstall the extension

To uninstall the Claude Code extension:

1. Open the Extensions view ( `Cmd+Shift+X` on Mac or `Ctrl+Shift+X` on Windows/Linux)
2. Search for "Claude Code"
3. Click **Uninstall**

To also remove extension data and reset all settings:

```
rm -rf ~/.vscode/globalStorage/anthropic.claude-code
```

For additional help, see the [troubleshooting guide](/docs/en/troubleshooting) .

### Next steps

Now that you have Claude Code set up in VS Code:

- [Explore common workflows](/docs/en/common-workflows) to get the most out of Claude Code
- [Set up MCP servers](/docs/en/mcp) to extend Claude's capabilities with external tools. Add servers using the CLI, then manage them with `/mcp` in the chat panel.
- [Configure Claude Code settings](/docs/en/settings) to customize allowed commands, hooks, and more. These settings are shared between the extension and CLI.

Was this page helpful?

Yes

No

[Computer use (preview)](/docs/en/computer-use) [JetBrains IDEs](/docs/en/jetbrains)

⌘ I


### JetBrains IDEs


Use Claude Code with JetBrains IDEs including IntelliJ, PyCharm, WebStorm, and more


Claude Code integrates with JetBrains IDEs through a dedicated plugin, providing features like interactive diff viewing, selection context sharing, and more.

### Supported IDEs

The Claude Code plugin works with most JetBrains IDEs, including:

- IntelliJ IDEA
- PyCharm
- Android Studio
- WebStorm
- PhpStorm
- GoLand

### Features

- **Quick launch** : Use `Cmd+Esc` (Mac) or `Ctrl+Esc` (Windows/Linux) to open Claude Code directly from your editor, or click the Claude Code button in the UI
- **Diff viewing** : Code changes can be displayed directly in the IDE diff viewer instead of the terminal
- **Selection context** : The current selection/tab in the IDE is automatically shared with Claude Code
- **File reference shortcuts** : Use `Cmd+Option+K` (Mac) or `Alt+Ctrl+K` (Linux/Windows) to insert file references (for example, @File#L1-99)
- **Diagnostic sharing** : Diagnostic errors (lint, syntax, etc.) from the IDE are automatically shared with Claude as you work

### Installation

#### Marketplace Installation

Find and install the [Claude Code plugin](https://plugins.jetbrains.com/plugin/27310-claude-code-beta-) from the JetBrains marketplace and restart your IDE. If you haven't installed Claude Code yet, see [our quickstart guide](/docs/en/quickstart) for installation instructions.

After installing the plugin, you may need to restart your IDE completely for it to take effect.

### Usage

#### From Your IDE

Run `claude` from your IDE's integrated terminal, and all integration features will be active.

#### From External Terminals

Use the `/ide` command in any external terminal to connect Claude Code to your JetBrains IDE and activate all features:

```
claude
```

```
/ide
```

If you want Claude to have access to the same files as your IDE, start Claude Code from the same directory as your IDE project root.

### Configuration

#### Claude Code Settings

Configure IDE integration through Claude Code's settings:

1. Run `claude`
2. Enter the `/config` command
3. Set the diff tool to `auto` for automatic IDE detection

#### Plugin Settings

Configure the Claude Code plugin by going to **Settings → Tools → Claude Code [Beta]** :

##### General Settings

- **Claude command** : Specify a custom command to run Claude (for example, `claude` , `/usr/local/bin/claude` , or `npx @anthropic-ai/claude-code` )
- **Suppress notification for Claude command not found** : Skip notifications about not finding the Claude command
- **Enable using Option+Enter for multi-line prompts** (macOS only): When enabled, Option+Enter inserts new lines in Claude Code prompts. Disable if experiencing issues with the Option key being captured unexpectedly (requires terminal restart)
- **Enable automatic updates** : Automatically check for and install plugin updates (applied on restart)

For WSL users: Set `wsl -d Ubuntu -- bash -lic "claude"` as your Claude command (replace `Ubuntu` with your WSL distribution name)

##### ESC Key Configuration

If the ESC key doesn't interrupt Claude Code operations in JetBrains terminals:

1. Go to **Settings → Tools → Terminal**
2. Either:
    - Uncheck "Move focus to the editor with Escape", or
    - Click "Configure terminal keybindings" and delete the "Switch focus to Editor" shortcut
3. Apply the changes

This allows the ESC key to properly interrupt Claude Code operations.

### Special Configurations

#### Remote Development

When using JetBrains Remote Development, you must install the plugin in the remote host via **Settings → Plugin (Host)** .

The plugin must be installed on the remote host, not on your local client machine.

#### WSL Configuration

WSL users may need additional configuration for IDE detection to work properly. See our [WSL troubleshooting guide](/docs/en/troubleshooting#jetbrains-ide-not-detected-on-wsl2) for detailed setup instructions.

WSL configuration may require:

- Proper terminal configuration
- Networking mode adjustments
- Firewall settings updates

### Troubleshooting

#### Plugin Not Working

- Ensure you're running Claude Code from the project root directory
- Check that the JetBrains plugin is enabled in the IDE settings
- Completely restart the IDE (you may need to do this multiple times)
- For Remote Development, ensure the plugin is installed in the remote host

#### IDE Not Detected

- Verify the plugin is installed and enabled
- Restart the IDE completely
- Check that you're running Claude Code from the integrated terminal
- For WSL users, see the [WSL troubleshooting guide](/docs/en/troubleshooting#jetbrains-ide-not-detected-on-wsl2)

#### Command Not Found

If clicking the Claude icon shows "command not found":

1. Verify Claude Code is installed: `npm list -g @anthropic-ai/claude-code`
2. Configure the Claude command path in plugin settings
3. For WSL users, use the WSL command format mentioned in the configuration section

### Security Considerations

When Claude Code runs in a JetBrains IDE with auto-edit permissions enabled, it may be able to modify IDE configuration files that can be automatically executed by your IDE. This may increase the risk of running Claude Code in auto-edit mode and allow bypassing Claude Code's permission prompts for bash execution. When running in JetBrains IDEs, consider:

- Using manual approval mode for edits
- Taking extra care to ensure Claude is only used with trusted prompts
- Being aware of which files Claude Code has access to modify

For additional help, see our [troubleshooting guide](/docs/en/troubleshooting) .

Was this page helpful?

Yes

No

[Visual Studio Code](/docs/en/vs-code) [Code Review](/docs/en/code-review)

⌘ I


### Get started with the desktop app


Install Claude Code on desktop and start your first coding session


The desktop app gives you Claude Code with a graphical interface: visual diff review, live app preview, GitHub PR monitoring with auto-merge, parallel sessions with Git worktree isolation, scheduled tasks, and the ability to run tasks remotely. No terminal required. Download Claude for your platform:

### macOS

Universal build for Intel and Apple Silicon

### Windows

For x64 processors

For Windows ARM64, download the [ARM64 installer](https://claude.ai/api/desktop/win32/arm64/setup/latest/redirect?utm_source=claude_code&utm_medium=docs) . Linux is not currently supported.

Claude Code requires a [Pro, Max, Team, or Enterprise subscription](https://claude.com/pricing?utm_source=claude_code&utm_medium=docs&utm_content=desktop_quickstart_pricing) .

This page walks through installing the app and starting your first session. If you're already set up, see [Use Claude Code Desktop](/docs/en/desktop) for the full reference. The desktop app has three tabs:

- **Chat** : General conversation with no file access, similar to claude.ai.
- **Cowork** : An autonomous background agent that works on tasks in a cloud VM with its own environment. It can run independently while you do other work.
- **Code** : An interactive coding assistant with direct access to your local files. You review and approve each change in real time.

Chat and Cowork are covered in the [Claude Desktop support articles](https://support.claude.com/en/collections/16163169-claude-desktop) . This page focuses on the **Code** tab.

### Install

1

Install and sign in

Download Claude for your platform and run the installer:

- [macOS](https://claude.ai/api/desktop/darwin/universal/dmg/latest/redirect?utm_source=claude_code&utm_medium=docs) : universal build for Intel and Apple Silicon
- [Windows x64](https://claude.ai/api/desktop/win32/x64/setup/latest/redirect?utm_source=claude_code&utm_medium=docs) : for x64 processors
- [Windows ARM64](https://claude.ai/api/desktop/win32/arm64/setup/latest/redirect?utm_source=claude_code&utm_medium=docs) : for ARM processors

Launch Claude from your Applications folder (macOS) or Start menu (Windows). Sign in with your Anthropic account.

2

Open the Code tab

Click the **Code** tab at the top center. If clicking Code prompts you to upgrade, you need to [subscribe to a paid plan](https://claude.com/pricing?utm_source=claude_code&utm_medium=docs&utm_content=desktop_quickstart_upgrade) first. If it prompts you to sign in online, complete the sign-in and restart the app. If you see a 403 error, see [authentication troubleshooting](/docs/en/desktop#403-or-authentication-errors-in-the-code-tab) .

The desktop app includes Claude Code. You don't need to install Node.js or the CLI separately. To use `claude` from the terminal, install the CLI separately. See [Get started with the CLI](/docs/en/quickstart) .

### Start your first session

With the Code tab open, choose a project and give Claude something to do.

1

Choose an environment and folder

Select **Local** to run Claude on your machine using your files directly. Click **Select folder** and choose your project directory.

Start with a small project you know well. It's the fastest way to see what Claude Code can do. On Windows, [Git](https://git-scm.com/downloads/win) must be installed for local sessions to work. Most Macs include Git by default.

You can also select:

- **Remote** : Run sessions on Anthropic's cloud infrastructure that continue even if you close the app. Remote sessions use the same infrastructure as [Claude Code on the web](/docs/en/claude-code-on-the-web) .
- **SSH** : Connect to a remote machine over SSH (your own servers, cloud VMs, or dev containers). Claude Code must be installed on the remote machine.

2

Choose a model

Select a model from the dropdown next to the send button. See [models](/docs/en/model-config#available-models) for a comparison of Opus, Sonnet, and Haiku. You cannot change the model after the session starts.

3

Tell Claude what to do

Type what you want Claude to do:

- Find a TODO comment and fix it
- Add tests for the main function
- Create a CLAUDE.md with instructions for this codebase

A [session](/docs/en/desktop#work-in-parallel-with-sessions) is a conversation with Claude about your code. Each session tracks its own context and changes, so you can work on multiple tasks without them interfering with each other.

4

Review and accept changes

By default, the Code tab starts in [Ask permissions mode](/docs/en/desktop#choose-a-permission-mode) , where Claude proposes changes and waits for your approval before applying them. You'll see:

1. A [diff view](/docs/en/desktop#review-changes-with-diff-view) showing exactly what will change in each file
2. Accept/Reject buttons to approve or decline each change
3. Real-time updates as Claude works through your request

If you reject a change, Claude will ask how you'd like to proceed differently. Your files aren't modified until you accept.

### Now what?

You've made your first edit. For the full reference on everything Desktop can do, see [Use Claude Code Desktop](/docs/en/desktop) . Here are some things to try next. **Interrupt and steer.** You can interrupt Claude at any point. If it's going down the wrong path, click the stop button or type your correction and press **Enter** . Claude stops what it's doing and adjusts based on your input. You don't have to wait for it to finish or start over. **Give Claude more context.** Type `@filename` in the prompt box to pull a specific file into the conversation, attach images and PDFs using the attachment button, or drag and drop files directly into the prompt. The more context Claude has, the better the results. See [Add files and context](/docs/en/desktop#add-files-and-context-to-prompts) . **Use skills for repeatable tasks.** Type `/` or click **+** → **Slash commands** to browse [built-in commands](/docs/en/commands) , [custom skills](/docs/en/skills) , and plugin skills. Skills are reusable prompts you can invoke whenever you need them, like code review checklists or deployment steps. **Review changes before committing.** After Claude edits files, a `+12 -1` indicator appears. Click it to open the [diff view](/docs/en/desktop#review-changes-with-diff-view) , review modifications file by file, and comment on specific lines. Claude reads your comments and revises. Click **Review code** to have Claude evaluate the diffs itself and leave inline suggestions. **Adjust how much control you have.** Your [permission mode](/docs/en/desktop#choose-a-permission-mode) controls the balance. Ask permissions (default) requires approval before every edit. Auto accept edits auto-accepts file edits for faster iteration. Plan mode lets Claude map out an approach without touching any files, which is useful before a large refactor. **Add plugins for more capabilities.** Click the **+** button next to the prompt box and select **Plugins** to browse and install [plugins](/docs/en/desktop#install-plugins) that add skills, agents, MCP servers, and more. **Preview your app.** Click the **Preview** dropdown to run your dev server directly in the desktop. Claude can view the running app, test endpoints, inspect logs, and iterate on what it sees. See [Preview your app](/docs/en/desktop#preview-your-app) . **Track your pull request.** After opening a PR, Claude Code monitors CI check results and can automatically fix failures or merge the PR once all checks pass. See [Monitor pull request status](/docs/en/desktop#monitor-pull-request-status) . **Put Claude on a schedule.** Set up [scheduled tasks](/docs/en/desktop-scheduled-tasks) to run Claude automatically on a recurring basis: a daily code review every morning, a weekly dependency audit, or a briefing that pulls from your connected tools. **Scale up when you're ready.** Open [parallel sessions](/docs/en/desktop#work-in-parallel-with-sessions) from the sidebar to work on multiple tasks at once, each in its own Git worktree. Send [long-running work to the cloud](/docs/en/desktop#run-long-running-tasks-remotely) so it continues even if you close the app, or [continue a session on the web or in your IDE](/docs/en/desktop#continue-in-another-surface) if a task takes longer than expected. [Connect external tools](/docs/en/desktop#extend-claude-code) like GitHub, Slack, and Linear to bring your workflow together.

### Coming from the CLI?

Desktop runs the same engine as the CLI with a graphical interface. You can run both simultaneously on the same project, and they share configuration (CLAUDE.md files, MCP servers, hooks, skills, and settings). For a full comparison of features, flag equivalents, and what's not available in Desktop, see [CLI comparison](/docs/en/desktop#coming-from-the-cli) .

### What's next

- [Use Claude Code Desktop](/docs/en/desktop) : permission modes, parallel sessions, diff view, connectors, and enterprise configuration
- [Troubleshooting](/docs/en/desktop#troubleshooting) : solutions to common errors and setup issues
- [Best practices](/docs/en/best-practices) : tips for writing effective prompts and getting the most out of Claude Code
- [Common workflows](/docs/en/common-workflows) : tutorials for debugging, refactoring, testing, and more

Was this page helpful?

Yes

No

[Schedule tasks on the web](/docs/en/web-scheduled-tasks) [Reference](/docs/en/desktop)

⌘ I


### Get started with Claude Code on the web


Run Claude Code in the cloud from your browser or phone. Connect a GitHub repository, submit a task, and review the PR without local setup.


Claude Code on the web is in research preview for Pro, Max, and Team users, and for Enterprise users with premium seats or Chat + Claude Code seats.

Claude Code on the web runs on Anthropic-managed cloud infrastructure instead of your machine. Submit tasks from [claude.ai/code](https://claude.ai/code) in your browser or the Claude mobile app. You'll need a GitHub repository to [get started](#connect-github-and-create-an-environment) . Claude clones it into an isolated virtual machine, makes changes, and pushes a branch for you to review. Sessions persist across devices, so a task you start on your laptop is ready to review from your phone later. Claude Code on the web works well for:

- **Parallel tasks** : run several independent tasks at once, each in its own session and branch, without managing multiple worktrees
- **Repos you don't have locally** : Claude clones the repo fresh every session, so you don't need it checked out
- **Tasks that don't need frequent steering** : submit a well-defined task, do something else, and review the result when Claude is done
- **Code questions and exploration** : understand a codebase or trace how a feature is implemented without a local checkout

For work that needs your local config, tools, or environment, running Claude Code locally or using [Remote Control](/docs/en/remote-control) is a better fit.

### How sessions run

When you submit a task:

1. **Clone and prepare** : your repository is cloned to an Anthropic-managed VM, and your [setup script](/docs/en/claude-code-on-the-web#setup-scripts) runs if configured.
2. **Configure network** : internet access is set based on your environment's [access level](/docs/en/claude-code-on-the-web#access-levels) .
3. **Work** : Claude analyzes code, makes changes, runs tests, and checks its work. You can watch and steer throughout, or step away and come back when it's done.
4. **Push the branch** : when Claude reaches a stopping point, it pushes its branch to GitHub. You review the diff, leave inline comments, create a PR, or send another message to keep going.

The session doesn't close when the branch is pushed. PR creation and further edits all happen within the same conversation.

### Compare ways to run Claude Code

Claude Code behaves the same everywhere. What changes is where code executes and whether your local config is available. The Desktop app offers both local and cloud sessions, so its answers below depend on which you choose:

|                                                   | On the web                                                                                                           | Remote Control               | Terminal CLI           | Desktop app                 |
|---------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|------------------------------|------------------------|-----------------------------|
| **Code runs on**                                  | Anthropic cloud VM                                                                                                   | Your machine                 | Your machine           | Your machine or cloud VM    |
| **You chat from**                                 | claude.ai or mobile app                                                                                              | claude.ai or mobile app      | Your terminal          | The Desktop UI              |
| **Uses your local config**                        | No, repo only                                                                                                        | Yes                          | Yes                    | Yes for local, no for cloud |
| **Requires GitHub**                               | Yes, or [bundle a local repo](/docs/en/claude-code-on-the-web#send-local-repositories-without-github) via `--remote` | No                           | No                     | Only for cloud sessions     |
| **Keeps running if you disconnect**               | Yes                                                                                                                  | While terminal stays open    | No                     | Depends on session type     |
| [**Permission modes**](/docs/en/permission-modes) | Auto accept edits, Plan                                                                                              | Ask, Auto accept edits, Plan | All modes              | Depends on session type     |
| **Network access**                                | Configurable per environment                                                                                         | Your machine's network       | Your machine's network | Depends on session type     |

See the [terminal quickstart](/docs/en/quickstart) , [Desktop app](/docs/en/desktop) , or [Remote Control](/docs/en/remote-control) docs to set those up.

### Connect GitHub and create an environment

Setup is a one-time process. If you already use the GitHub CLI, you can [do this from your terminal](#connect-from-your-terminal) instead of the browser.

1

Visit claude.ai/code

Go to [claude.ai/code](https://claude.ai/code) and sign in with your Anthropic account.

2

Install the Claude GitHub App

After signing in, claude.ai/code prompts you to connect GitHub. Follow the prompt to install the Claude GitHub App and grant it access to your repositories. Cloud sessions work with existing GitHub repositories, so to start a new project, [create an empty repository on GitHub](https://github.com/new) first.

3

Create your environment

After connecting GitHub, you'll be prompted to create a cloud environment. The environment controls what network access Claude has during sessions and what runs when a new session is created. See [Installed tools](/docs/en/claude-code-on-the-web#installed-tools) for what's available without any configuration. The form has these fields:

- **Name** : a display label. Useful when you have multiple environments for different projects or access levels.
- **Network access** : controls what the session can reach on the internet. The default, `Trusted` , allows connections to [common package registries](/docs/en/claude-code-on-the-web#default-allowed-domains) like npm, PyPI, and RubyGems while blocking general internet access.
- **Environment variables** : optional variables available in every session, in `.env` format. Don't wrap values in quotes, since quotes are stored as part of the value. These are visible to anyone who can edit this environment.
- **Setup script** : an optional Bash script that runs before Claude Code launches when a new session is created. Use it to install system tools the cloud VM doesn't include, like `apt install -y gh` , or to start services your project needs. See [Setup scripts](/docs/en/claude-code-on-the-web#setup-scripts) for examples and debugging tips.

For a first project, leave the defaults and click **Create environment** . You can [edit it later or create additional environments](/docs/en/claude-code-on-the-web#configure-your-environment) for different projects.

#### Connect from your terminal

If you already use the GitHub CLI ( `gh` ), you can set up Claude Code on the web without opening a browser. This requires the [Claude Code CLI](/docs/en/quickstart) . `/web-setup` reads your local `gh` token, links it to your Claude account, and creates a default cloud environment if you don't have one.

Organizations with [Zero Data Retention](/docs/en/zero-data-retention) enabled cannot use `/web-setup` or other cloud session features. If the GitHub CLI isn't installed or authenticated, `/web-setup` opens the browser onboarding flow instead.

1

Authenticate with the GitHub CLI

In your shell, authenticate the GitHub CLI if you haven't already:

```
gh auth login
```

2

Sign in to Claude

In the Claude Code CLI, run `/login` to sign in with your claude.ai account. Skip this step if you're already signed in.

3

Run /web-setup

In the Claude Code CLI, run:

```
/web-setup
```

This syncs your `gh` token to your Claude account. If you don't have a cloud environment yet, `/web-setup` creates one with Trusted network access and no setup script. You can [edit the environment or add variables](/docs/en/claude-code-on-the-web#configure-your-environment) afterward. Once `/web-setup` completes, you can start cloud sessions from your terminal with [`--remote`](/docs/en/claude-code-on-the-web#from-terminal-to-web) or set up recurring tasks with [`/schedule`](/docs/en/web-scheduled-tasks) .

### Start a task

With GitHub connected and an environment created, you're ready to submit tasks.

1

Select a repository and branch

From [claude.ai/code](https://claude.ai/code) or the Code tab in the Claude mobile app, click the repository selector below the input box and choose a repository for Claude to work in. Each repository shows a branch selector. Change it to start Claude from a feature branch instead of the default. You can add multiple repositories to work across them in one session.

2

Choose a permission mode

The mode dropdown next to the input defaults to **Auto accept edits** , where Claude makes changes and pushes a branch without stopping for approval. Switch to **Plan mode** if you want Claude to propose an approach and wait for your go-ahead before editing files. Cloud sessions don't offer Ask permissions, Auto mode, or Bypass permissions. See [Permission modes](/docs/en/permission-modes) for the full list.

3

Describe the task and submit

Type a description of what you want and press Enter. Be specific:

- Name the file or function: "Add a README with setup instructions" or "Fix the failing auth test in `tests/test_auth.py` " is better than "fix tests"
- Paste error output if you have it
- Describe the expected behavior, not just the symptom

Claude clones the repositories, runs your setup script if configured, and starts working. Each task gets its own session and its own branch, so you don't need to wait for one to finish before starting another.

### Review and iterate

When Claude finishes, review the changes, leave feedback on specific lines, and keep going until the diff looks right.

1

Open the diff view

A diff indicator shows lines added and removed across the session, for example `+42 -18` . Select it to open the diff view, with a file list on the left and changes on the right.

2

Leave inline comments

Select any line in the diff, type your feedback, and press Enter. Comments queue up until you send your next message, then they're bundled with it. Claude sees "at `src/auth.ts:47` , don't catch the error here" alongside your main instruction, so you don't have to describe where the problem is.

3

Create a pull request

When the diff looks right, select **Create PR** at the top of the diff view. You can open it as a full PR, a draft, or jump to GitHub's compose page with a generated title and description.

4

Keep iterating after the PR

The session stays live after the PR is created. Paste CI failure output or reviewer comments into the chat and ask Claude to address them. To have Claude monitor the PR automatically, see [Auto-fix pull requests](/docs/en/claude-code-on-the-web#auto-fix-pull-requests) .

### Troubleshoot setup

#### No repositories appear after connecting GitHub

The Claude GitHub App needs explicit access to each repository you want to use. On github.com, open **Settings → Applications → Claude → Configure** and verify your repo is listed under **Repository access** . Private repositories need the same authorization as public ones.

#### The page only shows a GitHub login button

Cloud sessions require a connected GitHub account. Connect via the browser flow above, or run `/web-setup` from your terminal if you use the GitHub CLI. If you'd rather not connect GitHub at all, see [Remote Control](/docs/en/remote-control) to run Claude Code on your own machine and monitor it from the web.

#### "Not available for the selected organization"

Enterprise organizations may need an admin to enable Claude Code on the web. Contact your Anthropic account team.

#### /web-setup returns "Unknown command"

`/web-setup` runs inside the Claude Code CLI, not your shell. Launch `claude` first, then type `/web-setup` at the prompt. If you typed it inside Claude Code and still see the error, your CLI is older than v2.1.80 or you're authenticated with an API key or third-party provider instead of a claude.ai subscription. Run `claude update` , then `/login` to sign in with your claude.ai account.

#### "No cloud environment available" when using --remote

You haven't created a cloud environment yet. Run `/web-setup` in the Claude Code CLI to create one, or visit [claude.ai/code](https://claude.ai/code) and follow the **Create your environment** step above.

#### Setup script failed

The setup script exited with a non-zero status, which blocks the session from starting. Common causes:

- A package install failed because the registry isn't in your [network access level](/docs/en/claude-code-on-the-web#access-levels) . `Trusted` covers most package managers; `None` blocks them all.
- The script references a file or path that doesn't exist in a fresh clone.
- A command that works locally needs a different invocation on Ubuntu.

To debug, add `set -x` at the top of the script to see which command failed. For non-critical commands, append `|| true` so they don't block session start.

#### Session keeps running after closing the tab

This is by design. Closing the tab or navigating away doesn't stop the session. It continues running in the background until Claude finishes the current task, then idles. From the sidebar, you can [archive a session](/docs/en/claude-code-on-the-web#archive-sessions) to hide it from your list, or [delete it](/docs/en/claude-code-on-the-web#delete-sessions) to remove it permanently.

### Next steps

Now that you can submit and review tasks, these pages cover what comes next: starting cloud sessions from your terminal, scheduling recurring work, and giving Claude standing instructions.

- [Use Claude Code on the web](/docs/en/claude-code-on-the-web) : the full reference, including teleporting sessions to your terminal, setup scripts, environment variables, and network config
- [Schedule tasks on the web](/docs/en/web-scheduled-tasks) : automate recurring work like daily PR reviews and dependency audits
- [CLAUDE.md](/docs/en/memory) : give Claude persistent instructions and context that load at the start of every session
- Install the Claude mobile app for [iOS](https://apps.apple.com/us/app/claude-by-anthropic/id6473753684) or [Android](https://play.google.com/store/apps/details?id=com.anthropic.claude) to monitor sessions from your phone. From the Claude Code CLI, `/mobile` shows a QR code.

Was this page helpful?

Yes

No

[Remote Control](/docs/en/remote-control) [Reference](/docs/en/claude-code-on-the-web)

⌘ I


### Use Claude Code Desktop


Get more out of Claude Code Desktop: computer use, Dispatch sessions from your phone, parallel sessions with Git isolation, visual diff review, app previews, PR monitoring, connectors, and enterprise configuration.


The Code tab within the Claude Desktop app lets you use Claude Code through a graphical interface instead of the terminal. Desktop adds these capabilities on top of the standard Claude Code experience:

- [Visual diff review](#review-changes-with-diff-view) with inline comments
- [Live app preview](#preview-your-app) with dev servers
- [Computer use](#let-claude-use-your-computer) to open apps and control your screen on macOS and Windows
- [GitHub PR monitoring](#monitor-pull-request-status) with auto-fix and auto-merge
- [Parallel sessions](#work-in-parallel-with-sessions) with automatic Git worktree isolation
- [Dispatch](#sessions-from-dispatch) integration: send a task from your phone, get a session here
- [Scheduled tasks](/docs/en/desktop-scheduled-tasks) that run Claude on a recurring schedule
- [Connectors](#connect-external-tools) for GitHub, Slack, Linear, and more
- Local, [SSH](#ssh-sessions) , and [cloud](#run-long-running-tasks-remotely) environments

New to Desktop? Start with [Get started](/docs/en/desktop-quickstart) to install the app and make your first edit.

This page covers [working with code](#work-with-code) , [computer use](#let-claude-use-your-computer) , [managing sessions](#manage-sessions) , [extending Claude Code](#extend-claude-code) , and [configuration](#environment-configuration) . It also includes a [CLI comparison](#coming-from-the-cli) and [troubleshooting](#troubleshooting) .

### Start a session

Before you send your first message, configure four things in the prompt area:

- **Environment** : choose where Claude runs. Select **Local** for your machine, **Remote** for Anthropic-hosted cloud sessions, or an [**SSH connection**](#ssh-sessions) for a remote machine you manage. See [environment configuration](#environment-configuration) .
- **Project folder** : select the folder or repository Claude works in. For remote sessions, you can add [multiple repositories](#run-long-running-tasks-remotely) .
- **Model** : pick a [model](/docs/en/model-config#available-models) from the dropdown next to the send button. The model is locked once the session starts.
- **Permission mode** : choose how much autonomy Claude has from the [mode selector](#choose-a-permission-mode) . You can change this during the session.

Type your task and press **Enter** to start. Each session tracks its own context and changes independently.

### Work with code

Give Claude the right context, control how much it does on its own, and review what it changed.

#### Use the prompt box

Type what you want Claude to do and press **Enter** to send. Claude reads your project files, makes changes, and runs commands based on your [permission mode](#choose-a-permission-mode) . You can interrupt Claude at any point: click the stop button or type your correction and press **Enter** . Claude stops what it's doing and adjusts based on your input. The **+** button next to the prompt box gives you access to file attachments, [skills](#use-skills) , [connectors](#connect-external-tools) , and [plugins](#install-plugins) .

#### Add files and context to prompts

The prompt box supports two ways to bring in external context:

- **@mention files** : type `@` followed by a filename to add a file to the conversation context. Claude can then read and reference that file. @mention is not available in remote sessions.
- **Attach files** : attach images, PDFs, and other files to your prompt using the attachment button, or drag and drop files directly into the prompt. This is useful for sharing screenshots of bugs, design mockups, or reference documents.

#### Choose a permission mode

Permission modes control how much autonomy Claude has during a session: whether it asks before editing files, running commands, or both. You can switch modes at any time using the mode selector next to the send button. Start with Ask permissions to see exactly what Claude does, then move to Auto accept edits or Plan mode as you get comfortable.

| Mode                   | Settings key        | Behavior                                                                                                                                                                                                                                                                                                              |
|------------------------|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Ask permissions**    | `default`           | Claude asks before editing files or running commands. You see a diff and can accept or reject each change. Recommended for new users.                                                                                                                                                                                 |
| **Auto accept edits**  | `acceptEdits`       | Claude auto-accepts file edits and common filesystem commands like `mkdir` , `touch` , and `mv` , but still asks before running other terminal commands. Use this when you trust file changes and want faster iteration.                                                                                              |
| **Plan mode**          | `plan`              | Claude reads files and runs commands to explore, then proposes a plan without editing your source code. Good for complex tasks where you want to review the approach first.                                                                                                                                           |
| **Auto**               | `auto`              | Claude executes all actions with background safety checks that verify alignment with your request. Reduces permission prompts while maintaining oversight. Currently a research preview. Available on Team, Enterprise, and API plans. Requires Claude Sonnet 4.6 or Opus 4.6. Enable in your Settings → Claude Code. |
| **Bypass permissions** | `bypassPermissions` | Claude runs without any permission prompts, equivalent to `--dangerously-skip-permissions` in the CLI. Enable in your Settings → Claude Code under "Allow bypass permissions mode". Only use this in sandboxed containers or VMs. Enterprise admins can disable this option.                                          |

The `dontAsk` permission mode is available only in the [CLI](/docs/en/permission-modes#allow-only-pre-approved-tools-with-dontask-mode) .

Start complex tasks in Plan mode so Claude maps out an approach before making changes. Once you approve the plan, switch to Auto accept edits or Ask permissions to execute it. See [explore first, then plan, then code](/docs/en/best-practices#explore-first-then-plan-then-code) for more on this workflow.

Remote sessions support Auto accept edits and Plan mode. Ask permissions is not available because remote sessions auto-accept file edits by default, and Bypass permissions is not available because the remote environment is already sandboxed. Enterprise admins can restrict which permission modes are available. See [enterprise configuration](#enterprise-configuration) for details.

#### Preview your app

Claude can start a dev server and open an embedded browser to verify its changes. This works for frontend web apps as well as backend servers: Claude can test API endpoints, view server logs, and iterate on issues it finds. In most cases, Claude starts the server automatically after editing project files. You can also ask Claude to preview at any time. By default, Claude [auto-verifies](#auto-verify-changes) changes after every edit. From the preview panel, you can:

- Interact with your running app directly in the embedded browser
- Watch Claude verify its own changes automatically: it takes screenshots, inspects the DOM, clicks elements, fills forms, and fixes issues it finds
- Start or stop servers from the **Preview** dropdown in the session toolbar
- Persist cookies and local storage across server restarts by selecting **Persist sessions** in the dropdown, so you don't have to re-login during development
- Edit the server configuration or stop all servers at once

Claude creates the initial server configuration based on your project. If your app uses a custom dev command, edit `.claude/launch.json` to match your setup. See [Configure preview servers](#configure-preview-servers) for the full reference. To clear saved session data, toggle **Persist preview sessions** off in Settings → Claude Code. To disable preview entirely, toggle off **Preview** in Settings → Claude Code.

#### Review changes with diff view

After Claude makes changes to your code, the diff view lets you review modifications file by file before creating a pull request. When Claude changes files, a diff stats indicator appears showing the number of lines added and removed, such as `+12 -1` . Click this indicator to open the diff viewer, which displays a file list on the left and the changes for each file on the right. To comment on specific lines, click any line in the diff to open a comment box. Type your feedback and press **Enter** to add the comment. After adding comments to multiple lines, submit all comments at once:

- **macOS** : press **Cmd+Enter**
- **Windows** : press **Ctrl+Enter**

Claude reads your comments and makes the requested changes, which appear as a new diff you can review.

#### Review your code

In the diff view, click **Review code** in the top-right toolbar to ask Claude to evaluate the changes before you commit. Claude examines the current diffs and leaves comments directly in the diff view. You can respond to any comment or ask Claude to revise. The review focuses on high-signal issues: compile errors, definite logic errors, security vulnerabilities, and obvious bugs. It does not flag style, formatting, pre-existing issues, or anything a linter would catch.

#### Monitor pull request status

After you open a pull request, a CI status bar appears in the session. Claude Code uses the GitHub CLI to poll check results and surface failures.

- **Auto-fix** : when enabled, Claude automatically attempts to fix failing CI checks by reading the failure output and iterating.
- **Auto-merge** : when enabled, Claude merges the PR once all checks pass. The merge method is squash. Auto-merge must be [enabled in your GitHub repository settings](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-auto-merge-for-pull-requests-in-your-repository) for this to work.

Use the **Auto-fix** and **Auto-merge** toggles in the CI status bar to enable either option. Claude Code also sends a desktop notification when CI finishes.

PR monitoring requires the [GitHub CLI (](https://cli.github.com/) [`gh`](https://cli.github.com/) [)](https://cli.github.com/) to be installed and authenticated on your machine. If `gh` is not installed, Desktop prompts you to install it the first time you try to create a PR.

### Let Claude use your computer

Computer use lets Claude open your apps, control your screen, and work directly on your machine the way you would. Ask Claude to test a native app in a mobile simulator, interact with a desktop tool that has no CLI, or automate something that only works through a GUI.

Computer use is a research preview on macOS and Windows that requires a Pro or Max plan. It is not available on Team or Enterprise plans. The Claude Desktop app must be running.

Computer use is off by default. [Enable it in Settings](#enable-computer-use) before Claude can control your screen. On macOS, you also need to grant Accessibility and Screen Recording permissions.

Unlike the [sandboxed Bash tool](/docs/en/sandboxing) , computer use runs on your actual desktop with access to whatever you approve. Claude checks each action and flags potential prompt injection from on-screen content, but the trust boundary is different. See the [computer use safety guide](https://support.claude.com/en/articles/14128542) for best practices.

#### When computer use applies

Claude has several ways to interact with an app or service, and computer use is the broadest and slowest. It tries the most precise tool first:

- If you have a [connector](#connect-external-tools) for a service, Claude uses the connector.
- If the task is a shell command, Claude uses Bash.
- If the task is browser work and you have [Claude in Chrome](/docs/en/chrome) set up, Claude uses that.
- If none of those apply, Claude uses computer use.

The [per-app access tiers](#app-permissions) reinforce this: browsers are capped at view-only, and terminals and IDEs at click-only, steering Claude toward the dedicated tool even when computer use is active. Screen control is reserved for things nothing else can reach, like native apps, hardware control panels, mobile simulators, or proprietary tools without an API.

#### Enable computer use

Computer use is off by default. If you ask Claude to do something that needs it while it's off, Claude tells you it could do the task if you enable computer use in Settings.

1

Update the desktop app

Make sure you have the latest version of Claude Desktop. Download or update at [claude.com/download](https://claude.com/download) , then restart the app.

2

Turn on the toggle

In the desktop app, go to **Settings > General** (under **Desktop app** ). Find the **Computer use** toggle and turn it on. On Windows, the toggle takes effect immediately and setup is complete. On macOS, continue to the next step. If you don't see the toggle, confirm you're on macOS or Windows with a Pro or Max plan, then update and restart the app.

3

Grant macOS permissions

On macOS, grant two system permissions before the toggle takes effect:

- **Accessibility** : lets Claude click, type, and scroll
- **Screen Recording** : lets Claude see what's on your screen

The Settings page shows the current status of each permission. If either is denied, click the badge to open the relevant System Settings pane.

#### App permissions

The first time Claude needs to use an app, a prompt appears in your session. Click **Allow for this session** or **Deny** . Approvals last for the current session, or 30 minutes in [Dispatch-spawned sessions](#sessions-from-dispatch) . The prompt also shows what level of control Claude gets for that app. These tiers are fixed by app category and can't be changed:

| Tier         | What Claude can do                                       | Applies to                  |
|--------------|----------------------------------------------------------|-----------------------------|
| View only    | See the app in screenshots                               | Browsers, trading platforms |
| Click only   | Click and scroll, but not type or use keyboard shortcuts | Terminals, IDEs             |
| Full control | Click, type, drag, and use keyboard shortcuts            | Everything else             |

Apps with broad reach, like terminals, Finder or File Explorer, and System Settings or Settings, show an extra warning in the prompt so you know what approving them grants. You can configure two settings in **Settings > General** (under **Desktop app** ):

- **Denied apps** : add apps here to reject them without prompting. Claude may still affect a denied app indirectly through actions in an allowed app, but it can't interact with the denied app directly.
- **Unhide apps when Claude finishes** : while Claude is working, your other windows are hidden so it interacts with only the approved app. When Claude finishes, hidden windows are restored unless you turn this setting off.

### Manage sessions

Each session is an independent conversation with its own context and changes. You can run multiple sessions in parallel, send work to the cloud, or let Dispatch start sessions for you from your phone.

#### Work in parallel with sessions

Click **+ New session** in the sidebar to work on multiple tasks in parallel. For Git repositories, each session gets its own isolated copy of your project using [Git worktrees](/docs/en/common-workflows#run-parallel-claude-code-sessions-with-git-worktrees) , so changes in one session don't affect other sessions until you commit them. Worktrees are stored in `<project-root>/.claude/worktrees/` by default. You can change this to a custom directory in Settings → Claude Code under "Worktree location". You can also set a branch prefix that gets prepended to every worktree branch name, which is useful for keeping Claude-created branches organized. To remove a worktree when you're done, hover over the session in the sidebar and click the archive icon. To include gitignored files like `.env` in new worktrees, create a [`.worktreeinclude`](/docs/en/common-workflows#copy-gitignored-files-to-worktrees) [file](/docs/en/common-workflows#copy-gitignored-files-to-worktrees) in your project root.

Session isolation requires [Git](https://git-scm.com/downloads) . Most Macs include Git by default. Run `git --version` in Terminal to check. On Windows, Git is required for the Code tab to work: [download Git for Windows](https://git-scm.com/downloads/win) , install it, and restart the app. If you run into Git errors, try a Cowork session to help troubleshoot your setup.

Use the filter icon at the top of the sidebar to filter sessions by status (Active, Archived) and environment (Local, Cloud). To rename a session or check context usage, click the session title in the toolbar at the top of the active session. When context fills up, Claude automatically summarizes the conversation and continues working. You can also type `/compact` to trigger summarization earlier and free up context space. See [the context window](/docs/en/how-claude-code-works#the-context-window) for details on how compaction works.

#### Run long-running tasks remotely

For large refactors, test suites, migrations, or other long-running tasks, select **Remote** instead of **Local** when starting a session. Remote sessions run on Anthropic's cloud infrastructure and continue even if you close the app or shut down your computer. Check back anytime to see progress or steer Claude in a different direction. You can also monitor remote sessions from [claude.ai/code](https://claude.ai/code) or the Claude iOS app. Remote sessions also support multiple repositories. After selecting a cloud environment, click the **+** button next to the repo pill to add additional repositories to the session. Each repo gets its own branch selector. This is useful for tasks that span multiple codebases, such as updating a shared library and its consumers. See [Claude Code on the web](/docs/en/claude-code-on-the-web) for more on how remote sessions work.

#### Continue in another surface

The **Continue in** menu, accessible from the VS Code icon in the bottom right of the session toolbar, lets you move your session to another surface:

- **Claude Code on the Web** : sends your local session to continue running remotely. Desktop pushes your branch, generates a summary of the conversation, and creates a new remote session with the full context. You can then choose to archive the local session or keep it. This requires a clean working tree, and is not available for SSH sessions.
- **Your IDE** : opens your project in a supported IDE at the current working directory.

#### Sessions from Dispatch

[Dispatch](https://support.claude.com/en/articles/13947068) is a persistent conversation with Claude that lives in the [Cowork](https://claude.com/product/cowork#dispatch-and-computer-use) tab. You message Dispatch a task, and it decides how to handle it. A task can end up as a Code session in two ways: you ask for one directly, such as "open a Claude Code session and fix the login bug", or Dispatch decides the task is development work and spawns one on its own. Tasks that typically route to Code include fixing bugs, updating dependencies, running tests, or opening pull requests. Research, document editing, and spreadsheet work stay in Cowork. Either way, the Code session appears in the Code tab's sidebar with a **Dispatch** badge. You get a push notification on your phone when it finishes or needs your approval. If you have [computer use](#let-claude-use-your-computer) enabled, Dispatch-spawned Code sessions can use it too. App approvals in those sessions expire after 30 minutes and re-prompt, rather than lasting the full session like regular Code sessions. For setup, pairing, and Dispatch settings, see the [Dispatch help article](https://support.claude.com/en/articles/13947068) . Dispatch requires a Pro or Max plan and is not available on Team or Enterprise plans. Dispatch is one of several ways to work with Claude when you're away from your terminal. See [Platforms and integrations](/docs/en/platforms#work-when-you-are-away-from-your-terminal) to compare it with Remote Control, Channels, Slack, and scheduled tasks.

### Extend Claude Code

Connect external services, add reusable workflows, customize Claude's behavior, and configure preview servers.

#### Connect external tools

For local and [SSH](#ssh-sessions) sessions, click the **+** button next to the prompt box and select **Connectors** to add integrations like Google Calendar, Slack, GitHub, Linear, Notion, and more. You can add connectors before or during a session. The **+** button is not available in remote sessions, but [scheduled tasks](/docs/en/web-scheduled-tasks) configure connectors at task creation time. To manage or disconnect connectors, go to Settings → Connectors in the desktop app, or select **Manage connectors** from the Connectors menu in the prompt box. Once connected, Claude can read your calendar, send messages, create issues, and interact with your tools directly. You can ask Claude what connectors are configured in your session. Connectors are [MCP servers](/docs/en/mcp) with a graphical setup flow. Use them for quick integration with supported services. For integrations not listed in Connectors, add MCP servers manually via [settings files](/docs/en/mcp#installing-mcp-servers) . You can also [create custom connectors](https://support.claude.com/en/articles/11175166-getting-started-with-custom-connectors-using-remote-mcp) .

#### Use skills

[Skills](/docs/en/skills) extend what Claude can do. Claude loads them automatically when relevant, or you can invoke one directly: type `/` in the prompt box or click the **+** button and select **Slash commands** to browse what's available. This includes [built-in commands](/docs/en/commands) , your [custom skills](/docs/en/skills#create-your-first-skill) , project skills from your codebase, and skills from any [installed plugins](/docs/en/plugins) . Select one and it appears highlighted in the input field. Type your task after it and send as usual.

#### Install plugins

[Plugins](/docs/en/plugins) are reusable packages that add skills, agents, hooks, MCP servers, and LSP configurations to Claude Code. You can install plugins from the desktop app without using the terminal. For local and [SSH](#ssh-sessions) sessions, click the **+** button next to the prompt box and select **Plugins** to see your installed plugins and their skills. To add a plugin, select **Add plugin** from the submenu to open the plugin browser, which shows available plugins from your configured [marketplaces](/docs/en/plugin-marketplaces) including the official Anthropic marketplace. Select **Manage plugins** to enable, disable, or uninstall plugins. Plugins can be scoped to your user account, a specific project, or local-only. Plugins are not available for remote sessions. For the full plugin reference including creating your own plugins, see [plugins](/docs/en/plugins) .

#### Configure preview servers

Claude automatically detects your dev server setup and stores the configuration in `.claude/launch.json` at the root of the folder you selected when starting the session. Preview uses this folder as its working directory, so if you selected a parent folder, subfolders with their own dev servers won't be detected automatically. To work with a subfolder's server, either start a session in that folder directly or add a configuration manually. To customize how your server starts, for example to use `yarn dev` instead of `npm run dev` or to change the port, edit the file manually or click **Edit configuration** in the Preview dropdown to open it in your code editor. The file supports JSON with comments.

```
{
"version" : "0.0.1" ,
"configurations" : [
{
"name" : "my-app" ,
"runtimeExecutable" : "npm" ,
"runtimeArgs" : [ "run" , "dev" ],
"port" : 3000
}
]
}
```

You can define multiple configurations to run different servers from the same project, such as a frontend and an API. See the [examples](#examples) below.

##### Auto-verify changes

When `autoVerify` is enabled, Claude automatically verifies code changes after editing files. It takes screenshots, checks for errors, and confirms changes work before completing its response. Auto-verify is on by default. Disable it per-project by adding `"autoVerify": false` to `.claude/launch.json` , or toggle it from the **Preview** dropdown menu.

```
{
"version" : "0.0.1" ,
"autoVerify" : false ,
"configurations" : [ ... ]
}
```

When disabled, preview tools are still available and you can ask Claude to verify at any time. Auto-verify makes it automatic after every edit.

##### Configuration fields

Each entry in the `configurations` array accepts the following fields:

| Field               | Type     | Description                                                                                                                                                                                                                                                               |
|---------------------|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `name`              | string   | A unique identifier for this server                                                                                                                                                                                                                                       |
| `runtimeExecutable` | string   | The command to run, such as `npm` , `yarn` , or `node`                                                                                                                                                                                                                    |
| `runtimeArgs`       | string[] | Arguments passed to `runtimeExecutable` , such as `["run", "dev"]`                                                                                                                                                                                                        |
| `port`              | number   | The port your server listens on. Defaults to 3000                                                                                                                                                                                                                         |
| `cwd`               | string   | Working directory relative to your project root. Defaults to the project root. Use `${workspaceFolder}` to reference the project root explicitly                                                                                                                          |
| `env`               | object   | Additional environment variables as key-value pairs, such as `{ "NODE_ENV": "development" }` . Don't put secrets here since this file is committed to your repo. To pass secrets to your dev server, set them in the [local environment editor](#local-sessions) instead. |
| `autoPort`          | boolean  | How to handle port conflicts. See below                                                                                                                                                                                                                                   |
| `program`           | string   | A script to run with `node` . See [when to use](#when-to-use-program-vs-runtimeexecutable) [`program`](#when-to-use-program-vs-runtimeexecutable) [vs](#when-to-use-program-vs-runtimeexecutable) [`runtimeExecutable`](#when-to-use-program-vs-runtimeexecutable)        |
| `args`              | string[] | Arguments passed to `program` . Only used when `program` is set                                                                                                                                                                                                           |

##### When to use program vs runtimeExecutable

Use `runtimeExecutable` with `runtimeArgs` to start a dev server through a package manager. For example, `"runtimeExecutable": "npm"` with `"runtimeArgs": ["run", "dev"]` runs `npm run dev` . Use `program` when you have a standalone script you want to run with `node` directly. For example, `"program": "server.js"` runs `node server.js` . Pass additional flags with `args` .

##### Port conflicts

The `autoPort` field controls what happens when your preferred port is already in use:

- **`true`** : Claude finds and uses a free port automatically. Suitable for most dev servers.
- **`false`** : Claude fails with an error. Use this when your server must use a specific port, such as for OAuth callbacks or CORS allowlists.
- **Not set (default)** : Claude asks whether the server needs that exact port, then saves your answer.

When Claude picks a different port, it passes the assigned port to your server via the `PORT` environment variable.

##### Examples

These configurations show common setups for different project types:

- Next.js
- Multiple servers
- Node.js script

This configuration runs a Next.js app using Yarn on port 3000:

```
{
"version" : "0.0.1" ,
"configurations" : [
{
"name" : "web" ,
"runtimeExecutable" : "yarn" ,
"runtimeArgs" : [ "dev" ],
"port" : 3000
}
]
}
```

For a monorepo with a frontend and an API server, define multiple configurations. The frontend uses `autoPort: true` so it picks a free port if 3000 is taken, while the API server requires port 8080 exactly:

```
{
"version" : "0.0.1" ,
"configurations" : [
{
"name" : "frontend" ,
"runtimeExecutable" : "npm" ,
"runtimeArgs" : [ "run" , "dev" ],
"cwd" : "apps/web" ,
"port" : 3000 ,
"autoPort" : true
},
{
"name" : "api" ,
"runtimeExecutable" : "npm" ,
"runtimeArgs" : [ "run" , "start" ],
"cwd" : "server" ,
"port" : 8080 ,
"env" : { "NODE_ENV" : "development" },
"autoPort" : false
}
]
}
```

To run a Node.js script directly instead of using a package manager command, use the `program` field:

```
{
"version" : "0.0.1" ,
"configurations" : [
{
"name" : "server" ,
"program" : "server.js" ,
"args" : [ "--verbose" ],
"port" : 4000
}
]
}
```

### Environment configuration

The environment you pick when [starting a session](#start-a-session) determines where Claude executes and how you connect:

- **Local** : runs on your machine with direct access to your files
- **Remote** : runs on Anthropic's cloud infrastructure. Sessions continue even if you close the app.
- **SSH** : runs on a remote machine you connect to over SSH, such as your own servers, cloud VMs, or dev containers

#### Local sessions

The desktop app does not always inherit your full shell environment. On macOS, when you launch the app from the Dock or Finder, it reads your shell profile, such as `~/.zshrc` or `~/.bashrc` , to extract `PATH` and a fixed set of Claude Code variables, but other variables you export there are not picked up. On Windows, the app inherits user and system environment variables but does not read PowerShell profiles. To set environment variables for local sessions and dev servers on any platform, open the environment dropdown in the prompt box, hover over **Local** , and click the gear icon to open the local environment editor. Variables you save here are stored encrypted on your machine and apply to every local session and preview server you start. You can also add variables to the `env` key in your `~/.claude/settings.json` file, though these reach Claude sessions only and not dev servers. See [environment variables](/docs/en/env-vars) for the full list of supported variables. [Extended thinking](/docs/en/common-workflows#use-extended-thinking-thinking-mode) is enabled by default, which improves performance on complex reasoning tasks but uses additional tokens. To disable thinking entirely, set `MAX_THINKING_TOKENS` to `0` in the local environment editor. On Opus 4.6 and Sonnet 4.6, any other `MAX_THINKING_TOKENS` value is ignored because adaptive reasoning controls thinking depth instead. To use a fixed thinking budget on these models, also set `CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING` to `1` .

#### Remote sessions

Remote sessions continue in the background even if you close the app. Usage counts toward your [subscription plan limits](/docs/en/costs) with no separate compute charges. You can create custom cloud environments with different network access levels and environment variables. Select the environment dropdown when starting a remote session and choose **Add environment** . See [the cloud environment](/docs/en/claude-code-on-the-web#the-cloud-environment) for details on configuring network access and environment variables.

#### SSH sessions

SSH sessions let you run Claude Code on a remote machine while using the desktop app as your interface. This is useful for working with codebases that live on cloud VMs, dev containers, or servers with specific hardware or dependencies. To add an SSH connection, click the environment dropdown before starting a session and select **+ Add SSH connection** . The dialog asks for:

- **Name** : a friendly label for this connection
- **SSH Host** : `user@hostname` or a host defined in `~/.ssh/config`
- **SSH Port** : defaults to 22 if left empty, or uses the port from your SSH config
- **Identity File** : path to your private key, such as `~/.ssh/id_rsa` . Leave empty to use the default key or your SSH config.

Once added, the connection appears in the environment dropdown. Select it to start a session on that machine. Claude runs on the remote machine with access to its files and tools. Claude Code must be installed on the remote machine. Once connected, SSH sessions support permission modes, connectors, plugins, and MCP servers.

### Enterprise configuration

Organizations on Team or Enterprise plans can manage desktop app behavior through admin console controls, managed settings files, and device management policies.

#### Admin console controls

These settings are configured through the [admin settings console](https://claude.ai/admin-settings/claude-code) :

- **Code in the desktop** : control whether users in your organization can access Claude Code in the desktop app
- **Code in the web** : enable or disable [web sessions](/docs/en/claude-code-on-the-web) for your organization
- **Remote Control** : enable or disable [Remote Control](/docs/en/remote-control) for your organization
- **Disable Bypass permissions mode** : prevent users in your organization from enabling bypass permissions mode

#### Managed settings

Managed settings override project and user settings and apply when Desktop spawns CLI sessions. You can set these keys in your organization's [managed settings](/docs/en/settings#settings-precedence) file or push them remotely through the admin console.

| Key                                        | Description                                                                                                                                                                                         |
|--------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `permissions.disableBypassPermissionsMode` | set to `"disable"` to prevent users from enabling Bypass permissions mode.                                                                                                                          |
| `disableAutoMode`                          | set to `"disable"` to prevent users from enabling [Auto](/docs/en/permission-modes#eliminate-prompts-with-auto-mode) mode. Removes Auto from the mode selector. Also accepted under `permissions` . |
| `autoMode`                                 | customize what the auto mode classifier trusts and blocks across your organization. See [Configure the auto mode classifier](/docs/en/permissions#configure-the-auto-mode-classifier) .             |

`permissions.disableBypassPermissionsMode` and `disableAutoMode` also work in user and project settings, but placing them in managed settings prevents users from overriding them. `autoMode` is read from user settings, `.claude/settings.local.json` , and managed settings, but not from the checked-in `.claude/settings.json` : a cloned repo cannot inject its own classifier rules. For the complete list of managed-only settings including `allowManagedPermissionRulesOnly` and `allowManagedHooksOnly` , see [managed-only settings](/docs/en/permissions#managed-only-settings) . Remote managed settings uploaded through the admin console currently apply to CLI and IDE sessions only. For Desktop-specific restrictions, use the admin console controls above.

#### Device management policies

IT teams can manage the desktop app through MDM on macOS or group policy on Windows. Available policies include enabling or disabling the Claude Code feature, controlling auto-updates, and setting a custom deployment URL.

- **macOS** : configure via `com.anthropic.Claude` preference domain using tools like Jamf or Kandji
- **Windows** : configure via registry at `SOFTWARE\Policies\Claude`

#### Authentication and SSO

Enterprise organizations can require SSO for all users. See [authentication](/docs/en/authentication) for plan-level details and [Setting up SSO](https://support.claude.com/en/articles/13132885-setting-up-single-sign-on-sso) for SAML and OIDC configuration.

#### Data handling

Claude Code processes your code locally in local sessions or on Anthropic's cloud infrastructure in remote sessions. Conversations and code context are sent to Anthropic's API for processing. See [data handling](/docs/en/data-usage) for details on data retention, privacy, and compliance.

#### Deployment

Desktop can be distributed through enterprise deployment tools:

- **macOS** : distribute via MDM such as Jamf or Kandji using the `.dmg` installer
- **Windows** : deploy via MSIX package or `.exe` installer. See [Deploy Claude Desktop for Windows](https://support.claude.com/en/articles/12622703-deploy-claude-desktop-for-windows) for enterprise deployment options including silent installation

For network configuration such as proxy settings, firewall allowlisting, and LLM gateways, see [network configuration](/docs/en/network-config) . For the full enterprise configuration reference, see the [enterprise configuration guide](https://support.claude.com/en/articles/12622667-enterprise-configuration) .

### Coming from the CLI?

If you already use the Claude Code CLI, Desktop runs the same underlying engine with a graphical interface. You can run both simultaneously on the same machine, even on the same project. Each maintains separate session history, but they share configuration and project memory via CLAUDE.md files. To move a CLI session into Desktop, run `/desktop` in the terminal. Claude saves your session and opens it in the desktop app, then exits the CLI. This command is available on macOS and Windows only.

When to use Desktop vs CLI: use Desktop when you want visual diff review, file attachments, or session management in a sidebar. Use the CLI when you need scripting, automation, third-party providers, or prefer a terminal workflow.

#### CLI flag equivalents

This table shows the desktop app equivalent for common CLI flags. Flags not listed have no desktop equivalent because they are designed for scripting or automation.

| CLI                                    | Desktop equivalent                                                                                                                       |
|----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `--model sonnet`                       | Model dropdown next to the send button, before starting a session                                                                        |
| `--resume` , `--continue`              | Click a session in the sidebar                                                                                                           |
| `--permission-mode`                    | Mode selector next to the send button                                                                                                    |
| `--dangerously-skip-permissions`       | Bypass permissions mode. Enable in Settings → Claude Code → "Allow bypass permissions mode". Enterprise admins can disable this setting. |
| `--add-dir`                            | Add multiple repos with the **+** button in remote sessions                                                                              |
| `--allowedTools` , `--disallowedTools` | Not available in Desktop                                                                                                                 |
| `--verbose`                            | Not available. Check system logs: Console.app on macOS, Event Viewer → Windows Logs → Application on Windows                             |
| `--print` , `--output-format`          | Not available. Desktop is interactive only.                                                                                              |
| `ANTHROPIC_MODEL` env var              | Model dropdown next to the send button                                                                                                   |
| `MAX_THINKING_TOKENS` env var          | Set in the local environment editor. See [environment configuration](#environment-configuration) .                                       |

#### Shared configuration

Desktop and CLI read the same configuration files, so your setup carries over:

- [**CLAUDE.md**](/docs/en/memory) and `CLAUDE.local.md` files in your project are used by both
- [**MCP servers**](/docs/en/mcp) configured in `~/.claude.json` or `.mcp.json` work in both
- [**Hooks**](/docs/en/hooks) and [**skills**](/docs/en/skills) defined in settings apply to both
- [**Settings**](/docs/en/settings) in `~/.claude.json` and `~/.claude/settings.json` are shared. Permission rules, allowed tools, and other settings in `settings.json` apply to Desktop sessions.
- **Models** : Sonnet, Opus, and Haiku are available in both. In Desktop, select the model from the dropdown next to the send button before starting a session. You cannot change the model during an active session.

**MCP servers: desktop chat app vs Claude Code** : MCP servers configured for the Claude Desktop chat app in `claude_desktop_config.json` are separate from Claude Code and will not appear in the Code tab. To use MCP servers in Claude Code, configure them in `~/.claude.json` or your project's `.mcp.json` file. See [MCP configuration](/docs/en/mcp#installing-mcp-servers) for details.

#### Feature comparison

This table compares core capabilities between the CLI and Desktop. For a full list of CLI flags, see the [CLI reference](/docs/en/cli-reference) .

| Feature                                                    | CLI                                                                          | Desktop                                                                                     |
|------------------------------------------------------------|------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Permission modes                                           | All modes including `dontAsk`                                                | Ask permissions, Auto accept edits, Plan mode, Auto, and Bypass permissions via Settings    |
| `--dangerously-skip-permissions`                           | CLI flag                                                                     | Bypass permissions mode. Enable in Settings → Claude Code → "Allow bypass permissions mode" |
| [Third-party providers](/docs/en/third-party-integrations) | Bedrock, Vertex, Foundry                                                     | Not available. Desktop connects to Anthropic's API directly.                                |
| [MCP servers](/docs/en/mcp)                                | Configure in settings files                                                  | Connectors UI for local and SSH sessions, or settings files                                 |
| [Plugins](/docs/en/plugins)                                | `/plugin` command                                                            | Plugin manager UI                                                                           |
| @mention files                                             | Text-based                                                                   | With autocomplete; local and SSH sessions only                                              |
| File attachments                                           | Not available                                                                | Images, PDFs                                                                                |
| Session isolation                                          | [`--worktree`](/docs/en/cli-reference) flag                                  | Automatic worktrees                                                                         |
| Multiple sessions                                          | Separate terminals                                                           | Sidebar tabs                                                                                |
| Recurring tasks                                            | Cron jobs, CI pipelines                                                      | [Scheduled tasks](/docs/en/desktop-scheduled-tasks)                                         |
| Computer use                                               | [Enable via](/docs/en/computer-use) [`/mcp`](/docs/en/computer-use) on macOS | [App and screen control](#let-claude-use-your-computer) on macOS and Windows                |
| Dispatch integration                                       | Not available                                                                | [Dispatch sessions](#sessions-from-dispatch) in the sidebar                                 |
| Scripting and automation                                   | [`--print`](/docs/en/cli-reference) , [Agent SDK](/docs/en/headless)         | Not available                                                                               |

#### What's not available in Desktop

The following features are only available in the CLI or VS Code extension:

- **Third-party providers** : Desktop connects to Anthropic's API directly. Use the [CLI](/docs/en/quickstart) with Bedrock, Vertex, or Foundry instead.
- **Linux** : the desktop app is available on macOS and Windows only.
- **Inline code suggestions** : Desktop does not provide autocomplete-style suggestions. It works through conversational prompts and explicit code changes.
- **Agent teams** : multi-agent orchestration is available via the [CLI](/docs/en/agent-teams) and [Agent SDK](/docs/en/headless) , not in Desktop.

### Troubleshooting

#### Check your version

To see which version of the desktop app you're running:

- **macOS** : click **Claude** in the menu bar, then **About Claude**
- **Windows** : click **Help** , then **About**

Click the version number to copy it to your clipboard.

#### 403 or authentication errors in the Code tab

If you see `Error 403: Forbidden` or other authentication failures when using the Code tab:

1. Sign out and back in from the app menu. This is the most common fix.
2. Verify you have an active paid subscription: Pro, Max, Team, or Enterprise.
3. If the CLI works but Desktop does not, quit the desktop app completely, not just close the window, then reopen and sign in again.
4. Check your internet connection and proxy settings.

#### Blank or stuck screen on launch

If the app opens but shows a blank or unresponsive screen:

1. Restart the app.
2. Check for pending updates. The app auto-updates on launch.
3. On Windows, check Event Viewer for crash logs under **Windows Logs → Application** .

#### "Failed to load session"

If you see `Failed to load session` , the selected folder may no longer exist, a Git repository may require Git LFS that isn't installed, or file permissions may prevent access. Try selecting a different folder or restarting the app.

#### Session not finding installed tools

If Claude can't find tools like `npm` , `node` , or other CLI commands, verify the tools work in your regular terminal, check that your shell profile properly sets up PATH, and restart the desktop app to reload environment variables.

#### Git and Git LFS errors

On Windows, Git is required for the Code tab to start local sessions. If you see "Git is required," install [Git for Windows](https://git-scm.com/downloads/win) and restart the app. If you see "Git LFS is required by this repository but is not installed," install Git LFS from [git-lfs.com](https://git-lfs.com/) , run `git lfs install` , and restart the app.

#### MCP servers not working on Windows

If MCP server toggles don't respond or servers fail to connect on Windows, check that the server is properly configured in your settings, restart the app, verify the server process is running in Task Manager, and review server logs for connection errors.

#### App won't quit

- **macOS** : press Cmd+Q. If the app doesn't respond, use Force Quit with Cmd+Option+Esc, select Claude, and click Force Quit.
- **Windows** : use Task Manager with Ctrl+Shift+Esc to end the Claude process.

#### Windows-specific issues

- **PATH not updated after install** : open a new terminal window. PATH updates only apply to new terminal sessions.
- **Concurrent installation error** : if you see an error about another installation in progress but there isn't one, try running the installer as Administrator.
- **ARM64** : Windows ARM64 devices are fully supported.

#### Cowork tab unavailable on Intel Macs

The Cowork tab requires Apple Silicon (M1 or later) on macOS. On Windows, Cowork is available on all supported hardware. The Chat and Code tabs work normally on Intel Macs.

#### "Branch doesn't exist yet" when opening in CLI

Remote sessions can create branches that don't exist on your local machine. Click the branch name in the session toolbar to copy it, then fetch it locally:

```
git fetch origin < branch-nam e >
git checkout < branch-nam e >
```

#### Still stuck?

- Search or file a bug on [GitHub Issues](https://github.com/anthropics/claude-code/issues)
- Visit the [Claude support center](https://support.claude.com/)

When filing a bug, include your desktop app version, your operating system, the exact error message, and relevant logs. On macOS, check Console.app. On Windows, check Event Viewer → Windows Logs → Application.

Was this page helpful?

Yes

No

[Get started](/docs/en/desktop-quickstart) [Scheduled tasks](/docs/en/desktop-scheduled-tasks)

⌘ I


---

# Troubleshooting


### Troubleshooting


Discover solutions to common issues with Claude Code installation and usage.


### Troubleshoot installation issues

If you'd rather skip the terminal entirely, the [Claude Code Desktop app](/docs/en/desktop-quickstart) lets you install and use Claude Code through a graphical interface. Download it for [macOS](https://claude.ai/api/desktop/darwin/universal/dmg/latest/redirect?utm_source=claude_code&utm_medium=docs) or [Windows](https://claude.com/download?utm_source=claude_code&utm_medium=docs) and start coding without any command-line setup.

Find the error message or symptom you're seeing:

| What you see                                                | Solution                                                                                                                 |
|-------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `command not found: claude` or `'claude' is not recognized` | [Fix your PATH](#command-not-found-claude-after-installation)                                                            |
| `syntax error near unexpected token '<'`                    | [Install script returns HTML](#install-script-returns-html-instead-of-a-shell-script)                                    |
| `curl: (56) Failure writing output to destination`          | [Download script first, then run it](#curl-56-failure-writing-output-to-destination)                                     |
| `Killed` during install on Linux                            | [Add swap space for low-memory servers](#install-killed-on-low-memory-linux-servers)                                     |
| `TLS connect error` or `SSL/TLS secure channel`             | [Update CA certificates](#tls-or-ssl-connection-errors)                                                                  |
| `Failed to fetch version` or can't reach download server    | [Check network and proxy settings](#check-network-connectivity)                                                          |
| `irm is not recognized` or `&& is not valid`                | [Use the right command for your shell](#windows-irm-or--not-recognized)                                                  |
| `Claude Code on Windows requires git-bash`                  | [Install or configure Git Bash](#windows-claude-code-on-windows-requires-git-bash)                                       |
| `Error loading shared library`                              | [Wrong binary variant for your system](#linux-wrong-binary-variant-installed-muslglibc-mismatch)                         |
| `Illegal instruction` on Linux                              | [Architecture mismatch](#illegal-instruction-on-linux)                                                                   |
| `dyld: cannot load` or `Abort trap` on macOS                | [Binary incompatibility](#dyld-cannot-load-on-macos)                                                                     |
| `Invoke-Expression: Missing argument in parameter list`     | [Install script returns HTML](#install-script-returns-html-instead-of-a-shell-script)                                    |
| `App unavailable in region`                                 | Claude Code is not available in your country. See [supported countries](https://www.anthropic.com/supported-countries) . |
| `unable to get local issuer certificate`                    | [Configure corporate CA certificates](#tls-or-ssl-connection-errors)                                                     |
| `OAuth error` or `403 Forbidden`                            | [Fix authentication](#authentication-issues)                                                                             |

If your issue isn't listed, work through these diagnostic steps.

### Debug installation problems

#### Check network connectivity

The installer downloads from `storage.googleapis.com` . Verify you can reach it:

```
curl -sI https://storage.googleapis.com
```

If this fails, your network may be blocking the connection. Common causes:

- Corporate firewalls or proxies blocking Google Cloud Storage
- Regional network restrictions: try a VPN or alternative network
- TLS/SSL issues: update your system's CA certificates, or check if `HTTPS_PROXY` is configured

If you're behind a corporate proxy, set `HTTPS_PROXY` and `HTTP_PROXY` to your proxy's address before installing. Ask your IT team for the proxy URL if you don't know it, or check your browser's proxy settings. This example sets both proxy variables, then runs the installer through your proxy:

```
export HTTP_PROXY = http :// proxy . example . com : 8080
export HTTPS_PROXY = http :// proxy . example . com : 8080
curl -fsSL https://claude.ai/install.sh | bash
```

#### Verify your PATH

If installation succeeded but you get a `command not found` or `not recognized` error when running `claude` , the install directory isn't in your PATH. Your shell searches for programs in directories listed in PATH, and the installer places `claude` at `~/.local/bin/claude` on macOS/Linux or `%USERPROFILE%\.local\bin\claude.exe` on Windows. Check if the install directory is in your PATH by listing your PATH entries and filtering for `local/bin` :

- macOS/Linux
- Windows PowerShell
- Windows CMD

```
echo $PATH | tr ':' '\n' | grep local/bin
```

If there's no output, the directory is missing. Add it to your shell configuration:

```
### Zsh (macOS default)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

### Bash (Linux default)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Alternatively, close and reopen your terminal. Verify the fix worked:

```
claude --version
```

```
$ env: PATH -split ';' | Select-String 'local\\bin'
```

If there's no output, add the install directory to your User PATH:

```
$currentPath = [ Environment ]::GetEnvironmentVariable( 'PATH' , 'User' )
[ Environment ]::SetEnvironmentVariable( 'PATH' , " $currentPath ; $ env: USERPROFILE \.local\bin" , 'User' )
```

Restart your terminal for the change to take effect. Verify the fix worked:

```
claude -- version
```

```
echo %PATH% | findstr /i "local\bin"
```

If there's no output, open System Settings, go to Environment Variables, and add `%USERPROFILE%\.local\bin` to your User PATH variable. Restart your terminal. Verify the fix worked:

```
claude --version
```

#### Check for conflicting installations

Multiple Claude Code installations can cause version mismatches or unexpected behavior. Check what's installed:

- macOS/Linux
- Windows PowerShell

List all `claude` binaries found in your PATH:

```
which -a claude
```

Check whether the native installer and npm versions are present:

```
ls -la ~/.local/bin/claude
```

```
ls -la ~/.claude/local/
```

```
npm -g ls @anthropic-ai/claude-code 2> /dev/null
```

```
where.exe claude
Test-Path " $ env: LOCALAPPDATA \Claude Code\claude.exe"
```

If you find multiple installations, keep only one. The native install at `~/.local/bin/claude` is recommended. Remove any extra installations: Uninstall an npm global install:

```
npm uninstall -g @anthropic-ai/claude-code
```

Remove a Homebrew install on macOS (use `claude-code@latest` if you installed that cask):

```
brew uninstall --cask claude-code
```

#### Check directory permissions

The installer needs write access to `~/.local/bin/` and `~/.claude/` . If installation fails with permission errors, check whether these directories are writable:

```
test -w ~/.local/bin && echo "writable" || echo "not writable"
test -w ~/.claude && echo "writable" || echo "not writable"
```

If either directory isn't writable, create the install directory and set your user as the owner:

```
sudo mkdir -p ~/.local/bin
sudo chown -R $( whoami ) ~/.local
```

#### Verify the binary works

If `claude` is installed but crashes or hangs on startup, run these checks to narrow down the cause. Confirm the binary exists and is executable:

```
ls -la $( which claude )
```

On Linux, check for missing shared libraries. If `ldd` shows missing libraries, you may need to install system packages. On Alpine Linux and other musl-based distributions, see [Alpine Linux setup](/docs/en/setup#alpine-linux-and-musl-based-distributions) .

```
ldd $( which claude ) | grep "not found"
```

Run a quick sanity check that the binary can execute:

```
claude --version
```

### Common installation issues

These are the most frequently encountered installation problems and their solutions.

#### Install script returns HTML instead of a shell script

When running the install command, you may see one of these errors:

```
bash: line 1: syntax error near unexpected token `<'
bash: line 1: `<!DOCTYPE html>'
```

On PowerShell, the same problem appears as:

```
Invoke-Expression: Missing argument in parameter list.
```

This means the install URL returned an HTML page instead of the install script. If the HTML page says "App unavailable in region," Claude Code is not available in your country. See [supported countries](https://www.anthropic.com/supported-countries) . Otherwise, this can happen due to network issues, regional routing, or a temporary service disruption. **Solutions:**

1. **Use an alternative install method** : On macOS or Linux, install via Homebrew: `brew install --cask claude-code` On Windows, install via WinGet: `winget install Anthropic.ClaudeCode`
2. **Retry after a few minutes** : the issue is often temporary. Wait and try the original command again.

#### command not found: claude after installation

The install finished but `claude` doesn't work. The exact error varies by platform:

| Platform    | Error message                                                          |
|-------------|------------------------------------------------------------------------|
| macOS       | `zsh: command not found: claude`                                       |
| Linux       | `bash: claude: command not found`                                      |
| Windows CMD | `'claude' is not recognized as an internal or external command`        |
| PowerShell  | `claude : The term 'claude' is not recognized as the name of a cmdlet` |

This means the install directory isn't in your shell's search path. See [Verify your PATH](#verify-your-path) for the fix on each platform.

#### curl: (56) Failure writing output to destination

The `curl ... | bash` command downloads the script and passes it directly to Bash for execution using a pipe ( `|` ). This error means the connection broke before the script finished downloading. Common causes include network interruptions, the download being blocked mid-stream, or system resource limits. **Solutions:**

1. **Check network stability** : Claude Code binaries are hosted on Google Cloud Storage. Test that you can reach it: `curl -fsSL https://storage.googleapis.com -o /dev/null` If the command completes silently, your connection is fine and the issue is likely intermittent. Retry the install command. If you see an error, your network may be blocking the download.
2. **Try an alternative install method** : On macOS or Linux: `brew install --cask claude-code` On Windows: `winget install Anthropic.ClaudeCode`

#### TLS or SSL connection errors

Errors like `curl: (35) TLS connect error` , `schannel: next InitializeSecurityContext failed` , or PowerShell's `Could not establish trust relationship for the SSL/TLS secure channel` indicate TLS handshake failures. **Solutions:**

1. **Update your system CA certificates** : On Ubuntu/Debian: `sudo apt-get update && sudo apt-get install ca-certificates` On macOS via Homebrew: `brew install ca-certificates`
2. **On Windows, enable TLS 1.2** in PowerShell before running the installer: `[ Net.ServicePointManager ]::SecurityProtocol = [ Net.SecurityProtocolType ]::Tls12 irm https: // claude.ai / install.ps1 | iex`
3. **Check for proxy or firewall interference** : corporate proxies that perform TLS inspection can cause these errors, including `unable to get local issuer certificate` . Set `NODE_EXTRA_CA_CERTS` to your corporate CA certificate bundle: `export NODE_EXTRA_CA_CERTS = / path / to / corporate-ca . pem` Ask your IT team for the certificate file if you don't have it. You can also try on a direct connection to confirm the proxy is the cause.
4. **On Windows, bypass certificate revocation checks** if you see `CRYPT_E_NO_REVOCATION_CHECK (0x80092012)` or `CRYPT_E_REVOCATION_OFFLINE (0x80092013)` . These mean curl reached the server but your network blocks the certificate revocation lookup, which is common behind corporate firewalls. Add `--ssl-revoke-best-effort` to the install command: `curl --ssl-revoke-best-effort -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd` Alternatively, install with `winget install Anthropic.ClaudeCode` , which avoids curl entirely.

#### Failed to fetch version from storage.googleapis.com

The installer couldn't reach the download server. This typically means `storage.googleapis.com` is blocked on your network. **Solutions:**

1. **Test connectivity directly** : `curl -sI https://storage.googleapis.com`
2. **If behind a proxy** , set `HTTPS_PROXY` so the installer can route through it. See [proxy configuration](/docs/en/network-config#proxy-configuration) for details. `export HTTPS_PROXY = http :// proxy . example . com : 8080 curl -fsSL https://claude.ai/install.sh | bash`
3. **If on a restricted network** , try a different network or VPN, or use an alternative install method: On macOS or Linux: `brew install --cask claude-code` On Windows: `winget install Anthropic.ClaudeCode`

#### Windows: irm or && not recognized

If you see `'irm' is not recognized` or `The token '&&' is not valid` , you're running the wrong command for your shell.

- **`irm`** **not recognized** : you're in CMD, not PowerShell. You have two options: Open PowerShell by searching for "PowerShell" in the Start menu, then run the original install command: `irm https: // claude.ai / install.ps1 | iex` Or stay in CMD and use the CMD installer instead: `curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd`
- **`&&`** **not valid** : you're in PowerShell but ran the CMD installer command. Use the PowerShell installer: `irm https: // claude.ai / install.ps1 | iex`

#### Install killed on low-memory Linux servers

If you see `Killed` during installation on a VPS or cloud instance:

```
Setting up Claude Code...
Installing Claude Code native build latest...
bash: line 142: 34803 Killed    "$binary_path" install ${TARGET:+"$TARGET"}
```

The Linux OOM killer terminated the process because the system ran out of memory. Claude Code requires at least 4 GB of available RAM. **Solutions:**

1. **Add swap space** if your server has limited RAM. Swap uses disk space as overflow memory, letting the install complete even with low physical RAM. Create a 2 GB swap file and enable it: `sudo fallocate -l 2G /swapfile sudo chmod 600 /swapfile sudo mkswap /swapfile sudo swapon /swapfile` Then retry the installation: `curl -fsSL https://claude.ai/install.sh | bash`
2. **Close other processes** to free memory before installing.
3. **Use a larger instance** if possible. Claude Code requires at least 4 GB of RAM.

#### Install hangs in Docker

When installing Claude Code in a Docker container, installing as root into `/` can cause hangs. **Solutions:**

1. **Set a working directory** before running the installer. When run from `/` , the installer scans the entire filesystem, which causes excessive memory usage. Setting `WORKDIR` limits the scan to a small directory: `WORKDIR /tmp RUN curl -fsSL https://claude.ai/install.sh | bash`
2. **Increase Docker memory limits** if using Docker Desktop: `docker build --memory=4g .`

#### Windows: Claude Desktop overrides claude CLI command

If you installed an older version of Claude Desktop, it may register a `Claude.exe` in the `WindowsApps` directory that takes PATH priority over Claude Code CLI. Running `claude` opens the Desktop app instead of the CLI. Update Claude Desktop to the latest version to fix this issue.

#### Windows: "Claude Code on Windows requires git-bash"

Claude Code on native Windows needs [Git for Windows](https://git-scm.com/downloads/win) , which includes Git Bash. **If Git is not installed** , download and install it from [git-scm.com/downloads/win](https://git-scm.com/downloads/win) . During setup, select "Add to PATH." Restart your terminal after installing. **If Git is already installed** but Claude Code still can't find it, set the path in your [settings.json file](/docs/en/settings) :

```
{
"env" : {
"CLAUDE_CODE_GIT_BASH_PATH" : "C: \\ Program Files \\ Git \\ bin \\ bash.exe"
}
}
```

If your Git is installed somewhere else, find the path by running `where.exe git` in PowerShell and use the `bin\bash.exe` path from that directory.

#### Linux: wrong binary variant installed (musl/glibc mismatch)

If you see errors about missing shared libraries like `libstdc++.so.6` or `libgcc_s.so.1` after installation, the installer may have downloaded the wrong binary variant for your system.

```
Error loading shared library libstdc++.so.6: No such file or directory
```

This can happen on glibc-based systems that have musl cross-compilation packages installed, causing the installer to misdetect the system as musl. **Solutions:**

1. **Check which libc your system uses** : `ldd /bin/ls | head -1` If it shows `linux-vdso.so` or references to `/lib/x86_64-linux-gnu/` , you're on glibc. If it shows `musl` , you're on musl.
2. **If you're on glibc but got the musl binary** , remove the installation and reinstall. You can also manually download the correct binary from the GCS bucket at `https://storage.googleapis.com/claude-code-dist-86c565f3-f756-42ad-8dfa-d59b1c096819/claude-code-releases/{VERSION}/manifest.json` . File a [GitHub issue](https://github.com/anthropics/claude-code/issues) with the output of `ldd /bin/ls` and `ls /lib/libc.musl*` .
3. **If you're actually on musl** (Alpine Linux), install the required packages: `apk add libgcc libstdc++ ripgrep`

#### Illegal instruction on Linux

If the installer prints `Illegal instruction` instead of the OOM `Killed` message, the downloaded binary doesn't match your CPU architecture. This commonly happens on ARM servers that receive an x86 binary, or on older CPUs that lack required instruction sets.

```
bash: line 142: 2238232 Illegal instruction    "$binary_path" install ${TARGET:+"$TARGET"}
```

**Solutions:**

1. **Verify your architecture** : `uname -m x86_64` means 64-bit Intel/AMD, `aarch64` means ARM64. If the binary doesn't match, [file a GitHub issue](https://github.com/anthropics/claude-code/issues) with the output.
2. **Try an alternative install method** while the architecture issue is resolved: `brew install --cask claude-code`

#### dyld: cannot load on macOS

If you see `dyld: cannot load` or `Abort trap: 6` during installation, the binary is incompatible with your macOS version or hardware.

```
dyld: cannot load 'claude-2.1.42-darwin-x64' (load command 0x80000034 is unknown)
Abort trap: 6
```

**Solutions:**

1. **Check your macOS version** : Claude Code requires macOS 13.0 or later. Open the Apple menu and select About This Mac to check your version.
2. **Update macOS** if you're on an older version. The binary uses load commands that older macOS versions don't support.
3. **Try Homebrew** as an alternative install method: `brew install --cask claude-code`

#### Windows installation issues: errors in WSL

You might encounter the following issues in WSL: **OS/platform detection issues** : if you receive an error during installation, WSL may be using Windows `npm` . Try:

- Run `npm config set os linux` before installation
- Install with `npm install -g @anthropic-ai/claude-code --force --no-os-check` . Do not use `sudo` .

**Node not found errors** : if you see `exec: node: not found` when running `claude` , your WSL environment may be using a Windows installation of Node.js. You can confirm this with `which npm` and `which node` , which should point to Linux paths starting with `/usr/` rather than `/mnt/c/` . To fix this, try installing Node via your Linux distribution's package manager or via [`nvm`](https://github.com/nvm-sh/nvm) . **nvm version conflicts** : if you have nvm installed in both WSL and Windows, you may experience version conflicts when switching Node versions in WSL. This happens because WSL imports the Windows PATH by default, causing Windows nvm/npm to take priority over the WSL installation. You can identify this issue by:

- Running `which npm` and `which node` - if they point to Windows paths (starting with `/mnt/c/` ), Windows versions are being used
- Experiencing broken functionality after switching Node versions with nvm in WSL

To resolve this issue, fix your Linux PATH to ensure the Linux node/npm versions take priority: **Primary solution: Ensure nvm is properly loaded in your shell** The most common cause is that nvm isn't loaded in non-interactive shells. Add the following to your shell configuration file ( `~/.bashrc` , `~/.zshrc` , etc.):

```
### Load nvm if it exists
export NVM_DIR = " $HOME /.nvm"
[ -s " $NVM_DIR /nvm.sh" ] && \. " $NVM_DIR /nvm.sh"
[ -s " $NVM_DIR /bash_completion" ] && \. " $NVM_DIR /bash_completion"
```

Or run directly in your current session:

```
source ~/.nvm/nvm.sh
```

**Alternative: Adjust PATH order** If nvm is properly loaded but Windows paths still take priority, you can explicitly prepend your Linux paths to PATH in your shell configuration:

```
export PATH = " $HOME /.nvm/versions/node/$( node -v )/bin: $PATH "
```

Avoid disabling Windows PATH importing via `appendWindowsPath = false` as this breaks the ability to call Windows executables from WSL. Similarly, avoid uninstalling Node.js from Windows if you use it for Windows development.

#### WSL2 sandbox setup

[Sandboxing](/docs/en/sandboxing) is supported on WSL2 but requires installing additional packages. If you see an error about missing `bubblewrap` or `socat` when running `/sandbox` , install the dependencies:

- Ubuntu/Debian
- Fedora

```
sudo apt-get install bubblewrap socat
```

```
sudo dnf install bubblewrap socat
```

WSL1 does not support sandboxing. If you see "Sandboxing requires WSL2", you need to upgrade to WSL2 or run Claude Code without sandboxing. Sandboxed commands cannot launch Windows binaries such as `cmd.exe` , `powershell.exe` , or executables under `/mnt/c/` . WSL hands these off to the Windows host over a Unix socket, which the sandbox blocks. If a command needs to invoke a Windows binary, add it to [`excludedCommands`](/docs/en/settings#sandbox-settings) so it runs outside the sandbox.

#### Permission errors during installation

If the native installer fails with permission errors, the target directory may not be writable. See [Check directory permissions](#check-directory-permissions) . If you previously installed with npm and are hitting npm-specific permission errors, switch to the native installer:

```
curl -fsSL https://claude.ai/install.sh | bash
```

### Permissions and authentication

These sections address login failures, token issues, and permission prompt behavior.

#### Repeated permission prompts

If you find yourself repeatedly approving the same commands, you can allow specific tools

to run without approval using the

`/permissions` command. See [Permissions docs](/docs/en/permissions#manage-permissions) .

#### Authentication issues

If you're experiencing authentication problems:

1. Run `/logout` to sign out completely
2. Close Claude Code
3. Restart with `claude` and complete the authentication process again

If the browser doesn't open automatically during login, press `c` to copy the OAuth URL to your clipboard, then paste it into your browser manually.

#### OAuth error: Invalid code

If you see `OAuth error: Invalid code. Please make sure the full code was copied` , the login code expired or was truncated during copy-paste. **Solutions:**

- Press Enter to retry and complete the login quickly after the browser opens
- Type `c` to copy the full URL if the browser doesn't open automatically
- If using a remote/SSH session, the browser may open on the wrong machine. Copy the URL displayed in the terminal and open it in your local browser instead.

#### 403 Forbidden after login

If you see `API Error: 403 {"error":{"type":"forbidden","message":"Request not allowed"}}` after logging in:

- **Claude Pro/Max users** : verify your subscription is active at [claude.ai/settings](https://claude.ai/settings)
- **Console users** : confirm your account has the "Claude Code" or "Developer" role assigned by your admin
- **Behind a proxy** : corporate proxies can interfere with API requests. See [network configuration](/docs/en/network-config) for proxy setup.

#### Model not found or not accessible

If you see `There's an issue with the selected model (...). It may not exist or you may not have access to it` , the API rejected the configured model name. Common causes:

- A typo in the model name passed to `--model`
- A stale or deprecated model ID saved in your settings
- An API key without access to that model on your current usage tier

Check where the model is set, in [priority order](/docs/en/model-config#setting-your-model) :

- The `--model` flag
- The `ANTHROPIC_MODEL` environment variable
- The `model` field in `.claude/settings.local.json`
- The `model` field in your project's `.claude/settings.json`
- The `model` field in `~/.claude/settings.json`

To clear a stale value, remove the `model` field from your settings or unset `ANTHROPIC_MODEL` , and Claude Code will fall back to the default model for your account. To browse models available to your account, start `claude` interactively and run `/model` to open the picker. For Vertex AI deployments, see [the Vertex AI troubleshooting section](/docs/en/google-vertex-ai#troubleshooting) .

#### "This organization has been disabled" with an active subscription

If you see `API Error: 400 ... "This organization has been disabled"` despite having an active Claude subscription, an `ANTHROPIC_API_KEY` environment variable is overriding your subscription. This commonly happens when an old API key from a previous employer or project is still set in your shell profile. When `ANTHROPIC_API_KEY` is present and you have approved it, Claude Code uses that key instead of your subscription's OAuth credentials. In non-interactive mode ( `-p` ), the key is always used when present. See [authentication precedence](/docs/en/authentication#authentication-precedence) for the full resolution order. To use your subscription instead, unset the environment variable and remove it from your shell profile:

```
unset ANTHROPIC_API_KEY
claude
```

Check `~/.zshrc` , `~/.bashrc` , or `~/.profile` for `export ANTHROPIC_API_KEY=...` lines and remove them to make the change permanent. Run `/status` inside Claude Code to confirm which authentication method is active.

#### OAuth login fails in WSL2

Browser-based login in WSL2 may fail if WSL can't open your Windows browser. Set the `BROWSER` environment variable:

```
export BROWSER = "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
claude
```

Or copy the URL manually: when the login prompt appears, press `c` to copy the OAuth URL, then paste it into your Windows browser.

#### "Not logged in" or token expired

If Claude Code prompts you to log in again after a session, your OAuth token may have expired. Run `/login` to re-authenticate. If this happens frequently, check that your system clock is accurate, as token validation depends on correct timestamps. On macOS, login can also fail when the Keychain is locked or its password is out of sync with your account password, which prevents Claude Code from saving credentials. Run `claude doctor` to check Keychain access. To unlock the Keychain manually, run `security unlock-keychain ~/Library/Keychains/login.keychain-db` . If unlocking doesn't help, open Keychain Access, select the `login` keychain, and choose Edit > Change Password for Keychain "login" to resync it with your account password.

### Configuration file locations

Claude Code stores configuration in several locations:

| File                          | Purpose                                                                                                     |
|-------------------------------|-------------------------------------------------------------------------------------------------------------|
| `~/.claude/settings.json`     | User settings (permissions, hooks, model overrides)                                                         |
| `.claude/settings.json`       | Project settings (checked into source control)                                                              |
| `.claude/settings.local.json` | Local project settings (not committed)                                                                      |
| `~/.claude.json`              | Global state (theme, OAuth, MCP servers)                                                                    |
| `.mcp.json`                   | Project MCP servers (checked into source control)                                                           |
| `managed-mcp.json`            | [Managed MCP servers](/docs/en/mcp#managed-mcp-configuration)                                               |
| Managed settings              | [Managed settings](/docs/en/settings#settings-files) (server-managed, MDM/OS-level policies, or file-based) |

On Windows, `~` refers to your user home directory, such as `C:\Users\YourName` . For details on configuring these files, see [Settings](/docs/en/settings) and [MCP](/docs/en/mcp) .

#### Resetting configuration

To reset Claude Code to default settings, you can remove the configuration files:

```
### Reset all user settings and state
rm ~/.claude.json
rm -rf ~/.claude/

### Reset project-specific settings
rm -rf .claude/
rm .mcp.json
```

This will remove all your settings, MCP server configurations, and session history.

### Performance and stability

These sections cover issues related to resource usage, responsiveness, and search behavior.

#### High CPU or memory usage

Claude Code is designed to work with most development environments, but may consume significant resources when processing large codebases. If you're experiencing performance issues:

1. Use `/compact` regularly to reduce context size
2. Close and restart Claude Code between major tasks
3. Consider adding large build directories to your `.gitignore` file

#### Auto-compaction stops with a thrashing error

If you see `Autocompact is thrashing: the context refilled to the limit...` , automatic compaction succeeded but a file or tool output immediately refilled the context window several times in a row. Claude Code stops retrying to avoid wasting API calls on a loop that isn't making progress. To recover:

1. Ask Claude to read the oversized file in smaller chunks, such as a specific line range or function, instead of the whole file
2. Run `/compact` with a focus that drops the large output, for example `/compact keep only the plan and the diff`
3. Move the large-file work to a [subagent](/docs/en/sub-agents) so it runs in a separate context window
4. Run `/clear` if the earlier conversation is no longer needed

#### Command hangs or freezes

If Claude Code seems unresponsive:

1. Press Ctrl+C to attempt to cancel the current operation
2. If unresponsive, you may need to close the terminal and restart

#### Search and discovery issues

If Search tool, `@file` mentions, custom agents, and custom skills aren't working, install system `ripgrep` :

```
### macOS (Homebrew)
brew install ripgrep

### Windows (winget)
winget install BurntSushi.ripgrep.MSVC

### Ubuntu/Debian
sudo apt install ripgrep

### Alpine Linux
apk add ripgrep

### Arch Linux
pacman -S ripgrep
```

Then set `USE_BUILTIN_RIPGREP=0` in your [environment](/docs/en/env-vars) .

#### Slow or incomplete search results on WSL

Disk read performance penalties when [working across file systems on WSL](https://learn.microsoft.com/en-us/windows/wsl/filesystems) may result in fewer-than-expected matches when using Claude Code on WSL. Search still functions, but returns fewer results than on a native filesystem.

`/doctor` will show Search as OK in this case.

**Solutions:**

1. **Submit more specific searches** : reduce the number of files searched by specifying directories or file types: "Search for JWT validation logic in the auth-service package" or "Find use of md5 hash in JS files".
2. **Move project to Linux filesystem** : if possible, ensure your project is located on the Linux filesystem ( `/home/` ) rather than the Windows filesystem ( `/mnt/c/` ).
3. **Use native Windows instead** : consider running Claude Code natively on Windows instead of through WSL, for better file system performance.

### IDE integration issues

If Claude Code does not connect to your IDE or behaves unexpectedly within an IDE terminal, try the solutions below.

#### JetBrains IDE not detected on WSL2

If you're using Claude Code on WSL2 with JetBrains IDEs and getting "No available IDEs detected" errors, this is likely due to WSL2's networking configuration or Windows Firewall blocking the connection.

##### WSL2 networking modes

WSL2 uses NAT networking by default, which can prevent IDE detection. You have two options: **Option 1: Configure Windows Firewall** (recommended)

1. Find your WSL2 IP address: `wsl hostname -I # Example output: 172.21.123.45`
2. Open PowerShell as Administrator and create a firewall rule: `New-NetFirewallRule - DisplayName "Allow WSL2 Internal Traffic" - Direction Inbound - Protocol TCP - Action Allow - RemoteAddress 172.21 . 0.0 / 16 - LocalAddress 172.21 . 0.0 / 16` Adjust the IP range based on your WSL2 subnet from step 1.
3. Restart both your IDE and Claude Code

**Option 2: Switch to mirrored networking** Add to `.wslconfig` in your Windows user directory:

```
[wsl2]
networkingMode =mirrored
```

Then restart WSL with `wsl --shutdown` from PowerShell.

These networking issues only affect WSL2. WSL1 uses the host's network directly and doesn't require these configurations.

For additional JetBrains configuration tips, see the [JetBrains IDE guide](/docs/en/jetbrains#plugin-settings) .

#### Report Windows IDE integration issues

If you're experiencing IDE integration problems on Windows, [create an issue](https://github.com/anthropics/claude-code/issues) with the following information:

- Environment type: native Windows (Git Bash) or WSL1/WSL2
- WSL networking mode, if applicable: NAT or mirrored
- IDE name and version
- Claude Code extension/plugin version
- Shell type: Bash, Zsh, PowerShell, etc.

#### Escape key not working in JetBrains IDE terminals

If you're using Claude Code in JetBrains terminals and the `Esc` key doesn't interrupt the agent as expected, this is likely due to a keybinding clash with JetBrains' default shortcuts. To fix this issue:

1. Go to Settings → Tools → Terminal
2. Either:
    - Uncheck "Move focus to the editor with Escape", or
    - Click "Configure terminal keybindings" and delete the "Switch focus to Editor" shortcut
3. Apply the changes

This allows the `Esc` key to properly interrupt Claude Code operations.

### Markdown formatting issues

Claude Code sometimes generates markdown files with missing language tags on code fences, which can affect syntax highlighting and readability in GitHub, editors, and documentation tools.

#### Missing language tags in code blocks

If you notice code blocks like this in generated markdown:

```
```
function example() {
return "hello";
}
```
```

Instead of properly tagged blocks like:

```
```javascript
function example () {
return "hello" ;
}
```
```

**Solutions:**

1. **Ask Claude to add language tags** : request "Add appropriate language tags to all code blocks in this markdown file."
2. **Use post-processing hooks** : set up automatic formatting hooks to detect and add missing language tags. See [Auto-format code after edits](/docs/en/hooks-guide#auto-format-code-after-edits) for an example of a PostToolUse formatting hook.
3. **Manual verification** : after generating markdown files, review them for proper code block formatting and request corrections if needed.

#### Inconsistent spacing and formatting

If generated markdown has excessive blank lines or inconsistent spacing: **Solutions:**

1. **Request formatting corrections** : ask Claude to "Fix spacing and formatting issues in this markdown file."
2. **Use formatting tools** : set up hooks to run markdown formatters like `prettier` or custom formatting scripts on generated markdown files.
3. **Specify formatting preferences** : include formatting requirements in your prompts or project [memory](/docs/en/memory) files.

#### Reduce markdown formatting issues

To minimize formatting issues:

- **Be explicit in requests** : ask for "properly formatted markdown with language-tagged code blocks"
- **Use project conventions** : document your preferred markdown style in [`CLAUDE.md`](/docs/en/memory)
- **Set up validation hooks** : use post-processing hooks to automatically verify and fix common formatting issues

### Get more help

If you're experiencing issues not covered here:

1. Use the `/feedback` command within Claude Code to report problems directly to Anthropic
2. Check the [GitHub repository](https://github.com/anthropics/claude-code) for known issues
3. Run `/doctor` to diagnose issues. It checks:
    - Installation type, version, and search functionality
    - Auto-update status and available versions
    - Invalid settings files (malformed JSON, incorrect types)
    - MCP server configuration errors
    - Keybinding configuration problems
    - Context usage warnings (large CLAUDE.md files, high MCP token usage, unreachable permission rules)
    - Plugin and agent loading errors
4. Ask Claude directly about its capabilities and features - Claude has built-in access to its documentation

Was this page helpful?

Yes

No

[Programmatic usage](/docs/en/headless)

⌘ I
