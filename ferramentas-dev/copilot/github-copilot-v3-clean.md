# GitHub Copilot Documentation


## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Core Concepts](#core-concepts)
- [Copilot CLI](#copilot-cli)
- [Cloud Agent](#cloud-agent)
- [Chat & Code Suggestions](#chat--code-suggestions)
- [Code Review](#code-review)
- [Selected Tutorials](#selected-tutorials)
- [Reference](#reference)


---

# Overview


### Features

You can use Copilot to:

- Get code suggestions as you type in your IDE.
- Chat with Copilot to get help with your code.
- Ask for help using the command line.
- Organize and share context with Copilot Spaces to get more relevant answers.
- Generate descriptions of changes in a pull request.
- Research, plan, make code changes, and create pull requests for you to review. Available in Copilot Pro+, Copilot Business, and Copilot Enterprise only.

Use Copilot in the following places:

- Your IDE
- GitHub Mobile, as a chat interface
- Windows Terminal Canary, through the Terminal Chat interface
- The command line, through the GitHub CLI
- The GitHub website

See [GitHub Copilot features](/en/copilot/about-github-copilot/github-copilot-features) .

### Get access

You can start using Copilot in several ways, depending on your role and needs.

#### Individuals

- **Try Copilot for free.** Use Copilot Free to explore core features with no paid plan required.
- **Subscribe to a paid plan.** Upgrade to Copilot Pro or Copilot Pro+ for full access to premium features and more generous usage limits.
    - Try [Copilot Pro for free](https://github.com/github-copilot/signup?ref_product=copilot&ref_type=trial&ref_style=text&ref_plan=pro) with a one-time 30-day trial.
- **Get free access if you're eligible.** Students, teachers, and open source maintainers may qualify for access to premium features at no cost. See [Access GitHub Copilot for free as a student](/en/copilot/how-tos/copilot-on-github/set-up-copilot/enable-copilot/set-up-for-students) and [Access Copilot Pro for free as a teacher or open source maintainer](/en/copilot/how-tos/copilot-on-github/set-up-copilot/enable-copilot/set-up-for-teachers-and-os-maintainers) .
- **Request access from your organization.** If your organization or enterprise has a GitHub Copilot plan, you can request access by going to [https://github.com/settings/copilot](https://github.com/settings/copilot) and request access under "Get Copilot from an organization."

See [Getting started with a GitHub Copilot plan](/en/copilot/managing-copilot/managing-copilot-as-an-individual-subscriber/getting-started-with-copilot-on-your-personal-account/getting-started-with-a-copilot-plan) for more information.

#### Organizations and enterprises

**Organization owners** can set up Copilot Business for their team by [contacting GitHub's Sales team](https://github.com/enterprise/contact?ref_product=copilot&ref_type=engagement&ref_style=text) . See [Subscribing to GitHub Copilot for your organization](/en/copilot/managing-copilot/managing-github-copilot-in-your-organization/subscribing-to-copilot-for-your-organization) .

If your organization is owned by an enterprise that has a Copilot subscription, you can ask your enterprise owner to enable Copilot for your organization. Go to [https://github.com/settings/copilot](https://github.com/settings/copilot) and request access under "Get Copilot from an organization."

**Enterprise owners** can set up Copilot Business or Copilot Enterprise for your enterprise by [contacting GitHub's Sales team](https://github.com/enterprise/contact?ref_product=copilot&ref_type=engagement&ref_style=text) . See [Subscribing to GitHub Copilot for your enterprise](/en/copilot/managing-copilot/managing-copilot-for-your-enterprise/subscribing-to-copilot-for-your-enterprise) .

If you don't need other GitHub features, you can create an enterprise account specifically for managing Copilot Business licenses. This gives you enterprise-grade authentication without charges for GitHub Enterprise licenses. See [About enterprise accounts for Copilot Business](/en/copilot/concepts/about-enterprise-accounts-for-copilot-business) .

### Next steps

- Learn more about Copilot features. See [GitHub Copilot features](/en/copilot/about-github-copilot/github-copilot-features) .
- Start using Copilot. See [Setting up GitHub Copilot](/en/copilot/setting-up-github-copilot) .

### Further reading

- [Frequently asked questions](https://github.com/features/copilot#faq) about GitHub Copilot
- [GitHub Copilot Trust Center](https://copilot.github.trust.page/)


### Comparing Copilot plans

The tables below show the features available in each Copilot plan.

|                                                       | Copilot Free   | Copilot Student   | Copilot Pro                              | Copilot Pro+      | Copilot Business                   | Copilot Enterprise                 |
|-------------------------------------------------------|----------------|-------------------|------------------------------------------|-------------------|------------------------------------|------------------------------------|
| Pricing                                               | Not applicable | Free              | $10 USD per month  (free for some users) | $39 USD per month | $19 USD per granted seat per month | $39 USD per granted seat per month |
| Premium requests                                      | 50 per month   | 300 per month     | 300 per month                            | 1500 per month    | 300 per user per month             | 1000 per user per month            |
| Purchase additional premium requests at $0.04/request |                |                   |                                          |                   |                                    |                                    |

#### Agents

| Agents                              | Copilot Free                       | Copilot Student   | Copilot Pro   | Copilot Pro+   | Copilot Business   | Copilot Enterprise   |
|-------------------------------------|------------------------------------|-------------------|---------------|----------------|--------------------|----------------------|
| Copilot cloud agent                 |                                    |                   |               |                |                    |                      |
| Agent mode                          |                                    |                   |               |                |                    |                      |
| Copilot code review                 | Only "Review selection" in VS Code |                   |               |                |                    |                      |
| Model Context Protocol (MCP)        |                                    |                   |               |                |                    |                      |
| Third-party Agents (public preview) |                                    |                   |               |                |                    |                      |

#### Chat

| Chat                                                                 | Copilot Free          | Copilot Student                | Copilot Pro                    | Copilot Pro+                   | Copilot Business               | Copilot Enterprise             |
|----------------------------------------------------------------------|-----------------------|--------------------------------|--------------------------------|--------------------------------|--------------------------------|--------------------------------|
| Copilot Chat in IDEs [1](#user-content-fn-1) [2](#user-content-fn-2) | 50 messages per month | Unlimited with included models | Unlimited with included models | Unlimited with included models | Unlimited with included models | Unlimited with included models |
| Inline chat                                                          |                       |                                |                                |                                |                                |                                |
| Slash commands                                                       |                       |                                |                                |                                |                                |                                |
| Copilot Chat in GitHub Mobile                                        |                       |                                |                                |                                |                                |                                |
| Copilot Chat in GitHub                                               |                       |                                |                                |                                |                                |                                |
| Copilot Chat in Windows Terminal                                     |                       |                                |                                |                                |                                |                                |
| Increased GitHub Models rate limits [3](#user-content-fn-3)          |                       |                                |                                |                                |                                |                                |
| Copilot Chat skills in IDEs [4](#user-content-fn-4)                  |                       |                                |                                |                                |                                |                                |

#### Models

| Available models in chat              | Copilot Free   | Copilot Student   | Copilot Pro   | Copilot Pro+   | Copilot Business   | Copilot Enterprise   |
|---------------------------------------|----------------|-------------------|---------------|----------------|--------------------|----------------------|
| Claude Haiku 4.5                      |                |                   |               |                |                    |                      |
| Claude Opus 4.5                       |                |                   |               |                |                    |                      |
| Claude Opus 4.6                       |                |                   |               |                |                    |                      |
| Claude Opus 4.6 (fast mode) (preview) |                |                   |               |                |                    |                      |
| Claude Sonnet 4                       |                |                   |               |                |                    |                      |
| Claude Sonnet 4.5                     |                |                   |               |                |                    |                      |
| Claude Sonnet 4.6                     |                |                   |               |                |                    |                      |
| Gemini 2.5 Pro                        |                |                   |               |                |                    |                      |
| Gemini 3 Flash                        |                |                   |               |                |                    |                      |
| Gemini 3.1 Pro                        |                |                   |               |                |                    |                      |
| GPT-4.1                               |                |                   |               |                |                    |                      |
| GPT-5 mini                            |                |                   |               |                |                    |                      |
| GPT-5.1                               |                |                   |               |                |                    |                      |
| GPT-5.2                               |                |                   |               |                |                    |                      |
| GPT-5.2-Codex                         |                |                   |               |                |                    |                      |
| GPT-5.3-Codex                         |                |                   |               |                |                    |                      |
| GPT-5.4                               |                |                   |               |                |                    |                      |
| GPT-5.4 mini                          |                |                   |               |                |                    |                      |
| Grok Code Fast 1                      |                |                   |               |                |                    |                      |
| Raptor mini                           |                |                   |               |                |                    |                      |
| Goldeneye                             |                |                   |               |                |                    |                      |

#### Inline suggestions

| Inline suggestions                                                      | Copilot Free               | Copilot Student   | Copilot Pro   | Copilot Pro+   | Copilot Business   | Copilot Enterprise   |
|-------------------------------------------------------------------------|----------------------------|-------------------|---------------|----------------|--------------------|----------------------|
| Real-time code suggestions with included models [5](#user-content-fn-5) | 2000 completions per month |                   |               |                |                    |                      |
| Next edit suggestions                                                   |                            |                   |               |                |                    |                      |

#### Customization

| Customization                               | Copilot Free   | Copilot Student   | Copilot Pro   | Copilot Pro+   | Copilot Business   | Copilot Enterprise   |
|---------------------------------------------|----------------|-------------------|---------------|----------------|--------------------|----------------------|
| Repository and personal custom instructions |                |                   |               |                |                    |                      |
| Organization custom instructions            |                |                   |               |                |                    |                      |
| Prompt files                                |                |                   |               |                |                    |                      |
| Model Context Protocol (MCP)                |                |                   |               |                |                    |                      |
| Block suggestions matching public code      |                |                   |               |                |                    |                      |
| Exclude specified files from Copilot        |                |                   |               |                |                    |                      |
| Organization-wide policy management         |                |                   |               |                |                    |                      |

#### Other features

|                                | Copilot Free   | Copilot Student   | Copilot Pro   | Copilot Pro+   | Copilot Business   | Copilot Enterprise   |
|--------------------------------|----------------|-------------------|---------------|----------------|--------------------|----------------------|
| Copilot pull request summaries |                |                   |               |                |                    |                      |
| Audit logs                     |                |                   |               |                |                    |                      |
| Content exclusion              |                |                   |               |                |                    |                      |
| Copilot CLI                    |                |                   |               |                |                    |                      |
| GitHub Spark (public preview)  |                |                   |               |                |                    |                      |

For more information, see [GitHub Copilot features](/en/copilot/about-github-copilot/github-copilot-features) .

### Ready to choose a plan?

Start using Copilot by signing up for the plan that best fits your needs.

- **Copilot Free** - Try Copilot with limited features and requests. [Start using Copilot Free](https://github.com/copilot?ref_product=copilot&ref_type=engagement&ref_style=text&ref_plan=free) .
- **GitHub Copilot Student** - Get access to Copilot's premium features for free. [Access GitHub Copilot Student](/en/copilot/how-tos/copilot-on-github/set-up-copilot/enable-copilot/set-up-for-students) .
- **Copilot Pro** - Get unlimited completions and access to premium models. [Subscribe to Copilot Pro](https://github.com/github-copilot/signup?ref_product=copilot&ref_type=purchase&ref_style=text&ref_plan=pro) .
- **Copilot Pro+** - Unlock advanced AI models, extended request limits, and extra capabilities. [Subscribe to Copilot Pro+](https://github.com/github-copilot/signup?ref_product=copilot&ref_type=purchase&ref_style=text&ref_plan=pro) .
- **Copilot Business** - For teams and organizations. [Contact Sales](https://github.com/enterprise/contact?ref_product=copilot&ref_type=purchase&ref_style=text) .
- **Copilot Enterprise** - For enterprises that need advanced features and centralized management. [Contact Sales](https://github.com/enterprise/contact?ref_product=copilot&ref_type=purchase&ref_style=text) .

### Footnotes

1. Copilot Chat in IDEs is available in Visual Studio Code, Visual Studio, JetBrains IDEs, Eclipse, and Xcode. [↩](#user-content-fnref-1)
2. Response times may vary during periods of high usage. [↩](#user-content-fnref-2)
3. For details about the increased rate limits, see [Prototyping with AI models](/en/github-models/prototyping-with-ai-models) . [↩](#user-content-fnref-3)
4. Copilot Chat skills in IDEs is available in Visual Studio Code and Visual Studio. [↩](#user-content-fnref-4)
5. Inline suggestions in IDEs is available in Visual Studio Code, Visual Studio, JetBrains IDEs, Azure Data Studio, Xcode, Vim/Neovim, and Eclipse. [↩](#user-content-fnref-5)


### GitHub Copilot features

#### Copilot Chat

A chat interface that lets you ask coding-related questions. GitHub Copilot Chat is available on the GitHub website, in GitHub Mobile, in supported IDEs *(Visual Studio Code, Visual Studio, JetBrains IDEs, Eclipse IDE, and Xcode)* , and in Windows Terminal. Users can also use skills with Copilot Chat. See [Asking GitHub Copilot questions in GitHub](/en/copilot/using-github-copilot/asking-github-copilot-questions-in-github) and [Asking GitHub Copilot questions in your IDE](/en/copilot/using-github-copilot/asking-github-copilot-questions-in-your-ide) .

#### Copilot cloud agent (formerly Copilot coding agent)

An autonomous AI agent that can research a repository, create an implementation plan, and make code changes on a branch. You can review the diff, iterate, and create a pull request when you're ready. You can also assign a GitHub issue to Copilot or ask it to open a pull request directly to complete a task. See [GitHub Copilot cloud agent](/en/copilot/how-tos/use-copilot-agents/cloud-agent) .

#### Third-party coding agents (public preview)

You can use third-party coding agents alongside Copilot cloud agent. See [About third-party agents](/en/copilot/concepts/agents/about-third-party-agents) .

#### Copilot CLI

A command line interface that lets you use Copilot from within the terminal. You can get answers to questions, or you can ask Copilot to make changes to your local files. You can also use Copilot CLI to interact with GitHub.com-for example, listing your open pull requests, or asking Copilot to create an issue. See [About GitHub Copilot CLI](/en/copilot/concepts/agents/about-copilot-cli) .

#### Copilot code review

AI-generated code review suggestions to help you write better code. See [Using GitHub Copilot code review](/en/copilot/using-github-copilot/code-review/using-copilot-code-review) .

Several tools in Copilot code review are in public preview and subject to change. See [About GitHub Copilot code review](/en/copilot/concepts/agents/code-review) .

#### Copilot pull request summaries

AI-generated summaries of the changes that were made in a pull request, which files they impact, and what a reviewer should focus on when they conduct their review. See [Creating a pull request summary with GitHub Copilot](/en/copilot/using-github-copilot/creating-a-pull-request-summary-with-github-copilot) .

#### Inline suggestions

Autocomplete-style suggestions from Copilot in supported IDEs (Visual Studio Code, Visual Studio, JetBrains IDEs, Azure Data Studio, Xcode, Vim/Neovim, and Eclipse). See [Getting code suggestions in your IDE with GitHub Copilot](/en/copilot/using-github-copilot/getting-code-suggestions-in-your-ide-with-github-copilot) .

If you use VS Code, Xcode, and Eclipse, you can also use next edit suggestions, which will predict the location of the next edit you are likely to make and suggest a completion for it.

#### Copilot Edits

Copilot Edits is available in Visual Studio Code, Visual Studio, and JetBrains IDEs. Use Copilot Edits to make changes across multiple files directly from a single Copilot Chat prompt. Copilot Edits has the following modes:

##### Edit mode

Edit mode is only available in Visual Studio Code and JetBrains IDEs.

Use edit mode when you want more granular control over the edits that Copilot proposes. In edit mode, you choose which files Copilot can make changes to, provide context to Copilot with each iteration, and decide whether or not to accept the suggested edits after each turn.

Edit mode is best suited to use cases where:

- You want to make a quick, specific update to a defined set of files.
- You want full control over the number of LLM requests Copilot uses.

##### Agent mode

Use agent mode when you have a specific task in mind and want to enable Copilot to autonomously edit your code. In agent mode, Copilot determines which files to make changes to, offers code changes and terminal commands to complete the task, and iterates to remediate issues until the original task is complete.

Agent mode is best suited to use cases where:

- Your task is complex, and involves multiple steps, iterations, and error handling.
- You want Copilot to determine the necessary steps to take to complete the task.
- The task requires Copilot to integrate with external applications, such as an MCP server.

#### Copilot custom instructions

Enhance Copilot Chat responses by providing contextual details on your preferences, tools, and requirements. See [About customizing GitHub Copilot responses](/en/copilot/concepts/about-customizing-github-copilot-chat-responses) .

#### Copilot Memory (public preview)

Copilot can deduce and store useful information about a repository, which Copilot cloud agent and Copilot code review can use to improve the quality of their output when working in that repository. For more information, see [About agentic memory for GitHub Copilot](/en/copilot/concepts/agents/copilot-memory) .

#### Copilot in GitHub Desktop

Automatically generate commit messages and descriptions with Copilot in GitHub Desktop based on the changes you make to your project.

#### Copilot Spaces

Organize and centralize relevant content-like code, docs, specs, and more-into Spaces that ground Copilot's responses in the right context for a specific task. See [About GitHub Copilot Spaces](/en/copilot/using-github-copilot/copilot-spaces/about-organizing-and-sharing-context-with-copilot-spaces) .

#### GitHub Spark (public preview)

Build and deploy full-stack applications using natural-language prompts that seamlessly integrate with the GitHub platform for advanced development. See [Building and deploying AI-powered apps with GitHub Spark](/en/copilot/tutorials/spark/build-apps-with-spark) .

### GitHub Copilot features for administrators

The following features are available to organization and enterprise owners with a Copilot Business or Copilot Enterprise plan.

#### Policy management

Manage policies for Copilot in your organization or enterprise. See [Managing policies and features for GitHub Copilot in your organization](/en/copilot/managing-copilot/managing-github-copilot-in-your-organization/setting-policies-for-copilot-in-your-organization/managing-policies-for-copilot-in-your-organization) and [Managing policies and features for GitHub Copilot in your enterprise](/en/copilot/managing-copilot/managing-copilot-for-your-enterprise/managing-policies-and-features-for-copilot-in-your-enterprise) .

#### Access management

Enterprise owners can specify which organizations in the enterprise can use Copilot, and organization owners can specify which organization members can use Copilot. See [Managing access to GitHub Copilot in your organization](/en/copilot/managing-copilot/managing-github-copilot-in-your-organization/managing-access-to-github-copilot-in-your-organization) and [Managing access to Copilot in your enterprise](/en/copilot/managing-copilot/managing-copilot-for-your-enterprise/managing-access-to-copilot-in-your-enterprise) .

#### Usage data

Review Copilot usage data within your organization or enterprise to inform how to manage access and drive adoption of Copilot. See [Reviewing user activity data for GitHub Copilot in your organization](/en/copilot/managing-copilot/managing-github-copilot-in-your-organization/reviewing-activity-related-to-github-copilot-in-your-organization/reviewing-user-activity-data-for-copilot-in-your-organization) and [Viewing Copilot license usage in your enterprise](/en/copilot/managing-copilot/managing-copilot-for-your-enterprise/managing-access-to-copilot-in-your-enterprise/viewing-copilot-license-usage-in-your-enterprise) .

#### Audit logs

Review audit logs for Copilot in your organization to understand what actions have been taken and by which users. See [Reviewing audit logs for GitHub Copilot](/en/copilot/managing-copilot/managing-github-copilot-in-your-organization/reviewing-activity-related-to-github-copilot-in-your-organization/reviewing-audit-logs-for-copilot-business) .

#### Exclude files

Configure Copilot to ignore certain files. This can be useful if you have files that you don't want to be available to Copilot. See [Excluding content from GitHub Copilot](/en/copilot/managing-copilot/managing-github-copilot-in-your-organization/setting-policies-for-copilot-in-your-organization/excluding-content-from-github-copilot) .

### Next steps

- To learn more about the plans available for GitHub Copilot, see [Plans for GitHub Copilot](/en/copilot/about-github-copilot/subscription-plans-for-github-copilot) .
- To start using Copilot, see [Setting up GitHub Copilot](/en/copilot/setting-up-github-copilot) .


---

# Getting Started


### Introduction

You can use Copilot to get answers to coding-related questions, such as how best to code something, how to fix a bug, or how someone else's code works. For full details of what Copilot can do, see [What is GitHub Copilot?](/en/copilot/about-github-copilot/what-is-github-copilot) .

Instructions for using Copilot differ depending on where you are using it. This version of the quickstart is for using Copilot on the GitHub website. Click the tabs above for instructions on using Copilot in other environments.

### Sign up for GitHub Copilot

[Get started for free](https://github.com/copilot?ref_product=copilot&ref_type=engagement&ref_style=button&ref_plan=free)

To use Copilot, you'll need a personal GitHub account with access to a Copilot plan. You can:

- Start with Copilot Free to explore limited features without subscribing to a plan.
- Upgrade to Copilot Pro or Copilot Pro+ to unlock more features, models, and request limits.

For more information about the different plans for GitHub Copilot, see [Plans for GitHub Copilot](/en/copilot/about-github-copilot/subscription-plans-for-github-copilot) .

### Asking your first question

1. On GitHub, navigate to a repository and open a file.
2. Click the Copilot icon ( ) at the top right of the file view.


4. Type a question in the "Ask Copilot" box at the bottom of the chat panel and press `Enter` . For example, you could enter: Copilot responds to your request in the panel.
    - Explain this file.
    - How could I improve this code?
    - How can I test this code?
5. You can continue the conversation by asking a follow-up question. For example, you could type "tell me more" to get Copilot to expand on its last comment.

### Other questions you can ask

There are many more things you can do with GitHub Copilot Chat in GitHub. For example:

- Ask a general question about software development
- Ask exploratory questions about a repository
- Find out about the changes in a pull request
- Ask a question about a specific issue or commit

For more information, see [Asking GitHub Copilot questions in GitHub](/en/copilot/github-copilot-chat/copilot-chat-in-github/using-github-copilot-chat-in-githubcom) .

### Next steps

- **Find out more about GitHub Copilot Chat** - See [Asking GitHub Copilot questions in your IDE](/en/copilot/github-copilot-chat/using-github-copilot-chat-in-your-ide) .
- **Get Copilot inline suggestions in an IDE** - See [Getting code suggestions in your IDE with GitHub Copilot](/en/enterprise-cloud@latest/copilot/using-github-copilot/using-github-copilot-code-suggestions-in-your-editor) .
- **Learn how to write effective prompts** - See [Prompt engineering for GitHub Copilot Chat](/en/copilot/using-github-copilot/prompt-engineering-for-github-copilot) .
- **Use Copilot on your mobile device** - See [Asking GitHub Copilot questions in GitHub Mobile](/en/copilot/github-copilot-chat/copilot-chat-in-github-mobile/using-github-copilot-chat-in-github-mobile) .
- **Use Copilot on the command line** - See [Using the GitHub CLI Copilot extension](/en/copilot/github-copilot-in-the-cli/using-github-copilot-in-the-cli) .

GitHub Copilot provides coding suggestions as you type in your editor. You can also ask Copilot coding-related questions, such as how best to code something, how to fix a bug, or how someone else's code works. For full details of what Copilot can do, see [What is GitHub Copilot?](/en/copilot/about-github-copilot/what-is-github-copilot) .

Instructions for using Copilot differ depending on where you are using it. This version of the quickstart is for Windows Terminal. Click the tabs above for instructions on using Copilot in other environments.

### Sign up for GitHub Copilot

[Get started for free](https://github.com/copilot?ref_product=copilot&ref_type=engagement&ref_style=button&ref_plan=free)

To use Copilot, you'll need a personal GitHub account with access to a Copilot plan. You can:

- Start with Copilot Free to explore limited features without subscribing to a plan.
- Upgrade to Copilot Pro or Copilot Pro+ to unlock more features, models, and request limits.

For more information about the different plans for GitHub Copilot, see [Plans for GitHub Copilot](/en/copilot/about-github-copilot/subscription-plans-for-github-copilot) .

### Prerequisites

- **Subscription to Copilot** . To use GitHub Copilot in Windows Terminal, you must have an active GitHub Copilot subscription. See [What is GitHub Copilot?](/en/copilot/about-github-copilot/what-is-github-copilot#getting-access-to-copilot) .
- **Windows Terminal Canary** . Terminal Chat is only available in [Windows Terminal Canary](https://github.com/microsoft/terminal?tab=readme-ov-file#installing-windows-terminal-canary) .

### Use Copilot in Terminal Chat

After you've installed Windows Terminal Canary, you can use Copilot in [Terminal Chat](https://learn.microsoft.com/windows/terminal/terminal-chat) to ask command line-related questions.

1. Open **Settings** from the dropdown menu.


3. Go to the **Terminal Chat (Experimental)** setting.


5. Under **Service Providers** , select **GitHub Copilot** and **Authenticate via GitHub** to sign in.

### Chat with GitHub Copilot

Note

If you have access to GitHub Copilot via your organization, you won't be able to use GitHub Copilot in Windows Terminal if your organization owner has disabled GitHub Copilot CLI. See [Managing policies and features for GitHub Copilot in your organization](/en/copilot/managing-github-copilot-in-your-organization/managing-policies-and-features-for-copilot-in-your-organization) .

1. Open **Terminal Chat (Experimental)** in the dropdown menu.
2. In the Terminal Chat chat window, type `how do i list all markdown files in my directory` then press `Enter` . Copilot's answer is displayed below your question.
3. Click on an answer to insert it to the command line.

### Next steps

- **Find out more about Copilot inline suggestions** - See [Getting code suggestions in your IDE with GitHub Copilot](/en/copilot/using-github-copilot/getting-code-suggestions-in-your-ide-with-github-copilot) .
- **Find out more about GitHub Copilot Chat** - See [Asking GitHub Copilot questions in your IDE](/en/copilot/using-github-copilot/copilot-chat/asking-github-copilot-questions-in-your-ide) .
- **Learn how to write effective prompts** - See [Prompt engineering for GitHub Copilot Chat](/en/copilot/using-github-copilot/copilot-chat/prompt-engineering-for-copilot-chat) .
- **Use Copilot on your mobile device** - See [Asking GitHub Copilot questions in GitHub Mobile](/en/copilot/using-github-copilot/copilot-chat/asking-github-copilot-questions-in-github-mobile) .
- **Use Copilot on the command line** - See [About GitHub Copilot CLI](/en/copilot/concepts/agents/about-copilot-cli) .
- **Configure Copilot in your editor** - You can enable or disable GitHub Copilot from within your editor, and create your own preferred keyboard shortcuts for Copilot. See [Configuring GitHub Copilot in your environment](/en/copilot/managing-copilot/configure-personal-settings/configuring-github-copilot-in-your-environment) .

GitHub Copilot provides coding suggestions as you type in your editor. You can also ask Copilot coding-related questions, such as how best to code something, how to fix a bug, or how someone else's code works. For full details of what Copilot can do, see [What is GitHub Copilot?](/en/copilot/about-github-copilot/what-is-github-copilot) .

Instructions for using Copilot differ depending on where you are using it. This version of the quickstart is for Visual Studio Code. Click the tabs above for instructions on using Copilot in other environments.

### Sign up for GitHub Copilot

[Get started for free](https://github.com/copilot?ref_product=copilot&ref_type=engagement&ref_style=button&ref_plan=free)

To use Copilot, you'll need a personal GitHub account with access to a Copilot plan. You can:

- Start with Copilot Free to explore limited features without subscribing to a plan.
- Upgrade to Copilot Pro or Copilot Pro+ to unlock more features, models, and request limits.

For more information about the different plans for GitHub Copilot, see [Plans for GitHub Copilot](/en/copilot/about-github-copilot/subscription-plans-for-github-copilot) .

### Prerequisites

- **Copilot subscription** - To use GitHub Copilot in VS Code, you must have an active GitHub Copilot subscription. See [What is GitHub Copilot?](/en/copilot/about-github-copilot/what-is-github-copilot#getting-access-to-copilot) .
- **Latest version of Visual Studio Code** . See the [Visual Studio Code download page](https://code.visualstudio.com/Download?ref_product=copilot&ref_type=engagement&ref_style=text) .
- **Sign in to GitHub in Visual Studio Code** . If you experience authentication issues, see [Troubleshooting common issues with GitHub Copilot](/en/copilot/troubleshooting-github-copilot/troubleshooting-issues-with-github-copilot-chat#troubleshooting-authentication-issues-in-your-editor) .

### Chat with GitHub Copilot

After you've installed the GitHub Copilot Chat extension, you can ask Copilot coding-related questions.

Note

If you have access to GitHub Copilot via your organization, you won't be able to use GitHub Copilot Chat if your organization owner has disabled chat. See [Managing policies and features for GitHub Copilot in your organization](/en/copilot/managing-github-copilot-in-your-organization/managing-policies-and-features-for-copilot-in-your-organization) .

1. Create a new folder for your project and open it in VS Code.
2. Open the Chat view by pressing `Control` + `Command` + `i` (Mac) / `Ctrl` + `Alt` + `i` (Windows/Linux) or by selecting the chat icon in the VS Code title bar.
3. At the bottom of the chat view, in the chat input field, type: `Create a complete task manager web application with the ability to add, delete, and mark tasks as completed. Include modern CSS styling and make it responsive. Use semantic HTML and ensure it's accessible. Separate markup, styles, and scripts into their own files.`
4. Press `Enter` . Watch as the agent generates the necessary files and code to implement your request. You should see it update the `index.html` file, create a `styles.css` file for styling, and a `script.js` file for functionality.
5. Review the generated files and select Keep to accept all the changes.

### Get your first inline suggestion

The following example uses JavaScript, however other languages will work similarly. GitHub Copilot provides suggestions for numerous languages and a wide variety of frameworks, but works especially well for Python, JavaScript, TypeScript, Ruby, Go, C# and C++. GitHub Copilot can also assist in query generation for databases, generating suggestions for APIs and frameworks, and can help with infrastructure as code development.

1. Open Visual Studio Code.
2. In Visual Studio Code, create a new JavaScript ( **.js* ) file.
3. In the JavaScript file, type the following function header. JavaScript `function calculateDaysBetweenDates ( begin, end ) {` GitHub Copilot will automatically suggest an entire function body in grayed text. The exact suggestion may vary.
4. To accept the suggestion, press `Tab` .

### Next steps

- **Find out more about Copilot inline suggestions** - See [Getting code suggestions in your IDE with GitHub Copilot](/en/copilot/using-github-copilot/getting-code-suggestions-in-your-ide-with-github-copilot) .
- **Find out more about GitHub Copilot Chat** - See [Asking GitHub Copilot questions in your IDE](/en/copilot/using-github-copilot/copilot-chat/asking-github-copilot-questions-in-your-ide) .
- **Learn how to write effective prompts** - See [Prompt engineering for GitHub Copilot Chat](/en/copilot/using-github-copilot/copilot-chat/prompt-engineering-for-copilot-chat) .
- **Use Copilot like a pro** - Learn how to write effective prompts for GitHub Copilot. For more information, see [Best practices for using GitHub Copilot in VS Code](https://code.visualstudio.com/docs/copilot/prompt-crafting) in the Visual Studio Code documentation.
- **Get familiar with next edit suggestions** - See [Navigating and accepting next edit suggestions](/en/copilot/how-tos/get-code-suggestions/get-ide-code-suggestions#navigating-and-accepting-next-edit-suggestions-1) .
- **Use Copilot on your mobile device** - See [Asking GitHub Copilot questions in GitHub Mobile](/en/copilot/using-github-copilot/copilot-chat/asking-github-copilot-questions-in-github-mobile) .
- **Use Copilot on the command line** - See [About GitHub Copilot CLI](/en/copilot/concepts/agents/about-copilot-cli) .
- **Configure Copilot in your editor** - You can enable or disable GitHub Copilot from within your editor, and create your own preferred keyboard shortcuts for Copilot. See [Configuring GitHub Copilot in your environment](/en/copilot/managing-copilot/configure-personal-settings/configuring-github-copilot-in-your-environment) .

GitHub Copilot provides coding suggestions as you type in your editor. You can also ask Copilot coding-related questions, such as how best to code something, how to fix a bug, or how someone else's code works. For full details of what Copilot can do, see [What is GitHub Copilot?](/en/copilot/about-github-copilot/what-is-github-copilot) .

Instructions for using Copilot differ depending on where you are using it. This version of the quickstart is for Visual Studio. Click the tabs above for instructions on using Copilot in other environments.

### Sign up for GitHub Copilot

[Get started for free](https://github.com/copilot?ref_product=copilot&ref_type=engagement&ref_style=button&ref_plan=free)

To use Copilot, you'll need a personal GitHub account with access to a Copilot plan. You can:

- Start with Copilot Free to explore limited features without subscribing to a plan.
- Upgrade to Copilot Pro or Copilot Pro+ to unlock more features, models, and request limits.

For more information about the different plans for GitHub Copilot, see [Plans for GitHub Copilot](/en/copilot/about-github-copilot/subscription-plans-for-github-copilot) .

### Prerequisites

- **Subscription to Copilot** . To use GitHub Copilot in Visual Studio, you must have an active GitHub Copilot subscription. See [What is GitHub Copilot?](/en/copilot/about-github-copilot/what-is-github-copilot#getting-access-to-copilot) .
- **Compatible version of Visual Studio** . To use GitHub Copilot in Visual Studio, you must have version 2022 17.8 or later of Visual Studio for Windows installed. For more information, see [Install Visual Studio](https://learn.microsoft.com/en-us/visualstudio/install/install-visual-studio?ref_product=copilot&ref_type=engagement&ref_style=text) in the Microsoft documentation.
- **GitHub Copilot extension for Visual Studio** . For instructions on how to install the Copilot extension, see [Install GitHub Copilot in Visual Studio](https://learn.microsoft.com/visualstudio/ide/visual-studio-github-copilot-install-and-states?ref_product=copilot&ref_type=engagement&ref_style=text) in the Microsoft documentation.
- **Add your GitHub account to Visual Studio** . See [Add your GitHub accounts to your Visual Studio keychain](https://learn.microsoft.com/en-us/visualstudio/ide/work-with-github-accounts?ref_product=copilot&ref_type=engagement&ref_style=text) in the Microsoft documentation.

### Chat with GitHub Copilot

After you've installed the GitHub Copilot extension, you can ask Copilot coding-related questions.

Note

If you have access to GitHub Copilot via your organization, you won't be able to use GitHub Copilot Chat if your organization owner has disabled chat. See [Managing policies and features for GitHub Copilot in your organization](/en/copilot/managing-github-copilot-in-your-organization/managing-policies-and-features-for-copilot-in-your-organization) .

1. Open an existing code file.
2. In the Visual Studio menu bar, click **View** , then click **GitHub Copilot Chat** .
3. In the Copilot Chat window, type `what does this file do` then press `Enter` . Copilot's answer is displayed below your question.
4. Select a line of code in the editor.
5. In the Copilot Chat window, type `explain this line` then press `Enter` .

### Get your first inline suggestion

The following example uses JavaScript, however other languages will work similarly. GitHub Copilot provides suggestions for numerous languages and a wide variety of frameworks, but works especially well for Python, JavaScript, TypeScript, Ruby, Go, C# and C++. GitHub Copilot can also assist in query generation for databases, generating suggestions for APIs and frameworks, and can help with infrastructure as code development.

1. Open Visual Studio.
2. In Visual Studio, create a new JavaScript ( **.js* ) file.
3. In the JavaScript file, type the following function header. JavaScript `function calculateDaysBetweenDates ( begin, end ) {` GitHub Copilot will automatically suggest an entire function body in grayed text. The exact suggestion may vary.
4. To accept the suggestion, press `Tab` .

### Next steps

- **Find out more about Copilot inline suggestions** - See [Getting code suggestions in your IDE with GitHub Copilot](/en/copilot/using-github-copilot/getting-code-suggestions-in-your-ide-with-github-copilot) .
- **Find out more about GitHub Copilot Chat** - See [Asking GitHub Copilot questions in your IDE](/en/copilot/using-github-copilot/copilot-chat/asking-github-copilot-questions-in-your-ide) .
- **Learn how to write effective prompts** - See [Prompt engineering for GitHub Copilot Chat](/en/copilot/using-github-copilot/copilot-chat/prompt-engineering-for-copilot-chat) .
- **Prompt like a pro** - Watch [Visual Studio Prompt Engineering with GitHub Copilot](https://www.youtube.com/watch?v=9hZsOeIINg8&list=PLReL099Y5nRckZDdcQ21UigO9pKa14yxC) on YouTube.
- **Use Copilot on your mobile device** - See [Asking GitHub Copilot questions in GitHub Mobile](/en/copilot/using-github-copilot/copilot-chat/asking-github-copilot-questions-in-github-mobile) .
- **Use Copilot on the command line** - See [About GitHub Copilot CLI](/en/copilot/concepts/agents/about-copilot-cli) .
- **Configure Copilot in your editor** - You can enable or disable GitHub Copilot from within your editor, and create your own preferred keyboard shortcuts for Copilot. See [Configuring GitHub Copilot in your environment](/en/copilot/managing-copilot/configure-personal-settings/configuring-github-copilot-in-your-environment) .

GitHub Copilot provides coding suggestions as you type in your editor. You can also ask Copilot coding-related questions, such as how best to code something, how to fix a bug, or how someone else's code works. For full details of what Copilot can do, see [What is GitHub Copilot?](/en/copilot/about-github-copilot/what-is-github-copilot) .

Instructions for using Copilot differ depending on where you are using it. This version of the quickstart is for JetBrains IDEs. Click the tabs above for instructions on using Copilot in other environments.

### Sign up for GitHub Copilot

[Get started for free](https://github.com/copilot?ref_product=copilot&ref_type=engagement&ref_style=button&ref_plan=free)

To use Copilot, you'll need a personal GitHub account with access to a Copilot plan. You can:

- Start with Copilot Free to explore limited features without subscribing to a plan.
- Upgrade to Copilot Pro or Copilot Pro+ to unlock more features, models, and request limits.

For more information about the different plans for GitHub Copilot, see [Plans for GitHub Copilot](/en/copilot/about-github-copilot/subscription-plans-for-github-copilot) .

### Prerequisites

- **Subscription to Copilot** . To use GitHub Copilot in a JetBrains IDE, you must have an active GitHub Copilot subscription. See [What is GitHub Copilot?](/en/copilot/about-github-copilot/what-is-github-copilot#getting-access-to-copilot) .
- **A compatible JetBrains IDE** . Copilot is supported in a large number of JetBrains IDEs. For a full list, see [Asking GitHub Copilot questions in your IDE](/en/copilot/github-copilot-chat/copilot-chat-in-ides/using-github-copilot-chat-in-your-ide?tool=jetbrains) .
- **Latest version of the GitHub Copilot extension** . See the [GitHub Copilot plugin](https://plugins.jetbrains.com/plugin/17718-github-copilot?ref_product=copilot&ref_type=engagement&ref_style=text) in the JetBrains Marketplace. For installation instructions, see [Installing the GitHub Copilot extension in your environment](/en/copilot/configuring-github-copilot/installing-the-github-copilot-extension-in-your-environment) .
- **Sign in to GitHub in your JetBrains IDE** . For authentication instructions, see [Installing the GitHub Copilot extension in your environment](/en/copilot/configuring-github-copilot/installing-the-github-copilot-extension-in-your-environment?tool=jetbrains#installing-the-github-copilot-plugin-in-your-jetbrains-ide) .

### Chat with GitHub Copilot

After you've installed the GitHub Copilot plugin, you can ask Copilot coding-related questions.

Note

If you have access to GitHub Copilot via your organization, you won't be able to use GitHub Copilot Chat if your organization owner has disabled chat. See [Managing policies and features for GitHub Copilot in your organization](/en/copilot/managing-github-copilot-in-your-organization/managing-policies-and-features-for-copilot-in-your-organization) .

1. Open an existing code file.
2. Open the Copilot Chat window by clicking the **Copilot Chat** icon at the right side of the JetBrains IDE window.


4. In the Copilot Chat window, type `what does this file do` then press `Enter` . Copilot's answer is displayed below your question.
5. Select a line of code in the editor.
6. In the Copilot Chat window, type `explain this line` then press `Enter` .

### Get your first inline suggestion

The following example uses JavaScript, however other languages will work similarly. GitHub Copilot provides suggestions for numerous languages and a wide variety of frameworks, but works especially well for Python, JavaScript, TypeScript, Ruby, Go, C# and C++. GitHub Copilot can also assist in query generation for databases, generating suggestions for APIs and frameworks, and can help with infrastructure as code development.

1. In your JetBrains editor, create a new JavaScript ( **.js* ) file.
2. In the JavaScript file, type the following function header. JavaScript `function calculateDaysBetweenDates ( begin, end ) {` GitHub Copilot will automatically suggest an entire function body in grayed text. The exact suggestion may vary.
3. To accept the suggestion, press `Tab` .

### Next steps

- **Find out more about Copilot inline suggestions** - See [Getting code suggestions in your IDE with GitHub Copilot](/en/copilot/using-github-copilot/getting-code-suggestions-in-your-ide-with-github-copilot) .
- **Find out more about GitHub Copilot Chat** - See [Asking GitHub Copilot questions in your IDE](/en/copilot/using-github-copilot/copilot-chat/asking-github-copilot-questions-in-your-ide) .
- **Learn how to write effective prompts** - See [Prompt engineering for GitHub Copilot Chat](/en/copilot/using-github-copilot/copilot-chat/prompt-engineering-for-copilot-chat) .
- **Use Copilot on your mobile device** - See [Asking GitHub Copilot questions in GitHub Mobile](/en/copilot/using-github-copilot/copilot-chat/asking-github-copilot-questions-in-github-mobile) .
- **Use Copilot on the command line** - See [About GitHub Copilot CLI](/en/copilot/concepts/agents/about-copilot-cli) .
- **Configure Copilot in your editor** - You can enable or disable GitHub Copilot from within your editor, and create your own preferred keyboard shortcuts for Copilot. See [Configuring GitHub Copilot in your environment](/en/copilot/managing-copilot/configure-personal-settings/configuring-github-copilot-in-your-environment) .

GitHub Copilot provides coding suggestions as you type in your editor. You can also ask Copilot coding-related questions, such as how best to code something, how to fix a bug, or how someone else's code works. For full details of what Copilot can do, see [What is GitHub Copilot?](/en/copilot/about-github-copilot/what-is-github-copilot) .

Instructions for using Copilot differ depending on where you are using it. This version of the quickstart is for XCode in MacOS. Click the tabs above for instructions on using Copilot in other environments.

### Sign up for GitHub Copilot

[Get started for free](https://github.com/copilot?ref_product=copilot&ref_type=engagement&ref_style=button&ref_plan=free)

To use Copilot, you'll need a personal GitHub account with access to a Copilot plan. You can:

- Start with Copilot Free to explore limited features without subscribing to a plan.
- Upgrade to Copilot Pro or Copilot Pro+ to unlock more features, models, and request limits.

For more information about the different plans for GitHub Copilot, see [Plans for GitHub Copilot](/en/copilot/about-github-copilot/subscription-plans-for-github-copilot) .

### Prerequisites

- **Subscription to Copilot** . To use GitHub Copilot in Xcode, you must have an active GitHub Copilot subscription. See [What is GitHub Copilot?](/en/copilot/about-github-copilot/what-is-github-copilot#getting-access-to-copilot) .
- **Latest version of the GitHub Copilot extension** . For installation instructions, see [Installing the GitHub Copilot extension in your environment](/en/copilot/how-tos/set-up/install-copilot-extension?tool=xcode) .
- **Sign in to GitHub in Xcode** . If you experience authentication issues, see [Troubleshooting common issues with GitHub Copilot](/en/copilot/troubleshooting-github-copilot/troubleshooting-issues-with-github-copilot-chat#troubleshooting-authentication-issues-in-your-editor) .

### Chat with GitHub Copilot

After you've installed the GitHub Copilot plugin, you can ask Copilot coding-related questions.

Note

If you have access to GitHub Copilot via your organization, you won't be able to use GitHub Copilot Chat if your organization owner has disabled chat. See [Managing policies and features for GitHub Copilot in your organization](/en/copilot/managing-github-copilot-in-your-organization/managing-policies-and-features-for-copilot-in-your-organization) .

1. Open an existing code file.
2. Click **Editor** in the menu bar, then click **GitHub Copilot** then **Open Chat** . Copilot Chat opens in a new window.
3. In the Copilot Chat window, select the file to indicate that you want to chat about this file.


5. Type `what does this file do` then press `Enter` . Copilot's answer is displayed below your question.
6. Select a line of code in the editor.
7. In the Copilot Chat window, type `explain this line` then press `Enter` .

### Get your first inline suggestion

The following example uses Swift, however other languages will work similarly.

1. Create a new file called `CalculateDays.swift` .
2. Type the following code in the new file: Swift `import Foundation func calculateDaysBetweenDates ( _ start : Date , _ end : Date )` GitHub Copilot adds a suggestion of code that continues this function. Suggestions are displayed in grayed text.
3. To accept the suggestion, press `Tab` .
4. Copilot will continue to make suggestions, each of which you can accept by pressing `Tab` .

### Next steps

- **Find out more about Copilot inline suggestions** - See [Getting code suggestions in your IDE with GitHub Copilot](/en/copilot/using-github-copilot/getting-code-suggestions-in-your-ide-with-github-copilot) .
- **Find out more about GitHub Copilot Chat** - See [Asking GitHub Copilot questions in your IDE](/en/copilot/using-github-copilot/copilot-chat/asking-github-copilot-questions-in-your-ide) .
- **Learn how to write effective prompts** - See [Prompt engineering for GitHub Copilot Chat](/en/copilot/using-github-copilot/copilot-chat/prompt-engineering-for-copilot-chat) .
- **Get familiar with next edit suggestions** - See [Navigating and accepting next edit suggestions](/en/copilot/how-tos/get-code-suggestions/get-ide-code-suggestions?tool=xcode#navigating-and-accepting-next-edit-suggestions-2) .
- **Use Copilot on your mobile device** - See [Asking GitHub Copilot questions in GitHub Mobile](/en/copilot/using-github-copilot/copilot-chat/asking-github-copilot-questions-in-github-mobile) .
- **Use Copilot on the command line** - See [About GitHub Copilot CLI](/en/copilot/concepts/agents/about-copilot-cli) .
- **Configure Copilot in your editor** - You can enable or disable GitHub Copilot from within your editor, and create your own preferred keyboard shortcuts for Copilot. See [Configuring GitHub Copilot in your environment](/en/copilot/managing-copilot/configure-personal-settings/configuring-github-copilot-in-your-environment) .

GitHub Copilot provides coding suggestions as you type in your editor. You can also ask Copilot coding-related questions, such as how best to code something, how to fix a bug, or how someone else's code works. For full details of what Copilot can do, see [What is GitHub Copilot?](/en/copilot/about-github-copilot/what-is-github-copilot) .

Instructions for using Copilot differ depending on where you are using it. This version of the quickstart is for Eclipse. Click the tabs above for instructions on using Copilot in other environments.

### Sign up for GitHub Copilot

[Get started for free](https://github.com/copilot?ref_product=copilot&ref_type=engagement&ref_style=button&ref_plan=free)

To use Copilot, you'll need a personal GitHub account with access to a Copilot plan. You can:

- Start with Copilot Free to explore limited features without subscribing to a plan.
- Upgrade to Copilot Pro or Copilot Pro+ to unlock more features, models, and request limits.

For more information about the different plans for GitHub Copilot, see [Plans for GitHub Copilot](/en/copilot/about-github-copilot/subscription-plans-for-github-copilot) .

### Prerequisites

- **Subscription to Copilot** . To use GitHub Copilot in Eclipse, you must have an active GitHub Copilot subscription. See [What is GitHub Copilot?](/en/copilot/about-github-copilot/what-is-github-copilot#getting-access-to-copilot) .
- **Latest version of the GitHub Copilot extension** . For installation instructions, see [Installing the GitHub Copilot extension in your environment](/en/copilot/how-tos/set-up/install-copilot-extension?tool=eclipse) .
- **Sign in to GitHub in Eclipse** . If you experience authentication issues, see [Troubleshooting common issues with GitHub Copilot](/en/copilot/troubleshooting-github-copilot/troubleshooting-issues-with-github-copilot-chat#troubleshooting-authentication-issues-in-your-editor) .

### Chat with GitHub Copilot

After you've installed the GitHub Copilot plugin, you can ask Copilot coding-related questions.

Note

If you have access to GitHub Copilot via your organization, you won't be able to use GitHub Copilot Chat if your organization owner has disabled chat. See [Managing policies and features for GitHub Copilot in your organization](/en/copilot/managing-github-copilot-in-your-organization/managing-policies-and-features-for-copilot-in-your-organization) .

1. Open an existing code file.
2. In the menu bar of Eclipse, click **Copilot** , then click **Open Chat** .
3. In the Copilot Chat window, type `what does this file do` then press `Enter` . Copilot's answer is displayed below your question.
4. Select a line of code in the editor.
5. In the Copilot Chat window, type `explain this line` then press `Enter` .

### Get your first inline suggestion

The following example uses Java, however other languages will work similarly.

1. Create a new Java class called `CalculateDaysBetween` .
2. Within the class add the following comment: Java `// Take 2 dates and return the number of days between them` GitHub Copilot adds a suggestion of code to use for this class. Suggestions are displayed in grayed text.
3. To accept the suggestion, press `Tab` .

### Next steps

- **Find out more about Copilot inline suggestions** - See [Getting code suggestions in your IDE with GitHub Copilot](/en/copilot/using-github-copilot/getting-code-suggestions-in-your-ide-with-github-copilot) .
- **Find out more about GitHub Copilot Chat** - See [Asking GitHub Copilot questions in your IDE](/en/copilot/using-github-copilot/copilot-chat/asking-github-copilot-questions-in-your-ide) .
- **Learn how to write effective prompts** - See [Prompt engineering for GitHub Copilot Chat](/en/copilot/using-github-copilot/copilot-chat/prompt-engineering-for-copilot-chat) .
- **Get familiar with next edit suggestions** - See [Navigating and accepting next edit suggestions](/en/copilot/how-tos/get-code-suggestions/get-ide-code-suggestions?tool=eclipse#navigating-and-accepting-next-edit-suggestions-3) .
- **Use Copilot on your mobile device** - See [Asking GitHub Copilot questions in GitHub Mobile](/en/copilot/using-github-copilot/copilot-chat/asking-github-copilot-questions-in-github-mobile) .
- **Use Copilot on the command line** - See [About GitHub Copilot CLI](/en/copilot/concepts/agents/about-copilot-cli) .
- **Configure Copilot in your editor** - You can enable or disable GitHub Copilot from within your editor, and create your own preferred keyboard shortcuts for Copilot. See [Configuring GitHub Copilot in your environment](/en/copilot/managing-copilot/configure-personal-settings/configuring-github-copilot-in-your-environment) .


### Understand Copilot's strengths and weaknesses

GitHub Copilot is an AI coding assistant that helps you write code faster and with less effort, allowing you to focus more energy on problem solving and collaboration. Before you start working with Copilot, it's important to understand when you should and shouldn't use it.

**Some of the things Copilot does best include:**

- Writing tests and repetitive code
- Debugging and correcting syntax
- Explaining and commenting code
- Generating regular expressions

**Copilot is not designed to:**

- Respond to prompts unrelated to coding and technology
- Replace your expertise and skills. Remember that you are in charge, and Copilot is a powerful tool at your service.

### Choose the right Copilot tool for the job

While Copilot inline suggestions and Copilot Chat share some functionality, the two tools are best used in different circumstances.

**Inline suggestions work best for:**

- Completing code snippets, variable names, and functions as you write them
- Generating repetitive code
- Generating code from inline comments in natural language
- Generating tests for test-driven development

**Alternatively, Copilot Chat is best suited for:**

- Answering questions about code in natural language
- Generating large sections of code, then iterating on that code to meet your needs
- Accomplishing specific tasks with keywords and skills. Copilot Chat has built-in keywords and skills designed to provide important context for prompts and accomplish common tasks quickly. Different types of keywords and skills are available in different Copilot Chat platforms. See [Asking GitHub Copilot questions in your IDE](/en/copilot/using-github-copilot/asking-github-copilot-questions-in-your-ide#using-keywords-in-your-prompt) .
- Completing a task as a specific persona. For example, you can tell Copilot Chat that it is a Senior C++ Developer who cares greatly about code quality, readability, and efficiency, then ask it to review your code.

### Create thoughtful prompts

Prompt engineering, or structuring your request so Copilot can easily understand and respond to it, plays a critical role in Copilot's ability to generate a valuable response. Here are a few quick tips you should remember while crafting your prompts:

- Break down complex tasks.
- Be specific about your requirements.
- Provide examples of things like input data, outputs, and implementations.
- Follow good coding practices.

To learn more, see [Prompt engineering for GitHub Copilot Chat](/en/copilot/using-github-copilot/copilot-chat/prompt-engineering-for-copilot-chat) .

### Check Copilot's work

While Copilot is very powerful, it is still a tool capable of making mistakes, and you should always validate the code it suggests. Use the following tips to ensure you are accepting accurate, secure suggestions:

- **Understand suggested code before you implement it.** To ensure you fully understand Copilot's suggestion, you can ask Copilot Chat to explain the code.
- **Review Copilot's suggestions carefully.** Consider not just the functionality and security of the suggested code, but also the readability and maintainability of the code moving forward.
- **Use automated tests and tooling to check Copilot's work.** With the help of tools like linting, code scanning, and IP scanning, you can automate an additional layer of security and accuracy checks.

Tip

Optionally, you may want to check Copilot's work for similarities to existing public code. If you don't want to use similar code, you can turn off suggestions matching public code. See [Managing GitHub Copilot policies as an individual subscriber](/en/copilot/managing-copilot/managing-copilot-as-an-individual-subscriber/managing-copilot-policies-as-an-individual-subscriber#enabling-or-disabling-suggestions-matching-public-code) or [Managing policies and features for GitHub Copilot in your organization](/en/copilot/managing-copilot/managing-github-copilot-in-your-organization/setting-policies-for-copilot-in-your-organization/managing-policies-for-copilot-in-your-organization#policies-for-suggestion-matching) .

### Guide Copilot towards helpful outputs

There are several adjustments you can make to steer Copilot towards more valuable responses:

- **Provide Copilot with helpful context:**
    - If you are using Copilot in your IDE, open relevant files and close irrelevant files.
    - In Copilot Chat, if a particular request is no longer helpful context, delete that request from the conversation. Alternatively, if none of the context of a particular conversation is helpful, start a new conversation.
    - If you are using Copilot Chat in GitHub, provide specific repositories, files, symbols, and more as context. See [Asking GitHub Copilot questions in GitHub](/en/copilot/using-github-copilot/asking-github-copilot-questions-in-githubcom) .
    - If you are using Copilot Chat in your IDE, use keywords to focus Copilot on a specific task or piece of context. See [Asking GitHub Copilot questions in your IDE](/en/copilot/using-github-copilot/asking-github-copilot-questions-in-your-ide#using-keywords-in-your-prompt) .
- **Rewrite your prompts to generate different responses.** If Copilot is not providing a helpful response, try rephrasing your prompt, or even breaking your request down into multiple smaller prompts.
- **Pick the best available suggestion.** When you are using inline suggestions, Copilot might offer more than one suggestion. You can use keyboard shortcuts to quickly look through all available suggestions. For the default keyboard shortcuts for your operating system, see [Configuring GitHub Copilot in your environment](/en/copilot/managing-copilot/configure-personal-settings/configuring-github-copilot-in-your-environment#keyboard-shortcuts-for-github-copilot) .
- **Provide feedback to improve future suggestions.** You can provide feedback in many ways:
    - For inline suggestions, accept or reject Copilot's suggestion.
    - For individual responses in Copilot Chat, click the thumbs up or thumbs down icons next to the response.
    - For Copilot Chat in your IDE, see [Asking GitHub Copilot questions in your IDE](/en/copilot/using-github-copilot/asking-github-copilot-questions-in-your-ide#sharing-feedback) for instructions specific to your environment.
    - For Copilot Chat in GitHub, leave a comment on the [feedback discussion](https://github.com/orgs/community/discussions/110314) .

### Stay up-to-date on Copilot's features

New features are regularly added to Copilot to create new abilities, build on existing features, and improve the user experience. To stay up-to-date with Copilot's features, see the [changelog](https://github.blog/changelog/label/copilot/) .


---

# Core Concepts


### About code suggestions in Visual Studio Code

Copilot in Visual Studio Code provides two kinds of code suggestions:

- **Next edit suggestions** Based on the edits you are making, Copilot both predicts the location of the next edit you'll want to make and what that edit should be. To enable next edit suggestions, see [Configuring GitHub Copilot in your environment](/en/copilot/managing-copilot/configure-personal-settings/configuring-github-copilot-in-your-environment#enabling-next-edit-suggestions) .
- **Ghost text suggestions** Copilot offers coding suggestions as you type. Start typing in the editor, and Copilot provides dimmed ghost text suggestions at your current cursor location. You can also describe something you want to do using natural language within a comment, and Copilot will suggest the code to accomplish your goal.

GitHub Copilot provides suggestions for numerous languages and a wide variety of frameworks, but works especially well for Python, JavaScript, TypeScript, Ruby, Go, C# and C++. GitHub Copilot can also assist in query generation for databases, generating suggestions for APIs and frameworks, and can help with infrastructure as code development.

### About code suggestions in JetBrains IDEs

Copilot offers inline suggestions as you type.

GitHub Copilot provides suggestions for numerous languages and a wide variety of frameworks, but works especially well for Python, JavaScript, TypeScript, Ruby, Go, C# and C++. GitHub Copilot can also assist in query generation for databases, generating suggestions for APIs and frameworks, and can help with infrastructure as code development.

### About code suggestions in Visual Studio

Copilot in Visual Studio provides two kinds of code suggestions:

- **Ghost text suggestions** Copilot offers coding suggestions as you type.
- **Next edit suggestions (public preview)** Based on the edits you are making, Copilot will predict the location of the next edit you are likely to make and suggest a completion for it. Suggestions may span a single symbol, an entire line, or multiple lines, depending on the scope of the potential change. To enable next edit suggestions, see [Configuring GitHub Copilot in your environment](/en/copilot/managing-copilot/configure-personal-settings/configuring-github-copilot-in-your-environment#enabling-next-edit-suggestions) .

GitHub Copilot provides suggestions for numerous languages and a wide variety of frameworks, but works especially well for Python, JavaScript, TypeScript, Ruby, Go, C# and C++. GitHub Copilot can also assist in query generation for databases, generating suggestions for APIs and frameworks, and can help with infrastructure as code development.

### About code suggestions in Vim/Neovim

GitHub Copilot provides inline suggestions as you type in Vim/Neovim.

### About code suggestions in Azure Data Studio

GitHub Copilot provides you with inline suggestions as you create SQL databases in Azure Data Studio.

### About code suggestions in Xcode

GitHub Copilot in Xcode provides two kinds of code suggestions:

- **Ghost text suggestions**
    - Copilot offers coding suggestions as you type. You can also describe something you want to do using natural language within a comment, and Copilot will suggest the code to accomplish your goal.
- **Next edit suggestions (public preview)**
    - Based on the edits you are making, Copilot will predict the location of the next edit you are likely to make and suggest a completion for it. Suggestions may span an entire line, or multiple lines, depending on the scope of the potential change. Next edit suggestions are enabled by default. To disable, see [Configuring GitHub Copilot in your environment](/en/copilot/how-tos/configure-personal-settings/configure-in-ide?tool=xcode#enabling-next-edit-suggestions-2) .

### About code suggestions in Eclipse

GitHub Copilot in Eclipse provides two kinds of code suggestions:

- **Ghost text suggestions**
    - Copilot offers coding suggestions as you type. You can also describe something you want to do using natural language within a comment, and Copilot will suggest the code to accomplish your goal.
- **Next edit suggestions (public preview)**
    - Based on the edits you are making, Copilot will predict the location of the next edit you are likely to make and suggest a completion for it. Suggestions may span a single symbol, an entire line, or multiple lines, depending on the scope of the potential change. To enable next edit suggestions, see [Configuring GitHub Copilot in your environment](/en/copilot/how-tos/configure-personal-settings/configure-in-ide?tool=eclipse#enabling-next-edit-suggestions-3) .

GitHub Copilot provides suggestions for numerous languages and a wide variety of frameworks, but works especially well for Python, JavaScript, TypeScript, Ruby, Go, C# and C++. GitHub Copilot can also assist in query generation for databases, generating suggestions for APIs and frameworks, and can help with infrastructure as code development.

### Code suggestions that match public code

GitHub Copilot checks each suggestion for matches with publicly available code. Matches may be discarded or suggested with a code reference, based on the setting of the "Suggestions matching public code" policy for your account or organization. See [GitHub Copilot code referencing](/en/copilot/concepts/completions/code-referencing) .

### Changing the model used for inline suggestions

You can switch the AI model that's used for Copilot inline suggestions if:

- An alternative model is currently available
- You are using the latest releases of VS Code with the latest version of the GitHub Copilot extension

Changing the model only affects Copilot ghost text suggestions. It does not affect Copilot next edit suggestions.

Note

The list of available models will change over time. When only one model is available for inline suggestions, the model picker will only show that model. Preview models and additional models will be added to the picker as they become available.

For details of how to switch the model for Copilot inline suggestions, see [Changing the AI model for GitHub Copilot inline suggestions](/en/copilot/how-tos/use-ai-models/change-the-completion-model) .

### Effects of switching the AI model

Changing the model that's used for Copilot inline suggestions does not affect the model that's used by Copilot next edit suggestions or Copilot Chat. See [Changing the AI model for GitHub Copilot Chat](/en/copilot/using-github-copilot/ai-models/changing-the-ai-model-for-copilot-chat) .

There are no changes to the data collection and usage policy if you change the AI model.

If you are on a Copilot Free plan, all completions count against your completions quota regardless of the model used. See [Plans for GitHub Copilot](/en/copilot/about-github-copilot/subscription-plans-for-github-copilot#comparing-copilot-subscriptions) .

The setting to enable or disable suggestions that match public code is applied irrespective of which model you choose. See [Finding public code that matches GitHub Copilot suggestions](/en/copilot/using-github-copilot/finding-public-code-that-matches-github-copilot-suggestions) .

### Enabling the model switcher

If you have a Copilot Free or Copilot Pro plan, the model switcher for Copilot inline suggestions is automatically enabled.

If you're using a Copilot Business or Copilot Enterprise plan, the organization or enterprise that provides your plan must enable the **Editor preview features** setting. See [Managing policies and features for GitHub Copilot in your organization](/en/enterprise-cloud@latest/copilot/managing-copilot/managing-github-copilot-in-your-organization/managing-policies-for-copilot-in-your-organization#enabling-copilot-features-in-your-organization) or [Managing policies and features for GitHub Copilot in your enterprise](/en/copilot/managing-copilot/managing-copilot-for-your-enterprise/managing-policies-and-features-for-copilot-in-your-enterprise#configuring-policies-for-github-copilot) .

### Changing the model used for inline suggestions

You can switch the AI model that's used for Copilot inline suggestions if:

- An alternative model is currently available
- You are using Visual Studio 17.14 Preview 2 or later

Note

The list of available models will change over time. When only one model is available for inline suggestions, the model picker will only show that model. Preview models and additional models will be added to the picker as they become available.

For details of how to switch the model for Copilot inline suggestions, see [Changing the AI model for GitHub Copilot inline suggestions](/en/copilot/how-tos/use-ai-models/change-the-completion-model) .

### Effects of switching the AI model

Changing the model that's used for Copilot inline suggestions does not affect the model that's used by Copilot next edit suggestions or Copilot Chat. See [Changing the AI model for GitHub Copilot Chat](/en/copilot/using-github-copilot/ai-models/changing-the-ai-model-for-copilot-chat) .

There are no changes to the data collection and usage policy if you change the AI model.

If you are on a Copilot Free plan, all completions count against your completions quota regardless of the model used. See [Plans for GitHub Copilot](/en/copilot/about-github-copilot/subscription-plans-for-github-copilot#comparing-copilot-subscriptions) .

The setting to enable or disable suggestions that match public code is applied irrespective of which model you choose. See [Finding public code that matches GitHub Copilot suggestions](/en/copilot/using-github-copilot/finding-public-code-that-matches-github-copilot-suggestions) .

### Enabling the model switcher

If you have a Copilot Free or Copilot Pro plan, the model switcher for Copilot inline suggestions is automatically enabled.

If you're using a Copilot Business or Copilot Enterprise plan, the organization or enterprise that provides your plan must enable the **Editor preview features** setting. See [Managing policies and features for GitHub Copilot in your organization](/en/enterprise-cloud@latest/copilot/managing-copilot/managing-github-copilot-in-your-organization/managing-policies-for-copilot-in-your-organization#enabling-copilot-features-in-your-organization) or [Managing policies and features for GitHub Copilot in your enterprise](/en/copilot/managing-copilot/managing-copilot-for-your-enterprise/managing-policies-and-features-for-copilot-in-your-enterprise#configuring-policies-for-github-copilot) .

### Changing the model used for inline suggestions

You can switch the AI model that's used for Copilot inline suggestions if:

- An alternative model is currently available
- You are using the latest release of JetBrains IDEs with the latest version of the GitHub Copilot extension

Note

The list of available models will change over time. When only one model is available for inline suggestions, the model picker will only show that model. Preview models and additional models will be added to the picker as they become available.

For details of how to switch the model for Copilot inline suggestions, see [Changing the AI model for GitHub Copilot inline suggestions](/en/copilot/how-tos/use-ai-models/change-the-completion-model) .

### Effects of switching the AI model

Changing the model that's used for Copilot inline suggestions does not affect the model that's used by Copilot next edit suggestions or Copilot Chat. See [Changing the AI model for GitHub Copilot Chat](/en/copilot/using-github-copilot/ai-models/changing-the-ai-model-for-copilot-chat) .

There are no changes to the data collection and usage policy if you change the AI model.

If you are on a Copilot Free plan, all completions count against your completions quota regardless of the model used. See [Plans for GitHub Copilot](/en/copilot/about-github-copilot/subscription-plans-for-github-copilot#comparing-copilot-subscriptions) .

The setting to enable or disable suggestions that match public code is applied irrespective of which model you choose. See [Finding public code that matches GitHub Copilot suggestions](/en/copilot/using-github-copilot/finding-public-code-that-matches-github-copilot-suggestions) .

### Enabling the model switcher

If you have a Copilot Free or Copilot Pro plan, the model switcher for Copilot inline suggestions is automatically enabled.

If you're using a Copilot Business or Copilot Enterprise plan, the organization or enterprise that provides your plan must enable the **Editor preview features** setting. See [Managing policies and features for GitHub Copilot in your organization](/en/enterprise-cloud@latest/copilot/managing-copilot/managing-github-copilot-in-your-organization/managing-policies-for-copilot-in-your-organization#enabling-copilot-features-in-your-organization) or [Managing policies and features for GitHub Copilot in your enterprise](/en/copilot/managing-copilot/managing-copilot-for-your-enterprise/managing-policies-and-features-for-copilot-in-your-enterprise#configuring-policies-for-github-copilot) .

### Programming languages included in the default model

The following programming languages and technologies are included in the training data for the default LLM used for Copilot inline suggestions:

- C
- C#
- C++
- Clojure
- CSS
- Dart
- Dockerfile
- Elixir
- Emacs Lisp
- Go
- Haskell
- HTML
- Java
- JavaScript
- Julia
- Jupyter Notebook
- Kotlin
- Lua
- MATLAB
- Objective-C
- Perl
- PHP
- PowerShell
- Python
- R
- Ruby
- Rust
- Scala
- Shell
- Swift
- TeX
- TypeScript
- Vue

### Next steps

- [Getting code suggestions in your IDE with GitHub Copilot](/en/copilot/how-tos/completions/getting-code-suggestions-in-your-ide-with-github-copilot)


### About Copilot code referencing in JetBrains IDEs

Copilot code referencing identifies and attributes code suggestions by linking them to their original public sources, helping you understand where the code originates.

If you, or your organization, have allowed suggestions that match public code, GitHub Copilot can provide you with details of the code that a suggestion matches. This happens:

- When you accept a Copilot inline suggestion in the editor.
- When a response in Copilot Chat includes matching code.

#### Code referencing for Copilot inline suggestions

When you accept a Copilot inline suggestion that matches code in a public GitHub repository, information about the matching code is logged. The log entry includes the URLs of files containing matching code, and the name of the license that applies to that code, if any was found. This allows you to review these references and decide how to proceed. For example, you can decide what attribution to use, or whether you want to remove this code from your project.

Note

- Code referencing for inline suggestions only occurs for matches of accepted Copilot suggestions. Code you have written, and Copilot suggestions you have altered, are not checked for matches to public code.
- Typically, matches to public code occur in less than one percent of Copilot suggestions, so you should not expect to see code references for many suggestions.

#### Code referencing for Copilot Chat

When Copilot Chat provides a response that includes code that matches code in a public GitHub repository, this is indicated at the end of the response with a link to display details of the matched code in the editor.

### About Copilot code referencing in Visual Studio Code

Copilot code referencing identifies and attributes code suggestions by linking them to their original public sources, helping you understand where the code originates.

If you, or your organization, have allowed suggestions that match public code, GitHub Copilot can provide you with details of the code that a suggestion matches. This happens:

- When you accept a Copilot inline suggestion in the editor.
- When a response in Copilot Chat includes matching code.

#### Code referencing for Copilot inline suggestions

When you accept a Copilot inline suggestion that matches code in a public GitHub repository, information about the matching code is logged. The log entry includes the URLs of files containing matching code, and the name of the license that applies to that code, if any was found. This allows you to review these references and decide how to proceed. For example, you can decide what attribution to use, or whether you want to remove this code from your project.

Note

- Code referencing for inline suggestions only occurs for matches of accepted Copilot suggestions. Code you have written, and Copilot suggestions you have altered, are not checked for matches to public code.
- Typically, matches to public code occur in less than one percent of Copilot suggestions, so you should not expect to see code references for many suggestions.

#### Code referencing for Copilot Chat

When Copilot Chat provides a response that includes code that matches code in a public GitHub repository, this is indicated at the end of the response with a link to display details of the matched code in the editor.

### About Copilot code referencing on GitHub.com

#### Code referencing for Copilot Chat

If you, or your organization, have allowed suggestions that match public code, then whenever a response from Copilot Chat includes matching code, details of the matches will be included in the response.

Note

Typically, matches to public code occur infrequently, so you should not expect to see code references in many Copilot Chat responses.

#### Code referencing for Copilot cloud agent

When Copilot generates code that matches code in a public GitHub repository, this is indicated in the agent session logs with a link to display details of the matched code. For more information, see [Tracking GitHub Copilot's sessions](/en/copilot/how-tos/use-copilot-agents/cloud-agent/track-copilot-sessions) .

### About Copilot code referencing in Visual Studio

Copilot code referencing identifies and attributes code suggestions by linking them to their original public sources, helping you understand where the code originates.

If you, or your organization, have allowed suggestions that match public code, GitHub Copilot can provide you with details of the code that a suggestion matches. This happens:

- When you accept a Copilot inline suggestion in the editor.
- When a response in Copilot Chat includes matching code.

#### Code referencing for Copilot inline suggestions

When you accept a Copilot inline suggestion that matches code in a public GitHub repository, information about the matching code is logged. The log entry includes the URLs of files containing matching code, and the name of the license that applies to that code, if any was found. This allows you to review these references and decide how to proceed. For example, you can decide what attribution to use, or whether you want to remove this code from your project.

Note

- Code referencing for inline suggestions only occurs for matches of accepted Copilot suggestions. Code you have written, and Copilot suggestions you have altered, are not checked for matches to public code.
- Typically, matches to public code occur in less than one percent of Copilot suggestions, so you should not expect to see code references for many suggestions.

#### Code referencing for Copilot Chat

When Copilot Chat provides a response that includes code that matches code in a public GitHub repository, this is indicated below the suggested code, with a link to display details of the matched code in the output log.

### How code referencing finds matching code

Copilot code referencing compares potential code suggestions and the surrounding code of about 150 characters against an index of all public repositories on GitHub.com.

Code in private GitHub repositories, or code outside of GitHub, is not included in the search process.

### Limitations

The search index is refreshed every few months. As a result, newly committed code, and code from public repositories deleted before the index was created, may not be included in the search. For the same reason, the search may return matches to code that has been deleted or moved since the index was created.

References to matching code are currently available in JetBrains IDEs, Visual Studio, Visual Studio Code, Copilot cloud agent, and on the GitHub website.

### Further reading

- [Finding public code that matches GitHub Copilot suggestions](/en/copilot/how-tos/completions/finding-public-code-that-matches-github-copilot-suggestions)
- [Managing GitHub Copilot policies as an individual subscriber](/en/copilot/how-tos/manage-your-account/managing-copilot-policies-as-an-individual-subscriber)
- [Managing policies and features for GitHub Copilot in your organization](/en/copilot/how-tos/administer/organizations/managing-policies-for-copilot-in-your-organization)


### Overview

GitHub Copilot Chat is the AI-powered chat interface for GitHub Copilot. It allows you to interact with AI models to get coding assistance, explanations, and suggestions in a conversational format.

Copilot Chat can help you with a variety of coding-related tasks, like offering you code suggestions, providing natural language descriptions of a piece of code's functionality and purpose, generating unit tests for your code, and proposing fixes for bugs in your code.

GitHub Copilot Chat is available in various environments:

- GitHub (the website)
- A range of IDEs such as Visual Studio Code, Xcode, and JetBrains IDEs
- GitHub Mobile
- GitHub Copilot CLI

Different environments may have different features and capabilities, but the core functionality remains consistent across platforms. To explore the functionality available in each environment, see the [GitHub Copilot Chat](/en/copilot/how-tos/chat) how-to guides and the [Tutorials for GitHub Copilot](/en/copilot/tutorials) .

### Limitations

Copilot Chat is designed to assist with coding tasks, but you remain responsible for reviewing and validating the code it generates. It may not always produce correct or optimal solutions, and it can sometimes generate code that contains security vulnerabilities or other issues. Always test and review the code before using it in production.

### Customizing Copilot Chat responses

GitHub Copilot in GitHub, Visual Studio Code, and Visual Studio can provide chat responses that are tailored to the way your team works, the tools you use, the specifics of your project, or your personal preferences, if you provide it with enough context to do so. Instead of repeating instructions in each prompt, you can create and save instructions for Copilot Chat to customize what responses you receive.

There are various ways you can create custom instructions for Copilot Chat. These fall into three main categories:

- **Personal instructions** : You can add personal instructions so that all the chat responses you, as a user, receive are tailored to your preferences.
- **Repository instructions** : You can store instructions files in a repository, so that all prompts asked in the context of the repository automatically include the instructions you've defined.
- **Organization instructions** : If you are an organization owner, you can create a custom instructions file for an organization, so that all prompts asked in the context of any repository owned by the organization automatically include the instructions you've defined.

For more information, see [Adding personal custom instructions for GitHub Copilot](/en/copilot/customizing-copilot/adding-personal-custom-instructions-for-github-copilot) , [Adding repository custom instructions for GitHub Copilot](/en/copilot/customizing-copilot/adding-repository-custom-instructions-for-github-copilot) and [Adding organization custom instructions for GitHub Copilot](/en/copilot/customizing-copilot/adding-organization-custom-instructions-for-github-copilot) .

### AI models for Copilot Chat

You can change the model Copilot uses to generate responses to chat prompts. You may find that different models perform better, or provide more useful responses, depending on the type of questions you ask. Options include premium models with advanced capabilities.  See [Changing the AI model for GitHub Copilot Chat](/en/copilot/using-github-copilot/ai-models/changing-the-ai-model-for-copilot-chat) .

### Extending Copilot Chat

Copilot Chat can be extended in a variety of ways to enhance its functionality and integrate it with other tools and services. This can include using the Model Context Protocol (MCP) to provide context-aware AI assistance, or connecting third-party tools to leverage GitHub's AI capabilities.

#### Extending Copilot Chat with MCP

MCP is an open standard that defines how applications share context with large language models (LLMs). MCP provides a standardized way to connect AI models to different data sources and tools, enabling them to work together more effectively.

You can configure MCP servers to provide context to Copilot Chat in various IDEs, such as Visual Studio Code and JetBrains IDEs. For Copilot Chat in GitHub, the GitHub MCP server is automatically configured, enabling Copilot Chat to perform a limited set of tasks, at your request, such as creating branches or merging pull requests. For more information, see [Extending GitHub Copilot Chat with Model Context Protocol (MCP) servers](/en/copilot/how-tos/context/model-context-protocol/extending-copilot-chat-with-mcp) and [Using the GitHub MCP Server in your IDE](/en/copilot/how-tos/context/model-context-protocol/using-the-github-mcp-server) .

#### Further reading

- [GitHub Copilot Chat](/en/copilot/how-tos/chat-with-copilot) how-to guides
- [Using GitHub Copilot CLI](/en/copilot/how-tos/use-copilot-agents/use-copilot-cli)
- [GitHub Copilot Chat Cookbook](/en/copilot/tutorials/copilot-chat-cookbook)


### Introduction

As a developer, when you start working on an existing codebase-perhaps as a new member of the development team-you can read the README for the repository, the coding conventions documentation, and other information to help you understand the repository and how you should work when updating or adding code. This will help you submit good quality pull requests. However, the quality of work you're able to deliver will steadily improve as you work on the codebase and learn more about it. In the same way, by allowing Copilot to build its own understanding of your repository, you can enable it to become increasingly effective over time.

Copilot can develop a persistent understanding of a repository by storing "memories."

Memories are tightly scoped pieces of information about a repository, that are deduced by Copilot as it works on the repository. Memories are:

- Repository-specific.
- Only created in response to Copilot activity initiated by users who have had Copilot Memory enabled.

Memories created by one part of Copilot can be used by another part of Copilot. So, for example, if Copilot cloud agent discovers how your repository handles database connections, Copilot code review can later apply that knowledge to spot inconsistent patterns in a pull request it is reviewing. Similarly, if Copilot code review learns about settings that must stay synchronized in two separate files, then Copilot cloud agent will know that if it alters the settings in one of those files it must update the other file accordingly.

### Benefits of using Copilot Memory

AI that is stateless and doesn't retain an understanding of a codebase between separate human/AI interactions, requires you either to repeatedly explain coding conventions and important details about specific code in your prompts, or to create detailed custom instructions files, which you must then maintain.

Copilot Memory:

- Reduces the burden of repeatedly providing the same details in your prompts.
- Reduces the need for regular, manual maintenance of custom instruction files.

By building and maintaining a persistent, repository-level memory, Copilot develops its own knowledge of your codebase, adapts to your coding requirements, and increases the value it can deliver over time.

### Where is Copilot Memory used?

Currently Copilot Memory is used by Copilot cloud agent and Copilot code review when these features are working on pull requests on the GitHub website, and by Copilot CLI. Memories are only created and used by Copilot when Copilot Memory has been enabled for the user initiating the Copilot operation.

Agentic memory will be extended to other parts of Copilot, and for personal and organizational scopes, in future releases.

### How memories are stored, retained and used

Each memory that Copilot generates is stored with citations. These are references to specific code locations that support the memory. When Copilot finds a memory that relates to the work it is doing, it checks the citations against the current codebase to validate that the information is still accurate and is relevant to the current branch. The memory is only used if it is successfully validated.

To avoid stale memories being retained, resulting in outdated information adversely affecting Copilot's decision making, memories are automatically deleted after 28 days.

If a memory is validated and used by Copilot, then a new memory with the same details may be stored, which increases the longevity of that memory.

Memories can be created from code in pull requests that were closed without being merged. However, the validation mechanism ensures that such memories will not affect Copilot's behavior if there is no substantiating evidence in the current codebase.

Copilot only creates memories in a repository in response to actions taken within that repository by people who have write permission for the repository, and for whom Copilot Memory has been enabled. Memories are repository scoped, not user scoped, so all memories stored for a repository are available for use in Copilot operations initiated by any user who has access to Copilot Memory for that repository. The memories stored for a repository can only be used in Copilot operations on that same repository. In this way, what Copilot learns about a repository stays within that repository, ensuring privacy and security.

If you are the owner of a repository where Copilot Memory is being used, you can review and manually delete the memories for that repository. See [Managing and curating Copilot Memory](/en/copilot/how-tos/use-copilot-agents/copilot-memory) .

### About enabling Copilot Memory

The ability to use Copilot Memory is granted to users, rather than being enabled for repositories. After Copilot Memory has been enabled for a user, Copilot will be able to use agentic memory in any repository in which that person uses GitHub Copilot.

For users who have an individual Copilot subscription to a Copilot Pro or Copilot Pro+ plan, Copilot Memory is enabled by default. These users can disable Copilot Memory in their personal Copilot settings on GitHub.

For enterprise and organization-managed Copilot subscriptions, Copilot Memory is turned off by default and can be enabled in the enterprise or organization settings. When enabled at the enterprise or organization level, Copilot Memory will be available to all organization members who receive a Copilot subscription from that organization.

For more information, see [Managing and curating Copilot Memory](/en/copilot/how-tos/use-copilot-agents/copilot-memory) .


### Introduction

You can use third-party coding agents alongside Copilot cloud agent to work asynchronously on your development tasks. You can assign an existing issue or give a prompt to an agent, which will work on the required changes and create a pull request. When the agent finishes, it will request a review from you, and you can leave pull request comments to ask the agent to iterate.

Coding agents are subject to the same security protections, mitigations, and limitations as Copilot cloud agent. To learn more about how you can use coding agents, see [About GitHub Copilot cloud agent](/en/copilot/concepts/agents/cloud-agent/about-cloud-agent) .

#### Where you can use coding agents

You can kick off tasks with coding agents in the following locations:

- **The Agents tab** : Select an agent under the prompt box in the [Agents tab](https://github.com/copilot/agents?ref_product=copilot&ref_type=engagement&ref_style=text&utm_source=docs-3p-agents-tab-cta&utm_medium=docs&utm_campaign=agent-3p-platform-feb-2026) , then kick off a new task and watch the agent get to work on a pull request.
- **Issues** : Assign the agent to an existing issue in a repository.
- **Pull requests** : Mention `@AGENT_NAME` in a comment on an existing pull request to ask it to make changes.
- On [**GitHub Mobile**](/en/copilot/how-tos/chat-with-copilot/chat-in-mobile) : From the **Home** view, click to start a new agent session.
- In [**Visual Studio Code**](https://code.visualstudio.com/docs/copilot/agents/overview#_create-an-agent-session) : Start a new session in the chat view, or delegate an existing session to a different agent.

#### Making coding agents available

Note

Third-party agents are available in the GitHub Copilot Pro, GitHub Copilot Pro+, GitHub Copilot Business, and GitHub Copilot Enterprise plans.

Before you can assign tasks to coding agents on GitHub, they must be enabled in your account policies.

- For **GitHub Copilot Pro and GitHub Copilot Pro+ subscribers** , see [Managing GitHub Copilot policies as an individual subscriber](/en/copilot/how-tos/manage-your-account/manage-policies#enabling-or-disabling-third-party-agents-in-your-repositories) .
- For **GitHub Copilot Business and GitHub Copilot Enterprise subscribers** , see [Managing policies and features for GitHub Copilot in your organization](/en/copilot/how-tos/administer-copilot/manage-for-organization/manage-policies) or [Managing policies and features for GitHub Copilot in your enterprise](/en/enterprise-cloud@latest/copilot/how-tos/administer-copilot/manage-for-enterprise/manage-enterprise-policies) .

These policies do not apply to **local** agents in Visual Studio Code. To configure agent settings in Visual Studio Code, see [Types of agents](https://code.visualstudio.com/docs/copilot/agents/overview#_types-of-agents) in the Visual Studio Code documentation. To adjust enterprise agent settings in Visual Studio Code, see [Enable or disable the use of agents](https://code.visualstudio.com/docs/enterprise/ai-settings#_enable-or-disable-the-use-of-agents) in the Visual Studio Code documentation.

### Supported coding agents

The following third-party agents are supported on GitHub:

- [Anthropic Claude](/en/copilot/concepts/agents/anthropic-claude)
- [OpenAI Codex](/en/copilot/concepts/agents/openai-codex)

### Usage costs

Coding agents consume **GitHub Actions minutes** and **GitHub Copilot premium requests** . Each agent **session** consumes one premium request.

Within your monthly usage allowance for GitHub Actions and premium requests, you can ask agents to work on coding tasks without incurring any additional costs.

For more information, see [GitHub Copilot licenses](/en/billing/managing-billing-for-your-products/managing-billing-for-github-copilot/about-billing-for-github-copilot) .

### Next steps

- To start managing agents, see [Managing cloud agents](/en/copilot/how-tos/use-copilot-agents/manage-agents) .
- To learn how AI models are hosted and served, see [Hosting of models for GitHub Copilot](/en/copilot/reference/ai-models/model-hosting) .


---

# Copilot CLI


### Introduction

The command-line interface (CLI) for GitHub Copilot allows you to use Copilot directly from your terminal. You can use it to answer questions, write and debug code, and interact with GitHub.com. For example, you can ask Copilot to make some changes to a project and create a pull request.

GitHub Copilot CLI gives you quick access to a powerful AI agent, without having to leave your terminal. It can help you complete tasks more quickly by working on your behalf, and you can work iteratively with GitHub Copilot CLI to build the code you need.

### Supported operating systems

- Linux
- macOS
- Windows from within Powershell and [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/about)

For installation instructions, see [Installing GitHub Copilot CLI](/en/copilot/how-tos/set-up/install-copilot-cli) .

### Modes of use

GitHub Copilot CLI has two user interfaces: interactive and programmatic.

#### Interactive interface

To start an interactive session, enter `copilot` . Within an interactive session, you can have a conversation with Copilot. You can prompt Copilot to perform one or more tasks, and you can give it feedback and steer the direction of the work.


The interactive interface has two modes. In addition to the default ask/execute mode there is also a **plan mode** in which Copilot will build a structured implementation plan for a task you want to complete.

Press `Shift` + `Tab` to cycle between modes. In plan mode, Copilot analyzes your request, asks clarifying questions to understand scope and requirements, and builds a plan before writing any code. This helps you catch misunderstandings before any code is written, and stay in control of complex, multi-step tasks.

#### Programmatic interface

You can also pass the CLI a single prompt directly on the command line. The CLI completes the task and then exits.

To use the CLI programmatically, include the `-p` or `--prompt` command-line option in your command. To allow Copilot to modify and execute files you should also use one of the approval options described later in this article-see [Allowing tools to be used without manual approval](#allowing-tools-to-be-used-without-manual-approval) ). For example:

Bash

```
copilot -p "Show me this week's commits and summarize them" --allow-tool= 'shell(git)'
```

Alternatively, you can use a script to output command-line options and pipe this to `copilot` . For example:

Bash

```
./script-outputting-options.sh | copilot
```

Caution

If you use an automatic approval option such as `--allow-all-tools` , Copilot has the same access as you do to files on your computer, and can run any shell commands that you can run, without getting your prior approval. See [Security considerations](#security-considerations) , later in this article.

### Use cases for GitHub Copilot CLI

The following sections provide examples of tasks you can complete with GitHub Copilot CLI.

#### Local tasks

- From within a project directory you can ask Copilot to make a change to the code in the project. For example: `Change the background-color of H1 headings to dark blue` Copilot finds the CSS file where H1 headings are defined and changes the color value.
- Ask Copilot to tell you about changes to a file: `Show me the last 5 changes made to the CHANGELOG.md file. Who changed the file, when, and give a brief summary of the changes they made`
- Use Copilot to help you improve the code, or documentation, in your project.
    - Suggest improvements to content.js
    - Rewrite the readme in this project to make it more accessible to newcomers
- Use Copilot to help you perform Git operations.
    - Commit the changes to this repo
    - Revert the last commit, leaving the changes unstaged
- Ask Copilot to create an application from scratch-for example, as a proof of concept. `Use the create-next-app kit and tailwind CSS to create a next.js app. The app should be a dashboard built with data from the GitHub API. It should track this project's build success rate, average build duration, number of failed builds, and automated test pass rate. After creating the app, give me easy to follow instructions on how to build, run, and view the app in my browser.`
- Ask Copilot to explain why a change it made is not working as expected, or tell Copilot to fix a problem with the last change it made. For example: `You said: "The application is now running on http://localhost:3002 and is fully functional!" but when I browse to that URL I get "This site can't be reached"`

#### Tasks involving GitHub.com

- Fetch and display details about your work from GitHub.com.
    - `List my open PRs` This lists your open pull requests from any repository on GitHub. For more specific results, include the repository name in your prompt:
    - List all open issues assigned to me in OWNER/REPO
- Ask Copilot to work on an issue: `I've been assigned this issue: https://github.com/octo-org/octo-repo/issues/1234. Start working on this for me in a suitably named branch.`
- Ask Copilot to make file changes and raise a pull request on GitHub.com. Copilot creates a pull request on GitHub.com, on your behalf. You are marked as the pull request author.
    - In the root of this repo, add a Node script called user-info.js that outputs information about the user who ran the script. Create a pull request to add this file to the repo on GitHub.
    - Create a PR that updates the README at https://github.com/octo-org/octo-repo, changing the subheading "How to run" to "Example usage"
- Ask Copilot to create an issue for you on GitHub.com. `Raise an improvement issue in octo-org/octo-repo. In src/someapp/somefile.py the `file = open('data.txt', 'r')` block opens a file but never closes it.`
- Ask Copilot to check the code changes in a pull request. `Check the changes made in PR https://github.com/octo-org/octo-repo/pull/57575. Report any serious errors you find in these changes.` Copilot responds in the CLI with a summary of any problems it finds.
- Manage pull requests from GitHub Copilot CLI.
    - Merge all of the open PRs that I've created in octo-org/octo-repo
    - Close PR #11 on octo-org/octo-repo
- Find specific types of issues. `Use the GitHub MCP server to find good first issues for a new team member to work on from octo-org/octo-repo` Note If you know that a specific MCP server can achieve a particular task, then specifying it in your prompt can help Copilot to deliver the results you want.
- Find specific GitHub Actions workflows. `List any Actions workflows in this repo that add comments to PRs`
- Create a GitHub Actions workflow. `Branch off from main and create a GitHub Actions workflow that will run on pull requests, or can be run manually. The workflow should run eslint to check for problems in the changes made in the PR. If warnings or errors are found these should be shown as messages in the diff view of the PR. I want to prevent code with errors from being merged into main so, if any errors are found, the workflow should cause the PR check to fail. Push the new branch and create a pull request.`

### Steering the conversation

You can interact with Copilot while it's thinking to steer the conversation:

- **Enqueue additional messages** : Send follow-up messages to steer the conversation in a different direction, or queue additional instructions for Copilot to process after it finishes its current response. This makes conversations feel more natural and keeps you in control.
- **Inline feedback on rejection** : When you reject a tool permission request, you can give Copilot inline feedback about the rejection so it can adapt its approach without stopping entirely. This makes the conversation flow more naturally when you want to guide Copilot away from certain actions.

### Automatic context management

GitHub Copilot CLI automatically manages your conversation context:

- **Auto-compaction** : When your conversation approaches 95% of the token limit, Copilot automatically compresses your history in the background without interrupting your workflow. This enables virtually infinite sessions.
- **Manual control** : Use `/compact` to manually compress context anytime. Press `Escape` to cancel if you change your mind.
- **Visualize usage** : The `/context` command shows a detailed token usage breakdown so you can understand how your context window is being used.

### Customizing GitHub Copilot CLI

You can customize GitHub Copilot CLI in a number of ways:

- **Custom instructions** : Custom instructions allow you to give Copilot additional context on your project and how to build, test and validate its changes. All custom instruction files now combine instead of using priority-based fallbacks. For more information, see [Adding custom instructions for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/add-custom-instructions) .
- **Model Context Protocol (MCP) servers** : MCP servers allow you to give Copilot access to different data sources and tools. For more information, see [Using GitHub Copilot CLI](/en/copilot/how-tos/use-copilot-agents/use-copilot-cli#add-an-mcp-server) .
- **Custom agents** : Custom agents allow you to create different specialized versions of Copilot for different tasks. For example, you could customize Copilot to be an expert frontend engineer following your team's guidelines. GitHub Copilot CLI includes specialized custom agents that it automatically delegates common tasks to. For more information, see [Using GitHub Copilot CLI](/en/copilot/how-tos/use-copilot-agents/use-copilot-cli#use-custom-agents) .
- **Hooks** : Hooks allow you to execute custom shell commands at key points during agent execution, enabling you to add validation, logging, security scanning, or workflow automation. See [About hooks](/en/copilot/concepts/agents/cloud-agent/about-hooks) .
- **Skills** : Skills allow you to enhance the ability of Copilot to perform specialized tasks with instructions, scripts, and resources. For more information, see [About agent skills](/en/copilot/concepts/agents/about-agent-skills) .
- **Copilot Memory** : Copilot Memory allows Copilot to build a persistent understanding of your repository by storing "memories", which are pieces of information about coding conventions, patterns, and preferences that Copilot deduces as it works. This reduces the need to repeatedly explain context in your prompts and makes future sessions more productive. For more information, see [About agentic memory for GitHub Copilot](/en/copilot/concepts/agents/copilot-memory) .

### Security considerations

When you use Copilot CLI, Copilot can perform tasks on your behalf, such as executing or modifying files, or running shell commands.

You should therefore always keep security considerations in mind when using Copilot CLI, just as you would when working directly with files yourself, or running commands directly in your terminal. You should always review suggested commands carefully when Copilot CLI requests your approval.

#### Trusted directories

Trusted directories control where Copilot CLI can read, modify, and execute files.

You should only launch Copilot CLI from directories that you trust. You should not use Copilot CLI in directories that may contain executable files you can't be sure you trust. Similarly, if you launch the CLI from a directory that contains sensitive or confidential data, or files that you don't want to be changed, you could inadvertently expose those files to risk. Typically, you should not launch Copilot CLI from your home directory.

Scoping of permissions is heuristic and GitHub does not guarantee that all files outside trusted directories will be protected. See [Risk mitigation](#risk-mitigation) .

When you start a GitHub Copilot CLI session, you'll be asked to confirm that you trust the files in, and below, the directory from which you launched the CLI. See [Configure GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/configure-copilot-cli#setting-trusted-directories) .

#### Allowed tools

The first time that Copilot needs to use a tool that could be used to modify or execute a file-for example, `touch` , `chmod` , `node` , or `sed` -it will ask you whether you want to allow it to use that tool.

Typically, you can choose from three options:

```
1. Yes
2. Yes, and approve TOOL for the rest of the running session
3. No, and tell Copilot what to do differently (Esc)
```

**Option 1** allows Copilot to run this particular command, this time only. The next time it needs to use this tool, it will ask you again.

**Option 2** allows Copilot to use this tool again, without asking you for permission, for the duration of the currently running session. It will ask for your approval again in new sessions, or if you resume the current session in the future. If you choose this option, you are allowing Copilot to use this tool in any way it thinks is appropriate. For example, if Copilot asks you to allow it to run the command `rm ./this-file.txt` , and you choose option 2, then Copilot can run any `rm` command (for example, `rm -rf ./*` ) during the current run of this session, without asking for your approval.

**Option 3** cancels the proposed command and allows you to tell Copilot to try a different approach.

##### Allowing tools to be used without manual approval

There are three command-line options that you can use, in either interactive or programmatic sessions, to determine tools that Copilot can use without asking for your approval:

- **`--allow-all-tools`** Allows Copilot to use any tool without asking for your approval. For example, you can use this option with a programmatic invocation of the CLI to allow Copilot to run any command. For example: `copilot -p "Revert the last commit" --allow-all-tools`
- **`--deny-tool`** Prevents Copilot from using a specific tool. This option takes precedence over the `--allow-all-tools` and `--allow-tool` options.
- **`--allow-tool`** Allows Copilot to use a specific tool without asking for your approval.

##### Using the approval options

The `--deny-tool` and `--allow-tool` options require one of the following arguments:

- `'shell(COMMAND)'` For example, `copilot --deny-tool='shell(rm)'` prevents Copilot from using any `rm` command. For `git` and `gh` commands, you can specify a particular first-level subcommand to allow or deny. For example: `copilot --deny-tool='shell(git push)'` The tool specification is optional. For example, `copilot --allow-tool='shell'` allows Copilot to use any shell command without individual approval.
- `'write'` This argument allows or denies tools-other than shell commands-permission to modify files. For example, `copilot --allow-tool='write'` allows Copilot to edit files without your individual approval.
- `'MCP_SERVER_NAME'` This argument allows or denies tools from the specified MCP server, where `MCP_SERVER_NAME` is the name of an MCP server that you have configured. Tools from the server are specified in parentheses, using the tool name that is registered with the MCP server. Using the server name without specifying a tool allows or denies all tools from that server. For example, `copilot --deny-tool='My-MCP-Server(tool_name)'` prevents Copilot from using the tool called `tool_name` from the MCP server called `My-MCP-Server` . You can find an MCP server's name by entering `/mcp` in the CLI's interactive interface, then selecting the server from the list that's displayed.

##### Combining approval options

You can use a combination of approval options to determine exactly which tools Copilot can use without asking for your approval.

For example, to prevent Copilot from using the `rm` and `git push` commands, but automatically allow all other tools, use:

```
copilot --allow-all-tools --deny-tool='shell(rm)' --deny-tool='shell(git push)'
```

To prevent Copilot from using the tool `tool_name` from the MCP server named `My-MCP-Server` , but allow all other tools from that server to be used without individual approval, use:

```
copilot --allow-tool='My-MCP-Server' --deny-tool='My-MCP-Server(tool_name)'
```

##### Security implications of automatic tool approval

It's important to be aware of the security implications of using the approval command-line options. These options allow Copilot to execute commands needed to complete your request, without giving you the opportunity to review and approve those commands before they are run. While this streamlines workflows, and allows headless operation of the CLI, it increases the risk of unintended actions being taken that might result in data loss or corruption, or other security issues.

You can control which tools Copilot CLI can use by responding to approval prompts when Copilot attempts to use a tool, by specifying permissions with command-line flags, or (in an interactive session) by using slash commands (such as `/allow-all` and `/yolo` . See [Configure GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/configure-copilot-cli#setting-allowed-tools) .

#### Risk mitigation

You can mitigate the risks associated with using the automatic approval options by running Copilot CLI in a restricted environment-such as a virtual machine, container, or dedicated system-with tightly controlled permissions and network access. This confines any potential damage that could occur when allowing Copilot to execute commands that you have not reviewed and verified.

#### Known MCP server policy limitations

Copilot CLI can't currently support the following organization-level MCP server policies:

- **MCP servers in Copilot** , which controls whether MCP servers can be used at all by Copilot.
- **MCP Registry URL** , which controls which MCP registry Copilot will allow MCP servers to be used from.

For more information about these policies, see [MCP server usage in your company](/en/copilot/concepts/mcp-management#mcp-policy-settings) .

### Model usage

The default model used by GitHub Copilot CLI is Claude Sonnet 4.5. GitHub reserves the right to change this model.

You can change the model used by GitHub Copilot CLI by using the `/model` slash command or the `--model` command-line option. Enter this command, then select a model from the list.

Each time you submit a prompt to Copilot in Copilot CLI's interactive interface, and each time you use Copilot CLI programmatically, your monthly quota of Copilot premium requests is reduced by one, multiplied by the multiplier shown in parentheses in the model list. For example, `Claude Sonnet 4.5 (1x)` indicates that with this model each time you submit a prompt your quota of premium requests is reduced by one. For information about premium requests, see [Requests in GitHub Copilot](/en/copilot/concepts/billing/copilot-requests) .

#### Using your own model provider

You can configure Copilot CLI to use your own model provider instead of GitHub-hosted models. This lets you connect to an OpenAI-compatible endpoint, Azure OpenAI, or Anthropic, including locally running models such as Ollama. You configure your model provider using environment variables.

| Environment variable        | Description                                                                                                                                               |
|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `COPILOT_PROVIDER_BASE_URL` | The base URL of your model provider's API endpoint.                                                                                                       |
| `COPILOT_PROVIDER_TYPE`     | The provider type: `openai` (default), `azure` , or `anthropic` . The `openai` type works with any OpenAI-compatible endpoint, including Ollama and vLLM. |
| `COPILOT_PROVIDER_API_KEY`  | Your API key for authenticating with the provider. Not required for providers that don't use authentication, such as a local Ollama instance.             |
| `COPILOT_MODEL`             | The model to use (required when using a custom provider). You can also set this with the `--model` command-line option.                                   |

Models used with Copilot CLI must support **tool calling** (function calling) and **streaming** . If the model does not support these capabilities, Copilot CLI will return an error. For best results, the model should have a context window of at least 128k tokens.

For details on how to configure your model provider, run `copilot help providers` in your terminal.

### Use Copilot CLI via ACP

ACP (the Agent Client Protocol) is an open standard for interacting with AI agents. It allows you to use Copilot CLI as an agent in any third-party tools, IDEs, or automation systems that support this protocol.

For more information, see [Copilot CLI ACP server](/en/copilot/reference/copilot-cli-reference/acp-server) .

### Feedback

If you have any feedback about GitHub Copilot CLI, please let us know by using the `/feedback` slash command in an interactive session and choosing one of the options. You can complete a private feedback survey, submit a bug report, or suggest a new feature.

### Further reading

- [Installing GitHub Copilot CLI](/en/copilot/how-tos/set-up/install-copilot-cli)
- [Using GitHub Copilot CLI](/en/copilot/how-tos/use-copilot-agents/use-copilot-cli)
- [Responsible use of GitHub Copilot CLI](/en/enterprise-cloud@latest/copilot/responsible-use/copilot-cli)


### About custom agents

Custom agents are specialized versions of the Copilot agent that you can tailor to your unique workflows, coding conventions, and use cases. They act like tailored teammates that follow your standards, use the right tools, and implement team-specific practices. You define these agents once instead of repeatedly providing the same instructions and context.

You define custom agents using Markdown files called agent profiles. These files specify prompts, tools, and MCP servers. This allows you to encode your conventions, frameworks, and desired outcomes directly into Copilot.

The agent profile defines the custom agent's behavior. When you assign the agent to a task or issue, it instantiates the custom agent.

In addition to any custom agents you define yourself, Copilot includes a set of pre-built custom agents. See [Built-in agents](#built-in-agents) .

### Agent profile format

Agent profiles are Markdown files with YAML frontmatter. In their simplest form, they include:

- **Name** (optional): A display name for the custom agent. If omitted, the agent's filename is used as its identifier and default display name.
- **Description** : Explains the agent's purpose and capabilities.
- **Prompt** : Custom instructions that define the agent's behavior and expertise.
- **Tools** (optional): Specific tools the agent can access. By default, agents can access all available tools, including built-in tools, and MCP server tools.

Agent profiles can also include MCP server configurations using the `mcp-servers` property.

#### Example agent profile

This example is a basic agent profile with name, description, and prompt configured.

```
---
name: readme-creator
description: Agent specializing in creating and improving README files

You are a documentation specialist focused on README files. Your scope is limited to README files or other related documentation files only - do not modify or analyze code files.

Focus on the following instructions:
- Create and update README.md files with clear project descriptions
- Structure README sections logically: overview, installation, usage, contributing
- Write scannable content with proper headings and formatting
- Add appropriate badges, links, and navigation elements
- Use relative links (e.g., `docs/CONTRIBUTING.md`) instead of absolute URLs for files within the repository
- Make links descriptive and add alt text to images
```

### Where you can configure custom agents

You can define agent profiles at different levels:

- **Repository level** : Create `.github/agents/CUSTOM-AGENT-NAME.md` in your repository for project-specific agents.
- **Organization or enterprise level** : Create `/agents/CUSTOM-AGENT-NAME.md` in a `.github-private` repository for broader availability.

For more information, see [Preparing to use custom agents in your organization](/en/copilot/how-tos/administer-copilot/manage-for-organization/prepare-for-custom-agents) and [Preparing to use custom agents in your enterprise](/en/copilot/how-tos/administer-copilot/manage-for-enterprise/manage-agents/prepare-for-custom-agents) .

### Built-in agents

In addition to the main Copilot agent, which processes your request when you submit a prompt, Copilot CLI includes the following built-in agents which the main agent can run as subagents to assist with common development tasks. These agents are optimized for efficiency and accuracy, leveraging the capabilities of the underlying language models and tools to provide high-quality assistance in their respective domains.

Copilot will automatically use an appropriate built-in agent based on your prompt and the current context. For example, the prompt `How does authentication work in this codebase?` will typically trigger the Explore agent, and using the `/research` slash command will trigger the Research agent.

- **explore** - A fast, lightweight codebase exploration agent. It uses code intelligence, grep, glob, view, and shell tools to search files and understand code structure. It will not change any files, so can be called in parallel to other subagents being run by the main Copilot agent. It has read-only access to GitHub MCP server tools.
- **task** - A command execution agent that runs development commands (tests, builds, linters, formatters, dependency installs) and reports results efficiently. It returns a brief summary on success, and full output on failure, keeping the main context clean. It has access to all of the tools the parent agent can use (excluding some that are not appropriate in a subagent context), with the same permissions granted or denied.
- **general-purpose** - This agent essentially has all of the same capabilities as the main Copilot agent. The main agent can run the general-purpose agent as a subagent to assist with any task that requires a separate context window, or to run in parallel when appropriate.
- **code-review** - Reviews code changes with an extremely high signal-to-noise ratio. This agent analyzes staged/unstaged changes and branch diffs, surfacing only issues that genuinely matter: bugs, security vulnerabilities, race conditions, memory leaks, and logic errors. It never comments on style or formatting. It will not make any changes to files.
- **research** - This agent operates as a staff-level software engineer and research specialist. It provides exhaustive, meticulously researched answers about codebases, APIs, libraries, and software architecture. It uses GitHub search/exploration tools, web fetch/search, and local tools. Unlike the other agents, the research agent can only be invoked by using the `/research` slash command. It cannot be automatically triggered by the main agent.

### Running agents as subagents

One of the benefits of using custom agents you have defined yourself-or the built-in agents-is that the main Copilot agent can run them as subagents with a separate context window. This means that your custom agent, or built-in agent, can focus on a specific subtask without cluttering the context window of the main agent.

Where appropriate, tasks performed by subagents can be run in parallel, allowing the overall task to be completed more quickly.

For more information, see [Comparing GitHub Copilot CLI customization features](/en/copilot/concepts/agents/copilot-cli/comparing-cli-features) .

### Next steps

To create your own custom agents, see:

- [Creating and using custom agents for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/create-custom-agents-for-cli)
- [Copilot customization cheat sheet](/en/copilot/reference/customization-cheat-sheet)


### What is a plugin?

- A distributable package that extends Copilot CLI's functionality.
- A bundle of components in a single installable unit.

### What plugins contain

A plugin can contain some or all of the following components:

- **Custom agents** - Specialized AI assistants ( `*.agent.md` files in `agents/` )
- **Skills** - Discrete callable capabilities (skills subdirectories in `skills/` , containing a `SKILL.md` file)
- **Hooks** - Event handlers that intercept agent behavior (a `hooks.json` file in the plugin root, or in `hooks/` )
- **MCP server configurations** - Model Context Protocol integrations (a `.mcp.json` file in the plugin root, or an `mcp.json` file in `.github/` )
- **LSP server configurations** - Language Server Protocol integrations (an `lsp.json` file in the plugin root, or in `.github/` )

### Why use plugins?

Plugins provide the following benefits:

- Reusability across projects
- Team standardization of CLI configuration
- Share domain expertise (for example, by providing the skills of a Rails expert, or a Kubernetes expert)
- Encapsulate complex MCP server setups

### Where can I get plugins?

You can install plugins from:

- A marketplace
- A repository
- A local path

A marketplace is a location where developers can publish, discover, install, and manage plugins. It's a bit like an app store-but for plugins.

Examples of marketplaces include:

- [copilot-plugins](https://github.com/github/copilot-plugins) (added by default)
- [awesome-copilot](https://github.com/github/awesome-copilot) (added by default)
- [claude-code-plugins](https://github.com/anthropics/claude-code)
- [claudeforge-marketplace](https://github.com/claudeforge/marketplace)

For more about adding marketplaces and installing plugins from them, see [Finding and installing plugins for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/plugins-finding-installing) .

### Plugins compared with manual configuration

Any functionality that you could add with a plugin, you could also add by configuring Copilot CLI manually-for example, by adding custom agent profiles or MCP servers. However, plugins provide several advantages over manual configuration:

| Feature    | Manual configuration in a repository   | Plugin                    |
|------------|----------------------------------------|---------------------------|
| Scope      | Single repository                      | Any project               |
| Sharing    | Manual copy/paste                      | `/plugin install` command |
| Versioning | Git history                            | Marketplace versions      |
| Discovery  | Searching repositories                 | Marketplace browsing      |

### Further reading

- [Creating a plugin for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/plugins-creating)
- [GitHub Copilot CLI plugin reference](/en/copilot/reference/cli-plugin-reference)
- [GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli)


### Introduction

Copilot CLI is a terminal-based AI agent that can answer questions, plan work, and complete tasks on your behalf. It's designed to be highly extensible, and there are various ways in which you can customize its behavior and extend its capabilities.

This article explains the difference between:

- **Custom instructions** These tell Copilot **how to behave** in general. For example, to ensure any code that Copilot writes conforms to your coding standards. [Find out more](#custom-instructions) .
- **Skills** These tell Copilot **how to handle a specific kind of task** . For example, to use a particular tool when working on a specific type of task. [Find out more](#skills) .
- **Tools** These **provide abilities** . For example, for finding and modifying files, or for interacting with parts of GitHub. [Find out more](#tools) .
- **MCP servers** These **add collections of tools** that allow Copilot to interact with external services. [Find out more](#mcp-servers) .
- **Hooks** These let you **run your own logic at specific lifecycle moments** . For example, you can run a specific script every time a CLI session starts or ends. [Find out more](#hooks) .
- **Subagents** These are **delegated agent processes** , tied to the main agent and used to perform specific tasks separately from the main agent process. They have their own context window, which can be populated without affecting the main agent's context. [Find out more](#subagents) .
- **Custom agents** These are **definitions of specialized abilities** , designed to perform specific tasks. The main CLI agent can delegate a task to a subagent, using a custom agent profile, to apply specialist knowledge and a particular approach to the task. For example, a custom agent might perform the role of a React reviewer, a docs writer, a security auditor, or a test generator. [Find out more](#custom-agents) .
- **Plugins** These are **packages** that can deliver preconfigured customizations such as skills, hooks, custom agents, and MCP servers. [Find out more](#plugins) .

### Custom instructions

#### What are custom instructions?

**Custom instructions** are persistent guidance that the Copilot CLI loads from instruction files at the start of a session.

Copilot will find and load instruction files from a number of default locations in the repository, such as `AGENTS.md` and `.github/copilot-instructions.md` , or from your home directory at `$HOME/.copilot/copilot-instructions.md` .

You can use the `--no-custom-instructions` flag to avoid loading these instructions.

#### What problem do custom instructions solve?

Custom instructions help you:

- Keep Copilot aligned with your coding conventions and preferences.
- Apply team or organization standards consistently.
- Avoid having to include repetitive reminders to Copilot in every prompt.

#### When should you use custom instructions?

Use custom instructions for:

- Style and quality rules Example: "Prefer small PRs, write tests, and avoid changing public APIs without discussion."
- Repository conventions Example: "Use `pnpm` , keep changelog entries in `CHANGELOG.md` , run `pnpm test` before committing."
- Communication preferences Example: "Explain tradeoffs briefly, then provide the recommended choice."

#### When shouldn't you use custom instructions?

Avoid or keep them minimal when:

- You only want the behavior in one workflow (use a **skill** instead).
- Your instructions are so large/specific they distract Copilot from the immediate task (prefer a **skill** or a **custom agent** ).

#### Find out more about custom instructions

See [Adding custom instructions for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/add-custom-instructions) .

### Skills

#### What is a skill?

A **skill** is, minimally, a Markdown file containing instructions that Copilot can use to perform tasks in a specific context. The name and skill description allow Copilot to determine whether it should use the skill for a given task. If it decides to use the skill, it will load the instructions and follow them to complete the task.

Skills can optionally reference other files, stored within the skill directory. These can include scripts that Copilot can run when the skill is used.

#### What problem does a skill solve?

Skills help you:

- Standardize how Copilot performs tasks in a specific context (for example, when performing a code review).
- Provide "just-in-time" instructions without permanently changing Copilot's behavior.
- Avoid overloading Copilot's context window with instructions that are not relevant to the current task.

#### How do you access skills?

You can manually invoke a skill by using a slash command. For example, `/Markdown-Checker check README.md` . Use `/skills list` to list the available skills.

Copilot CLI automatically invokes skills when it detects one that is relevant to the current task.

#### When should you use a skill?

Use a skill when you want:

- A repeatable set of instructions or functionality to be available for a type of task. Example: a documentation skill that checks that user-facing documentation is updated when frontend code is changed.
- A consistent output format. Example: a "release note draft" skill that ensures Copilot uses a template to create a release note.
- A workflow you sometimes need, but not always. Example: a "deep refactor" skill you only enable during migrations.

#### When shouldn't you use a skill?

Avoid skills when:

- The guidance should **apply to everything** you do (use **custom instructions** instead).
- You need new capabilities (you may need an **MCP server** to add tools, or a **custom agent** for specialization).

#### Find out more about agent skills

See [About agent skills](/en/copilot/concepts/agents/about-agent-skills) .

### Tools

#### What is a tool?

A **tool** is an ability that Copilot uses to get something done-like searching files, viewing file contents, editing, running a task, or invoking a skill. Some tools are built in, and others can be added through MCP servers.

#### What problem do tools solve?

Tools let the CLI:

- Gather accurate context (using read/search tools).
- Make changes safely (using edit tools).
- Execute commands and validate outcomes (potentially using subagents).

#### When should you use tools?

You typically don't call tools directly-Copilot decides to use tools as needed. You can allow or deny use of tools, either for a specific task, for the current session, or for all of your Copilot CLI sessions.

You'll see Copilot using tools when you:

- Ask Copilot to search the repository for something, update a file, or run tests.
- Invoke a skill-which triggers the `skill` tool.
- Ask Copilot to perform a task that requires it to use a tool supplied by an MCP server.
- Task Copilot to complete a complex task and it decides to delegate to a subagent-which triggers the `task` tool.

#### Find out more about allowing or denying tools

See [About GitHub Copilot CLI](/en/copilot/concepts/agents/about-copilot-cli#allowed-tools) .

### MCP servers

#### What is an MCP server?

An **MCP server** is a service that allows AI applications, such as Copilot CLI, to connect to external data sources and tools.

Adding an MCP server to Copilot CLI provides additional capabilities, by allowing you to use tools supplied by that MCP server. For example, you could add an MCP server that provides tools for interacting with an online calendar application, or a support ticketing system.

#### What problem do MCP servers solve?

MCP servers help when the built-in tools aren't enough. They can:

- Connect Copilot CLI to external systems.
- Add purpose-built tools (for example, for working with APIs, databases, or image generation).
- Standardize safe access patterns for non-repository resources.

#### When should you use an MCP server?

Use an MCP server when you need:

- Integration with external data or systems. Example: `How many support tickets have been opened this month for Product X?`
- Domain-specific actions that you want the CLI to perform on your behalf. Example: `Message the bug-watch channel: Only 2 support tickets raised this month for Product X.`

#### When shouldn't you use an MCP server?

Avoid adding MCP servers when:

- Built-in tools already cover your needs.

#### Find out more about MCP servers

See [About Model Context Protocol (MCP)](/en/copilot/concepts/context/mcp) .

### Hooks

#### What is a hook?

**Hooks** allow you to specify that, at a given point in a session lifecycle, Copilot CLI will execute a shell command you have defined.

| Hook                          | When it runs                                |
|-------------------------------|---------------------------------------------|
| `preToolUse` / `postToolUse`  | Before/after a tool runs.                   |
| `userPromptSubmitted`         | When a user submits a prompt.               |
| `sessionStart` / `sessionEnd` | At the start/end of a session.              |
| `errorOccurred`               | When an error occurs.                       |
| `agentStop`                   | When the main agent stops without an error. |
| `subagentStop`                | When a subagent completes.                  |

#### What problem do hooks solve?

Hooks help when you want **programmable control or observability** around Copilot CLI behavior, such as:

- **Enforcing guardrails** -block or warn before certain tools run.
- **Adding logging/telemetry**
- **Customizing retry/abort behavior on recoverable errors**
- **Adding "policy" checks** -for example, to prevent edits to protected paths.
- **Intercepting the moment a subagent finishes** -before results return to the parent agent.

Hooks are useful when you need more control than skills or custom instructions can provide. While skills and instructions guide Copilot's behavior through prompts, hooks ensure that operations you have defined will be performed at specific moments-for example, to block a tool from running, or to log activity when a session ends.

#### When should you usehooks?

Use hooks when you want:

- **Tool guardrails**
    - Example: before `bash` runs, require that the specific command matches an allowlist.
    - Example: before `edit` runs, block changes under `infra/` unless a ticket ID is present.
- **Session lifecycle automation**
    - Example: when the agent stops, archive the transcript of the session to a storage location.
- **Error handling policy**
    - Example: on rate limit errors, automatically choose "retry" with a capped retry count.
- **Subagent workflow control**
    - Example: when a subagent finishes, validate its output before passing results back to the main agent.

#### When shouldn't you use hooks?

Avoid hooks when:

- You just need consistent prompting or workflow instructions (use **skills** ).
- You want persistent preferences and standards (use **custom instructions** ).
- You need new external capabilities (use **MCP servers** and tools).
- Maintaining configuration that can affect every session may be problematic for you.

#### Find out more about hooks

See [Using hooks with GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/use-hooks) .

### Subagents

#### What is a subagent?

A **subagent** is the execution of a separate AI agent that the main agent of a Copilot CLI session spins up to do a specific piece of work.

Copilot CLI uses a subagent when the main agent decides that delegating a chunk of work to a separate agent is the best way to complete the user's request.

#### What problem do subagents solve?

Subagents help Copilot:

- Keep the context window of the main agent in a CLI session focused, by offloading a chunk of work to a separate agent.
- Parallelize work, where necessary, by running certain tasks in the background.
- Run a custom agent separately from the main agent, performing specialist work with a different approach to the work carried out by the main agent.

#### When are subagents used?

Copilot is likely to use a subagent for:

- Codebase exploration For example, listing all endpoints in an API.
- Command execution for complex tasks For example, running a test suite, or building a large project and analyzing the results.
- Reviewing changes For example, reviewing staged changes and identifying potential security issues.
- Complex multi-step work For example, implementing a feature with several changes.
- For using custom agents If you've defined a custom agent and it's eligible for inference ( `infer` is not set to `false` ), Copilot may choose to delegate work to that custom agent by spinning up a subagent with the custom agent's configuration.

### Custom agents

#### What is a custom agent?

**Custom agents** provide Copilot with specialist knowledge about a particular subject, and define a particular approach that Copilot should use when working in that area. You can think of a custom agent as a "persona" that Copilot can adopt when working on certain tasks.

Copilot CLI has several built-in custom agents. For example, the `explore` , `task` , `research` , `code-review` , and `general-purpose` agents. You can also define your own custom agents, to meet your specific needs.

You define a custom agent in a Markdown file with YAML frontmatter. The file contains:

- A description of the agent's role and expertise
- A list of allowed tools (or all tools)
- Optional MCP server connections
- An optional `infer` setting-when enabled, Copilot will automatically delegate to this agent when it detects a task that matches the agent's specialty.

#### What problem do custom agents solve?

Custom agents help when you need:

- Specialist knowledge to be applied consistently in a particular context.
- Different tool permissions for different work, as defined in the custom agent configuration.
- To allow the main agent's context window to stay focused on the main task, with the custom agent's own context window being used for the specialist work it performs.

#### When should you use a custom agent?

Use a custom agent when you want:

- A specialized reviewer or helper Example: Create a "react-reviewer" custom agent that focuses on work involving React patterns.
- Safer permissions Example: A custom agent that can only `view/grep/glob` (read-only) for auditing.
- Optional auto-delegation Example: Set `infer: true` in the custom agent configuration so that Copilot can automatically use this custom agent when appropriate.

#### When shouldn't you use a custom agent?

Avoid custom agents when:

- You only need guidance text (a **skill** can be a lighter-weight solution).
- You don't need specialization and the default agent performs tasks well.

#### Find out more about custom agents

See [Custom agents configuration](/en/copilot/reference/custom-agents-configuration) .

### Plugins

#### What is a plugin?

A **plugin** is an installable package that can deliver a bundle of functionality to Copilot. A plugin can include any combination of the other customization features. For example, skills, custom agents, hooks, and MCP server configurations.

Copilot includes plugin management commands (install, update, list, uninstall) and supports installing from a marketplace or directly from a GitHub repository.

#### What problem do plugins solve?

Plugins help you:

- Easily add a bundle of functionality to Copilot without having to manually configure each piece.
- Package and distribute a custom configuration-potentially a combination of skills, custom agents, hooks, and MCP servers-to your team, or to the public.
- Alter available functionality without having to manually copy files into directories.

#### When should you use a plugin?

Use a plugin when:

- You want a team-wide bundle Example: A company-wide engineering plugin that includes:
    - Skills for incident response.
    - A custom agent for code review.
    - An MCP server for internal services.
- You want easy installation and updates Example: Install a plugin initially, then update it regularly using `/plugin update PLUGIN-NAME` .

#### When shouldn't you use a plugin?

Avoid plugins when:

- You're experimenting locally and don't need distribution (use local skills, custom instructions, or custom agents).
- You only need a small one-off workflow. A single skill file may be simpler.

### Putting it together: choosing the right option

| Requirement                                                                                             | Best option                                                  |
|---------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| I want Copilot to always follow our repository conventions.                                             | **Custom instructions**                                      |
| I want a repeatable workflow I can invoke on demand.                                                    | **Skills**                                                   |
| I want Copilot to answer questions and carry out work in my repository.                                 | Copilot requests permission to use the appropriate **tools** |
| I want guardrails, policy, or automation around tool use and session events.                            | **Hooks**                                                    |
| I need Copilot to be able to use tools provided by an external service.                                 | **MCP servers**                                              |
| When working on particular tasks, I want Copilot to operate as a specialist with a constrained toolset. | **Custom agent**                                             |
| I want Copilot to carry out a complex task on my behalf.                                                | Copilot automatically uses **subagents** when appropriate.   |
| I want to add a package of functionality to Copilot CLI without configuring it manually myself.         | **Plugin**                                                   |

### Further reading

- [GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli)


### Overview

Typically, when you use Copilot CLI interactively, you submit a prompt and then wait for Copilot CLI to respond before giving the next instruction. This back-and-forth interaction continues until the task is done.

Autopilot mode allows Copilot CLI to work through a task without waiting for your input after each step. Once you give the initial instruction, Copilot CLI works through each step autonomously until it determines the task is complete.

The difference between the CLI's standard interactive mode and autopilot mode is like the difference between working on a task with a coworker, where they do most of the work, but check back with you periodically, versus handing the task over to your colleague, saying "Here's what I need-let me know when you're finished."

In autopilot mode, Copilot keeps on going until one of these happens:

- The agent determines that the task is complete.
- A problem occurs that prevents further progress.
- You press `Ctrl` + `C` to stop the agent from continuing.
- The maximum continuation limit is reached (if set).

To switch into autopilot mode during an interactive session, press `Shift` + `Tab` and cycle through the available modes until you reach autopilot mode, then enter your prompt. Use the same keypress to switch from autopilot mode back to the standard interactive mode.

### Benefits of autopilot mode

- **Hands-off automation:** Copilot completes tasks without needing your input after the initial instruction.
- **Efficiency:** Ideal for well-defined tasks like writing tests, refactoring files, or fixing CI failures. Autopilot is particularly suited for large tasks that require long-running, multi-step sessions.
- **Batch operations:** Useful for scripting and CI workflows where you want Copilot to run to completion.
- **Safety:** Autopilot mode allows Copilot to take multiple self-directed steps to finish your task. `--max-autopilot-continues` limits how many steps it can take before stopping, to avoid infinite loops. Also, in autopilot mode, Copilot cannot carry out any actions that require permission unless you explicitly grant it full permissions.

### Things to consider

- **Task suitability:** Autopilot mode is best for well-defined tasks. It is not ideal for open-ended exploration, feature development without a clear goal, or tasks where you want to guide the ongoing work. Copilot will do its best to complete any task, but it may struggle with vague or ambiguous instructions or tasks that require nuanced judgment calls along the way. This may result in a set of code changes that aren't what you expected and can't be used without remedial work.
- **Trust:** You need to trust Copilot to make reasonable decisions. Autopilot mode works best when you grant it approval for all permissions. This is equivalent to running Copilot CLI with the `--allow-all` option. You should be aware that this gives the CLI permission to make any changes it deems necessary to complete the task, including altering and deleting files.
- **Cost:** Autopilot mode uses premium requests in the same way that these are used when you are working in the standard interactive interface. In the standard mode, one premium request is used when you submit your initial prompt, and then an additional premium request is used each time you reply to a question in the CLI and the agent uses your response to interact with the AI model. The same applies in autopilot mode, except that you are not involved in initiating the next step, so the use of additional premium requests happens without your direct involvement. The billable premium request usage is determined using a multiplier. The multiplier varies depending on which model you use. Use the `/model` slash command to see the currently selected model and its multiplier, and change the model if required. For more information, see [Requests in GitHub Copilot](/en/copilot/concepts/billing/copilot-requests) and [About billing for individual GitHub Copilot plans](/en/copilot/concepts/billing/billing-for-individuals#about-premium-requests) . Each time the agent continues autonomously it will display a message in the CLI telling you how many premium requests have been used by that continuation step-taking account of the model multiplier-for example: `Continuing autonomously (3 premium requests)` .

### Permissions

When entering autopilot mode, if you have not already granted Copilot all permissions, a message is displayed prompting you to choose between three options:

```
1. Enable all permissions (recommended)
2. Continue with limited permissions
3. Cancel (Esc)
```

You will get the best results from autopilot mode if you enable all permissions. If you choose to continue with limited permissions, Copilot will automatically deny any tool requests that require approval, which may prevent it from completing certain tasks. You can change your mind later and grant full permissions, during an autopilot session, by using the `/allow-all` command (or its alias `/yolo` ).

### Comparing autopilot mode, --allow-all , and --no-ask-user

`--allow-all` , and its alias `--yolo` , are permissions-related options that you can pass to the `copilot` command when you start an interactive session. For a full list of available options, see [GitHub Copilot CLI command reference](/en/copilot/reference/copilot-cli-reference/cli-command-reference#command-line-options) .

The `--allow-all` and `--yolo` options allow the CLI agent to use all tools, paths, and URLs. You can also set these permissions during an interactive session, by using the `/allow-all` or `/yolo` slash commands.

Note

Entering `/allow-all` and `/yolo` enables permissions for the current session. Entering these slash commands again does not disable permissions-in other words, these commands don't toggle permissions on and off.

With `--allow-all` , you are still in the normal interactive flow. Copilot will still stop and ask you what you want it to do when it reaches a decision point. However, when Copilot CLI needs to do something that would normally require approval, such as using tools, paths, or URLs, it will go ahead without asking for permission.

The `--no-ask-user` option suppresses clarifying questions that Copilot would normally ask. Instead the agent must make decisions on its own, rather than asking for your input. This provides a degree of autonomy. However, unlike autopilot mode, `--no-ask-user` does not allow the agent to continue working on a task through successive steps where interaction with the AI model is required. With this option, the CLI won't use additional premium requests, after your initial prompt, without your involvement.

### Typical workflow for using autopilot mode

Autopilot mode is ideal for implementing a large, detailed plan of work. Often you will find it useful to switch to autopilot mode after working with Copilot in plan mode to create an implementation plan. For more information about plan mode, see [Best practices for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/cli-best-practices#2-plan-before-you-code) .

For example:

- Start an interactive Copilot CLI session. Optionally, you can include the `--allow-all` option to grant permissions, and the `--max-autopilot-continues` option to set a maximum continuation limit for autopilot mode during the session. For example, you could start the session with `copilot --allow-all --max-autopilot-continues 10` to give the agent permission to use all tools, paths, and URLs, and set a maximum continuation limit for autopilot to 10.
- When the interactive session starts, if you're prompted to trust the files in the current folder, accept this option.
- Press `Shift` + `Tab` to switch to plan mode, enter a prompt describing what you want to achieve, then work with Copilot to create a detailed plan.
- Once you have a plan that you are happy with, use the option that the CLI presents to "Accept plan and build on autopilot".
- If you're prompted about permissions, choose the option to enable all permissions.
- Leave Copilot to implement the plan. You can check in on its progress periodically.

### Using autopilot mode programmatically

You can use autopilot mode when you run Copilot CLI programmatically, for example when you pass Copilot a prompt on the command line, or when you use the CLI as part of a script or CI workflow. Doing so allows you to automate tasks end-to-end without needing to interact with the CLI after the initial command.

Use the `--allow-all` (or `--yolo` ) option to grant Copilot permission to use all tools, paths, and URLs. You can include the `--max-autopilot-continues` option to set a maximum continuation limit to prevent runaway loops. This is especially important in programmatic contexts where you won't be there to intervene if something goes wrong.

Example usage:

```
copilot --autopilot --yolo --max-autopilot-continues 10 -p "YOUR PROMPT HERE"
```

### Summary

Use autopilot mode when you want Copilot to take over a task and work to completion without your involvement. It's best for clear, well-defined tasks where you trust Copilot to make reasonable decisions.

### Further reading

- [Using GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/use-copilot-cli#get-copilot-to-work-autonomously)
- [Running tasks in parallel with the /fleet command](/en/copilot/concepts/agents/copilot-cli/fleet)
- [GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli)


### Introduction

The `/fleet` slash command in Copilot CLI is designed to take an implementation plan and break it down into smaller, independent tasks that can be executed in parallel by subagents. This allows for faster completion of complex requests that involve multiple steps.

This article gives an overview of the `/fleet` slash command. For details of how to use it, see [Speeding up task completion with the /fleet command](/en/copilot/how-tos/copilot-cli/speeding-up-task-completion) .

### How /fleet works

When you use the `/fleet` command, the main Copilot agent analyzes the prompt and determines whether it can be divided into smaller subtasks. It will assess, based on the nature of the subtasks and their dependencies, whether these can be efficiently executed by subagents. If it decides to assign some or all of the subtasks to subagents, it will act as orchestrator, managing the workflow and dependencies between the subtasks. Where possible, the orchestrator agent will run the subagents in parallel, allowing the whole task to be completed more quickly.

### Benefits of using /fleet

- **Speed of task completion** : The main benefit of using the `/fleet` command is that a large, multi-part task can be completed more quickly by running subtasks in parallel. Whether parts of a large task can be worked on in parallel will be determined by the dependencies between the subtasks. Some tasks, such as creating a suite of tests for a new feature, are well suited to parallelization and will typically complete faster when you use the `/fleet` slash command.
- **Specialization** : If you've defined custom agents that are specialized for certain types of work, these may be used by the subagents. This allows for specialization, with the subagents using the custom agents best suited to the specific subtask they are working on. By default, subagents use a low-cost AI model. However, you can tell Copilot to use a specific model for part of the work. For example, within a larger prompt, you could specify `... Use GPT-5.3-Codex, to create ... Use Claude Opus 4.5, to analyze ...` . If a subagent uses a custom agent profile that specifies a particular AI model, then that model will be used by the subagent. Using a specific model may produce better quality results for particular types of subtask. If custom agents are available, Copilot will decide whether to use one to complete a particular subtask. However, if you know that a specific custom agent is well-suited to a particular subtask, you can specify this in your prompt by using `@CUSTOM-AGENT-NAME` . For example, within a larger prompt: `... Use @test-writer to create comprehensive unit tests for ...` . For more information, see [Creating and using custom agents for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/create-custom-agents-for-cli) .
- **Context window** : Each subagent has its own context window, separate from the main agent and other subagents. This allows each subagent to focus on its specific task without being overwhelmed by the full context of the larger task.

### When should you use /fleet ?

- **Large or complex tasks** : When your request involves multiple independent steps, such as refactoring several files, updating dependencies, or running tests across modules.
- **Parallelizable work** : If your task can be split into subtasks that don't depend on each other.
- **Automated workflows** : When you want the quickest possible completion of a large task-for example, when you're using autopilot mode to allow Copilot to work autonomously.

### Points to consider

- **Premium request usage** : When you submit a prompt in the CLI and Copilot interacts with the selected large language model (LLM) to generate a response, this consumes premium requests. The number of premium requests consumed depends on the model that's currently selected. More interactions with the LLM result in more premium requests being consumed. Each subagent can interact with the LLM independently of the main agent, so splitting work up into smaller tasks that are run by subagents may result in more LLM interactions than if the work was handled by the main agent. Using `/fleet` in a prompt may therefore cause more premium requests to be consumed. The billable premium request usage is determined using a multiplier. The multiplier varies depending on which model you use. Use the `/model` slash command to see the currently selected model and its multiplier, and change the model if required. For more information, see [Requests in GitHub Copilot](/en/copilot/concepts/billing/copilot-requests) and [About billing for individual GitHub Copilot plans](/en/copilot/concepts/billing/billing-for-individuals#about-premium-requests) .
- **Task composition** : Work is best suited to execution by multiple subagents if it can be decomposed into independent subtasks. If your request is inherently sequential, using the `/fleet` slash command mode may not provide any benefit.

### Relationship between /fleet and autopilot mode

The `/fleet` slash command is often used in autopilot mode, but these are distinct features that can be used independently:

- **Autopilot mode** allows Copilot to continue working autonomously until a task is complete, auto-responding to requests that would otherwise require user intervention.
- **`/fleet`** is all about using subagents to execute tasks in parallel, while the main agent manages the overall workflow. You can use the `/fleet` slash command in interactive sessions independently of autopilot mode.

A typical workflow for using `/fleet` in autopilot mode might look like this:

1. Press `Shift` + `Tab` to switch into plan mode and work with Copilot CLI to create an implementation plan.
2. Recognize that the completed plan contains multiple elements and looks like a good candidate for `/fleet` .
3. Select the **Accept plan and build on autopilot + /fleet** option that's displayed when the plan is complete.

For more information about autopilot mode, see [Allowing GitHub Copilot CLI to work autonomously](/en/copilot/concepts/agents/copilot-cli/autopilot) .

### Further reading

- [Speeding up task completion with the /fleet command](/en/copilot/how-tos/copilot-cli/speeding-up-task-completion)
- [GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli)
- [Using GitHub Copilot CLI](/en/copilot/how-tos/use-copilot-agents/use-copilot-cli)


### Introduction

Copilot CLI's `/research` slash command is a powerful tool for deep research and investigation. When you enter `/research` followed by details of what you want to know about, Copilot activates a specialized research agent that gathers and processes information from your codebase, from relevant GitHub repositories, and from the web. This built-in custom agent produces a comprehensive Markdown report with citations, along with a brief summary in the CLI. You can view the full report and save it as a gist on GitHub, making it easy to share.

The command is designed to provide exhaustive, well-cited answers to complex questions about codebases, APIs, libraries, software architecture and other technical topics.

### Using the /research slash command

In an interactive CLI session, enter:

Copilot prompt

```
/research TOPIC
```

Where `TOPIC` is a natural language description of what you want to find out about.

Depending on the permissions you have given the CLI, Copilot may ask you to grant permission for it to create a directory in which to store data as it compiles the research.

When the research is complete, Copilot shows you a summary of the key findings, and gives you a link to a Markdown file containing the full report.

### Viewing and sharing a research report

You can use the link displayed when the research completes to view the full report in your default editor for Markdown files.

Alternatively, press `Ctrl` + `Y` to open the current session's most recent research report in the terminal.

Note

The application used to display a report when you press `Ctrl` + `Y` is determined by the value of the `COPILOT_EDITOR` , `VISUAL` , or `EDITOR` environment variables (in that order of precedence). If none of these are set, the CLI will use vi on Linux or vim on macOS.

To share the report you can either save it to a file or create a GitHub gist.

1. To create a gist enter: Copilot prompt `/share gist research` To save to a file, enter: Copilot prompt `/share file research [PATH]` If you omit the `[PATH]` parameter, the file will be saved to the current working directory with a filename based on the research topic.
2. Use the up/down and enter keys to select the report you want to share from the list of research reports you've created during the current session. The URL of the gist, or the path to the file, is displayed in the CLI.

### Benefits of /research

- **Depth over speed** : Normal chat is optimized for quick answers. `/research` is optimized for thoroughness. It produces reports that can be hundreds of lines long, with architecture diagrams, code snippets, and citations.
- **Saved and shareable output** : Reports are saved to disk as Markdown files. You can view and share them at any time. This makes the research output a permanent artifact, rather than a transient chat message.
- **Works across repositories** : When logged into GitHub, the agent can search across your organization's repositories, fetch files from any public or accessible private repository, and search the web-it's not limited to your local codebase.
- **Query-type adaptation** : Rather than generating a standard, one-size-fits-all report, the response format automatically adapts to whether you're asking a how-to question, a conceptual question, or requesting a technical deep-dive.
- **Autonomous operation** : The agent never interrupts you with clarifying questions. It makes reasonable assumptions and explicitly documents them in a "Confidence Assessment" section.

### Example prompts for /research

#### Codebase architecture

Copilot prompt

```
/research What is the architecture of this codebase?
```

**Why it works well** : The research agent has access to `grep` , `glob` , and `view` tools scoped to your current working directory. It can explore the full project tree, read key files, and synthesize an architectural overview-something a normal chat response might do only superficially. The agent will typically produce architecture diagrams, component breakdowns, and data flow descriptions.

#### How a specific technology works

Copilot prompt

```
/research How does React implement concurrent rendering?
```

**Why it works well** : The agent uses specialized tools to pull information from the internet, and to look at actual React source code on GitHub. It's instructed to prioritize code over documentation and provide file paths with line numbers.

#### Understanding internal implementation patterns

Copilot prompt

```
/research How are feature flags implemented at our organization?
```

**Why it works well** : The agent is explicitly instructed to "always prioritize internal/private implementations over public/open-source alternatives" and to search the organization's repositories first using `org:ORGNAME` queries. It knows to look for internal naming patterns like `-hub` , `-service` , `-client` .

#### Comparing technologies or approaches

Copilot prompt

```
/research What's the difference between JWT and session-based authentication?
```

**Why it works well** : The agent adapts its response to "Conceptual/Explanatory Questions" with narrative explanations, trade-offs, and design decisions. It will typically use tables for comparisons of three or more items.

#### Process/how-to questions

Copilot prompt

```
/research How do I add an endpoint to the API?
```

**Why it works well** : The agent is trained to detect query type and provide step-by-step guidance with links to relevant docs, contacts, and systems for process/how-to type questions.

#### Deep-diving into a specific codebase component

Copilot prompt

```
/research How is the session management system implemented in this repo?
```

**Why it works well** : Combining local tools ( `grep` , `glob` , `view` ) with the agent's instructions to "trace imports, calls, and type references" and "follow dependencies" means it will walk through the actual implementation, not just give a high-level answer.

### *When you might not want to use /research*

- **Quick, simple questions** : If you just want to know "What does this function do?" or "Fix this bug", a normal chat message is faster and more appropriate. `/research` is designed for questions requiring extensive investigation.
- **When you need code changes** : `/research` produces a report, not code modifications. It uses the `create` tool to save the report file, but does not use `edit` , `bash` , or other code-modification tools. If you need the agent to actually change your code, use a normal prompt (typically starting in plan mode).
- **Time-sensitive interactions** : Research takes longer than a normal response because the agent makes many tool calls (searching code, fetching files, searching the web). If you need a quick answer in the flow of coding, normal chat is better.

### Considerations and things to be aware of

- **Reports are tied to your session** : Research reports are stored in a session-specific research directory. If you start a new session, previous research won't be available within the CLI when you use the `Ctrl` + `Y` shortcut or the `/share` slash command. However, you can access previous reports from the appropriate `~/.copilot/session-state/SESSION-ID/research/` directory. In Linux or macOS, you can use the following command at a terminal command prompt to list the 10 most recent CLI session directories: Bash `ls -dtl ~/.copilot/session-state/*/ | head -10`
- **The research agent uses a specific model** : The research agent is hard-coded to use a particular AI model (see [GitHub Copilot CLI command reference](/en/copilot/reference/copilot-cli-reference/cli-command-reference#built-in-agents) ). The model selection is not configurable via the `/model` command. The research agent always uses the defined model regardless of what model you've selected for your main session.
- **Report quality varies by query type** : The agent classifies your query into three types and adapts its response accordingly: The way you phrase your prompt may affect the agent's choice of research classification. For example, if you want a technical deep-dive but you phrase your question as "What is X?", you might get a conceptual answer. In this situation you could rephrase your prompt to be more explicit about the type of report you want Copilot to produce. For example: "Give me a technical deep-dive into X, with architecture diagrams and code examples."
    - **Process questions** → step-by-step guidance (minimal code).
    - **Conceptual questions** → narrative explanation with context.
    - **Technical deep-dives** → full architecture diagrams, component sections, and code examples.

### Further reading

- [GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli)
- [Using GitHub Copilot CLI](/en/copilot/how-tos/use-copilot-agents/use-copilot-cli)


### Introduction

Every time you use Copilot CLI, a complete set of data about your session-including your prompts, Copilot's responses, the tools that were used, and details of files that were modified-is recorded on your local machine. Over time, this builds up a rich history of what you've worked on, how you've worked, and what Copilot has done for you.

This session data powers several features:

- **Resuming sessions** : You can pick up where you left off in any previous session.
- **Asking questions about your history** : You can ask Copilot questions about your past work, and it will query your session data to answer them.
- **The** **`/chronicle`** **slash command** : A set of purpose-built subcommands that generate standup reports, personalized tips, and suggestions for improving your custom instructions-all derived from your session history.

This conceptual article explains how session data is stored, and how you can leverage it to enhance your workflow. For a practical guide to resuming a session, asking Copilot about your CLI sessions, and using the `/chronicle` slash command, see [Using GitHub Copilot CLI session data](/en/copilot/how-tos/copilot-cli/chronicle) .

Note

The `/chronicle` command, and Copilot's ability to answer questions about your session history, are currently experimental features and are only available if you have used the `/experimental on` slash command, or the `--experimental` command line option.

### How session data is stored

Every Copilot CLI session is persisted as a set of files in the `~/.copilot/session-state/` directory on your machine. The data for each session contains a complete record of the session. These files allow you to resume an interactive CLI session.

In addition to the session files, Copilot CLI stores structured session data in a local SQLite database, referred to as the session store. This data is a subset of the full data stored in the session files. The session store is what powers the `/chronicle` slash command and it also allows Copilot to answer questions you ask about your past work.

#### Privacy and data locality

All session data is stored locally in your home directory and is only accessible to your user account. Copilot reads this data on your machine when you ask questions about your interactions with the CLI, or when you use the `/chronicle` slash command. Session data such as your previous prompts, context data, and responses you received may be sent to the AI model, just as they would be in any normal Copilot CLI interaction.

If you want to remove data for a particular CLI session, you can delete the relevant session directory from `~/.copilot/session-state/` . You can clear all session data by deleting everything under `~/.copilot/session-state/` . After doing this you must manually reindex the session store. See the [Reindexing the session store](#reindexing-the-session-store) later in this article.

### About the /chronicle slash command

The `/chronicle SUBCOMMAND` command uses the data in the session store to provide insights and suggestions about your use of Copilot CLI.

You can enter the following commands in an interactive CLI session:

- `/chronicle standup` : Generates a short report summarizing what you worked on in your recent CLI sessions, including branch names, pull request links, and status checks.
- `/chronicle tips` : Provides personalized tips for using Copilot CLI more effectively.
- `/chronicle improve` : Analyzes your session history to identify patterns where Copilot may have misunderstood your intent or where there was a lot of back-and-forth, and generates custom instructions to help Copilot better understand you in the future.
- `/chronicle reindex` : Rebuilds the session store from your session history files.

### Benefits of /chronicle and the session data

- **Self-improving workflow** : The `improve` subcommand creates a feedback loop that helps you to refine your custom instructions. Over time, this makes the agent more effective for your specific project.
- **Effortless standup reports** : Instead of manually reconstructing what you did yesterday, `/chronicle standup` generates a standup summary from your actual session data.
- **Personalized coaching** : The `tips` subcommand acts as a personal productivity coach that knows both what Copilot CLI can do and how you actually use it. It bridges the gap between available features and your current workflow.
- **Talk to your coding history** : The session store lets Copilot answer any question that your past sessions might help with-from recalling a bug fix you did last week to analyzing your prompting patterns over time.
- **Local and private** : All session data-both the raw JSONL files and the SQLite session store-stays on your machine. Nothing is uploaded or shared beyond the normal AI model interactions that happen in any Copilot CLI session. You have full control over your data and can delete it at any time.

### When should you use these features?

- **At the start of your day** : Run `/chronicle standup last 3 days` to generate a reminder of what you worked on recently and the CLI session you were working in.
- **Periodically, to level up** : Run `/chronicle tips` every week or two to discover features and workflow improvements you might be missing.
- **When Copilot keeps making the same mistake** : Run `/chronicle improve` to identify the pattern and generate custom instructions to fix it.
- **To recall past work** : Ask a free-form question like "Have I worked on anything related to the payments API?" and Copilot will search your history.
- **To continue previous work** : Use `copilot --continue` or `copilot --resume` to pick up where you left off.

### Reindexing the session store

The session store is populated incrementally during a CLI session. Data for a session is written to disk in a session-specific subdirectory of `~/.copilot/session-state/` . This also happens periodically during a session, and also when the session ends.

You can reindex the session store from the session files on disk, although typically you will never need to do this.

Situations where you might need to reindex include:

- **Indexing old sessions** : If you have old session files on disk that were created before the session store existed, reindexing will populate the session store with data from those sessions.
- **Session deletion** : If you want to delete a session from your history you can delete the session directory and then reindex the session store.
- **Migrating/recovering sessions** : If you moved your session files to another machine, or restored them from a backup, without also moving/restoring the session store file ( `~/.copilot/session-store.db` ), you can use the reindex command to recreate the session store.
- **File corruption** : If the session store file ( `~/.copilot/session-store.db` ) becomes corrupted, or is accidentally deleted, you can recover the session store from the session files.
- **Unexpected termination** : If a session terminates unexpectedly (for example, due to a crash or power loss) before data held in memory has been flushed to the session store you may be able to populate the session store with the missing data if it was written to disk, in the session files, prior to the termination.

To reindex the session store, use the following slash command in an interactive CLI session:

Copilot prompt

```
/chronicle reindex
```

### Further reading

- [Using GitHub Copilot CLI session data](/en/copilot/how-tos/copilot-cli/chronicle)
- [GitHub Copilot CLI command reference](/en/copilot/reference/copilot-cli-reference/cli-command-reference)


### About the context window

When you use GitHub Copilot CLI, every message you send, every response from Copilot, every tool call and its result, and the system instructions that define Copilot's behavior are all held in a **context window** . The context window is the total amount of information that the AI model can consider at one time when generating a response.

The context window has a fixed size, measured in tokens, that varies by model. Tokens typically consist of short, commonly used words, and fragments of multi-syllable words. As your conversation progresses, the context window fills up with:

- **System instructions and tool definitions** : The built-in instructions that tell Copilot how to behave, plus the schemas of all available tools. These are always present and take up a fixed portion of the context window.
- **Your messages** : Every prompt you send.
- **Copilot's responses** : Everything Copilot says back to you.
- **Tool calls and results** : When Copilot reads files, runs commands, or searches your codebase, both the request and the output are added to the context. Tool results can be especially large-for example, if a tool reads a long file or runs a command that produces extensive output.

All of this accumulates in the context window. In a long or complex session, the context window can fill up.

#### Why the context window matters

The context window is what gives Copilot its "memory" of your conversation. Everything inside the context window is available for Copilot to reference when responding to you.

This means that in a very long session, Copilot might not be able to hold the entire conversation history at once. Copilot CLI therefore has context management features that effectively allow you to continue a conversation with Copilot for as long as you need.

### Checking your context usage

You can check how much of the context window is currently in use by entering the `/context` slash command. This displays a visual breakdown of your token usage, showing:

- **System/Tools** : The fixed overhead of system instructions and tool definitions.
- **Messages** : The space used by your conversation history.
- **Free Space** : How much room is left for new messages.
- **Buffer** : A reserved portion that triggers automatic context management.


You might want to use the `/context` slash command when:

- You're in a long session and want to know how much space is left.
- Copilot seems to be forgetting earlier parts of the conversation.
- You want to understand whether compaction has occurred, or is likely to occur soon.

### Compaction

Compaction is the process that allows GitHub Copilot CLI to support long-running sessions without hitting the limits of the context window.

#### When compaction happens

When your conversation reaches approximately 80% of the context window's capacity, Copilot CLI automatically starts compacting the context in the background. This leaves a buffer of approximately 20% so that tool calls can continue to run while compaction is in progress. If the context fills to approximately 95% before compaction finishes, Copilot CLI pauses briefly to wait for compaction to complete before continuing.

You can also trigger compaction manually at any time by entering the `/compact` command. This is useful if you're about to start a new phase of work and want to free up context space proactively. Press `Esc` to cancel a manual compaction if you change your mind.

#### What compaction does

When compaction runs, Copilot CLI:

1. Takes a snapshot of the current conversation history.
2. Sends the full conversation to the AI model with a special prompt that asks it to generate a structured summary. The summary captures the goals of the conversation, what was done, key technical details, important files, and planned next steps.
3. Replaces the old conversation history with the summary, along with any original user instructions and the current state of any plans or to-do lists.
4. Keeps any messages that were added while compaction was running in the background.

The result is that the conversation history is compressed into a much smaller summary, freeing up the majority of the context window for new work. Copilot uses this summary to maintain continuity-it knows what was discussed, what was decided, and what to do next-even though the original messages have been replaced.

#### What compaction does not preserve

Compaction is a summarization process, so some detail is inevitably lost. The summary captures the key points, but fine-grained details-such as the exact wording of every message, the full output of every command, or minor decisions made early in a long conversation-may not be included. If you need Copilot to recall a very specific detail from much earlier in the session, it may not have that information after compaction.

#### What would happen without compaction

Without compaction, once the context window filled up, Copilot would have to fall back to simply dropping old messages from the conversation history-removing them without any summary or record. This would mean losing context abruptly, with no way for Copilot to know what was in the deleted messages. Compaction avoids this by replacing the history with an intelligent summary rather than discarding it.

### Checkpoints

Every time compaction happens-whether automatically or manually-a **checkpoint** is created. A checkpoint is a saved copy of the compaction summary, stored as a numbered, titled file in your session's workspace.

#### Viewing checkpoints

To see all checkpoints in your current session, enter:

Copilot prompt

```
/session checkpoints
```

This lists every checkpoint with its number and title:

```
Checkpoint History (3 total):
  3. Refactoring authentication module
  2. Implementing user dashboard
  1. Initial planning and setup
```

Use the checkpoint number to view the full content of any checkpoint. For example, to view checkpoint 2, enter:

Copilot prompt

```
/session checkpoints 2
```

#### When checkpoints are useful

- **Reviewing what happened** : After a long session with multiple compactions, earlier phases of the conversation are no longer in the active context. Checkpoints let you read back through what Copilot did at each compaction.
- **Verifying continuity** : If you want to check that Copilot's summary accurately captured your earlier work before continuing, you can review the most recent checkpoint.
- **Debugging confusion** : If Copilot seems to have forgotten a decision or is going in a direction that contradicts earlier work, checking checkpoints can reveal what was preserved during compaction and what might have been summarized differently than you expected.

Note

- Checkpoints are created automatically. You don't need to manage them-they're there if you need them. For most sessions, you won't need to look at checkpoints at all.
- You can't reverse a compaction once it has completed.

### Using long-running sessions

Automatic compaction allows you to continue working in a long-running session without worrying about hitting the limits of the context window. There are times when this is very useful, and other times when you might prefer to start a fresh session.

#### When long sessions are useful

Long-running sessions work well when:

- You're working on a multi-phase task, such as building a feature that requires scaffolding, implementation, testing, and then creating a pull request.
- You're iterating on a problem and want Copilot to retain the context of what's been tried and what hasn't worked.
- You're doing exploratory work across a codebase and building up shared understanding with Copilot over time.

#### When to start a fresh session

Starting a new session is better when:

- You're switching to an unrelated task. Copilot doesn't need the context of your previous work, and a clean context window means more space for the new task.
- The conversation has gone through many compactions, and you feel that important context is being lost in the summarization process.
- You want a clean slate-for example, if work went in a wrong direction and you'd rather start over than have Copilot try to reconcile earlier decisions with a new approach.

Tip

You can resume previous sessions at any time using the `/resume` command. This lets you pick up where you left off, including any checkpoints that were created during that session.

### Further reading

- [GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli)
- [Using GitHub Copilot CLI](/en/copilot/how-tos/use-copilot-agents/use-copilot-cli)
- [GitHub Copilot CLI command reference](/en/copilot/reference/copilot-cli-reference/cli-command-reference)


### Introduction

GitHub Copilot CLI is a powerful terminal-native AI coding assistant that brings agentic capabilities directly to your command line. The Copilot CLI offers deep flexibility, GitHub workflow integration, and the ability to work autonomously on complex tasks while maintaining full user control.

This guide will help you start using the CLI.

### Installation

Use one of these commands:

- **Cross-platform (npm)** Prerequisite: Node.js 22 or later. Bash `npm install -g @github/copilot`
- **Windows (WinGet)** Bash `winget install GitHub.Copilot`
- **macOS/Linux (Homebrew)** Bash `brew install copilot-cli`

### Starting the CLI for the first time

1. In the terminal, navigate to the project directory where you want to use Copilot CLI.
2. Start an interactive CLI session: `copilot`
3. In the CLI interface, enter `/login` and follow the on-screen prompts to authenticate with your GitHub account. You'll only have to do this the first time you use the CLI.
4. When prompted, confirm that you trust that the files in the current directory are suitable for use with an AI tool. Note Copilot won't make changes to your files without your explicit approval.
5. Try asking Copilot a question, for example: Copilot prompt `Give me an overview of this project.`

### Core shortcuts to master

| Shortcut     | Action                                   |
|--------------|------------------------------------------|
| `Esc`        | Cancel the current operation             |
| `Ctrl` + `C` | Cancel if thinking, clear input, or exit |
| `Ctrl` + `L` | Clear the screen                         |
| `@`          | Mention files to include in context      |
| `/`          | Show slash commands                      |
| `?`          | Show tabbed help                         |
| `↑` and `↓`  | Navigate the command history             |

For a full list of shortcuts and available commands, enter:

```
/help
```

### Using GitHub Copilot CLI non-interactively

You can also enter a command and get a response from Copilot directly in your terminal, without starting an interactive session.

To do this, pass a prompt to the CLI with the `-p` flag. For example:

```
copilot -p "In Git, how can I apply a commit from another branch"
```

The `-p` flag allows you to use GitHub Copilot CLI programmatically within scripts, for example to automate tasks using AI.

You can add the `-s` flag to tell the CLI to output only Copilot's response, omitting the additional usage information.

```
copilot -sp "YOUR PROMPT HERE"
```

For details of other flags you can use programmatically, and for more information, enter:

```
copilot help
```

or:

```
copilot help TOPIC
```

where TOPIC is one of the topics listed in the help output.

### Next steps

Find out more about Copilot CLI:

- [About GitHub Copilot CLI](/en/copilot/concepts/agents/about-copilot-cli)
- [Using GitHub Copilot CLI](/en/copilot/how-tos/use-copilot-agents/use-copilot-cli)
- [Best practices for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/cli-best-practices)
- [Get started with GitHub Copilot CLI: A free hands-on course](https://developer.microsoft.com/blog/get-started-with-github-copilot-cli-a-free-hands-on-course)


### Introduction

GitHub Copilot CLI is a terminal-native AI coding assistant that brings agentic capabilities directly to your command line. Copilot CLI can operate like a chatbot, answering your questions, but its true power lies in its ability to work autonomously as your coding partner, allowing you to delegate tasks and oversee its work.

This article provides tips for getting the most out of Copilot CLI, from using the various CLI commands effectively to managing the CLI's access to files. Consider these tips as starting points, then experiment to find out what works best for your workflows.

Note

GitHub Copilot CLI is continually evolving. Use the `/help` command to see the most up to date information.

### 1. Customize your environment

#### Use custom instructions files

Copilot CLI automatically reads instructions from multiple locations, allowing you to define organization-wide standards and repository-specific conventions.

**Supported locations (in order of discovery):**

| Location                                    | Scope                 |
|---------------------------------------------|-----------------------|
| `~/.copilot/copilot-instructions.md`        | All sessions (global) |
| `.github/copilot-instructions.md`           | Repository            |
| `.github/instructions/**/*.instructions.md` | Repository (modular)  |
| `AGENTS.md` (in Git root or cwd)            | Repository            |
| `Copilot.md` , `GEMINI.md` , `CODEX.md`     | Repository            |

##### Best practice

Repository instructions **always take precedence** over global instructions. Use this to enforce team conventions. For example, this is a simple `.github/copilot-instructions.md` file.

```
### Build Commands
- `npm run build` - Build the project - `npm run test` - Run all tests - `npm run lint:fix` - Fix linting issues ## Code Style
- Use TypeScript strict mode - Prefer functional components over class components - Always add JSDoc comments for public APIs ## Workflow
- Run `npm run lint:fix && npm test` after making changes - Commit messages follow conventional commits format - Create feature branches from `main`
```

Tip

Keep instructions concise and actionable. Lengthy instructions can dilute effectiveness.

For more information, see [About customizing GitHub Copilot responses](/en/copilot/concepts/prompting/response-customization?tool=webui) .

#### Configure allowed tools

Manage which tools Copilot can run without asking for permission. When Copilot requests permission for an action, you can typically choose either to allow it just this time, or allow the tool to be used for the rest of the CLI session.

To reset previously approved tools, use:

```
/reset-allowed-tools
```

You can also preconfigure allowed tools via CLI flags:

```
copilot --allow-tool= 'shell(git:*)' --deny-tool= 'shell(git push)'
```

**Common permission patterns:**

- `shell(git:*)` - Allow all Git commands
- `shell(npm run:*)` - Allow all npm scripts
- `shell(npm run test:*)` - Allow npm test commands
- `write` - Allow file writes

#### Select your preferred model

Use `/model` to choose from available models based on your task complexity:

| Model                         | Best For                                                       | Tradeoffs                                                                                                      |
|-------------------------------|----------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| **Claude Opus 4.5** (default) | Complex architecture, difficult debugging, nuanced refactoring | Most capable but uses more [premium requests](/en/copilot/concepts/billing/copilot-requests#model-multipliers) |
| **Claude Sonnet 4.5**         | Day-to-day coding, most routine tasks                          | Fast, cost-effective, handles most work well                                                                   |
| **GPT-5.2 Codex**             | Code generation, code review, straightforward implementations  | Excellent for reviewing code produced by other models                                                          |

**Recommendations:**

- **Opus 4.5** is ideal for tasks requiring deep reasoning, complex system design, subtle bug investigation, or extensive context understanding.
- **Switch to Sonnet 4.5** for routine tasks where speed and cost efficiency matter-it handles the majority of everyday coding effectively.
- **Use Codex** for high-volume code generation and as a second opinion for reviewing code produced by other models.

You can switch models mid-session with `/model` as task complexity changes.

If your organization or enterprise has configured custom models using their own LLM provider API keys, those models also appear in `/model` at the bottom of the list.

#### Use your own model provider

You can configure Copilot CLI to use your own model provider instead of GitHub-hosted models. Run `copilot help providers` for full setup instructions.

**Key considerations:**

- Your model must support **tool calling** (function calling) and **streaming** . Copilot CLI returns an error if either capability is missing.
- For best results, use a model with a context window of at least 128k tokens.
- Built-in sub-agents ( `/review` , `/task` , explore, `/fleet` ) automatically inherit your provider configuration.
- Premium request cost estimates are hidden when using your own provider. Token usage (input, output, and cache counts) is still displayed.
- `/delegate` only works if you are also signed in to GitHub. It transfers the session to GitHub's server-side Copilot, not your provider.

See [Using your own model provider](/en/copilot/concepts/agents/copilot-cli/about-copilot-cli#using-your-own-model-provider) .

### 2. Plan before you code

#### Plan mode

**Models achieve higher success rates when given a concrete plan to follow.** In plan mode, Copilot will create a structured implementation plan before any code is written.

Press `Shift` + `Tab` to toggle between normal mode and plan mode. In plan mode, all prompts you enter will trigger the plan workflow.

Alternatively, you can use  the `/plan` command in normal mode to achieve the same effect.

**Example prompt (from normal mode):**

```
/plan Add OAuth2 authentication with Google and GitHub providers
```

**What happens:**

- Copilot analyzes your request and codebase.
- **Asks clarifying questions** to align on requirements and approach.
- Creates a structured implementation plan with checkboxes.
- Saves the plan to `plan.md` in your session folder.
- **Waits for your approval** before implementing.

You can press `Ctrl` + `y` to view and edit the plan in your default editor for Markdown files.

**Example plan output:**

```
### Implementation Plan: OAuth2 Authentication
### Overview Add social authentication using OAuth2 with Google and GitHub providers. ## Tasks
- [ ] Install dependencies (passport, passport-google-oauth20, passport-github2) - [ ] Create authentication routes in `/api/auth`
- [ ] Implement passport strategies for each provider - [ ] Add session management middleware - [ ] Create login/logout UI components - [ ] Add environment variables for OAuth credentials - [ ] Write integration tests ## Detailed Steps
1. **Dependencies** : Add to package.json... 2. **Routes** : Create `/api/auth/google` and `/api/auth/github` ...
```

#### When to use plan mode

| Scenario                           | Use plan mode?   |
|------------------------------------|------------------|
| Complex multi-file changes         |                  |
| Refactoring with many touch points |                  |
| New feature implementation         |                  |
| Quick bug fixes                    |                  |
| Single file changes                |                  |

#### The explore → plan → code → commit workflow

For best results on complex tasks:

- **Explore** : `Read the authentication files but don't write code yet`
- **Plan** : `/plan Implement password reset flow`
- **Review** : Check the plan, suggest modifications
- **Implement** : `Proceed with the plan`
- **Verify** : `Run the tests and fix any failures`
- **Commit** : `Commit these changes with a descriptive message`

### 3. Leverage infinite sessions

#### Automatic context window management

Copilot CLI features **infinite sessions** . You don't need to worry about running out of context. The system automatically manages context through intelligent compaction that summarizes conversation history while preserving essential information.

**Session storage location:**

```
~/.copilot/session-state/{session-id}/
├── events.jsonl      # Full session history
├── workspace.yaml    # Metadata
├── plan.md           # Implementation plan (if created)
├── checkpoints/      # Compaction history
└── files/            # Persistent artifacts
```

Note

If you ever need to manually trigger compaction, use `/compact` . This is rarely necessary since the system handles it automatically.

#### Session management commands

To view information about the current CLI session, enter:

```
/session
```

To view a list of any session checkpoints, enter:

```
/session checkpoints
```

Note

A checkpoint is created when session context is compacted, and allows you to view the summary context that Copilot created.

To view the details of a specific checkpoint, enter:

```
/session checkpoints NUMBER
```

where NUMBER specifies the checkpoint you want to display.

To view any temporary files that have been created during the current session-for example, artifacts created by Copilot that shouldn't be saved to the repository-enter:

```
/session files
```

To view the current plan (if Copilot has generated one), enter:

```
/session plan
```

#### Best practice: Keep sessions focused

While infinite sessions allow long-running work, focused sessions produce better results:

- Use `/clear` or `/new` between unrelated tasks.
- This resets context and improves response quality.
- Think of it like starting a fresh conversation with a colleague.

#### The /context command

Visualize your current context usage with `/context` . It shows a breakdown of:

- System/tools tokens
- Message history tokens
- Free space
- Buffer allocation

### 4. Delegate work effectively

#### The /delegate command

**Offload work to run in the cloud using Copilot cloud agent.** This is particularly powerful for:

- Tasks that can run asynchronously.
- Changes to other repositories.
- Long-running operations you don't want to wait for.

**Example prompt:**

```
/delegate Add dark mode support to the settings page
```

**What happens:**

- Your request is sent to Copilot cloud agent.
- The agent creates a pull request with the changes.
- You can continue working locally while the cloud agent works.

#### When to use /delegate

| Use `/delegate`              | Work locally            |
|------------------------------|-------------------------|
| Tangential tasks             | Core feature work       |
| Documentation updates        | Debugging               |
| Refactoring separate modules | Interactive exploration |

### 5. Common workflows

#### Codebase onboarding

Use Copilot CLI as your pair programming partner when joining a new project. For example, you could ask Copilot:

- How is logging configured in this project?
- What's the pattern for adding a new API endpoint?
- Explain the authentication flow
- Where are the database migrations?

#### Test-driven development

Pair with Copilot CLI to develop tests.

- Write failing tests for the user registration flow
- *Review and approve the tests.*
- Now implement code to make all tests pass
- *Review the implementation.*
- Commit with message "feat: add user registration"

#### Code review assistance

- /review Use Opus 4.5 and Codex 5.2 to review the changes in my current branch against `main`. Focus on potential bugs and security issues.

#### Git operations

Copilot excels at Git workflows:

- What changes went into version `2.3.0`?
- Create a PR for this branch with a detailed description
- Rebase this branch against `main`
- Resolve the merge conflicts in `package.json`

#### Bug investigation

- The `/api/users` endpoint returns 500 errors intermittently. Search the codebase and logs to identify the root cause.

#### Refactoring

- `/plan Migrate all class components to functional components with hooks` Then answer the questions Copilot asks. Review the plan it creates, and ask Copilot to make changes if necessary. When you are happy with the plan you can prompt: `Implement this plan`

### 6. Advanced patterns

#### Work across multiple repositories

**Copilot CLI provides flexible multi-repository workflows** -a key differentiator for teams working on microservices, monorepos, or related projects.

**Option 1: Run from a parent directory**

```
### Navigate to a parent directory containing multiple repos
cd ~/projects
copilot
```

Copilot can now access and work across all child repositories simultaneously. This is ideal for:

- Microservices architectures
- Making coordinated changes across related repos
- Refactoring shared patterns across projects

**Option 2: Use** **`/add-dir`** **to expand access**

```
### Start in one repo, then add others (requires full paths) copilot
/add-dir /Users/me/projects/backend-service
/add-dir /Users/me/projects/shared-libs
/add-dir /Users/me/projects/documentation
```

**View and manage allowed directories:**

```
/list-dirs
```

**Example workflow: coordinated API changes**

```
I need to update the user authentication API. The changes span:

- @/Users/me/projects/api-gateway (routing changes)
- @/Users/me/projects/auth-service (core logic)
- @/Users/me/projects/frontend (client updates)

Start by showing me the current auth flow across all three repos.
```

This multi-repository capability enables:

- Cross-cutting refactors (update a shared pattern everywhere)
- API contract changes with client updates
- Documentation that references multiple codebases
- Dependency upgrades across a monorepo

#### Using images for UI work

Copilot can work with visual references. Simply **drag and drop** images directly into the CLI input, or reference image files:

```
Implement this design: @mockup.png
Match the layout and spacing exactly
```

#### Checklists for complex migrations

For large-scale changes:

```
Run the linter and write all errors to `migration-checklist.md` as a checklist.
Then fix each issue one by one, checking them off as you go.
```

#### Autonomous task completion

Switch into autopilot mode to allow Copilot to work autonomously on a task until it is complete. This is ideal for long-running tasks that don't require constant supervision. For more information, see [Allowing GitHub Copilot CLI to work autonomously](/en/copilot/concepts/agents/copilot-cli/autopilot) .

Optionally, you can usually speed up large tasks by using the `/fleet` slash command at the start of your prompt to allow Copilot to break the task into parallel subtasks that are run by subagents. For more information, see [Running tasks in parallel with the /fleet command](/en/copilot/concepts/agents/copilot-cli/fleet) .

### 7. Team guidelines

#### Recommended repository setup

- **Create** **`.github/copilot-instructions.md`** with:
    - Build and test commands
    - Code style guidelines
    - Required checks before commits
    - Architecture decisions
- **Establish conventions** for:
    - When to use `/plan` (complex features, refactoring)
    - When to use `/delegate` (tangential work)
    - Code review processes with AI assistance

#### Security considerations

- Copilot CLI requires explicit approval for potentially destructive operations.
- Review all proposed changes before accepting.
- Use permission allowlists judiciously.
- Never commit secrets. Copilot is designed to avoid this, but always verify.

#### Measuring productivity

Track metrics like:

- Time from issue to pull request
- Number of iterations before merge
- Code review feedback cycles
- Test coverage improvements

### Getting help

From the command line, you can display help by using the command: `copilot -h` .

For help on various topics enter:

```
copilot help TOPIC
```

where `TOPIC` can be one of: `config` , `commands` , `environment` , `logging` , or `permissions` .

#### Within the CLI

For help within the CLI, enter:

```
/help
```

To view usage statistics, enter:

```
/usage
```

To submit private feedback to GitHub about Copilot CLI, raise a bug report, or submit a feature request, enter:

```
/feedback
```

### Hands-on practice

Try the [Creating applications with Copilot CLI](https://github.com/skills/create-applications-with-the-copilot-cli) Skills exercise for practical experience building an application with Copilot CLI.

Here is what you will learn:

- Install Copilot CLI
- Use the issue template to create an issue
- Generate a Node.js CLI calculator app
- Expand calculator functionality
- Write unit tests for calculator functions
- Create, review, and merge your pull request

### Further reading

- [About GitHub Copilot CLI](/en/copilot/concepts/agents/about-copilot-cli)
- [Using GitHub Copilot CLI](/en/copilot/how-tos/use-copilot-agents/use-copilot-cli)
- [GitHub Copilot CLI command reference](/en/copilot/reference/copilot-cli-reference/cli-command-reference)
- [Copilot plans and pricing](https://github.com/features/copilot/plans)


### Prerequisites

- **An active GitHub Copilot subscription** . See [Copilot plans](https://github.com/features/copilot/plans?ref_product=copilot&ref_type=engagement&ref_style=text) .
- (On Windows) **PowerShell** v6 or higher

If you have access to GitHub Copilot via your organization or enterprise, you cannot use Copilot CLI if your organization owner or enterprise administrator has disabled it in the organization or enterprise settings. See [Managing policies and features for GitHub Copilot in your organization](/en/copilot/managing-copilot/managing-github-copilot-in-your-organization/managing-github-copilot-features-in-your-organization/managing-policies-for-copilot-in-your-organization) .

### Installing or updating Copilot CLI

You can install Copilot CLI using WinGet (Windows), Homebrew (macOS and Linux), npm (all platforms), or an install script (macOS and Linux).

#### Installing with npm (all platforms)

Prerequisite: Node.js 22 or later.

Shell

```
npm install -g @github/copilot
```

Note

If you have `ignore-scripts=true` in your `~/.npmrc` file, you must use the command:

Shell

```
npm_config_ignore_scripts=false npm install -g @github/copilot
```

To install the prerelease version:

Shell

```
npm install -g @github/copilot@prerelease
```

#### Installing with WinGet (Windows)

PowerShell

```
winget install GitHub.Copilot
```

To install the prerelease version:

PowerShell

```
winget install GitHub.Copilot.Prerelease
```

#### Installing with Homebrew (macOS and Linux)

Shell

```
brew install copilot-cli
```

To install the prerelease version:

Shell

```
brew install copilot-cli@prerelease
```

#### Installing with the install script (macOS and Linux)

Shell

```
curl -fsSL https://gh.io/copilot-install | bash
```

Or:

Shell

```
wget -qO- https://gh.io/copilot-install | bash
```

To run as root and install to `/usr/local/bin` , use `| sudo bash` .

To install to a custom directory, set the `PREFIX` environment variable. It defaults to `/usr/local` when run as root or `$HOME/.local` when run as a non-root user.

To install a specific version, set the `VERSION` environment variable. It defaults to the latest version.

For example, to install version `v0.0.369` to a custom directory:

Shell

```
curl -fsSL https://gh.io/copilot-install | VERSION="v0.0.369" PREFIX="$HOME/custom" bash
```

#### Download from GitHub.com

You can download the executables directly from [the](https://github.com/github/copilot-cli/releases/) [`copilot-cli`](https://github.com/github/copilot-cli/releases/) [repository](https://github.com/github/copilot-cli/releases/) .

Download the executable for your platform, unpack it, and run.

### Authenticating with Copilot CLI

On first launch, if you're not currently logged in to GitHub, you'll be prompted to use the `/login` slash command. Enter this command and follow the on-screen instructions to authenticate. For more information on the authentication process, see [Authenticating GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/authenticate-copilot-cli) .

#### Authenticating with a personal access token

You can also authenticate using a fine-grained personal access token with the "Copilot Requests" permission enabled.

1. Visit [Fine-grained personal access tokens](https://github.com/settings/personal-access-tokens/new) .
2. Under "Permissions," click **Add permissions** and select **Copilot Requests** .
3. Click **Generate token** .
4. Export the token in your terminal or environment configuration. Use the `COPILOT_GITHUB_TOKEN` , `GH_TOKEN` , or `GITHUB_TOKEN` environment variable (in order of precedence).

### Next steps

You can now use Copilot from the command line. See [Using GitHub Copilot CLI](/en/copilot/how-tos/use-copilot-agents/use-copilot-cli) .


### About authentication

If you use your own LLM provider API keys (BYOK), GitHub authentication is not required.

Authentication is required for any other GitHub Copilot CLI usage.

When authentication is required, Copilot CLI supports three methods. The method you use depends on whether you are working interactively or in an automated environment.

- **OAuth device flow** : The default and recommended method for interactive use. When you run `/login` in Copilot CLI, the CLI generates a one-time code and directs you to authenticate in your browser. This is the simplest way to authenticate.
- **Environment variables** : Recommended for CI/CD pipelines, containers, and non-interactive environments. You set a supported token as an environment variable ( `COPILOT_GITHUB_TOKEN` , `GH_TOKEN` , or `GITHUB_TOKEN` ), and the CLI uses it automatically without prompting.
- **GitHub CLI fallback** : If you have GitHub CLI ( `gh` ) (note: the `gh` CLI, not `copilot` ) installed and authenticated, Copilot CLI can use its token automatically. This is the lowest priority method and activates only when no other credentials are found.

Once authenticated, Copilot CLI remembers your login and automatically uses the token for all Copilot API requests. You can log in with multiple accounts, and the CLI will remember the last-used account. Token lifetime and expiration depend on how the token was created on your account or organization settings.

### Unauthenticated use

If you configure Copilot CLI to use your own LLM provider API keys (BYOK), GitHub authentication is **not required** . Copilot CLI can connect directly to your configured provider without a GitHub account or token.

However, without GitHub authentication, the following features are **not available** :

- `/delegate` : Requires Copilot cloud agent, which runs on GitHub's servers
- GitHub MCP server: Requires authentication to access GitHub APIs
- GitHub Code Search: Requires authentication to query GitHub's search index

You can combine BYOK with GitHub authentication to get the best of both: your preferred model for AI responses, plus access to GitHub-hosted features like `/delegate` and code search.

#### Offline mode

If you set the `COPILOT_OFFLINE` environment variable to `true` , Copilot CLI runs without contacting GitHub's servers. In offline mode:

- No GitHub authentication is attempted.
- The CLI only makes network requests to your configured BYOK provider.
- Telemetry is fully disabled.

Offline mode is **only fully air-gapped** if your BYOK provider is local or otherwise within the same isolated environment (for example, a model running on-premises with no external network access). If `COPILOT_PROVIDER_BASE_URL` points to a remote or internet-accessible endpoint, prompts and code context will still be sent over the network to that provider. Without offline mode, even when using BYOK without GitHub authentication, telemetry is still sent normally.

#### Supported token types

| Token type                | Prefix        | Supported   | Notes                                                  |
|---------------------------|---------------|-------------|--------------------------------------------------------|
| OAuth token (device flow) | `gho_`        | Yes         | Default method via `copilot login`                     |
| Fine-grained PAT          | `github_pat_` | Yes         | Must include required permissions **Copilot Requests** |
| GitHub App user-to-server | `ghu_`        | Yes         | Via environment variable                               |
| Classic PAT               | `ghp_`        | No          | Not supported by Copilot CLI                           |

#### How Copilot CLI stores credentials

By default, the CLI stores your OAuth token in your operating system's keychain under the service name `copilot-cli` :

| Platform   | Keychain                           |
|------------|------------------------------------|
| macOS      | Keychain Access                    |
| Windows    | Credential Manager                 |
| Linux      | libsecret (GNOME Keyring, KWallet) |

If the system keychain is unavailable-for example, on a headless Linux server without `libsecret` installed-the CLI prompts you to store the token in a plaintext configuration file at `~/.copilot/config.json` .

When you run a command, Copilot CLI checks for credentials in the following order:

1. `COPILOT_GITHUB_TOKEN` environment variable
2. `GH_TOKEN` environment variable
3. `GITHUB_TOKEN` environment variable
4. OAuth token from the system keychain
5. GitHub CLI ( `gh auth token` ) fallback

Note

- An environment variable silently overrides a stored OAuth token. If you set `GH_TOKEN` for another tool, the CLI uses that token instead of the OAuth token from `copilot login` . To avoid unexpected behavior, unset environment variables you do not intend the CLI to use.
- When you configure BYOK provider environment variables (for example, `COPILOT_PROVIDER_BASE_URL` , `COPILOT_PROVIDER_API_KEY` ), Copilot CLI uses these for AI model requests regardless of your GitHub authentication status. GitHub tokens are only needed for GitHub-hosted features.

### Authenticating with OAuth

The OAuth device flow is the default authentication method for interactive use. You can authenticate by running `/login` from Copilot CLI or `copilot login` from your terminal.

#### Authenticate with /login

1. From Copilot CLI, run `/login` . Bash `/login`
2. Select the account you want to authenticate with. For GitHub Enterprise Cloud with data residency, enter the hostname of your instance `What account do you want to log into? 1. GitHub.com 2. GitHub Enterprise Cloud with data residency (*.ghe.com)`
3. The CLI displays a one-time user code and automatically copies it to your clipboard and opens your browser. `Waiting for authorization... Enter one-time code: 1234-5678 at https://github.com/login/device Press any key to copy to clipboard and open browser...`
4. Navigate to the verification URL at `https://github.com/login/device` if your browser did not open automatically.
5. Paste the one-time code in the field on the page.
6. If your organization uses SAML SSO, click **Authorize** next to each organization you want to grant access to.
7. Review the requested permissions and click **Authorize GitHub Copilot CLI** .
8. Return to your terminal. The CLI displays a success message when authentication is complete. `Signed in successfully as Octocat. You can now use Copilot.`

#### Authenticate with copilot login

1. From the terminal, run `copilot login` . If you are using GitHub Enterprise Cloud with data residency, pass the hostname of your instance. Bash `copilot login` For GitHub Enterprise Cloud: Bash `copilot login --host HOSTNAME` The CLI displays a one-time user code and automatically copies it to your clipboard and opens your browser. `To authenticate, visit https://github.com/login/device and enter code 1234-5678.`
2. Navigate to the verification URL at `https://github.com/login/device` if your browser did not open automatically.
3. Paste the one-time code in the field on the page.
4. If your organization uses SAML SSO, click **Authorize** next to each organization you want to grant access to.
5. Review the requested permissions and click **Authorize GitHub Copilot CLI** .
6. Return to your terminal. The CLI displays a success message when authentication is complete. `Signed in successfully as Octocat.`

### Authenticating with environment variables

For non-interactive environments, you can authenticate by setting an environment variable with a supported token. This is ideal for CI/CD pipelines, containers, or headless servers.

1. Visit [Fine-grained personal access tokens](https://github.com/settings/personal-access-tokens/new) .
2. Under "Permissions," click **Add permissions** and select **Copilot Requests** .
3. Click **Generate token** .
4. Export the token in your terminal or environment configuration. Use the `COPILOT_GITHUB_TOKEN` , `GH_TOKEN` , or `GITHUB_TOKEN` environment variable (in order of precedence).

### Authenticating with GitHub CLI

If you have GitHub CLI installed and authenticated, Copilot CLI can use its token as a fallback. This method has the lowest priority and activates only when no environment variables are set and no stored token is found.

1. Verify that GitHub CLI is authenticated. Bash `gh auth status` If you use GitHub Enterprise Cloud with data residency, verify the correct hostname is authenticated. Bash `gh auth status --hostname HOSTNAME`
2. Run `copilot` . The Copilot CLI uses the GitHub CLI token automatically.
3. Run `/user` to verify your authenticated account in the CLI.

### Switching between accounts

Copilot CLI supports multiple accounts. You can list available accounts and switch between them from within the CLI.

To list available accounts, run

`/user list` from the Copilot CLI prompt.

To switch to a different account, type

`/user switch` on the prompt.

To add another account, run `copilot login` from a new terminal session, or run the login command from within the CLI and authorize with the other account.

### Signing out and removing credentials

To sign out, type `/logout` at the Copilot CLI prompt. This removes the locally stored token but does not revoke it on GitHub.

To revoke the OAuth app authorization on GitHub and prevent it from being used elsewhere, follow these steps.

1. Navigate to **Settings** > **Applications** > **Authorized OAuth Apps** .
2. Navigate to your settings page:
    1. In the upper-right corner of any page on GitHub, click your profile picture.
    2. Click **Settings** .
3. In the left sidebar, click **Applications** .
4. Under **Authorized OAuth Apps** , click next to **GitHub CLI** to expand the menu and select **Revoke** .


### Introduction

Copilot CLI has several configuration options that control what it can access and do on your behalf.

This article shows you how to set trusted directories, configure access for tools, and grant permissions to file paths and URLs.

#### Prerequisites

- Install the Copilot CLI. See [Installing GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/install-copilot-cli) .

### Setting trusted directories

Trusted directories control where Copilot CLI can read, modify, and execute files. Trusting a directory has security implications, see [Security considerations](/en/copilot/concepts/agents/about-copilot-cli#trusted-directories) .

#### Choosing to trust a directory

When you start a GitHub Copilot CLI session, you'll be asked to confirm that you trust the files in, and below, the directory from which you launched the CLI.

You can choose to trust the current directory for:

- The currently running session only
- This and future sessions

If you choose to trust the directory for future sessions, the trusted directory prompt will not be displayed again. You should only choose this second option if you are sure that this location will always be a safe place for Copilot to operate.

#### Editing trusted directories

You can edit the list of permanently trusted directories.

1. Open the CLI's `config.json` file. By default, it's stored in a `.copilot` folder under your home directory:
    - **macOS/Linux** : `~/.copilot/config.json`
    - **Windows** : `$HOME\.copilot\config.json`

You can change the config location by setting the `COPILOT_HOME` environment variable.

1. Edit the contents of the `trusted_folders` array.

### Setting allowed tools

You can control which tools Copilot CLI can use, either by responding to approval prompts when Copilot attempts to use a tool, or by specifying permissions via command-line flags.

Be aware that allowing tool access has security implications, see [Security considerations](/en/copilot/concepts/agents/about-copilot-cli#allowed-tools) .

In this section, you can learn how to:

- [Allow a tool for the first time](#allowing-a-tool-for-the-first-time)
- [Allow tools to be used without manual approval](#allowing-tools-to-be-used-without-manual-approval)
- [Specify which tool you want to allow or deny](#specifying-which-tool-you-want-to-allow-or-deny)
- [Allow some tools while denying others](#allowing-some-tools-while-denying-others)
- [Limit available tools](#limiting-available-tools)

#### Allowing a tool for the first time

The first time that Copilot needs to use a tool that may require approval-for example, for example, `touch` , `chmod` , `node` , or `sed` -it will ask you whether you want to allow it to run. Whether you're prompted can depend on the tool and how it's being used (such as the arguments provided or whether the tool has been previously approved).

1. Prompt Copilot to perform a task that requires a tool. For example: `copilot -p "Create a new file called README.md with a project description"`
2. Choose from one of the three options:
    - `1. Yes` Choose this option to allow Copilot to run this particular command, this time only. The next time it needs to use this tool, it will ask you again.
    - `2. Yes, and approve TOOL for the rest of the running session` Choose this option to allow Copilot to use this tool for the duration of the currently running session. It will ask for your approval again in new sessions, or if you resume the current session in the future. If you choose this option, you are allowing Copilot to use this tool in any way it thinks is appropriate. For example, if Copilot asks you to allow it to run the command `rm ./this-file.txt` , and you choose option 2, then Copilot can run any `rm` command (for example, `rm -rf ./*` ) during the current run of this session, without asking for your approval.
    - `3. No, and tell Copilot what to do differently (Esc)` Choose this option to cancel the proposed command and instruct Copilot to try a different approach.

#### Allowing tools to be used without manual approval

You can use command-line flags to designate tools that Copilot can use without asking for your approval.

##### Allowing all tools

Use the `--allow-all-tools` to allow Copilot to use any tool without asking for your approval.

- For example: `copilot -p "Revert the last commit" --allow-all-tools`

##### Denying a tool

Use `--deny-tool` to prevent Copilot from using a specific tool.

- For example: `copilot --deny-tool='shell(git push)'`

This option takes precedence over the `--allow-all-tools` and `--allow-tool` options.

##### Allowing a tool

Use `--allow-tool` to allow Copilot to use a specific tool without asking for your approval.

- For example: `copilot --allow-tool='shell'`

#### Specifying which tool you want to allow or deny

To use the `--deny-tool` and `--allow-tool` options, you must specify what type of tool you want to allow or deny:

- [Shell commands](#allowing-or-denying-shell-commands)
- ['Write' tools](#allowing-or-denying-write-tools)
- [MCP server tools](#allowing-or-denying-mcp-server-tools)

##### Allowing or denying shell commands

Use `shell(COMMAND)` to allow or deny a specific shell command.

- For example, to prevent Copilot from using any `rm` command, use: `copilot --deny-tool='shell(rm)'`

For `git` and `gh` commands, specify a particular first-level subcommand to allow or deny.

- For example, to prevent Copilot from using `git push` , use: `copilot --deny-tool='shell(git push)'`

The tool specification is optional. For example, `copilot --allow-tool='shell'` allows Copilot to use any shell command without individual approval.

##### Allowing or denying 'write' tools

Use `'write'` to allow or deny tools-other than shell commands-permission to modify files.

- For example, to allow Copilot to edit files without your individual approval, use: `copilot --allow-tool='write'`

##### Allowing or denying MCP server tools

Use `'MCP_SERVER_NAME'` to allow or deny a specific tool from the specified MCP server.

- For example, to prevent Copilot from using the tool called `tool_name` from the MCP server called `My-MCP-Server` , use: `copilot --deny-tool='My-MCP-Server(tool_name)'`

`MCP_SERVER_NAME` is the name of an MCP server that you have configured.

Tools from the server are specified in parentheses, using the tool name that is registered with the MCP server.

Using the server name without specifying a tool allows or denies all tools from that server.

You can find an MCP server's name by entering `/mcp` in the interactive mode of Copilot CLI and selecting the server from the list that's displayed.

#### Allowing some tools while denying others

To determine exactly which tools Copilot can use without asking for your approval, you can use a combination of approval options. For example:

- To prevent Copilot from using the `rm` and `git push` commands, but automatically allow all other tools, use: `copilot --allow-all-tools --deny-tool='shell(rm)' --deny-tool='shell(git push)'`
- To prevent Copilot from using the tool `tool_name` from the MCP server named `My-MCP-Server` , but allow all other tools from that server to be used without individual approval, use: `copilot --allow-tool='My-MCP-Server' --deny-tool='My-MCP-Server(tool_name)'`

#### Limiting available tools

To restrict Copilot to a specific set of tools, use `--available-tools` .

Tools not included in this list will not be available to Copilot.

### Setting path permissions

Path permissions control which directories and files Copilot can access.

By default, Copilot CLI can access the current working directory, its subdirectories, and the system temp directory.

Path permissions apply to shell commands, file operations (create, edit, view), and search tools (such as `grep` and glob patterns). For shell commands, paths are heuristically extracted by tokenizing command text and identifying tokens that look like paths.

Warning

Path detection for shell commands has limitations:

- Paths embedded in complex shell constructs may not be detected.
- Only a specific set of environment variables are expanded ( `HOME` , `TMPDIR` , `PWD` , and similar). Custom variables like `$MY_PROJECT_DIR` are not expanded and may not be validated correctly.
- Symlinks are resolved for existing files, but not for files being created.

#### Allowing access to all paths

To disable path verification and allow access to any path, use the `--allow-all-paths` flag when starting Copilot CLI.

#### Disallowing access to the temp directory

To disallow access to the temp directory, use `--disallow-temp-dir` .

### Setting URL permissions

URL permissions control which external URLs Copilot can access. By default, all URLs require approval before access is granted.

URL permissions apply to the `web_fetch` tool and a curated list of shell commands that access the network (such as `curl` , `wget` , and `fetch` ). For shell commands, URLs are extracted using regex patterns.

Warning

URL detection for shell commands has limitations:

- URLs in file contents, config files, or environment variables read by commands are not detected.
- Obfuscated URLs (such as split strings or escape sequences) may not be detected.
- HTTP and HTTPS are treated as different protocols and require separate approval.

URL permissions can be persisted for the session or permanently.

#### Disabling URL verification

To disable URL verification, use the `--allow-all-urls` flag.

#### Pre-approving specific domains

To pre-approve specific domains, use `--allow-url=DOMAIN` .

- For example, `--allow-url=github.com` .

#### Denying specific domains

To deny specific domains, use `--deny-url=DOMAIN` .

- For example, `--deny-url=github.com` .

### Allowing all tools, paths, and URLs

To allow all tools, paths and URLs, use `--allow-all` , or its alias, `--yolo` .

This flag combines:

- `--allow-all-tools` (skip tool approval).
- `--allow-all-paths` (disable path verification).
- `--allow-all-urls` (disables URL verification).

Tip

During an interactive session, you can also enable all permissions with the `/allow-all` or `/yolo` slash commands.

### Further reading

- [Customize GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot)
- [GitHub Copilot CLI configuration directory](/en/copilot/reference/copilot-cli-reference/cli-config-dir-reference)


### Authentication errors

If you encounter authentication errors, use the table below to identify the cause and resolution.

| Issue                               | Cause                                     | Fix                                     | More information                                                            |
|-------------------------------------|-------------------------------------------|-----------------------------------------|-----------------------------------------------------------------------------|
| No authentication information found | No credentials stored                     | Run `copilot login`                     | [No authentication information found](#no-authentication-information-found) |
| 401 Unauthorized                    | Token revoked or insufficient permissions | Generate token with permissions         | [Token expired or revoked](#token-expired-or-revoked)                       |
| Token (classic) rejected            | Token (classic) ( `ghp_` )                | Use fine-grained personal access token  | [Token (classic) rejected](#token-classic-rejected)                         |
| 403 Forbidden or policy denied      | Copilot license or enterprise/org policy  | Check subscription and org settings     | [Access denied](#access-denied)                                             |
| Keychain unavailable                | Missing system keychain                   | Install `libsecret` or accept plaintext | [Keychain access failure](#keychain-access-failure)                         |
| Wrong account                       | Multiple accounts or env var override     | Check env vars, use `/user switch`      | [Wrong account](#wrong-account)                                             |

### No authentication information found

Copilot CLI displays the following error:

```
Error: No authentication information found
Copilot can be authenticated with GitHub using an OAuth Token or a Fine-Grained Personal Access Token
```

#### Cause

No credentials exist in any of the checked locations.

#### Fix

Use the following steps to find where authentication is missing and restore access.

##### Check your authentication status

Bash

```
gh auth status
```

If you see a message indicating that you're not logged in, log in with `gh auth login` or use the OAuth flow with `copilot login` .

##### Check whether an authentication environment variable is set

If you are using an environment variable, check whether the `COPILOT_GITHUB_TOKEN` , `GH_TOKEN` , or `GITHUB_TOKEN` environment variable is set:

Bash

```
echo $COPILOT_GITHUB_TOKEN
```

If the command prints nothing, the variable is not set. Set the variable to a valid token. To generate a token, see [Authenticating GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/authenticate-copilot-cli#authenticate-with-a-personal-access-token-pat) .

Bash

```
export $COPILOT_GITHUB_TOKEN =PERSONAL_ACCESS_TOKEN
```

##### macOS keychain

Bash

```
security find-generic-password -s copilot-cli
```

If no item is found, authenticate again with `/login` or `copilot login` to create one.

If an item is found but authentication still fails, remove the saved credential then authenticate again with

`/login` or `copilot login` :

Bash

```
security delete-generic-password -s copilot-cli
```

### Token expired or revoked

Copilot CLI displays the following error:

```
Error: Authentication failed

Your GitHub token may be invalid, expired, or lacking the required permissions.

To resolve this, try the following:
  • Start 'copilot' and run the '${LOGIN_COMMAND}' command to re-authenticate
  • If using a Fine-Grained PAT, ensure it has the 'Copilot Requests' permission enabled
  • If using COPILOT_GITHUB_TOKEN, GH_TOKEN or GITHUB_TOKEN environment variable, verify the token is valid and not expired
  • Run 'gh auth status' to check your current authentication status
```

#### Cause

The token was revoked, has expired, or was created without the required permissions.

#### Fix

Review the token's status and permissions on GitHub. The token must have the **Copilot Requests** permission. Generate a new token with the required permissions if necessary.

### Token (classic) rejected

A token starting with `ghp_` is silently ignored and the CLI behaves as if no token is set.

#### Cause

Classic personal access tokens are not supported by Copilot CLI.

#### Fix

Generate a fine-grained personal access token with the required scopes.

### Access denied

Copilot CLI displays one of the following errors:

```
Error: Access denied by policy settings

Your Copilot CLI policy setting may be preventing access. This can happen when:
  • Your organization has restricted Copilot access
  • Your Copilot subscription does not include this feature
  • Required policies have not been enabled by your administrator

To resolve this, visit your Copilot settings:
```

#### Cause

An organization policy blocks GitHub Copilot CLI, or the user account lacks a GitHub Copilot license.

#### Fix

- Check that your account has an active GitHub Copilot license.
- Ask your organization admin to enable GitHub Copilot CLI in the organization policy.

### Keychain access failure

During login, the CLI prompts you about the system keychain being unavailable and asks whether to store credentials in plaintext.

```
System keychain unavailable. Store token in plaintext config file? (y/N)
```

#### Cause

The system keychain is not accessible. This may occur on Linux systems without `libsecret` , headless servers, or systems with a permission issue.

#### Fix

Follow the steps for your operating system to restore secure credential storage.

##### macOS or Windows

On macOS, confirm Keychain Access app is available, and you can unlock your login keychain.

On Windows, confirm Credential Manager is available, and you can access the Windows Vault.

If you can't access the system credential manager, use plaintext storage (if prompted) or authenticate using an environment variable token, then rerun

`/login` or `copilot login` .

##### Linux

On Linux, use the system keyring or store credentials in plaintext.

1. Check whether `secret-tool` is installed: Bash `command -v secret-tool`
2. If `secret-tool` is not found or the search command returns no results, install `libsecret` and its dependencies. For example, on Debian and Ubuntu you could use the following command." Bash `sudo apt install libsecret-1-0 gnome-keyring seahorse`
3. Once `secret-tool` is installed, search the keyring for a saved credential: Bash `secret-tool search copilot-cli` If the command returns one or more results, credentials exist in the keyring. Run `copilot login` in the terminal or `/login` in Copilot CLI again.

### Wrong account

The wrong user is authenticated, or the token belongs to the wrong organization.

#### Cause

Multiple accounts are stored, or an environment variable is overriding the stored token.

#### Fix

To switch accounts, use `/user switch` at the CLI prompt, or sign out with `/logout` and run `/login` with the correct account.


### Introduction

Copilot CLI uses a variety of tools to complete tasks for you. It can execute shell commands, read and write files, search your codebase, fetch web content, and delegate tasks to specialized sub-agents.

While read-only operations like searching, reading files, and running read-only shell commands are allowed automatically, tools that can modify your system-such as running destructive shell commands, editing files, or accessing URLs-require your explicit approval before Copilot can use them. This helps avoid your use of the CLI resulting in changes you didn't intend because, for example, a shell command can do anything your user account can do: install packages, delete files, push code, or make network requests.

You can allow or deny permissions for tools either when you start the CLI or during your interactive session. If you haven't granted permission prior to starting a session, Copilot CLI will prompt you for permission each time it needs to perform a potentially destructive action. You can choose to allow the tool this one time, or for the remainder of the session.

### Layers of tool controls

There are two layers of control you can use when specifying tool permissions in command-line options. You can:

- Restrict the choice of tools available to the AI model.
- Allow or deny permission for specific tools.

### Restricting the choice of tools available to the AI model

The `--available-tools` and `--excluded-tools` options restrict the set of tools that the AI model is aware of, and can therefore choose from, when it determines how to complete a task.

- `--available-tools` disables all tools other than those you specify.
- `--excluded-tools` disables only the specified tools.

If you use both options together, the CLI will apply the allowlist specified by `--available-tools` and ignore the denylist specified by `--excluded-tools` .

If a tool is not in the available set, the AI model won't be able to use it at all, even if you specify it with the `--allow-tool` option. In an interactive session where you do not specify an available tool set, the AI model may try to use a tool, only to be denied. The `--available-tools` and `--excluded-tools` options prevent you wasting interactions with the model in this way.

#### Example use case

You are starting a CLI session to run benchmarking on your project and you want to avoid the AI model from even attempting to use web search.

```
copilot --excluded-tools= 'web_fetch, web_search'
```

Note

For full details of the syntax for these and other command-line options mentioned in this article, see [GitHub Copilot CLI command reference](/en/copilot/reference/copilot-cli-reference/cli-command-reference#command-line-options) .

### Allowing or denying permission for specific tools

The `--allow-tool` and `--deny-tool` options allow or deny permission for specific tools, or tool subcommands.

The value for each of these options is a comma-separated list of tool kinds, which can optionally specify exact tools and subcommand patterns.

If you specify a tool with `--allow-tool` , the AI model can choose to use that tool without prompting you for permission. If you specify a tool with `--deny-tool` , the AI model cannot use that tool at all, even if it would be the best choice for completing a task.

Deny rules always take precedence over allow rules, even when `--allow-all` is set.

#### Examples

| Option                                                                                                   | Effect                                                                                                                                                                                                    |
|----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--allow-tool=shell`                                                                                     | Allow all shell commands.                                                                                                                                                                                 |
| `--allow-tool='shell(git commit)'`                                                                       | Allow the `git commit` command.                                                                                                                                                                           |
| `--allow-tool='shell(git:*)' --deny-tool='shell(git push)'`                                              | Allow all `git` commands except `git push` .                                                                                                                                                              |
| `--deny-tool=write`                                                                                      | Deny all file writing operations.                                                                                                                                                                         |
| `--allow-tool='read, write(.github/copilot-instructions.md)'`                                            | Allow all read operations, and allow write operations for a specific file.                                                                                                                                |
| `--allow-tool='MyMCP(create_issue), MyMCP(delete_issue)'`                                                | Allow the `create_issue` and `delete_issue` tools from the `MyMCP` MCP server.                                                                                                                            |
| `--available-tools='bash,edit,view,grep,glob' --allow-tool='shell(git:*)' --deny-tool='shell(git push)'` | Combine both layers of control for a restricted CLI session. Copilot can explore the code, make edits, and commit changes, but can't reach the internet, run arbitrary subagents, or push to Git history. |

For details of the supported tool kinds, see [GitHub Copilot CLI command reference](/en/copilot/reference/copilot-cli-reference/cli-command-reference#tool-permission-patterns) .

### Permissive options

The following command-line options give Copilot CLI permission to use all available tools.

- `--allow-all-tools` - Full access to the available tools.
- `--allow-all` or `--yolo` - Equivalent to using all of the `--allow-all-tools` , `--allow-all-paths` , and `--allow-all-urls` options when starting the CLI. Within an interactive session, you can use the `/allow-all` or `/yolo` slash commands to allow all tools without needing to restart the session. Note It is strongly recommended that you only use these options in an isolated environment. You should never use an alias to apply one of these options every time you start Copilot CLI, as doing so would allow Copilot to use any tool without your explicit permission every time you use the CLI, which could lead to unintended consequences.

### Resetting permissions

The `/reset-allowed-tools` slash command revokes all permissions you granted during the current interactive session. This applies equally to permissions you gave by responding to prompts, and to the use of the `/allow-all` or `/yolo` slash commands.

Using `/reset-allowed-tools` resets the permissions to the default, or to the state defined by any command-line options you used when you started Copilot CLI. For example, if you start a Copilot CLI interactive session with the option `--allow-tool='shell(git:*)'` , and then you allow and deny further permissions during the session by responding to prompts, when you then use the `/reset-allowed-tools` command, the CLI's permissions return to the original `--allow-tool='shell(git:*)'` state, with no other permissions allowed or denied. As you continue to work in the session, you will be prompted again if Copilot needs additional permissions.

### Further reading

- [Best practices for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/cli-best-practices#configure-allowed-tools)
- [GitHub Copilot CLI command reference](/en/copilot/reference/copilot-cli-reference/cli-command-reference)


### Overview

You can use GitHub Copilot CLI to programmatically run Copilot prompts. There are two main ways to do this:

- Run a Copilot CLI prompt directly from your terminal.
- Write a script or automation that leverages Copilot CLI.

This guide will walk you through a simple use case for each option.

### Run a prompt from the command line

When you want to pass Copilot CLI a prompt without initiating an interactive session, use the `-p` flag.

Shell

```
copilot -p "Summarize what this file does: ./README.md"
```

Any prompt you would type in an interactive session works with `-p` .

### Use Copilot CLI in a script

The real power of programmatic mode comes from writing scripts to automate AI-powered tasks. Within a script, you can generate the prompt, or replace parts of a prompt with dynamic content, and then capture the output or pass it to another part of the script.

Let's create a script that finds all files larger than 10 MB in the current directory, uses Copilot CLI to generate a brief description of each file, and then emails a summary report.

1. In your repository, create a new file called `find_large_files.sh` and add the following content. Bash `#!/bin/bash # Find files over 10 MB, use Copilot CLI to describe them, and email a summary EMAIL_TO= "user@example.com" SUBJECT= "Large file found" BODY= "" while IFS= read -r -d '' file; do size=$( du -h " $file " | cut -f1) description=$(copilot -p "Describe this file briefly: $file " -s 2>/dev/null) BODY+= "File: $file " $ ' ' "Size: $size " $ ' ' "Description: $description " $ '  ' done < <(find . - type f -size +10M -print0) if [ -z " $BODY " ]; then echo "No files over 10MB found." exit 0 fi echo -e "To: $EMAIL_TO  Subject: $SUBJECT    $BODY " | sendmail " $EMAIL_TO " echo "Email sent to $EMAIL_TO with large file details."`
2. Make the script executable. Shell `chmod +x find_large_files.sh`
3. Run the script. Shell `./find_large_files.sh`

This script leverages Copilot CLI to generate descriptions of the files you are searching for, so you can quickly understand the contents of large files without opening them.

You can also automatically trigger these scripts in response to events, such as a new file being added to a directory, or on a schedule using cron jobs or CI/CD pipelines.

### Further reading

- [Running GitHub Copilot CLI programmatically](/en/copilot/how-tos/copilot-cli/automate-copilot-cli/run-cli-programmatically)
- [Automating tasks with Copilot CLI and GitHub Actions](/en/copilot/how-tos/copilot-cli/automate-copilot-cli/automate-with-actions)
- [GitHub Copilot CLI programmatic reference](/en/copilot/reference/copilot-cli-reference/cli-programmatic-reference)


### Introduction

You can pass a prompt directly to Copilot CLI in a single command, without entering an interactive session. This allows you to use Copilot directly from the terminal, but also allows you to use the CLI programmatically in scripts, CI/CD pipelines, and automation workflows.

To use Copilot CLI programmatically you can do either of the following.

- Use the `copilot` command with the `-p` or `--prompt` command-line option, followed by your prompt: Shell `copilot -p "Explain this file: ./complex.ts"`
- Pipe a prompt to the `copilot` command: Shell `echo "Explain this file: ./complex.ts" | copilot` Note Piped input is ignored if you also provide a prompt with the `-p` or `--prompt` option.

### Tips for using Copilot CLI programmatically

- **Provide precise prompts** - clear, unambiguous instructions produce better results than vague requests. The more context you give-file names, function names, the exact change-the less guesswork Copilot has to do.
- **Quote prompts carefully** - use single quotes around your prompt if you want to avoid shell interpretation of special characters.
- **Always give minimal permissions** - use the `--allow-tool=[TOOLS...]` and `--allow-url=[URLs...]` command-line options to give Copilot permission to use only the tools and access that are necessary to complete the task. Avoid using overly permissive options (such as `--allow-all` ) unless you are working in a sandbox environment.
- **Use** **`-s`** **(silent)** when capturing output. This suppresses session metadata so you get clean text.
- **Use** **`--no-ask-user`** to prevent the agent from attempting to ask clarifying questions.
- **Set a model explicitly** with `--model` for consistent behavior across environments.

See [GitHub Copilot CLI programmatic reference](/en/copilot/reference/copilot-cli-reference/cli-programmatic-reference) for options that are particularly useful when running Copilot CLI programmatically.

### CI/CD integration

A common use case for running Copilot CLI programmatically is to include a CLI command in a CI/CD workflow step.

This extract from a GitHub Actions workflow shows a simple example of running a Copilot CLI command.

```
### Workflow step using Copilot CLI
- name: Generate test coverage report env: COPILOT_GITHUB_TOKEN: ${{ secrets.PERSONAL_ACCESS_TOKEN }} run: |
    copilot -p "Run the test suite and produce a coverage summary" \
      -s --allow-tool='shell(npm:*), write' --no-ask-user
```

For more information, see [Automating tasks with Copilot CLI and GitHub Actions](/en/copilot/how-tos/copilot-cli/automate-copilot-cli/automate-with-actions) .

### Examples of programmatic usage

#### Generate a commit message

Bash

```
copilot -p 'Write a commit message in plain text for the staged changes' -s \
  --allow-tool= 'shell(git:*)'
```

#### Summarize a file

Bash

```
copilot -p 'Summarize what src/auth/login.ts does in no more than 100 words' -s
```

#### Write tests for a module

Bash

```
copilot -p 'Write unit tests for src/utils/validators.ts' \
  --allow-tool= 'write, shell(npm:*), shell(npx:*)'
```

#### Fix lint errors

Bash

```
copilot -p 'Fix all ESLint errors in this project' \
  --allow-tool= 'write, shell(npm:*), shell(npx:*), shell(git:*)'
```

#### Explain a diff

Bash

```
copilot -p 'Explain the changes in the latest commit on this branch and flag any potential issues' -s
```

#### Code review a branch

Use `/review` slash command to have the built-in `code-review` agent review the code changes on the current branch.

Bash

```
copilot -p '/review the changes on this branch compared to main. Focus on bugs and security issues.' \
  -s --allow-tool= 'shell(git:*)'
```

#### Generate documentation

Bash

```
copilot -p 'Generate JSDoc comments for all exported functions in src/api/' \
  --allow-tool=write
```

#### Export a session

Save the full session transcript to a Markdown file on the local filesystem.

Bash

```
copilot -p "Audit this project's dependencies for vulnerabilities" \
  --allow-tool= 'shell(npm:*), shell(npx:*)' \
  --share= './audit-report.md'
```

Save the session transcript to a gist on GitHub.com for easy sharing.

Bash

```
copilot -p 'Summarize the architecture of this project' --share-gist
```

Note

Gists are not available to Enterprise Managed Users, or if you use GitHub Enterprise Cloud with data residency (*.ghe.com).

### Shell scripting patterns

#### Capture Copilot's output in a variable

Bash

```
result=$(copilot -p 'What version of Node.js does this project require? \
  Give the number only. No other text.' -s) echo "Required Node version: $result "
```

#### Use in a conditional

Bash

```
if copilot -p 'Does this project have any TypeScript errors? Reply only YES or NO.' -s \
  | grep -qi "no" ; then echo "No type errors found."
else echo "Type errors detected."
fi
```

#### Process multiple files

Bash

```
for file in src/api/*.ts; do echo "--- Reviewing $file ---" | tee -a review-results.md
  copilot -p "Review $file for error handling issues" -s --allow-all-tools | tee -a review-results.md done
```

### Further reading

- [GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli)
- [GitHub Copilot CLI programmatic reference](/en/copilot/reference/copilot-cli-reference/cli-programmatic-reference)
- [GitHub Copilot CLI command reference](/en/copilot/reference/copilot-cli-reference/cli-command-reference#command-line-options)


### Using Copilot CLI in an Actions workflow

You can define a job in a GitHub Actions workflow that: installs Copilot CLI on the runner, authenticates it, runs it in programmatic mode, and then handles the results. Programmatic mode is designed for scripts and automation and lets you pass a prompt non-interactively.

Workflows can follow this pattern:

1. **Trigger** : Start the workflow on a schedule, in response to repository events, or manually.
2. **Setup** : Checkout code, set up environment.
3. **Install** : Install GitHub Copilot CLI on the runner.
4. **Authenticate** : Ensure the CLI has the necessary permissions to access the repository and make changes.
5. **Run Copilot CLI** : Invoke Copilot CLI with a prompt describing the task you want to automate.

#### Example workflow

The following workflow generates details of changes made today in the default branch of the repository and displays these details as the summary for the workflow run.

YAML

```
name: Daily summary
on: workflow_dispatch: # Run this workflow daily at 5:30pm UTC schedule: - cron: '30 17 * * *'
permissions: contents: read
jobs: daily-summary: runs-on: ubuntu-latest steps: - name: Checkout uses: actions/checkout@v5 with: fetch-depth: 0 - name: Set up Node.js environment uses: actions/setup-node@v4 - name: Install Copilot CLI run: npm install -g @github/copilot - name: Run Copilot CLI env: COPILOT_GITHUB_TOKEN: ${{ secrets.PERSONAL_ACCESS_TOKEN }} run: |
          copilot -p "Review the git log for this repository and write a bullet point summary of all code changes that were made today, with links to the relevant commit on GitHub. Above the bullet list give a description (max 100 words) summarizing the changes made. Write the details to summary.md" --allow-tool='shell(git:*)' --allow-tool=write --no-ask-user
          cat summary.md >> "$GITHUB_STEP_SUMMARY"
```

The following sections explain each part of this workflow.

### Trigger

In this example, the workflow runs on a daily schedule and can also be triggered manually.

The `workflow_dispatch` trigger lets you run the workflow manually from the **Actions** tab of your repository on GitHub, which is useful when testing changes to your prompt or workflow configuration.

The `schedule` trigger runs the workflow automatically at a specified time using cron syntax.

YAML

```
on: # Allows manual triggering of this workflow workflow_dispatch: # Run this workflow daily at 11:55pm UTC schedule: - cron: '55 23 * * *'
```

### Setup

Set up the job so Copilot CLI can access your repository and run on the Actions runner. This allows Copilot CLI to analyze the repository context, when generating the daily summary.

The `permissions` block defines the scope granted to the built-in `GITHUB_TOKEN` . Because this workflow reads repository data and prints a summary to the logs, it requires `contents: read` .

YAML

```
permissions: contents: read
jobs: daily-summary: runs-on: ubuntu-latest steps: - name: Checkout uses: actions/checkout@v5 with: fetch-depth: 0
```

### Install

Install Copilot CLI on the runner so your workflow can invoke it as a command. You can install GitHub Copilot CLI using any supported installation method. For a full list of installation options, see [Installing GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/install-copilot-cli) .

In this example, the workflow installs GitHub Copilot CLI globally with npm.

YAML

```
- name: Set up Node.js environment uses: actions/setup-node@v4
- name: Install Copilot CLI run: npm install -g @github/copilot
```

### Authenticate

To allow Copilot CLI to run on an Actions runner, you need to authenticate a GitHub user account with a valid Copilot license.

**Step 1: Create a personal access token (PAT) with the "Copilot Requests" permission:**

1. Go to your personal settings for creating a fine-grained personal access token: [github.com/settings/personal-access-tokens/new](https://github.com/settings/personal-access-tokens/new?ref_product=copilot&ref_type=engagement&ref_style=text) .
2. Create a new PAT with the "Copilot Requests" permission.
3. Copy the token value.

**Step 2: Store the PAT as an Actions repository secret:**

1. In your repository, go to **Settings** > **Secrets and variables** > **Actions** and click **New repository secret** .
2. Give the secret a name that you will use in the workflow. In this example we're using `PERSONAL_ACCESS_TOKEN` as the name of the secret.
3. Paste the token value into the "Secret" field and click **Add secret** .

The workflow sets a special environment variable with the value of the repository secret. Copilot CLI supports several special environment variables for authentication. In this example, the workflow uses `COPILOT_GITHUB_TOKEN` , which is specific to Copilot CLI and allows you to set different permissions for Copilot than you might use elsewhere with the built-in `GITHUB_TOKEN` environment variable.

YAML

```
- name: Run Copilot CLI env: COPILOT_GITHUB_TOKEN: ${{ secrets.PERSONAL_ACCESS_TOKEN }}
```

### Run Copilot CLI

Use `copilot -p PROMPT [OPTIONS]` to run the CLI programmatically and exit when the command completes.

The CLI prints its response to standard output, which is recorded in the log for the Actions workflow run. However, to make the details of changes easier to access, this example adds this information to the summary for the workflow run.

YAML

```
run: |
    copilot -p "Review the git log for this repository and write a bullet point summary of all code changes that were made today, with links to the relevant commit on GitHub. Above the bullet list give a description (max 100 words) summarizing the changes made. Write the details to summary.md" --allow-tool='shell(git:*)' --allow-tool=write --no-ask-user
    cat summary.md >> "$GITHUB_STEP_SUMMARY"
```

This example uses several options after the CLI prompt:

- `--allow-tool='shell(git:*)'` allows Copilot to run Git commands to analyze the repository history. This is necessary to generate the summary of recent changes.
- `--allow-tool='write'` allows Copilot to write the generated summary to a file on the runner.
- `--no-ask-user` prevents the CLI from prompting for user input, which is important when running in an automated workflow where there is no user to respond to requests for additional input.

### Next steps

After you confirm the workflow generates a summary of changes, you can adapt the same pattern to other automation tasks. Start by changing the prompt you pass to `copilot -p PROMPT` , then decide what you want to do with the output. For example, you could:

- Create a pull request to update a changelog file in the repository with the day's changes.
- Email the summary to the repository maintainers.

### Further reading

- [GitHub Copilot CLI command reference](/en/copilot/reference/copilot-cli-reference/cli-command-reference)
- [GitHub Actions documentation](/en/actions)
- [Running GitHub Copilot CLI programmatically](/en/copilot/how-tos/copilot-cli/automate-copilot-cli/run-cli-programmatically)


### Custom instructions

You can provide Copilot with instructions for how it should respond. Whenever you ask Copilot a question, or task it to perform some work, a copy of these instructions is added to your prompt. This allows you, for example, to provide details of your project's coding standards, without having to manually tell Copilot about them each time you start a conversation.

For more information, see [Adding custom instructions for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/add-custom-instructions) .

### Hooks

Hooks let you run your own shell commands at key points during a Copilot CLI session. By defining hooks, you can automate specific operations to be triggered when certain events occur: such as the start or end of a session, whenever someone submits a prompt, after the agent completes a task, or when an error occurs.

For example, you could set up a hook to automatically run tests after Copilot makes changes to code files.

For more information, see [Using hooks with GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/use-hooks) .

### Skills

Skills are folders of instructions, scripts, and resources that Copilot can load to improve its performance on specialized tasks. By adding skills to your project, you can give Copilot extra knowledge or tools for particular workflows, technologies, or domains.

For more information, see [Creating agent skills for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/create-skills) .

### Custom agents

Custom agents let you define specific expertise and behavior for the CLI when it works on a particular type of task. Custom agents are run as subagents-separately to the main agent that responds to a prompt-with their own context window. This allows Copilot to offload work to custom agents without cluttering the main agent's context window, and to use the expertise of a custom agent when it's a good fit for the task at hand.

You can define the toolset available to a custom agent, so that the tools the agent can use are appropriate to its role. For example, a custom agent that works as a reviewer would typically not be permitted to make changes to code files.

For more information, see [Creating and using custom agents for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/create-custom-agents-for-cli) .

### MCP servers

The Model Context Protocol (MCP) allows you to add external tools and data sources to Copilot CLI. By adding MCP servers to Copilot CLI you can add functionality such as the ability to:

- Query databases
- Access issue tracking systems
- Integrate with CI/CD pipelines
- Generate design diagrams
- Search specialist documentation sources
- Book tickets online
- Integrate with a calendar application

For more information, see [About Model Context Protocol (MCP)](/en/copilot/concepts/context/mcp) .

### Plugins

Copilot CLI plugins are distributable packages that provide a simple way to extend the functionality of the CLI.

They bundle multiple customization components together into a single installable unit. You can install plugins directly from a repository, from a plugin marketplace, or from a local path.

For more information, see [About plugins for GitHub Copilot CLI](/en/copilot/concepts/agents/copilot-cli/about-cli-plugins) .


### Types of custom instructions

GitHub Copilot CLI supports the following types of custom instructions.

#### Repository-wide custom instructions

These apply to all requests made in the context of a repository.

These are specified in a `copilot-instructions.md` file in the `.github` directory at the root of the repository. See [Creating repository-wide custom instructions](#creating-repository-wide-custom-instructions) .

#### Path-specific custom instructions

These apply to requests made in the context of files that match a specified path.

These are specified in one or more `NAME.instructions.md` files within or below the `.github/instructions` directory at the root of the repository, or within or below a `.github/instructions` directory in the current working directory. See [Creating path-specific custom instructions](#creating-path-specific-custom-instructions) .

If the path you specify in these instructions matches a file that Copilot is working on, and a repository-wide custom instructions file also exists, then the instructions from both files are used. You should avoid potential conflicts between instructions as Copilot's choice between conflicting instructions is non-deterministic.

#### Agent instructions

These are used by various AI agents.

You can create one or more `AGENTS.md` files. These can be located in the repository's root directory, in the current working directory, or in any of the directories specified by a comma-separated list of paths in the `COPILOT_CUSTOM_INSTRUCTIONS_DIRS` environment variable.

Instructions in the `AGENTS.md` file in the root directory, if found, are treated as primary instructions. If an `AGENTS.md` file and a `.github/copilot-instructions.md` file are both found at the root of the repository, the instructions in both files are used.

Instructions found in other `AGENTS.md` files are treated as additional instructions. Any primary instructions that are found are likely to have more effect on Copilot's responses than additional instructions.

For more information, see the [agentsmd/agents.md repository](https://github.com/agentsmd/agents.md) .

Alternatively, you can use `CLAUDE.md` and `GEMINI.md` files. These must be located at the root of the repository.

#### Local instructions

These apply within a specific local environment.

You can specify instructions within your own home directory, by creating a file at `$HOME/.copilot/copilot-instructions.md` .

You can also set the `COPILOT_CUSTOM_INSTRUCTIONS_DIRS` environment variable to a comma-separated list of directories. Copilot CLI will look for an `AGENTS.md` file, and any `.github/instructions/**/*.instructions.md` files, in each of these directories.

### Creating repository-wide custom instructions

1. In the root of your repository, create a file named `.github/copilot-instructions.md` . Create the `.github` directory if it does not already exist.
2. Add natural language instructions to the file, in Markdown format. Whitespace between instructions is ignored, so the instructions can be written as a single paragraph, each on a new line, or separated by blank lines for legibility. For help on writing effective custom instructions, see [About customizing GitHub Copilot responses](/en/copilot/concepts/prompting/response-customization#writing-effective-custom-instructions) .

### Creating path-specific custom instructions

1. Create the `.github/instructions` directory if it does not already exist.
2. Optionally, create subdirectories of `.github/instructions` to organize your instruction files.
3. Create one or more `NAME.instructions.md` files, where `NAME` indicates the purpose of the instructions. The file name must end with `.instructions.md` .
4. At the start of the file, create a frontmatter block containing the `applyTo` keyword. Use glob syntax to specify what files or directories the instructions apply to. For example: `--- applyTo: "app/models/ **/ *.rb" ---` You can specify multiple patterns by separating them with commas. For example, to apply the instructions to all TypeScript files in the repository, you could use the following frontmatter block: `--- applyTo: " **/ *.ts,* */* .tsx" ---` Glob examples:
    - `*` - will all match all files in the current directory.
    - `**` or `**/*` - will all match all files in all directories.
    - `*.py` - will match all `.py` files in the current directory.
    - `**/*.py` - will recursively match all `.py` files in all directories.
    - `src/*.py` - will match all `.py` files in the `src` directory. For example, `src/foo.py` and `src/bar.py` but *not* `src/foo/bar.py` .
    - `src/**/*.py` - will recursively match all `.py` files in the `src` directory. For example, `src/foo.py` , `src/foo/bar.py` , and `src/foo/bar/baz.py` .
    - `**/subdir/**/*.py` - will recursively match all `.py` files in any `subdir` directory at any depth. For example, `subdir/foo.py` , `subdir/nested/bar.py` , `parent/subdir/baz.py` , and `deep/parent/subdir/nested/qux.py` , but *not* `foo.py` at a path that does not contain a `subdir` directory.
5. Optionally, to prevent the file from being used by either Copilot cloud agent or Copilot code review, add the `excludeAgent` keyword to the frontmatter block. Use either `"code-review"` or `"cloud-agent"` . For example, the following file will only be read by Copilot cloud agent. `--- applyTo: " **" excludeAgent: "code-review" ---` If the `excludeAgent` keyword is not included in the front matterblock, both Copilot code review and Copilot cloud agent will use your instructions.
6. Add your custom instructions in natural language, using Markdown format. Whitespace between instructions is ignored, so the instructions can be written as a single paragraph, each on a new line, or separated by blank lines for legibility.

Did you successfully add a custom instructions file to your repository?

[Yes](https://docs.github.io/success-test/yes.html) [No](https://docs.github.io/success-test/no.html)

### Custom instructions in use

The instructions in the file(s) are available for use by Copilot as soon as you save the file(s). Instructions are automatically added to requests that you submit to Copilot.

If you make changes to your custom instructions during a CLI session, your changes are available for use by Copilot the next time you submit a prompt in the current, or future, sessions.

### Further reading

- [Support for different types of custom instructions](/en/copilot/reference/custom-instructions-support)
- [Custom instructions](/en/copilot/tutorials/customization-library/custom-instructions) -a curated collection of examples
- [Using custom instructions to unlock the power of Copilot code review](/en/copilot/tutorials/use-custom-instructions)


### Creating a hook in a repository on GitHub

1. Create a new `hooks.json` file with the name of your choice in the `.github/hooks/` folder of your repository. The hooks configuration file **must be present** on your repository's default branch to be used by Copilot cloud agent. For GitHub Copilot CLI, hooks are loaded from your current working directory.
2. In your text editor, copy and paste the following hook template. Remove any hooks you don't plan on using from the `hooks` array. JSON `{ "version" : 1 , "hooks" : { "sessionStart" : [ ... ] , "sessionEnd" : [ ... ] , "userPromptSubmitted" : [ ... ] , "preToolUse" : [ ... ] , "postToolUse" : [ ... ] , "errorOccurred" : [ ... ] } }`
3. Configure your hook syntax under the `bash` or `powershell` keys, or directly reference script files you have created.
    - This example runs a script that outputs the start date of the session to a log file using the `sessionStart` hook: JSON `"sessionStart" : [ { "type" : "command" , "bash" : "echo \"Session started: $(date)\" >> logs/session.log" , "powershell" : "Add-Content -Path logs/session.log -Value \"Session started: $(Get-Date)\"" , "cwd" : "." , "timeoutSec" : 10 } ] ,`
    - This example calls out to an external `log-prompt` script: JSON `"userPromptSubmitted" : [ { "type" : "command" , "bash" : "./scripts/log-prompt.sh" , "powershell" : "./scripts/log-prompt.ps1" , "cwd" : "scripts" , "env" : { "LOG_LEVEL" : "INFO" } } ] ,` For a full reference on the input JSON from agent sessions along with sample scripts, see [Hooks configuration](/en/copilot/reference/hooks-configuration) .
4. Commit the file to the repository and merge it into the default branch. Your hooks will now run during agent sessions.

### Troubleshooting

If you run into problems using hooks, use the following table to troubleshoot.

| Issue                   | Action                                                                                                                                                                                                                                                                                                                                                           |
|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Hooks are not executing | - Verify the JSON file is in the `.github/hooks/` directory. - Check for valid JSON syntax (for example, `jq . hooks.json` ). - Ensure `version: 1` is specified in your `hooks.json` file. - Verify the script you are calling from your hook is executable ( `chmod +x script.sh` ) - Check that the script has a proper shebang (for example, `#!/bin/bash` ) |
| Hooks are timing out    | - The default timeout is 30 seconds. Increase `timeoutSec` in the configuration if needed. - Optimize script performance by avoiding unnecessary operations.                                                                                                                                                                                                     |
| Invalid JSON output     | - Ensure the output is on a single line. - On Unix, use `jq -c` to compact and validate the JSON output. - On Windows, use the `ConvertTo-Json -Compress` command in PowerShell to do the same.                                                                                                                                                                  |

### Debugging

You can debug hooks using the following methods:

- **Enable verbose logging** in the script to inspect the input data and trace script execution. Shell `# !/bin/bash set -x # Enable bash debug mode INPUT=$(cat) echo "DEBUG: Received input" >&2 echo "$INPUT" >&2 # ... rest of script`
- **Test hooks locally** by piping test input into your hook to validate its behavior: Shell `# Create test input echo '{"timestamp":1704614400000,"cwd":"/tmp","toolName":"bash","toolArgs":"{\"command\":\"ls\"}"}' | ./my-hook.sh # Check exit code echo $? # Validate output is valid JSON ./my-hook.sh | jq .`

### Further reading

- [Hooks configuration](/en/copilot/reference/hooks-configuration)
- [About GitHub Copilot cloud agent](/en/copilot/concepts/agents/cloud-agent/about-cloud-agent)
- [About GitHub Copilot CLI](/en/copilot/concepts/agents/about-copilot-cli)
- [Customizing the development environment for GitHub Copilot cloud agent](/en/copilot/how-tos/use-copilot-agents/cloud-agent/customize-the-agent-environment)


### Creating and adding a skill

To create an agent skill you write a `SKILL.md` file and, optionally, other resources, such as supplementary Markdown files, or scripts, which you reference in the `SKILL.md` instructions.

To add a skill, you save the `SKILL.md` file, and any subsidiary resources, to a location where Copilot knows to look for skills. This can be within a repository, or within your home directory.

1. Create a `skills` directory in one of the supported locations to store your skill and any others you may want to create in the future. For **project skills** , specific to a single repository, create and use a `.github/skills` , `.claude/skills` , or `.agents/skills` directory in your repository. For **personal skills** , shared across projects, create and use a `~/.copilot/skills` , `~/.claude/skills` , or `~/.agents/skills` directory in your home directory.
2. Create a subdirectory for your new skill. Each skill should have its own directory (for example, `.github/skills/webapp-testing` ). Skill subdirectory names should be lowercase and use hyphens for spaces.
3. In your skill subdirectory, create a `SKILL.md` file containing your skill's instructions. Important Skill files must be named `SKILL.md` . `SKILL.md` files are Markdown files with YAML frontmatter. In their simplest form, they include:
    - YAML frontmatter
        - **name** (required): A unique identifier for the skill. This must be lowercase, using hyphens for spaces. Typically, this matches the name of the skill's directory.
        - **description** (required): A description of what the skill does, and when Copilot should use it.
        - **license** (optional): A description of the license that applies to this skill.
    - A Markdown body, with the instructions, examples and guidelines for Copilot to follow.
4. Optionally, add scripts, examples or other resources to your skill's directory. For more information, see " [Enabling a skill to run a script](#enabling-a-skill-to-run-a-script) ."

#### Example SKILL.md file

For a **project skill** , this file would be located in a `.github/skills/github-actions-failure-debugging` directory of your repository.

For a **personal skill** , this file would be located in a `~/.copilot/skills/github-actions-failure-debugging` directory.

Markdown

```
---
name: github-actions-failure-debugging description: Guide for debugging failing GitHub Actions workflows. Use this when asked to debug failing GitHub Actions workflows.
--- To debug failing GitHub Actions workflows in a pull request, follow this process, using tools provided from the GitHub MCP Server: 1. Use the `list_workflow_runs` tool to look up recent workflow runs for the pull request and their status 2. Use the `summarize_job_log_failures` tool to get an AI summary of the logs for failed jobs, to understand what went wrong without filling your context windows with thousands of lines of logs 3. If you still need more information, use the `get_job_logs` or `get_workflow_run_logs` tool to get the full, detailed failure logs 4. Try to reproduce the failure yourself in your own environment. 5. Fix the failing build. If you were able to reproduce the failure yourself, make sure it is fixed before committing your changes.
```

#### Enabling a skill to run a script

When a skill is invoked, Copilot automatically discovers all of the files in the skill's directory and makes them available alongside the skill's instructions. This means you can include scripts or other resources in the skill directory and reference them in your `SKILL.md` instructions.

To create a skill that runs a script:

1. **Add the script to your skill's directory.** For example, a skill for converting SVG images to PNG might have the following structure. `.github/skills/image-convert/ ├── SKILL.md └── convert-svg-to-png.sh`
2. **Optionally pre-approve the tools the skill needs.** In your `SKILL.md` frontmatter, you can use the `allowed-tools` field to list the tools Copilot may use without asking for confirmation each time. If a tool is not listed in the `allowed-tools` field, Copilot will prompt you for permission before using it. `--- name: image-convert description: Converts SVG images to PNG format. Use when asked to convert SVG files. allowed-tools: shell ---` Warning Only pre-approve the `shell` or `bash` tools if you have reviewed this skill and any referenced scripts, and you fully trust their source. Pre-approving `shell` or `bash` removes the confirmation step for running terminal commands and can allow attacker-controlled skills or prompt injections to execute arbitrary commands in your environment. When in doubt, omit `shell` and `bash` from `allowed-tools` so that Copilot must ask for your explicit confirmation before running terminal commands.
3. **Write instructions that tell Copilot how to use the script.** In the Markdown body of `SKILL.md` , describe when and how to run the script. `When asked to convert an SVG to PNG, run the `convert-svg-to-png.sh` script from this skill's base directory, passing the input SVG file path as the first argument.`

### Using agent skills

When performing tasks, Copilot will decide when to use your skills based on your prompt and the skill's description.

When Copilot chooses to use a skill, the `SKILL.md` file will be injected in the agent's context, giving the agent access to your instructions. It can then follow those instructions and use any scripts or examples you may have included in the skill's directory.

To tell Copilot to use a specific skill, include the skill name in your prompt, preceded by a forward slash. For example, if you have a skill named "frontend-design" you could use a prompt such as:

```
Use the /frontend-design skill to create a responsive navigation bar in React.
```

#### Skills commands in the CLI

- **List the currently available skills:** use the command `/skills list` or the prompt: `What skills do you have?`
- **Enable or disable specific skills:** use the command `/skills` and then use the up and down keys on your keyboard, and the space bar, to toggle skills on or off.
- **Find out more about a skill** (including its location): use the command `/skills info` .
- **Add a skills location:** to add an alternative location in which to store skills, use the command `/skills add` .
- **Reload skills:** if you have added a skill during a CLI session, you can add it using the command `/skills reload` to avoid having to restart the CLI to use it.
- **Remove skills:** to remove a skill that you have added directly-not via a plugin-use the command `/skills remove SKILL-DIRECTORY` . To remove skills added as part of a plugin you must manage the plugin itself. Use the `info` subcommand to find out which plugin a skill came from.

### Skills versus custom instructions

You can use both skills and custom instructions to teach Copilot how to work in your repository and how to perform specific tasks.

We recommend using **custom instructions** for simple instructions relevant to almost every task (for example information about your repository's coding standards), and **skills** for more detailed instructions that Copilot should only access when relevant.

To learn more about repository custom instructions, see [Adding repository custom instructions for GitHub Copilot](/en/copilot/how-tos/configure-custom-instructions/add-repository-instructions) .

To learn more about how skills differ from other customization features, see [Comparing GitHub Copilot CLI customization features](/en/copilot/concepts/agents/copilot-cli/comparing-cli-features) .


### Adding an MCP server

Note

The GitHub MCP server is built into Copilot CLI and is already available without any additional configuration. The steps below are for adding other MCP servers.

You can add MCP servers using the interactive `/mcp add` command within the CLI, or by editing the configuration file directly.

For installation instructions, available tools, and URLs for specific MCP servers, see the [GitHub MCP Registry](https://github.com/mcp) .

#### Using the /mcp add command

1. In interactive mode, enter `/mcp add` . A configuration form is displayed. Use `Tab` to navigate between fields.
2. Next to **Server Name** , enter a unique name for the MCP server. This is the name you will use to refer to the server.
3. Next to **Server Type** , select a type by pressing the corresponding number. The following types are available:
    - **Local** or **STDIO** : starts a local process and communicates over standard input/output ( `stdin` / `stdout` ). Both options work the same way. **STDIO** is the standard MCP protocol type name, so choose this if you want your configuration to be compatible with VS Code, the Copilot cloud agent, and other MCP clients.
    - **HTTP** or **SSE** : connects to a remote MCP server. **HTTP** uses the Streamable HTTP transport. **SSE** uses the legacy HTTP with Server-Sent Events transport, which is deprecated in the MCP specification but still supported for backwards compatibility.
4. The remaining fields depend on the server type you selected:
    - If you chose **Local** or **STDIO** :
        - Next to **Command** , enter the command to start the server, including any arguments. For example, `npx @playwright/mcp@latest` . This corresponds to both the `command` and `args` properties in the JSON configuration file.
        - Next to **Environment Variables** , optionally specify environment variables the server needs, such as API keys or tokens, as JSON key-value pairs. For example, `{"API_KEY": "YOUR-API-KEY"}` . The `PATH` variable is automatically inherited from your environment. All other environment variables must be configured here.
    - If you chose **HTTP** or **SSE** :
        - Next to **URL** , paste the remote server URL. For example, `https://mcp.context7.com/mcp` .
        - Next to **HTTP Headers** , optionally specify HTTP headers as JSON. For example, `{"CONTEXT7_API_KEY": "YOUR-API-KEY"}` .
5. Next to **Tools** , specify which tools from the server should be available. Enter `*` to include all tools, or provide a comma-separated list of tool names (no quotes needed). The default is `*` .
6. Press `Ctrl` + `S` to save the configuration. The MCP server is added and available immediately without restarting the CLI.

#### Editing the configuration file

You can also add MCP servers by editing the configuration file at `~/.copilot/mcp-config.json` . This is useful if you want to share configurations or add multiple servers at once.

The following example shows a configuration file with a local server and a remote HTTP server:

JSON

```
{ "mcpServers" : { "playwright" : { "type" : "local" , "command" : "npx" , "args" : [ "@playwright/mcp@latest" ] , "env" : { } , "tools" : [ "*" ] } , "context7" : { "type" : "http" , "url" : "https://mcp.context7.com/mcp" , "headers" : { "CONTEXT7_API_KEY" : "YOUR-API-KEY" } , "tools" : [ "*" ] } }
}
```

For more information on MCP server configuration, see [Extending GitHub Copilot cloud agent with the Model Context Protocol (MCP)](/en/copilot/how-tos/use-copilot-agents/cloud-agent/extend-cloud-agent-with-mcp#writing-a-json-configuration-for-mcp-servers) .

### Managing MCP servers

You can manage your configured MCP servers using the following `/mcp` commands in Copilot CLI.

- **List configured MCP servers:** Use the command `/mcp show` . This displays all configured MCP servers and their current status.
- **View details about a specific server:** Use the command `/mcp show SERVER-NAME` . This displays the status of the specified server and the list of tools it provides.
- **Edit a server's configuration:** Use the command `/mcp edit SERVER-NAME` .
- **Delete a server:** Use the command `/mcp delete SERVER-NAME` .
- **Disable a server:** Use the command `/mcp disable SERVER-NAME` . A disabled server remains configured but is not used by Copilot for the current session.
- **Enable a previously disabled server:** Use the command `/mcp enable SERVER-NAME` .

### Using MCP servers

Once you have added an MCP server, Copilot can automatically use the tools it provides when relevant to your prompt. You can also directly reference an MCP server and specific tools in a prompt to ensure they are used.

### Further reading

- [About Model Context Protocol (MCP)](/en/copilot/concepts/about-mcp)
- [Extending GitHub Copilot Chat with Model Context Protocol (MCP) servers](/en/copilot/how-tos/provide-context/use-mcp/extend-copilot-chat-with-mcp)
- [Extending GitHub Copilot cloud agent with the Model Context Protocol (MCP)](/en/copilot/how-tos/use-copilot-agents/cloud-agent/extend-cloud-agent-with-mcp)


### Introduction

Custom agents allow you to tailor Copilot's expertise for specific tasks.

When you prompt Copilot to carry out a task it may choose to use one of your custom agents, if Copilot determines that the agent's expertise is a good fit for the task.

Work performed by a custom agent is carried out using a subagent, which is a temporary agent spun up to complete the task. The subagent has its own context window, which can be populated by information that is not relevant to the main agent. In this way, especially for larger tasks, parts of the work can be offloaded to custom agents, without cluttering the main agent's context window. The main agent can then focus on higher-level planning and coordination.

For more information, see [About custom agents](/en/copilot/concepts/agents/copilot-cli/about-custom-agents) .

### Creating a custom agent

Each custom agent is defined by a Markdown file with an `.agent.md` extension. You can create these files yourself, or you can add them from within the CLI, as described in the following steps.

1. In interactive mode, enter `/agent` .
2. Select **Create new agent** from the list of options.
3. Choose between the options to create the custom agent in the repository or in your home directory: Note If you have custom agents with the same name in both locations, the one in your home directory will be used, rather than the one in the repository.
    - **Project** ( `.github/agents/` )
    - **User** ( `~/.copilot/agents/` )
4. Choose whether to get Copilot to create the custom agent file, or create it yourself. **Option 1: Use Copilot** Enter details of the agent you want to create. Describe the agent's expertise and when the agent should be used. Copilot will take the description you enter and use it to write an agent profile for you. For example, you could enter: `I am a security expert. I check code files thoroughly for potential security issues. Use me whenever a security review/check/audit is requested for one or more code files, or when the word "seccheck" is used in a prompt in reference to code files. I will identify potential problems, such as code that: - Exposes secrets or credentials - Allows cross-site scripting - Allows SQL injection - Contains vulnerable dependencies - Allows authentication to be bypassed If any problems are identified, create a single GitHub issue in this repository on GitHub.com with details of problems, giving full details of each issue, including, but not limited to, risk level and recommended fix.` After Copilot finishes generating the initial agent profile it displays the following options: If you choose to review the content, the agent file is opened in your default editor. You can review and make changes, if required, before continuing the agent creation process in the CLI. To complete the creation process, choose **Continue** . **Option 2: Create the agent profile manually** When you choose to create the agent file yourself, you'll be guided through a series of prompts to fill in the necessary information to create the agent profile.
    - Continue
    - Review content
    - Try again
    - Quit
    1. Enter a name for the agent. The name you enter is the name that's displayed when you list the available agents. A version of this will be used as the name of the agent file-for example, if you enter "Security expert", the agent file will be named `security-expert.agent.md` . Tip For ease of use when using a custom agent programmatically, it's recommended that you choose a name consisting only of lowercase letters and hyphens.
    2. Enter a description that states what expertise this agent has and when it should be used.
    3. Enter instructions for how the agent should behave, including any specific guidelines, actions it should take or constraints it should follow.
5. Choose which tools your custom agent should have access to. By default, custom agents have access to all tools. If you restrict an agent's access, a `tools` specification is added to the agent file.
6. Restart the CLI to load your new custom agent.

### Using a custom agent

Custom agents can be used in the following ways:

- **Slash command** Enter `/agent` in interactive mode and choose from the list of available custom agents. Then enter a prompt that will be passed to the selected agent. Note The CLI's default agents are not included in this list. For more information about the default agents, see [Using GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/use-copilot-cli#use-custom-agents) .
- **Explicit instruction** Tell Copilot to use a specific agent. For example: `Use the security-auditor agent on all files in the /src/app directory`
- **By inference** Use a prompt that will trigger the use of a particular agent based on the description in the agent file. For example: `Check all TypeScript files in or under the src directory for potential security problems` or (where "seccheck" is defined as a trigger word in the agent profile): `seccheck /src/app/validator.go` Copilot will automatically infer the agent you want to use.
- **Programmatically** Specify the custom agent you want to use with the command-line option. For example: `copilot --agent security-auditor --prompt "Check /src/app/validator.go"` Where `security-auditor` is the file name of the custom agent profile, without the `.agent.md` extension. Typically, but not necessarily, this is the same as the `name` value in the agent profile.

### Further reading

- [Comparing GitHub Copilot CLI customization features](/en/copilot/concepts/agents/copilot-cli/comparing-cli-features)
- [Custom agents configuration](/en/copilot/reference/custom-agents-configuration)
- [GitHub Copilot CLI command reference](/en/copilot/reference/copilot-cli-reference/cli-command-reference#custom-agents-reference)
- [Custom agents](/en/copilot/tutorials/customization-library/custom-agents) -a curated collection of examples


### Prerequisites

- Copilot CLI is installed. See [Installing GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/install-copilot-cli) .
- You have an API key from a supported LLM provider, or you have a local model running (such as Ollama).

### Supported providers

Copilot CLI supports three provider types:

| Provider type   | Compatible services                                                                                                                    |
|-----------------|----------------------------------------------------------------------------------------------------------------------------------------|
| `openai`        | OpenAI, Ollama, vLLM, Foundry Local, and any other OpenAI Chat Completions API-compatible endpoint. This is the default provider type. |
| `azure`         | Azure OpenAI Service.                                                                                                                  |
| `anthropic`     | Anthropic (Claude models).                                                                                                             |

For additional examples, run `copilot help providers` in your terminal.

### Model requirements

Models must support **tool calling** (also called function calling) and **streaming** . If a model does not support either capability, Copilot CLI returns an error. For best results, use a model with a context window of at least 128k tokens.

### Configuring your provider

You configure your model provider by setting environment variables before starting Copilot CLI.

| Environment variable        | Required   | Description                                                                                                                |
|-----------------------------|------------|----------------------------------------------------------------------------------------------------------------------------|
| `COPILOT_PROVIDER_BASE_URL` | Yes        | The base URL of your model provider's API endpoint.                                                                        |
| `COPILOT_PROVIDER_TYPE`     | No         | The provider type: `openai` (default), `azure` , or `anthropic` .                                                          |
| `COPILOT_PROVIDER_API_KEY`  | No         | Your API key for the provider. Not required for providers that do not use authentication, such as a local Ollama instance. |
| `COPILOT_MODEL`             | Yes        | The model identifier to use. You can also set this with the `--model` command-line flag.                                   |

### Connecting to an OpenAI-compatible endpoint

Use the following steps if you are connecting to OpenAI, Ollama, vLLM, Foundry Local, or any other endpoint that is compatible with the OpenAI Chat Completions API.

1. Set environment variables for your provider. For example, for a local Ollama instance: `export COPILOT_PROVIDER_BASE_URL=http://localhost:11434 export COPILOT_MODEL=YOUR-MODEL-NAME` Replace `YOUR-MODEL-NAME` with the name of the model you have pulled in Ollama (for example, `llama3.2` ).
2. For a remote OpenAI endpoint, also set your API key. `export COPILOT_PROVIDER_BASE_URL=https://api.openai.com/v1 export COPILOT_PROVIDER_API_KEY=YOUR-OPENAI-API-KEY export COPILOT_MODEL=YOUR-MODEL-NAME` Replace `YOUR-OPENAI-API-KEY` with your OpenAI API key and `YOUR-MODEL-NAME` with the model you want to use (for example, `gpt-4o` ).
3. Start Copilot CLI.

```
copilot
```

### Connecting to Azure OpenAI

1. Set the environment variables for Azure OpenAI. `export COPILOT_PROVIDER_BASE_URL=https://YOUR-RESOURCE-NAME.openai.azure.com/openai/deployments/YOUR-DEPLOYMENT-NAME export COPILOT_PROVIDER_TYPE=azure export COPILOT_PROVIDER_API_KEY=YOUR-AZURE-API-KEY export COPILOT_MODEL=YOUR-DEPLOYMENT-NAME` Replace the following placeholders:
    - `YOUR-RESOURCE-NAME` : your Azure OpenAI resource name
    - `YOUR-DEPLOYMENT-NAME` : the name of your model deployment
    - `YOUR-AZURE-API-KEY` : your Azure OpenAI API key
2. Start Copilot CLI.

```
copilot
```

### Connecting to Anthropic

1. Set the environment variables for Anthropic: `export COPILOT_PROVIDER_TYPE=anthropic export COPILOT_PROVIDER_BASE_URL=https://api.anthropic.com export COPILOT_PROVIDER_API_KEY=YOUR-ANTHROPIC-API-KEY export COPILOT_MODEL=YOUR-MODEL-NAME` Replace `YOUR-ANTHROPIC-API-KEY` with your Anthropic API key and YOUR-MODEL-NAME with the Claude model you want to use (for example, `claude-opus-4-5` ).
2. Start Copilot CLI.

```
copilot
```

### Running in offline mode

You can run Copilot CLI in offline mode to prevent it from contacting GitHub's servers. This is designed for isolated environments where the CLI should communicate only with your local or on-premises model provider.

Important

Offline mode only guarantees full network isolation if your provider is also local or within the same isolated environment. If `COPILOT_PROVIDER_BASE_URL` points to a remote endpoint, your prompts and code context are still sent over the network to that provider.

1. Configure your provider environment variables as described in Configuring your provider.
2. Set the offline mode environment variable: `export COPILOT_OFFLINE=true`
3. Start Copilot CLI.

```
copilot
```


### Introduction

Plugins are packages that extend the functionality of Copilot CLI. You can install a plugin from a marketplace that you have registered with the CLI, from a Git repository, or from a local path.

For more information, see [About plugins for GitHub Copilot CLI](/en/copilot/concepts/agents/copilot-cli/about-cli-plugins) .

Note

You can find help on using plugins by entering `copilot plugin [SUBCOMMAND] --help` in the terminal.

### Finding plugins

Plugins are collected together in marketplaces. A marketplace is a registry of plugins that you can browse and install from. You can add a marketplace to your CLI configuration, which allows you to use the CLI to browse and install plugins from that marketplace-see [Adding plugin marketplaces](#adding-plugin-marketplaces) . Copilot comes with two marketplaces already registered by default: `copilot-plugins` and `awesome-copilot` .

Alternatively, you can search for plugin marketplaces online and then add a plugin directly from a repository.

To use the CLI to browse the plugins in one of your registered marketplaces:

1. **Check which marketplaces are currently registered.** In the terminal, list the available marketplaces by entering: Shell `copilot plugin marketplace list` Alternatively, in an interactive session, enter: Copilot prompt `/plugin marketplace list`
2. **Browse the plugins in a registered marketplace.** From the list of registered marketplaces, copy the name of the marketplace you want to browse-for example, `awesome-copilot` -then enter the following command, replacing `MARKETPLACE-NAME` : Shell `copilot plugin marketplace browse MARKETPLACE-NAME`

### Installing plugins

Typically, you'll install a plugin from one of your registered marketplaces. However, you can also install a plugin directly from a Git repository, or from a local path.

For information on how to register additional marketplaces, see [Adding and removing plugin marketplaces](#adding-and-removing-plugin-marketplaces) .

#### Install from a registered marketplace

Shell

```
copilot plugin install PLUGIN-NAME@MARKETPLACE-NAME
```

For example, to install the `database-data-management` plugin from the `awesome-copilot` marketplace enter:

Shell

```
copilot plugin install database-data-management@awesome-copilot
```

Alternatively, in an interactive session, enter:

Copilot prompt

```
/plugin install PLUGIN-NAME@MARKETPLACE-NAME
```

#### Install directly from an online Git repository

You can install a plugin directly from a repository, rather than doing so using a registered marketplace.

To install a plugin directly from a repository **on GitHub.com** , enter:

Shell

```
copilot plugin install OWNER/REPO
```

To install a plugin from **any online Git repository** , enter:

Shell

```
copilot plugin install URL-OF-GIT-REPO
```

For example, `copilot plugin install https://gitlab.com/OWNER/REPO.git` .

Important

For these commands to work, the repository must contain a `plugin.json` file in a `.plugin` , `.github/plugin` , or `.claude-plugin` directory, or at the root of the repository.

To install a plugin directly from a repository on GitHub.com where the `plugin.json` file is located somewhere other than `.github/plugin` , `.claude-plugin` , or the repository root-for example, if you are installing a plugin directly from a marketplace repository such as [anthropics/claude-code](https://github.com/anthropics/claude-code) -enter:

Shell

```
copilot plugin install OWNER/REPO:PATH/TO/PLUGIN
```

Where `PATH/TO/PLUGIN` is the path from the root of the repository to a directory that contains `plugin.json` , `.github/plugin/plugin.json` or `.claude-plugin/plugin.json` .

For example, `copilot plugin install anthropics/claude-code:plugins/frontend-design`

#### Install from a local path

Shell

```
copilot plugin install ./PATH/TO/PLUGIN
```

### Managing installed plugins

```
copilot plugin list # View installed plugins copilot plugin update PLUGIN-NAME # Update plugin to latest version copilot plugin uninstall PLUGIN-NAME # Remove plugin completely
```

### Where plugins are stored

Plugins installed from a marketplace are stored at: `~/.copilot/installed-plugins/MARKETPLACE/PLUGIN-NAME/` . Plugins installed directly (for example, from a local path) are stored at: `~/.copilot/installed-plugins/_direct/SOURCE-ID/` .

### Adding plugin marketplaces

To add a marketplace to the list of registered marketplaces, enter the following command in the terminal:

Shell

```
copilot plugin marketplace add OWNER/REPO
```

Where OWNER/REPO identifies a repository on GitHub.com that has been configured as a CLI plugin marketplace.

For example to add the `claude-code-plugins` marketplace, hosted at [https://github.com/anthropics/claude-code](https://github.com/anthropics/claude-code) , enter:

Shell

```
copilot plugin marketplace add anthropics/claude-code
```

Alternatively, in an interactive session, enter:

Copilot prompt

```
/plugin marketplace add OWNER/REPO
```

If a marketplace is located on the local file system, instead of on GitHub.com, use the path to the marketplace directory instead of OWNER/REPO. For example:

Shell

```
copilot plugin marketplace add /PATH/TO/MARKETPLACE-DIRECTORY
```

If a marketplace is located in a Git repository that is not hosted on GitHub.com, use the URL of the Git repository. For example:

Shell

```
copilot plugin marketplace add https://gitlab.com/OWNER/REPO.git
```

### Removing plugin marketplaces

To remove a marketplace from the CLI enter:

Shell

```
copilot plugin marketplace remove MARKETPLACE-NAME
```

Or, in an interactive session:

Copilot prompt

```
/plugin marketplace remove MARKETPLACE-NAME
```

Note

- When adding a marketplace you reference the marketplace using the OWNER/REPO of the GitHub repository that has been configured as a marketplace. When removing a marketplace, however, you reference the name of the marketplace as it appears in your list of registered marketplaces.
- If you attempt to remove a marketplace that has plugins installed, the command will fail with an error message that lists the plugins that are currently installed from that marketplace. Add the `--force` option to the command to remove the marketplace and uninstall all plugins that were installed from that marketplace.

### Further reading

- [Creating a plugin for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/plugins-creating)
- [Creating a plugin marketplace for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/plugins-marketplace)


### Introduction

Plugins are packages that extend the functionality of Copilot CLI. See [About plugins for GitHub Copilot CLI](/en/copilot/concepts/agents/copilot-cli/about-cli-plugins) .

Note

You can find help on using plugins by entering `copilot plugin [SUBCOMMAND] --help` in the terminal.

### Plugin structure

A plugin consists of a directory with a specific structure. At minimum, it must contain a `plugin.json` manifest file at the root of the directory. It can also contain any combination of agents, skills, hooks, and MCP server configurations.

#### Example plugin structure

```
my-plugin/
├── plugin.json           # Required manifest
├── agents/               # Custom agents (optional)
│   └── helper.agent.md
├── skills/               # Skills (optional)
│   └── deploy/
│       └── SKILL.md
├── hooks.json            # Hook configuration (optional)
└── .mcp.json             # MCP server config (optional)
```

### Creating a plugin

1. Create a directory for your plugin.
2. Add a `plugin.json` manifest file to the root of the directory. **Example** **`plugin.json`** **file** JSON `{ "name" : "my-dev-tools" , "description" : "React development utilities" , "version" : "1.2.0" , "author" : { "name" : "Jane Doe" , "email" : "jane@example.com" } , "license" : "MIT" , "keywords" : [ "react" , "frontend" ] , "agents" : "agents/" , "skills" : [ "skills/" , "extra-skills/" ] , "hooks" : "hooks.json" , "mcpServers" : ".mcp.json" }` For details of the full set of fields you can include in this file, see [GitHub Copilot CLI plugin reference](/en/copilot/reference/cli-plugin-reference#pluginjson) .
3. Add some components to your plugin by creating the appropriate files and directories for agents, skills, hooks, and MCP server configurations. For example:
    1. Add an agent by creating a `NAME.agent.md` file in an `agents` subdirectory. Markdown `--- name: my-agent description: Helps with specific tasks tools: ["bash", "edit", "view"] --- You are a specialized assistant that...`
    2. Add a skill by creating a `skills/NAME` subdirectory of your plugin directory, where `NAME` is the name of your skill. Then, within this subdirectory, create a `SKILL.md` file that defines the skill. For example, to create a "deploy" skill, create `skills/deploy/SKILL.md` : Markdown `--- name: deploy description: Deploy the current project to... --- Instructions for the skill...`
4. Install your plugin locally, so that you can test it as you develop it. For example, where `./my-plugin` is the path to your plugin directory, enter: Shell `copilot plugin install ./my-plugin`
5. Verify that the plugin loaded successfully by viewing your list of installed plugins: Shell `copilot plugin list` Or you can start a new interactive session and enter: Copilot prompt `/plugin list`
6. Verify that the agents, skills, hooks, and MCP server configurations you defined are loaded correctly. For example, in an interactive session, to check that custom agents defined in the plugin were loaded, enter: Copilot prompt `/agent` To check that skills defined in the plugin were loaded, enter: Copilot prompt `/skills list`
7. Use the functionality provided by your plugin's components to verify that each component works as expected.
8. Iterate on your plugin development, as required. Important When you install a plugin its components are cached and the CLI reads from the cache for subsequent sessions. To pick up changes made to a local plugin install it again: Shell `copilot plugin install ./my-plugin`
9. After you have finished testing, you can uninstall the local version of your plugin by entering: Shell `copilot plugin uninstall NAME` Note To uninstall a plugin, use the name of the plugin as specified in the `name` field of the plugin's `plugin.json` manifest file, not the path to the plugin's directory.

### Distributing your plugin

To distribute your plugin, you can add it to a marketplace. See [Creating a plugin marketplace for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/plugins-marketplace) .

### Further reading

- [Finding and installing plugins for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/plugins-finding-installing)
- [GitHub Copilot CLI plugin reference](/en/copilot/reference/cli-plugin-reference)


### Introduction

Plugin marketplaces are registries of plugins for Copilot CLI. They can be located on GitHub.com, in any other online Git hosting service, or on your local or shared file system. By creating a marketplace and adding your plugins to it, you can make it easy for other users to find and install your plugins.

Note

You can find help on using plugins by entering `copilot plugin [SUBCOMMAND] --help` in the terminal.

### Prerequisite

You have created one or more plugins that you want to share. See [Creating a plugin for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/plugins-creating) .

### Creating a plugin marketplace

1. Create a `marketplace.json` file that provides metadata about your marketplace and lists the plugins that are available in the marketplace. Note The `marketplace.json` file is the only required component of a plugin marketplace. Adding it to a repository allows Copilot CLI to recognize the repository as a plugin marketplace, and provides an easy way for users to install plugins. **Example** **`marketplace.json`** **file** JSON `{ "name" : "my-marketplace" , "owner" : { "name" : "Your Organization" , "email" : "plugins@example.com" } , "metadata" : { "description" : "Curated plugins for our team" , "version" : "1.0.0" } , "plugins" : [ { "name" : "frontend-design" , "description" : "Create a professional-looking GUI ..." , "version" : "2.1.0" , "source" : "./plugins/frontend-design" } , { "name" : "security-checks" , "description" : "Check for potential security vulnerabilities ..." , "version" : "1.3.0" , "source" : "./plugins/security-checks" } ] }` Online examples: The top-level `plugins` field is an array of plugin objects, each containing metadata about a plugin, including its name, description, version, and source. The value of the `source` field for each plugin is the path to the plugin's directory, relative to the root of the repository. It is not necessary to use `./` at the start of the path. For example, `"./plugins/plugin-name"` and `"plugins/plugin-name"` resolve to the same directory. For details of the full set of fields you can include in this file, see [GitHub Copilot CLI plugin reference](/en/copilot/reference/cli-plugin-reference#marketplacejson) .
    - [marketplace.json](https://github.com/github/copilot-plugins/blob/main/.github/plugin/marketplace.json) in the [github/copilot-plugins](https://github.com/github/copilot-plugins) repository.
    - [marketplace.json](https://github.com/github/awesome-copilot/blob/main/.github/plugin/marketplace.json) in the [github/awesome-copilot](https://github.com/github/awesome-copilot) repository.
2. Add the `marketplace.json` file to the `.github/plugin` directory of a repository. Note Copilot CLI also looks for the `marketplace.json` file in the `.claude-plugin/` directory.
3. For each plugin defined in the `marketplace.json` file, add the relevant plugin directory to the appropriate location in the repository. For example, if your `marketplace.json` file includes a plugin with `"source": "./plugins/frontend-design"` , add the `frontend-design` plugin directory to the `plugins` directory at the root of your repository.
4. Share the repository with your intended users, and provide them with instructions to add the marketplace to Copilot CLI. For example, if your repository is hosted on GitHub in the `octo-org/octo-repo` repository, instruct users to enter: Shell `copilot plugin marketplace add octo-org/octo-repo`

### Further reading

- [Finding and installing plugins for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/plugins-finding-installing)
- [GitHub Copilot CLI plugin reference](/en/copilot/reference/cli-plugin-reference)


### Connecting to VS Code

Copilot CLI can automatically connect to VS Code when you start a CLI session. Additionally, during an interactive session, you can choose to connect to any workspace that is currently open in VS Code on the local machine.

#### Automatic connection at startup

When you start Copilot CLI, it checks whether the current working directory from which you started the CLI matches any workspace folder you have open in VS Code in trusted mode. If there is a match, the CLI connects to the relevant VS Code instance. The connection happens regardless of where you are using Copilot CLI: in a built-in terminal in VS Code, or in an external terminal application running in a separate window.

If Copilot CLI successfully connects to VS Code, the environment message that's displayed at startup will include either "Visual Studio Code connected" or "Visual Studio Code - Insiders connected."

If you have the same workspace open in more than one VS Code window, the CLI connects to one of them automatically. It cannot connect to multiple IDE instances at the same time. If you prefer to connect to a different instance of VS Code, you can switch by using the `/ide` command.

Note

If you are using GitHub Codespaces, a CLI session running locally cannot connect to a VS Code workspace running in the remote codespace. You can, however, connect when you use the CLI inside the codespace-that is, within VS Code's built-in terminal or in an SSH session on the remote codespace host.

#### Manual connection during an interactive session

If you open a workspace in VS Code after starting Copilot CLI, or if you started the CLI from a directory that doesn't match any open workspace, you can use the `/ide` slash command to manually connect to a VS Code workspace. The workspace you want to connect to must be currently open in trusted mode in VS Code.

### Managing the connection with the /ide slash command

Use the `/ide` slash command in an interactive Copilot CLI session to:

- **View** the current connection status-for example, if you want to check which workspace is currently connected.
- **Connect** to a different VS Code workspace.
- **Disconnect** from VS Code.

You can also toggle the following settings from the `/ide` menu:

- **Auto-connect to matching IDE workspace** -controls whether the CLI automatically connects to a matching VS Code workspace at startup.
- **Open file edit diffs in IDE** -controls whether proposed file changes are shown as diffs in a VS Code editor tab.

### Using VS Code context in prompts

When Copilot CLI is connected to VS Code, it receives your current editor selection whenever the selection changes. The selection is displayed under your prompt in the CLI, aligned to the right. This selection indicator is updated whenever you select different code in VS Code.

This allows you to select some code in VS Code and then use a prompt such as:

```
Debug this
```

Alternatively, you can select some code but ask Copilot about the whole file:

```
Explain this file
```

### Reviewing file changes as diffs

When you ask Copilot to make changes to a file in the workspace, VS Code displays the proposed changes as a diff in a new editor tab. This makes it easy to see exactly what Copilot is proposing. Use the accept (✓) or reject (✗) buttons in the top-right of the diff view to apply or discard the changes. Once you accept or reject the diff, the pending file-edit permission is resolved and the CLI continues its workflow.

Note

- The diff view is not shown if you have allowed Copilot to edit files without your approval-for example, using the `--allow-all` or `--yolo` command-line options, or the `/allow-all` or `/yolo` slash commands. Instead, the proposed changes are applied directly to the file in the workspace without showing a diff, and the CLI continues immediately with the updated file content.
- If you prefer not to use the diff view in VS Code you can turn this feature off in the `/ide` menu. When you turn this off, the proposed file changes are displayed in the CLI.

### Viewing and resuming CLI sessions in VS Code

You can read the transcript of any Copilot CLI session for the current workspace from within VS Code.

1. Open the **Copilot Chat** side bar in VS Code.
2. Click the Sessions icon ( ) at the top right of the Chat panel to display the Sessions view. The Sessions view lists your most recent Copilot sessions, with the most recent at the top.
3. Click a session to read the full input and output text. For CLI sessions, the transcript is identical to what was displayed in the terminal during that session.

If you have run a CLI session for the current workspace that you have not yet viewed in the Sessions view, a dot icon and an unread count are shown next to the Chat icon in the VS Code title bar. Click it to toggle a filtered list of unread sessions. Click it again to clear the filter and view all sessions.


To continue a CLI session in VS Code's integrated terminal, right-click the session in the Sessions view and choose **Resume in Terminal** . This is a quick way to pick up work from an external terminal window without losing any session context.

### Further reading

- [GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli)


### Prerequisite

Install Copilot CLI. See [Installing GitHub Copilot CLI](/en/copilot/how-tos/set-up/install-copilot-cli) .

### Using Copilot CLI

1. In your terminal, navigate to a folder that contains code you want to work with.
2. Enter `copilot` to start Copilot CLI. Copilot will ask you to confirm that you trust the files in this folder. Important During this GitHub Copilot CLI session, Copilot may attempt to read, modify, and execute files in and below this folder. You should only proceed if you trust the files in this location. For more information about trusted directories, see [About GitHub Copilot CLI](/en/copilot/concepts/agents/about-copilot-cli#trusted-directories) .
3. Choose one of the options: **1. Yes, proceed** : Copilot can work with the files in this location for this session only. **2. Yes, and remember this folder for future sessions** : You trust the files in this folder for this and future sessions. You won't be asked again when you start Copilot CLI from this folder. Only choose this option if you are sure that it will always be safe for Copilot to work with files in this location. **3. No, exit (Esc)** : End your Copilot CLI session.
4. If you are not currently logged in to GitHub, you'll be prompted to use the `/login` slash command. Enter this command and follow the on-screen instructions to authenticate.
5. Enter a prompt in the CLI. This can be a simple chat question, or a request for Copilot to perform a specific task, such as fixing a bug, adding a feature to an existing application, or creating a new application. For some examples of prompts, see [About GitHub Copilot CLI](/en/copilot/concepts/agents/about-copilot-cli) .
6. When Copilot wants to use a tool that could modify or execute files-for example, `touch` , `chmod` , `node` , or `sed` -it will ask you to approve the use of the tool. Choose one of the options: **1. Yes** : Allow Copilot to use this tool. The next time Copilot wants to use this tool, it will ask you to approve it again. **2. Yes, and approve TOOL for the rest of the running session** : Allow Copilot to use this tool-with any options-without asking again, for the rest of the currently running session. Any pending parallel permission requests of the same type will be auto-approved. You will have to approve the command again in future sessions. Choosing this option is useful for many tools-such as `chmod` -as it avoids you having to approve similar commands repeatedly in the same session. However, be aware of the security implications of this option. For example, choosing this option for the command `rm` would allow Copilot to delete any file in the current directory or its subdirectories without asking for your approval. **3. No, and tell Copilot what to do differently (Esc)** : Copilot will not run the command. Instead, it ends the current operation and awaits your next prompt. You can tell Copilot to continue the task but using a different approach. For example, if you ask Copilot to create a bash script but you do not want to use the script Copilot suggests, you can stop the current operation and enter a new prompt, such as: `Continue the previous task but include usage instructions in the script` . When you reject a tool permission request, you can also give Copilot inline feedback about the rejection so it can adapt its approach without stopping entirely.

### Tips

Optimize your experience with Copilot CLI with the following tips.

#### Stop a currently running operation

If you enter a prompt and then decide you want to stop Copilot from completing the task while it is still "Thinking," press `Esc` .

#### Use plan mode

Plan mode lets you collaborate with Copilot on an implementation plan before any code is written. Press `Shift` + `Tab` to cycle in and out of plan mode.

#### Include a specific file in your prompt

To add a specific file to your prompt, use `@` followed by the relative path to the file. For example: `Explain @config/ci/ci-required-checks.yml` or `Fix the bug in @src/app.js` . This adds the contents of the file to your prompt as context for Copilot.

When you start typing a file path, the matching paths are displayed below the prompt box. Use the arrow keys to select a path and press `Tab` to complete the path in your prompt.

#### Work with files in a different location

To complete a task, Copilot may need to work with files that are outside the current working directory. If a prompt you have entered in an interactive session requires Copilot to modify a file outside the current location, it will ask you to approve access to the file's directory.

You can also add a trusted directory manually at any time by using the slash command:

```
/add-dir /path/to/directory
```

If all of the files you want to work with are in a different location, you can switch the current working directory without starting a new Copilot CLI session by using either the `/cwd` or `/cd` slash commands:

```
/cwd /path/to/directory
```

#### Run shell commands

You can prepend your input with `!` to directly run shell commands, without making a call to the model.

```
!git clone https://github.com/github/copilot-cli
```

#### Resume an interactive session

You can use the `--resume` command-line option or the `/resume` slash command to select and resume an interactive CLI session, allowing you to pick up right where you left off, with the saved context. You can kick off a Copilot cloud agent session on GitHub, and then use GitHub Copilot CLI to bring that session to your local environment.

Tip

To quickly resume the most recently closed local session, enter this in your terminal:

```
copilot --continue
```

#### Use custom instructions

You can enhance Copilot's performance, by adding custom instructions to the repository you are working in. Custom instructions are natural language descriptions saved in Markdown files in the repository. They are automatically included in prompts you enter while working in that repository. This helps Copilot to better understand the context of your project and how to respond to your prompts.

Copilot CLI supports:

- Repository-wide instructions in the `.github/copilot-instructions.md` file.
- Path-specific instructions files: `.github/instructions/**/*.instructions.md` .
- Agent files such as `AGENTS.md` .

For more information, see [Adding custom instructions for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/add-custom-instructions) .

#### Use custom agents

A custom agent is a specialized version of Copilot. Custom agents help Copilot handle unique workflows, particular coding conventions, and specialist use cases.

Copilot CLI includes a default group of custom agents for common tasks:

| Agent           | Description                                                                                                                                                               |
|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Explore         | Performs quick codebase analysis, allowing you to ask questions about your code without adding to your main context.                                                      |
| Task            | Executes commands such as tests and builds, providing brief summaries on success and full output on failure.                                                              |
| General-purpose | Handles complex, multi-step tasks that require the full toolset and high-quality reasoning, running in a separate context to keep your main conversation clearly focused. |
| Code-review     | Reviews changes with a focus on surfacing only genuine issues, minimizing noise.                                                                                          |

The AI model being used by the CLI can choose to delegate a task to a subsidiary subagent process, that operates using a custom agent with specific expertise, if it judges that this would result in the work being completed more effectively. The model may equally choose to handle the work directly in the main agent.

You can define your own custom agents using Markdown files, called agent profiles, that specify what expertise the agent should have, what tools it can use, and any specific instructions for how it should respond.

You can define custom agents at the user, repository, or organization/enterprise level:

| Type                                           | Location                                                                                 | Scope                                                       |
|------------------------------------------------|------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| User-level custom agent                        | local `~/.copilot/agents` directory                                                      | All projects                                                |
| Repository-level custom agent                  | `.github/agents` directory in your local and remote repositories                         | Current project                                             |
| Organization and Enterprise-level custom agent | `/agents` directory in the `.github-private` repository in an organization or enterprise | All projects under your organization and enterprise account |

In the case of naming conflicts, a system-level agent overrides a repository-level agent, and the repository-level agent would override an organization-level agent.

Custom agents can be used in three ways:

- Using the slash command in the CLI's interactive interface to select from the list of available custom agents: `/agent`
- Calling out to custom agent directly in a prompt: `Use the refactoring agent to refactor this code block` Copilot will automatically infer the agent you want to use.
- Specifying the custom agent you want to use with the command-line option. For example: `copilot --agent=refactor-agent --prompt "Refactor this code block"`

For more information, see [Creating custom agents for Copilot cloud agent](/en/copilot/how-tos/use-copilot-agents/cloud-agent/create-custom-agents) .

#### Use skills

You can create skills to enhance the ability of Copilot to perform specialized tasks with instructions, scripts, and resources.

For more information, see [Creating agent skills for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/create-skills) .

#### Add an MCP server

Copilot CLI comes with the GitHub MCP server already configured. This MCP server allows you to interact with resources on GitHub.com-for example, allowing you to merge pull requests from the CLI.

To extend the functionality available to you in Copilot CLI, you can add more MCP servers:

1. Use the following slash command: `/mcp add`
2. Fill in the details for the MCP server you want to add, using the `Tab` key to move between fields.
3. Press `Ctrl` + `S` to save the details.

Details of your configured MCP servers are stored in the `mcp-config.json` file, which is located, by default, in the `~/.copilot` directory. This location can be changed by setting the `COPILOT_HOME` environment variable. For information about the JSON structure of a server definition, see [Extending GitHub Copilot cloud agent with the Model Context Protocol (MCP)](/en/copilot/how-tos/use-copilot-agents/cloud-agent/extend-cloud-agent-with-mcp#writing-a-json-configuration-for-mcp-servers) .

#### Context management

Copilot CLI provides several slash commands to help you monitor and manage your context window:

- `/usage` : Lets you view your session statistics, including:
    - The amount of premium requests used in the current session
    - The session duration
    - The total lines of code edited
    - A breakdown of token usage per model
- `/context` : Provides a visual overview of your current token usage
- `/compact` : Manually compresses your conversation history to free up context space

GitHub Copilot CLI automatically compresses your history in the background when your conversation approaches 95% of the token limit, without interrupting your workflow.

#### Enable all permissions

For situations where you trust Copilot to run freely, you can use the `--allow-all` or `--yolo` flags to enable all permissions at once.

#### Toggle reasoning visibility

Press `Ctrl` + `T` to show or hide the model's reasoning process while it generates a response. This setting persists across sessions, allowing you to observe how Copilot works through complex problems.

### Find out more

For a complete list of the command line options and slash commands that you can use with Copilot CLI, do one of the following:

- Enter `?` in the prompt box in an interactive session.
- Enter `copilot help` in your terminal.

For additional information use one of the following commands in your terminal:

- **Configuration settings** : `copilot help config` You can adjust the configuration settings by editing the `config.json` file, which is located, by default, in the `~/.copilot` directory. This location can be changed by setting the `COPILOT_HOME` environment variable.
- **Environment variables** that affect Copilot CLI: `copilot help environment`
- **Available logging levels** : `copilot help logging`
- **Permissions** for allowing or denying tool use: `copilot help permissions`

### Feedback

If you have any feedback about GitHub Copilot CLI, please let us know by using the `/feedback` slash command in an interactive session and choosing one of the options. You can complete a private feedback survey, submit a bug report, or suggest a new feature.

### Next steps

Copilot CLI can operate as a conversational assistant, answering questions and helping you write code interactively. Beyond chat, Copilot CLI supports a range of agentic modes that allow you to delegate tasks with greater autonomy.

You can work with agents in Copilot CLI to support a full task lifecycle, from delegating work to reviewing results:

- **Delegate tasks autonomously** : Run Copilot CLI in autopilot mode to complete multi-step tasks without requiring approval at each step. See [Delegating tasks to GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/use-copilot-cli-agents/delegate-tasks-to-cca) .
- **Invoke custom agents** : Invoke specialized agents tailored to specific tasks, such as code review, documentation, or security audits. See [Invoking custom agents](/en/copilot/how-tos/copilot-cli/use-copilot-cli-agents/invoke-custom-agents) .
- **Steer agents** : Guide and refine agent behavior during task execution to keep work on track. See [Steering agents in GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/use-copilot-cli-agents/steer-agents) .
- **Request a code review** : Use Copilot CLI to get an AI-powered review of your code changes. See [Requesting a code review with GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/use-copilot-cli-agents/agentic-code-review) .

### Further reading

- [Best practices for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/cli-best-practices)
- [GitHub Copilot CLI command reference](/en/copilot/reference/copilot-cli-reference/cli-command-reference)
- [GitHub Copilot CLI configuration directory](/en/copilot/reference/copilot-cli-reference/cli-config-dir-reference)
- [Copilot CLI ACP server](/en/copilot/reference/copilot-cli-reference/acp-server)


### Get Copilot to work autonomously

You can tell Copilot to use its best judgment to complete a task autonomously, rather than the CLI prompting you for input at each decision point within a task. You do this by using the CLI's autopilot mode.

There are two ways to use autopilot mode:

- **Interactively:** In an interactive session, press `Shift` + `Tab` until you see "autopilot" in the status bar. If prompted to choose permissions for autopilot mode, allow full permissions, then enter your prompt.
- **Programmatically:** Pass the CLI a prompt directly in a command, and include the `--autopilot` option. For example, to use autopilot mode with full permissions, restricting it to 10 continuations, enter `copilot --autopilot --yolo --max-autopilot-continues 10 -p "YOUR PROMPT HERE"` .

For more information, see [Allowing GitHub Copilot CLI to work autonomously](/en/copilot/concepts/agents/copilot-cli/autopilot) .

### Delegate tasks to Copilot cloud agent

The delegate command lets you push your current session to Copilot cloud agent on GitHub. This lets you hand off work while preserving all the context Copilot needs to complete your task.

You can delegate a task using the slash command, followed by a prompt:

```
/delegate complete the API integration tests and fix any failing edge cases
```

Alternatively, prefix a prompt with `&` to delegate it:

```
& complete the API integration tests and fix any failing edge cases
```

Copilot will ask to commit any of your unstaged changes as a checkpoint in a new branch it creates. Copilot cloud agent will open a draft pull request, make changes in the background, and request a review from you.

Copilot will provide a link to the pull request and agent session on GitHub once the session begins.

### Next steps

To learn how to invoke specialized agents tailored to specific tasks, such as code review, documentation, or security audits, see [Invoking custom agents](/en/copilot/how-tos/copilot-cli/use-copilot-cli-agents/invoke-custom-agents) .


### Use custom agents

A custom agent is a specialized version of Copilot. Custom agents help Copilot handle unique workflows, particular coding conventions, and specialist use cases.

Copilot CLI includes a default group of custom agents for common tasks:

| Agent           | Description                                                                                                                                                               |
|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Explore         | Performs quick codebase analysis, allowing you to ask questions about your code without adding to your main context.                                                      |
| Task            | Executes commands such as tests and builds, providing brief summaries on success and full output on failure.                                                              |
| General-purpose | Handles complex, multi-step tasks that require the full toolset and high-quality reasoning, running in a separate context to keep your main conversation clearly focused. |
| Code-review     | Reviews changes with a focus on surfacing only genuine issues, minimizing noise.                                                                                          |

The AI model being used by the CLI can choose to delegate a task to a subsidiary subagent process, that operates using a custom agent with specific expertise, if it judges that this would result in the work being completed more effectively. The model may equally choose to handle the work directly in the main agent.

You can define your own custom agents using Markdown files, called agent profiles, that specify what expertise the agent should have, what tools it can use, and any specific instructions for how it should respond.

You can define custom agents at the user, repository, or organization/enterprise level:

| Type                                           | Location                                                                                 | Scope                                                       |
|------------------------------------------------|------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| User-level custom agent                        | local `~/.copilot/agents` directory                                                      | All projects                                                |
| Repository-level custom agent                  | `.github/agents` directory in your local and remote repositories                         | Current project                                             |
| Organization and Enterprise-level custom agent | `/agents` directory in the `.github-private` repository in an organization or enterprise | All projects under your organization and enterprise account |

In the case of naming conflicts, a system-level agent overrides a repository-level agent, and the repository-level agent would override an organization-level agent.

Custom agents can be used in three ways:

- Using the slash command in the CLI's interactive interface to select from the list of available custom agents: `/agent`
- Calling out to the custom agent directly in a prompt: `Use the refactoring agent to refactor this code block` Copilot will automatically infer the agent you want to use.
- Specifying the custom agent you want to use with the command-line option. For example: `copilot --agent=refactor-agent --prompt "Refactor this code block"`

For more information, see [Creating custom agents for Copilot cloud agent](/en/copilot/how-tos/use-copilot-agents/cloud-agent/create-custom-agents) .

### Use skills

You can create skills to enhance the ability of Copilot to perform specialized tasks with instructions, scripts, and resources.

For more information, see [Creating agent skills for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/create-skills) .

### Add an MCP server

Copilot CLI comes with the GitHub MCP server already configured. This MCP server allows you to interact with resources on GitHub.com-for example, allowing you to merge pull requests from the CLI.

To extend the functionality available to you in Copilot CLI, you can add more MCP servers:

1. Use the following slash command: `/mcp add`
2. Fill in the details for the MCP server you want to add, using the `Tab` key to move between fields.
3. Press `Ctrl` + `S` to save the details.

Details of your configured MCP servers are stored in the `mcp-config.json` file, which is located, by default, in the `~/.copilot` directory. This location can be changed by setting the `COPILOT_HOME` environment variable. For information about the JSON structure of a server definition, see [Extending GitHub Copilot cloud agent with the Model Context Protocol (MCP)](/en/copilot/how-tos/use-copilot-agents/cloud-agent/extend-cloud-agent-with-mcp#writing-a-json-configuration-for-mcp-servers) .

For more detailed information on adding and managing MCP servers in Copilot CLI, see [Adding MCP servers for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/add-mcp-servers) .

### Next steps

To learn how to guide and refine agent behavior during task execution to keep work on track, see [Steering agents in GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/use-copilot-cli-agents/steer-agents) .


### Steer the conversation while Copilot is thinking

You can interact with Copilot while it's thinking. Send follow-up messages to steer the conversation in a different direction, or queue additional instructions for Copilot to process after it finishes its current response.

Steering lets you:

- Interrupt an agent that is heading in the wrong direction.
- Provide inline feedback when rejecting a tool permission request.
- Refine or clarify the task scope partway through execution.

### Next steps

To learn how to use Copilot CLI to get an AI-powered review of your code changes, see [Requesting a code review with GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/use-copilot-cli-agents/agentic-code-review) .


### About agentic code review

You can use the `/review` slash command to have Copilot analyze code changes without leaving the CLI. This lets you get quick feedback on your changes prior to committing.

1. Type `/review` and optionally specify a prompt, path, or file pattern to narrow the review scope, then press `Enter` .
2. If Copilot proposes running a command (for example, to inspect a diff or verify a file), review the command, then use the arrow keys to choose an option and press `Enter` .
    - Select **Yes** to run the command.
    - Select **No** to skip the command and tell Copilot what to do differently.
3. Read the feedback that Copilot provides about your changes and apply any suggested improvements in your code editor.

### Further reading

- [Automating tasks with Copilot CLI and GitHub Actions](/en/copilot/how-tos/copilot-cli/automate-with-actions)
- [Adding custom instructions for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/add-custom-instructions)


### Using the /fleet slash command

To use the `/fleet` slash command, enter the command followed by your prompt.

#### Typical workflow

Typically, you'll use the `/fleet` slash command after creating an implementation plan.

1. In an interactive CLI session, press `Shift` + `Tab` to switch into plan mode.
2. Enter a prompt describing the feature you want to add or the change you want to make.
3. Work with Copilot in plan mode to create an implementation plan.
4. Once the plan is complete, select one of the following options:
    - **Accept plan and build on autopilot + /fleet** to allow Copilot to use subagents and work autonomously to implement the plan without any further input.
    - **Exit plan mode and I will prompt myself** and then enter a prompt such as `/fleet implement the plan` . Copilot will start working on the plan, using subagents to run parts of the work in parallel where possible. It may ask you to answer questions or make decisions as it works through the plan.

#### Monitoring progress

Use the `/tasks` slash command to see a list of background tasks relating to the current session. This will include any subtasks handled by subagents when you use the `/fleet` command.

Use up and down keyboard keys to navigate through the list of background tasks. For each subagent task, you can:

- Press `Enter` to view details. When the subtask is complete, you will see a summary of what was done.
- Press `k` to kill the process.
- Press `r` to remove completed or killed subtasks from the list.

Press `Esc` to exit the task list and return to the main CLI prompt.

### Further reading

- [GitHub Copilot CLI command reference](/en/copilot/reference/copilot-cli-reference/cli-command-reference#slash-commands-in-the-interactive-interface)


### Resuming a previous session

You can resume a previous interactive CLI session either from the command line, or during an active session.

- **From the command line** , run `copilot --continue` to resume your most recent session. If you want to choose from a list of recent sessions, run `copilot --resume` to open a session picker that lists your recent sessions. Alternatively, if you know the session ID of the session you want to resume, you can run `copilot --resume SESSION-ID` to jump straight into it.
- **During an interactive session** , type `/resume` to switch to a different session. A picker is displayed showing your recent sessions. Alternatively, you can enter `/resume SESSION-ID` to jump straight into a specific session.

Note

You can find the ID of a current interactive session by using the `/session` slash command. The session ID is also displayed when you exit an interactive session.

When you resume a session, Copilot loads the full conversation history, so you can continue exactly where you left off.

### Renaming a session

When you use the `--resume` command line option, or the `/resume` slash command, your recent sessions are listed. The final column of the list shows the session name, which helps you to identify the session you want to resume. If you have a session you return to frequently you might want to give it a custom name to make it easier to find in the list.

To remame a session:

1. In an interactive session, if you want to rename a session other than the current session use the `/resume` slash command to switch to the session you want to rename.
2. Type `/rename NEW_NAME` to rename the current session. You do not need to enclose the name in quotes. For example, `/rename Improve test coverage` .

### Sharing a session

You can save the content of the current session as either a Markdown file or a private gist on GitHub.com. This allows you to share your prompts and Copilot's responses with others, or store a record of your work outside of the CLI.

To share a session as a gist, type the following in an interactive session:

Copilot prompt

```
/share gist
```

To export the session conversation as a Markdown file, type:

Copilot prompt

```
/share file [PATH-TO-FILE]
```

If you don't specify a file path, the Markdown file is saved in the current working directory with the name `copilot-session-SESSIONID.md` .

### Using the /chronicle slash command

Note

The `/chronicle` command, and Copilot's ability to answer questions about your session history, are currently experimental features and are only available if you have used the `/experimental on` slash command, or the `--experimental` command line option.

The `/chronicle` slash command provides a set of subcommands that generate specific types of insights from your session history. While you can ask Copilot free-form questions about your sessions at any time, `/chronicle` subcommands provide a quick way to get specific insights.

When you type `/chronicle` without arguments, a picker is displayed that lets you choose from the available subcommands:

| Subcommand   | Description                                                    |
|--------------|----------------------------------------------------------------|
| `standup`    | Generate a standup report from your recent work.               |
| `tips`       | Get personalized tips based on your usage patterns.            |
| `improve`    | Suggest improvements to your Copilot custom instructions file. |
| `reindex`    | Rebuild the session store index from your session history.     |

You can also invoke a subcommand directly, without using the picker-for example, `/chronicle standup` .

#### /chronicle standup

This generates a short report based on your Copilot CLI sessions, by default from the last 24 hours. Copilot looks at which branches you worked on, what you accomplished, and any GitHub pull requests or issues you referenced. It groups the output by completion status, with each item labeled by its branch, and checks the current status of any linked pull requests.

##### Example standup summary

```
Standup for March 13 2026:

✅ Done

myapp-repo repo maintenance (main branch)

 - Synced local, cleaned files, audited deps, reviewed architecture
 - Session: 69a027e4-9b7b-493e-922e-107acd25abab

🚧 In Progress

MyApp configuration (suppress-start-message branch, myapp-repo)

 - Suppressing startup init prompt message
 - Session: 3034d813-3e1f-413a-b3d9-15427ef8c19c
```

You can append additional context to the command to customize the output. For example, you can tell Copilot to use a different time period, rather than the default last 24 hours:

Copilot prompt

```
/chronicle standup for the last 3 days
```

#### /chronicle tips

This analyzes your recent sessions to understand how you work and how you use Copilot CLI. It then provides 3-5 personalized recommendations. Copilot examines your actual prompts, the tools you use, and the features you haven't tried yet. It cross-references this with the full set of available CLI features-including any custom agents and skills you've set up in the repository-to find opportunities you might be missing.

Tips are grounded in your real usage data, giving you specific suggestions rather than generic advice.

##### Example tips

The following is an example of the main points from a `/chronicle tips` response. In an actual response, each point is explained in more detail.

```
1. Use @ to mention files instead of pasting content
2. Iterate within a session - don't start over
3. Try /research for your exploration work
4. Turn recurring prompts into a custom agent
5. Use plan mode for multi-step work
```

You can focus the tips on a specific area by appending context after `/chronicle tips` . For example:

Copilot prompt

```
/chronicle tips for better prompting
```

#### /chronicle improve

This does a deep dive into your session history to find places where Copilot struggled to provide the kind of response or results you were looking for, or where you had to course-correct by providing follow-up prompts. On the basis of this research, it suggests improvements to your `.github/copilot-instructions.md` custom instructions file.

Capturing project-specific knowledge as custom instructions is a powerful way to improve Copilot's performance when working on your project. For more information, see [Adding custom instructions for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/add-custom-instructions) .

Note

Unlike other times Copilot uses your session data to answer questions or generate insights, the scope of the `improve` subcommand is limited to data for the current repository or working directory. This ensures the recommendations are relevant to the project you're currently working on.

Copilot looks for friction signals-repeated test failures, build errors that required multiple attempts, user messages that corrected or redirected the agent, and patterns that recur across sessions. It then presents 3-5 specific recommendations, each explaining the problem it found and the instruction that would address it.

For example, Copilot might find that it repeatedly tried to use `jest` for your project that uses `vitest` , or that it kept generating imports in a style that doesn't match your codebase conventions. The suggested instructions would prevent these mistakes in future sessions.

After presenting its recommendations, Copilot asks which ones you'd like to apply. By default all recommendations are selected but you can use the arrow keys on your keyboard to move to any of the recommendations then press the space bar to toggle the suggestion off. After choosing which recommendations to apply, press `Enter` . Copilot then creates or updates the `.github/copilot-instructions.md` file.

### Asking questions about your session history

You don't need to use a slash command to take advantage of your session history. If Copilot determines that you are asking about your use of the CLI it will automatically use the session store to provide the context for a response.

Note

By default, the answers to questions about your interactions with Copilot CLI are based on all of your recorded sessions, irrespective of the repository or branch you are currently working in.

Here are some examples of the kinds of questions you might ask:

#### Insights about tasks

Copilot prompt

```
Using what you know about my sessions, what type of tasks give me one-shot successes and which do I have to iterate on most?
```

Copilot will analyze your conversations, looking for times when an initial response was not followed by related prompts, and times when there was a series of iterative prompts and responses.

#### Reduce premium request usage

Copilot prompt

```
Based on my previous CLI sessions, how could I prompt you in a way that would cost less?
```

Copilot will look at your session patterns-prompt length, number of continuation steps, and tool call frequency-and suggest ways to achieve the same results with fewer interactions.

#### Find your most productive times

Copilot prompt

```
Look at data for previous sessions. What time of day am I most and least effective at getting good results from Copilot?
```

Copilot will query session timestamps and outcomes to identify when your interactions tend to be most efficient.

#### Recall past work

Copilot prompt

```
Have I worked on anything related to authentication in the last month?
```

Copilot uses full-text search across your session history to find relevant sessions, then summarizes what you did.

### Further reading

- [About GitHub Copilot CLI session data](/en/copilot/concepts/agents/copilot-cli/chronicle)
- [GitHub Copilot CLI command reference](/en/copilot/reference/copilot-cli-reference/cli-command-reference)


---

# Cloud Agent


### Overview of Copilot cloud agent (formerly Copilot coding agent)

With Copilot cloud agent, GitHub Copilot can work independently in the background to complete tasks, just like a human developer.

Copilot cloud agent can:

- Research a repository
- Create implementation plans
- Fix bugs
- Implement incremental new features
- Improve test coverage
- Update documentation
- Address technical debt
- Resolve merge conflicts

When you delegate tasks to Copilot cloud agent, you can:

- Use the agents panel or other agents entry points on GitHub.com to have Copilot research, plan, and make code changes on a branch, then iterate before creating a pull request. You can also specify in your prompt that you want a pull request created right away. See [Research, plan, and iterate on code changes with Copilot cloud agent](/en/copilot/how-tos/use-copilot-agents/cloud-agent/research-plan-iterate) .
- Ask Copilot to open a new pull request from other entry points, including GitHub Issues and Visual Studio Code. See [Asking GitHub Copilot to create a pull request](/en/copilot/how-tos/use-copilot-agents/cloud-agent/create-a-pr) .
- Mention `@copilot` in a comment on an existing pull request to ask it to make changes. See [Asking GitHub Copilot to make changes to an existing pull request](/en/copilot/how-tos/use-copilot-agents/cloud-agent/make-changes-to-an-existing-pr) .
- Assign security alerts to Copilot from security campaigns. See [Fixing alerts in a security campaign](/en/code-security/code-scanning/managing-code-scanning-alerts/fixing-alerts-in-security-campaign#assigning-alerts-to-copilot-cloud-agent) .

Copilot cloud agent will evaluate the task it has been assigned based on the prompt you give it.

While working on a coding task, Copilot cloud agent has access to its own ephemeral development environment, powered by GitHub Actions, where it can explore your code, make changes, execute automated tests and linters and more.

Note

Deep research, planning, and iterating on code changes before creating a pull request are only available with Copilot cloud agent on GitHub.com. Cloud agent integrations (such as Azure Boards, JIRA, Linear, Slack, or Teams) only support creating a pull request directly.

#### Benefits over traditional AI workflows

When used effectively, Copilot cloud agent offers productivity benefits over traditional AI assistants in IDEs:

- With **AI assistants in IDEs** , coding happens **locally** . Individual developers pair in **synchronous** sessions with the AI assistant. Decisions made during the session are **untracked** and lost to time unless committed. Although the assistant helps write code, the developer still has a lot of **manual steps** to do: create the branch, write commit messages, push the changes, open the PR, write the PR description, get a review, iterate in the IDE, and repeat. These steps take time and effort that may be hard to justify for simple or routine issues.
- With **Copilot cloud agent** , all coding and iterating happens **on GitHub** . You can ask Copilot to **research** a repository, **create a plan** , and **make code changes** on a branch-all before opening a pull request. You can create multiple custom agents that specialize in different types of tasks. Copilot **automates** branch creation, commit message writing, and pushing. Developers let the agents **work in the background** and then chooses to **create a pull request** when ready. Working on GitHub adds **transparency** , with every step happening in a commit and being viewable in logs, and opens up **collaboration** opportunities for the entire team.

### Copilot cloud agent versus agent mode

Copilot cloud agent is distinct from the "agent mode" feature available in your IDE. Copilot cloud agent works autonomously in a GitHub Actions-powered environment to complete development tasks assigned through GitHub issues or GitHub Copilot Chat prompts. It can research a repository, create a plan, make code changes on a branch, and optionally open a pull request. In contrast, agent mode in your IDE makes autonomous edits directly in your local development environment. For more information about agent mode, see [Asking GitHub Copilot questions in your IDE](/en/copilot/using-github-copilot/copilot-chat/asking-github-copilot-questions-in-your-ide) .

### Streamlining software development with Copilot cloud agent

Assigning tasks to Copilot cloud agent can enhance your software development workflow.

For example, you can assign Copilot cloud agent to straightforward issues on your backlog by selecting "Copilot" as the assignee. This allows you to spend less time on these issues and more time on more complex or interesting work, or work that requires a high degree of creative thinking. Copilot cloud agent can work on "nice to have" issues that improve the quality of your codebase or product, but often remain on the backlog while you focus on more urgent work.

Having Copilot cloud agent as an additional coding resource also allows you to start tasks that you might not have otherwise started due to lack of resources. For example, you might create issues to refactor code or add more logging, and then immediately assign these to Copilot.

You can also use Copilot cloud agent to research a repository and create a plan before any code is written, helping you understand how a codebase works or agree on an approach before committing to changes. See [Research, plan, and iterate on code changes with Copilot cloud agent](/en/copilot/how-tos/use-copilot-agents/cloud-agent/research-plan-iterate) .

Copilot cloud agent can start a task, which you then pick up and continue working on yourself. By assigning the initial work to Copilot, you free up time that you would otherwise have spent doing repetitive tasks, such as setting up the scaffolding for a new project.

You can create specialized custom agents for different tasks. For example, you might create a custom agent specialized for frontend development that focuses on React components and styling, a documentation agent that excels at writing and updating technical documentation, or a testing agent that specializes in generating comprehensive unit tests. Each custom agent can be tailored with specific prompts and tools suited to its particular task.

### Measuring pull request outcomes for Copilot cloud agent

Enterprise administrators and organization owners can use Copilot usage metrics to analyze pull request outcomes for pull requests created by Copilot cloud agent.

The Copilot usage metrics APIs include pull request lifecycle metrics such as:

- The total number of pull requests created and merged
- The number of pull requests created by Copilot cloud agent that have been merged
- Median time to merge for merged pull requests, including pull requests created by Copilot cloud agent

These metrics can help you track adoption of Copilot cloud agent and monitor changes in pull request throughput and time to merge over time. See [GitHub Copilot usage metrics](/en/copilot/concepts/copilot-usage-metrics/copilot-metrics) .

### Integrating Copilot cloud agent with third-party tools

You can also invoke Copilot cloud agent from external tools, allowing you to assign tasks to Copilot, provide context, and open pull requests without leaving your workflow. See [About Copilot integrations](/en/copilot/concepts/tools/about-copilot-integrations)

### Making Copilot cloud agent available

Before you can assign tasks to Copilot cloud agent, it must be enabled.

Copilot cloud agent is available with the GitHub Copilot Pro, GitHub Copilot Pro+, GitHub Copilot Business and GitHub Copilot Enterprise plans.

If you are a GitHub Copilot Business or GitHub Copilot Enterprise subscriber, an administrator must enable the relevant policy before you can use the agent.

Repository owners can choose to opt out some or all repositories from Copilot cloud agent.

For more information, see [Managing access to GitHub Copilot cloud agent](/en/copilot/concepts/agents/cloud-agent/access-management) .

### AI models for Copilot cloud agent

Depending on how you start your Copilot cloud agent task, you may be able to select the model used by Copilot cloud agent. You may find that different models perform better, or provide more useful responses, depending on the type of tasks you give Copilot.

For more information, see [Changing the AI model for GitHub Copilot cloud agent](/en/copilot/how-tos/use-copilot-agents/cloud-agent/changing-the-ai-model) .

### Enhancing Copilot cloud agent's knowledge of a repository

The more Copilot cloud agent knows about the code in your repository, the tools you use, and your coding standards and practices, the more effective it will become. There are two ways you can enhance Copilot cloud agent's knowledge of a repository.

- **Custom instructions** These are short, natural-language statements that you write and store as one or more files in a repository. If you are the owner of an organization on GitHub you can also define custom instructions in the settings for your organization. For more information, see [About customizing GitHub Copilot responses](/en/copilot/concepts/prompting/response-customization?tool=webui#about-repository-custom-instructions) .
- **Copilot Memory** (public preview) If you have a Copilot Pro or Copilot Pro+ plan, you can enable Copilot Memory. This allows Copilot to store useful details it has worked out for itself about a repository. Copilot cloud agent can then use this information when it is working in that repository. For more information, see [About agentic memory for GitHub Copilot](/en/copilot/concepts/agents/copilot-memory) .

### Copilot cloud agent usage costs

Copilot cloud agent uses GitHub Actions minutes and Copilot premium requests.

Within your monthly usage allowance for GitHub Actions and premium requests, you can ask Copilot cloud agent to work on coding tasks without incurring any additional costs.

For more information, see [GitHub Copilot licenses](/en/billing/managing-billing-for-your-products/managing-billing-for-github-copilot/about-billing-for-github-copilot#allowance-usage-for-copilot-cloud-agent) .

### Customizing Copilot cloud agent

You can customize Copilot cloud agent in a number of ways:

- **Custom instructions** : Custom instructions allow you to give Copilot additional context on your project and how to build, test and validate its changes. For more information, see [Adding repository custom instructions for GitHub Copilot](/en/copilot/how-tos/configure-custom-instructions/add-repository-instructions) .
- **Model Context Protocol (MCP) servers** : MCP servers allow you to give Copilot access to different data sources and tools. For more information, see [Extending GitHub Copilot cloud agent with the Model Context Protocol (MCP)](/en/copilot/how-tos/use-copilot-agents/cloud-agent/extend-cloud-agent-with-mcp) .
- **Custom agents** : Custom agents allow you to create different specialized versions of Copilot for different tasks. For example, you could customize Copilot to be an expert frontend engineer following your team's guidelines. For more information, see [About custom agents](/en/copilot/concepts/agents/cloud-agent/about-custom-agents) .
- **Hooks** : Hooks allow you to execute custom shell commands at key points during agent execution, enabling you to add validation, logging, security scanning, or workflow automation. For more information, see [About hooks](/en/copilot/concepts/agents/cloud-agent/about-hooks) .
- **Skills** : Skills allow you to enhance the ability of Copilot to perform specialized tasks with instructions, scripts, and resources. For more information, see [About agent skills](/en/copilot/concepts/agents/about-agent-skills) .

### Limitations of Copilot cloud agent

Copilot cloud agent has certain limitations in its software development workflow and compatibility with other features.

#### Limitations in Copilot cloud agent's software development workflow

- **Copilot can only make changes in the repository specified when you start a task** . Copilot cannot make changes across multiple repositories in one run.
- **By default, Copilot can only access context in the repository specified when you start a task** . The Copilot MCP server is configured by default to allow Copilot to access context (for example issues and historic pull requests) in the repository where it is working. You can, however, configure broader access. See [Extending GitHub Copilot cloud agent with the Model Context Protocol (MCP)](/en/copilot/how-tos/use-copilot-agents/cloud-agent/extend-cloud-agent-with-mcp) .
- **Copilot can only work on one branch at a time** and can open exactly one pull request to address each task it is assigned.

#### Limitations in Copilot cloud agent's compatibility with other features

- **Copilot isn't able to comply with certain rules that may be configured for your repository** . If you have configured a ruleset or branch protection rule that isn't compatible with Copilot cloud agent, access to the agent will be blocked. For example, a rule that only allows specific commit authors can prevent Copilot cloud agent from creating or updating pull requests. If the rule is configured using rulesets, you can add Copilot as a bypass actor to enable access. See [Creating rulesets for a repository](/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository#granting-bypass-permissions-for-your-branch-or-tag-ruleset) .
- **Copilot cloud agent doesn't account for content exclusions** . Content exclusions allow administrators to configure Copilot to ignore certain files. When using Copilot cloud agent, Copilot will not ignore these files, and will be able to see and update them. See [Excluding content from GitHub Copilot](/en/copilot/managing-copilot/configuring-and-auditing-content-exclusion/excluding-content-from-github-copilot) .
- **Copilot cloud agent only works with repositories hosted on GitHub** . If your repository is stored using a different code hosting platform, Copilot won't be able to work on it.

### Hands-on practice

Try the [Expand your team with Copilot cloud agent](https://github.com/skills/expand-your-team-with-copilot/?ref_product=copilot&ref_type=engagement&ref_style=text) Skills exercise for practical experience with Copilot cloud agent.

### Further reading

- [GitHub Copilot cloud agent](/en/copilot/how-tos/use-copilot-agents/cloud-agent) how-to articles
- [About custom agents](/en/copilot/concepts/agents/cloud-agent/about-custom-agents)
- [Responsible use of GitHub Copilot cloud agent on GitHub.com](/en/copilot/responsible-use/copilot-cloud-agent)


### About agents

AI agents are autonomous systems that can evaluate their environment, make decisions, and take actions to complete tasks. Agents can break down complex tasks into steps, use various tools and resources, plan their approach, and adapt based on human feedback until they accomplish their assigned objective.

Agents bring automation and assistance to every stage of the software development process on GitHub. You can run multiple agent sessions concurrently, allowing you to efficiently delegate work items.

Alongside Copilot, you can use Anthropic Claude and OpenAI Codex, giving you more flexibility and choice to find the right agent for a task. See [About third-party agents](/en/copilot/concepts/agents/about-third-party-agents) .

Utilizing custom agents you can build out a team of task-specific agents with customized system prompts to handle simpler tasks like writing tests and refactoring, giving you bandwidth to prioritize problem-solving and collaboration. See [About custom agents](/en/copilot/concepts/agents/cloud-agent/about-custom-agents) .

Model choice allows you to choose from a selection of AI models to use with your agents, each with its own particular strengths. See [Supported AI models in GitHub Copilot](/en/copilot/reference/ai-models/supported-models) .

To learn more about Copilot cloud agent, see [About GitHub Copilot cloud agent](/en/copilot/concepts/agents/cloud-agent/about-cloud-agent) .

### Managing agents

When utilizing GitHub's agentic features, you can use the **Agents** tab within a repository that has Copilot cloud agent enabled to initiate, monitor, and manage agent sessions without leaving your workflow. You can also use the [Agents page](https://github.com/copilot/agents?ref_product=copilot&ref_type=engagement&ref_style=text) to view and start agent sessions. To learn how to enable Copilot cloud agent, see [Managing access to GitHub Copilot cloud agent](/en/copilot/concepts/agents/cloud-agent/access-management) .

From the Agents tab, you can:

- **Kick off new agent tasks** : Select an AI model of your choice, and optionally choose from third-party agents or custom agents best suited for the task. See [Asking GitHub Copilot to create a pull request](/en/copilot/how-tos/use-copilot-agents/cloud-agent/create-a-pr) .
- **Monitor live session logs** : Once the agent starts working, you can click any agent session to open the session log and follow its progress and thought process in real time.
- **Track active sessions** : You can view all active agent sessions that have been started in the repository.
- **Steer agents mid-session** : If you realize you didn't scope a request correctly, or want the agent to use a specific tool or service, you can step in and provide **steering input** without stopping the run. Steering uses **one premium request** per message. See [Tracking GitHub Copilot's sessions](/en/copilot/how-tos/use-copilot-agents/cloud-agent/track-copilot-sessions#steering-a-copilot-session-from-the-agents-tab) .
- **Open a session in VS Code or GitHub Copilot CLI** : When you want to start working on changes to an agent session in your local development environment, click **Open in VS Code** or **Continue in GitHub Copilot CLI** to bring the session to your local machine. Note Opening a session in VS Code requires the latest versions of VS Code, the GitHub Copilot extension, and the GitHub Pull Requests extension.
- **Review and merge agent code** : Once the agent completes a session, you can jump to the pull request to review the changes, request further improvements, or approve and merge. See [Reviewing a pull request created by GitHub Copilot](/en/copilot/how-tos/use-copilot-agents/cloud-agent/review-copilot-prs) .

### Next steps

To start managing agents, see [Managing cloud agents](/en/copilot/how-tos/use-copilot-agents/manage-agents) .


### About custom agents

Custom agents are specialized versions of the Copilot agent that you can tailor to your unique workflows, coding conventions, and use cases. They act like tailored teammates that follow your standards, use the right tools, and implement team-specific practices. You define these agents once instead of repeatedly providing the same instructions and context.

You define custom agents using Markdown files called agent profiles. These files specify prompts, tools, and MCP servers. This allows you to encode your conventions, frameworks, and desired outcomes directly into Copilot.

The agent profile defines the custom agent's behavior. When you assign the agent to a task or issue, it instantiates the custom agent.

### Agent profile format

Agent profiles are Markdown files with YAML frontmatter. In their simplest form, they include:

- **Name** (optional): A display name for the custom agent. If omitted, the agent's filename is used as its identifier and default display name.
- **Description** : Explains the agent's purpose and capabilities.
- **Prompt** : Custom instructions that define the agent's behavior and expertise.
- **Tools** (optional): Specific tools the agent can access. By default, agents can access all available tools, including built-in tools, and MCP server tools.

Agent profiles can also include MCP server configurations using the `mcp-servers` property.

#### Example agent profile

This example is a basic agent profile with name, description, and prompt configured.

```
---
name: readme-creator
description: Agent specializing in creating and improving README files

You are a documentation specialist focused on README files. Your scope is limited to README files or other related documentation files only - do not modify or analyze code files.

Focus on the following instructions:
- Create and update README.md files with clear project descriptions
- Structure README sections logically: overview, installation, usage, contributing
- Write scannable content with proper headings and formatting
- Add appropriate badges, links, and navigation elements
- Use relative links (e.g., `docs/CONTRIBUTING.md`) instead of absolute URLs for files within the repository
- Make links descriptive and add alt text to images
```

### Where you can configure custom agents

You can define agent profiles at different levels:

- **Repository level** : Create `.github/agents/CUSTOM-AGENT-NAME.md` in your repository for project-specific agents.
- **Organization or enterprise level** : Create `/agents/CUSTOM-AGENT-NAME.md` in a `.github-private` repository for broader availability.

For more information, see [Preparing to use custom agents in your organization](/en/copilot/how-tos/administer-copilot/manage-for-organization/prepare-for-custom-agents) and [Preparing to use custom agents in your enterprise](/en/copilot/how-tos/administer-copilot/manage-for-enterprise/manage-agents/prepare-for-custom-agents) .

### Where you can use custom agents

Note

Custom agents are in public preview for JetBrains IDEs, Eclipse, and Xcode, and subject to change.

Once you create custom agents, they become available to:

- **Copilot cloud agent on GitHub.com** : The agents tab and panel, issue assignment, and pull requests
- **Copilot cloud agent in IDEs** : Visual Studio Code, JetBrains IDEs, Eclipse, and Xcode
- **GitHub Copilot CLI**

You can use agent profiles directly in Visual Studio Code, JetBrains IDEs, Eclipse, and Xcode. Some properties may function differently or be ignored between environments.

For more information on using custom agents in Visual Studio Code, see [Custom agents in VS Code](https://code.visualstudio.com/docs/copilot/customization/custom-agents) .

### Next steps

To create your own custom agents, see:

- [Creating custom agents for Copilot cloud agent](/en/copilot/how-tos/use-copilot-agents/cloud-agent/create-custom-agents)
- [Creating and using custom agents for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/create-custom-agents-for-cli)
- [Copilot customization cheat sheet](/en/copilot/reference/customization-cheat-sheet)


### About hooks

Hooks enable you to execute custom shell commands at strategic points in an agent's workflow, such as when an agent session starts or ends, or before and after a prompt is entered or a tool is called.

Hooks receive detailed information about agent actions via JSON input, enabling context-aware automation. For example, you can use hooks to:

- Programmatically approve or deny tool executions.
- Utilize built-in security features like secret scanning to prevent credential leaks.
- Implement custom validation rules and audit logging for compliance.

Copilot agents support hooks stored in JSON files in your repository at `.github/hooks/*.json` .

Hooks are available for use with:

- Copilot cloud agent on GitHub
- GitHub Copilot CLI in the terminal

### Types of hooks

The following types of hooks are available:

- **sessionStart** : Executed when a new agent session begins or when resuming an existing session. Can be used to initialize environments, log session starts for auditing, validate project state, and set up temporary resources.
- **sessionEnd** : Executed when the agent session completes or is terminated. Can be used to cleanup temporary resources, generate and archive session reports and logs, or send notifications about session completion.
- **userPromptSubmitted** : Executed when the user submits a prompt to the agent. Can be used to log user requests for auditing and usage analysis.
- **preToolUse** : Executed before the agent uses any tool (such as `bash` , `edit` , `view` ). This is the most powerful hook as it can **approve or deny tool executions** . Use this hook to block dangerous commands, enforce security policies and coding standards, require approval for sensitive operations, or log tool usage for compliance.
- **postToolUse** : Executed after a tool completes execution (whether successful or failed). Can be used to log execution results, track usage statistics, generate audit trails, monitor performance metrics, and send failure alerts.
- **agentStop** : Executed when the main agent has finished responding to your prompt.
- **subagentStop** : Executed when a subagent completes, before returning results to the parent agent.
- **errorOccurred** : Executed when an error occurs during agent execution. Can be used to log errors for debugging, send notifications, track error patterns, and generate reports.

To see a complete reference of hook types with example use cases, best practices, and advanced patterns, see [Hooks configuration](/en/copilot/reference/hooks-configuration) .

### Hook configuration format

You configure hooks using a special JSON format. The JSON must contain a `version` field with a value of `1` and a `hooks` object containing arrays of hook definitions.

JSON

```
{ "version" : 1 , "hooks" : { "sessionStart" : [ { "type" : "command" , "bash" : "string (optional)" , "powershell" : "string (optional)" , "cwd" : "string (optional)" , "env" : { "KEY" : "value" } , "timeoutSec" : 30 } ] , }
}
```

The hook object can contain the following keys:

| Property     | Required              | Description                                                                    |
|--------------|-----------------------|--------------------------------------------------------------------------------|
| `type`       | Yes                   | Must be `"command"`                                                            |
| `bash`       | Yes (on Unix systems) | Path to the bash script to execute                                             |
| `powershell` | Yes (on Windows)      | Path to the PowerShell script to execute                                       |
| `cwd`        | No                    | Working directory for the script (relative to repository root)                 |
| `env`        | No                    | Additional environment variables that are merged with the existing environment |
| `timeoutSec` | No                    | Maximum execution time in seconds (default: 30)                                |

### Example hook configuration file

This is an example configuration file that lives in `~/.github/hooks/project-hooks.json` within a repository.

JSON

```
{ "version" : 1 , "hooks" : { "sessionStart" : [ { "type" : "command" , "bash" : "echo \"Session started: $(date)\" >> logs/session.log" , "powershell" : "Add-Content -Path logs/session.log -Value \"Session started: $(Get-Date)\"" , "cwd" : "." , "timeoutSec" : 10 } ] , "userPromptSubmitted" : [ { "type" : "command" , "bash" : "./scripts/log-prompt.sh" , "powershell" : "./scripts/log-prompt.ps1" , "cwd" : "scripts" , "env" : { "LOG_LEVEL" : "INFO" } } ] , "preToolUse" : [ { "type" : "command" , "bash" : "./scripts/security-check.sh" , "powershell" : "./scripts/security-check.ps1" , "cwd" : "scripts" , "timeoutSec" : 15 } , { "type" : "command" , "bash" : "./scripts/log-tool-use.sh" , "powershell" : "./scripts/log-tool-use.ps1" , "cwd" : "scripts" } ] , "postToolUse" : [ { "type" : "command" , "bash" : "cat >> logs/tool-results.jsonl" , "powershell" : "$input | Add-Content -Path logs/tool-results.jsonl" } ] , "sessionEnd" : [ { "type" : "command" , "bash" : "./scripts/cleanup.sh" , "powershell" : "./scripts/cleanup.ps1" , "cwd" : "scripts" , "timeoutSec" : 60 } ] }
}
```

### Performance considerations

Hooks run synchronously and block agent execution. To ensure a responsive experience, keep the following considerations in mind:

- **Minimize execution time** : Keep hook execution time under 5 seconds when possible.
- **Optimize logging** : Use asynchronous logging, like appending to files, rather than synchronous I/O.
- **Use background processing** : For expensive operations, consider background processing.
- **Cache results** : Cache expensive computations when possible.

### Security considerations

To ensure security is maintained when using hooks, keep the following considerations in mind:

- **Always validate and sanitize the input processed by hooks** . Untrusted input could lead to unexpected behavior.
- **Use proper shell escaping when constructing commands** . This prevents command injection vulnerabilities.
- **Never log sensitive data, such as tokens or passwords** .
- **Ensure hook scripts and logs have the appropriate permissions** .
- **Be cautious with hooks that make external network calls** . These can introduce latency, failures, or expose data to third parties.
- **Set appropriate timeouts to prevent resource exhaustion** . Long-running hooks can block agent execution and degrade performance.

### Next steps

To start creating hooks, see [Using hooks with GitHub Copilot agents](/en/copilot/how-tos/use-copilot-agents/cloud-agent/use-hooks) .


### Overview

Copilot cloud agent is an AI-powered software development agent that can work autonomously on issues or developer requests. It raises draft pull requests to propose a fix and iterates on the changes in response to feedback.

If you are a GitHub Copilot Enterprise or GitHub Copilot Business subscriber, Copilot cloud agent is disabled by default and must be enabled by an administrator before it is available for use.

If you are a GitHub Copilot Pro or Pro+ subscriber, Copilot cloud agent is enabled by default.

Once enabled, you can use Copilot cloud agent in any repository, provided that an administrator hasn't opted the repository out.

### Copilot cloud agent policies for Copilot Business and Copilot Enterprise

For GitHub Copilot Business and GitHub Copilot Enterprise subscribers, the ability to use Copilot cloud agent is controlled by policy settings defined at the organization level. See [Adding GitHub Copilot cloud agent to your organization](/en/copilot/how-tos/administer-copilot/manage-for-organization/add-copilot-cloud-agent) .

If the organization is owned by an enterprise, enablement may be controlled at the enterprise level. See [Managing GitHub Copilot cloud agent in your enterprise](/en/enterprise-cloud@latest/copilot/how-tos/administer-copilot/manage-for-enterprise/manage-agents/manage-copilot-cloud-agent) .

### Opting repositories out of Copilot cloud agent

By default, users with Copilot cloud agent enabled can use it in all repositories.

Enterprise administrators and organization owners (for organization-owned repositories) and users (for user-owned repositories) can opt out repositories and prevent Copilot cloud agent from being used in those repositories.

For information on disabling Copilot cloud agent in some or all repositories owned by an organization, see [Adding GitHub Copilot cloud agent to your organization](/en/copilot/how-tos/administer-copilot/manage-for-organization/add-copilot-cloud-agent) .

For information on disabling Copilot cloud agent in all repositories owned by an enterprise, see [Managing GitHub Copilot cloud agent in your enterprise](/en/enterprise-cloud@latest/copilot/how-tos/administer-copilot/manage-for-enterprise/manage-agents/manage-copilot-cloud-agent) .

For information on disabling Copilot cloud agent in repositories owned by your personal user account, see [Managing GitHub Copilot policies as an individual subscriber](/en/copilot/managing-copilot/managing-copilot-as-an-individual-subscriber/managing-your-copilot-plan/managing-copilot-policies-as-an-individual-subscriber#enabling-or-disabling-copilot-cloud-agent) .

### Further reading

- [GitHub Copilot cloud agent](/en/copilot/how-tos/use-copilot-agents/cloud-agent)
- [Customizing the development environment for GitHub Copilot cloud agent](/en/copilot/how-tos/use-copilot-agents/cloud-agent/customize-the-agent-environment)
- [Customizing or disabling the firewall for GitHub Copilot cloud agent](/en/copilot/how-tos/use-copilot-agents/cloud-agent/customize-the-agent-firewall)
- [Extending GitHub Copilot cloud agent with the Model Context Protocol (MCP)](/en/copilot/how-tos/use-copilot-agents/cloud-agent/extend-cloud-agent-with-mcp)
- [Piloting GitHub Copilot cloud agent in your organization](/en/copilot/tutorials/cloud-agent/pilot-cloud-agent)


### Overview

The Model Context Protocol (MCP) is an open standard that defines how applications share context with large language models (LLMs). MCP provides a standardized way to connect AI models to different data sources and tools, enabling them to work together more effectively.

You can use MCP to extend the capabilities of Copilot cloud agent by connecting it to other tools and services.

The agent can use tools provided by local and remote MCP servers. Some MCP servers are configured by default to provide the best experience for getting started.

For more information on MCP, see [the official MCP documentation](https://modelcontextprotocol.io/introduction) . For information on some of the currently available MCP servers, see [the MCP servers repository](https://github.com/modelcontextprotocol/servers/tree/main) .

Note

- Copilot cloud agent only supports tools provided by MCP servers. It does not support resources or prompts.
- Copilot cloud agent does not currently support remote MCP servers that leverage OAuth for authentication and authorization.

### Default MCP servers

The following MCP servers are configured automatically for Copilot cloud agent:

- **GitHub** : The GitHub MCP server gives Copilot access to GitHub data like issues and pull requests. To learn more, see [Using the GitHub MCP Server in your IDE](/en/copilot/customizing-copilot/using-model-context-protocol/using-the-github-mcp-server) .
    - By default, the GitHub MCP server connects to GitHub using a specially scoped token that only has read-only access to the current repository. You can customize it to use a different token with broader access. For more details, see [Extending GitHub Copilot cloud agent with the Model Context Protocol (MCP)](/en/copilot/how-tos/use-copilot-agents/cloud-agent/extend-cloud-agent-with-mcp#customizing-the-built-in-github-mcp-server) .
- **Playwright** : The [Playwright MCP server](https://github.com/microsoft/playwright-mcp) gives Copilot access to web pages, including the ability to read, interact and take screenshots.
    - By default, the Playwright MCP server is only able to access web resources hosted within Copilot's own environment, accessible on `localhost` or `127.0.0.1` .

### Setting up MCP servers in a repository

Repository administrators can configure MCP servers for use within that repository. This is done via a JSON-formatted configuration that specifies the details of the MCP servers that Copilot cloud agent can use.

Once MCP servers are configured for use within a repository, the tools specified in the configuration will be available to Copilot cloud agent during each assigned task.

Copilot will use available tools autonomously, and will not ask for approval before use.

For details of how to set up MCP servers for Copilot cloud agent in a repository, see [Extending GitHub Copilot cloud agent with the Model Context Protocol (MCP)](/en/copilot/how-tos/use-copilot-agents/cloud-agent/extend-cloud-agent-with-mcp) .

### MCP servers for custom agents

You can also configure MCP servers for custom agents.

MCP servers configured in custom agents are available only to that specific agent and follow the same processing order as other MCP configurations, with custom agent MCP settings processed after default servers but before repository-level configurations.

For more information on configuring MCP servers for custom agents, see [Custom agents configuration](/en/copilot/reference/custom-agents-configuration#mcp-server-configuration-details) .

### Best practices

- Enabling third-party MCP servers for use may impact the performance of the agent and the quality of the outputs. Review the third-party MCP server thoroughly and ensure that it meets your organization's requirements.
- By default, Copilot cloud agent does not have access to write MCP server tools. However, some MCP servers do contain such tools. Be sure to review the tools available in the MCP server you want to use. Update the `tools` field in the MCP configuration with only the necessary tooling.
- Carefully review the configured MCP servers prior to saving the configuration to ensure the correct servers are configured for use.


### Unvalidated code can introduce vulnerabilities

By default, Copilot cloud agent checks code it generates for security issues and gets a second opinion on its code with Copilot code review. It attempts to resolve issues identified prior to completing the pull request. This improves code quality and reduces the likelihood of the code generated by Copilot cloud agent introducing problems such as hardcoded secrets, insecure dependencies, and other vulnerabilities. Copilot cloud agent's security validation **does not require** a GitHub Secret Protection, GitHub Code Security, or GitHub Advanced Security license.

- **CodeQL** is used to identify code security issues.
- Newly introduced dependencies are checked against the **GitHub Advisory Database** for malware advisories, and for any CVSS-rated High or Critical vulnerabilities.
- **Secret scanning** is used to detect sensitive information such as API keys, tokens, and other secrets.
- Details about the analysis performed and the actions taken by Copilot cloud agent can be reviewed in the session log. See [Tracking GitHub Copilot's sessions](/en/copilot/how-tos/use-copilot-agents/cloud-agent/track-copilot-sessions) .

Optionally, you can disable one or more of the code quality and security validation tools used by Copilot cloud agent. See [Configuring settings for GitHub Copilot cloud agent](/en/copilot/how-tos/use-copilot-agents/cloud-agent/configuring-agent-settings) .

### Copilot cloud agent can push code changes to your repository

To mitigate this risk, GitHub:

- **Limits who can trigger the agent.** Only users with write access to the repository can trigger Copilot cloud agent to work. Comments from users without write access are never presented to the agent.
- **Limits the branch the agent can push to.** Copilot cloud agent only has the ability to push to a single branch. When the agent is triggered by mentioning `@copilot` on an existing pull request, Copilot has write access to the pull request's branch. In other cases, a new `copilot/` branch is created for Copilot, and the agent can only push to that branch. The agent is also subject to any branch protections and required checks for the working repository.
- **Limits the agent's credentials.** Copilot cloud agent can only perform simple push operations. It cannot directly run `git push` or other Git commands.
- **Requires human review before merging.** Draft pull requests created by Copilot cloud agent must be reviewed and merged by a human. Copilot cloud agent cannot mark its pull requests as "Ready for review" and cannot approve or merge a pull request.
- **Restricts GitHub Actions workflow runs.** By default, workflows are not triggered until Copilot cloud agent's code is reviewed and a user with write access to the repository clicks the **Approve and run workflows** button. Optionally, you can configure Copilot to allow workflows to run automatically. See [Reviewing a pull request created by GitHub Copilot](/en/copilot/how-tos/use-copilot-agents/cloud-agent/review-copilot-prs#managing-github-actions-workflow-runs) .
- **Prevents the user who asked Copilot cloud agent to create a pull request from approving it.** This maintains the expected controls in the "Required approvals" rule and branch protection. See [Available rules for rulesets](/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets) .

### Copilot cloud agent has access to sensitive information

Copilot cloud agent has access to code and other sensitive information, and could leak it, either accidentally or due to malicious user input.

To mitigate this risk, GitHub **restricts Copilot cloud agent's access to the internet** . See [Customizing or disabling the firewall for GitHub Copilot cloud agent](/en/copilot/how-tos/use-copilot-agents/cloud-agent/customize-the-agent-firewall) .

### AI prompts can be vulnerable to injection

Users can include hidden messages in issues assigned to Copilot cloud agent or comments left for Copilot cloud agent as a form of [prompt injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) .

To mitigate this risk, GitHub **filters hidden characters before passing user input to Copilot cloud agent** : For example, text entered as an HTML comment in an issue or pull request comment is not passed to Copilot cloud agent.

### Administrators can lose sight of agents' work

To mitigate this risk, Copilot cloud agent is designed to be auditable and traceable.

- Copilot cloud agent's commits are authored by Copilot, with the developer who assigned the issue or requested the change to the pull request marked as the co-author. This makes it easier to identify code generated by Copilot cloud agent and who started the task.
- Copilot cloud agent's commits are signed, so they appear as "Verified" on GitHub. This provides confidence that the commits were made by Copilot cloud agent and have not been altered.
- Session logs and audit log events are available to administrators.
- The commit message for each agent-authored commit includes a link to the agent session logs, for code review and auditing. See [Tracking GitHub Copilot's sessions](/en/copilot/how-tos/use-copilot-agents/cloud-agent/track-copilot-sessions) .


---

# Chat & Code Suggestions


### Ask general software questions

You can ask Copilot Chat general software questions. For example:

- tell me about nodejs web server frameworks
- how can I create an Express app
- @terminal how to update an npm package

### Ask questions about your project

You can ask Copilot Chat questions about your project.

- what sorting algorithm does this function use
- @workspace how are notifications scheduled
- #file:gameReducer.js #file:gameInit.js how are these files related

To give Copilot the correct context, try some of these strategies:

- Highlight relevant lines of code.
- Use chat variables like `#selection` , `#file` , `#editor` , `#codebase` , or `#git` .
- Use the `@workspace` chat participant.

### Write code

You can ask Copilot to write code for you. For example:

- write a function to sum all numbers in a list
- add error handling to this function
- @workspace add form validation, similar to the newsletter page

When Copilot returns a code block, the response includes options to copy the code, or to insert the code at your cursor, into a new file, or into the terminal.

### Ask questions about alerts from GitHub Advanced Security features

You can ask Copilot about security alerts in repositories in your organization from GitHub Advanced Security features (code scanning, secret scanning, and Dependabot alerts). For example:

- How would I fix this alert?
- How many alerts do I have on this pull request?
- Which line of code is this code scanning alert referencing?
- What library is affected by this Dependabot alert?

### Set up a new project

Use the `/new` slash command to set up a new project. For example:

- /new react app with typescript
- /new python django web application
- /new node.js express server

Copilot will suggest a directory structure and provide a button to create the suggested files and contents. To preview a suggested file, select the file name in the suggested directory structure.

Use the `/newNotebook` slash command to set up a new Jupyter notebook. For example:

- /newNotebook retrieve the titanic dataset and use Seaborn to plot the data

### Fix, improve, and refactor code

If your active file contains an error, use the `/fix` slash command to ask Copilot to fix the error.

You can also make general requests to improve or refactor your code.

- how would you improve this code?
- translate this code to C#
- add error handling to this function

### Write tests

Use the `/tests` slash command to ask Copilot to write tests for the active file or selected code. For example:

- /tests
- /tests using the Jest framework
- /tests ensure the function rejects an empty list

The `/tests` slash command writes tests for existing code. If you prefer to write tests before writing code (test driven development), omit the `/tests` command. For example:

- Add tests for a JavaScript function that should sum a list of integers

### Ask questions about Visual Studio Code

Use the `@vscode` chat participant to ask specific questions about Visual Studio Code. For example:

- @vscode tell me how to debug a node.js app
- @vscode how do I change my Visual Studio Code colors
- @vscode how can I change key bindings

### Ask questions about the command line

Use the `@terminal` chat participant to ask specific questions about the command line. For example:

- @terminal find the largest file in the src directory
- `@terminal #terminalLastCommand` to explain the last command and any errors

### Ask general software questions

You can ask Copilot Chat general software questions. For example:

- tell me about nodejs web server frameworks
- how can I create an Express app
- what's the process for updating an npm package

### Ask questions about your project

You can ask Copilot Chat questions about your project. To give Copilot the correct context, try some of these strategies:

- Highlight relevant lines of code.
- Open the relevant file.
- Use `#file` to tell Copilot to reference specific files.
- Use `#solution` to tell Copilot to reference the active file.

For example:

- what sorting algorithm does this function use
- #file:gameReducer.js what happens when a new game is requested

### Write code

You can ask Copilot to write code for you. For example:

- write a function to sum all numbers in a list
- add error handling to this function

When Copilot returns a code block, the response includes options to copy the code, insert the code into a new file, or preview the code output.

### Ask questions about alerts from GitHub Advanced Security features

You can ask Copilot about security alerts in repositories in your organization from GitHub Advanced Security features (code scanning, secret scanning, and Dependabot alerts). For example:

- How would I fix this alert?
- How many alerts do I have on this pull request?
- Which line of code is this code scanning alert referencing?
- What library is affected by this Dependabot alert?

### Fix, improve, and refactor code

If your active file contains an error, use the `/fix` slash command to ask Copilot to fix the error.

You can also make general requests to improve or refactor your code.

- how would you improve this code?
- translate this code to C#
- add error handling to this function

### Write tests

Use the `/tests` slash command to ask Copilot to write tests for the active file or selected code. For example:

- /tests
- /tests using the Jest framework
- /tests ensure the function rejects an empty list

The `/tests` slash command writes tests for existing code. If you prefer to write tests before writing code (test driven development), omit the `/tests` command. For example:

- Add tests for a JavaScript function that should sum a list of integers

### Ask general software questions

You can ask Copilot Chat general software questions. For example:

- tell me about nodejs web server frameworks
- how can I create an Express app
- what's the process for updating an npm package

### Ask questions about your project

You can ask Copilot Chat questions about your project. To give Copilot the correct context, try some of these strategies:

- Highlight relevant lines of code.
- Open the relevant file.
- Add the file as a reference. For information about how to use file references, see [Asking GitHub Copilot questions in your IDE](/en/copilot/using-github-copilot/asking-github-copilot-questions-in-your-ide?tool=jetbrains#file-references) .
- Use the `@project` chat participant.

For example:

- what sorting algorithm does this function use
- `how are these files related` (with references to the files in question)
- @project how are notifications scheduled

### Write code

You can ask Copilot to write code for you. For example:

- write a function to sum all numbers in a list
- add error handling to this function

When Copilot returns a code block, the response includes options to copy the code or to insert the code at your cursor.

### Fix, improve, and refactor code

If your active file contains an error, use the `/fix` slash command to ask Copilot to fix the error.

You can also make general requests to improve or refactor your code.

- how would you improve this code?
- translate this code to C#
- add error handling to this function

### Write tests

Use the `/tests` slash command to ask Copilot to write tests for the active file or selected code. For example:

- /tests
- /tests using the Jest framework
- /tests ensure the function rejects an empty list

The `/tests` slash command writes tests for existing code. If you prefer to write tests before writing code (test driven development), omit the `/tests` command. For example:

- Add tests for a JavaScript function that should sum a list of integers

### Ask general software questions

You can ask Copilot Chat general software questions. For example:

- tell me about nodejs web server frameworks
- how can I create an Express app
- what's the process for updating an npm package

### Ask questions about files your project

You can ask Copilot Chat questions about the file that's currently displayed in the editor, or about files you have attached to your conversation in the Copilot Chat panel. To give Copilot the correct context:

- Open the relevant file in the editor.
- Click the paperclip icon in the Copilot Chat panel, then search for and select files you want to attach to the conversation.

For example:

- how can I make this file run faster
- `how are these files related` (with two or more attached files)
- explain the getSearchReplaceRules function

### Write code

You can ask Copilot to write code for you. For example:

- write a TypeScript function to sum all numbers in a list
- using the comments in this file, create appropriate Node JavaScript

When Copilot returns a code block, the response includes options to copy the code.

### Fix, improve, and refactor code

If your active file contains an error, use the `/fix` slash command to ask Copilot to fix the error.

You can also make general requests to improve or refactor your code.

- how would you improve the code in this file
- translate this code to C#
- add error handling to the main function

### Write tests

Use the `/tests` slash command to ask Copilot to write tests for the active file or selected code. For example:

- /tests
- /tests using the Jest framework
- /tests ensure the function rejects an empty list

The `/tests` slash command writes tests for existing code. If you prefer to write tests before writing code (test driven development), omit the `/tests` command. For example:

- Add tests for a JavaScript function that should sum a list of integers


### Introduction

This guide describes how to use Copilot Chat and agents to automate coding tasks by breaking them into steps, using tools to read files, edit code, and run commands, and self-correcting when something goes wrong. You can also ask general questions about software development, or specific questions about the code in your project. For more information, see [About GitHub Copilot Chat](/en/copilot/concepts/about-github-copilot-chat) .

### Prerequisites

- **Access to GitHub Copilot** . See [What is GitHub Copilot?](/en/copilot/about-github-copilot/what-is-github-copilot#getting-access-to-copilot) .
- **Latest version of Visual Studio Code** . See the [Visual Studio Code download page](https://code.visualstudio.com/Download?ref_product=copilot&ref_type=engagement&ref_style=text) .
- **Sign in to GitHub in Visual Studio Code** . If you experience authentication issues, see [Troubleshooting common issues with GitHub Copilot](/en/copilot/troubleshooting-github-copilot/troubleshooting-issues-with-github-copilot-chat#troubleshooting-authentication-issues-in-your-editor) .

If you have access to GitHub Copilot via your organization, you won't be able to use GitHub Copilot Chat if your organization owner has disabled chat. See [Managing policies and features for GitHub Copilot in your organization](/en/copilot/managing-github-copilot-in-your-organization/managing-policies-and-features-for-copilot-in-your-organization) .

Note

If you don't see the **Agent** option in the mode selector, your enterprise or organization administrator may have disabled agent mode for your IDE.

### Copilot Chat agents

You can use Copilot Chat in the following modes:

- [Agent mode](#agent-mode) : to get Copilot to autonomously accomplish a set task.
- [Plan mode](#plan-mode) : to get Copilot to create detailed implementation plans to ensure all requirements are met.
- [Ask mode](#ask-mode) : to get answers to coding questions and get Copilot to provide code suggestions.

To switch between modes, use the agents dropdown at the bottom of the chat view.

#### Agent mode

Use agent mode when you have a specific task in mind and want to enable Copilot to autonomously edit your code. In agent mode, Copilot determines which files to make changes to, offers code changes and terminal commands to complete the task, and iterates to remediate issues until the original task is complete.

Agent mode is best suited to use cases where:

- Your task is complex, and involves multiple steps, iterations, and error handling.
- You want Copilot to determine the necessary steps to take to complete the task.
- The task requires Copilot to integrate with external applications, such as an MCP server.

##### Using agents

1. If the chat view is not already displayed, select **Open Chat** from the Copilot Chat menu.
2. At the bottom of the chat view, ensure **Agent** is selected from the agents dropdown.
3. Submit a prompt. In response to your prompt, Copilot streams the edits in the editor, updates the working set, and if necessary, runs terminal commands, if necessary.
4. Review and iterate on changes or run a code review.

You can also [click this link](vscode://GitHub.Copilot-Chat/chat?mode=agent&ref_product=copilot&ref_type=engagement&ref_style=text) to go directly to agent mode in VS Code.

For more information, see [Chat overview](https://aka.ms/vscode-copilot-agent) in the Visual Studio Code documentation.

When you use agent mode, each prompt you enter counts as one premium request, multiplied by the model's multiplier. For example, if you're using the included model-which has a multiplier of 0-your prompts won't consume any premium requests. Copilot may take several follow-up actions to complete your task, but these follow-up actions do **not** count toward your premium request usage. Only the prompts you enter are billed-tool calls or background steps taken by the agent are not charged.

The total number of premium requests you use depends on how many prompts you enter and which model you select. See [Requests in GitHub Copilot](/en/copilot/managing-copilot/managing-copilot-as-an-individual-subscriber/monitoring-usage-and-entitlements/avoiding-unexpected-copilot-costs) .

##### Using subagents

You can use subagents to delegate tasks to an isolated agent with its own context window within your chat session. The subagent operates independently without pausing for user feedback and returns the final result to the main chat session.

Subagents are best suited for situations where:

- You want to delegate complex, multi-step tasks like research or analysis without interrupting your main session.
- You need to process large amounts of information or multiple documents that would clutter your primary context window.
- You want to explore different approaches or perspectives independently without mixing contexts together.

Subagents use the same tools and AI model as the main session, but they cannot create other subagents.

##### Enabling subagents

1. In the Copilot Chat window, click the tools icon.
2. Enable the `runSubagent` tool.

If you use custom prompt files or custom agents, ensure you specify the `runSubagent` tool in the `tools` frontmatter property. See [Creating custom agents for Copilot cloud agent](/en/copilot/how-tos/use-copilot-agents/cloud-agent/create-custom-agents#creating-a-custom-agent-profile-in-visual-studio-code) , and [Use prompt files in VS Code](https://code.visualstudio.com/docs/copilot/customization/prompt-files) in the Visual Studio Code documentation.

##### Invoking subagents

Subagents can be invoked in different ways:

- **Automatic delegation** . Copilot will analyze the description of your request, the description field of your configured custom agents, and the current context and available tools to automatically choose a subagent. For example, this prompt would automatically delegate the task to a **refactor-specialist** custom agent: `Suggest ways to refactor this legacy code.`
- **Direct invocation** . You can directly call the subagent in your prompt: `Use the testing subagent to write unit tests for the authentication module.`
- **Calling the #runSubagent tool.** `Evaluate the #file:databaseSchema using #runSubagent and generate an optimized data-migration plan.`

When the subagent completes its task, its results appear back in the main chat session, ready for follow-up questions or next steps.

#### Plan mode

Plan mode helps you to create detailed implementation plans before executing them. This ensures that all requirements are considered and addressed before any code changes are made. The plan agent does not make any code changes until the plan is reviewed and approved by you. Once approved, you can hand off the plan to the default agent or save it for further refinement, review, or team discussions.

The plan agent is designed to:

- Research the task comprehensively using read-only tools and codebase analysis to identify requirements and constraints.
- Break down the task into manageable, actionable steps and include open questions about ambiguous requirements.
- Present a concise plan draft, based on a standardized plan format, for user review and iteration.

##### Using the plan agent

1. If the chat view is not already displayed, select **Open Chat** from the Copilot Chat menu.
2. At the bottom of the chat view, select **Plan** from the agents dropdown.
3. Type a prompt that describes a task, such as adding a feature to an existing application, refactoring code, fixing a bug, or creating an initial version of a new application. For example: `Create a simple to-do web app with HTML, CSS, and JS files.` After a few moments, the plan agent outputs a plan in the chat view. The plan provides a high-level summary and a breakdown of steps, including any open questions for clarification.
4. Review the plan and answer any questions the agent has asked. You can iterate multiple times to clarify requirements, adjust scope, or answer questions.
5. Once the plan is complete you can:
    - Click **Start Implementation** to switch Copilot Chat to agent mode and start an agent session to implement the required changes, based on the implementation plan.
    - Click **Open in Editor** to switch Copilot Chat to agent mode and start an agent session that generates Markdown, in a tab of your editor, with the details of the implementation plan. You can start to work through the plan yourself, or save the plan as a Markdown file for later use.

For more information, see [Planning with agents in VS Code](https://code.visualstudio.com/docs/copilot/agents/planning) in the Visual Studio Code documentation.

#### Ask mode

Ask mode is optimized for answering questions about your codebase, coding, and general technology concepts. Use ask mode when you want to understand how something works, explore ideas, or get help with coding tasks.

##### Using the ask agent

1. If the chat view is not already displayed, select **Open Chat** from the Copilot Chat menu.
2. At the bottom of the chat view, select **Ask** from the agents dropdown.
3. Type a prompt in the prompt box and press `Enter` .

### Submitting prompts

You can give the agent a high-level description of what you want to build and it gets to work. Each task runs inside an agent session, a persistent conversation you can track, pause, resume, or hand off to another agent.

1. To open the chat view, click the chat icon in the title bar of Visual Studio Code. If the chat icon is not displayed, right-click the title bar and make sure that **Command Center** is selected.


3. Enter a prompt in the prompt box. For an introduction to the kinds of prompts you can use, see [Getting started with prompts for GitHub Copilot Chat in your IDE](/en/copilot/get-started/getting-started-with-prompts-for-copilot-chat) .
4. Evaluate Copilot's response, and make a follow-up request if needed. The response may contain text, code blocks, buttons, images, URIs, and file trees. The response often includes interactive elements. For example, the response may include a menu to insert a code block, or a button to invoke a Visual Studio Code command. To see the files that Copilot Chat used to generate the response, select the **Used** ***n*** **references** dropdown at the top of the response. The references may include a link to a custom instructions file for your repository. This file contains additional information that is automatically added to all of your chat questions to improve the quality of the responses. For more information, see [Adding repository custom instructions for GitHub Copilot](/en/copilot/customizing-copilot/adding-repository-custom-instructions-for-github-copilot) .

### Using keywords in your prompt

You can use special keywords to help Copilot understand your prompt. For examples, see [Getting started with prompts for GitHub Copilot Chat in your IDE](/en/copilot/get-started/getting-started-with-prompts-for-copilot-chat) .

#### Chat participants

Chat participants are like domain experts who have a specialty that they can help you with.

Copilot Chat can infer relevant chat participants based on your natural language prompt, improving discovery of advanced capabilities without you having to explicitly specify the participant you want to use in your prompt.

Note

Automatic inference for chat participants is currently in public preview and is subject to change.

Alternatively, you can manually specify a chat participant to scope your prompt to a specific domain. To do this, type `@` in the chat prompt box, followed by a chat participant name.

For a list of available chat participants, type `@` in the chat prompt box. See also [GitHub Copilot Chat cheat sheet](/en/copilot/using-github-copilot/github-copilot-chat-cheat-sheet?tool=vscode#chat-participants) or [Chat participants](https://code.visualstudio.com/docs/copilot/copilot-chat#_chat-participants) in the Visual Studio Code documentation.

#### Slash commands

Use slash commands to avoid writing complex prompts for common scenarios. To use a slash command, type `/` in the chat prompt box, followed by a command.

To see all available slash commands, type `/` in the chat prompt box. See also [GitHub Copilot Chat cheat sheet](/en/copilot/using-github-copilot/github-copilot-chat-cheat-sheet?tool=vscode#slash-commands) or [Slash commands](https://code.visualstudio.com/docs/copilot/reference/copilot-vscode-features#_slash-commands) in the Visual Studio Code documentation.

#### Chat variables

Use chat variables to include specific context in your prompt. To use a chat variable, type `#` in the chat prompt box, followed by a chat variable.

To see all available chat variables, type `#` in the chat prompt box. See also [GitHub Copilot Chat cheat sheet](/en/copilot/using-github-copilot/github-copilot-chat-cheat-sheet?tool=vscode#chat-variables) .

### Using GitHub skills for Copilot

Copilot's GitHub-specific skills expand the type of information Copilot can provide. To access these skills in Copilot Chat, include `@github` in your question.

When you add `@github` to a question, Copilot dynamically selects an appropriate skill, based on the content of your question. You can also explicitly ask Copilot Chat to use a particular skill. You can do this in two ways:

- Use natural language to ask Copilot Chat to use a skill. For example, `@github Search the web to find the latest GPT model from OpenAI.`
- To specifically invoke a web search you can include the `#web` variable in your question. For example, `@github #web What is the latest LTS of Node.js?`

You can generate a list of currently available skills by asking Copilot: `@github What skills are available?`

### Using Model Context Protocol (MCP) servers

You can use MCP to extend the capabilities of Copilot Chat by integrating it with a wide range of existing tools and services. For additional information, see [About Model Context Protocol (MCP)](/en/copilot/concepts/context/mcp) .

### AI models for Copilot Chat

You can change the model Copilot uses to generate responses to chat prompts. You may find that different models perform better, or provide more useful responses, depending on the type of questions you ask. Options include premium models with advanced capabilities.  See [Changing the AI model for GitHub Copilot Chat](/en/copilot/using-github-copilot/ai-models/changing-the-ai-model-for-copilot-chat) .

### Additional ways to access Copilot Chat

In addition to submitting prompts through the chat view, you can submit prompts in other ways:

- **Quick chat:** To open the quick chat dropdown, enter `Shift` + `Optin` + `Command` + `L` (Mac) / `Ctrl` + `Shift` + `Alt` + `L` (Windows/Linux).
- **Inline:** To start an inline chat directly in the editor or integrated terminal, enter `Command` + `i` (Mac) / `Ctrl` + `i` (Windows/Linux).
- **Smart actions:** To submit prompts via the context menu, right click in your editor, select **Copilot** in the menu that appears, then select one of the actions. Smart actions can also be accessed via the sparkle icon that sometimes appears when you select a line of code.

See [inline chat](https://code.visualstudio.com/docs/copilot/copilot-chat#_inline-chat) , [quick chat](https://code.visualstudio.com/docs/copilot/copilot-chat#_quick-chat) , and [chat smart actions](https://code.visualstudio.com/docs/copilot/copilot-chat#_chat-smart-actions) in the Visual Studio Code documentation for more details.

### Using images in Copilot Chat

Note

- If you're using a Copilot Business or Copilot Enterprise plan, the organization or enterprise that provides your plan must enable the **Editor preview features** setting. See [Managing policies and features for GitHub Copilot in your organization](/en/enterprise-cloud@latest/copilot/managing-copilot/managing-github-copilot-in-your-organization/managing-policies-for-copilot-in-your-organization#enabling-copilot-features-in-your-organization) or [Managing policies and features for GitHub Copilot in your enterprise](/en/copilot/managing-copilot/managing-copilot-for-your-enterprise/managing-policies-and-features-for-copilot-in-your-enterprise#configuring-policies-for-github-copilot) .

You can attach images to your chat prompts and then ask Copilot about the images. For example, you can attach:

- A screenshot of a code snippet and ask Copilot to explain the code.
- A mockup of the user interface for an application and ask Copilot to generate the code.
- A flowchart and ask Copilot to describe the processes shown in the image.
- A screenshot of a web page and ask Copilot to generate HTML for a similar page.

Note

The following types of image file are supported: JPEG ( `.jpg` , `.jpeg` ), PNG ( `.png` ), GIF ( `.gif` ), or WEBP ( `.webp` ).

#### Attaching images to your chat prompt

1. Do one of the following:
    - Copy an image and paste it into the chat view.
    - Drag and drop one or more image file from your operating system's file explorer-or from the Explorer in VS Code-into the chat view.
    - Right-click an image file in the VS Code Explorer and click **Copilot** then **Add File to Chat** .
2. Type your prompt into the chat view to accompany the image. For example, `explain this diagram` , `describe each of these images in detail` , `what does this error message mean` .

### Sharing feedback

To indicate whether a response was helpful, use the thumbs up and thumbs down icons that appear next to the response.

To leave feedback about the GitHub Copilot Chat extension, open an issue in the [microsoft/vscode-copilot-release](https://github.com/microsoft/vscode-copilot-release/issues) repository.

### Further reading

- [Prompt engineering for GitHub Copilot Chat](/en/copilot/using-github-copilot/prompt-engineering-for-github-copilot)
- [Using Copilot Chat in VS Code](https://code.visualstudio.com/docs/copilot/copilot-chat) and [Getting started with GitHub Copilot in VS Code](https://code.visualstudio.com/docs/copilot/getting-started) in the Visual Studio Code documentation
- [Asking GitHub Copilot questions in GitHub](/en/copilot/github-copilot-enterprise/copilot-chat-in-github/using-github-copilot-chat-in-githubcom)
- [Responsible use of GitHub Copilot Chat in your IDE](/en/copilot/github-copilot-chat/about-github-copilot-chat)
- [GitHub Terms for Additional Products and Features](/en/site-policy/github-terms/github-terms-for-additional-products-and-features#github-copilot)
- [GitHub Copilot Trust Center](https://copilot.github.trust.page/)
- [GitHub Copilot FAQ](https://github.com/features/copilot#faq)

### Prerequisites

- **Access to GitHub Copilot** . See [What is GitHub Copilot?](/en/copilot/about-github-copilot/what-is-github-copilot#getting-access-to-copilot) .
- **Visual Studio 2022 version 17.8 or later** . See [Install Visual Studio](https://learn.microsoft.com/visualstudio/install/install-visual-studio) in the Visual Studio documentation. *Visual Studio 17.10 and later have the GitHub Copilot and GitHub Copilot Chat extensions built in. You don't need to install them separately.*
    - *For Visual Studio 17.8 and 17.9:*
        - **GitHub Copilot extension** . See [Install GitHub Copilot in Visual Studio](https://learn.microsoft.com/visualstudio/ide/visual-studio-github-copilot-install-and-states?ref_product=copilot&ref_type=engagement&ref_style=text) in the Visual Studio documentation.
        - **GitHub Copilot Chat extension** . See [Install GitHub Copilot in Visual Studio](https://learn.microsoft.com/visualstudio/ide/visual-studio-github-copilot-install-and-states?ref_product=copilot&ref_type=engagement&ref_style=text) in the Visual Studio documentation.
- **Sign in to GitHub in Visual Studio** . If you experience authentication issues, see [Troubleshooting common issues with GitHub Copilot](/en/copilot/troubleshooting-github-copilot/troubleshooting-issues-with-github-copilot-chat#troubleshooting-authentication-issues-in-your-editor) .

If you have access to GitHub Copilot via your organization, you won't be able to use GitHub Copilot Chat if your organization owner has disabled chat. See [Managing policies and features for GitHub Copilot in your organization](/en/copilot/managing-github-copilot-in-your-organization/managing-policies-and-features-for-copilot-in-your-organization) .

### Submitting prompts

You can ask Copilot Chat to give you code suggestions, explain code, generate unit tests, and suggest code fixes.

1. In the Visual Studio menu bar, click **View** , then click **GitHub Copilot Chat** .
2. In the Copilot Chat window, enter a prompt, then press **Enter** . For example prompts, see [Getting started with prompts for GitHub Copilot Chat in your IDE](/en/copilot/get-started/getting-started-with-prompts-for-copilot-chat) .
3. Evaluate Copilot's response, and submit a follow up prompt if needed. The response often includes interactive elements. For example, the response may include buttons to copy, insert, or preview the result of a code block. To see the files that Copilot Chat used to generate the response, click the **References** link below the response. The references may include a link to a custom instructions file for your repository. This file contains additional information that is automatically added to all of your chat questions to improve the quality of the responses. For more information, see [Adding repository custom instructions for GitHub Copilot](/en/copilot/customizing-copilot/adding-repository-custom-instructions-for-github-copilot) .

### Using keywords in your prompt

You can use special keywords to help Copilot understand your prompt.

#### Slash commands

Use slash commands to avoid writing complex prompts for common scenarios. To use a slash command, type `/` in the chat prompt box, followed by a command.

To see all available slash commands, type `/` in the chat prompt box. See also [GitHub Copilot Chat cheat sheet](/en/copilot/using-github-copilot/github-copilot-chat-cheat-sheet?tool=vscode#slash-commands) or [Slash commands](https://learn.microsoft.com/visualstudio/ide/copilot-chat-context#slash-commands) in the Visual Studio documentation.

#### References

By default, Copilot Chat will reference the file that you have open or the code that you have selected. You can also use `#` followed by a file name, file name and line numbers, or `solution` to reference a specific file, lines, or solution.

See also [GitHub Copilot Chat cheat sheet](/en/copilot/using-github-copilot/github-copilot-chat-cheat-sheet?tool=visualstudio#references) or [Reference](https://learn.microsoft.com/visualstudio/ide/copilot-chat-context#reference) in the Visual Studio documentation.

### Using GitHub skills for Copilot (preview)

Note

The `@github` chat participant is currently in preview, and only available in [Visual Studio 2022 Preview 2](https://visualstudio.microsoft.com/vs/preview/) onwards.

Copilot's GitHub-specific skills expand the type of information Copilot can provide. To access these skills in Copilot Chat in Visual Studio, include `@github` in your question.

When you add `@github` to a question, Copilot dynamically selects an appropriate skill, based on the content of your question. You can also explicitly ask Copilot Chat to use a particular skill. For example, `@github Search the web to find the latest GPT4 model from OpenAI.`

You can generate a list of currently available skills by asking Copilot: `@github What skills are available?`

### Using Model Context Protocol (MCP) servers

You can use MCP to extend the capabilities of Copilot Chat by integrating it with a wide range of existing tools and services. For additional information, see [About Model Context Protocol (MCP)](/en/copilot/concepts/context/mcp) .

### AI models for Copilot Chat

You can change the model Copilot uses to generate responses to chat prompts. You may find that different models perform better, or provide more useful responses, depending on the type of questions you ask. Options include premium models with advanced capabilities.  See [Changing the AI model for GitHub Copilot Chat](/en/copilot/using-github-copilot/ai-models/changing-the-ai-model-for-copilot-chat) .

### Additional ways to access Copilot Chat

In addition to submitting prompts through the chat window, you can submit prompts inline. To start an inline chat, right click in your editor window and select **Ask Copilot** .

See [Ask questions in the inline chat view](https://learn.microsoft.com/visualstudio/ide/visual-studio-github-copilot-chat#ask-questions-in-the-inline-chat-view) in the Visual Studio documentation for more details.

### Copilot Edits

Note

- This feature is currently in public preview and subject to change.
- Available in Visual Studio 17.14 and later.

Copilot Edits lets you make changes across multiple files from a single Copilot Chat prompt

Use agent mode when you have a specific task in mind and want to enable Copilot to autonomously edit your code. In agent mode, Copilot determines which files to make changes to, offers code changes and terminal commands to complete the task, and iterates to remediate issues until the original task is complete.

#### Using agent mode

1. In the Visual Studio menu bar, click **View** , then click **GitHub Copilot Chat** .
2. At the bottom of the chat panel, select **Agent** from the agents dropdown.
3. Submit a prompt. In response to your prompt, Copilot streams the edits in the editor, updates the working set, and if necessary, suggests terminal commands to run.
4. Review the changes. If Copilot suggested terminal commands, confirm whether or not Copilot can run them. In response, Copilot iterates and performs additional actions to complete the task in your original prompt.

When you use Copilot agent mode, each prompt you enter counts as one premium request, multiplied by the model's multiplier. For example, if you're using the included model-which has a multiplier of 0-your prompts won't consume any premium requests. Copilot may take several follow-up actions to complete your task, but these follow-up actions do **not** count toward your premium request usage. Only the prompts you enter are billed-tool calls or background steps taken by the agent are not charged.

### Using images in Copilot Chat

Note

- If you're using a Copilot Business or Copilot Enterprise plan, the organization or enterprise that provides your plan must enable the **Editor preview features** setting. See [Managing policies and features for GitHub Copilot in your organization](/en/enterprise-cloud@latest/copilot/managing-copilot/managing-github-copilot-in-your-organization/managing-policies-for-copilot-in-your-organization#enabling-copilot-features-in-your-organization) or [Managing policies and features for GitHub Copilot in your enterprise](/en/copilot/managing-copilot/managing-copilot-for-your-enterprise/managing-policies-and-features-for-copilot-in-your-enterprise#configuring-policies-for-github-copilot) .

You can attach images to your chat prompts and then ask Copilot about the images. For example, you can attach:

- A screenshot of a code snippet and ask Copilot to explain the code.
- A mockup of the user interface for an application and ask Copilot to generate the code.
- A flowchart and ask Copilot to describe the processes shown in the image.
- A screenshot of a web page and ask Copilot to generate HTML for a similar page.

Note

The following types of image file are supported: JPEG ( `.jpg` , `.jpeg` ), PNG ( `.png` ), GIF ( `.gif` ), or WEBP ( `.webp` ).

#### Attaching images to your chat prompt

1. If you see the AI model picker at the bottom right of the chat view, select one of the models that supports adding images to prompts:
2. Do one of the following: You can add multiple images if required.
    - Copy an image and paste it into the chat view.
    - Click the paperclip icon at the bottom right of the chat view, click **Upload Image** , browse to the image file you want to attach, select it and click **Open** .
3. Type your prompt into the chat view to accompany the image. For example, `explain this image` , or `describe each of these images in detail` .

### Sharing feedback

To share feedback about Copilot Chat, you can use the **Send feedback** button in Visual Studio. For more information on providing feedback for Visual Studio, see the [Visual Studio Feedback](https://learn.microsoft.com/en-us/visualstudio/ide/how-to-report-a-problem-with-visual-studio?view=vs-2022) documentation.

1. In the top right corner of the Visual Studio window, click the **Send feedback** button.


3. Choose the option that best describes your feedback.
    - To report a bug, click **Report a problem** .
    - To request a feature, click **Suggest a feature** .

### Further reading

- [Prompt engineering for GitHub Copilot Chat](/en/copilot/using-github-copilot/prompt-engineering-for-github-copilot)
- [Using GitHub Copilot Chat in Visual Studio in the Microsoft Learn documentation](https://learn.microsoft.com/visualstudio/ide/visual-studio-github-copilot-chat?view=vs-2022#use-copilot-chat-in-visual-studio)
- [Tips to improve GitHub Copilot Chat results in the Microsoft Learn documentation](https://learn.microsoft.com/en-us/visualstudio/ide/copilot-chat-context?view=vs-2022)
- [Asking GitHub Copilot questions in GitHub](/en/copilot/github-copilot-enterprise/copilot-chat-in-github/using-github-copilot-chat-in-githubcom)
- [Responsible use of GitHub Copilot Chat in your IDE](/en/copilot/github-copilot-chat/about-github-copilot-chat)
- [GitHub Terms for Additional Products and Features](/en/site-policy/github-terms/github-terms-for-additional-products-and-features#github-copilot)
- [GitHub Copilot Trust Center](https://copilot.github.trust.page/)
- [GitHub Copilot FAQ](https://github.com/features/copilot#faq)

### Prerequisites

- **Access to GitHub Copilot** . See [What is GitHub Copilot?](/en/copilot/about-github-copilot/what-is-github-copilot#getting-access-to-copilot) .
- **Compatible JetBrains IDE** . GitHub Copilot is compatible with the following IDEs: See the [JetBrains IDEs](https://www.jetbrains.com/products/?ref_product=copilot&ref_type=engagement&ref_style=button) tool finder to download.
    - IntelliJ IDEA (Ultimate, Community, Educational)
    - Android Studio
    - AppCode
    - CLion
    - Code With Me Guest
    - DataGrip
    - DataSpell
    - GoLand
    - JetBrains Client
    - MPS
    - PhpStorm
    - PyCharm (Professional, Community, Educational)
    - Rider
    - RubyMine
    - RustRover
    - WebStorm
    - Writerside
- **Latest version of the GitHub Copilot extension** . See the [GitHub Copilot plugin](https://plugins.jetbrains.com/plugin/17718-github-copilot?ref_product=copilot&ref_type=engagement&ref_style=text) in the JetBrains Marketplace. For installation instructions, see [Installing the GitHub Copilot extension in your environment](/en/copilot/configuring-github-copilot/installing-the-github-copilot-extension-in-your-environment) .
- **Sign in to GitHub in your JetBrains IDE** . For authentication instructions, see [Installing the GitHub Copilot extension in your environment](/en/copilot/configuring-github-copilot/installing-the-github-copilot-extension-in-your-environment?tool=jetbrains#installing-the-github-copilot-plugin-in-your-jetbrains-ide) .

If you have access to GitHub Copilot via your organization, you won't be able to use GitHub Copilot Chat if your organization owner has disabled chat. See [Managing policies and features for GitHub Copilot in your organization](/en/copilot/managing-github-copilot-in-your-organization/managing-policies-and-features-for-copilot-in-your-organization) .

### Submitting prompts

You can ask Copilot Chat to give you code suggestions, explain code, generate unit tests, and suggest code fixes.

1. Open the Copilot Chat window by clicking the **GitHub Copilot Chat** icon at the right side of the JetBrains IDE window.


3. Enter a prompt in the prompt box. For example prompts, see [Getting started with prompts for GitHub Copilot Chat in your IDE](/en/copilot/get-started/getting-started-with-prompts-for-copilot-chat) .
4. Evaluate Copilot's response, and submit a follow up prompt if needed. The response often includes interactive elements. For example, the response may include buttons to copy or insert a code block. To see the files that Copilot Chat used to generate the response, click the **References** link below the response. The references may include a link to a custom instructions file for your repository. This file contains additional information that is automatically added to all of your chat questions to improve the quality of the responses. For more information, see [Adding repository custom instructions for GitHub Copilot](/en/copilot/how-tos/custom-instructions/adding-repository-custom-instructions-for-github-copilot) .

### Supplementing your prompt

You can use slash commands and file references to help Copilot understand your what you are asking it to do.

#### Slash commands

Use slash commands to avoid writing complex prompts for common scenarios. To use a slash command, type `/` in the chat prompt box, followed by a command.

To see all available slash commands, type `/` in the chat prompt box. See also [GitHub Copilot Chat cheat sheet](/en/copilot/using-github-copilot/github-copilot-chat-cheat-sheet?tool=jetbrains#slash-commands-2) .

#### File references

By default, Copilot Chat will reference the file that you have open or the code that you have selected. You can also tell Copilot Chat which files to reference by dragging a file into the chat prompt box. Alternatively, you can right click on a file, select **GitHub Copilot** , then select **Reference File in Chat** .

### Using GitHub skills for Copilot

Copilot's GitHub-specific skills expand the type of information Copilot can provide. To access these skills in Copilot Chat, include `@github` in your question.

When you add `@github` to a question, Copilot dynamically selects an appropriate skill, based on the content of your question. You can also explicitly ask Copilot Chat to use a particular skill. You can do this in two ways:

- Use natural language to ask Copilot Chat to use a skill. For example, `@github Search the web to find the latest GPT model from OpenAI.`
- To specifically invoke a web search you can include the `#web` variable in your question. For example, `@github #web What is the latest LTS of Node.js?`

You can generate a list of currently available skills by asking Copilot: `@github What skills are available?`

### Using Model Context Protocol (MCP) servers

You can use MCP to extend the capabilities of Copilot Chat by integrating it with a wide range of existing tools and services. For additional information, see [About Model Context Protocol (MCP)](/en/copilot/concepts/context/mcp) .

### AI models for Copilot Chat

You can change the model Copilot uses to generate responses to chat prompts. You may find that different models perform better, or provide more useful responses, depending on the type of questions you ask. Options include premium models with advanced capabilities.  See [Changing the AI model for GitHub Copilot Chat](/en/copilot/using-github-copilot/ai-models/changing-the-ai-model-for-copilot-chat) .

### Additional ways to access Copilot Chat

- **Built-in requests** . In addition to submitting prompts through the chat window, you can submit built-in requests by right clicking in a file, selecting **GitHub Copilot** , then selecting one of the options.
- **Inline** . You can submit a chat prompt inline, and scope it to a highlighted code block or your current file.
    - To start an inline chat, right click on a code block or anywhere in your current file, hover over **GitHub Copilot** , then select **Copilot: Inline Chat** , or enter `Ctrl` + `Shift` + `I` .

### Copilot Edits

Use Copilot Edits to make changes across multiple files directly from a single Copilot Chat prompt. Copilot Edits has the following modes:

- [Edit mode](#edit-mode-1) lets Copilot make controlled edits to multiple files.
- [Agent mode](#agent-mode-1) lets Copilot autonomously accomplish a set task.

#### Edit mode

Edit mode is only available in Visual Studio Code and JetBrains IDEs.

Use edit mode when you want more granular control over the edits that Copilot proposes. In edit mode, you choose which files Copilot can make changes to, provide context to Copilot with each iteration, and decide whether or not to accept the suggested edits after each turn.

Edit mode is best suited to use cases where:

- You want to make a quick, specific update to a defined set of files.
- You want full control over the number of LLM requests Copilot uses.

##### Using edit mode

1. To start an edit session, click **Copilot** in the menu bar, then select **Open GitHub Copilot Chat** .
2. At the top of the chat panel, click **Copilot Edits** .
3. Add relevant files to the *working set* to indicate to GitHub Copilot which files you want to work on. You can add all open files by clicking **Add all open files** or individually search for single files.
4. Submit a prompt. In response to your prompt, Copilot Edits determines which files in your *working set* to change and adds a short description of the change.
5. Review the changes and **Accept** or **Discard** the edits for each file.

#### Agent mode

Use agent mode when you have a specific task in mind and want to enable Copilot to autonomously edit your code. In agent mode, Copilot determines which files to make changes to, offers code changes and terminal commands to complete the task, and iterates to remediate issues until the original task is complete.

Agent mode is best suited to use cases where:

- Your task is complex, and involves multiple steps, iterations, and error handling.
- You want Copilot to determine the necessary steps to take to complete the task.
- The task requires Copilot to integrate with external applications, such as an MCP server.

##### Using agent mode

1. To start an edit session using agent mode, click **Copilot** in the menu bar, then select **Open GitHub Copilot Chat** .
2. At the top of the chat panel, click the **Agent** tab.
3. Submit a prompt. In response to your prompt, Copilot streams the edits in the editor, updates the working set, and if necessary, suggests terminal commands to run.
4. Review the changes. If Copilot suggested terminal commands, confirm whether or not Copilot can run them. In response, Copilot iterates and performs additional actions to complete the task in your original prompt.

When you use agent mode, each prompt you enter counts as one premium request, multiplied by the model's multiplier. For example, if you're using the included model-which has a multiplier of 0-your prompts won't consume any premium requests. Copilot may take several follow-up actions to complete your task, but these follow-up actions do **not** count toward your premium request usage. Only the prompts you enter are billed-tool calls or background steps taken by the agent are not charged.

The total number of premium requests you use depends on how many prompts you enter and which model you select. See [Requests in GitHub Copilot](/en/copilot/managing-copilot/managing-copilot-as-an-individual-subscriber/monitoring-usage-and-entitlements/avoiding-unexpected-copilot-costs) .

#### Using subagents

You can use subagents to delegate tasks to an isolated agent with its own context window within your chat session. The subagent operates independently without pausing for user feedback and returns the final result to the main chat session.

Subagents are best suited for situations where:

- You want to delegate complex, multi-step tasks like research or analysis without interrupting your main session.
- You need to process large amounts of information or multiple documents that would clutter your primary context window.
- You want to explore different approaches or perspectives independently without mixing contexts together.

Subagents use the same tools and AI model as the main session, but they cannot create other subagents.

To use subagents, you **must have custom agents configured in your environment** . See [Creating custom agents for Copilot cloud agent](/en/copilot/how-tos/use-copilot-agents/cloud-agent/create-custom-agents) .

##### Enabling subagents

To enable subagents:

1. Click **Tools** in the menu bar, then click **GitHub Copilot** , then **Edit Settings** .
2. In the popup menu, click **Chat** , then click the **Enable Subagent** checkbox.

##### Invoking subagents

Subagents can be invoked in different ways:

- **Automatic delegation** . Copilot will analyze the description of your request, the description field of your configured custom agents, and the current context and available tools to automatically choose a subagent. For example, this prompt would automatically delegate the task to a **refactor-specialist** custom agent: `Suggest ways to refactor this legacy code.`
- **Direct invocation** . You can directly call the subagent in your prompt: `Use the testing subagent to write unit tests for the authentication module.`

When the subagent completes its task, its results appear back in the main chat session, ready for follow-up questions or next steps.

### Using plan mode

Plan mode helps you to create detailed implementation plans before executing them. This ensures that all requirements are considered and addressed before any code changes are made. The plan agent does not make any code changes until the plan is reviewed and approved by you. Once approved, you can hand off the plan to the default agent or save it for further refinement, review, or team discussions.

The plan agent is designed to:

- Research the task comprehensively using read-only tools and codebase analysis to identify requirements and constraints.
- Break down the task into manageable, actionable steps and include open questions about ambiguous requirements.
- Present a concise plan draft, based on a standardized plan format, for user review and iteration.

To use plan mode:

1. If it is not already displayed, open the Copilot Chat panel by clicking the **GitHub Copilot Chat** icon at the right side of the JetBrains IDE window.
2. At the bottom of the Copilot Chat panel, select **Plan** from the agents dropdown.
3. Type a prompt that describes a task, such as adding a feature to an existing application, refactoring code, fixing a bug, or creating an initial version of a new application. For example: `Create a simple to-do web app with HTML, CSS, and JS files.`
4. Submit the prompt. After a few moments, the plan agent outputs a plan in the chat panel. The plan provides a high-level summary and a breakdown of steps, including any open questions for clarification.
5. Review the plan and answer any questions the agent has asked. You can iterate multiple times to clarify requirements, adjust scope, or answer questions.
6. Once the plan is complete you can:
    - Click **Start Implementation** to switch Copilot Chat to agent mode and start an agent session to implement the required changes, based on the implementation plan.
    - Click **Open in Editor** to switch Copilot Chat to agent mode and start an agent session that generates Markdown, in a tab of your editor, with the details of the implementation plan. You can start to work through the plan yourself, or save the plan as a Markdown file for later use.

### Sharing feedback

To share feedback about Copilot Chat, you can use the **share feedback** link in JetBrains.

1. At the right side of the JetBrains IDE window, click the **Copilot Chat** icon to open the Copilot Chat window.


3. At the top of the Copilot Chat window, click the **share feedback** link.


### Further reading

- [Prompt engineering for GitHub Copilot Chat](/en/copilot/using-github-copilot/prompt-engineering-for-github-copilot)
- [Asking GitHub Copilot questions in GitHub](/en/copilot/github-copilot-enterprise/copilot-chat-in-github/using-github-copilot-chat-in-githubcom)
- [Responsible use of GitHub Copilot Chat in your IDE](/en/copilot/github-copilot-chat/about-github-copilot-chat)
- [GitHub Pre-release License Terms](/en/site-policy/github-terms/github-copilot-pre-release-terms)
- [GitHub Terms for Additional Products and Features](/en/site-policy/github-terms/github-terms-for-additional-products-and-features#github-copilot)
- [GitHub Copilot Trust Center](https://copilot.github.trust.page/)
- [GitHub Copilot FAQ](https://github.com/features/copilot#faq)

### Prerequisites

- **Access to GitHub Copilot** . See [What is GitHub Copilot?](/en/copilot/about-github-copilot/what-is-github-copilot#getting-access-to-copilot) .
- **Latest version of the GitHub Copilot extension** . For installation instructions, see [Installing the GitHub Copilot extension in your environment](/en/copilot/configuring-github-copilot/installing-the-github-copilot-extension-in-your-environment) .
- **Sign in to GitHub in Xcode** . If you experience authentication issues, see [Troubleshooting common issues with GitHub Copilot](/en/copilot/troubleshooting-github-copilot/troubleshooting-issues-with-github-copilot-chat#troubleshooting-authentication-issues-in-your-editor) .

If you have access to GitHub Copilot via your organization, you won't be able to use GitHub Copilot Chat if your organization owner has disabled chat. See [Managing policies and features for GitHub Copilot in your organization](/en/copilot/managing-github-copilot-in-your-organization/managing-policies-and-features-for-copilot-in-your-organization) .

### Submitting prompts

You can ask Copilot Chat to give you code suggestions, explain code, generate unit tests, and suggest code fixes.

1. To open the chat window, click **Editor** in the menu bar, then click **GitHub Copilot** then **Open Chat** . Copilot Chat opens in a new window.
2. Enter a prompt in the prompt box. For example prompts, see [Getting started with prompts for GitHub Copilot Chat in your IDE](/en/copilot/get-started/getting-started-with-prompts-for-copilot-chat) .
3. Evaluate Copilot's response, and submit a follow up prompt if needed. The response often includes interactive elements. For example, the response may include buttons to copy or insert a code block. To see the files that Copilot Chat used to generate the response, click the **References** link below the response. The references may include a link to a custom instructions file for your repository. This file contains additional information that is automatically added to all of your chat questions to improve the quality of the responses. For more information, see [Adding repository custom instructions for GitHub Copilot](/en/copilot/how-tos/custom-instructions/adding-repository-custom-instructions-for-github-copilot) .

### Using Model Context Protocol (MCP) servers

You can use MCP to extend the capabilities of Copilot Chat by integrating it with a wide range of existing tools and services. For additional information, see [About Model Context Protocol (MCP)](/en/copilot/concepts/context/mcp) .

### AI models for Copilot Chat

You can change the model Copilot uses to generate responses to chat prompts. You may find that different models perform better, or provide more useful responses, depending on the type of questions you ask. Options include premium models with advanced capabilities.  See [Changing the AI model for GitHub Copilot Chat](/en/copilot/using-github-copilot/ai-models/changing-the-ai-model-for-copilot-chat) .

### Using keywords in your prompt

You can use special keywords to help Copilot understand your prompt.

#### Slash commands

Use slash commands to avoid writing complex prompts for common scenarios. To use a slash command, type `/` in the chat prompt box, followed by a command.

To see all available slash commands, type `/` in the chat prompt box. For more information, see [GitHub Copilot Chat cheat sheet](/en/copilot/using-github-copilot/github-copilot-chat-cheat-sheet?tool=xcode#slash-commands) .

### Using plan mode

Note

Plan mode is currently in public preview and subject to change.

Plan mode helps you to create detailed implementation plans before executing them. This ensures that all requirements are considered and addressed before any code changes are made. The plan agent does not make any code changes until the plan is reviewed and approved by you. Once approved, you can hand off the plan to the default agent or save it for further refinement, review, or team discussions.

The plan agent is designed to:

- Research the task comprehensively using read-only tools and codebase analysis to identify requirements and constraints.
- Break down the task into manageable, actionable steps and include open questions about ambiguous requirements.
- Present a concise plan draft, based on a standardized plan format, for user review and iteration.

To use plan mode:

1. If it is not already displayed, open the Copilot Chat window by clicking **Editor** in the menu bar, then clicking **GitHub Copilot** then **Open Chat** .
2. At the bottom of the Copilot Chat window, select **Plan** from the agents dropdown.
3. Type a prompt that describes a task, such as adding a feature to an existing application, refactoring code, fixing a bug, or creating an initial version of a new application. For example: `Create a simple to-do app with Swift files.`
4. Submit the prompt. After a few moments, the plan agent outputs a plan in the chat panel. The plan provides a high-level summary and a breakdown of steps, including any open questions for clarification.
5. Review the plan and answer any questions the agent has asked. You can iterate multiple times to clarify requirements, adjust scope, or answer questions.
6. Once the plan is complete you can:
    - Click **Start Implementation** to switch Copilot Chat to agent mode and start an agent session to implement the required changes, based on the implementation plan.
    - Click **Open in Editor** to switch Copilot Chat to agent mode and start an agent session that generates Markdown, in a tab of your editor, with the details of the implementation plan. You can start to work through the plan yourself, or save the plan as a Markdown file for later use.

### Using Copilot agent mode

Use agent mode when you have a specific task in mind and want to enable Copilot to autonomously edit your code. In agent mode, Copilot determines which files to make changes to, offers code changes and terminal commands to complete the task, and iterates to remediate issues until the original task is complete.

Agent mode is best suited to use cases where:

- Your task is complex, and involves multiple steps, iterations, and error handling.
- You want Copilot to determine the necessary steps to take to complete the task.
- The task requires Copilot to integrate with external applications, such as an MCP server.

#### Using agent mode

1. If it is not already displayed, open the Copilot Chat window by clicking **Editor** in the menu bar, then clicking **GitHub Copilot** then **Open Chat** .
2. At the bottom of the chat panel, select **Agent** from the agents dropdown.
3. Optionally, add relevant files to the *working set* view to indicate to Copilot which files you want to work on.
4. Submit a prompt. In response to your prompt, Copilot streams the edits in the editor, updates the working set, and if necessary, suggests terminal commands to run.
5. Review the changes. If Copilot suggested terminal commands, confirm whether or not Copilot can run them. In response, Copilot iterates and performs additional actions to complete the task in your original prompt.

When you use agent mode, each prompt you enter counts as one premium request, multiplied by the model's multiplier. For example, if you're using the included model-which has a multiplier of 0-your prompts won't consume any premium requests. Copilot may take several follow-up actions to complete your task, but these follow-up actions do **not** count toward your premium request usage. Only the prompts you enter are billed-tool calls or background steps taken by the agent are not charged.

The total number of premium requests you use depends on how many prompts you enter and which model you select. See [Requests in GitHub Copilot](/en/copilot/managing-copilot/managing-copilot-as-an-individual-subscriber/monitoring-usage-and-entitlements/avoiding-unexpected-copilot-costs) .

#### Using subagents

You can use subagents to delegate tasks to an isolated agent with its own context window within your chat session. The subagent operates independently without pausing for user feedback and returns the final result to the main chat session.

Subagents are best suited for situations where:

- You want to delegate complex, multi-step tasks like research or analysis without interrupting your main session.
- You need to process large amounts of information or multiple documents that would clutter your primary context window.
- You want to explore different approaches or perspectives independently without mixing contexts together.

Subagents use the same tools and AI model as the main session, but they cannot create other subagents.

To use subagents, you **must have custom agents configured in your environment** . See [Creating custom agents for Copilot cloud agent](/en/copilot/how-tos/use-copilot-agents/cloud-agent/create-custom-agents) .

##### Enabling subagents

1. Click **Editor** in the menu bar, then click **GitHub Copilot** then **Open GitHub Copilot for Xcode Settings** .
2. Click **Advanced** in the chat panel, then under **Chat Settings** click the **Enable Subagents** toggle.

##### Invoking subagents

Subagents can be invoked in different ways:

- **Automatic delegation** . Copilot will analyze the description of your request, the description field of your configured custom agents, and the current context and available tools to automatically choose a subagent. For example, this prompt would automatically delegate the task to a **refactor-specialist** custom agent: `Suggest ways to refactor this legacy code.`
- **Direct invocation** . You can directly call the subagent in your prompt: `Use the testing subagent to write unit tests for the authentication module.`

When the subagent completes its task, its results appear back in the main chat session, ready for follow-up questions or next steps.

### File references

By default, Copilot Chat will reference the file that you have open or the code that you have selected. To attach a specific file as reference, click in the chat prompt box.

### Chat management

You can open a conversation thread for each Xcode IDE to keep discussions organized across different contexts. You can also revisit previous conversations and reference past suggestions through the chat history.

### Sharing feedback

To indicate whether a response was helpful, use or that appear next to the response.

### Further reading

- [Prompt engineering for GitHub Copilot Chat](/en/copilot/using-github-copilot/prompt-engineering-for-github-copilot)
- [Asking GitHub Copilot questions in GitHub](/en/copilot/github-copilot-enterprise/copilot-chat-in-github/using-github-copilot-chat-in-githubcom)
- [Responsible use of GitHub Copilot Chat in your IDE](/en/copilot/github-copilot-chat/about-github-copilot-chat)
- [GitHub Pre-release License Terms](/en/site-policy/github-terms/github-copilot-pre-release-terms)
- [GitHub Terms for Additional Products and Features](/en/site-policy/github-terms/github-terms-for-additional-products-and-features#github-copilot)
- [GitHub Copilot Trust Center](https://copilot.github.trust.page/)
- [GitHub Copilot FAQ](https://github.com/features/copilot#faq)

### Prerequisites

- **Access to Copilot** . See [What is GitHub Copilot?](/en/copilot/about-github-copilot/what-is-github-copilot#getting-access-to-copilot) .
- **Compatible version of Eclipse** . To use the GitHub Copilot extension, you must have Eclipse version 2024-09 or above. See the [Eclipse download page](https://www.eclipse.org/downloads/packages/) .
- If you are a member of an organization or enterprise with a Copilot Business or Copilot Enterprise plan, the "MCP servers in Copilot" policy must be enabled in order to use MCP with Copilot.
- **Latest version of the GitHub Copilot extension** . Download this from the [Eclipse Marketplace](https://aka.ms/copiloteclipse?ref_product=copilot&ref_type=engagement&ref_style=text) . For more information, see [Installing the GitHub Copilot extension in your environment](/en/copilot/managing-copilot/configure-personal-settings/installing-the-github-copilot-extension-in-your-environment?tool=eclipse) .
- **Sign in to GitHub in Eclipse** . If you experience authentication issues, see [Troubleshooting common issues with GitHub Copilot](/en/copilot/troubleshooting-github-copilot/troubleshooting-issues-with-github-copilot-chat#troubleshooting-authentication-issues-in-your-editor) .

If you have access to GitHub Copilot via your organization, you won't be able to use GitHub Copilot Chat if your organization owner has disabled chat. See [Managing policies and features for GitHub Copilot in your organization](/en/copilot/managing-github-copilot-in-your-organization/managing-policies-and-features-for-copilot-in-your-organization) .

### Submitting prompts

You can ask Copilot Chat to give you code suggestions, explain code, generate unit tests, and suggest code fixes.

1. To open the Copilot Chat panel, click the Copilot icon ( ) in the status bar at the bottom of Eclipse, then click **Open Chat** .
2. Enter a prompt in the prompt box, then press `Enter` . For an introduction to the kinds of prompts you can use, see [Getting started with prompts for GitHub Copilot Chat in your IDE](/en/copilot/get-started/getting-started-with-prompts-for-copilot-chat) .
3. Evaluate Copilot's response, and make a follow up request if needed.

### Using keywords in your prompt

You can use special keywords to help Copilot understand your prompt. For examples, see [Getting started with prompts for GitHub Copilot Chat in your IDE](/en/copilot/get-started/getting-started-with-prompts-for-copilot-chat) .

#### Slash commands

Use slash commands to avoid writing complex prompts for common scenarios. To use a slash command, type `/` in the chat prompt box, followed by a command. For example, use `/explain` to ask Copilot to explain the code in the file currently displayed in the editor.

To see all available slash commands, type `/` in the chat prompt box.

### Using Model Context Protocol (MCP) servers

You can use MCP to extend the capabilities of Copilot Chat by integrating it with a wide range of existing tools and services. For additional information, see [About Model Context Protocol (MCP)](/en/copilot/concepts/context/mcp) .

### AI models for Copilot Chat

You can change the model Copilot uses to generate responses to chat prompts. You may find that different models perform better, or provide more useful responses, depending on the type of questions you ask. Options include premium models with advanced capabilities.  See [Changing the AI model for GitHub Copilot Chat](/en/copilot/using-github-copilot/ai-models/changing-the-ai-model-for-copilot-chat) .

### Using plan mode

Note

Plan mode is currently in public preview and subject to change.

Plan mode helps you to create detailed implementation plans before executing them. This ensures that all requirements are considered and addressed before any code changes are made. The plan agent does not make any code changes until the plan is reviewed and approved by you. Once approved, you can hand off the plan to the default agent or save it for further refinement, review, or team discussions.

The plan agent is designed to:

- Research the task comprehensively using read-only tools and codebase analysis to identify requirements and constraints.
- Break down the task into manageable, actionable steps and include open questions about ambiguous requirements.
- Present a concise plan draft, based on a standardized plan format, for user review and iteration.

To use plan mode:

1. If it is not already displayed, open the Copilot Chat panel by clicking the Copilot icon ( ) in the status bar at the bottom of Eclipse, then clicking **Open Chat** .
2. At the bottom of the chat panel, select **Plan** from the agents dropdown.
3. Type a prompt that describes a task, such as adding a feature to an existing application, refactoring code, fixing a bug, or creating an initial version of a new application. For example: `Create a simple to-do app using JavaFX.`
4. Submit the prompt. After a few moments, the plan agent outputs a plan in the chat panel. The plan provides a high-level summary and a breakdown of steps, including any open questions for clarification.
5. Review the plan and answer any questions the agent has asked. You can iterate multiple times to clarify requirements, adjust scope, or answer questions.
6. Once the plan is complete you can:
    - Click **Start Implementation** to switch Copilot Chat to agent mode and start an agent session to implement the required changes, based on the implementation plan.
    - Click **Open in Editor** to switch Copilot Chat to agent mode and start an agent session that generates Markdown, in a tab of your editor, with the details of the implementation plan. You can start to work through the plan yourself, or save the plan as a Markdown file for later use.

### Using Copilot agent mode

Use agent mode when you have a specific task in mind and want to enable Copilot to autonomously edit your code. In agent mode, Copilot determines which files to make changes to, offers code changes and terminal commands to complete the task, and iterates to remediate issues until the original task is complete.

Agent mode is best suited to use cases where:

- Your task is complex, and involves multiple steps, iterations, and error handling.
- You want Copilot to determine the necessary steps to take to complete the task.
- The task requires Copilot to integrate with external applications, such as an MCP server.

To use agent mode:

1. Open the Copilot Chat panel by clicking the Copilot icon ( ) in the status bar at the bottom of Eclipse, then clicking **Open Chat** .
2. At the bottom of the chat panel, select **Agent** from the agents dropdown.
3. Submit a prompt. In response to your prompt, Copilot streams the edits in the editor, updates the working set, and if necessary, suggests terminal commands to run.
4. Review the changes. If Copilot suggested terminal commands, confirm whether or not Copilot can run them. In response, Copilot iterates and performs additional actions to complete the task in your original prompt.

When you use agent mode, each prompt you enter counts as one premium request, multiplied by the model's multiplier. For example, if you're using the included model-which has a multiplier of 0-your prompts won't consume any premium requests. Copilot may take several follow-up actions to complete your task, but these follow-up actions do **not** count toward your premium request usage. Only the prompts you enter are billed-tool calls or background steps taken by the agent are not charged.

The total number of premium requests you use depends on how many prompts you enter and which model you select. See [Requests in GitHub Copilot](/en/copilot/managing-copilot/managing-copilot-as-an-individual-subscriber/monitoring-usage-and-entitlements/avoiding-unexpected-copilot-costs) .

#### Using subagents

You can use subagents to delegate tasks to an isolated agent with its own context window within your chat session. The subagent operates independently without pausing for user feedback and returns the final result to the main chat session.

Subagents are best suited for situations where:

- You want to delegate complex, multi-step tasks like research or analysis without interrupting your main session.
- You need to process large amounts of information or multiple documents that would clutter your primary context window.
- You want to explore different approaches or perspectives independently without mixing contexts together.

Subagents use the same tools and AI model as the main session, but they cannot create other subagents.

To use subagents, you **must have custom agents configured in your environment** . See [Creating custom agents for Copilot cloud agent](/en/copilot/how-tos/use-copilot-agents/cloud-agent/create-custom-agents) .

##### Enabling subagents

1. Click the icon in the status bar.
2. In the popup menu, click **Edit Preferences** .
3. Under **Chat** , click the **Enable sub-agent** check box

##### Invoking subagents

Subagents can be invoked in different ways:

- **Automatic delegation** . Copilot will analyze the description of your request, the description field of your configured custom agents, and the current context and available tools to automatically choose a subagent. For example, this prompt would automatically delegate the task to a **refactor-specialist** custom agent: `Suggest ways to refactor this legacy code.`
- **Direct invocation** . You can directly call the subagent in your prompt: `Use the testing subagent to write unit tests for the authentication module.`

When the subagent completes its task, its results appear back in the main chat session, ready for follow-up questions or next steps.

### Further reading

- [Prompt engineering for GitHub Copilot Chat](/en/copilot/using-github-copilot/prompt-engineering-for-github-copilot)
- [Asking GitHub Copilot questions in GitHub](/en/copilot/github-copilot-enterprise/copilot-chat-in-github/using-github-copilot-chat-in-githubcom)
- [Responsible use of GitHub Copilot Chat in your IDE](/en/copilot/github-copilot-chat/about-github-copilot-chat)
- [GitHub Terms for Additional Products and Features](/en/site-policy/github-terms/github-terms-for-additional-products-and-features#github-copilot)
- [GitHub Copilot Trust Center](https://copilot.github.trust.page/)
- [GitHub Copilot FAQ](https://github.com/features/copilot#faq)


### Prerequisites

- **Access to GitHub Copilot** . See [What is GitHub Copilot?](/en/copilot/about-github-copilot/what-is-github-copilot#getting-access-to-copilot) .
- **Windows Terminal Canary installed** . For installation instructions, see [Installing Windows Terminal Canary](https://github.com/microsoft/terminal?tab=readme-ov-file#installing-windows-terminal-canary) .
- **GitHub Copilot connected to Terminal Chat** . See [Quickstart for GitHub Copilot](/en/copilot/quickstart?tool=windowsterminal) .

If you have access to GitHub Copilot via your organization or enterprise, you cannot use Copilot in Windows Terminal if your organization owner or enterprise administrator has disabled Copilot CLI. See [Managing policies and features for GitHub Copilot in your organization](/en/copilot/managing-copilot/managing-github-copilot-in-your-organization/managing-github-copilot-features-in-your-organization/managing-policies-for-copilot-in-your-organization) .

### Getting command explanations and suggestions

In the Terminal Chat chat window, type a question (for example, `how do i list all markdown files in my directory` ) then press `Enter` .

Copilot's answer is displayed below your question.

Click on an answer to insert it to the command line.

### Sharing feedback

To send feedback to Windows Terminal about the quality of a suggestion, open an issue in the [Windows Terminal repository](https://github.com/microsoft/terminal/issues) .

### Further reading

- [Terminal Chat](https://learn.microsoft.com/windows/terminal/terminal-chat#setting-up-terminal-chat) in the Microsoft Learn documentation


---

# Code Review


### Introduction

Copilot code review reviews code written in any language, and provides feedback. It reviews your code from multiple angles to identify issues and suggest fixes. You can apply suggested changes with a couple of clicks.

This article provides an overview of Copilot code review. To learn how to request a code review from Copilot, see [Using GitHub Copilot code review](/en/copilot/how-tos/agents/copilot-code-review/using-copilot-code-review) .

### Availability

Copilot code review is supported in:

- GitHub.com
- GitHub Mobile
- VS Code
- Visual Studio
- Xcode
- JetBrains IDEs

Copilot code review is a premium feature available with these plans:

- Copilot Pro
- Copilot Pro+
- Copilot Business
- Copilot Enterprise

See [Copilot plans](https://github.com/features/copilot/plans?ref_product=copilot&ref_type=purchase&ref_style=text) .

If you receive Copilot from an organization, your organization must enable the **Copilot code review** option in the Copilot policy settings. This applies to reviews on GitHub.com or in GitHub Mobile. See [Managing policies and features for GitHub Copilot in your organization](/en/copilot/how-tos/administer/organizations/managing-policies-for-copilot-in-your-organization) .

### Copilot code review without a Copilot license

Organization members **without a Copilot license** can use Copilot code review on GitHub.com. An enterprise administrator or organization owner must enable it. This capability is available to organizations on **Copilot Business** and **Copilot Enterprise** plans.

#### Enabling code review for users without a license

To allow organization members without a Copilot license to use Copilot code review, you must enable two policies:

1. **Premium request paid usage** . Enable this policy first. It allows the enterprise or organization to incur charges for Copilot code review premium request usage.
2. **Allow members without a Copilot license to use Copilot code review in GitHub.com** . This sub-policy enables Copilot code review for users without a license.

The second policy has these characteristics:

- It is disabled by default.
- Once this policy is set it at the enterprise level, it becomes **visible, but not editable** at the organization level.
- The policy is **most restrictive** . Copilot code review is only available in repositories where you explicitly enable the policy.

#### How it works for users without a license

When both policies are enabled, users without a Copilot license can request a review from Copilot code review on their pull requests in the organization's repositories.

In repositories where automatic code review is enabled, Copilot automatically reviews all pull requests. This happens regardless of whether the author has a Copilot license.

Copilot code review for users without a license is not available in IDEs.

### Excluded files

Some file types are excluded from Copilot code review:

- Dependency management files, such as package.json and Gemfile.lock
- Log files
- SVG files

If you include these file types in a pull request, Copilot code review will not review the file.

For more information, see [Files excluded from GitHub Copilot code review](/en/copilot/reference/review-excluded-files) .

### Agentic capabilities for Copilot code review

Note

- Copilot code review has capabilities that are in public preview and subject to change. The [GitHub Pre-release License Terms](/en/site-policy/github-terms/github-pre-release-license-terms) apply to your use of preview features.

Copilot code review utilizes agentic capabilities to extend its functionality.

- **Full project context gathering** . This provides more specific, accurate, and contextually aware code reviews. This capability analyzes your entire repository to better understand the context of code changes. Full project context gathering is generally available.
- **The ability to pass suggestions to Copilot cloud agent** . This automates creating a new pull request against your branch with the suggested fixes applied. Passing suggestions to Copilot cloud agent is in public preview and subject to change.

These capabilities are enabled automatically for Copilot Pro or Copilot Pro+ plans.

If GitHub Actions is unavailable or if Actions workflows used by Copilot code review fail, reviews will still be generated. However, they will not include the additional features provided by the agentic capabilities.

#### Usage of GitHub Actions runners for agentic capabilities in code review

Copilot code review uses free minutes for GitHub Actions to run the agentic capabilities, including full project context gathering and any capabilities in public preview. By default, Copilot code review uses GitHub-hosted runners. You can also upgrade to larger GitHub-hosted runners for better performance.

Note

Usage of larger GitHub-hosted runners is billed per-minute and may incur additional GitHub Actions charges.

You do not need to have GitHub Actions enabled in your organization or enterprise to use the agentic capabilities in code review.

If your organization has disabled GitHub-hosted runners, the agentic capabilities will not be available. In this case, code reviews will fall back to a more limited review. Organizations in this situation can use self-hosted runners.

For more information on configuring runners, see [Configuring runners for GitHub Copilot code review](/en/copilot/how-tos/copilot-on-github/set-up-copilot/configure-runners) .

### Code review monthly quota

Each time Copilot reviews a pull request or reviews code in your IDE, your monthly quota of Copilot premium requests is reduced by one.

If a repository is configured to automatically request a code review from Copilot for all new pull requests, the premium request usage is applied to the pull request author's quota. If a review is manually requested by another user, the usage is applied to that user's quota instead.

If a pull request is created by GitHub Actions or by a bot, the usage will apply to:

- The user who triggered the workflow, if that user can be identified.
- A designated billing owner.

#### What happens when you reach your quota

When you reach your monthly quota, you will not be able to get a code review from Copilot until your quota resets. To continue to use code reviews before your quota resets, you will need to upgrade your Copilot plan or enable additional premium requests.

#### Users without a Copilot license or plan that includes Copilot code review

Users without access to Copilot code review do not have a monthly premium request quota. This includes users who have no Copilot license and users on the Copilot Free plan, which does not include Copilot code review.

When Copilot code review is enabled for these users, any premium requests they generate are billed directly to the organization or enterprise as paid overage usage. This applies to both manually requested reviews and automatic code reviews.

Premium requests generated by these users are not attributed to any Copilot plan quota. They appear as overage usage in billing reports and premium request analytics. Users with a Copilot license that includes code review continue to consume premium requests from their assigned plan quota.

### Model usage

Copilot code review is a purpose-built product that uses a carefully tuned mix of models, prompts, and system behaviors to deliver consistent, high-quality feedback across a wide range of codebases. Model switching is not supported, as changing the model is likely to compromise reliability, user experience, and the quality of review comments.

Note

Copilot code review may use models that are not enabled on your organization's "Models" settings page. The "Models" settings page only controls Copilot Chat.

Since Copilot code review is generally available, all model usage will be subject to the generally available terms. See [Managing policies and features for GitHub Copilot in your organization](/en/copilot/how-tos/administer-copilot/manage-for-organization/manage-policies) .

### Validating Copilot code reviews

Copilot is not guaranteed to spot all problems or issues in a pull request. Sometimes it will make mistakes. Always validate Copilot's feedback carefully. Supplement Copilot's feedback with a human review.

For more information, see [Responsible use of GitHub Copilot code review](/en/copilot/responsible-use-of-github-copilot-features/responsible-use-of-github-copilot-code-review) .

### Enhancing Copilot's knowledge of a repository

The more Copilot knows about the code in your repository, the tools you use, and your coding standards and practices, the more accurate and useful its reviews will become. You can enhance Copilot's knowledge of your repositories in two ways.

#### Custom instructions

These are short, natural-language statements that you write and store as one or more files in a repository. If you are the owner of an organization on GitHub, you can also define custom instructions in the settings for your organization. For more information, see [About customizing GitHub Copilot responses](/en/copilot/concepts/prompting/response-customization?tool=webui#about-repository-custom-instructions) .

#### Copilot Memory (public preview)

If you have a Copilot Pro or Copilot Pro+ plan, you can enable Copilot Memory. This allows Copilot to store useful details it has learned about a repository. Copilot can then use this information when it reviews pull requests in that repository. For more information, see [About agentic memory for GitHub Copilot](/en/copilot/concepts/agents/copilot-memory) .

### About automatic pull request reviews

By default, Copilot only reviews a pull request if you assign it to the pull request. However, you can configure automatic reviews.

- **Individual users** on the Copilot Pro or Copilot Pro+ plan can configure Copilot to automatically review all pull requests they create.
- **Repository owners** can configure Copilot to automatically review all pull requests in the repository that are created by people with access to Copilot.
- **Organization owners** can configure Copilot to automatically review all pull requests in some or all of the repositories in the organization where the pull request is created by a Copilot user.

#### Triggering an automatic pull request review

The triggers for automatic code review depend on the configuration settings.

- Basic setting:
    - When you create a pull request as an "Open" pull request.
    - The first time you switch a "Draft" pull request to "Open".
- Review new pushes:
    - Every time you push a new commit to the pull request.
- Review draft pull requests:
    - Pull requests are automatically reviewed while they are still drafts, before you switch them to "Open".

For full instructions, see [Configuring automatic code review by GitHub Copilot](/en/copilot/how-tos/agents/copilot-code-review/configuring-automatic-code-review-by-copilot) .

Note

Unless Copilot has been configured to review each push to a pull request, it will only review a pull request once. If you make changes to the pull request after it has been automatically reviewed and you want Copilot to re-review it, you can request this manually. Click the button next to Copilot's name in the **Reviewers** menu.

### Getting detailed code quality feedback for your whole repository

GitHub Copilot code review reviews your code in pull requests and provides feedback. If you want actionable feedback on the reliability and maintainability of your whole repository, enable GitHub Code Quality. See [About GitHub Code Quality](/en/code-security/code-quality/concepts/about-code-quality) .

### Further reading

- [Using GitHub Copilot code review](/en/copilot/how-tos/agents/copilot-code-review/using-copilot-code-review)


---

# Selected Tutorials


### Introduction

If you've been assigned to work on a project that you're not familiar with-or you've found an interesting open source project that you want to contribute to-you'll need some understanding of the codebase before you can start making changes. This guide will show you how to use GitHub Copilot Chat to explore a codebase and quickly learn about the project.

### Working with Copilot Chat

Throughout this guide, we'll work with Copilot Chat on GitHub.com, which you can find at [github.com/copilot](https://github.com/copilot?ref_product=copilot&ref_type=engagement&ref_style=text) .

### Attaching a codebase

Before Copilot Chat can help you, you need to attach the codebase you want to explore.

1. On GitHub, navigate to [github.com/copilot](https://github.com/copilot?ref_product=copilot&ref_type=engagement&ref_style=text) .
2. In the text box, click **Add repositories, files, and spaces** , then click **Repositories** .
3. Search for and select the repository you want to explore.

Copilot Chat now has access to the code in that repository, and you can start asking questions about it.

### Example prompts

The following prompts are examples of the kind of questions you can ask Copilot to help you find out about a codebase.

#### General questions

- Based on the code in this repository, give me an overview of the architecture of the codebase. Provide evidence.
- Which languages are used in this repo? Show the percentages for each language.
- What are the core algorithms implemented in this repo?
- What design patterns are used in this repository? Give a brief explanation of each pattern that you find, and an example of code from this repository that uses the pattern, with a link to the file.

#### Specific questions

Whether these questions are useful will depend on the codebase you're exploring.

- How do I build this project?
- Where is authentication handled in this codebase?
- Analyze the code in this repository and tell me about the entry points for this application.
- Describe the data flow in this application.
- Analyze the code in this repository and tell me what application-level security mechanisms are employed. Provide references.

### Understanding the files in a directory

Use Copilot to help you understand the purpose of the files in a directory, or individual files.

To find out about the files in a directory:

1. Navigate to the directory on GitHub.com.
2. In the top right corner of the page, click the Copilot icon ( ) to open Copilot Chat. Copilot will use the directory contents as context for your question.
3. Ask Copilot: `Explain the files in this directory` .

To find out about a specific file:

1. Open the file on GitHub.com.
2. In the top right corner of the page, click the Copilot icon ( ) to open Copilot Chat. Copilot will use the file contents as context for your question.
3. For a small file, ask Copilot: `Explain this file` .
4. For a large file, ask: `Explain what this file does. Start with an overview of the purpose of the file. Then, in appropriately headed sections, go through each part of the file and explain what it does in detail.`

### Understanding specific lines of code

Use Copilot to help you understand specific lines of code in a file.

To find out about a specific line of code:

1. On GitHub, navigate to a repository and open a file.
2. Select the lines by clicking the line number for the first line you want to select, holding down `Shift` and clicking the line number for the last line you want to select.
3. To ask your own question about the selected lines, click the Copilot icon ( ) to the right of your selection. This displays the GitHub Copilot Chat panel with the selected lines indicated as the context of your question.
4. To ask a predefined question, click the downward-pointing button beside the Copilot icon, then choose one of the options.


6. If you clicked the Copilot icon, type a question in the prompt box at the bottom of the chat panel and press `Enter` .

### Understanding a specific file or symbol

Use Copilot to help you understand the purpose of a specific file or symbol in the codebase. A symbol is a named entity in the code, such as a function, class, or variable.

1. On GitHub, navigate to a repository and open a file.
2. At the top of the file, click the Copilot icon ( ) to open Copilot Chat. Copilot will display the file contents in a split screen as context for your question.
3. If you want to ask about a specific symbol, highlight the symbol in the file.
4. In the prompt box, type a question about the file or highlighted symbol, and press `Enter` . Copilot replies in the chat panel. Tip Copilot's ability to answer natural language questions like these in a repository context is optimized when the semantic code search index for the repository is up to date. For more information, see [Indexing repositories for GitHub Copilot](/en/copilot/managing-copilot/managing-github-copilot-in-your-organization/managing-github-copilot-features-in-your-organization/indexing-repositories-for-copilot-chat) .

### Finding out about commits

One good way to familiarize yourself with a project is to look at the recent work that's been happening. You can do this by browsing the recent commits.

1. On GitHub, navigate to the main page of the repository.
2. On the main page of the repository, above the file list, click **commits** .


4. Click a commit message to display a diff view for that commit.
5. In the Copilot Chat panel, enter: `What does this commit do?` .
6. If necessary, you can follow up by entering: `Explain in more detail` .

### Using the Insights tab

In addition to using Copilot to help you become familiar with a project, you can also use the **Insights** tab on GitHub.com. This gives you a high-level overview of the repository.

For more information, see [Using Pulse to view a summary of repository activity](/en/repositories/viewing-activity-and-data-for-your-repository/using-pulse-to-view-a-summary-of-repository-activity) and [Viewing a project's contributors](/en/repositories/viewing-activity-and-data-for-your-repository/viewing-a-projects-contributors) .

### Further reading

- [Asking GitHub Copilot questions in GitHub](/en/copilot/using-github-copilot/copilot-chat/asking-github-copilot-questions-in-github)


### Introduction

GitHub Copilot can assist you in developing tests quickly and improving productivity. In this article, we'll demonstrate how you can use Copilot to write both unit and integration tests. While Copilot performs well when generating tests for basic functions, complex scenarios require more detailed prompts and strategies. This article will walk through practical examples of using Copilot to break down tasks and verify code correctness.

### Prerequisites

Before getting started you must have the following:

- A [GitHub Copilot subscription plan](/en/copilot/about-github-copilot/subscription-plans-for-github-copilot) .
- Visual Studio, Visual Studio Code, or any JetBrains IDE.
- The [GitHub Copilot extension](/en/copilot/managing-copilot/configure-personal-settings/installing-the-github-copilot-extension-in-your-environment) installed in your IDE.

### Writing unit tests with Copilot Chat

In this section, we'll explore how to use GitHub Copilot Chat to generate unit tests for a Python class. This example demonstrates how you can use Copilot to create unit tests for a class like `BankAccount` . We will show you how to prompt Copilot to generate tests, execute them, and verify the results.

#### Example class: BankAccount

Let's start with a class `BankAccount` that contains methods for depositing, withdrawing, and getting the balance of an account. Create a new file `bank_account.py` in a GitHub repository and add the following `BankAccount` class in Python.

```
class BankAccount : def __init__ ( self, initial_balance= 0 ): if initial_balance < 0 : raise ValueError( "Initial balance cannot be negative." ) self .balance = initial_balance def deposit ( self, amount ): if amount <= 0 : raise ValueError( "Deposit amount must be positive." ) self .balance += amount def withdraw ( self, amount ): if amount <= 0 : raise ValueError( "Withdrawal amount must be positive." ) if amount > self .balance: raise ValueError( "Cannot withdraw more than the current balance." ) self .balance -= amount def get_balance ( self ): return self .balance
```

#### Generating unit tests with Copilot Chat

You can prompt Copilot with a request to write tests that cover a range of scenarios. Be specific with your test requirements to get the best results. The following is an example prompt you can use to generate unit tests for the `BankAccount` class.

**Prompt:** *Develop a comprehensive suite of unit tests for the BankAccount() function in Python. Write multiple test methods that cover a wide range of scenarios, including edge cases, exception handling, and data validation.*

This prompt does a few important things:

- **Scope:** It explicitly asks for a comprehensive suite of unit tests, so that Copilot generates more than just basic tests, covering a variety of potential use cases.
- **Scenarios:** The prompt specifies that the tests should include:
    - **Edge cases:** Tests for boundary conditions (e.g., negative values for deposits or withdrawals, withdrawing more than the available balance).
    - **Exception handling:** Tests for errors that should be raised when certain invalid actions occur (e.g., withdrawing more than the available balance, depositing negative amounts).
    - **Data validation:** Ensures that inputs are correctly validated and handled by the methods.

Copilot will give you a response like the following.

```
import unittest from bank_account import BankAccount class TestBankAccount (unittest.TestCase): def setUp ( self ): self .account = BankAccount() def test_initial_balance ( self ): self .assertEqual( self .account.get_balance(), 0 ) def test_deposit_positive_amount ( self ): self .account.deposit( 100 ) self .assertEqual( self .account.get_balance(), 100 ) def test_withdraw_within_balance ( self ): self .account.deposit( 100 ) self .account.withdraw( 50 ) self .assertEqual( self .account.get_balance(), 50 ) def test_deposit_negative_amount_raises_error ( self ): with self .assertRaises(ValueError): self .account.deposit(- 100 ) def test_withdraw_negative_amount_raises_error ( self ): with self .assertRaises(ValueError): self .account.withdraw(- 50 ) def test_withdraw_more_than_balance_raises_error ( self ): self .account.deposit( 100 ) with self .assertRaises(ValueError): self .account.withdraw( 200 ) def test_initial_balance_negative_raises_error ( self ): with self .assertRaises(ValueError):
            BankAccount(- 100 ) if __name__ == '__main__' :
    unittest.main()
```

#### Running the tests cases

The test suite generated by Copilot includes a test case for each scenario specified in the prompt. Each function in the `BankAccount` class is tested for a range of conditions, including valid operations like deposits and withdrawals, as well as edge cases such as negative amounts and attempts to withdraw more than the available balance.

Once Copilot has generated the test suite to your satisfaction, add the code to a new file `test_bank_account.py` . You can ask it how to run the tests.

**Prompt:** *"How do I run these unit tests in Python using the unittest framework?"*

Copilot will give you the following bash command.

```
python -m unittest test_bank_account.py
```

After running the tests, you will see the output in your terminal or IDE. If all tests pass, you can be confident that your `BankAccount` class is working as expected.

##### Slash command

Additionally, you can prompt Copilot to write a full suite of unit tests with the `/tests` slash command. Ensure that you have the file open on the current tab of your IDE and Copilot will generate unit tests for that file. The tests that Copilot generates may not cover all scenarios, so you should always review the generated code and add any additional tests that may be necessary.

Tip

If you ask Copilot to write tests for a code file that is not already covered by unit tests, you can provide Copilot with useful context by opening one or more existing test files in adjacent tabs in your editor. Copilot will be able to see the testing framework you use and will be more likely to write a test that is consistent with your existing tests.

Copilot will generate a unit test suite such as the following.

```
import unittest from bank_account import BankAccount class TestBankAccount (unittest.TestCase): def setUp ( self ): self .account = BankAccount() def test_initial_balance ( self ): self .assertEqual( self .account.get_balance(), 0 )
```

### Writing integration tests with Copilot

Integration tests are essential for ensuring that the various components of your system work correctly when combined. In this section, we'll extend our `BankAccount` class to include interactions with an external service `NotificationSystem` and use mocks to test the system's behavior without needing real connections. The goal of the integration tests is to verify the interaction between the `BankAccount` class and the `NotificationSystem` services, ensuring that they work together correctly.

#### Example class: BankAccount with notification services

Let's update the `BankAccount` class to include interactions with an external service such as a `NotificationSystem` that sends notifications to users. `NotificationSystem` represents the integration that would need to be tested.

Update the `BankAccount` class in the `bank_account.py` file with the following code snippet.

```
class BankAccount : def __init__ ( self, initial_balance= 0 , notification_system= None ): if initial_balance < 0 : raise ValueError( "Initial balance cannot be negative." ) self .balance = initial_balance self .notification_system = notification_system def deposit ( self, amount ): if amount <= 0 : raise ValueError( "Deposit amount must be positive." ) self .balance += amount if self .notification_system: self .notification_system.notify( f"Deposited {amount} , new balance: {self.balance} " ) def withdraw ( self, amount ): if amount <= 0 : raise ValueError( "Withdrawal amount must be positive." ) if amount > self .balance: raise ValueError( "Cannot withdraw more than the current balance." ) self .balance -= amount if self .notification_system: self .notification_system.notify( f"Withdrew {amount} , new balance: {self.balance} " ) def get_balance ( self ): return self .balance
```

Here we'll break down our request for Copilot to write integration tests for the `BankAccount` class into smaller, more manageable pieces. This will help Copilot generate more accurate and relevant tests.

**Prompt:** *"Write integration tests for the* *`deposit`* *function in the* *`BankAccount`* *class. Use mocks to simulate the* *`NotificationSystem`* *and verify that it is called correctly after a deposit."*

This prompt does a few important things:

- **Scope:** It specifies integration tests, focusing on the interaction between the `deposit` function and the `NotificationSystem` , rather than just unit tests.
- **Mocks:** It explicitly asks for the use of mocks to simulate the `NotificationSystem` , ensuring that the interaction with external systems is tested without relying on their actual implementation.
- **Verification:** The prompt emphasizes verifying that the `NotificationSystem` is called correctly after a deposit, ensuring that the integration between the components works as expected.
- **Specificity:** The prompt clearly states the method ( `deposit` ) and the class ( `BankAccount` ) to be tested.

Tip

If Copilot is producing invalid tests, provide examples of inputs and outputs for the function you want to test. This will help Copilot evaluate the expected behavior of the function.

Copilot will generate a test suite like the following.

```
import unittest from unittest.mock import Mock from bank_account import BankAccount class TestBankAccountIntegration (unittest.TestCase): def setUp ( self ): self .notification_system = Mock() def test_deposit_with_notification ( self ):
        account = BankAccount(initial_balance= 100 , notification_system= self .notification_system)
        account.deposit( 50 ) self .assertEqual(account.get_balance(), 150 ) self .notification_system.notify.assert_called_once_with( "Deposited 50, new balance: 150" ) if __name__ == '__main__' :
    unittest.main()
```

Add the generated code to a new file `test_bank_account_integration.py` .

#### Improving on the test cases

The prompt above generated a single test case that verifies the `NotificationSystem` is called when a valid deposit is made. However, it doesn't cover cases where an error is raised during the deposit. In those scenarios, the `NotificationSystem` should not be called. We need to add a test case that handles invalid deposits and ensure the notification system is not triggered.

**Prompt:** *"Add a test case for invalid deposit amounts to verify the function raises the correct exceptions and that the* *`NotificationService`* *is not called."*

Copilot will generate a test case such as the following.

```
def test_deposit_negative_amount_raises_error ( self ):
        account = BankAccount(initial_balance= 100 , notification_system= self .notification_system) with self .assertRaises(ValueError):
            account.deposit( 0 ) self .notification_system.notify.assert_not_called()
```

#### Asking about areas of improvement

Now that we've written test cases to validate the integration functionality for deposits, this is a great opportunity to search for improvements within the test suite. While the current tests are functional, we can prompt Copilot to evalulate code coverage and suggest areas of improvement.

**Prompt:** *"What additional tests should be included to ensure full coverage for the integration between the* *`BankAccount`* *class and the* *`NotificationSystem`* *?"*

Prompting Copilot with this question can help you identify missing test cases that may have been overlooked.

In this situation, while we tested valid and invalid deposits, we haven't yet covered the withdrawal functionality.

Copilot will generate an updated test suite such as the following.

Click to expand the full generated code example

```
import unittest from unittest.mock import Mock from bank_account import BankAccount class TestBankAccountIntegration (unittest.TestCase): def setUp ( self ): self .notification_system = Mock() def test_deposit_with_notification ( self ):
        account = BankAccount(initial_balance= 100 , notification_system= self .notification_system)
        account.deposit( 50 ) self .assertEqual(account.get_balance(), 150 ) self .notification_system.notify.assert_called_once_with( "Deposited 50, new balance: 150" ) def test_deposit_negative_amount_raises_error ( self ):
        account = BankAccount(initial_balance= 100 , notification_system= self .notification_system) with self .assertRaises(ValueError):
            account.deposit(- 50 ) self .notification_system.notify.assert_not_called() def test_deposit_zero_amount_raises_error ( self ):
        account = BankAccount(initial_balance= 100 , notification_system= self .notification_system) with self .assertRaises(ValueError):
            account.deposit( 0 ) self .notification_system.notify.assert_not_called() def test_withdraw_with_notification ( self ):
        account = BankAccount(initial_balance= 100 , notification_system= self .notification_system)
        account.withdraw( 30 ) self .assertEqual(account.get_balance(), 70 ) self .notification_system.notify.assert_called_once_with( "Withdrew 30, new balance: 70" ) def test_withdraw_exceeding_balance_raises_error ( self ):
        account = BankAccount(initial_balance= 100 , notification_system= self .notification_system) with self .assertRaises(ValueError):
            account.withdraw( 150 ) self .notification_system.notify.assert_not_called() def test_withdraw_negative_amount_raises_error ( self ):
        account = BankAccount(initial_balance= 100 , notification_system= self .notification_system) with self .assertRaises(ValueError):
            account.withdraw(- 30 ) self .notification_system.notify.assert_not_called() def test_withdraw_zero_amount_raises_error ( self ):
        account = BankAccount(initial_balance= 100 , notification_system= self .notification_system) with self .assertRaises(ValueError):
            account.withdraw( 0 ) self .notification_system.notify.assert_not_called() def test_initial_negative_balance_raises_error ( self ): with self .assertRaises(ValueError):
            BankAccount(initial_balance=- 100 , notification_system= self .notification_system) if __name__ == '__main__' :
    unittest.main()
```

Once Copilot has generated the test suite to your satisfaction, run the tests with command below to verify the results.

```
python -m unittest test_bank_account_integration.py
```

### Using Copilot Spaces to improve test suggestions

Copilot Spaces is a feature that allows you to organize and share task-specific context with Copilot. This can help improve the relevance of the suggestions you receive. By providing Copilot with more context about your project, you can get better test suggestions.

For example, you could create a space that includes:

- The module you're testing (like `payments.js` )
- The current test suite (like `payments.test.js` )
- A test coverage report or notes about what's missing

In the space, you can ask Copilot questions like:

What test cases are missing in `payments.test.js` based on the logic in `payments.js` ?

Or:

Write a unit test for the refund logic in `refund.js` , following the structure in the existing test suite.

For more information about using Copilot Spaces, see [About GitHub Copilot Spaces](/en/copilot/using-github-copilot/copilot-spaces/about-organizing-and-sharing-context-with-copilot-spaces) .


### Introduction

Refactoring code is the process of restructuring existing code without changing its behavior. The benefits of refactoring include improving code readability, reducing complexity, making the code easier to maintain, and allowing new features to be added more easily.

This article gives you some ideas for using Copilot to refactor code in your IDE.

Note

Example responses are included in this article. GitHub Copilot Chat may give you different responses from the ones shown here.

### Understanding code

Before you modify existing code you should make sure you understand its purpose and how it currently works. Copilot can help you with this.

1. Select the relevant code in your IDE's editor.
2. Open inline chat:
    - **In VS Code:** Press `Command` + `i` (Mac) or `Ctrl` + `i` (Windows/Linux).
    - **In Visual Studio:** Press `Alt` + `/` .
    - **In JetBrains IDEs:** Press `Control` + `Shift` + `i` (Mac) or `Ctrl` + `Shift` + `g` (Windows/Linux).
3. In the input box for inline chat, type a forward slash ( `/` ).
4. In the dropdown list, select **/explain** and press `Enter` .
5. If the explanation that Copilot returns is more than a few lines, click **View in Chat** to allow you to read the explanation more easily.

### Optimizing inefficient code

Copilot can help you to optimize code - for example, to make the code run more quickly.

#### Example code

In the two sections below, we'll use the following example bash script to demonstrate how to optimize inefficient code:

```
#!/bin/bash
### Find all .txt files and count lines in each
for file in $(find . - type f -name "*.txt" ); do wc -l " $file "
done
```

#### Use the Copilot Chat panel

Copilot can tell you whether code, like the example bash script, can be optimized.

1. Select either the `for` loop or the entire contents of the file.
2. Open Copilot Chat by clicking the chat icon in the activity bar or by using the keyboard shortcut:
    - **VS Code and Visual Studio:** `Control` + `Command` + `i` (Mac) / `Ctrl` + `Alt` + `i` (Windows/Linux)
    - **JetBrains:** `Control` + `Shift` + `c`
3. In the input box at the bottom of the chat panel, type: `Can this script be improved?` Copilot replies with a suggestion that will make the code more efficient.
4. To apply the suggested change:
    - **In VS Code and JetBrains:** Hover over the suggestion in the chat panel and click the **Insert At Cursor** icon.


    - **In Visual Studio:** Click **Preview** then, in the comparison view, click **Accept** .


#### Use Copilot inline chat

Alternatively, if you already know that existing code, like the example bash script, is inefficient:

1. Select either the `for` loop or the entire contents of the file.
2. Open inline chat:
    - **In VS Code:** Press `Command` + `i` (Mac) or `Ctrl` + `i` (Windows/Linux).
    - **In Visual Studio:** Press `Alt` + `/` .
    - **In JetBrains IDEs:** Press `Control` + `Shift` + `i` (Mac) or `Ctrl` + `Shift` + `g` (Windows/Linux).
3. Type `optimize` and press `Enter` . Copilot suggests revised code. For example: `find . - type f -name "*.txt" - exec wc -l {} +` This is more efficient than the original code, shown earlier in this article, because using `-exec ... +` allows `find` to pass multiple files to `wc` at once rather than calling `wc` once for each `*.txt` file that's found.
4. Assess Copilot's suggestion and, if you agree with the change:
    - **In VS Code and Visual Studio:** Click **Accept** .
    - **In JetBrains:** Click the Preview icon (double arrows), then click the Apply All Diffs icon (double angle brackets).

As with all Copilot suggestions, you should always check that the revised code runs without errors and produces the correct result.

### Cleaning up repeated code

Avoiding repetition will make your code easier to revise and debug. For example, if the same calculation is performed more than once at different places in a file, you could move the calculation to a function.

In the following very simple JavaScript example, the same calculation (item price multiplied by number of items sold) is performed in two places.

```
let totalSales = 0 ; let applePrice = 3 ; let applesSold = 100 ;
totalSales += applePrice * applesSold; let orangePrice = 5 ; let orangesSold = 50 ;
totalSales += orangePrice * orangesSold; console . log ( `Total: ${totalSales} ` );
```

You can ask Copilot to move the repeated calculation into a function.

1. Select the entire contents of the file.
2. Open inline chat:
    - **In VS Code:** Press `Command` + `i` (Mac) or `Ctrl` + `i` (Windows/Linux).
    - **In Visual Studio:** Press `Alt` + `/` .
    - **In JetBrains IDEs:** Press `Control` + `Shift` + `i` (Mac) or `Ctrl` + `Shift` + `g` (Windows/Linux).
3. Type: `move repeated calculations into functions` and press `Enter` . Copilot suggests revised code. For example: `function calculateSales ( price, quantity ) { return price * quantity; } let totalSales = 0 ; let applePrice = 3 ; let applesSold = 100 ; totalSales += calculateSales (applePrice, applesSold); let orangePrice = 5 ; let orangesSold = 50 ; totalSales += calculateSales (orangePrice, orangesSold); console . log ( `Total: ${totalSales} ` );`
4. Assess Copilot's suggestion and, if you agree with the change:
    - **In VS Code and Visual Studio:** Click **Accept** .
    - **In JetBrains:** Click the Preview icon (double arrows), then click the Apply All Diffs icon (double angle brackets).

As with all Copilot suggestions, you should always check that the revised code runs without errors and produces the correct result.

### Making code more concise

If code is unnecessarily verbose it can be difficult to read and maintain. Copilot can suggest a more concise version of selected code.

In the following example, this Python code outputs the area of a rectangle and a circle, but could be written more concisely:

```
def calculate_area_of_rectangle ( length, width ):
    area = length * width return area def calculate_area_of_circle ( radius ): import math
    area = math.pi * (radius ** 2 ) return area

length_of_rectangle = 10 width_of_rectangle = 5 area_of_rectangle = calculate_area_of_rectangle(length_of_rectangle, width_of_rectangle) print ( f"Area of rectangle: {area_of_rectangle} " )

radius_of_circle = 7 area_of_circle = calculate_area_of_circle(radius_of_circle) print ( f"Area of circle: {area_of_circle} " )
```

1. Select the entire contents of the file.
2. Open inline chat:
    - **In VS Code:** Press `Command` + `i` (Mac) or `Ctrl` + `i` (Windows/Linux).
    - **In Visual Studio:** Press `Alt` + `/` .
    - **In JetBrains IDEs:** Press `Control` + `Shift` + `i` (Mac) or `Ctrl` + `Shift` + `g` (Windows/Linux).
3. Type: `make this more concise` and press `Enter` . Copilot suggests revised code. For example: `import math def calculate_area_of_rectangle ( length, width ): return length * width def calculate_area_of_circle ( radius ): return math.pi * (radius ** 2 ) print ( f"Area of rectangle: {calculate_area_of_rectangle( 10 , 5 )} " ) print ( f"Area of circle: {calculate_area_of_circle( 7 )} " )`
4. Assess Copilot's suggestion and, if you agree with the change:
    - **In VS Code and Visual Studio:** Click **Accept** .
    - **In JetBrains:** Click the Preview icon (double arrows), then click the Apply All Diffs icon (double angle brackets).

As with all Copilot suggestions, you should always check that the revised code runs without errors and produces the correct result.

### Splitting up complex units of code

Large methods or functions that perform multiple operations are likely to offer fewer opportunities for reuse than smaller, simpler functions that are focused on performing a particular operation. They may also be more difficult to understand and debug.

Copilot can help you to split up complex blocks of code into smaller units that are more suitable for reuse.

The following Python code is a very simple example, but it shows the principle of splitting up a single function into two functions that perform particular operations.

```
import pandas as pd from pandas.io.formats.style import Styler def process_data ( item, price ): # Cleanse data item = item.strip() # Strip whitespace from item price = price.strip() # Strip whitespace from price price = float (price) # Convert price to a float # More cleansing operations here # Create and print a DataFrame data = { 'Item' : [item], 'Price' : [price]}
    df = pd.DataFrame(data) print (df.to_string(index= False )) # Example usage item = "   Apple " price = " 1.25" process_data(item, price)
```

To split up the `process_data` function:

1. Put the cursor in the function name.
2. Open inline chat:
    - **In VS Code:** Press `Command` + `i` (Mac) or `Ctrl` + `i` (Windows/Linux).
    - **In Visual Studio:** Press `Alt` + `/` .
    - **In JetBrains IDEs:** Press `Control` + `Shift` + `i` (Mac) or `Ctrl` + `Shift` + `g` (Windows/Linux).
3. Type: `split into 2 separate functions: one for cleansing data, the other for printing` and press `Enter` . Copilot suggests revised code. For example: `def cleanse_data ( item, price ): # Cleanse data item = item.strip() # Strip whitespace from item price = price.strip() # Strip whitespace from price price = float (price) # Convert price to a float return item, price def print_data ( item, price ): # Create and print a DataFrame data = { 'Item' : [item], 'Price' : [price]} df = pd.DataFrame(data) print (df.to_string(index= False )) def process_data ( item, price ): item, price = cleanse_data(item, price) print_data(item, price)` Note The example suggestion shown above replaces the existing function with three functions. The third function simply calls the other two functions, and allows for existing code that calls the `process_data` function.
4. Assess Copilot's suggestion and, if you agree with the change:
    - **In VS Code and Visual Studio:** Click **Accept** .
    - **In JetBrains:** Click the Preview icon (double arrows), then click the Apply All Diffs icon (double angle brackets).

As with all Copilot suggestions, you should always check that the revised code runs without errors and produces the correct result.

### Rewrite conditional code for better readability

There are often several ways to write code that does, or does not, get executed depending on various conditions. Some conditional structures are better suited than others to particular use cases, and choosing an alternative conditional structure can sometimes make the code easier to read.

This Java method uses a series of `if` and `else if` statements to determine which operation to perform:

```
public String getSound (String animal) { if (animal == null ) {
      System.out.println( "Oops! A null animal?" );
  } else if (animal.equalsIgnoreCase( "Dog" )) { return "Bark" ;
  } else if ( animal.equalsIgnoreCase( "Cat" )) { return "Meow" ;
  } else if ( animal.equalsIgnoreCase( "Bird" )) { return "Tweet" ;
  } return "Unknown" ;
}
```

A `switch` statement might be a better way of applying the same logic.

1. Put the cursor in the method name.
2. Open inline chat:
    - **In VS Code:** Press `Command` + `i` (Mac) or `Ctrl` + `i` (Windows/Linux).
    - **In Visual Studio:** Press `Alt` + `/` .
    - **In JetBrains IDEs:** Press `Control` + `Shift` + `i` (Mac) or `Ctrl` + `Shift` + `g` (Windows/Linux).
3. Type: `rewrite the condition to use a switch and use Java 21 syntax with null case, but also add documentation and provide a better function name` . This demonstrates how you can use a single prompt to ask Copilot to make multiple changes.
4. Press `Enter` . Copilot suggests revised code. For example: `/** * Retrieves the sound made by a given animal. * * @param animal The name of the animal. * @return The sound made by the animal, or "Unknown" if the animal is not recognized. */ public String getAnimalSound (String animal) { return switch (animal) { case null -> { System.out.println( "Oops! A null animal?" ); yield "Unknown" ; } case String a when a.equalsIgnoreCase( "Dog" ) -> "Bark" ; case String a when a.equalsIgnoreCase( "Cat" ) -> "Meow" ; case String a when a.equalsIgnoreCase( "Bird" ) -> "Tweet" ; default -> "Unknown" ; }; }`
5. Assess Copilot's suggestion and, if you agree with the change:
    - **In VS Code and Visual Studio:** Click **Accept** .
    - **In JetBrains:** Click the Preview icon (double arrows), then click the Apply All Diffs icon (double angle brackets).

As with all Copilot suggestions, you should always check that the revised code runs without errors and produces the correct result.

### Reformat code to use a different structure

Suppose you have this function in JavaScript:

```
function listRepos ( o, p ) { return fetch ( `https://api.github.com/orgs/ ${o} /repos?per_page= ${ parseInt (p)} ` )
   . then ( ( response ) => response. json ())
   . then ( ( data ) => data);
}
```

If your coding standards require you to use the arrow notation for functions, and descriptive names for parameters, you can use Copilot to help you make these changes.

1. Put the cursor in the function name.
2. Open inline chat:
    - **In VS Code:** Press `Command` + `i` (Mac) or `Ctrl` + `i` (Windows/Linux).
    - **In Visual Studio:** Press `Alt` + `/` .
    - **In JetBrains IDEs:** Press `Control` + `Shift` + `i` (Mac) or `Ctrl` + `Shift` + `g` (Windows/Linux).
3. Type: `use arrow notation and better parameter names` and press `Enter` . Copilot suggests revised code. For example: `const listRepos = ( org, perPage ) => { return fetch ( `https://api.github.com/orgs/ ${org} /repos?per_page= ${ parseInt (perPage)} ` ) . then ( response => response. json ()) . then ( data => data); };`

### Improving the name of a symbol

Note

- VS Code and Visual Studio only.
- Support for this feature depends on having the appropriate language extension installed in your IDE for the language you are using. Not all language extensions support this feature.

Well chosen names can help to make code easier to maintain. Copilot in VS Code and Visual Studio can suggest alternative names for symbols such as variables or functions.

1. Put the cursor in the symbol name.
2. Press `F2` .
3. **Visual Studio only:** Press `Ctrl` + `Space` . Copilot suggests alternative names.


5. In the dropdown list, select one of the suggested names. The name is changed throughout the project.


### Project overview

It's important to define what you want your product to do. In the planning phase of the software development lifecycle (SDLC), you turn ideas into actionable tasks by breaking down your project into epics, features, and smaller pieces of work. This helps you organize your thoughts, set priorities, and prepare your team for development.

When you use Copilot, you drive this process. Copilot can suggest a structure and fill in details, but the best results come when you have a sense of how you want the work to be organized. Copilot works with your input to help you refine, expand, and document your plan.

In this scenario you'll plan a new shopping website that will allow users to:

- Browse a product catalog with categories and search
- Add items to a shopping cart
- Complete secure checkouts

Your goal is to use Copilot to quickly turn this vision into a structured project plan, creating epics and detailed issues that capture each part of your site.

### Set up repository

Set up a repository with GitHub Issues enabled. See [Creating a new repository](/en/repositories/creating-and-managing-repositories/creating-a-new-repository) .

By default, issues are enabled for new repositories. If you would like to use an existing repository but don't see the **Issues** tab, follow these steps to enable issues:

1. From the repository, select **Settings** .
2. Under "Features", check the **Issues** box.

### Generate project issues

With the repository set up, you can use Copilot to turn your project vision into a set of actionable issues.

#### Start in Copilot in GitHub

1. Navigate to [https://github.com/copilot](https://github.com/copilot?ref_product=copilot&ref_type=engagement&ref_style=text) .
2. Using the chat panel, attach the repository for the shopping website. This allows Copilot to access the repository and create issues directly within it.

#### Create an epic issue

1. Enter a detailed project description as your prompt. For example: `I'm planning to create a shopping website in React and Node.js. The site should allow users to browse products by category, search for items, add products to a cart, and complete checkout. Please help me plan the project by creating issues and breaking it down into epics, features, and tasks.`
2. Submit your prompt. Copilot will generate an issue tree, typically with an epic at the top and sub-issues for each main feature or task


### Navigate the issue tree

1. Click the epic to view its details in the workbench. Navigate through the workbench to explore the issue tree.
2. Each issue typically includes a title and description. Additional metadata such as labels or assignees, can be edited directly in the workbench.
3. You can expand or collapse sub-issues to focus on specific parts of the project. The issue tree provides a clear overview of your project structure, making it easy to navigate between epics, features, and tasks.
4. In this first iteration of the draft, Copilot may generate only high-level issues. You can refine these issues further by breaking them down into smaller tasks or features. Let's refine the issue "Feature: UI Skeleton and Navigation". Prompt Copilot with: `Can you break down the issue "Feature: UI Skeleton and Navigation" into smaller tasks?` Copilot will generate multiple new sub-issues such as:
    - Task: Set up React project structure and initial files
    - Task: Create placeholder pages for main routes
    - Task: Implement site-wide navigation bar component
    - Task: Integrate navigation with routing
    - Task: Add basic responsive layout
5. Repeat this process for the remaining feature issues in the epic.


#### Improve issue descriptions

After you finish generating the issue tree you may notice that Copilot's issue descriptions may be brief or unclear. To make them actionable, refine each issue as needed.

1. Start with the newly generated issue such as "Task: Create placeholder pages for main routes". Prompt Copilot with: `Can you improve the description for "Task: Create placeholder pages for main routes"? Please provide a detailed technical summary, list the main routes to be included, outline the steps for implementation, and specify what should be delivered for this task. Please add any relevant code snippets.`
2. Copilot will generate a new version of the draft issue "Task: Create placeholder pages for main routes." At the top-left of the issue, click the versioning drop-down and select **Version 2** to review the new changes.
3. Review and decide whether to keep Copilot's revised version, edit further, or prompt again for more detail. Copilot can add code snippets into the draft to improve clarity and provide immediate context for these issues.
4. Repeat this process for other issues in the epic, refining descriptions and breaking down tasks as needed.
5. Once you're satisfied with the issue descriptions, click **Create all** to create the issues in your repository.

### Unlink issues

If Copilot generates a sub-issue that doesn't belong to the issue tree, you can unlink it from the issue tree.

1. In the workbench issue tree, click next to the sub-issue, then click **Unlink sub-issue** .
2. The issue will be unlinked from its parent and will no longer appear under that epic in the tree.

### Next steps

Now that you've generated and refined your project issues, you can assign them to the right team members or even to Copilot itself for further assistance. To learn more about how to assign Copilot or contributors to issues, and how to continue planning and implementing your project with Copilot's agentic features, see [Asking GitHub Copilot to create a pull request](/en/copilot/how-tos/use-copilot-agents/cloud-agent/create-a-pr) .

### Further reading

- [Using GitHub Copilot to create or update issues](/en/copilot/how-tos/copilot-on-github/copilot-for-github-tasks/use-copilot-to-create-or-update-issues)
- [Piloting GitHub Copilot cloud agent in your organization](/en/copilot/tutorials/cloud-agent/pilot-cloud-agent)
- [Best practices for using GitHub Copilot to work on tasks](/en/copilot/tutorials/cloud-agent/get-the-best-results)
- [Speeding up development work with GitHub Copilot Spaces](/en/copilot/tutorials/speed-up-development-work)


### Introduction

Copilot allows you to create a whole new application from scratch, add features, or alter the user interface, without writing a line of code yourself. You can work with Copilot, entering prompts in the chat view-using the AI as your coding partner-and leave all of the actual coding to Copilot.

In this tutorial you'll work this way, in VS Code or in a JetBrains IDE, to create a personal time-tracking web app.

This method works well for developing a proof of concept, creating a draft of an application that you'll  develop further using a more conventional approach to software development, or creating applications for your own personal use.

Note

The responses shown in this article are examples. Copilot Chat responses are non-deterministic, so you may get different responses from the ones shown here.

### Who is this tutorial for?

- **Learner:** You're learning how to create software applications. You can learn a lot from working with Copilot and seeing how it implements your requests.
- **Non-developer:** You're a product manager, or working in another role outside of an engineering team. You want to quickly create a proof of concept application to demonstrate some particular functionality. You're mainly concerned with the user experience, rather than the details of the code.
- **Individual:** You want to create an application to provide some useful functionality to help you in your daily work or home life. The application will run locally on your computer, and only you will use it, so you're not overly concerned about how the code was put together.

This tutorial is not intended for experienced developers with an established practice of writing code in an editor. Experienced developers will use Copilot in a different way-as tool for problem solving and increased productivity. In this tutorial, we'll work within chat and leave Copilot to do all the work in the editor.

### How long will this take?

There are many variables that may affect how long you might take to complete this tutorial. However, you should allow for at least two hours. At any time you can return to it later, picking up from where you left off in the same conversation in Copilot Chat.

### Prerequisites

Before getting started you must have the following:

- A [GitHub Copilot subscription plan](/en/copilot/about-github-copilot/subscription-plans-for-github-copilot) .
- One of these IDEs:
    - Visual Studio Code
    - Any JetBrains IDE that supports Copilot, with the GitHub Copilot extension for JetBrains installed. See [Installing the GitHub Copilot extension in your environment](/en/copilot/how-tos/set-up/install-copilot-extension?tool=jetbrains) .
- Some experience of using Copilot Chat in either Visual Studio Code or JetBrains. If you've never used Copilot Chat before, see [Asking GitHub Copilot questions in your IDE](/en/copilot/how-tos/chat-with-copilot/chat-in-ide) .

### Preparation

We'll create a time-tracking app in a new GitHub repository.

1. In the GitHub website, create a new private repository for your application, including an initial README file. See [Creating a new repository](/en/repositories/creating-and-managing-repositories/creating-a-new-repository) .
2. Clone a copy of the repository to your local machine. See [Cloning a repository](/en/repositories/creating-and-managing-repositories/cloning-a-repository) .
3. In your local copy of the repository, create a new branch to work in. For example, in a terminal, use the command: Bash `git checkout -b BRANCH-NAME`

### Researching with Copilot

1. In VS Code, or your JetBrains IDE, open the repository directory as a new project or workspace.
2. Close any editor tabs that are currently open. Working in an empty project or workspace, with no editor tabs open, prevents Copilot being influenced by any code or information in those tabs.
3. Open a terminal window in the IDE.
4. Open Copilot Chat and check, at the bottom of the chat view, that **Ask** is the currently selected chat mode. If it is not, select **Ask** from the chat mode dropdown.


6. Choose a model from the models dropdown. Note The responses referred to in this tutorial were received while using Claude Sonnet 4.5. Other models will respond differently, but you should get roughly similar results. Claude Sonnet 4.5 is a good choice, if it's available, as it provides useful commentary in the chat view, explaining what it is doing, and giving detailed summaries when it has finished coding. If Claude Sonnet 4.5 is not available, set the model to **Auto** or select a model of your choice.
7. Enter this prompt in the chat: Copilot prompt `I need to keep a daily log of what I've spent my time on. I want to build a time-tracking application to help me do this. Throughout the day I want to use the app to record what I'm working on as I move between tasks. At the end of the day it should show me the total time I've spent on each item. What are the typical features of such an app? What do I need to consider when building this app?` Copilot responds with details to answer your questions.
8. Consider Copilot's response and ask for more information, as required, to clarify your thoughts about your application. For example, you might ask: Copilot prompt `Data storage: the application will run locally on my laptop. What's the best way to persistently store data so that I'll have access to historic time-tracking data? Data structure: How should I structure the data for this application? There will be tasks and projects. Each chunk of time will be associated with a task and some, but not all, tasks will be associated with projects. I will want to see totals for: each task, each project, each task per project.`
9. Continue to ask questions in the same chat conversation, to build up a clearer idea of the application you want to build. Keep your chat conversation open, as Copilot will use this in the next series of steps.

### Planning the implementation

You can now start planning for an initial implementation of your application. It's a good idea to begin with a basic version of the application which you can iterate on. This makes it easier to get the fundamental functionality working, before adding features.

1. In the same Copilot Chat conversation you used in the previous section, switch from ask mode to plan mode by selecting **Plan** from the chat mode dropdown at the bottom of the chat view.
2. Enter this prompt: Copilot prompt `I want to build a time-tracking application that allows me to keep track of how much time I spend on tasks during my working day. This should be a web app that runs locally on my computer. Plan the implementation of a basic, initial version of this application. This first version should allow me to: - Add, edit and delete projects and tasks - each a name with a maximum of about 50 characters - Quickly click to select a project and task and record the start time - Click another task to stop the current timer, recording the stop time, and recording the start time for the new task - Pause/resume/end the current task - Display the totals of times I have spent on each: task, project, and task per project. Time is always recorded for a specific task. A task can optionally be associated with a project. Store data for each day, but for this version do not include any user interface or functionality for looking at historical data, or compiling statistics. The initial version of the application should be limited to today's time tracking. Notes: - Tasks never overlap - Time should be accurate to the minute by recording the day, hour, minute tasks are started and stopped and calculating the duration from this - Design the web UI for display on a desktop monitor - Keep things very simple for the initial version. Do not add any other features not mentioned in this prompt` Copilot replies with something like: [Plan: Build Time-Tracking Web Application](#plan-build-time-tracking-web-application) A single-page web application for tracking daily task time using vanilla HTML/CSS/JavaScript with localStorage. The app displays projects/tasks in a sidebar, shows an active timer, and calculates daily totals by task and project. Steps Further Considerations
3. Answer the "Further Considerations" questions Copilot raised. For example, you could respond with this prompt: Copilot prompt `- Date handling: only consider the local date. This app is only going to be used by one person in one timezone. Set the day boundary to 4 am. - Time display: Show elapsed time as HH:MM. Don't track seconds. - Unassigned tasks: I've changed my mind. All tasks should be associated with a project, but there should be a built-in project called "No Project" (which the user can't delete or rename). All tasks should be associated with this project until the user chooses another project. If the user is changing tasks for the same project then they should be able to do this with one click (assuming the new task has already been defined). If they are doing the same task but for a different project, this should also be possible to change with one click (assuming the new project is already defined). If they want to track time for a different task in a different project then they should be able to do this with 2 clicks.` Copilot may respond with further questions for your consideration.
4. You can answer some or all of the questions, or, if you feel that the plan has enough detail, you can skip to the next stage. Copilot's responses are non-deterministic, so the questions it asks will vary, but let's assume its response included these questions: **New task default project** - When adding a new task without specifying project, assign to "No Project" or to currently active/selected project? Recommend: Currently selected/active project with "No Project" as fallback. **Timer precision edge case** - If user switches tasks within the same minute (e.g., 10:30:15 to 10:30:45), should this create a 0-minute entry or be ignored? Recommend: Ignore and treat as immediate switch without recording. You might decide to respond to these questions by entering the prompt: Copilot prompt `New task default project - When adding a new task without actively specifying a project, use the currently selected/active project with "No Project" as the default when the user has not actively selected any other project. Timer precision edge case - If user switches or ends tasks within the same minute as the start time then delete this entry. Only time entries of more than one minute should be recorded.`
5. Continue iterating if you feel there are further questions that need answered.
6. Keep your chat conversation open, as Copilot will use this in the next series of steps.

### Building your application with Copilot cloud agent

When you think the plan contains enough detail:

1. Click **Start Implementation** in the Copilot Chat view. Depending on your IDE this will either start the agent immediately, or it will add "Start Implementation" as a prompt, which you should then submit. Notice that chat mode switches from "Plan" to "Agent".
2. Copilot will request your permission to perform actions such as editing sensitive files, running commands, or adding files to Git. Copilot will begin to implement an initial version of your application.
    - **In VS Code:** click **Allow** . Alternatively, click the arrow on the **Allow** button and click **Allow in this Session** in the dropdown menu.


    - **In JetBrains:**
        - When asked if you want to add a file to Git, select the **Don't ask again** checkbox, then click **Add** .
        - When asked about running a command, click **Continue** .


5. If Copilot finishes its response without completing the installation, or if Copilot appears to have stalled, take the following remedial actions, as required:
    - **Missing component** If Copilot says it cannot proceed because a required component needs to be installed (for example, Node.js), you can enter a prompt asking Copilot to download and install the missing component.
    - **Process is taking a long time** Some steps may take several minutes to complete. Be patient and allow the agent to complete each part of the process. An animated spinner icon indicates that the agent is currently working on a command. For example, in JetBrains IDEs:


    - **Input required** Occasionally Copilot will run a command that requires some manual input. If Copilot appears to have stalled, check the IDE's terminal window to see if a command requires action from you. Copilot will wait for you to enter a response in the terminal before continuing.
    - **Error messages** If you get any error messages while developing the application, copy the error message into the chat prompt box and ask Copilot to fix the problem. Note You may have to iterate with Copilot in this way, asking it to debug and fix problems, until it has a working application that you can view in your browser.
    - **Copilot's response appears to be stuck** If the spinner icon is displayed in a response but, after waiting for several minutes-and having checked that your input is not required in the terminal-nothing is happening, you can stop and restart the conversation. Click the cancel button at the bottom of the chat view. For example, in VS Code: Then enter the prompt: Copilot prompt `Your previous response stalled. Try again, picking up from where you left off.`


8. Typically, towards the end of the coding process, Copilot will request your permission to open an untrusted web page for the application: Give your permission for this.


10. When the agent finishes work on the application it will display a summary of what it built in the chat panel. Typically it will provide a link to the running application. The time tracker application may also be displayed in a browser tab in your IDE. This can be useful for confirming that the page is available. However, you should always check the application in your default browser to verify a realistic user experience. Click the link in the chat panel to open the application in your default browser. If a link isn't displayed in Copilot's chat response, wait a few minutes as the agent may be working on deploying the application. If a link is still not displayed you can prompt Copilot to display one: Copilot prompt `Confirm the implementation is complete. If so, give me a link to the running application.`
11. If the final message from Copilot gives you instructions for starting the application (such as running `npm start` ), rather than supplying a link to the running application, you can ask Copilot to run the command for you and check that it completed successfully. For example: Copilot prompt `Run npm start for me and confirm everything is working` If Copilot isn't able to run all of the commands itself, it will provide you with commands that you can copy and paste into the terminal.

### Testing your application

1. View your application's web page. Below are some examples of a time tracking application created by Copilot: Note The application that Copilot generates for you might look quite different to the examples shown above.


4. Try using the application. Add a couple of projects and a selection of tasks, then start tracking time on a task.
5. As you try out this first draft of the application, make a note of the two or three most important things that need to be changed. In subsequent steps you'll work on fixing these. For now, don't spend time noting down everything you want to change. Just identify the most pressing things that need to be fixed first. You'll have time to get everything working and looking the way you want it to later in the process.
6. If the application doesn't load, or an error is displayed, describe the problem in the chat prompt box, copying and pasting any error messages, and ask Copilot to debug and fix the problem.
7. After you have reviewed the initial draft of the application and established that it runs in at least a rudimentary fashion, return to your IDE.
8. ***Optional*** *: if you're familiar with the type of code Copilot is writing for you.* Display the files that Copilot has changed in the editor and review the changes. You can make your own changes if required.
9. Click **Keep** (in VS Code) or **Accept All** (in JetBrains IDEs), in the Copilot Chat view, to accept the changes and remove the diff lines from the editor. You now have a base version of your application that you can iterate on to improve and extend the functionality and user interface.
10. Commit the changes to Git. It's always a good idea to commit changes at each successful iteration so you can easily return to a previous version if you decide you don't like some changes that Copilot has made for you.
11. Close any open editor tabs, but keep the Copilot Chat view open as you'll continue working in the same chat conversation.

### Iterating on changes

1. After committing the initial version to git you can make a change to the application, fixing one of the things you noted when you reviewed the site. For example, the implementation may have tied tasks to project, so that a task created for one project doesn't show up when you select another project. To change this-allowing you to create tasks that can be used for any project-enter another prompt into the same conversation, while still in agent mode for Copilot. You could use a prompt such as: Copilot prompt `Tasks should not be tied to projects in the user interface, as they currently are. When the user selects a project, allow them to choose any currently defined task. There should be a many-to-many relationship between projects and tasks.`
2. Again, Copilot is likely to ask you to approve changes it needs to make to the code. Click **Approve** or **Continue** .
3. Once Copilot completes the change, return to your browser and refresh the page.
4. Review the revised application and tell Copilot if the change was not implemented correctly. You may spot more than one thing that needs fixed, but to allow Copilot to focus on one thing at a time, you should restrict each prompt to a single task and iterate on this in a series of prompts and responses, as necessary, until the problem is fixed. Then, move on to the next thing you want to change.
5. If there's a problem with the layout of the web page-for example, overlapping or badly aligned elements-you can take a screenshot, paste it into the chat and enter a prompt such as: Copilot prompt `This part of the web page looks bad. Fix it.`
6. After a change is implemented satisfactorily, click **Keep** or **Accept All** and commit the changes.
7. ***Optional*** *:* Depending on your working practices-for example, if you are working in a development team-you may decide at this point, and after each significant change, to raise a pull request. This will allow you to have changes reviewed and merged into the default branch of the repository so that other people can work on the code.
8. Continue iterating on your application. For example, you might want to give the app a different style of user interface. In this case, still in agent mode, you could prompt Copilot: Copilot prompt `I don't like the look of the user interface. Suggest some alternative web UI libraries I could choose to give the app a more formal, business-like appearance.` Copilot will list some UI libraries.
9. Choose one of the libraries and ask Copilot to use it. For example: Copilot prompt `Alter the user interface to use Bootstrap 5. I want the app to look like a professionally designed business application.` Copilot will rework the application to use your chosen user interface library.
10. Check the results and commit the changes if you are happy with the revised look of the application.
11. A common requirement for a time tracker application is the ability to output your timesheet. For example, if you are a contractor, you might be required to submit a timesheet along with your invoice. So let's add the ability to generate a PDF. Use this prompt: Copilot prompt `Add a button to the user interface which generates a PDF timesheet for the work the user has tracked today. The timesheet should show the total time spent on task for each of the defined project for which data has been recorded today. Under this show the total time spent on each project. Then show the total time spent on each task irrespective of projects. Finally show a chronological lists of tasks performed during the day with the time spent on each.`

#### Example application

After working with Copilot to build, extend and improve your time tracker, the application might look something like this:

Screenshot an example of a time tracker app with data added by a user.


### Improving your software project

1. Copilot can help you make your project more robust by adding and running tests. Tests help to prevent bugs from getting into the codebase. You could prompt Copilot in agent mode: Copilot prompt `Add a comprehensive suite of tests for this application. These should include unit tests, integration tests, component tests, database tests, and end-to-end tests. Locate the tests in a `tests` directory. Run the tests and fix any problems that are identified.`
2. It's always a good idea to have a README file in your project. The README should provide an overview of the project and give instructions for using the application. You can ask Copilot to create or update the README file: Copilot prompt `Add or update a README.md file. This should provide an introduction to the application, describing its primary use and highlighting its features. It should give easy to follow user instructions for using the application in the browser. It should provide admin instructions, explaining how to deploy the application. Finally it should give an overview of the technologies used to build the application and some basic information for developers on how to maintain the code and extend the application.`
3. Now that you've added the initial code for the application to the repository, you should add a custom instructions file for Copilot. The custom instructions file improves Copilot's responses in a repository by providing repository-specific guidance and implementation preferences. To add a custom instructions file:
    - **In VS Code:** click the "Configure Chat" cog icon, at the top of the chat view, and click **Generate Chat Instructions** .
    - **In JetBrains IDEs:** in agent mode of Copilot Chat, submit a prompt such as: Copilot prompt `Analyze this codebase and create or update `.github/copilot-instructions.md` to guide AI agents. Discover essential knowledge for immediate productivity: - Architecture: major components, service boundaries, data flows, and structural decisions - Developer workflows: builds, tests, debugging commands - Project conventions that differ from common practices - Integration points and cross-component communication Guidelines: - Merge intelligently if file exists - Write ~20-50 concise lines with markdown structure - Include specific codebase examples - Focus on THIS project's approaches, not generic advice - Document discoverable patterns, not aspirational patterns - Reference key files/directories that exemplify important patterns`
4. Review the instructions file. If you think the instructions need some more details, you can add these to the file manually and save it. For example, you could add an instruction about running tests, if the file does not already contain an instruction about this: Text `## Running tests Always run the complete test suite after completing a batch of code changes, to ensure the changes do not break or adversely affect any part of the application. Fix any test failures and then run the tests again to verify the fix.`

### Next steps

- Continue iterating on this project, making improvements to your time tracker.
- Using the same methodology, create another application.
- Find out about another way you can create applications without writing the code yourself. See [About GitHub Spark](/en/copilot/concepts/spark) .


### Prerequisites

- Familiarity with shell scripting (Bash or PowerShell)
- Basic understanding of JSON configuration files
- Access to a repository where Copilot CLI is used
- `jq` installed (for the Bash examples)

### 1. Define an organizational policy

Before you write any hook scripts, decide which actions should be allowed automatically and which should require human review.

A clear policy helps you avoid over-blocking while still reducing risk.

#### Identify commands that always require review

Start by identifying patterns that should never be auto-executed by Copilot CLI. Common examples include:

- **Privilege escalation** : `sudo` , `su` , `runas`
- **Destructive system operations** : `rm -rf /` , `mkfs` , `dd` , `format`
- **Download-and-execute patterns** : `curl ... | bash` , `wget ... | sh` , PowerShell `iex (irm ...)`

These commands can have irreversible effects if executed unintentionally.

#### Decide what to log

When you use hooks, you can capture information about how Copilot CLI is used in a repository, including prompts submitted by users and tools that Copilot CLI attempts to run.

At minimum, most organizations log:

- The timestamp and repository path
- The prompt text (or a redacted form)
- The tool name and tool arguments
- Any policy decision (for example, a denied command and its reason)

Avoid logging secrets or credentials. If prompts or commands may contain sensitive data, apply redaction before writing logs.

This tutorial uses a local `.github/hooks/logs` directory as a simple, illustrative example. These log files are **not intended to be committed to the repository** and typically live only on a developer's machine.

In production environments, many organizations forward hook events to a centralized logging or observability system instead of writing logs locally. This allows teams to apply consistent redaction, access controls, retention policies, and monitoring across repositories and users.

#### Align with stakeholders

Before enforcing policies, review them with:

- Security or compliance teams, to confirm risk boundaries
- Platform or infrastructure teams, who may need broader permissions
- Development teams, so they understand what will be blocked and why

Clear expectations make policy enforcement easier to adopt and maintain.

### 2. Set up repository hook files

Throughout this tutorial, you'll use **repository-scoped hooks** stored in the repository under `.github/hooks/` . These hooks apply whenever Copilot CLI runs from within this repository.

Note

Copilot agents load hook configuration files from `.github/hooks/*.json` in the repository. Hooks run synchronously and can block execution.

#### Create the directory structure

From the repository root, create directories for your hook configuration, scripts, and logs:

Bash

```
mkdir -p .github/hooks/scripts mkdir -p .github/hooks/logs
```

Add `.github/hooks/logs/` to .gitignore so local audit logs aren't committed:

Bash

```
echo ".github/hooks/logs/" >> .gitignore
```

This tutorial uses the following structure:

```
.github/
└── hooks/
    ├── copilot-cli-policy.json
    ├── logs/
    │   └── audit.jsonl
    └── scripts/
        ├── session-banner.sh
        ├── session-banner.ps1
        ├── log-prompt.sh
        ├── log-prompt.ps1
        ├── pre-tool-policy.sh
        └── pre-tool-policy.ps1
```

#### Create a hook configuration file

Create a hook configuration file at `.github/hooks/copilot-cli-policy.json` .

This file defines which hooks run, when they run, and which scripts they execute.

JSON

```
{ "version" : 1 , "hooks" : { "sessionStart" : [ { "type" : "command" , "bash" : "./scripts/session-banner.sh" , "powershell" : "./scripts/session-banner.ps1" , "cwd" : ".github/hooks" , "timeoutSec" : 10 } ] , "userPromptSubmitted" : [ { "type" : "command" , "bash" : "./scripts/log-prompt.sh" , "powershell" : "./scripts/log-prompt.ps1" , "cwd" : ".github/hooks" , "timeoutSec" : 10 } ] , "preToolUse" : [ { "type" : "command" , "bash" : "./scripts/pre-tool-policy.sh" , "powershell" : "./scripts/pre-tool-policy.ps1" , "cwd" : ".github/hooks" , "timeoutSec" : 15 } ] }
}
```

#### Understand what this configuration does

This configuration sets up three hooks:

- `sessionStart` : Shows an informational message when a new agent session starts or resumes.
- `userPromptSubmitted` : Runs whenever a user submits a prompt.
- `preToolUse` : Runs before a tool executes and can explicitly allow or deny execution.

#### Commit and share the hook configuration

When you're ready to share the hook configuration with collaborators (for example, via a pull request or in a test repository), commit the hook configuration and scripts. Don't commit any local audit logs.

Bash

```
git add .github/hooks/copilot-cli-policy.json .github/hooks/scripts
git commit -m "Add Copilot CLI hook configuration" git push
```

At this point, Copilot CLI can discover your hook configuration, even though you haven't created the hook scripts yet.

### 3. Add a policy banner at session start

Use a `sessionStart` hook to display a banner whenever a new Copilot CLI session starts or resumes. This makes it clear to developers that organizational policies are active.

The `sessionStart` hook receives contextual information such as the current working directory and the initial prompt. Any output from this hook is ignored by Copilot CLI, which makes it suitable for informational messages.

#### Create the session banner script (Bash)

Create `.github/hooks/scripts/session-banner.sh` :

Bash

```
#!/bin/bash
set -euo pipefail cat << 'EOF' COPILOT CLI POLICY ACTIVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Prompts and tool use may be logged for auditing
• High-risk commands may be blocked automatically
• If something is blocked, follow the guidance shown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF exit 0
```

#### Create the session banner script (PowerShell)

Create `.github/hooks/scripts/session-banner.ps1` :

PowerShell

```
$ErrorActionPreference = "Stop"
Write-Host @"
COPILOT CLI POLICY ACTIVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Prompts and tool use may be logged for auditing
• High-risk commands may be blocked automatically
• If something is blocked, follow the guidance shown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"@
exit 0
```

#### Test the session banner

You can test the banner scripts directly:

```
.github/hooks/scripts/session-banner.sh # or, for PowerShell .github/hooks/scripts/session-banner.ps1
```

When you run either script, you should see the policy banner displayed in your terminal.

### 4. Log prompts for auditing

Use the `userPromptSubmitted` hook to record when users submit prompts to Copilot CLI. This hook runs whenever a prompt is sent, before any tools are invoked.

The hook receives structured JSON input that includes the timestamp, current working directory, and full prompt text. The output of this hook is ignored.

Important

Prompts may contain sensitive information. Apply redaction and follow your organization's data handling and retention policies when logging this data.

#### Create the prompt logging script (Bash)

Create `.github/hooks/scripts/log-prompt.sh` :

Bash

```
#!/bin/bash
set -euo pipefail

INPUT= " $(cat) " TIMESTAMP_MS= " $(echo " $INPUT " | jq -r '.timestamp // empty') " CWD= " $(echo " $INPUT " | jq -r '.cwd // empty') "
### This example logs only metadata, not the full prompt, to avoid storing
### potentially sensitive data. Adjust to match your organization's needs. LOG_DIR= ".github/hooks/logs"
mkdir -p " $LOG_DIR "
chmod 700 " $LOG_DIR " jq -n \
  --arg ts " $TIMESTAMP_MS " \
  --arg cwd " $CWD " \ '{event:"userPromptSubmitted", timestampMs:$ts, cwd:$cwd}' \
  >> " $LOG_DIR /audit.jsonl"
exit 0
```

#### Create the prompt logging script (PowerShell)

Create `.github/hooks/scripts/log-prompt.ps1` :

PowerShell

```
$ErrorActionPreference = "Stop"
$inputObj = [ Console ]::In.ReadToEnd() | ConvertFrom-Json
$timestampMs = $inputObj .timestamp $cwd = $inputObj .cwd $prompt = $inputObj .prompt # Optional example redaction. Adjust to match your organization's needs.
$redactedPrompt = $prompt -replace 'ghp_[A-Za-z0-9]{20,}' , '[REDACTED_TOKEN]'
$logDir = ".github/hooks/logs"
if ( -not ( Test-Path $logDir )) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null } $logEntry = @ {
  event       = "userPromptSubmitted" timestampMs = $timestampMs cwd         = $cwd prompt      = $redactedPrompt } | ConvertTo-Json -Compress
Add-Content -Path " $logDir /audit.jsonl" -Value $logEntry
exit 0
```

#### Test the prompt logging script

You can test the scripts directly by piping example input.

```
echo '{"timestamp":1704614500000,"cwd":"/repo","prompt":"List all branches"}' \
  | .github/hooks/scripts/log-prompt.sh # or, for PowerShell
echo '{"timestamp":1704614500000,"cwd":"/repo","prompt":"List all branches"}' |
  .github/hooks/scripts/log-prompt.ps1
```

After running the script, check `.github/hooks/logs/audit.jsonl` for a new log entry.

Bash

```
cat .github/hooks/logs/audit.jsonl
```

At this point, prompts submitted to Copilot CLI in this repository are recorded for auditing.

### 5. Enforce policies with preToolUse

Use the `preToolUse` hook to evaluate a tool call **before it runs** . This hook can allow execution (by doing nothing) or deny execution (by returning a structured response).

#### Understand the preToolUse input

The `preToolUse` hook input includes:

- `toolName` : The tool that Copilot CLI is about to run (for example, `bash` )
- `toolArgs` : A **JSON string** containing that tool's arguments

Because `toolArgs` is a JSON string, your script must parse it before reading fields like `command` .

Important

Tool arguments and commands may contain sensitive information such as API tokens, passwords, or other credentials. Apply redaction before logging this data and follow your organization's security policies. Consider logging only non-sensitive metadata (tool name, timestamp, policy decision) and directing audit events to a secured, centralized logging system with appropriate access controls and retention policies.

#### Create the policy script

Next, create a policy script. This example:

- Logs all attempted tool usage.
- Applies deny rules only to bash commands.
- Blocks high-risk patterns such as privilege escalation, destructive operations, and download-and-execute commands.

To let you validate the deny flow safely, the script also includes a temporary demo rule that blocks a harmless test command. After confirming that hooks work as expected, remove the demo rule and replace it with patterns that reflect your organization's policies.

##### Example script (Bash)

Create `.github/hooks/scripts/pre-tool-policy.sh` :

Bash

```
#!/bin/bash
set -euo pipefail

INPUT= " $(cat) " TOOL_NAME= " $(echo " $INPUT " | jq -r '.toolName // empty') " TOOL_ARGS_RAW= " $(echo " $INPUT " | jq -r '.toolArgs // empty') " # JSON string LOG_DIR= ".github/hooks/logs"
mkdir -p " $LOG_DIR "
### Example redaction logic.
### GitHub does not currently provide built-in secret redaction for hooks.
### This example shows one possible approach; many organizations prefer to
### forward events to a centralized logging system that handles redaction.
### Redact sensitive patterns before logging.
### Adjust these patterns to match your organization's needs. REDACTED_TOOL_ARGS= " $(echo " $TOOL_ARGS_RAW " | \
  sed -E 's/ghp_[A-Za-z0-9]{20,}/[REDACTED_TOKEN]/g' | \
  sed -E 's/gho_[A-Za-z0-9]{20,}/[REDACTED_TOKEN]/g' | \
  sed -E 's/ghu_[A-Za-z0-9]{20,}/[REDACTED_TOKEN]/g' | \
  sed -E 's/ghs_[A-Za-z0-9]{20,}/[REDACTED_TOKEN]/g' | \
  sed -E 's/Bearer [A-Za-z0-9_\-\.]+/Bearer [REDACTED]/g' | \
  sed -E 's/--password[= ][^ ]+/--password=[REDACTED]/g' | \
  sed -E 's/--token[= ][^ ]+/--token=[REDACTED]/g') "
### Log attempted tool use with redacted toolArgs. jq -n \
  --arg tool " $TOOL_NAME " \
  --arg toolArgs " $REDACTED_TOOL_ARGS " \ '{event:"preToolUse", toolName:$tool, toolArgs:$toolArgs}' \
  >> " $LOG_DIR /audit.jsonl"
### Only enforce command rules for bash.
if [ " $TOOL_NAME " != "bash" ]; then exit 0 fi
### Parse toolArgs JSON string.
### If toolArgs isn't valid JSON for some reason, allow (and rely on logs).
if ! echo " $TOOL_ARGS_RAW " | jq -e . >/dev/null 2>&1; then exit 0 fi COMMAND= " $(echo " $TOOL_ARGS_RAW " | jq -r '.command // empty') "
### ---------------------------------------------------------------------------
### Demo-only deny rule for safe testing.
### This blocks a harmless test command so you can validate the deny flow.
### Remove this rule after confirming your hooks work as expected.
### ---------------------------------------------------------------------------
if echo " $COMMAND " | grep -q "COPILOT_HOOKS_DENY_DEMO" ; then deny "Blocked demo command (test rule). Remove this rule after validating hooks."
fi
deny () { local reason= " $1 " # Redact sensitive patterns from command before logging. local redacted_cmd= " $(echo " $COMMAND " | \
    sed -E 's/ghp_[A-Za-z0-9]{20,}/[REDACTED_TOKEN]/g' | \
    sed -E 's/gho_[A-Za-z0-9]{20,}/[REDACTED_TOKEN]/g' | \
    sed -E 's/ghu_[A-Za-z0-9]{20,}/[REDACTED_TOKEN]/g' | \
    sed -E 's/ghs_[A-Za-z0-9]{20,}/[REDACTED_TOKEN]/g' | \
    sed -E 's/Bearer [A-Za-z0-9_\-\.]+/Bearer [REDACTED]/g' | \
    sed -E 's/--password[= ][^ ]+/--password=[REDACTED]/g' | \
    sed -E 's/--token[= ][^ ]+/--token=[REDACTED]/g') " # Log the denial decision with redacted command. jq -n \
    --arg cmd " $redacted_cmd " \
    --arg r " $reason " \ '{event:"policyDeny", toolName:"bash", command:$cmd, reason:$r}' \
    >> " $LOG_DIR /audit.jsonl" # Return a denial response. jq -n \
    --arg r " $reason " \ '{permissionDecision:"deny", permissionDecisionReason:$r}' exit 0
} # Privilege escalation
if echo " $COMMAND " | grep -qE '\b(sudo|su|runas)\b' ; then deny "Privilege escalation requires manual approval."
fi
### Destructive filesystem operations targeting root
if echo " $COMMAND " | grep -qE 'rm\s+-rf\s*/($|\s)|rm\s+.*-rf\s*/($|\s)' ; then deny "Destructive operations targeting the filesystem root require manual approval."
fi
### System-level destructive operations
if echo " $COMMAND " | grep -qE '\b(mkfs|dd|format)\b' ; then deny "System-level destructive operations are not allowed via automated execution."
fi
### Download-and-execute patterns
if echo " $COMMAND " | grep -qE 'curl.*\|\s*(bash|sh)|wget.*\|\s*(bash|sh)' ; then deny "Download-and-execute patterns require manual approval."
fi
### Allow by default
exit 0
```

##### Create the policy script (PowerShell)

Create `.github/hooks/scripts/pre-tool-policy.ps1` :

PowerShell

```
$ErrorActionPreference = "Stop"
$inputObj = [ Console ]::In.ReadToEnd() | ConvertFrom-Json
$toolName = $inputObj .toolName $toolArgsRaw = $inputObj .toolArgs # JSON string
$logDir = ".github/hooks/logs"
if ( -not ( Test-Path $logDir )) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null } # Example redaction logic.
### GitHub does not currently provide built-in secret redaction for hooks.
### This example shows one possible approach; many organizations prefer to
### forward events to a centralized logging system that handles redaction.
### Redact sensitive patterns before logging.
### Adjust these patterns to match your organization's needs.
$redactedToolArgs = $toolArgsRaw ` -replace 'ghp_[A-Za-z0-9]{20,}' , '[REDACTED_TOKEN]' ` -replace 'gho_[A-Za-z0-9]{20,}' , '[REDACTED_TOKEN]' ` -replace 'ghu_[A-Za-z0-9]{20,}' , '[REDACTED_TOKEN]' ` -replace 'ghs_[A-Za-z0-9]{20,}' , '[REDACTED_TOKEN]' ` -replace 'Bearer [A-Za-z0-9_\-\.]+' , 'Bearer [REDACTED]' ` -replace '--password[= ][^ ]+' , '--password=[REDACTED]' ` -replace '--token[= ][^ ]+' , '--token=[REDACTED]'
### Log attempted tool use with redacted toolArgs. ( @ {
  event    = "preToolUse" toolName = $toolName toolArgs = $redactedToolArgs } | ConvertTo-Json -Compress ) | Add-Content -Path " $logDir /audit.jsonl"
if ( $toolName -ne "bash" ) { exit 0 } # Parse toolArgs JSON string.
$toolArgs = $null
try { $toolArgs = $toolArgsRaw | ConvertFrom-Json } catch { exit 0 } $command = $toolArgs .command # ---------------------------------------------------------------------------
### Demo-only deny rule for safe testing.
### This blocks a harmless test command so you can validate the deny flow.
### Remove this rule after confirming your hooks work as expected.
### ---------------------------------------------------------------------------
if ( $command -match 'COPILOT_HOOKS_DENY_DEMO' ) {
  Deny "Blocked demo command (test rule). Remove this rule after validating hooks." } function Deny ([string] $reason ) { # Redact sensitive patterns from command before logging. $redactedCommand = $command ` -replace 'ghp_[A-Za-z0-9]{20,}' , '[REDACTED_TOKEN]' ` -replace 'gho_[A-Za-z0-9]{20,}' , '[REDACTED_TOKEN]' ` -replace 'ghu_[A-Za-z0-9]{20,}' , '[REDACTED_TOKEN]' ` -replace 'ghs_[A-Za-z0-9]{20,}' , '[REDACTED_TOKEN]' ` -replace 'Bearer [A-Za-z0-9_\-\.]+' , 'Bearer [REDACTED]' ` -replace '--password[= ][^ ]+' , '--password=[REDACTED]' ` -replace '--token[= ][^ ]+' , '--token=[REDACTED]' # Log the denial decision with redacted command. ( @ {
    event    = "policyDeny" toolName = "bash" command  = $redactedCommand reason   = $reason } | ConvertTo-Json -Compress ) | Add-Content -Path " $logDir /audit.jsonl" ( @ {
    permissionDecision = "deny" permissionDecisionReason = $reason } | ConvertTo-Json -Compress ) exit 0 } if ( $command -match '\b(sudo|su|runas)\b' ) { Deny "Privilege escalation requires manual approval." } if ( $command -match 'rm\s+-rf\s*/(\s|$)|rm\s+.*-rf\s*/(\s|$)' ) { Deny "Destructive operations targeting the filesystem root require manual approval." } if ( $command -match '\b(mkfs|dd|format)\b' ) { Deny "System-level destructive operations are not allowed via automated execution." } if ( $command -match 'curl.*\|\s*(bash|sh)|wget.*\|\s*(bash|sh)' ) { Deny "Download-and-execute patterns require manual approval." } exit 0
```

#### Test the policy script

You can test the scripts by piping example `preToolUse` input.

Allow example:

```
echo '{"toolName":"bash","toolArgs":"{\"command\":\"git status\"}"}' \
  | .github/hooks/scripts/pre-tool-policy.sh # or, for PowerShell
echo '{"toolName":"bash","toolArgs":"{\"command\":\"git status\"}"}' |
  .github/hooks/scripts/pre-tool-policy.ps1
```

Deny example:

```
echo '{"toolName":"bash","toolArgs":"{\"command\":\"sudo rm -rf /\"}"}' \
  | .github/hooks/scripts/pre-tool-policy.sh # or, for PowerShell
echo '{"toolName":"bash","toolArgs":"{\"command\":\"sudo rm -rf /\"}"}' |
  .github/hooks/scripts/pre-tool-policy.ps1
```

After running the deny example, check `.github/hooks/logs/audit.jsonl` for a new denial log entry.

```
{ "permissionDecision" : "deny" , "permissionDecisionReason" : "Privilege escalation requires manual approval." }
```

At this point, high-risk `bash` commands are blocked from auto-execution in this repository.

### 6. Test end-to-end in the repository

Once you've created the configuration file and scripts, verify that hooks run as expected when you use Copilot CLI in this repository.

#### Validate your hook configuration file

Check that your hook configuration file is valid JSON:

Bash

```
jq '.' < .github/hooks/copilot-cli-policy.json
```

#### Verify script permissions (Unix-based systems)

On macOS and Linux, confirm your Bash scripts are executable:

Bash

```
chmod +x .github/hooks/scripts/*.sh
```

#### Run a basic session

Start a new Copilot CLI session in the repository:

Bash

```
copilot -p "Show me the status of this repository"
```

Expected results:

- You see the policy banner (from `sessionStart` ).
- A new entry is added to `.github/hooks/logs/audit.jsonl` (from `userPromptSubmitted` ).

#### Trigger tool use and verify logging

Run a prompt that causes Copilot CLI to use a tool (for example, bash):

Bash

```
copilot -p "Show me the last 5 git commits"
```

Expected results:

- A `preToolUse` entry is added to `.github/hooks/logs/audit.jsonl` .
- If the tool call is allowed, execution proceeds normally.

#### Test a denied command

The example policy script includes a temporary demo rule that blocks commands containing the string `COPILOT_HOOKS_DENY_DEMO` . This allows you to validate the deny flow safely without running destructive commands.

Run a prompt that would trigger a denied command:

Bash

```
copilot -p "Run a test command: echo COPILOT_HOOKS_DENY_DEMO"
```

Expected results:

- Copilot CLI does not execute the command.
- Your hook returns a denial response with a clear reason.
- A `policyDeny` entry is written to `.github/hooks/logs/audit.jsonl` .

After confirming that the deny flow works correctly, remove the demo rule from your script and replace it with deny patterns that reflect your organization's policies.

#### Inspect your audit logs

To view recent entries:

Bash

```
tail -n 50 .github/hooks/logs/audit.jsonl
```

To filter only denied decisions:

Bash

```
jq 'select(.event=="policyDeny")' .github/hooks/logs/audit.jsonl
```

### 7. Roll out safely across teams

After validating your hooks in a single repository, roll them out gradually to avoid disrupting development workflows.

#### Choose a rollout strategy

Common rollout approaches include:

- **Logging-first rollout (recommended)** : Start by logging prompts and tool usage without denying execution. Review logs for a period of time, then introduce deny rules once you understand common usage patterns.
- **Team-by-team rollout** : Deploy hooks to one team or repository at a time, gather feedback, then expand to additional teams.
- **Risk-based rollout** : Start with repositories that handle sensitive systems or production infrastructure, then expand to lower-risk repositories.

#### Communicate expectations

Before enforcing deny rules, make sure developers understand:

- That hooks are active in the repository
- Which types of commands may be blocked
- How to proceed if a command is denied

Clear communication reduces confusion and support requests.

#### Keep policies maintainable

As usage evolves:

- Store hook configuration and scripts in version control.
- Review audit logs periodically to detect new risk patterns.
- Update deny rules incrementally rather than adding broad matches.
- Document why each deny rule exists, especially for high-impact restrictions.

#### Handle exceptions carefully

Some teams (for example, infrastructure or platform teams) may require broader permissions. To handle this safely:

- Maintain separate hook configurations for different repositories.
- Keep exceptions narrow and well-documented.
- Avoid ad-hoc local bypasses that undermine auditability.

### Further reading

For troubleshooting hooks, see [Using hooks with GitHub Copilot agents](/en/copilot/how-tos/use-copilot-agents/cloud-agent/use-hooks#troubleshooting) .


---

# Reference


### About GitHub Copilot enhancements

You can enhance your experience of Copilot Chat with a variety of commands and options. Finding the right command or option for the task you are working on can help you achieve your goals more efficiently. This cheat sheet provides a quick reference to the most common commands and options for using Copilot Chat.

For information about how to get started with Copilot Chat in the GitHub website, see [Asking GitHub Copilot questions in GitHub](/en/copilot/using-github-copilot/asking-github-copilot-questions-in-github) .

### Mentions

Use `@` mentions in to attach relevant context directly to your conversations. Type `@` in the chat prompt box to display a list of items you can attach, such as:

- Discussions
- Extensions
- Files
- Issues
- Pull requests
- Repositories

### Slash commands

Use slash commands to avoid writing complex prompts for common scenarios. To use a slash command, type `/` in the chat prompt box, followed by the command name.

Available slash commands may vary, depending on your environment and the context of your chat. To view a list of currently available slash commands, type `/` in the chat prompt box of your current environment. Below is a list of some of the most common slash commands for using Copilot Chat.

| Command   | Description              |
|-----------|--------------------------|
| `/clear`  | Clear conversation.      |
| `/delete` | Delete a conversation.   |
| `/new`    | Start a new conversation |
| `/rename` | Rename a conversation.   |

### MCP skills

Below is a list of the MCP skills that are currently available in Copilot Chat in GitHub, and example prompts you can use to invoke them. You do not need to use the MCP skill name in your prompt; you can simply ask Copilot Chat to perform the task.

| Skill                        | Example prompt                                                                                                                                                                |
|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `create_branch`              | Create a new branch called [BRANCH-NAME] in the repository [USERNAME/REPO-NAME].                                                                                              |
| `create_or_update_file`      | Add a new file named `hello-world.md` to my [BRANCH-NAME] of [USERNAME/REPO-NAME] with the content: "Hello, world! This file was created from Copilot Chat in GitHub!"        |
| `push_files`                 | Push the files `test.md` with the content "This is a test file" and `test-again.md` with the content "This is another test file" to the [BRANCH-NAME] in [USERNAME/REPO-NAME] |
| `update_pull_request_branch` | Update the branch for pull request [PR-number] in [USERNAME/REPO-NAME] with the latest changes from the base branch.                                                          |
| `merge_pull_request`         | Merge pull request [PR-Number] in [USERNAME/REPO-NAME]                                                                                                                        |
| `get_me`                     | Tell me about myself.                                                                                                                                                         |
| `search_users`               | Search for users with the name "Mona Octocat"                                                                                                                                 |

For more information about using MCP skills in Copilot Chat, see [Using the GitHub MCP Server in your IDE](/en/copilot/how-tos/context/model-context-protocol/using-the-github-mcp-server) .

This version of this article is for Copilot in Visual Studio Code. For other versions of this article, click the tabs above.

### About GitHub Copilot enhancements

You can enhance your experience of Copilot Chat with a variety of commands and options. Finding the right command or option for the task you are working on can help you achieve your goals more efficiently. This cheat sheet provides a quick reference to the most common commands and options for using Copilot Chat.

For information about how to get started with Copilot Chat in Visual Studio Code, see [Asking GitHub Copilot questions in your IDE](/en/copilot/using-github-copilot/asking-github-copilot-questions-in-your-ide) .

### Slash commands

Use slash commands to avoid writing complex prompts for common scenarios. To use a slash command, type `/` in the chat prompt box, followed by the command name.

Available slash commands may vary, depending on your environment and the context of your chat. To view a list of currently available slash commands, type `/` in the chat prompt box of your current environment. Below is a list of some of the most common slash commands for using Copilot Chat.

| Command           | Description                                         |
|-------------------|-----------------------------------------------------|
| `/clear`          | Start a new chat session.                           |
| `/explain`        | Explain how the code in your active editor works.   |
| `/fix`            | Propose a fix for problems in the selected code.    |
| `/fixTestFailure` | Find and fix a failing test.                        |
| `/help`           | Quick reference and basics of using GitHub Copilot. |
| `/new`            | Create a new project.                               |
| `/tests`          | Generate unit tests for the selected code.          |

### Chat variables

Use chat variables to include specific context in your prompt. To use a chat variable, type `#` in the chat prompt box, followed by a chat variable.

| Variable     | Description                                            |
|--------------|--------------------------------------------------------|
| `#block`     | Includes the current block of code in the prompt.      |
| `#class`     | Includes the current class in the prompt.              |
| `#comment`   | Includes the current comment in the prompt.            |
| `#file`      | Includes the current file's content in the prompt.     |
| `#function`  | Includes the current function or method in the prompt. |
| `#line`      | Includes the current line of code in the prompt.       |
| `#path`      | Includes the file path in the prompt.                  |
| `#project`   | Includes the project context in the prompt.            |
| `#selection` | Includes the currently selected text in the prompt.    |
| `#sym`       | Includes the current symbol in the prompt.             |

### Chat participants

Chat participants are like domain experts who have a specialty that they can help you with. You can specify a chat participant by typing `@` in the chat prompt box, followed by a chat participant name. To see all available chat participants, type `@` in the chat prompt box.

Below is a list of some of the most common chat participants for using Copilot Chat.

| Variable     | Description                                                                                                                                                                                                         |
|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `@azure`     | Has context about Azure services and how to use, deploy and manage them. Use `@azure` when you want help with Azure. The `@azure` chat participant is currently in public preview and is subject to change.         |
| `@github`    | Allows you to use GitHub-specific Copilot skills. See [Asking GitHub Copilot questions in your IDE](/en/copilot/using-github-copilot/asking-github-copilot-questions-in-your-ide#using-github-skills-for-copilot) . |
| `@terminal`  | Has context about the Visual Studio Code terminal shell and its contents. Use `@terminal` when you want help creating or debugging terminal commands.                                                               |
| `@vscode`    | Has context about Visual Studio Code commands and features. Use `@vscode` when you want help with Visual Studio Code.                                                                                               |
| `@workspace` | Has context about the code in your workspace. Use `@workspace` when you want Copilot to consider the structure of your project, how different parts of your code interact, or design patterns in your project.      |

This version of this article is for Copilot in Visual Studio. For other versions of this article, click the tabs above.

### About GitHub Copilot enhancements

You can enhance your experience of Copilot Chat with a variety of commands and options. Finding the right command or option for the task you are working on can help you achieve your goals more efficiently. This cheat sheet provides a quick reference to the most common commands and options for using Copilot Chat.

For information about how to get started with Copilot Chat in Visual Studio, see [Asking GitHub Copilot questions in your IDE](/en/copilot/using-github-copilot/asking-github-copilot-questions-in-your-ide) .

### Slash commands

Use slash commands to avoid writing complex prompts for common scenarios. To use a slash command, type `/` in the chat prompt box, followed by the command name.

Available slash commands may vary, depending on your environment and the context of your chat. To view a list of currently available slash commands, type `/` in the chat prompt box of your current environment. Below is a list of some of the most common slash commands for using Copilot Chat.

| Command     | Description                                            |
|-------------|--------------------------------------------------------|
| `/doc`      | Add documentation comment for this symbol.             |
| `/explain`  | Explain how the code in your active editor works.      |
| `/fix`      | Propose a fix for problems in the selected code.       |
| `/help`     | Quick reference and basics of using GitHub Copilot.    |
| `/optimize` | Analyze and improve running time of the selected code. |
| `/tests`    | Generate unit tests for the selected code.             |

### References

By default, Copilot Chat will reference the file that you have open or the code that you have selected. You can also use # followed by a file name, file name and line numbers, or solution to reference a specific file, lines, or solution.

| Example                                              | Description                         |
|------------------------------------------------------|-------------------------------------|
| `Where are the tests in #MyFile.cs?`                 | References a specific file          |
| `How are these files related #MyFile.cs #MyFile2.cs` | References multiple files           |
| `Explain this function #MyFile.cs: 66-72?`           | References specific lines in a file |
| `Is there a delete method in this #solution?`        | References the current file         |

This version of this article is for Copilot in JetBrains. For other versions of this article, click the tabs above.

### About GitHub Copilot enhancements

You can enhance your experience of Copilot Chat with a variety of commands and options. Finding the right command or option for the task you are working on can help you achieve your goals more efficiently. This cheat sheet provides a quick reference to the most common commands and options for using Copilot Chat.

For information about how to get started with Copilot Chat in JetBrains, see [Asking GitHub Copilot questions in your IDE](/en/copilot/using-github-copilot/asking-github-copilot-questions-in-your-ide) .

### Slash commands

Use slash commands to avoid writing complex prompts for common scenarios. To use a slash command, type `/` in the chat prompt box, followed by the command name.

Available slash commands may vary, depending on your environment and the context of your chat. To view a list of currently available slash commands, type `/` in the chat prompt box of your current environment. Below is a list of some of the most common slash commands for using Copilot Chat.

| Command    | Description                                         |
|------------|-----------------------------------------------------|
| `/explain` | Explain how the code in your active editor works.   |
| `/fix`     | Propose a fix for problems in the selected code.    |
| `/help`    | Quick reference and basics of using GitHub Copilot. |
| `/tests`   | Generate unit tests for the selected code.          |

This version of this article is for Copilot in Xcode. For other versions of this article, click the tabs above.

### About GitHub Copilot enhancements

You can enhance your experience of Copilot Chat with a variety of commands and options. Finding the right command or option for the task you are working on can help you achieve your goals more efficiently. This cheat sheet provides a quick reference to the most common commands and options for using Copilot Chat.

For information about how to get started with Copilot Chat in Xcode, see [Asking GitHub Copilot questions in your IDE](/en/copilot/using-github-copilot/asking-github-copilot-questions-in-your-ide) .

### Slash commands

Use slash commands to avoid writing complex prompts for common scenarios. To use a slash command, type `/` in the chat prompt box, followed by the command name.

Available slash commands may vary, depending on your environment and the context of your chat. To view a list of currently available slash commands, type `/` in the chat prompt box of your current environment. Below is a list of the slash commands for using Copilot Chat.

| Command     | Description                                        |
|-------------|----------------------------------------------------|
| `/doc`      | Generate documentation for this symbol.            |
| `/explain`  | Provide an explanation for the selected code.      |
| `/fix`      | Suggest fixes for code errors and typos.           |
| `/simplify` | Simplify the current code selection.               |
| `/tests`    | Create a unit test for the current code selection. |


### Feature overview

This table shows what each customization feature is and where it lives.

| Feature                                                                                              | What it is                                                                                     | Filename and location                                                                                                                                                                                                                                                                |
|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Custom instructions](/en/copilot/concepts/prompting/response-customization)                         | Always-on context that automatically applies to every interaction within its defined scope     | `.github/copilot-instructions.md` (repo-wide), `.github/instructions/*.instructions.md` (path-specific), `AGENTS.md` (third-party agents), or personal/org settings via UI on GitHub                                                                                                 |
| [Prompt files](/en/copilot/concepts/prompting/response-customization?tool=vscode#about-prompt-files) | Reusable, standalone prompt template with input variables                                      | `.github/prompts/*.prompt.md`                                                                                                                                                                                                                                                        |
| [Custom agents](/en/copilot/concepts/agents/cloud-agent/about-custom-agents)                         | Specialist persona with its own instructions, tool restrictions, and context                   | `.github/agents/AGENT-NAME.md` (repo), `agents/AGENT-NAME.md` in `.github-private` repo (org/enterprise), or user profile                                                                                                                                                            |
| [Subagents](/en/copilot/how-tos/chat-with-copilot/chat-in-ide#using-subagents)                       | Separate agent spawned by the main agent to handle delegated work in an isolated context       | N/A (runtime process, not a user-configured file)                                                                                                                                                                                                                                    |
| [Agent skills](/en/copilot/concepts/agents/about-agent-skills)                                       | Folder of instructions, scripts, and resources that Copilot loads when relevant to a task      | `.github/skills/<skill-name>/SKILL.md` , `.claude/skills/<skill-name>/SKILL.md` , or `.agents/skills/<skill-name>/SKILL.md` (project); `~/.copilot/skills/<skill-name>/SKILL.md` , `~/.claude/skills/<skill-name>/SKILL.md` , or `~/.agents/skills/<skill-name>/SKILL.md` (personal) |
| [Hooks](/en/copilot/concepts/agents/cloud-agent/about-hooks)                                         | Custom shell commands that execute deterministically at specific points in an agent's workflow | `.github/hooks/*.json`                                                                                                                                                                                                                                                               |
| [MCP servers](/en/copilot/concepts/context/mcp)                                                      | Connection to external systems, APIs, and databases                                            | `mcp.json` (path varies by IDE), repo settings on GitHub (cloud agent), or `mcp-servers` property in custom agent configurations                                                                                                                                                     |

### Usage comparison

This table helps you decide which customization feature to use.

| Feature                                                                                              | How to trigger                                                                   | Best for                                                                                         | Example use cases                                                                                                     |
|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| [Custom instructions](/en/copilot/concepts/prompting/response-customization)                         | Automatic                                                                        | Standards, guidelines, or expectations that apply broadly across a context                       | Enforce coding standards, accessibility rules, review checklists                                                      |
| [Prompt files](/en/copilot/concepts/prompting/response-customization?tool=vscode#about-prompt-files) | Manual: reference directly in chat or use the prompt file picker                 | Focused single tasks you run once with different inputs each time                                | Generate unit tests, run a code review checklist                                                                      |
| [Custom agents](/en/copilot/concepts/agents/cloud-agent/about-custom-agents)                         | Manual: select from the agent dropdown in your IDE, on GitHub, or in Copilot CLI | Projects or processes with distinct stages that need specialized capabilities or strict handoffs | React reviewer agent, read-only auditing agent                                                                        |
| [Subagents](/en/copilot/how-tos/chat-with-copilot/chat-in-ide#using-subagents)                       | Automatic, or reference a subagent directly in your prompt                       | Complex subtasks that should run in isolation from the main agent                                | Codebase research, running test suites                                                                                |
| [Agent skills](/en/copilot/concepts/agents/about-agent-skills)                                       | Automatic: chosen by Copilot when relevant to your prompt                        | Multi-step workflows with bundled assets that should be loaded as needed                         | GitHub Actions failure debugging, deployment procedures, release note drafting                                        |
| [Hooks](/en/copilot/concepts/agents/cloud-agent/about-hooks)                                         | Automatic: at configured lifecycle events                                        | Tasks that need to run at a specific point in the agent lifecycle, with guaranteed execution     | Run a formatter after every file edit, approve or deny tool executions, prevent credential leaks with secret scanning |
| [MCP servers](/en/copilot/concepts/context/mcp)                                                      | Automatic, or ask for a specific tool by name                                    | Tasks that require access to external tools or real-time data                                    | Manage issues and PRs (GitHub MCP server), automate browser testing (Playwright MCP server)                           |

### IDE and surface support

This table shows which customization features are supported in each IDE and surface. For the full Copilot feature matrix, see [Copilot feature matrix](/en/copilot/reference/copilot-feature-matrix#features-by-ide) .

GitHub recommends using the latest stable IDE, Copilot CLI, and Copilot extension versions to get the best Copilot experience.

**Key:**

- ✓ = supported
- ✗ = not supported
- P = under preview

| Feature             | VS Code   | Visual Studio   | JetBrains IDEs   | Eclipse   | Xcode   | GitHub .com   | Copilot CLI   |
|---------------------|-----------|-----------------|------------------|-----------|---------|---------------|---------------|
| Custom instructions | ✓         | ✓               | P                | P         | P       | ✓             | ✓             |
| Prompt files        | ✓         | ✓               | P                | ✗         | P       | ✗             | ✗             |
| Custom agents       | ✓         | ✗               | P                | P         | P       | ✓             | ✓             |
| Subagents           | ✓         | ✗               | P                | P         | P       | ✗             | ✓             |
| Agent skills        | ✓         | ✗               | P                | ✗         | ✗       | ✓             | ✓             |
| Hooks               | P         | ✗               | ✗                | ✗         | ✗       | ✓             | ✓             |
| MCP servers         | ✓         | ✓               | ✓                | ✓         | ✓       | ✓             | ✓             |

For a detailed breakdown of which types of custom instructions are supported in each IDE and surface, see [Support for different types of custom instructions](/en/copilot/reference/custom-instructions-support) .

### Further reading

- [Customization library](/en/copilot/tutorials/customization-library) -a curated collection of examples


### Supported AI models in Copilot

This table lists the AI models available in Copilot, along with their release status and availability in different modes.

| Model name                            | Provider                 | Release status           | Agent mode   | Ask mode   | Edit mode   |
|---------------------------------------|--------------------------|--------------------------|--------------|------------|-------------|
| GPT-4.1                               | OpenAI                   | GA                       |              |            |             |
| GPT-5 mini                            | OpenAI                   | GA                       |              |            |             |
| GPT-5.1                               | OpenAI                   | Closing down: 2026-04-15 |              |            |             |
| GPT-5.2                               | OpenAI                   | GA                       |              |            |             |
| GPT-5.2-Codex                         | OpenAI                   | GA                       |              |            |             |
| GPT-5.3-Codex                         | OpenAI                   | GA                       |              |            |             |
| GPT-5.4                               | OpenAI                   | GA                       |              |            |             |
| GPT-5.4 mini                          | OpenAI                   | GA                       |              |            |             |
| Claude Haiku 4.5                      | Anthropic                | GA                       |              |            |             |
| Claude Opus 4.5                       | Anthropic                | GA                       |              |            |             |
| Claude Opus 4.6                       | Anthropic                | GA                       |              |            |             |
| Claude Opus 4.6 (fast mode) (preview) | Anthropic                | Public preview           |              |            |             |
| Claude Sonnet 4                       | Anthropic                | GA                       |              |            |             |
| Claude Sonnet 4.5                     | Anthropic                | GA                       |              |            |             |
| Claude Sonnet 4.6                     | Anthropic                | GA                       |              |            |             |
| Gemini 2.5 Pro                        | Google                   | GA                       |              |            |             |
| Gemini 3 Flash                        | Google                   | Public preview           |              |            |             |
| Gemini 3.1 Pro                        | Google                   | Public preview           |              |            |             |
| Grok Code Fast 1                      | xAI                      | GA                       |              |            |             |
| Raptor mini                           | Fine-tuned GPT-5 mini    | Public preview           |              |            |             |
| Goldeneye                             | Fine-tuned GPT-5.1-Codex | Public preview           |              |            |             |

### Model retirement history

The following table lists AI models that are retired or scheduled for retirement from Copilot, along with their retirement dates and suggested alternatives.

| Model name                 | Retirement date   | Suggested alternative   |
|----------------------------|-------------------|-------------------------|
| GPT-5.1                    | 2026-04-15        | GPT-5.3-Codex           |
| GPT-5.1-Codex              | 2026-04-01        | GPT-5.3-Codex           |
| GPT-5.1-Codex-Max          | 2026-04-01        | GPT-5.3-Codex           |
| GPT-5.1-Codex-Mini         | 2026-04-01        | GPT-5.3-Codex           |
| Gemini 3 Pro               | 2026-03-26        | Gemini 3.1 Pro          |
| Claude Opus 4.1            | 2026-02-17        | Claude Opus 4.6         |
| GPT-5                      | 2026-02-17        | GPT-5.2                 |
| GPT-5-Codex                | 2026-02-17        | GPT-5.2-Codex           |
| Claude Sonnet 3.5          | 2025-11-06        | Claude Haiku 4.5        |
| Claude Opus 4              | 2025-10-23        | Claude Opus 4.6         |
| Claude Sonnet 3.7          | 2025-10-23        | Claude Sonnet 4.6       |
| Claude Sonnet 3.7 Thinking | 2025-10-23        | Claude Sonnet 4.6       |
| Gemini 2.0 Flash           | 2025-10-23        | Gemini 2.5 Pro          |
| o1-mini                    | 2025-10-23        | GPT-5 mini              |
| o3                         | 2025-10-23        | GPT-5.2                 |
| o3-mini                    | 2025-10-23        | GPT-5 mini              |
| o4-mini                    | 2025-10-23        | GPT-5 mini              |

### Supported AI models per client

The following table shows which models are available in each client.

Note

- When you use Copilot Chat in supported IDEs, **Auto** will automatically select the best model for you based on availability. You can manually choose a different model to override this selection. See [About Copilot auto model selection](/en/copilot/concepts/auto-model-selection) and [Changing the AI model for GitHub Copilot Chat](/en/copilot/how-tos/use-ai-models/change-the-chat-model?tool=vscode) .
- GPT-5-Codex is supported in Visual Studio Code v1.104.1 or higher.

| Model                                 | GitHub.com   | Copilot CLI   | Visual Studio Code   | Visual Studio   | Eclipse   | Xcode   | JetBrains IDEs   |
|---------------------------------------|--------------|---------------|----------------------|-----------------|-----------|---------|------------------|
| Claude Haiku 4.5                      |              |               |                      |                 |           |         |                  |
| Claude Opus 4.5                       |              |               |                      |                 |           |         |                  |
| Claude Opus 4.6                       |              |               |                      |                 |           |         |                  |
| Claude Opus 4.6 (fast mode) (preview) |              |               |                      |                 |           |         |                  |
| Claude Sonnet 4                       |              |               |                      |                 |           |         |                  |
| Claude Sonnet 4.5                     |              |               |                      |                 |           |         |                  |
| Claude Sonnet 4.6                     |              |               |                      |                 |           |         |                  |
| Gemini 2.5 Pro                        |              |               |                      |                 |           |         |                  |
| Gemini 3 Flash                        |              |               |                      |                 |           |         |                  |
| Gemini 3.1 Pro                        |              |               |                      |                 |           |         |                  |
| GPT-4.1                               |              |               |                      |                 |           |         |                  |
| GPT-5 mini                            |              |               |                      |                 |           |         |                  |
| GPT-5.1                               |              |               |                      |                 |           |         |                  |
| GPT-5.2                               |              |               |                      |                 |           |         |                  |
| GPT-5.2-Codex                         |              |               |                      |                 |           |         |                  |
| GPT-5.3-Codex                         |              |               |                      |                 |           |         |                  |
| GPT-5.4                               |              |               |                      |                 |           |         |                  |
| GPT-5.4 mini                          |              |               |                      |                 |           |         |                  |
| Grok Code Fast 1                      |              |               |                      |                 |           |         |                  |
| Raptor mini                           |              |               |                      |                 |           |         |                  |
| Goldeneye                             |              |               |                      |                 |           |         |                  |

### Supported AI models per Copilot plan

The following table shows which AI models are available in each Copilot plan. For more information about the plans, see [Plans for GitHub Copilot](/en/copilot/about-github-copilot/plans-for-github-copilot) .

| Available models in chat              | Copilot Free   | Copilot Student   | Copilot Pro   | Copilot Pro+   | Copilot Business   | Copilot Enterprise   |
|---------------------------------------|----------------|-------------------|---------------|----------------|--------------------|----------------------|
| Claude Haiku 4.5                      |                |                   |               |                |                    |                      |
| Claude Opus 4.5                       |                |                   |               |                |                    |                      |
| Claude Opus 4.6                       |                |                   |               |                |                    |                      |
| Claude Opus 4.6 (fast mode) (preview) |                |                   |               |                |                    |                      |
| Claude Sonnet 4                       |                |                   |               |                |                    |                      |
| Claude Sonnet 4.5                     |                |                   |               |                |                    |                      |
| Claude Sonnet 4.6                     |                |                   |               |                |                    |                      |
| Gemini 2.5 Pro                        |                |                   |               |                |                    |                      |
| Gemini 3 Flash                        |                |                   |               |                |                    |                      |
| Gemini 3.1 Pro                        |                |                   |               |                |                    |                      |
| GPT-4.1                               |                |                   |               |                |                    |                      |
| GPT-5 mini                            |                |                   |               |                |                    |                      |
| GPT-5.1                               |                |                   |               |                |                    |                      |
| GPT-5.2                               |                |                   |               |                |                    |                      |
| GPT-5.2-Codex                         |                |                   |               |                |                    |                      |
| GPT-5.3-Codex                         |                |                   |               |                |                    |                      |
| GPT-5.4                               |                |                   |               |                |                    |                      |
| GPT-5.4 mini                          |                |                   |               |                |                    |                      |
| Grok Code Fast 1                      |                |                   |               |                |                    |                      |
| Raptor mini                           |                |                   |               |                |                    |                      |
| Goldeneye                             |                |                   |               |                |                    |                      |

### Model multipliers

Note

The multiplier for these models are subject to change.

- Claude Sonnet 4.6
- GPT-5.4 mini

Each model has a premium request multiplier, based on its complexity and resource usage. If you are on a paid Copilot plan, your premium request allowance is deducted according to this multiplier.

For more information about premium requests, see [Requests in GitHub Copilot](/en/copilot/managing-copilot/monitoring-usage-and-entitlements/about-premium-requests) .

| Model                                 | Multiplier for **paid plans**   | Multiplier for **Copilot Free**   |
|---------------------------------------|---------------------------------|-----------------------------------|
| Claude Haiku 4.5                      | 0.33                            | 1                                 |
| Claude Opus 4.5                       | 3                               | Not applicable                    |
| Claude Opus 4.6                       | 3                               | Not applicable                    |
| Claude Opus 4.6 (fast mode) (preview) | 30                              | Not applicable                    |
| Claude Sonnet 4                       | 1                               | Not applicable                    |
| Claude Sonnet 4.5                     | 1                               | Not applicable                    |
| Claude Sonnet 4.6                     | 1                               | Not applicable                    |
| Gemini 2.5 Pro                        | 1                               | Not applicable                    |
| Gemini 3 Flash                        | 0.33                            | Not applicable                    |
| Gemini 3.1 Pro                        | 1                               | Not applicable                    |
| GPT-4.1                               | 0                               | 1                                 |
| GPT-4o                                | 0                               | 1                                 |
| GPT-5 mini                            | 0                               | 1                                 |
| GPT-5.1                               | 1                               | Not applicable                    |
| GPT-5.2                               | 1                               | Not applicable                    |
| GPT-5.2-Codex                         | 1                               | Not applicable                    |
| GPT-5.3-Codex                         | 1                               | Not applicable                    |
| GPT-5.4                               | 1                               | Not applicable                    |
| GPT-5.4 mini                          | 0.33                            | Not applicable                    |
| Grok Code Fast 1                      | 0.25                            | 1                                 |
| Raptor mini                           | 0                               | 1                                 |
| Goldeneye                             | Not applicable                  | 1                                 |

### Fallback and long-term support (LTS) models

For more information about fallback and LTS models, see [Base and long-term support (LTS) models](/en/copilot/concepts/fallback-and-lts-models) .

### Evaluation models

GitHub Copilot offers access to evaluation models-including top-performing open source and open-weight models-to provide the most advanced coding suggestions available.

Note

Testing of evaluation models has revealed that some may perform worse than other models on security-related or other categories of prompts. Customers are encouraged to validate code, including code security, using a range of models and thorough human review before incorporating suggestions into production.

Evaluation models may be added, updated, or removed without notice. Availability and rate limits may differ from generally available models.

### Next steps

- For task-based guidance on selecting a model, see [AI model comparison](/en/copilot/reference/ai-models/model-comparison) .
- To configure which models are available to you, see [Configuring access to AI models in GitHub Copilot](/en/copilot/using-github-copilot/ai-models/configuring-access-to-ai-models-in-copilot) .
- To learn how to change your current model, see [Changing the AI model for GitHub Copilot Chat](/en/copilot/using-github-copilot/ai-models/changing-the-ai-model-for-copilot-chat) or [Changing the AI model for GitHub Copilot inline suggestions](/en/copilot/how-tos/use-ai-models/change-the-completion-model) .
- To learn more about Responsible Use and Responsible AI, see [Copilot Trust Center](https://copilot.github.trust.page/) and [Responsible use of GitHub Copilot features](/en/copilot/responsible-use-of-github-copilot-features) .
- To learn how Copilot Chat serves different AI models, see [Hosting of models for GitHub Copilot](/en/copilot/reference/ai-models/model-hosting) .


### Comparison of AI models for GitHub Copilot

GitHub Copilot supports multiple AI models with different capabilities. The model you choose affects the quality and relevance of responses by Copilot Chat and Copilot inline suggestions. Some models offer lower latency, while others offer fewer hallucinations or better performance on specific tasks. This guide helps you pick the best model based on your task, not just model names.

Note

- Different models have different premium request multipliers, which can affect how much of your monthly usage allowance is consumed. For details, see [Requests in GitHub Copilot](/en/copilot/managing-copilot/monitoring-usage-and-entitlements/about-premium-requests) .
- When you use Copilot Chat in supported IDEs, **Auto** will automatically select the best model for you based on availability. You can manually choose a different model to override this selection. See [About Copilot auto model selection](/en/copilot/concepts/auto-model-selection) and [Changing the AI model for GitHub Copilot Chat](/en/copilot/how-tos/use-ai-models/change-the-chat-model?tool=vscode) .

#### Recommended models by task

Use this table to find a suitable model quickly, see more detail in the sections below.

| Model                                 | Task area                                 | Excels at (primary use case)                                                 | Further reading                                                                                                            |
|---------------------------------------|-------------------------------------------|------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| GPT-4.1                               | General-purpose coding and writing        | Fast, accurate code completions and explanations                             | [GPT-4.1 model card](https://openai.com/index/gpt-4-1/)                                                                    |
| GPT-5 mini                            | General-purpose coding and writing        | Fast, accurate code completions and explanations                             | [GPT-5 mini model card](https://cdn.openai.com/gpt-5-system-card.pdf)                                                      |
| GPT-5.1                               | Deep reasoning and debugging              | Multi-step problem solving and architecture-level code analysis              | [GPT-5.1 model card](https://cdn.openai.com/pdf/4173ec8d-1229-47db-96de-06d87147e07e/5_1_system_card.pdf)                  |
| GPT-5.2                               | Deep reasoning and debugging              | Multi-step problem solving and architecture-level code analysis              | [GPT-5.2 model card](https://cdn.openai.com/pdf/3a4153c8-c748-4b71-8e31-aecbde944f8d/oai_5_2_system-card.pdf)              |
| GPT-5.2-Codex                         | Agentic software development              | Agentic tasks                                                                | [GPT-5.2-Codex model card](https://cdn.openai.com/pdf/ac7c37ae-7f4c-4442-b741-2eabdeaf77e0/oai_5_2_Codex.pdf)              |
| GPT-5.3-Codex                         | Agentic software development              | Agentic tasks                                                                | [GPT-5.3-Codex model card](https://deploymentsafety.openai.com/gpt-5-3-codex)                                              |
| GPT-5.4                               | Deep reasoning and debugging              | Multi-step problem solving and architecture-level code analysis              | [GPT-5.4 model card](https://deploymentsafety.openai.com/gpt-5-4-thinking/introduction)                                    |
| GPT-5.4 mini                          | Agentic software development              | Codebase exploration and is especially effective when using grep-style tools | Not available                                                                                                              |
| Claude Haiku 4.5                      | Fast help with simple or repetitive tasks | Fast, reliable answers to lightweight coding questions                       | [Claude Haiku 4.5 model card](https://assets.anthropic.com/m/99128ddd009bdcb/Claude-Haiku-4-5-System-Card.pdf)             |
| Claude Opus 4.5                       | Deep reasoning and debugging              | Complex problem-solving challenges, sophisticated reasoning                  | [Claude Opus 4.5 model card](https://assets.anthropic.com/m/64823ba7485345a7/Claude-Opus-4-5-System-Card.pdf)              |
| Claude Opus 4.6                       | Deep reasoning and debugging              | Complex problem-solving challenges, sophisticated reasoning                  | [Claude Opus 4.6 model card](https://www-cdn.anthropic.com/14e4fb01875d2a69f646fa5e574dea2b1c0ff7b5.pdf)                   |
| Claude Opus 4.6 (fast mode) (preview) | Deep reasoning and debugging              | Complex problem-solving challenges, sophisticated reasoning                  | Not available                                                                                                              |
| Claude Sonnet 4.0                     | Deep reasoning and debugging              | Performance and practicality, perfectly balanced for coding workflows        | [Claude Sonnet 4.0 model card](https://www-cdn.anthropic.com/6be99a52cb68eb70eb9572b4cafad13df32ed995.pdf)                 |
| Claude Sonnet 4.5                     | General-purpose coding and agent tasks    | Complex problem-solving challenges, sophisticated reasoning                  | [Claude Sonnet 4.5 model card](https://assets.anthropic.com/m/12f214efcc2f457a/original/Claude-Sonnet-4-5-System-Card.pdf) |
| Claude Sonnet 4.6                     | General-purpose coding and agent tasks    | Complex problem-solving challenges, sophisticated reasoning                  | [Claude Sonnet 4.6 model card](https://www-cdn.anthropic.com/78073f739564e986ff3e28522761a7a0b4484f84.pdf)                 |
| Gemini 2.5 Pro                        | Deep reasoning and debugging              | Complex code generation, debugging, and research workflows                   | [Gemini 2.5 Pro model card](https://storage.googleapis.com/model-cards/documents/gemini-2.5-pro.pdf)                       |
| Gemini 3 Flash                        | Fast help with simple or repetitive tasks | Fast, reliable answers to lightweight coding questions                       | [Gemini 3 Flash model card](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Flash-Model-Card.pdf)       |
| Gemini 3.1 Pro                        | Deep reasoning and debugging              | Effective and efficient edit-then-test loops with high tool precision        | not applicable                                                                                                             |
| Grok Code Fast 1                      | General-purpose coding and writing        | Fast, accurate code completions and explanations                             | [Grok Code Fast 1 model card](https://data.x.ai/2025-08-20-grok-4-model-card.pdf)                                          |
| Qwen2.5                               | General-purpose coding and writing        | Code generation, reasoning, and code repair / debugging                      | [Qwen2.5 model card](https://arxiv.org/pdf/2409.12186)                                                                     |
| Raptor mini                           | General-purpose coding and writing        | Fast, accurate code completions and explanations                             | Coming soon                                                                                                                |

### Task: General-purpose coding and writing

Use these models for common development tasks that require a balance of quality, speed, and cost efficiency. These models are a good default when you don't have specific requirements.

| Model            | Why it's a good fit                                                                                                                             |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| GPT-5.3-Codex    | Delivers higher-quality code on complex engineering tasks like features, tests, debugging, refactors, and reviews without lengthy instructions. |
| GPT-5 mini       | Reliable default for most coding and writing tasks. Fast, accurate, and works well across languages and frameworks.                             |
| Grok Code Fast 1 | Specialized for coding tasks. Performs well on code generation, and debugging across multiple languages.                                        |
| Raptor mini      | Specialized for fast, accurate inline suggestions and explanations.                                                                             |

#### When to use these models

Use one of these models if you want to:

- Write or review functions, short files, or code diffs.
- Generate documentation, comments, or summaries.
- Explain errors or unexpected behavior quickly.
- Work in a non-English programming environment.

#### When to use a different model

If you're working on complex refactoring, architectural decisions, or multi-step logic, consider a model from [Deep reasoning and debugging](#task-deep-reasoning-and-debugging) . For faster, simpler tasks like repetitive edits or one-off code suggestions, see [Fast help with simple or repetitive tasks](#task-fast-help-with-simple-or-repetitive-tasks) .

### Task: Fast help with simple or repetitive tasks

These models are optimized for speed and responsiveness. They're ideal for quick edits, utility functions, syntax help, and lightweight prototyping. You'll get fast answers without waiting for unnecessary depth or long reasoning chains.

#### Recommended models

| Model            | Why it's a good fit                                                                                   |
|------------------|-------------------------------------------------------------------------------------------------------|
| Claude Haiku 4.5 | Balances fast responses with quality output. Ideal for small tasks and lightweight code explanations. |

#### When to use these models

Use one of these models if you want to:

- Write or edit small functions or utility code.
- Ask quick syntax or language questions.
- Prototype ideas with minimal setup.
- Get fast feedback on simple prompts or edits.

#### When to use a different model

If you're working on complex refactoring, architectural decisions, or multi-step logic, see [Deep reasoning and debugging](#task-deep-reasoning-and-debugging) .

For tasks that need stronger general-purpose reasoning or more structured output, see

[General-purpose coding and writing](#task-general-purpose-coding-and-writing) .

### Task: Deep reasoning and debugging

These models are designed for tasks that require step-by-step reasoning, complex decision-making, or high-context awareness. They work well when you need structured analysis, thoughtful code generation, or multi-file understanding.

#### Recommended models

| Model             | Why it's a good fit                                                                                                                                             |
|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GPT-5 mini        | Delivers deep reasoning and debugging with faster responses and lower resource usage than GPT-5. Ideal for interactive sessions and step-by-step code analysis. |
| GPT-5.4           | Great at complex reasoning, code analysis, and technical decision-making.                                                                                       |
| Claude Sonnet 4.6 | Improves on Sonnet 4.5 with more reliable completions and smarter reasoning under pressure.                                                                     |
| Claude Opus 4.6   | Anthropic's most powerful model. Improves on Claude Opus 4.5.                                                                                                   |
| Gemini 3.1 Pro    | Advanced reasoning across long contexts and scientific or technical analysis.                                                                                   |
| Goldeneye         | Complex problem-solving challenges and sophisticated reasoning.                                                                                                 |

#### When to use these models

Use one of these models if you want to:

- Debug complex issues with context across multiple files.
- Refactor large or interconnected codebases.
- Plan features or architecture across layers.
- Weigh trade-offs between libraries, patterns, or workflows.
- Analyze logs, performance data, or system behavior.

#### When to use a different model

For fast iteration or lightweight tasks, see [Fast help with simple or repetitive tasks](#task-fast-help-with-simple-or-repetitive-tasks) .

For general development workflows or content generation, see

[General-purpose coding and writing](#task-general-purpose-coding-and-writing) .

### Task: Working with visuals (diagrams, screenshots)

Use these models when you want to ask questions about screenshots, diagrams, UI components, or other visual input. These models support multimodal input and are well suited for front-end work or visual debugging.

| Model             | Why it's a good fit                                                                                                                                                       |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GPT-5 mini        | Reliable default for most coding and writing tasks. Fast, accurate, and supports multimodal input for visual reasoning tasks. Works well across languages and frameworks. |
| Claude Sonnet 4.6 | Improves on Sonnet 4.5 with more reliable completions and smarter reasoning under pressure.                                                                               |
| Gemini 3.1 Pro    | Deep reasoning and debugging, ideal for complex code generation, debugging, and research workflows.                                                                       |

#### When to use these models

Use one of these models if you want to:

- Ask questions about diagrams, screenshots, or UI components.
- Get feedback on visual drafts or workflows.
- Understand front-end behavior from visual context.

Tip

If you're using a model in a context that doesn't support image input (like a code editor), you won't see visual reasoning benefits. You may be able to use an MCP server to get access to visual input indirectly. See [Extending GitHub Copilot Chat with Model Context Protocol (MCP) servers](/en/copilot/customizing-copilot/using-model-context-protocol/extending-copilot-chat-with-mcp) .

#### When to use a different model

If your task involves deep reasoning or large-scale refactoring, consider a model from [Deep reasoning and debugging](#task-deep-reasoning-and-debugging) . For text-only tasks or simpler code edits, see [Fast help with simple or repetitive tasks](#task-fast-help-with-simple-or-repetitive-tasks) .

### Next steps

Choosing the right model helps you get the most out of Copilot. If you're not sure which model to use, start with a general-purpose option like GPT-4.1, then adjust based on your needs.

- For detailed model specs and pricing, see [Supported AI models in GitHub Copilot](/en/copilot/using-github-copilot/ai-models/supported-ai-models-in-copilot) .
- For more examples of how to use different models, see [Comparing AI models using different tasks](/en/copilot/using-github-copilot/ai-models/comparing-ai-models-using-different-tasks) .
- To switch between models, refer to [Changing the AI model for GitHub Copilot Chat](/en/copilot/using-github-copilot/ai-models/changing-the-ai-model-for-copilot-chat) or [Changing the AI model for GitHub Copilot inline suggestions](/en/copilot/how-tos/use-ai-models/change-the-completion-model) .
- To learn how Copilot Chat serves different AI models, see [Hosting of models for GitHub Copilot](/en/copilot/reference/ai-models/model-hosting) .


### OpenAI models

Used for:

- GPT-4.1
- GPT-5 mini
- GPT-5.1
- GPT-5.2
- GPT-5.2-Codex
- GPT-5.3-Codex
- GPT-5.4
- GPT-5.4 mini

These models are hosted by OpenAI and GitHub's Azure infrastructure.

OpenAI makes the [following data commitment](https://openai.com/enterprise-privacy/) : *We [OpenAI] do not train models on customer business data* . Data processing follows OpenAI's enterprise privacy comments.

GitHub maintains a [zero data retention agreement](https://platform.openai.com/docs/guides/your-data) with OpenAI.

All input requests and output responses processed by GitHub Copilot's models continue to pass through GitHub Copilot's, content filtering systems. These filters include checks for public code matches (when applied) as well as mechanisms to detect and block harmful or offensive content.

### OpenAI models fine-tuned by Microsoft

Used for:

- Raptor mini
- Goldeneye

These models are deployed on GitHub managed Azure OpenAI tenant.

### Anthropic models

Used for:

- Claude Haiku 4.5
- Claude Sonnet 4.5
- Claude Opus 4.5
- Claude Opus 4.6
- Claude Opus 4.6 (fast mode) (preview)
- Claude Sonnet 4
- Claude Sonnet 4.6

These models are hosted by Amazon Web Services, Anthropic PBC, and Google Cloud Platform. GitHub has provider agreements in place to ensure data is not used for training. Additional details for each provider are included below:

- Amazon Bedrock: Amazon makes the [following data commitments](https://docs.aws.amazon.com/bedrock/latest/userguide/data-protection.html) : *Amazon Bedrock doesn't store or log your prompts and completions. Amazon Bedrock doesn't use your prompts and completions to train any AWS models and doesn't distribute them to third parties* .

- Anthropic PBC: GitHub maintains a [zero data retention agreement](https://privacy.anthropic.com/en/articles/8956058-i-have-a-zero-retention-agreement-with-anthropic-what-products-does-it-apply-to) with Anthropic for generally available Anthropic features in GitHub Copilot. Some Anthropic features in beta or public preview-including tool search via the Messages API-are not covered by this agreement. For these features, data may be retained by Anthropic in accordance with [Anthropic's ZDR documentation](https://platform.claude.com/docs/en/build-with-claude/zero-data-retention) . GitHub will update this page as ZDR coverage changes.

- Google Cloud: [Google commits to not training on GitHub data as part of their service terms](https://cloud.google.com/vertex-ai/generative-ai/docs/data-governance) . GitHub is additionally not subject to prompt logging for abuse monitoring.

To provide better service quality and reduce latency, GitHub uses [prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) . You can read more about prompt caching on [Anthropic PBC](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) , [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html) , and [Google Cloud](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude-prompt-caching) .

When using Claude, input prompts and output completions continue to run through GitHub Copilot's content filters for public code matching, when applied, along with those for harmful or offensive content.

### Google models

Used for:

- Gemini 2.5 Pro
- Gemini 3 Flash
- Gemini 3.1 Pro

GitHub Copilot uses Gemini 3.1 Pro, Gemini 3 Flash, and Gemini 2.5 Pro hosted on Google Cloud Platform (GCP). When using Gemini models, prompts and metadata are sent to GCP, which makes the [following data commitment](https://cloud.google.com/vertex-ai/generative-ai/docs/data-governance) : *Gemini doesn't use your prompts, or its responses, as data to train its models.*

To provide better service quality and reduce latency, GitHub uses [prompt caching](https://cloud.google.com/vertex-ai/generative-ai/docs/data-governance#customer_data_retention_and_achieving_zero_data_retention) .

When using Gemini models, input prompts and output completions continue to run through GitHub Copilot's content filters for public code matching, when applied, along with those for harmful or offensive content.

### xAI models

These models are hosted on xAI. xAI operates Grok Code Fast 1 in GitHub Copilot under a zero data retention API policy. This means xAI commits that user content (both inputs sent to the model and outputs generated by the model):

Will **not** be:

- Logged for any purpose, including human review
- Saved to disk or retained in any form, including as metadata
- Accessible by xAI personnel
- Used for model training

Will **only** :

- Exist temporarily in RAM for the minimum time required to process and respond to each request
- Be immediately deleted from memory once the response is delivered

When using xAI, input prompts and output completions continue to run through GitHub Copilot's content filters for public code matching, when applied, along with those for harmful or offensive content.

For more information, see [xAI's enterprise terms of service](https://x.ai/legal/terms-of-service-enterprise) on the xAI website.

### Inline suggestions

Inline suggestions, including ghost text and next edit suggestions, are powered by models hosted on Azure for Copilot Business and Copilot Enterprise plans. Copilot Free and Copilot Student user models are hosted on Fireworks AI.


### Features by IDE

The following table shows supported Copilot features in the latest version of each IDE.

| Feature               | VS Code   | Visual Studio   | JetBrains   | Eclipse   | Xcode   | NeoVim   |
|-----------------------|-----------|-----------------|-------------|-----------|---------|----------|
| .NET Upgrade Agent    | ✗         | ✓               | ✗           | ✗         | ✗       | ✗        |
| Agent skills          | ✓         | ✗               | P           | ✗         | ✗       | ✗        |
| Agent mode            | ✓         | ✓               | ✓           | ✓         | ✓       | ✗        |
| BYOK                  | P         | ✓               | P           | P         | P       | ✗        |
| Chat                  | ✓         | ✓               | ✓           | ✓         | ✓       | ✗        |
| Checkpoints           | ✓         | ✓               | ✓           | ✗         | ✓       | ✗        |
| Code completion       | ✓         | ✓               | ✓           | ✓         | ✓       | ✓        |
| Code referencing      | ✓         | ✓               | ✓           | ✓         | ✗       | ✗        |
| Copilot code review   | ✓         | ✓               | ✓           | ✗         | ✓       | ✗        |
| Custom agents         | ✓         | P               | P           | ✓         | P       | ✗        |
| Custom instructions   | ✓         | ✓               | P           | P         | P       | ✗        |
| Edit mode             | ✓         | C               | ✓           | ✗         | ✗       | ✗        |
| Java Upgrade Agent    | P         | ✗               | ✗           | ✗         | ✗       | ✗        |
| MCP                   | ✓         | ✓               | ✓           | ✓         | ✓       | ✗        |
| Next edit suggestions | ✓         | ✓               | P           | P         | P       | ✗        |
| Prompt files          | ✓         | ✓               | P           | ✗         | P       | ✗        |
| Vision                | P         | ✓               | P           | ✓         | P       | ✗        |
| Workspace indexing    | ✓         | ✓               | ✓           | ✓         | ✗       | ✗        |

### Features by VS Code version

The following table shows supported Copilot features across recent vesions of the IDE.

### VS Code latest releases

| Feature               | 1.108.0   | 1.107.0   | 1.106.0   | 1.105.0   | 1.104.0   |
|-----------------------|-----------|-----------|-----------|-----------|-----------|
| .NET Upgrade Agent    | ✗         | ✗         | ✗         | ✗         | ✗         |
| Agent skills          | ✓         | ✗         | ✗         | ✗         | ✗         |
| Agent mode            | ✓         | ✓         | ✓         | ✓         | ✓         |
| BYOK                  | P         | P         | P         | P         | P         |
| Chat                  | ✓         | ✓         | ✓         | ✓         | ✓         |
| Checkpoints           | ✓         | ✓         | ✓         | ✓         | ✓         |
| Code completion       | ✓         | ✓         | ✓         | ✓         | ✓         |
| Code referencing      | ✓         | ✓         | ✓         | ✓         | ✓         |
| Copilot code review   | ✓         | ✓         | ✓         | ✓         | ✓         |
| Custom agents         | ✓         | ✓         | P         | P         | P         |
| Custom instructions   | ✓         | ✓         | ✓         | ✓         | ✓         |
| Edit mode             | ✓         | ✓         | ✓         | ✓         | ✓         |
| Java Upgrade Agent    | P         | P         | P         | P         | P         |
| MCP                   | ✓         | ✓         | ✓         | ✓         | ✓         |
| Next edit suggestions | ✓         | ✓         | ✓         | ✓         | ✓         |
| Prompt files          | ✓         | ✓         | P         | P         | P         |
| Vision                | P         | P         | P         | P         | P         |
| Workspace indexing    | ✓         | ✓         | ✓         | ✓         | ✓         |

### VS Code 2025 releases

| Feature               | 1.108.0   | 1.107.0   | 1.106.0   | 1.105.0   | 1.104.0   | 1.103.0   | 1.102.0   | 1.101.0   | 1.100.0   | 1.99.0   | 1.98.0   | 1.97.0   |
|-----------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|----------|----------|----------|
| .NET Upgrade Agent    | ✗         | ✗         | ✗         | ✗         | ✗         | ✗         | ✗         | ✗         | ✗         | ✗        | ✗        | ✗        |
| Agent skills          | ✓         | ✗         | ✗         | ✗         | ✗         | ✗         | ✗         | ✗         | ✗         | ✗        | ✗        | ✗        |
| Agent mode            | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓        | P        | P        |
| BYOK                  | P         | P         | P         | P         | P         | P         | P         | P         | P         | P        | P        | P        |
| Chat                  | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓        | ✓        | ✓        |
| Checkpoints           | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✗         | ✗         | ✗         | ✗        | ✗        | ✗        |
| Code completion       | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓        | ✓        | ✓        |
| Code referencing      | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓        | ✓        | ✓        |
| Copilot code review   | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓        | P        | P        |
| Custom agents         | ✓         | ✓         | P         | P         | P         | P         | P         | P         | ✗         | ✗        | ✗        | ✗        |
| Custom instructions   | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓        | ✓        | P        |
| Edit mode             | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓        | ✓        | ✓        |
| Java Upgrade Agent    | P         | P         | P         | P         | P         | P         | P         | P         | P         | P        | P        | P        |
| MCP                   | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | P         | P         | P        | ✗        | ✗        |
| Next edit suggestions | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓        | P        | P        |
| Prompt files          | ✓         | ✓         | P         | P         | P         | P         | P         | P         | P         | P        | P        | P        |
| Vision                | P         | P         | P         | P         | P         | P         | P         | P         | P         | P        | P        | P        |
| Workspace indexing    | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓         | ✓        | ✓        | ✓        |

### VS Code 2024 releases

| Feature               | 1.96.0   | 1.95.0   | 1.94.0   |
|-----------------------|----------|----------|----------|
| .NET Upgrade Agent    | ✗        | ✗        | ✗        |
| Agent skills          | ✗        | ✗        | ✗        |
| Agent mode            | ✗        | ✗        | ✗        |
| BYOK                  | P        | P        | P        |
| Chat                  | ✓        | ✓        | ✓        |
| Checkpoints           | ✗        | ✗        | ✗        |
| Code completion       | ✓        | ✓        | ✓        |
| Code referencing      | ✓        | ✓        | ✓        |
| Copilot code review   | P        | P        | ✗        |
| Custom agents         | ✗        | ✗        | ✗        |
| Custom instructions   | P        | P        | ✗        |
| Edit mode             | P        | P        | ✗        |
| Java Upgrade Agent    | P        | P        | P        |
| MCP                   | ✗        | ✗        | ✗        |
| Next edit suggestions | ✗        | ✗        | ✗        |
| Prompt files          | ✗        | ✗        | ✗        |
| Vision                | ✗        | ✗        | ✗        |
| Workspace indexing    | ✓        | ✓        | ✗        |

### VS Code 2023 releases

| Feature               | 1.80.0   |
|-----------------------|----------|
| .NET Upgrade Agent    | ✗        |
| Agent skills          | ✗        |
| Agent mode            | ✗        |
| BYOK                  | P        |
| Chat                  | ✓        |
| Checkpoints           | ✗        |
| Code completion       | ✓        |
| Code referencing      | ✗        |
| Copilot code review   | ✗        |
| Custom agents         | ✗        |
| Custom instructions   | ✗        |
| Edit mode             | ✗        |
| Java Upgrade Agent    | P        |
| MCP                   | ✗        |
| Next edit suggestions | ✗        |
| Prompt files          | ✗        |
| Vision                | ✗        |
| Workspace indexing    | ✗        |

### VS Code 2022 releases

| Feature               | 1.70.0   |
|-----------------------|----------|
| .NET Upgrade Agent    | ✗        |
| Agent skills          | ✗        |
| Agent mode            | ✗        |
| BYOK                  | P        |
| Chat                  | ✓        |
| Checkpoints           | ✗        |
| Code completion       | ✓        |
| Code referencing      | ✗        |
| Copilot code review   | ✗        |
| Custom agents         | ✗        |
| Custom instructions   | ✗        |
| Edit mode             | ✗        |
| Java Upgrade Agent    | P        |
| MCP                   | ✗        |
| Next edit suggestions | ✗        |
| Prompt files          | ✗        |
| Vision                | ✗        |
| Workspace indexing    | ✗        |

### VS Code 2021 releases

| Feature               | 1.60.0   | 1.57.0   |
|-----------------------|----------|----------|
| .NET Upgrade Agent    | ✗        | ✗        |
| Agent skills          | ✗        | ✗        |
| Agent mode            | ✗        | ✗        |
| BYOK                  | P        | P        |
| Chat                  | ✗        | ✗        |
| Checkpoints           | ✗        | ✗        |
| Code completion       | ✓        | P        |
| Code referencing      | ✗        | ✗        |
| Copilot code review   | ✗        | ✗        |
| Custom agents         | ✗        | ✗        |
| Custom instructions   | ✗        | ✗        |
| Edit mode             | ✗        | ✗        |
| Java Upgrade Agent    | P        | P        |
| MCP                   | ✗        | ✗        |
| Next edit suggestions | ✗        | ✗        |
| Prompt files          | ✗        | ✗        |
| Vision                | ✗        | ✗        |
| Workspace indexing    | ✗        | ✗        |

### Features by Visual Studio version

The following table shows supported Copilot features across recent vesions of the IDE.

### Visual Studio latest releases

| Feature               | 18.0.0   | 17.14.13   | 17.14.6   | 17.14.0   | 17.13.0   |
|-----------------------|----------|------------|-----------|-----------|-----------|
| .NET Upgrade Agent    | ✓        | P          | P         | P         | P         |
| Agent skills          | ✗        | ✗          | ✗         | ✗         | ✗         |
| Agent mode            | ✓        | ✓          | ✓         | P         | ✗         |
| BYOK                  | ✓        | ✗          | ✗         | ✗         | ✗         |
| Chat                  | ✓        | ✓          | ✓         | ✓         | ✓         |
| Checkpoints           | ✓        | ✓          | ✓         | ✓         | ✗         |
| Code completion       | ✓        | ✓          | ✓         | ✓         | ✓         |
| Code referencing      | ✓        | ✓          | ✓         | ✓         | ✓         |
| Copilot code review   | ✓        | ✓          | ✓         | ✓         | ✓         |
| Custom agents         | P        | ✗          | ✗         | ✗         | ✗         |
| Custom instructions   | ✓        | ✓          | ✓         | ✓         | ✓         |
| Edit mode             | C        | ✗          | ✓         | ✓         | ✓         |
| MCP                   | ✓        | ✓          | P         | P         | ✗         |
| Next edit suggestions | ✓        | ✓          | ✓         | ✓         | ✗         |
| Prompt files          | ✓        | ✗          | ✗         | ✗         | ✗         |
| Vision                | ✓        | ✓          | ✓         | ✓         | P         |
| Workspace indexing    | ✓        | ✗          | ✗         | ✗         | ✗         |

### Visual Studio 2025 releases

| Feature               | 18.0.0   | 17.14.13   | 17.14.6   | 17.14.0   | 17.13.0   |
|-----------------------|----------|------------|-----------|-----------|-----------|
| .NET Upgrade Agent    | ✓        | P          | P         | P         | P         |
| Agent skills          | ✗        | ✗          | ✗         | ✗         | ✗         |
| Agent mode            | ✓        | ✓          | ✓         | P         | ✗         |
| BYOK                  | ✓        | ✗          | ✗         | ✗         | ✗         |
| Chat                  | ✓        | ✓          | ✓         | ✓         | ✓         |
| Checkpoints           | ✓        | ✓          | ✓         | ✓         | ✗         |
| Code completion       | ✓        | ✓          | ✓         | ✓         | ✓         |
| Code referencing      | ✓        | ✓          | ✓         | ✓         | ✓         |
| Copilot code review   | ✓        | ✓          | ✓         | ✓         | ✓         |
| Custom agents         | P        | ✗          | ✗         | ✗         | ✗         |
| Custom instructions   | ✓        | ✓          | ✓         | ✓         | ✓         |
| Edit mode             | C        | ✗          | ✓         | ✓         | ✓         |
| MCP                   | ✓        | ✓          | P         | P         | ✗         |
| Next edit suggestions | ✓        | ✓          | ✓         | ✓         | ✗         |
| Prompt files          | ✓        | ✗          | ✗         | ✗         | ✗         |
| Vision                | ✓        | ✓          | ✓         | ✓         | P         |
| Workspace indexing    | ✓        | ✗          | ✗         | ✗         | ✗         |

### Features by JetBrains version

The following table shows supported Copilot features across recent vesions of the GitHub Copilot Extension for the IDE.

### JetBrains latest releases

| Feature               | 1.5.66   | 1.5.65   | 1.5.64   | 1.5.63   | 1.5.62   |
|-----------------------|----------|----------|----------|----------|----------|
| Agent skills          | P        | P        | P        | ✗        | ✗        |
| Agent mode            | ✓        | ✓        | ✓        | ✓        | ✓        |
| BYOK                  | P        | P        | P        | P        | P        |
| Chat                  | ✓        | ✓        | ✓        | ✓        | ✓        |
| Checkpoints           | ✓        | ✓        | ✓        | ✓        | ✓        |
| Code completion       | ✓        | ✓        | ✓        | ✓        | ✓        |
| Code referencing      | ✓        | ✓        | ✓        | ✓        | ✓        |
| Copilot code review   | ✓        | ✓        | ✓        | ✓        | ✓        |
| Custom instructions   | P        | P        | P        | P        | P        |
| Custom agents         | P        | P        | P        | P        | P        |
| Edit mode             | ✓        | ✓        | ✓        | ✓        | ✓        |
| MCP                   | ✓        | ✓        | ✓        | ✓        | ✓        |
| Next edit suggestions | P        | P        | P        | P        | P        |
| Prompt files          | P        | P        | P        | P        | P        |
| Vision                | P        | P        | P        | P        | P        |
| Workspace indexing    | ✓        | ✓        | ✓        | ✓        | ✓        |

### JetBrains 2026 releases

| Feature               | 1.5.66   | 1.5.65   | 1.5.64   | 1.5.63   |
|-----------------------|----------|----------|----------|----------|
| Agent skills          | P        | P        | P        | ✗        |
| Agent mode            | ✓        | ✓        | ✓        | ✓        |
| BYOK                  | P        | P        | P        | P        |
| Chat                  | ✓        | ✓        | ✓        | ✓        |
| Checkpoints           | ✓        | ✓        | ✓        | ✓        |
| Code completion       | ✓        | ✓        | ✓        | ✓        |
| Code referencing      | ✓        | ✓        | ✓        | ✓        |
| Copilot code review   | ✓        | ✓        | ✓        | ✓        |
| Custom instructions   | P        | P        | P        | P        |
| Custom agents         | P        | P        | P        | P        |
| Edit mode             | ✓        | ✓        | ✓        | ✓        |
| MCP                   | ✓        | ✓        | ✓        | ✓        |
| Next edit suggestions | P        | P        | P        | P        |
| Prompt files          | P        | P        | P        | P        |
| Vision                | P        | P        | P        | P        |
| Workspace indexing    | ✓        | ✓        | ✓        | ✓        |

### JetBrains 2025 releases

| Feature               | 1.5.62   | 1.5.54   | 1.5.53   | 1.5.49   | 1.5.45   | 1.5.43   | 1.5.41   | 1.5.0   | 1.0.1   |
|-----------------------|----------|----------|----------|----------|----------|----------|----------|---------|---------|
| Agent skills          | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       | ✗       |
| Agent mode            | ✓        | ✓        | ✓        | ✓        | P        | ✗        | ✗        | ✗       | ✗       |
| BYOK                  | P        | P        | P        | P        | P        | P        | P        | P       | P       |
| Chat                  | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓       | ✗       |
| Checkpoints           | ✓        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       | ✗       |
| Code completion       | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓       | ✓       |
| Code referencing      | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓       | ✓       |
| Copilot code review   | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓       | ✓       |
| Custom instructions   | P        | P        | P        | P        | P        | P        | P        | ✗       | ✗       |
| Custom agents         | P        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       | ✗       |
| Edit mode             | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | P        | ✗       | ✗       |
| MCP                   | ✓        | ✓        | ✓        | P        | P        | ✗        | ✗        | ✗       | ✗       |
| Next edit suggestions | P        | P        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       | ✗       |
| Prompt files          | P        | P        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       | ✗       |
| Vision                | P        | P        | P        | P        | P        | P        | P        | ✗       | ✗       |
| Workspace indexing    | ✓        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       | ✗       |

### JetBrains 2024 releases

| Feature               | 1.5.39   | 1.4.0   |
|-----------------------|----------|---------|
| Agent skills          | ✗        | ✗       |
| Agent mode            | ✗        | ✗       |
| BYOK                  | P        | P       |
| Chat                  | ✓        | P       |
| Checkpoints           | ✗        | ✗       |
| Code completion       | ✓        | ✓       |
| Code referencing      | ✓        | ✓       |
| Copilot code review   | ✓        | ✓       |
| Custom instructions   | ✗        | ✗       |
| Custom agents         | ✗        | ✗       |
| Edit mode             | P        | ✗       |
| MCP                   | ✗        | ✗       |
| Next edit suggestions | ✗        | ✗       |
| Prompt files          | ✗        | ✗       |
| Vision                | ✗        | ✗       |
| Workspace indexing    | ✗        | ✗       |

### Features by Eclipse version

The following table shows supported Copilot features across recent vesions of the GitHub Copilot Extension for the IDE.

### Eclipse latest releases

| Feature               | 0.14.0   | 0.13.0   | 0.12.0   | 0.11.0   | 0.10.0   |
|-----------------------|----------|----------|----------|----------|----------|
| Agent skills          | ✗        | ✗        | ✗        | ✗        | ✗        |
| Agent mode            | ✓        | ✓        | ✓        | ✓        | ✓        |
| BYOK                  | P        | P        | P        | ✗        | ✗        |
| Chat                  | ✓        | ✓        | ✓        | ✓        | ✓        |
| Checkpoints           | ✗        | ✗        | ✗        | ✗        | ✗        |
| Code completion       | ✓        | ✓        | ✓        | ✓        | ✓        |
| Code referencing      | ✓        | ✓        | ✓        | ✓        | ✓        |
| Copilot code review   | ✗        | ✗        | ✗        | ✗        | ✗        |
| Custom agents         | ✓        | ✓        | ✗        | ✗        | ✗        |
| Custom instructions   | P        | P        | P        | P        | P        |
| Java Upgrade Agent    | ✗        | ✗        | ✗        | ✗        | ✗        |
| MCP                   | ✓        | ✓        | ✓        | ✓        | ✓        |
| Next edit suggestions | P        | P        | ✗        | ✗        | ✗        |
| Prompt files          | ✗        | ✗        | ✗        | ✗        | ✗        |
| Vision                | ✓        | ✓        | ✓        | ✓        | ✓        |
| Workspace indexing    | ✓        | ✓        | ✓        | ✓        | ✓        |

### Eclipse 2025 releases

| Feature               | 0.14.0   | 0.13.0   | 0.12.0   | 0.11.0   | 0.10.0   | 0.9.0   | 0.8.0   | 0.7.0   | 0.6.0   | 0.5.0   | 0.4.0   | 0.3.0   | 0.2.0   | 0.1.0   |
|-----------------------|----------|----------|----------|----------|----------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| Agent skills          | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |
| Agent mode            | ✓        | ✓        | ✓        | ✓        | ✓        | ✓       | P       | P       | P       | ✗       | ✗       | ✗       | ✗       | ✗       |
| BYOK                  | P        | P        | P        | ✗        | ✗        | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |
| Chat                  | ✓        | ✓        | ✓        | ✓        | ✓        | ✓       | ✓       | ✓       | ✓       | ✓       | P       | P       | ✗       | ✗       |
| Checkpoints           | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |
| Code completion       | ✓        | ✓        | ✓        | ✓        | ✓        | ✓       | ✓       | ✓       | ✓       | ✓       | ✓       | ✓       | P       | P       |
| Code referencing      | ✓        | ✓        | ✓        | ✓        | ✓        | ✓       | ✓       | ✓       | ✓       | ✓       | ✓       | ✓       | ✓       | ✓       |
| Copilot code review   | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |
| Custom agents         | ✓        | ✓        | ✗        | ✗        | ✗        | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |
| Custom instructions   | P        | P        | P        | P        | P        | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |
| Java Upgrade Agent    | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |
| MCP                   | ✓        | ✓        | ✓        | ✓        | ✓        | P       | P       | P       | P       | ✗       | ✗       | ✗       | ✗       | ✗       |
| Next edit suggestions | P        | P        | ✗        | ✗        | ✗        | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |
| Prompt files          | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |
| Vision                | ✓        | ✓        | ✓        | ✓        | ✓        | ✓       | ✓       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |
| Workspace indexing    | ✓        | ✓        | ✓        | ✓        | ✓        | ✓       | ✓       | ✓       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |

### Features by Xcode version

The following table shows supported Copilot features across recent vesions of the GitHub Copilot Extension for the IDE.

### Xcode latest releases

| Feature               | 0.46.0   | 0.45.0   | 0.44.0   | 0.43.0   | 0.42.0   |
|-----------------------|----------|----------|----------|----------|----------|
| Agent skills          | ✗        | ✗        | ✗        | ✗        | ✗        |
| Agent mode            | ✓        | ✓        | ✓        | ✓        | ✓        |
| BYOK                  | P        | P        | P        | P        | P        |
| Chat                  | ✓        | ✓        | ✓        | ✓        | ✓        |
| Checkpoints           | ✓        | ✓        | ✓        | ✗        | ✗        |
| Code completion       | ✓        | ✓        | ✓        | ✓        | ✓        |
| Code referencing      | ✗        | ✗        | ✗        | ✗        | ✗        |
| Copilot code review   | ✓        | ✓        | ✓        | ✓        | ✓        |
| Custom agents         | P        | P        | ✗        | ✗        | ✗        |
| Custom instructions   | P        | P        | P        | P        | P        |
| MCP                   | ✓        | ✓        | ✓        | ✓        | ✓        |
| Next edit suggestions | P        | P        | ✗        | ✗        | ✗        |
| Prompt files          | P        | P        | P        | P        | P        |
| Vision                | P        | P        | P        | P        | P        |
| Workspace indexing    | ✗        | ✗        | ✗        | ✗        | ✗        |

### Xcode 2025 releases

| Feature               | 0.46.0   | 0.45.0   | 0.44.0   | 0.43.0   | 0.42.0   | 0.41.0   | 0.40.0   | 0.39.0   | 0.38.0   | 0.37.0   | 0.36.0   | 0.35.0   | 0.34.0   | 0.33.0   | 0.32.0   | 0.31.0   | 0.30.0   |
|-----------------------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| Agent skills          | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        |
| Agent mode            | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | P        | P        | P        | ✗        | ✗        | ✗        | ✗        | ✗        |
| BYOK                  | P        | P        | P        | P        | P        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        |
| Chat                  | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | P        | ✗        |
| Checkpoints           | ✓        | ✓        | ✓        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        |
| Code completion       | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | P        |
| Code referencing      | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        |
| Copilot code review   | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        |
| Custom agents         | P        | P        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        |
| Custom instructions   | P        | P        | P        | P        | P        | P        | P        | P        | P        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        |
| MCP                   | ✓        | ✓        | ✓        | ✓        | ✓        | ✓        | P        | P        | P        | P        | P        | P        | ✗        | ✗        | ✗        | ✗        | ✗        |
| Next edit suggestions | P        | P        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        |
| Prompt files          | P        | P        | P        | P        | P        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        |
| Vision                | P        | P        | P        | P        | P        | P        | P        | P        | P        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        |
| Workspace indexing    | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        |

### Xcode 2024 releases

| Feature               | 0.29.0   | 0.28.0   | 0.27.0   | 0.26.0   | 0.25.0   | 0.24.0   | 0.23.0   | 0.0.0   |
|-----------------------|----------|----------|----------|----------|----------|----------|----------|---------|
| Agent skills          | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       |
| Agent mode            | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       |
| BYOK                  | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       |
| Chat                  | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       |
| Checkpoints           | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       |
| Code completion       | P        | P        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       |
| Code referencing      | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       |
| Copilot code review   | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       |
| Custom agents         | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       |
| Custom instructions   | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       |
| MCP                   | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       |
| Next edit suggestions | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       |
| Prompt files          | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       |
| Vision                | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       |
| Workspace indexing    | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗        | ✗       |

### Features by NeoVim version

The following table shows supported Copilot features across recent vesions of the GitHub Copilot Extension for the IDE.

### NeoVim latest releases

| Feature               | 1.18.0   | 1.17.0   | 1.16.0   | 1.15.0   | 1.14.0   |
|-----------------------|----------|----------|----------|----------|----------|
| Agent skills          | ✗        | ✗        | ✗        | ✗        | ✗        |
| Agent mode            | ✗        | ✗        | ✗        | ✗        | ✗        |
| BYOK                  | ✗        | ✗        | ✗        | ✗        | ✗        |
| Chat                  | ✗        | ✗        | ✗        | ✗        | ✗        |
| Checkpoints           | ✗        | ✗        | ✗        | ✗        | ✗        |
| Code completion       | ✓        | ✓        | ✓        | ✓        | ✓        |
| Code referencing      | ✗        | ✗        | ✗        | ✗        | ✗        |
| Copilot code review   | ✗        | ✗        | ✗        | ✗        | ✗        |
| Custom agents         | ✗        | ✗        | ✗        | ✗        | ✗        |
| Custom instructions   | ✗        | ✗        | ✗        | ✗        | ✗        |
| MCP                   | ✗        | ✗        | ✗        | ✗        | ✗        |
| Next edit suggestions | ✗        | ✗        | ✗        | ✗        | ✗        |
| Prompt files          | ✗        | ✗        | ✗        | ✗        | ✗        |
| Vision                | ✗        | ✗        | ✗        | ✗        | ✗        |
| Workspace indexing    | ✗        | ✗        | ✗        | ✗        | ✗        |

### NeoVim 2024 releases

| Feature               | 1.18.0   | 1.17.0   | 1.16.0   | 1.15.0   | 1.14.0   |
|-----------------------|----------|----------|----------|----------|----------|
| Agent skills          | ✗        | ✗        | ✗        | ✗        | ✗        |
| Agent mode            | ✗        | ✗        | ✗        | ✗        | ✗        |
| BYOK                  | ✗        | ✗        | ✗        | ✗        | ✗        |
| Chat                  | ✗        | ✗        | ✗        | ✗        | ✗        |
| Checkpoints           | ✗        | ✗        | ✗        | ✗        | ✗        |
| Code completion       | ✓        | ✓        | ✓        | ✓        | ✓        |
| Code referencing      | ✗        | ✗        | ✗        | ✗        | ✗        |
| Copilot code review   | ✗        | ✗        | ✗        | ✗        | ✗        |
| Custom agents         | ✗        | ✗        | ✗        | ✗        | ✗        |
| Custom instructions   | ✗        | ✗        | ✗        | ✗        | ✗        |
| MCP                   | ✗        | ✗        | ✗        | ✗        | ✗        |
| Next edit suggestions | ✗        | ✗        | ✗        | ✗        | ✗        |
| Prompt files          | ✗        | ✗        | ✗        | ✗        | ✗        |
| Vision                | ✗        | ✗        | ✗        | ✗        | ✗        |
| Workspace indexing    | ✗        | ✗        | ✗        | ✗        | ✗        |

### NeoVim 2023 releases

| Feature               | 1.13.0   | 1.12.0   | 1.11.0   | 1.10.0   | 1.9.0   |
|-----------------------|----------|----------|----------|----------|---------|
| Agent skills          | ✗        | ✗        | ✗        | ✗        | ✗       |
| Agent mode            | ✗        | ✗        | ✗        | ✗        | ✗       |
| BYOK                  | ✗        | ✗        | ✗        | ✗        | ✗       |
| Chat                  | ✗        | ✗        | ✗        | ✗        | ✗       |
| Checkpoints           | ✗        | ✗        | ✗        | ✗        | ✗       |
| Code completion       | ✓        | ✓        | ✓        | ✓        | ✓       |
| Code referencing      | ✗        | ✗        | ✗        | ✗        | ✗       |
| Copilot code review   | ✗        | ✗        | ✗        | ✗        | ✗       |
| Custom agents         | ✗        | ✗        | ✗        | ✗        | ✗       |
| Custom instructions   | ✗        | ✗        | ✗        | ✗        | ✗       |
| MCP                   | ✗        | ✗        | ✗        | ✗        | ✗       |
| Next edit suggestions | ✗        | ✗        | ✗        | ✗        | ✗       |
| Prompt files          | ✗        | ✗        | ✗        | ✗        | ✗       |
| Vision                | ✗        | ✗        | ✗        | ✗        | ✗       |
| Workspace indexing    | ✗        | ✗        | ✗        | ✗        | ✗       |

### NeoVim 2022 releases

| Feature               | 1.8.0   | 1.7.0   | 1.6.0   | 1.5.0   | 1.4.0   | 1.3.0   | 1.2.0   | 1.1.0   |
|-----------------------|---------|---------|---------|---------|---------|---------|---------|---------|
| Agent skills          | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |
| Agent mode            | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |
| BYOK                  | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |
| Chat                  | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |
| Checkpoints           | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |
| Code completion       | ✓       | ✓       | ✓       | ✓       | ✓       | ✓       | ✓       | ✓       |
| Code referencing      | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |
| Copilot code review   | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |
| Custom agents         | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |
| Custom instructions   | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |
| MCP                   | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |
| Next edit suggestions | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |
| Prompt files          | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |
| Vision                | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |
| Workspace indexing    | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       | ✗       |

### NeoVim 2021 releases

| Feature               | 1.0.0   | 0.0.1   |
|-----------------------|---------|---------|
| Agent skills          | ✗       | ✗       |
| Agent mode            | ✗       | ✗       |
| BYOK                  | ✗       | ✗       |
| Chat                  | ✗       | ✗       |
| Checkpoints           | ✗       | ✗       |
| Code completion       | ✓       | P       |
| Code referencing      | ✗       | ✗       |
| Copilot code review   | ✗       | ✗       |
| Custom agents         | ✗       | ✗       |
| Custom instructions   | ✗       | ✗       |
| MCP                   | ✗       | ✗       |
| Next edit suggestions | ✗       | ✗       |
| Prompt files          | ✗       | ✗       |
| Vision                | ✗       | ✗       |
| Workspace indexing    | ✗       | ✗       |


### Keyboard shortcuts for macOS

| Action                                                        | Shortcut                       |
|---------------------------------------------------------------|--------------------------------|
| Accept an inline suggestion                                   | `Tab`                          |
| Dismiss an inline suggestion                                  | `Esc`                          |
| Show next inline suggestion                                   | `Option (⌥) or Alt` + `]`      |
| Show previous inline suggestion                               | `Option (⌥) or Alt` + `[`      |
| Trigger inline suggestion                                     | `Option (⌥)` + `\`             |
| Open GitHub Copilot (additional suggestions in separate pane) | `Option (⌥) or Alt` + `Return` |

### Keyboard shortcuts for Windows

| Action                                                        | Shortcut        |
|---------------------------------------------------------------|-----------------|
| Accept an inline suggestion                                   | `Tab`           |
| Dismiss an inline suggestion                                  | `Esc`           |
| Show next inline suggestion                                   | `Alt` + `]`     |
| Show previous inline suggestion                               | `Alt` + `[`     |
| Trigger inline suggestion                                     | `Alt` + `\`     |
| Open GitHub Copilot (additional suggestions in separate pane) | `Alt` + `Enter` |

### Keyboard shortcuts for Linux

| Action                                                        | Shortcut        |
|---------------------------------------------------------------|-----------------|
| Accept an inline suggestion                                   | `Tab`           |
| Dismiss an inline suggestion                                  | `Esc`           |
| Show next inline suggestion                                   | `Alt` + `]`     |
| Show previous inline suggestion                               | `Alt` + `[`     |
| Trigger inline suggestion                                     | `Alt` + `\`     |
| Open GitHub Copilot (additional suggestions in separate pane) | `Alt` + `Enter` |

You can use the default keyboard shortcuts for inline suggestions in Visual Studio when using GitHub Copilot. You can search for each keyboard shortcut by its command name in the Keyboard Shortcuts editor.

| Action                          | Shortcut    | Command name            |
|---------------------------------|-------------|-------------------------|
| Show next inline suggestion     | `Alt` + `.` | Edit.NextSuggestion     |
| Show previous inline suggestion | `Alt` + `,` | Edit.PreviousSuggestion |

You can use the default keyboard shortcuts for GitHub Copilot in Visual Studio Code. Search keyboard shortcuts by command name in the Keyboard Shortcuts editor.

### Keyboard shortcuts for macOS

| Action                                                        | Shortcut              | Command name                             |
|---------------------------------------------------------------|-----------------------|------------------------------------------|
| Accept an inline suggestion                                   | `Tab`                 | editor.action.inlineSuggest.commit       |
| Dismiss an inline suggestion                                  | `Esc`                 | editor.action.inlineSuggest.hide         |
| Show next inline suggestion                                   | `Option (⌥)` + `]`    | editor.action.inlineSuggest.showNext     |
| Show previous inline suggestion                               | `Option (⌥)` + `[`    | editor.action.inlineSuggest.showPrevious |
| Trigger inline suggestion                                     | `Option (⌥)` + `\`    | editor.action.inlineSuggest.trigger      |
| Open GitHub Copilot (additional suggestions in separate pane) | `Ctrl` + `Return`     | github.copilot.generate                  |
| Toggle GitHub Copilot on/off                                  | *No default shortcut* | github.copilot.toggleCopilot             |

### Keyboard shortcuts for Windows

| Action                                                        | Shortcut              | Command name                             |
|---------------------------------------------------------------|-----------------------|------------------------------------------|
| Accept an inline suggestion                                   | `Tab`                 | editor.action.inlineSuggest.commit       |
| Dismiss an inline suggestion                                  | `Esc`                 | editor.action.inlineSuggest.hide         |
| Show next inline suggestion                                   | `Alt` + `]`           | editor.action.inlineSuggest.showNext     |
| Show previous inline suggestion                               | `Alt` + `[`           | editor.action.inlineSuggest.showPrevious |
| Trigger inline suggestion                                     | `Alt` + `\`           | editor.action.inlineSuggest.trigger      |
| Open GitHub Copilot (additional suggestions in separate pane) | `Ctrl` + `Enter`      | github.copilot.generate                  |
| Toggle GitHub Copilot on/off                                  | *No default shortcut* | github.copilot.toggleCopilot             |

### Keyboard shortcuts for Linux

| Action                                                        | Shortcut              | Command name                             |
|---------------------------------------------------------------|-----------------------|------------------------------------------|
| Accept an inline suggestion                                   | `Tab`                 | editor.action.inlineSuggest.commit       |
| Dismiss an inline suggestion                                  | `Esc`                 | editor.action.inlineSuggest.hide         |
| Show next inline suggestion                                   | `Alt` + `]`           | editor.action.inlineSuggest.showNext     |
| Show previous inline suggestion                               | `Alt` + `[`           | editor.action.inlineSuggest.showPrevious |
| Trigger inline suggestion                                     | `Alt` + `\`           | editor.action.inlineSuggest.trigger      |
| Open GitHub Copilot (additional suggestions in separate pane) | `Ctrl` + `Enter`      | github.copilot.generate                  |
| Toggle GitHub Copilot on/off                                  | *No default shortcut* | github.copilot.toggleCopilot             |

You can use the default keyboard shortcuts for inline suggestions in Xcode when using GitHub Copilot. Alternatively, you can rebind the shortcuts to your preferred keyboard shortcuts for each specific command.

| Action                                | Shortcut         |
|---------------------------------------|------------------|
| Accept the first line of a suggestion | `Tab`            |
| View full suggestion                  | Hold `Option`    |
| Accept full suggestion                | `Option` + `Tab` |

You can use the default keyboard shortcuts for inline suggestions in Eclipse when using GitHub Copilot.

| Action                                   | Shortcut                                                               |
|------------------------------------------|------------------------------------------------------------------------|
| Accept an inline suggestion              | `Tab`                                                                  |
| Accept next word of an inline suggestion | `Command` + `→` (Mac) or `Ctrl` + `→` (Windows)                        |
| Dismiss an inline suggestion             | `Esc`                                                                  |
| Trigger inline suggestion                | `Option (⌥)` + `Command` + `/` (Mac) or `Alt` + `Ctrl` + `/` (Windows) |

You can rebind the keyboard shortcuts in Vim/Neovim when using GitHub Copilot to use your preferred keyboard shortcuts for each specific command. For more information, see the [Map](https://neovim.io/doc/user/map.html) article in the Neovim documentation.


### Command-line commands

| Command                | Purpose                                                                                                                                     |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| `copilot`              | Launch the interactive user interface.                                                                                                      |
| `copilot help [topic]` | Display help information. Help topics include: `config` , `commands` , `environment` , `logging` , `permissions` , and `providers` .        |
| `copilot init`         | Initialize Copilot custom instructions for this repository.                                                                                 |
| `copilot update`       | Download and install the latest version.                                                                                                    |
| `copilot version`      | Display version information and check for updates.                                                                                          |
| `copilot login`        | Authenticate with Copilot via the OAuth device flow. Accepts `--host HOST` to specify the GitHub host URL (default: `https://github.com` ). |
| `copilot logout`       | Sign out of GitHub and remove stored credentials.                                                                                           |
| `copilot plugin`       | Manage plugins and plugin marketplaces.                                                                                                     |

### Global shortcuts in the interactive interface

| Shortcut              | Purpose                                                                                                                                                           |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `@ FILENAME`          | Include file contents in the context.                                                                                                                             |
| `Ctrl` + `X` then `/` | After you have started typing a prompt, this allows you to run a slash command-for example, if you want to change the model without having to retype your prompt. |
| `Esc`                 | Cancel the current operation.                                                                                                                                     |
| `! COMMAND`           | Execute a command in your local shell, bypassing Copilot.                                                                                                         |
| `Ctrl` + `C`          | Cancel operation / clear input. Press twice to exit.                                                                                                              |
| `Ctrl` + `D`          | Shutdown.                                                                                                                                                         |
| `Ctrl` + `L`          | Clear the screen.                                                                                                                                                 |
| `Shift` + `Tab`       | Cycle between standard, plan, and autopilot mode.                                                                                                                 |

### Timeline shortcuts in the interactive interface

| Shortcut   | Purpose                                                                                                                    |
|------------|----------------------------------------------------------------------------------------------------------------------------|
| ctrl+o     | While there is nothing in the prompt input, this expands recent items in Copilot's response timeline to show more details. |
| ctrl+e     | While there is nothing in the prompt input, this expands all items in Copilot's response timeline.                         |
| ctrl+t     | Expand/collapse display of reasoning in responses.                                                                         |

### Navigation shortcuts in the interactive interface

| Shortcut           | Purpose                                                                                                |
|--------------------|--------------------------------------------------------------------------------------------------------|
| `Ctrl` + `A`       | Move to beginning of the line (when typing).                                                           |
| `Ctrl` + `B`       | Move to the previous character.                                                                        |
| `Ctrl` + `E`       | Move to end of the line (when typing).                                                                 |
| `Ctrl` + `F`       | Move to the next character.                                                                            |
| `Ctrl` + `G`       | Edit the prompt in an external editor.                                                                 |
| `Ctrl` + `H`       | Delete the previous character.                                                                         |
| `Ctrl` + `K`       | Delete from cursor to end of the line. If the cursor is at the end of the line, delete the line break. |
| `Ctrl` + `U`       | Delete from cursor to beginning of the line.                                                           |
| `Ctrl` + `W`       | Delete the previous word.                                                                              |
| `Home`             | Move to the start of the current line.                                                                 |
| `End`              | Move to the end of the current line.                                                                   |
| `Ctrl` + `Home`    | Move to the start of the text.                                                                         |
| `Ctrl` + `End`     | Move to the end of the text.                                                                           |
| `Meta` + `←` / `→` | Move the cursor by a word.                                                                             |
| `↑` / `↓`          | Navigate the command history.                                                                          |

### Slash commands in the interactive interface

| Command                                                                                                   | Purpose                                                                                                                                                         |
|-----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `/add-dir PATH`                                                                                           | Add a directory to the allowed list for file access.                                                                                                            |
| `/agent`                                                                                                  | Browse and select from available agents (if any).                                                                                                               |
| `/allow-all` , `/yolo`                                                                                    | Enable all permissions (tools, paths, and URLs).                                                                                                                |
| `/changelog [SUMMARIZE] [VERSION]`                                                                        | Display the CLI changelog with an optional AI-generated summary.                                                                                                |
| `/clear [PROMPT]` , `/new [PROMPT]`                                                                       | Start a new conversation.                                                                                                                                       |
| `/compact`                                                                                                | Summarize the conversation history to reduce context window usage.                                                                                              |
| `/context`                                                                                                | Show the context window token usage and visualization.                                                                                                          |
| `/copy`                                                                                                   | Copy the last response to the clipboard.                                                                                                                        |
| `/cwd` , `/cd [PATH]`                                                                                     | Change the working directory or display the current directory.                                                                                                  |
| `/delegate [PROMPT]`                                                                                      | Delegate changes to a remote repository with an AI-generated pull request.                                                                                      |
| `/diff`                                                                                                   | Review the changes made in the current directory.                                                                                                               |
| `/exit` , `/quit`                                                                                         | Exit the CLI.                                                                                                                                                   |
| `/experimental [on|off|show]`                                                                   | Toggle, set, or show experimental features.                                                                                                                     |
| `/feedback`                                                                                               | Provide feedback about the CLI.                                                                                                                                 |
| `/fleet [PROMPT]`                                                                                         | Enable parallel subagent execution of parts of a task. See [Running tasks in parallel with the /fleet command](/en/copilot/concepts/agents/copilot-cli/fleet) . |
| `/help`                                                                                                   | Show the help for interactive commands.                                                                                                                         |
| `/ide`                                                                                                    | Connect to an IDE workspace.                                                                                                                                    |
| `/init`                                                                                                   | Initialize Copilot custom instructions and agentic features for this repository.                                                                                |
| `/instructions`                                                                                           | View and toggle custom instruction files.                                                                                                                       |
| `/list-dirs`                                                                                              | Display all of the directories for which file access has been allowed.                                                                                          |
| `/login`                                                                                                  | Log in to Copilot.                                                                                                                                              |
| `/logout`                                                                                                 | Log out of Copilot.                                                                                                                                             |
| `/lsp [show|test|reload|help] [SERVER-NAME]`                                               | Manage the language server configuration.                                                                                                                       |
| `/mcp [show|add|edit|delete|disable|enable|auth|reload] [SERVER-NAME]` | Manage the MCP server configuration.                                                                                                                            |
| `/model` , `/models [MODEL]`                                                                              | Select the AI model you want to use.                                                                                                                            |
| `/on-air` , `/streamer-mode`                                                                              | Toggle streamer mode (hides preview model names).                                                                                                               |
| `/plan [PROMPT]`                                                                                          | Create an implementation plan before coding.                                                                                                                    |
| `/plugin [marketplace|install|uninstall|update|list] [ARGS...]`                       | Manage plugins and plugin marketplaces.                                                                                                                         |
| `/pr [view|create|fix|auto]`                                                               | Operate on pull requests for the current branch.                                                                                                                |
| `/rename [NAME]`                                                                                          | Rename the current session (auto-generates a name if omitted; alias for `/session rename` ).                                                                    |
| `/reset-allowed-tools`                                                                                    | Reset the list of allowed tools.                                                                                                                                |
| `/restart`                                                                                                | Restart the CLI, preserving the current session.                                                                                                                |
| `/resume [SESSION-ID]`                                                                                    | Switch to a different session by choosing from a list (optionally specify a session ID).                                                                        |
| `/review [PROMPT]`                                                                                        | Run the code review agent to analyze changes.                                                                                                                   |
| `/session [checkpoints [n]|files|plan|rename NAME]`                                        | Show session information and a workspace summary. Use the subcommands for details.                                                                              |
| `/share [file|gist] [session|research] [PATH]`                                                  | Share the session to a Markdown file or GitHub gist.                                                                                                            |
| `/skills [list|info|add|remove|reload] [ARGS...]`                                     | Manage skills for enhanced capabilities.                                                                                                                        |
| `/terminal-setup`                                                                                         | Configure the terminal for multiline input support ( `Shift` + `Enter` and `Ctrl` + `Enter` ).                                                                  |
| `/theme [show|set|list] [auto|THEME-ID]`                                                   | View or configure the terminal theme.                                                                                                                           |
| `/usage`                                                                                                  | Display session usage metrics and statistics.                                                                                                                   |
| `/undo` , `/rewind`                                                                                       | Rewind the last turn and revert file changes.                                                                                                                   |
| `/user [show|list|switch]`                                                                      | Manage the current GitHub user.                                                                                                                                 |

For a complete list of available slash commands enter `/help` in the CLI's interactive interface.

### Command-line options

| Option                                        | Purpose                                                                                                                                                                                                                                                                                                                    |
|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--acp`                                       | Start the Agent Client Protocol server.                                                                                                                                                                                                                                                                                    |
| `--add-dir=PATH`                              | Add a directory to the allowed list for file access (can be used multiple times).                                                                                                                                                                                                                                          |
| `--add-github-mcp-tool=TOOL`                  | Add a tool to enable for the GitHub MCP server, instead of the default CLI subset (can be used multiple times). Use `*` for all tools.                                                                                                                                                                                     |
| `--add-github-mcp-toolset=TOOLSET`            | Add a toolset to enable for the GitHub MCP server, instead of the default CLI subset (can be used multiple times). Use `all` for all toolsets.                                                                                                                                                                             |
| `--additional-mcp-config=JSON`                | Add an MCP server for this session only. The server configuration can be supplied as a JSON string or a file path (prefix with `@` ). Augments the configuration from `~/.copilot/mcp-config.json` . Overrides any installed MCP server configuration with the same name.                                                  |
| `--agent=AGENT`                               | Specify a custom agent to use.                                                                                                                                                                                                                                                                                             |
| `--allow-all`                                 | Enable all permissions (equivalent to `--allow-all-tools --allow-all-paths --allow-all-urls` ).                                                                                                                                                                                                                            |
| `--allow-all-paths`                           | Disable file path verification and allow access to any path.                                                                                                                                                                                                                                                               |
| `--allow-all-tools`                           | Allow all tools to run automatically without confirmation. Required when using the CLI programmatically (env: `COPILOT_ALLOW_ALL` ).                                                                                                                                                                                       |
| `--allow-all-urls`                            | Allow access to all URLs without confirmation.                                                                                                                                                                                                                                                                             |
| `--allow-tool=TOOL ...`                       | Tools the CLI has permission to use. Will not prompt for permission. For multiple tools, use a quoted, comma-separated list.                                                                                                                                                                                               |
| `--allow-url=URL ...`                         | Allow access to specific URLs or domains. For multiple URLs, use a quoted, comma-separated list.                                                                                                                                                                                                                           |
| `--autopilot`                                 | Enable autopilot continuation in prompt mode. See [Allowing GitHub Copilot CLI to work autonomously](/en/copilot/concepts/agents/copilot-cli/autopilot) .                                                                                                                                                                  |
| `--available-tools=TOOL ...`                  | Only these tools will be available to the model. For multiple tools, use a quoted, comma-separated list.                                                                                                                                                                                                                   |
| `--banner`                                    | Show the startup banner.                                                                                                                                                                                                                                                                                                   |
| `--bash-env`                                  | Enable `BASH_ENV` support for bash shells.                                                                                                                                                                                                                                                                                 |
| `--config-dir=PATH`                           | Set the configuration directory (default: `~/.copilot` ).                                                                                                                                                                                                                                                                  |
| `--continue`                                  | Resume the most recent session.                                                                                                                                                                                                                                                                                            |
| `--deny-tool=TOOL ...`                        | Tools the CLI does not have permission to use. Will not prompt for permission. For multiple tools, use a quoted, comma-separated list.                                                                                                                                                                                     |
| `--deny-url=URL ...`                          | Deny access to specific URLs or domains, takes precedence over `--allow-url` . For multiple URLs, use a quoted, comma-separated list.                                                                                                                                                                                      |
| `--disable-builtin-mcps`                      | Disable all built-in MCP servers (currently: `github-mcp-server` ).                                                                                                                                                                                                                                                        |
| `--disable-mcp-server=SERVER-NAME`            | Disable a specific MCP server (can be used multiple times).                                                                                                                                                                                                                                                                |
| `--disable-parallel-tools-execution`          | Disable parallel execution of tools (LLM can still make parallel tool calls, but they will be executed sequentially).                                                                                                                                                                                                      |
| `--disallow-temp-dir`                         | Prevent automatic access to the system temporary directory.                                                                                                                                                                                                                                                                |
| `--effort=LEVEL` , `--reasoning-effort=LEVEL` | Set the reasoning effort level ( `low` , `medium` , `high` ).                                                                                                                                                                                                                                                              |
| `--enable-all-github-mcp-tools`               | Enable all GitHub MCP server tools, instead of the default CLI subset. Overrides the `--add-github-mcp-toolset` and `--add-github-mcp-tool` options.                                                                                                                                                                       |
| `--enable-reasoning-summaries`                | Request reasoning summaries for OpenAI models that support it.                                                                                                                                                                                                                                                             |
| `--excluded-tools=TOOL ...`                   | These tools will not be available to the model. For multiple tools, use a quoted, comma-separated list.                                                                                                                                                                                                                    |
| `--experimental`                              | Enable experimental features (use `--no-experimental` to disable).                                                                                                                                                                                                                                                         |
| `-h` , `--help`                               | Display help.                                                                                                                                                                                                                                                                                                              |
| `-i PROMPT` , `--interactive=PROMPT`          | Start an interactive session and automatically execute this prompt.                                                                                                                                                                                                                                                        |
| `--log-dir=DIRECTORY`                         | Set the log file directory (default: `~/.copilot/logs/` ).                                                                                                                                                                                                                                                                 |
| `--log-level=LEVEL`                           | Set the log level (choices: `none` , `error` , `warning` , `info` , `debug` , `all` , `default` ).                                                                                                                                                                                                                         |
| `--max-autopilot-continues=COUNT`             | Maximum number of continuation messages in autopilot mode (default: unlimited). See [Allowing GitHub Copilot CLI to work autonomously](/en/copilot/concepts/agents/copilot-cli/autopilot) .                                                                                                                                |
| `--model=MODEL`                               | Set the AI model you want to use.                                                                                                                                                                                                                                                                                          |
| `--mouse[=VALUE]`                             | Enable mouse support in alt screen mode. VALUE can be `on` (default) or `off` . When enabled, the CLI captures mouse events in alt screen mode-scroll wheel, clicks, etc. When disabled, the terminal's native mouse behavior is preserved. Once set the setting is persisted by being written to your configuration file. |
| `--no-ask-user`                               | Disable the `ask_user` tool (the agent works autonomously without asking questions).                                                                                                                                                                                                                                       |
| `--no-auto-update`                            | Disable downloading CLI updates automatically.                                                                                                                                                                                                                                                                             |
| `--no-bash-env`                               | Disable `BASH_ENV` support for bash shells.                                                                                                                                                                                                                                                                                |
| `--no-color`                                  | Disable all color output.                                                                                                                                                                                                                                                                                                  |
| `--no-custom-instructions`                    | Disable loading of custom instructions from `AGENTS.md` and related files.                                                                                                                                                                                                                                                 |
| `--no-experimental`                           | Disable experimental features.                                                                                                                                                                                                                                                                                             |
| `--no-mouse`                                  | Disable mouse support.                                                                                                                                                                                                                                                                                                     |
| `--output-format=FORMAT`                      | FORMAT can be `text` (default) or `json` (outputs JSONL: one JSON object per line).                                                                                                                                                                                                                                        |
| `-p PROMPT` , `--prompt=PROMPT`               | Execute a prompt programmatically (exits after completion).                                                                                                                                                                                                                                                                |
| `--plain-diff`                                | Disable rich diff rendering (syntax highlighting via the diff tool specified by your git config).                                                                                                                                                                                                                          |
| `--plugin-dir=DIRECTORY`                      | Load a plugin from a local directory (can be used multiple times).                                                                                                                                                                                                                                                         |
| `--resume=SESSION-ID`                         | Resume a previous interactive session by choosing from a list (optionally specify a session ID).                                                                                                                                                                                                                           |
| `-s` , `--silent`                             | Output only the agent response (without usage statistics), useful for scripting with `-p` .                                                                                                                                                                                                                                |
| `--screen-reader`                             | Enable screen reader optimizations.                                                                                                                                                                                                                                                                                        |
| `--secret-env-vars=VAR ...`                   | Redact an environment variable from shell and MCP server environments (can be used multiple times). For multiple variables, use a quoted, comma-separated list. The values in the `GITHUB_TOKEN` and `COPILOT_GITHUB_TOKEN` environment variables are redacted from output by default.                                     |
| `--share=PATH`                                | Share a session to a Markdown file after completion of a programmatic session (default path: `./copilot-session-<ID>.md` ).                                                                                                                                                                                                |
| `--share-gist`                                | Share a session to a secret GitHub gist after completion of a programmatic session.                                                                                                                                                                                                                                        |
| `--stream=MODE`                               | Enable or disable streaming mode (mode choices: `on` or `off` ).                                                                                                                                                                                                                                                           |
| `-v` , `--version`                            | Show version information.                                                                                                                                                                                                                                                                                                  |
| `--yolo`                                      | Enable all permissions (equivalent to `--allow-all` ).                                                                                                                                                                                                                                                                     |

For a complete list of commands and options, run `copilot help` .

### Tool availability values

The `--available-tools` and `--excluded-tools` options support the following values for specifying tools:

#### Shell tools

| Tool name                         | Description                      |
|-----------------------------------|----------------------------------|
| `bash` / `powershell`             | Execute commands                 |
| `read_bash` / `read_powershell`   | Read output from a shell session |
| `write_bash` / `write_powershell` | Send input to a shell session    |
| `stop_bash` / `stop_powershell`   | Terminate a shell session        |
| `list_bash` / `list_powershell`   | List active shell sessions       |

#### File operation tools

| Tool name     | Description                                                       |
|---------------|-------------------------------------------------------------------|
| `view`        | Read files or directories                                         |
| `create`      | Create new files                                                  |
| `edit`        | Edit files via string replacement                                 |
| `apply_patch` | Apply patches (used by some models instead of `edit` / `create` ) |

#### Agent and task delegation tools

| Tool name     | Description                   |
|---------------|-------------------------------|
| `task`        | Run sub-agents                |
| `read_agent`  | Check background agent status |
| `list_agents` | List available agents         |

#### Other tools

| Tool name                         | Description                                |
|-----------------------------------|--------------------------------------------|
| `grep` (or `rg` )                 | Search for text in files                   |
| `glob`                            | Find files matching patterns               |
| `web_fetch`                       | Fetch and parse web content                |
| `skill`                           | Invoke custom skills                       |
| `ask_user`                        | Ask the user a question                    |
| `report_intent`                   | Report what the agent plans to do          |
| `show_file`                       | Display a file prominently                 |
| `fetch_copilot_cli_documentation` | Look up CLI documentation                  |
| `update_todo`                     | Update task checklist                      |
| `store_memory`                    | Persist facts across sessions              |
| `task_complete`                   | Signal task is done (autopilot only)       |
| `exit_plan_mode`                  | Exit plan mode                             |
| `sql`                             | Query session data (experimental)          |
| `lsp`                             | Language server refactoring (experimental) |

### Tool permission patterns

The `--allow-tool` and `--deny-tool` options accept permission patterns in the format `Kind(argument)` . The argument is optional-omitting it matches all tools of that kind.

| Kind        | Description                       | Example patterns                             |
|-------------|-----------------------------------|----------------------------------------------|
| `shell`     | Shell command execution           | `shell(git push)` , `shell(git:*)` , `shell` |
| `write`     | File creation or modification     | `write` , `write(src/*.ts)`                  |
| `read`      | File or directory reads           | `read` , `read(.env)`                        |
| SERVER-NAME | MCP server tool invocation        | `MyMCP(create_issue)` , `MyMCP`              |
| `url`       | URL access via web-fetch or shell | `url(github.com)` , `url(https://*.api.com)` |
| `memory`    | Storing facts to agent memory     | `memory`                                     |

For `shell` rules, the `:*` suffix matches the command stem followed by a space, preventing partial matches. For example, `shell(git:*)` matches `git push` and `git pull` but does not match `gitea` .

Deny rules always take precedence over allow rules, even when `--allow-all` is set.

```
### Allow all git commands except git push copilot --allow-tool='shell(git:*)' --deny-tool='shell(git push)' # Allow a specific MCP server tool copilot --allow-tool='MyMCP(create_issue)' # Allow all tools from a server copilot --allow-tool='MyMCP'
```

### Environment variables

| Variable                            | Description                                                                                                                                                                                                                                                                                                    |
|-------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `COPILOT_MODEL`                     | Set the AI model.                                                                                                                                                                                                                                                                                              |
| `COPILOT_ALLOW_ALL`                 | Set to `true` to allow all permissions automatically (equivalent to `--allow-all` ).                                                                                                                                                                                                                           |
| `COPILOT_AUTO_UPDATE`               | Set to `false` to disable automatic updates.                                                                                                                                                                                                                                                                   |
| `COPILOT_CUSTOM_INSTRUCTIONS_DIRS`  | Comma-separated list of additional directories for custom instructions.                                                                                                                                                                                                                                        |
| `COPILOT_SKILLS_DIRS`               | Comma-separated list of additional directories for skills.                                                                                                                                                                                                                                                     |
| `COPILOT_EDITOR`                    | Editor command for interactive editing (checked after `$VISUAL` and `$EDITOR` ). Defaults to `vi` if none are set.                                                                                                                                                                                             |
| `COPILOT_GITHUB_TOKEN`              | Authentication token. Takes precedence over `GH_TOKEN` and `GITHUB_TOKEN` .                                                                                                                                                                                                                                    |
| `COPILOT_HOME`                      | Override the configuration and state directory. Default: `$HOME/.copilot` .                                                                                                                                                                                                                                    |
| `COPILOT_CACHE_HOME`                | Override the cache directory (used for marketplace caches, auto-update packages, and other ephemeral data). See [GitHub Copilot CLI configuration directory](/en/copilot/reference/copilot-cli-reference/cli-config-dir-reference#changing-the-location-of-the-configuration-directory) for platform defaults. |
| `GH_TOKEN`                          | Authentication token. Takes precedence over `GITHUB_TOKEN` .                                                                                                                                                                                                                                                   |
| `GITHUB_TOKEN`                      | Authentication token.                                                                                                                                                                                                                                                                                          |
| `USE_BUILTIN_RIPGREP`               | Set to `false` to use the system ripgrep instead of the bundled version.                                                                                                                                                                                                                                       |
| `PLAIN_DIFF`                        | Set to `true` to disable rich diff rendering.                                                                                                                                                                                                                                                                  |
| `COLORFGBG`                         | Fallback for dark/light terminal background detection.                                                                                                                                                                                                                                                         |
| `COPILOT_CLI_ENABLED_FEATURE_FLAGS` | Comma-separated list of feature flags to enable (for example, `"SOME_FEATURE,SOME_OTHER_FEATURE"` ).                                                                                                                                                                                                           |

### Configuration file settings

Settings cascade from user to repository to local, with more specific scopes overriding more general ones. Command-line flags and environment variables always take the highest precedence.

| Scope      | Location                              | Purpose                                                                                                           |
|------------|---------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| User       | `~/.copilot/config.json`              | Global defaults for all repositories. Use the `COPILOT_HOME` environment variable to specify an alternative path. |
| Repository | `.github/copilot/settings.json`       | Shared repository configuration (committed to the repository).                                                    |
| Local      | `.github/copilot/settings.local.json` | Personal overrides (add this to `.gitignore` ).                                                                   |

#### User settings ( ~/.copilot/config.json )

| Key                                | Type                                                                                                            | Default                     | Description                                                                                                                        |
|------------------------------------|-----------------------------------------------------------------------------------------------------------------|-----------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `allowed_urls`                     | `string[]`                                                                                                      | `[]`                        | URLs or domains allowed without prompting.                                                                                         |
| `autoUpdate`                       | `boolean`                                                                                                       | `true`                      | Automatically download CLI updates.                                                                                                |
| `banner`                           | `"always"` | `"once"` | `"never"`                                                                     | `"once"`                    | Animated banner display frequency.                                                                                                 |
| `bashEnv`                          | `boolean`                                                                                                       | `false`                     | Enable `BASH_ENV` support for bash shells.                                                                                         |
| `beep`                             | `boolean`                                                                                                       | `true`                      | Play an audible beep when attention is required.                                                                                   |
| `compactPaste`                     | `boolean`                                                                                                       | `true`                      | Collapse large pastes into compact tokens.                                                                                         |
| `custom_agents.default_local_only` | `boolean`                                                                                                       | `false`                     | Only use local custom agents.                                                                                                      |
| `denied_urls`                      | `string[]`                                                                                                      | `[]`                        | URLs or domains blocked (takes precedence over `allowed_urls` ).                                                                   |
| `experimental`                     | `boolean`                                                                                                       | `false`                     | Enable experimental features.                                                                                                      |
| `includeCoAuthoredBy`              | `boolean`                                                                                                       | `true`                      | Add a `Co-authored-by` trailer to git commits made by the agent.                                                                   |
| `companyAnnouncements`             | `string[]`                                                                                                      | `[]`                        | Custom messages shown randomly on startup.                                                                                         |
| `logLevel`                         | `"none"` | `"error"` | `"warning"` | `"info"` | `"debug"` | `"all"` | `"default"` | `"default"`                 | Logging verbosity.                                                                                                                 |
| `model`                            | `string`                                                                                                        | varies                      | AI model to use (see the `/model` command).                                                                                        |
| `powershellFlags`                  | `string[]`                                                                                                      | `["-NoProfile", "-NoLogo"]` | Flags passed to PowerShell ( `pwsh` ) on startup. Windows only.                                                                    |
| `effortLevel`                      | `string`                                                                                                        | `"medium"`                  | Reasoning effort level for extended thinking (e.g., `"low"` , `"medium"` , `"high"` , `"xhigh"` ). Higher levels use more compute. |
| `renderMarkdown`                   | `boolean`                                                                                                       | `true`                      | Render Markdown in terminal output.                                                                                                |
| `screenReader`                     | `boolean`                                                                                                       | `false`                     | Enable screen reader optimizations.                                                                                                |
| `stream`                           | `boolean`                                                                                                       | `true`                      | Enable streaming responses.                                                                                                        |
| `storeTokenPlaintext`              | `boolean`                                                                                                       | `false`                     | Store authentication tokens in plain text in the configuration file when no system keychain is available.                          |
| `streamerMode`                     | `boolean`                                                                                                       | `false`                     | Hide preview model names and quota details (useful when demonstrating Copilot CLI).                                                |
| `theme`                            | `"auto"` | `"dark"` | `"light"`                                                                       | `"auto"`                    | Terminal color theme.                                                                                                              |
| `trusted_folders`                  | `string[]`                                                                                                      | `[]`                        | Folders with pre-granted file access.                                                                                              |
| `mouse`                            | `boolean`                                                                                                       | `true`                      | Enable mouse support in alt screen mode.                                                                                           |
| `respectGitignore`                 | `boolean`                                                                                                       | `true`                      | Exclude gitignored files from the `@` file picker.                                                                                 |
| `disableAllHooks`                  | `boolean`                                                                                                       | `false`                     | Disable all hooks.                                                                                                                 |
| `hooks`                            | `object`                                                                                                        | -                           | Inline user-level hook definitions.                                                                                                |
| `updateTerminalTitle`              | `boolean`                                                                                                       | `true`                      | Show the current intent in the terminal title.                                                                                     |

#### Repository settings ( .github/copilot/settings.json )

Repository settings apply to everyone who works in the repository. Only a subset of settings is supported at the repository level. Unsupported keys are ignored.

| Key                      | Type                      | Merge behavior                                | Description                                       |
|--------------------------|---------------------------|-----------------------------------------------|---------------------------------------------------|
| `companyAnnouncements`   | `string[]`                | Replaced-repository takes precedence          | Messages shown randomly on startup.               |
| `enabledPlugins`         | `Record<string, boolean>` | Merged-repository overrides user for same key | Declarative plugin auto-install.                  |
| `extraKnownMarketplaces` | `Record<string, {...}>`   | Merged-repository overrides user for same key | Plugin marketplaces available in this repository. |

#### Local settings ( .github/copilot/settings.local.json )

Create `.github/copilot/settings.local.json` in the repository, for personal overrides that should not be committed. Add this file to `.gitignore` .

The local configuration file uses the same schema as the repository configuration file ( `.github/copilot/settings.json` ) and takes precedence over it.

### Project initialization for Copilot

When you use the command `copilot init` , or the slash command `/init` within an interactive session, Copilot analyzes your codebase and writes or updates a `.github/copilot-instructions.md` file in the repository. This custom instructions file contains project-specific guidance that will improve future CLI sessions.

You will typically use `copilot init` , or `/init` , when you start a new project, or when you start using Copilot CLI in an existing repository.

The `copilot-instructions.md` file that's created or updated typically documents:

- Build, test, and lint commands.
- High-level architecture.
- Codebase-specific conventions.

If the file already exists, Copilot suggests improvements which you can choose to apply or reject.

The CLI looks for the `copilot-instructions.md` file on startup, and if it's missing, it displays the message:

💡 No copilot instructions found. Run /init to generate a copilot-instructions.md file for this project.

If you don't want to create this file, you can permanently hide this startup message by using the `/init suppress` slash command, which adds a `suppress_init_folders` setting for this repository to your Copilot configuration file.

For more information, see [Adding repository custom instructions for GitHub Copilot](/en/copilot/how-tos/configure-custom-instructions/add-repository-instructions) .

### Hooks reference

Hooks are external commands that execute at specific lifecycle points during a session, enabling custom automation, security controls, and integrations. Hook configuration files are loaded automatically from `.github/hooks/*.json` in your repository.

#### Hook configuration format

Hook configuration files use JSON format with version `1` .

##### Command hooks

Command hooks run shell scripts and are supported on all hook types.

```
{ "version" : 1 , "hooks" : { "preToolUse" : [ { "type" : "command" , "bash" : "your-bash-command" , "powershell" : "your-powershell-command" , "cwd" : "optional/working/directory" , "env" : { "VAR" : "value" } , "timeoutSec" : 30 } ] }
}
```

| Field        | Type        | Required                     | Description                                                                  |
|--------------|-------------|------------------------------|------------------------------------------------------------------------------|
| `type`       | `"command"` | Yes                          | Must be `"command"` .                                                        |
| `bash`       | string      | One of `bash` / `powershell` | Shell command for Unix.                                                      |
| `powershell` | string      | One of `bash` / `powershell` | Shell command for Windows.                                                   |
| `cwd`        | string      | No                           | Working directory for the command (relative to repository root or absolute). |
| `env`        | object      | No                           | Environment variables to set (supports variable expansion).                  |
| `timeoutSec` | number      | No                           | Timeout in seconds. Default: `30` .                                          |

##### Prompt hooks

Prompt hooks auto-submit text as if the user typed it. They are only supported on `sessionStart` and run before any initial prompt passed via `--prompt` . The text can be a natural language prompt or a slash command.

```
{ "version" : 1 , "hooks" : { "sessionStart" : [ { "type" : "prompt" , "prompt" : "Your prompt text or /slash-command" } ] }
}
```

| Field    | Type       | Required   | Description                                                          |
|----------|------------|------------|----------------------------------------------------------------------|
| `type`   | `"prompt"` | Yes        | Must be `"prompt"` .                                                 |
| `prompt` | string     | Yes        | Text to submit-can be a natural language message or a slash command. |

#### Hook events

| Event                 | Fires when                                                                                                                                                                                                                                        | Output processed                                                                               |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| `sessionStart`        | A new or resumed session begins.                                                                                                                                                                                                                  | No                                                                                             |
| `sessionEnd`          | The session terminates.                                                                                                                                                                                                                           | No                                                                                             |
| `userPromptSubmitted` | The user submits a prompt.                                                                                                                                                                                                                        | No                                                                                             |
| `preToolUse`          | Before each tool executes.                                                                                                                                                                                                                        | Yes - can allow, deny, or modify.                                                              |
| `postToolUse`         | After each tool completes successfully.                                                                                                                                                                                                           | Yes - can replace the successful result (SDK programmatic hooks only).                         |
| `postToolUseFailure`  | After a tool completes with a failure.                                                                                                                                                                                                            | Yes - can provide recovery guidance via `additionalContext` (exit code `2` for command hooks). |
| `agentStop`           | The main agent finishes a turn.                                                                                                                                                                                                                   | Yes - can block and force continuation.                                                        |
| `subagentStop`        | A subagent completes.                                                                                                                                                                                                                             | Yes - can block and force continuation.                                                        |
| `subagentStart`       | A subagent is spawned (before it runs). Returns `additionalContext` prepended to the subagent's prompt. Supports `matcher` to filter by agent name.                                                                                               | No - cannot block creation.                                                                    |
| `preCompact`          | Context compaction is about to begin (manual or automatic). Supports `matcher` to filter by trigger ( `"manual"` or `"auto"` ).                                                                                                                   | No - notification only.                                                                        |
| `permissionRequest`   | Before showing a permission dialog to the user, after rule-based checks find no matching allow or deny rule. Supports `matcher` regex on `toolName` .                                                                                             | Yes - can allow or deny programmatically.                                                      |
| `errorOccurred`       | An error occurs during execution.                                                                                                                                                                                                                 | No                                                                                             |
| `notification`        | Fires asynchronously when the CLI emits a system notification (shell completion, agent completion or idle, permission prompts, elicitation dialogs). Fire-and-forget: never blocks the session. Supports `matcher` regex on `notification_type` . | Optional - can inject `additionalContext` into the session.                                    |

#### Hook event input payloads

Each hook event delivers a JSON payload to the hook handler. Two payload formats are supported, selected by the event name used in the hook configuration:

- **camelCase format** - Configure the event name in camelCase (for example, `sessionStart` ). Fields use camelCase.
- **VS Code compatible format** - Configure the event name in PascalCase (for example, `SessionStart` ). Fields use snake\_case to match the VS Code Copilot extension format.

##### sessionStart / SessionStart

**camelCase input:**

```
{ sessionId : string ; timestamp : number ; // Unix timestamp in milliseconds cwd : string ; source : "startup" | "resume" | "new" ; initialPrompt ?: string ;
}
```

**VS Code compatible input:**

```
{ hook_event_name : "SessionStart" ; session_id : string ; timestamp : string ; // ISO 8601 timestamp cwd : string ; source : "startup" | "resume" | "new" ; initial_prompt ?: string ;
}
```

##### sessionEnd / SessionEnd

**camelCase input:**

```
{ sessionId : string ; timestamp : number ; cwd : string ; reason : "complete" | "error" | "abort" | "timeout" | "user_exit" ;
}
```

**VS Code compatible input:**

```
{ hook_event_name : "SessionEnd" ; session_id : string ; timestamp : string ; // ISO 8601 timestamp cwd : string ; reason : "complete" | "error" | "abort" | "timeout" | "user_exit" ;
}
```

##### userPromptSubmitted / UserPromptSubmit

**camelCase input:**

```
{ sessionId : string ; timestamp : number ; cwd : string ; prompt : string ;
}
```

**VS Code compatible input:**

```
{ hook_event_name : "UserPromptSubmit" ; session_id : string ; timestamp : string ; // ISO 8601 timestamp cwd : string ; prompt : string ;
}
```

##### preToolUse / PreToolUse

**camelCase input:**

```
{ sessionId : string ; timestamp : number ; cwd : string ; toolName : string ; toolArgs : unknown ;
}
```

**VS Code compatible input:**

When configured with the PascalCase event name `PreToolUse` , the payload uses snake\_case field names to match the VS Code Copilot extension format:

```
{ hook_event_name : "PreToolUse" ; session_id : string ; timestamp : string ; // ISO 8601 timestamp cwd : string ; tool_name : string ; tool_input : unknown ; // Tool arguments (parsed from JSON string when possible) }
```

##### postToolUse / PostToolUse

**camelCase input:**

```
{ sessionId : string ; timestamp : number ; cwd : string ; toolName : string ; toolArgs : unknown ; toolResult : { resultType : "success" ; textResultForLlm : string ;
    }
}
```

**VS Code compatible input:**

```
{ hook_event_name : "PostToolUse" ; session_id : string ; timestamp : string ; // ISO 8601 timestamp cwd : string ; tool_name : string ; tool_input : unknown ; tool_result : { result_type : "success" | "failure" | "denied" | "error" ; text_result_for_llm : string ;
    }
}
```

##### postToolUseFailure / PostToolUseFailure

**camelCase input:**

```
{ sessionId : string ; timestamp : number ; cwd : string ; toolName : string ; toolArgs : unknown ; error : string ;
}
```

**VS Code compatible input:**

```
{ hook_event_name : "PostToolUseFailure" ; session_id : string ; timestamp : string ; // ISO 8601 timestamp cwd : string ; tool_name : string ; tool_input : unknown ; error : string ;
}
```

##### agentStop / Stop

**camelCase input:**

```
{ sessionId : string ; timestamp : number ; cwd : string ; transcriptPath : string ; stopReason : "end_turn" ;
}
```

**VS Code compatible input:**

```
{ hook_event_name : "Stop" ; session_id : string ; timestamp : string ; // ISO 8601 timestamp cwd : string ; transcript_path : string ; stop_reason : "end_turn" ;
}
```

##### subagentStart

**Input:**

```
{ sessionId : string ; timestamp : number ; cwd : string ; transcriptPath : string ; agentName : string ; agentDisplayName ?: string ; agentDescription ?: string ;
}
```

##### subagentStop / SubagentStop

**camelCase input:**

```
{ sessionId : string ; timestamp : number ; cwd : string ; transcriptPath : string ; agentName : string ; agentDisplayName ?: string ; stopReason : "end_turn" ;
}
```

**VS Code compatible input:**

```
{ hook_event_name : "SubagentStop" ; session_id : string ; timestamp : string ; // ISO 8601 timestamp cwd : string ; transcript_path : string ; agent_name : string ; agent_display_name ?: string ; stop_reason : "end_turn" ;
}
```

##### errorOccurred / ErrorOccurred

**camelCase input:**

```
{ sessionId : string ; timestamp : number ; cwd : string ; error : { message : string ; name : string ; stack ?: string ;
    }; errorContext : "model_call" | "tool_execution" | "system" | "user_input" ; recoverable : boolean ;
}
```

**VS Code compatible input:**

```
{ hook_event_name : "ErrorOccurred" ; session_id : string ; timestamp : string ; // ISO 8601 timestamp cwd : string ; error : { message : string ; name : string ; stack ?: string ;
    }; error_context : "model_call" | "tool_execution" | "system" | "user_input" ; recoverable : boolean ;
}
```

##### preCompact / PreCompact

**camelCase input:**

```
{ sessionId : string ; timestamp : number ; cwd : string ; transcriptPath : string ; trigger : "manual" | "auto" ; customInstructions : string ;
}
```

**VS Code compatible input:**

```
{ hook_event_name : "PreCompact" ; session_id : string ; timestamp : string ; // ISO 8601 timestamp cwd : string ; transcript_path : string ; trigger : "manual" | "auto" ; custom_instructions : string ;
}
```

#### preToolUse decision control

The `preToolUse` hook can control tool execution by writing a JSON object to stdout.

| Field                      | Values                         | Description                                                     |
|----------------------------|--------------------------------|-----------------------------------------------------------------|
| `permissionDecision`       | `"allow"` , `"deny"` , `"ask"` | Whether the tool executes. Empty output uses default behavior.  |
| `permissionDecisionReason` | string                         | Reason shown to the agent. Required when decision is `"deny"` . |
| `modifiedArgs`             | object                         | Substitute tool arguments to use instead of the originals.      |

#### agentStop / subagentStop decision control

| Field      | Values                | Description                                                       |
|------------|-----------------------|-------------------------------------------------------------------|
| `decision` | `"block"` , `"allow"` | `"block"` forces another agent turn using `reason` as the prompt. |
| `reason`   | string                | Prompt for the next turn when `decision` is `"block"` .           |

#### permissionRequest decision control

The `permissionRequest` hook fires when a tool-level permission dialog is about to be shown. It fires after rule-based permission checks find no matching allow or deny rule. Use it to approve or deny tool calls programmatically-especially useful in pipe mode ( `-p` ) and CI environments where no interactive prompt is available.

**Matcher:** Optional regex tested against `toolName` . When set, the hook fires only for matching tool names.

Output JSON to stdout to control the permission decision:

| Field       | Values               | Description                                                    |
|-------------|----------------------|----------------------------------------------------------------|
| `behavior`  | `"allow"` , `"deny"` | Whether to approve or deny the tool call.                      |
| `message`   | string               | Reason fed back to the LLM when denying.                       |
| `interrupt` | boolean              | When `true` combined with `"deny"` , stops the agent entirely. |

Return empty output or `{}` to fall through to the default behavior (show the user dialog, or deny in pipe mode). Exit code `2` is treated as a deny; if the hook also outputs JSON on stdout, those fields are merged with the deny decision.

#### notification hook

The `notification` hook fires asynchronously when the CLI emits a system notification. These hooks are fire-and-forget: they never block the session, and any errors are logged and skipped.

**Input:**

```
{ sessionId : string ; timestamp : number ; cwd : string ; hook_event_name : "Notification" ; message : string ; // Human-readable notification text title ?: string ; // Short title (e.g., "Permission needed", "Shell completed") notification_type : string ; // One of the types listed below }
```

**Notification types:**

| Type                       | When it fires                                                                         |
|----------------------------|---------------------------------------------------------------------------------------|
| `shell_completed`          | A background (async) shell command finishes                                           |
| `shell_detached_completed` | A detached shell session completes                                                    |
| `agent_completed`          | A background sub-agent finishes (completed or failed)                                 |
| `agent_idle`               | A background agent finishes a turn and enters idle state (waiting for `write_agent` ) |
| `permission_prompt`        | The agent requests permission to execute a tool                                       |
| `elicitation_dialog`       | The agent requests additional information from the user                               |

**Output:**

```
{ additionalContext ?: string ; // Injected into the session as a user message }
```

If `additionalContext` is returned, the text is injected into the session as a prepended user message. This can trigger further agent processing if the session is idle. Return `{}` or empty output to take no action.

**Matcher:** Optional regex on `notification_type` . The pattern is anchored as `^(?:pattern)$` . Omit `matcher` to receive all notification types.

#### Tool names for hook matching

| Tool name    | Description                       |
|--------------|-----------------------------------|
| `bash`       | Execute shell commands (Unix).    |
| `powershell` | Execute shell commands (Windows). |
| `view`       | Read file contents.               |
| `edit`       | Modify file contents.             |
| `create`     | Create new files.                 |
| `glob`       | Find files by pattern.            |
| `grep`       | Search file contents.             |
| `web_fetch`  | Fetch web pages.                  |
| `task`       | Run subagent tasks.               |

If multiple hooks of the same type are configured, they execute in order. For `preToolUse` , if any hook returns `"deny"` , the tool is blocked. For `postToolUseFailure` command hooks, exiting with code `2` causes stderr to be returned as recovery guidance for the assistant. Hook failures (non-zero exit codes or timeouts) are logged and skipped-they never block agent execution.

### MCP server configuration

MCP servers provide additional tools to the CLI agent. Configure persistent servers in `~/.copilot/mcp-config.json` . Use `--additional-mcp-config` to add servers for a single session.

#### Transport types

| Type              | Description                                       | Required fields    |
|-------------------|---------------------------------------------------|--------------------|
| `local` / `stdio` | Local process communicating via stdin/stdout.     | `command` , `args` |
| `http`            | Remote server using streamable HTTP transport.    | `url`              |
| `sse`             | Remote server using Server-Sent Events transport. | `url`              |

#### Local server configuration fields

| Field     | Required   | Description                                                                          |
|-----------|------------|--------------------------------------------------------------------------------------|
| `command` | Yes        | Command to start the server.                                                         |
| `args`    | Yes        | Command arguments (array).                                                           |
| `tools`   | Yes        | Tools to enable: `["*"]` for all, or a list of specific tool names.                  |
| `env`     | No         | Environment variables. Supports `$VAR` , `${VAR}` , and `${VAR:-default}` expansion. |
| `cwd`     | No         | Working directory for the server.                                                    |
| `timeout` | No         | Tool call timeout in milliseconds.                                                   |
| `type`    | No         | `"local"` or `"stdio"` . Default: `"local"` .                                        |

#### Remote server configuration fields

| Field               | Required   | Description                                                                                                                                                                    |
|---------------------|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `type`              | Yes        | `"http"` or `"sse"` .                                                                                                                                                          |
| `url`               | Yes        | Server URL.                                                                                                                                                                    |
| `tools`             | Yes        | Tools to enable.                                                                                                                                                               |
| `headers`           | No         | HTTP headers. Supports variable expansion.                                                                                                                                     |
| `oauthClientId`     | No         | Static OAuth client ID (skips dynamic registration).                                                                                                                           |
| `oauthPublicClient` | No         | Whether the OAuth client is public. Default: `true` .                                                                                                                          |
| `oidc`              | No         | Enable OIDC token injection. When `true` , injects a `GITHUB_COPILOT_OIDC_MCP_TOKEN` environment variable (local servers) or a `Bearer Authorization` header (remote servers). |
| `timeout`           | No         | Tool call timeout in milliseconds.                                                                                                                                             |

#### OAuth re-authentication

Remote MCP servers that use OAuth may show a `needs-auth` status when a token expires or when a different account is required. Use `/mcp auth <server-name>` to trigger a fresh OAuth flow. This opens a browser authentication prompt, allowing you to sign in or switch accounts. After completing the flow, the server reconnects automatically.

#### Filter mapping

Control how MCP tool output is processed using the `filterMapping` field in a server's configuration.

| Mode                | Description                                   |
|---------------------|-----------------------------------------------|
| `none`              | No filtering.                                 |
| `markdown`          | Format output as Markdown.                    |
| `hidden_characters` | Remove hidden or control characters. Default. |

#### Built-in MCP servers

The CLI includes built-in MCP servers that are available without additional setup.

| Server              | Description                                                                              |
|---------------------|------------------------------------------------------------------------------------------|
| `github-mcp-server` | GitHub API integration: issues, pull requests, commits, code search, and GitHub Actions. |
| `playwright`        | Browser automation: navigate, click, type, screenshot, and form handling.                |
| `fetch`             | HTTP requests via the `fetch` tool.                                                      |
| `time`              | Time utilities: `get_current_time` and `convert_time` .                                  |

Use `--disable-builtin-mcps` to disable all built-in servers, or `--disable-mcp-server SERVER-NAME` to disable a specific one.

#### MCP server trust levels

MCP servers are loaded from multiple sources, each with a different trust level.

| Source                                              | Trust level   | Review required     |
|-----------------------------------------------------|---------------|---------------------|
| Built-in                                            | High          | No                  |
| Repository ( `.github/mcp.json` )                   | Medium        | Recommended         |
| Workspace ( `.mcp.json` , `.vscode/mcp.json` )      | Medium        | Recommended         |
| Dev Container ( `.devcontainer/devcontainer.json` ) | Medium        | Recommended         |
| User config ( `~/.copilot/mcp-config.json` )        | User-defined  | User responsibility |
| Remote servers                                      | Low           | Always              |

All MCP tool invocations require explicit permission. This applies even to read-only operations on external services.

### Skills reference

Skills are Markdown files that extend what the CLI can do. Each skill lives in its own directory containing a `SKILL.md` file. When invoked (via `/SKILL-NAME` or automatically by the agent), the skill's content is injected into the conversation.

#### Skill frontmatter fields

| Field                      | Type               | Required   | Description                                                                                                                   |
|----------------------------|--------------------|------------|-------------------------------------------------------------------------------------------------------------------------------|
| `name`                     | string             | Yes        | Unique identifier for the skill. Letters, numbers, and hyphens only. Max 64 characters.                                       |
| `description`              | string             | Yes        | What the skill does and when to use it. Max 1024 characters.                                                                  |
| `allowed-tools`            | string or string[] | No         | Comma-separated list or YAML array of tools that are automatically allowed when the skill is active. Use `"*"` for all tools. |
| `user-invocable`           | boolean            | No         | Whether users can invoke the skill with `/SKILL-NAME` . Default: `true` .                                                     |
| `disable-model-invocation` | boolean            | No         | Prevent the agent from automatically invoking this skill. Default: `false` .                                                  |

#### Skill locations

Skills are loaded from these locations in priority order (first found wins for duplicate names).

| Location                 | Scope     | Description                                                                   |
|--------------------------|-----------|-------------------------------------------------------------------------------|
| `.github/skills/`        | Project   | Project-specific skills.                                                      |
| `.agents/skills/`        | Project   | Alternative project location.                                                 |
| `.claude/skills/`        | Project   | Claude-compatible location.                                                   |
| Parent `.github/skills/` | Inherited | Monorepo parent directory support.                                            |
| `~/.copilot/skills/`     | Personal  | Personal skills for all projects.                                             |
| `~/.agents/skills/`      | Personal  | Agent skills shared across all projects.                                      |
| `~/.claude/skills/`      | Personal  | Claude-compatible personal location.                                          |
| Plugin directories       | Plugin    | Skills from installed plugins.                                                |
| `COPILOT_SKILLS_DIRS`    | Custom    | Additional directories (comma-separated).                                     |
| (bundled with CLI)       | Built-in  | Skills shipped with the CLI. Lowest priority-overridable by any other source. |

#### Commands (alternative skill format)

Commands are an alternative to skills stored as individual `.md` files in `.claude/commands/` . The command name is derived from the filename. Command files use a simplified format (no `name` field required) and support `description` , `allowed-tools` , and `disable-model-invocation` . Commands have lower priority than skills with the same name.

### Custom agents reference

Custom agents are specialized AI agents defined in Markdown files. The filename (minus extension) becomes the agent ID. Use `.agent.md` or `.md` as the file extension.

#### Built-in agents

| Agent             | Default model       | Description                                                                                                                                                                                          |
|-------------------|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `code-review`     | claude-sonnet-4.5   | High signal-to-noise code review. Analyzes diffs for bugs, security issues, and logic errors.                                                                                                        |
| `critic`          | complementary model | Rubber-duck adversarial feedback on proposals, designs, and implementations. Identifies weak points and suggests improvements. Available for Claude models. Experimental-requires `--experimental` . |
| `explore`         | claude-haiku-4.5    | Fast codebase exploration. Searches files, reads code, and answers questions. Returns focused answers under 300 words. Safe to run in parallel.                                                      |
| `general-purpose` | claude-sonnet-4.5   | Full-capability agent for complex multi-step tasks. Runs in a separate context window.                                                                                                               |
| `research`        | claude-sonnet-4.6   | Deep research agent. Generates a report based on information in your codebase, in relevant repositories, and on the web.                                                                             |
| `task`            | claude-haiku-4.5    | Command execution (tests, builds, lints). Returns brief summary on success, full output on failure.                                                                                                  |

#### Custom agent frontmatter fields

| Field         | Type     | Required   | Description                                                                    |
|---------------|----------|------------|--------------------------------------------------------------------------------|
| `description` | string   | Yes        | Description shown in the agent list and `task` tool.                           |
| `infer`       | boolean  | No         | Allow auto-delegation by the main agent. Default: `true` .                     |
| `mcp-servers` | object   | No         | MCP servers to connect. Uses the same schema as `~/.copilot/mcp-config.json` . |
| `model`       | string   | No         | AI model for this agent. When unset, inherits the outer agent's model.         |
| `name`        | string   | No         | Display name. Defaults to the filename.                                        |
| `tools`       | string[] | No         | Tools available to the agent. Default: `["*"]` (all tools).                    |

#### Custom agent locations

| Scope   | Location                                    |
|---------|---------------------------------------------|
| Project | `.github/agents/` or `.claude/agents/`      |
| User    | `~/.copilot/agents/` or `~/.claude/agents/` |
| Plugin  | `<plugin>/agents/`                          |

Project-level agents take precedence over user-level agents. Plugin agents have the lowest priority.

### Permission approval responses

When the CLI prompts for permission to execute an operation, you can respond with the following keys.

| Key   | Effect                                                  |
|-------|---------------------------------------------------------|
| `y`   | Allow this specific request once.                       |
| `n`   | Deny this specific request once.                        |
| `!`   | Allow all similar requests for the rest of the session. |
| `#`   | Deny all similar requests for the rest of the session.  |
| `?`   | Show detailed information about the request.            |

Session approvals reset when you run `/clear` or start a new session.

### OpenTelemetry monitoring

Copilot CLI can export traces and metrics via [OpenTelemetry](https://opentelemetry.io/) (OTel), giving you visibility into agent interactions, LLM calls, tool executions, and token usage. All signal names and attributes follow the [OTel GenAI Semantic Conventions](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/) .

OTel is off by default with zero overhead. It activates when any of the following conditions are met:

- COPILOT\_OTEL\_ENABLED=true
- `OTEL_EXPORTER_OTLP_ENDPOINT` is set
- `COPILOT_OTEL_FILE_EXPORTER_PATH` is set

#### OTel environment variables

| Variable                                             | Default          | Description                                                                                                  |
|------------------------------------------------------|------------------|--------------------------------------------------------------------------------------------------------------|
| `COPILOT_OTEL_ENABLED`                               | `false`          | Explicitly enable OTel. Not required if `OTEL_EXPORTER_OTLP_ENDPOINT` is set.                                |
| `OTEL_EXPORTER_OTLP_ENDPOINT`                        | -                | OTLP endpoint URL. Setting this automatically enables OTel.                                                  |
| `COPILOT_OTEL_EXPORTER_TYPE`                         | `otlp-http`      | Exporter type: `otlp-http` or `file` . Auto-selects `file` when `COPILOT_OTEL_FILE_EXPORTER_PATH` is set.    |
| `OTEL_SERVICE_NAME`                                  | `github-copilot` | Service name in resource attributes.                                                                         |
| `OTEL_RESOURCE_ATTRIBUTES`                           | -                | Extra resource attributes as comma-separated `key=value` pairs. Use percent-encoding for special characters. |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | `false`          | Capture full prompt and response content. See [Content capture](#content-capture) .                          |
| `OTEL_LOG_LEVEL`                                     | -                | OTel diagnostic log level: `NONE` , `ERROR` , `WARN` , `INFO` , `DEBUG` , `VERBOSE` , `ALL` .                |
| `COPILOT_OTEL_FILE_EXPORTER_PATH`                    | -                | Write all signals to this file as JSON-lines. Setting this automatically enables OTel.                       |
| `COPILOT_OTEL_SOURCE_NAME`                           | `github.copilot` | Instrumentation scope name for tracer and meter.                                                             |
| `OTEL_EXPORTER_OTLP_HEADERS`                         | -                | Auth headers for the OTLP exporter (for example, `Authorization=Bearer token` ).                             |

#### Traces

The runtime emits a hierarchical span tree for each agent interaction. Each tree contains an `invoke_agent` root span, with `chat` and `execute_tool` child spans.

##### invoke\_agent span attributes

Wraps the entire agent invocation: all LLM calls and tool executions for one user message.

- **Top-level sessions** use span kind `CLIENT` (remote service invocation) with `server.address` and `server.port` .
- **Subagent invocations** (for example, explore, task) use span kind `INTERNAL` (in-process) without server attributes.

| Attribute                                  | Description                                          | Span kind     |
|--------------------------------------------|------------------------------------------------------|---------------|
| `gen_ai.operation.name`                    | `invoke_agent`                                       | Both          |
| `gen_ai.provider.name`                     | Provider (for example, `github` , `anthropic` )      | Both          |
| `gen_ai.agent.id`                          | Session identifier                                   | Both          |
| `gen_ai.agent.name`                        | Agent name (when available)                          | Both          |
| `gen_ai.agent.description`                 | Agent description (when available)                   | Both          |
| `gen_ai.agent.version`                     | Runtime version                                      | Both          |
| `gen_ai.conversation.id`                   | Session identifier                                   | Both          |
| `gen_ai.request.model`                     | Requested model                                      | Both          |
| `gen_ai.response.finish_reasons`           | `["stop"]` or `["error"]`                            | Both          |
| `gen_ai.usage.input_tokens`                | Total input tokens (all turns)                       | Both          |
| `gen_ai.usage.output_tokens`               | Total output tokens (all turns)                      | Both          |
| `gen_ai.usage.cache_read.input_tokens`     | Cached input tokens read                             | Both          |
| `gen_ai.usage.cache_creation.input_tokens` | Cached input tokens created                          | Both          |
| `github.copilot.turn_count`                | Number of LLM round-trips                            | Both          |
| `github.copilot.cost`                      | Monetary cost                                        | Both          |
| `github.copilot.aiu`                       | AI units consumed                                    | Both          |
| `server.address`                           | Server hostname                                      | `CLIENT` only |
| `server.port`                              | Server port                                          | `CLIENT` only |
| `error.type`                               | Error class name (on error)                          | Both          |
| `gen_ai.input.messages`                    | Full input messages as JSON (content capture only)   | Both          |
| `gen_ai.output.messages`                   | Full output messages as JSON (content capture only)  | Both          |
| `gen_ai.system_instructions`               | System prompt content as JSON (content capture only) | Both          |
| `gen_ai.tool.definitions`                  | Tool schemas as JSON (content capture only)          | Both          |

##### chat span attributes

One span per LLM request. Span kind: `CLIENT` .

| Attribute                                  | Description                                                |
|--------------------------------------------|------------------------------------------------------------|
| `gen_ai.operation.name`                    | `chat`                                                     |
| `gen_ai.provider.name`                     | Provider name                                              |
| `gen_ai.request.model`                     | Requested model                                            |
| `gen_ai.conversation.id`                   | Session identifier                                         |
| `gen_ai.response.id`                       | Response ID                                                |
| `gen_ai.response.model`                    | Resolved model                                             |
| `gen_ai.response.finish_reasons`           | Stop reasons                                               |
| `gen_ai.usage.input_tokens`                | Input tokens this turn                                     |
| `gen_ai.usage.output_tokens`               | Output tokens this turn                                    |
| `gen_ai.usage.cache_read.input_tokens`     | Cached tokens read                                         |
| `gen_ai.usage.cache_creation.input_tokens` | Cached tokens created                                      |
| `github.copilot.cost`                      | Turn cost                                                  |
| `github.copilot.aiu`                       | AI units consumed this turn                                |
| `github.copilot.server_duration`           | Server-side duration                                       |
| `github.copilot.initiator`                 | Request initiator                                          |
| `github.copilot.turn_id`                   | Turn identifier                                            |
| `github.copilot.interaction_id`            | Interaction identifier                                     |
| `github.copilot.time_to_first_chunk`       | Time to first streaming chunk, in seconds (streaming only) |
| `server.address`                           | Server hostname                                            |
| `server.port`                              | Server port                                                |
| `error.type`                               | Error class name (on error)                                |
| `gen_ai.input.messages`                    | Full prompt messages as JSON (content capture only)        |
| `gen_ai.output.messages`                   | Full response messages as JSON (content capture only)      |
| `gen_ai.system_instructions`               | System prompt content as JSON (content capture only)       |

##### execute\_tool span attributes

One span per tool call. Span kind: `INTERNAL` .

| Attribute                    | Description                                         |
|------------------------------|-----------------------------------------------------|
| `gen_ai.operation.name`      | `execute_tool`                                      |
| `gen_ai.provider.name`       | Provider name (when available)                      |
| `gen_ai.tool.name`           | Tool name (for example, `readFile` )                |
| `gen_ai.tool.type`           | `function`                                          |
| `gen_ai.tool.call.id`        | Tool call identifier                                |
| `gen_ai.tool.description`    | Tool description                                    |
| `error.type`                 | Error class name (on error)                         |
| `gen_ai.tool.call.arguments` | Tool input arguments as JSON (content capture only) |
| `gen_ai.tool.call.result`    | Tool output as JSON (content capture only)          |

#### Metrics

##### GenAI convention metrics

| Metric                                          | Type      | Unit   | Description                                 |
|-------------------------------------------------|-----------|--------|---------------------------------------------|
| `gen_ai.client.operation.duration`              | Histogram | s      | LLM API call and agent invocation duration  |
| `gen_ai.client.token.usage`                     | Histogram | tokens | Token counts by type ( `input` / `output` ) |
| `gen_ai.client.operation.time_to_first_chunk`   | Histogram | s      | Time to receive first streaming chunk       |
| `gen_ai.client.operation.time_per_output_chunk` | Histogram | s      | Inter-chunk latency after first chunk       |

##### Vendor-specific metrics

| Metric                              | Type      | Unit   | Description                                          |
|-------------------------------------|-----------|--------|------------------------------------------------------|
| `github.copilot.tool.call.count`    | Counter   | calls  | Tool invocations by `gen_ai.tool.name` and `success` |
| `github.copilot.tool.call.duration` | Histogram | s      | Tool execution latency by `gen_ai.tool.name`         |
| `github.copilot.agent.turn.count`   | Histogram | turns  | LLM round-trips per agent invocation                 |

#### Span events

Lifecycle events recorded on the active `chat` or `invoke_agent` span.

| Event                                        | Description                          | Key attributes                                                                                                                                                                                                                                                   |
|----------------------------------------------|--------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `github.copilot.hook.start`                  | A hook began executing               | `github.copilot.hook.type` , `github.copilot.hook.invocation_id`                                                                                                                                                                                                 |
| `github.copilot.hook.end`                    | A hook completed successfully        | `github.copilot.hook.type` , `github.copilot.hook.invocation_id`                                                                                                                                                                                                 |
| `github.copilot.hook.error`                  | A hook failed                        | `github.copilot.hook.type` , `github.copilot.hook.invocation_id` , `github.copilot.hook.error_message`                                                                                                                                                           |
| `github.copilot.session.truncation`          | Conversation history was truncated   | `github.copilot.token_limit` , `github.copilot.pre_tokens` , `github.copilot.post_tokens` , `github.copilot.pre_messages` , `github.copilot.post_messages` , `github.copilot.tokens_removed` , `github.copilot.messages_removed` , `github.copilot.performed_by` |
| `github.copilot.session.compaction_start`    | History compaction began             | None                                                                                                                                                                                                                                                             |
| `github.copilot.session.compaction_complete` | History compaction completed         | `github.copilot.success` , `github.copilot.pre_tokens` , `github.copilot.post_tokens` , `github.copilot.tokens_removed` , `github.copilot.messages_removed` , `github.copilot.message` (content capture only)                                                    |
| `github.copilot.skill.invoked`               | A skill was invoked                  | `github.copilot.skill.name` , `github.copilot.skill.path` , `github.copilot.skill.plugin_name` , `github.copilot.skill.plugin_version`                                                                                                                           |
| `github.copilot.session.shutdown`            | Session is shutting down             | `github.copilot.shutdown_type` , `github.copilot.total_premium_requests` , `github.copilot.lines_added` , `github.copilot.lines_removed` , `github.copilot.files_modified_count`                                                                                 |
| `github.copilot.session.abort`               | User cancelled the current operation | `github.copilot.abort_reason`                                                                                                                                                                                                                                    |
| `exception`                                  | Session error                        | `github.copilot.error_type` , `github.copilot.error_status_code` , `github.copilot.error_provider_call_id`                                                                                                                                                       |

#### Resource attributes

All signals carry these resource attributes.

| Attribute         | Value                                                    |
|-------------------|----------------------------------------------------------|
| `service.name`    | `github-copilot` (configurable via `OTEL_SERVICE_NAME` ) |
| `service.version` | Runtime version                                          |

#### Content capture

By default, no prompt content, responses, or tool arguments are captured-only metadata like model names, token counts, and durations. To capture full content, set `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` .

Warning

Content capture may include sensitive information such as code, file contents, and user prompts. Only enable this in trusted environments.

When content capture is enabled, the following attributes are populated.

| Attribute                    | Content                       |
|------------------------------|-------------------------------|
| `gen_ai.input.messages`      | Full prompt messages (JSON)   |
| `gen_ai.output.messages`     | Full response messages (JSON) |
| `gen_ai.system_instructions` | System prompt content (JSON)  |
| `gen_ai.tool.definitions`    | Tool schemas (JSON)           |
| `gen_ai.tool.call.arguments` | Tool input arguments          |
| `gen_ai.tool.call.result`    | Tool output                   |

### Feature flag reference

Feature flags enable functionality that is not yet generally available. Enable flags via the `COPILOT_CLI_ENABLED_FEATURE_FLAGS` environment variable (comma-separated list) or by using the `/experimental` slash command.

| Flag                                  | Tier                  | Description                                                                                     |
|---------------------------------------|-----------------------|-------------------------------------------------------------------------------------------------|
| `RUBBER_DUCK_AGENT`                   | experimental          | Rubber-duck subagent for adversarial feedback on code and designs (available for Claude models) |
| `BACKGROUND_SESSIONS`                 | experimental          | Multiple concurrent sessions with background management                                         |
| `MULTI_TURN_AGENTS`                   | experimental          | Multi-turn subagent message passing via `write_agent`                                           |
| `EXTENSIONS`                          | experimental          | Programmatic extensions with custom tools and hooks                                             |
| `QUEUED_COMMANDS`                     | staff-or-experimental | Queue commands with `Ctrl` + `Enter` while the agent runs                                       |
| `PERSISTED_PERMISSIONS`               | staff-or-experimental | Persist tool permissions across sessions per location                                           |
| `SESSION_STORE`                       | staff-or-experimental | SQLite-based session store for cross-session history                                            |
| `COMPUTER_USE`                        | staff                 | Built-in computer use MCP server (screen capture and mouse/keyboard control)                    |
| `copilot-feature-agentic-memory`      | on                    | Persistent memory tools across sessions                                                         |
| `COPILOT_SWE_AGENT_BACKGROUND_AGENTS` | on                    | Background agent task execution                                                                 |

### Further reading

- [GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli)
- [GitHub Copilot CLI plugin reference](/en/copilot/reference/copilot-cli-reference/cli-plugin-reference)
- [GitHub Copilot CLI programmatic reference](/en/copilot/reference/copilot-cli-reference/cli-programmatic-reference)
- [GitHub Copilot CLI configuration directory](/en/copilot/reference/copilot-cli-reference/cli-config-dir-reference)


### CLI commands

You can use the following commands in the terminal to manage plugins for Copilot CLI.

| Command                                        | Description                                                                                                                                                                                                    |
|------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `copilot plugin install SPECIFICATION`         | Install a plugin. See [Plugin specification for](#plugin-specification-for-install-command) [`install`](#plugin-specification-for-install-command) [command](#plugin-specification-for-install-command) below. |
| `copilot plugin uninstall NAME`                | Remove a plugin                                                                                                                                                                                                |
| `copilot plugin list`                          | List installed plugins                                                                                                                                                                                         |
| `copilot plugin update NAME`                   | Update a plugin                                                                                                                                                                                                |
| `copilot plugin marketplace add SPECIFICATION` | Register a marketplace                                                                                                                                                                                         |
| `copilot plugin marketplace list`              | List registered marketplaces                                                                                                                                                                                   |
| `copilot plugin marketplace browse NAME`       | Browse marketplace plugins                                                                                                                                                                                     |
| `copilot plugin marketplace remove NAME`       | Unregister a marketplace                                                                                                                                                                                       |

#### Plugin specification for install command

| Format         | Example                      | Description                          |
|----------------|------------------------------|--------------------------------------|
| Marketplace    | `plugin@marketplace`         | Plugin from a registered marketplace |
| GitHub         | `OWNER/REPO`                 | Root of a GitHub repository          |
| GitHub  subdir | `OWNER/REPO:PATH/TO/PLUGIN`  | Subdirectory in a repository         |
| Git URL        | `https://github.com/o/r.git` | Any Git URL                          |
| Local path     | `./my-plugin` or `/abs/path` | Local directory                      |

### plugin.json

All plugins consist of a plugin directory containing, at minimum, a manifest file named `plugin.json` located at the root of the plugin directory. See [Creating a plugin for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/plugins-creating) .

#### Required field

| Field   | Type   | Description                                                            |
|---------|--------|------------------------------------------------------------------------|
| `name`  | string | Kebab-case plugin name (letters, numbers, hyphens only). Max 64 chars. |

#### Optional metadata fields

| Field         | Type     | Description                                              |
|---------------|----------|----------------------------------------------------------|
| `description` | string   | Brief description. Max 1024 chars.                       |
| `version`     | string   | Semantic version (e.g., `1.0.0` ).                       |
| `author`      | object   | `name` (required), `email` (optional), `url` (optional). |
| `homepage`    | string   | Plugin homepage URL.                                     |
| `repository`  | string   | Source repository URL.                                   |
| `license`     | string   | License identifier (e.g., `MIT` ).                       |
| `keywords`    | string[] | Search keywords.                                         |
| `category`    | string   | Plugin category.                                         |
| `tags`        | string[] | Additional tags.                                         |

#### Component path fields

These tell the CLI where to find your plugin's components. All are optional. The CLI uses default conventions if omitted.

| Field        | Type                   | Default   | Description                                                                    |
|--------------|------------------------|-----------|--------------------------------------------------------------------------------|
| `agents`     | string | string[] | `agents/` | Path(s) to agent directories ( `.agent.md` files).                             |
| `skills`     | string | string[] | `skills/` | Path(s) to skill directories ( `SKILL.md` files).                              |
| `commands`   | string | string[] | -         | Path(s) to command directories.                                                |
| `hooks`      | string | object   | -         | Path to a hooks config file, or an inline hooks object.                        |
| `mcpServers` | string | object   | -         | Path to an MCP config file (e.g., `.mcp.json` ), or inline server definitions. |
| `lspServers` | string | object   | -         | Path to an LSP config file, or inline server definitions.                      |

#### Example plugin.json file

JSON

```
{ "name" : "my-dev-tools" , "description" : "React development utilities" , "version" : "1.2.0" , "author" : { "name" : "Jane Doe" , "email" : "jane@example.com" } , "license" : "MIT" , "keywords" : [ "react" , "frontend" ] , "agents" : "agents/" , "skills" : [ "skills/" , "extra-skills/" ] , "hooks" : "hooks.json" , "mcpServers" : ".mcp.json"
}
```

### marketplace.json

You can create a plugin marketplace-which people can use to discover and install your plugins-by creating a `marketplace.json` file and saving it to the `.github/plugin/` directory of the repository. You can also store the `marketplace.json` file in your local file system. For example, saving the file as `/PATH/TO/my-marketplace/.github/plugin/marketplace.json` allows you to add it to the CLI using the following command:

```
copilot plugin marketplace add /PATH/TO/my-marketplace
```

Note

Copilot CLI also looks for the `marketplace.json` file in the `.claude-plugin/` directory.

For more information, see [Creating a plugin marketplace for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/plugins-marketplace) .

#### Example marketplace.json file

JSON

```
{ "name" : "my-marketplace" , "owner" : { "name" : "Your Organization" , "email" : "plugins@example.com" } , "metadata" : { "description" : "Curated plugins for our team" , "version" : "1.0.0" } , "plugins" : [ { "name" : "frontend-design" , "description" : "Create a professional-looking GUI ..." , "version" : "2.1.0" , "source" : "./plugins/frontend-design" } , { "name" : "security-checks" , "description" : "Check for potential security vulnerabilities ..." , "version" : "1.3.0" , "source" : "./plugins/security-checks" } ]
}
```

Note

The value of the `source` field for each plugin is the path to the plugin's directory, relative to the root of the repository. It is not necessary to use `./` at the start of the path. For example, `"./plugins/plugin-name"` and `"plugins/plugin-name"` resolve to the same directory.

#### marketplace.json fields

##### Top-level fields

| Field      | Type   | Required   | Description                                   |
|------------|--------|------------|-----------------------------------------------|
| `name`     | string | Yes        | Kebab-case marketplace name. Max 64 chars.    |
| `owner`    | object | Yes        | `{ name, email? }` - marketplace owner info.  |
| `plugins`  | array  | Yes        | List of plugin entries (see the table below). |
| `metadata` | object | No         | `{ description?, version?, pluginRoot? }`     |

##### Plugin entry fields (objects within the plugins array)

| Field         | Type                   | Required   | Description                                                                                                                                                                                                     |
|---------------|------------------------|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `name`        | string                 | Yes        | Kebab-case plugin name. Max 64 chars.                                                                                                                                                                           |
| `source`      | string | object   | Yes        | Where to fetch the plugin (relative path, GitHub, or URL).                                                                                                                                                      |
| `description` | string                 | No         | Plugin description. Max 1024 chars.                                                                                                                                                                             |
| `version`     | string                 | No         | Plugin version.                                                                                                                                                                                                 |
| `author`      | object                 | No         | `{ name, email?, url? }`                                                                                                                                                                                        |
| `homepage`    | string                 | No         | Plugin homepage URL.                                                                                                                                                                                            |
| `repository`  | string                 | No         | Source repository URL.                                                                                                                                                                                          |
| `license`     | string                 | No         | License identifier.                                                                                                                                                                                             |
| `keywords`    | string[]               | No         | Search keywords.                                                                                                                                                                                                |
| `category`    | string                 | No         | Plugin category.                                                                                                                                                                                                |
| `tags`        | string[]               | No         | Additional tags.                                                                                                                                                                                                |
| `commands`    | string | string[] | No         | Path(s) to command directories.                                                                                                                                                                                 |
| `agents`      | string | string[] | No         | Path(s) to agent directories.                                                                                                                                                                                   |
| `skills`      | string | string[] | No         | Path(s) to skill directories.                                                                                                                                                                                   |
| `hooks`       | string | object   | No         | Path to hooks config or inline hooks object.                                                                                                                                                                    |
| `mcpServers`  | string | object   | No         | Path to MCP config or inline server definitions.                                                                                                                                                                |
| `lspServers`  | string | object   | No         | Path to LSP config or inline server definitions.                                                                                                                                                                |
| `strict`      | boolean                | No         | When `true` (the default), plugins must conform to the full schema and validation rules. When `false` , relaxed validation is used, allowing more flexibility-especially for direct installs or legacy plugins. |

### File locations

| Item                 | Path                                                                                                                                                            |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Installed plugins    | `~/.copilot/installed-plugins/MARKETPLACE/PLUGIN-NAME` (installed via a marketplace) and `~/.copilot/installed-plugins/_direct/SOURCE-ID/` (installed directly) |
| Marketplace cache    | Platform cache directory: `~/.cache/copilot/marketplaces/` (Linux), `~/Library/Caches/copilot/marketplaces/` (macOS). Overridable with `COPILOT_CACHE_HOME` .   |
| Plugin manifest      | `.plugin/plugin.json` , `plugin.json` , `.github/plugin/plugin.json` , or `.claude-plugin/plugin.json` (checked in this order)                                  |
| Marketplace manifest | `marketplace.json` , `.plugin/marketplace.json` , `.github/plugin/marketplace.json` , or `.claude-plugin/marketplace.json` (checked in this order)              |
| Agents               | `agents/` (default, overridable in manifest)                                                                                                                    |
| Skills               | `skills/` (default, overridable in manifest)                                                                                                                    |
| Hooks config         | `hooks.json` or `hooks/hooks.json`                                                                                                                              |
| MCP config           | `.mcp.json` , `.vscode/mcp.json` , `.devcontainer/devcontainer.json` , `.github/mcp.json`                                                                       |
| LSP config           | `lsp.json` or `.github/lsp.json`                                                                                                                                |

### Loading order and precedence

If you install multiple plugins it's possible that some custom agents, skills, MCP servers, or tools supplied via MCP servers have duplicate names. In this situation, the CLI determines which component to use based on a precedence order.

- **Agents and skills** use first-found-wins precedence. If you have a project-level custom agent or skill with the same name or ID as one in a plugin you install, the agent or skill in the plugin is silently ignored. The plugin cannot override project-level or personal configurations. Custom agents are deduplicated using their ID, which is derived from its file name (for example, if the file is named `reviewer.agent.md` , the agent ID is `reviewer` ). Skills are deduplicated by their name field inside the `SKILL.md` file.
- **MCP servers** use last-wins precedence. If you install a plugin that defines an MCP server with the same server name as an MCP server you have already installed, the plugin's definition takes precedence. You can use the `--additional-mcp-config` command-line option to override an MCP server configuration with the same name, installed using a plugin.
- **Built-in tools and agents** are always present and cannot be overridden by user-defined components.

The following diagram illustrates the loading order and precedence rules.

```
┌──────────────────────────────────────────────────────────────────┐
│  BUILT-IN - HARDCODED, ALWAYS PRESENT                            │
│  • tools: bash, view, apply_patch, glob, rg, task, ...           │
│  • agents: explore, task, code-review, general-purpose, research │
└────────────────────────┬─────────────────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────────────────────┐
  │  CUSTOM AGENTS - FIRST LOADED IS USED (dedup by ID)                 │
  │  1. ~/.copilot/agents/           (user, .github convention)         │
  │  2. <project>/.github/agents/    (project)                          │
  │  3. <parents>/.github/agents/    (inherited, monorepo)              │
  │  4. ~/.claude/agents/            (user, .claude convention)         │
  │  5. <project>/.claude/agents/    (project)                          │
  │  6. <parents>/.claude/agents/    (inherited, monorepo)              │
  │  7. PLUGIN: agents/ dirs         (plugin, by install order)         │
  │  8. Remote org/enterprise agents (remote, via API)                  │
  └──────────────────────┬──────────────────────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────────────────────┐
  │  AGENT SKILLS - FIRST LOADED IS USED (dedup by name)                │
  │  1. <project>/.github/skills/        (project)                      │
  │  2. <project>/.agents/skills/        (project)                      │
  │  3. <project>/.claude/skills/        (project)                      │
  │  4. <parents>/.github/skills/ etc.   (inherited)                    │
  │  5. ~/.copilot/skills/               (personal-copilot)             │
  │  6. ~/.agents/skills/                (personal-agents)              │
  │  7. ~/.claude/skills/                (personal-claude)              │
  │  8. PLUGIN: skills/ dirs             (plugin)                       │
  │  9. COPILOT_SKILLS_DIRS env + config (custom)                       │
  │  --- then commands (.claude/commands/), skills override commands ---│
  └──────────────────────┬──────────────────────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────────────────────┐
  │  MCP SERVERS - LAST LOADED IS USED (dedup by server name)           │
  │  1. ~/.copilot/mcp-config.json       (lowest priority)              │
  │  2. .vscode/mcp.json                 (workspace)                    │
  │  3. PLUGIN: MCP configs              (plugins)                      │
  │  4. --additional-mcp-config flag     (highest priority)             │
  └─────────────────────────────────────────────────────────────────────┘
```

### Further reading

- [GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli)
- [GitHub Copilot CLI command reference](/en/copilot/reference/copilot-cli-reference/cli-command-reference)
- [GitHub Copilot CLI programmatic reference](/en/copilot/reference/copilot-cli-reference/cli-programmatic-reference)


### Command line options

There are a number of command-line options that are particularly useful when running Copilot CLI programmatically.

| Option                       | Description                                                                                                                                                                                                                                                                                     |
|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-p PROMPT`                  | Execute a prompt in non-interactive mode. The CLI runs the prompt and exits when done.                                                                                                                                                                                                          |
| `-s`                         | Suppress stats and decoration, outputting only the agent's response. Ideal for piping output in scripts.                                                                                                                                                                                        |
| `--add-dir=DIRECTORY`        | Add a directory to the allowed-paths list. This can be used multiple times to add multiple directories. Useful when the agent needs to read/write outside the current working directory.                                                                                                        |
| `--agent=AGENT`              | Specify a custom agent to use.                                                                                                                                                                                                                                                                  |
| `--allow-all` (or `--yolo` ) | Allow the CLI all permissions. Equivalent to `--allow-all-tools --allow-all-paths --allow-all-urls` .                                                                                                                                                                                           |
| `--allow-all-paths`          | Disable file-path verification entirely. Simpler alternative to `--add-dir` when path restrictions aren't needed.                                                                                                                                                                               |
| `--allow-all-tools`          | Allow every tool to run without explicit permission for each tool.                                                                                                                                                                                                                              |
| `--allow-all-urls`           | Allow access to all URLs without explicit permission for each URL.                                                                                                                                                                                                                              |
| `--allow-tool=TOOL ...`      | Selectively grant permission for a specific tool. For multiple tools, use a quoted, comma-separated list.                                                                                                                                                                                       |
| `--allow-url=URL ...`        | Allow the agent to fetch a specific URL or domain. Useful when a workflow needs web access to known endpoints. For multiple URLs, use a quoted, comma-separated list.                                                                                                                           |
| `--deny-tool=TOOL ...`       | Deny a specific tool. Useful for restricting what the agent can do in a locked-down workflow. For multiple tools, use a quoted, comma-separated list.                                                                                                                                           |
| `--model=MODEL`              | Choose the AI model (for example, `gpt-5.2` or `claude-sonnet-4.6` ). Useful for pinning a model in reproducible workflows. See [Choosing a model](#choosing-a-model) below.                                                                                                                    |
| `--no-ask-user`              | Prevent the agent from pausing to seek additional user input.                                                                                                                                                                                                                                   |
| `--secret-env-vars=VAR ...`  | An environment variable whose value you want redacted in output. For multiple variables, use a quoted, comma-separated list. Essential for preventing secrets being exposed in logs. The values in the `GITHUB_TOKEN` and `COPILOT_GITHUB_TOKEN` environment variables are redacted by default. |
| `--share=PATH`               | Export the session transcript to a markdown file after non-interactive completion (defaults to `./copilot-session-<ID>.md` ). Useful for auditing or archiving what the agent did. Note that session transcripts may contain sensitive information.                                             |
| `--share-gist`               | Publish the session transcript as a secret GitHub gist after completion. Convenient for sharing results from CI. Note that session transcripts may contain sensitive information.                                                                                                               |

### Tools for the --allow-tool option

You can specify various kinds of tools with the `--allow-tool` option.

| Kind of tool   | What it controls                                                                                                                                                                                                                           |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| shell          | Executing shell commands.                                                                                                                                                                                                                  |
| write          | Creating or modifying files.                                                                                                                                                                                                               |
| read           | Reading files or directories.                                                                                                                                                                                                              |
| url            | Fetching content from a URL.                                                                                                                                                                                                               |
| memory         | Storing new facts to the agent's persistent memory. This does not affect using existing memories. See [About agentic memory for GitHub Copilot](/en/copilot/concepts/agents/copilot-memory) .                                              |
| MCP-SERVER     | Invoking tools from a specific MCP server. Use the server's configured name as the identifier-for example, `github` . See [Adding MCP servers for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/add-mcp-servers) . |

#### Tool filters

The `shell` , `write` , `url` , and MCP server tool kinds allow you to specify a filter, in parentheses, to control which specific tools are allowed.

| Kind of tool   | Example                                  | Explanation of the example                                                     |
|----------------|------------------------------------------|--------------------------------------------------------------------------------|
| **shell**      | `shell(git:*)`                           | Allow all Git subcommands ( `git push` , `git status` , etc.).                 |
|                | `shell(npm test)`                        | Allow the exact command `npm test` .                                           |
| **write**      | `write(.github/copilot-instructions.md)` | Allow the CLI to write to this specific path.                                  |
|                | `write(README.md)`                       | Allow the CLI to write to any file whose path ends with `/README.md` .         |
| **url**        | `url(github.com)`                        | Allow the CLI to access HTTPS URLs on github.com.                              |
|                | `url(http://localhost:3000)`             | Allow the CLI to access the local dev server with explicit protocol and port.  |
|                | `url(https://*.github.com)`              | Allow the CLI to access any GitHub subdomain (for example, `api.github.com` ). |
|                | `url(https://docs.github.com/copilot/*)` | Allow access to Copilot documentation at this site.                            |
| **MCP-SERVER** | `github(create_issue)`                   | Allow only the `create_issue` tool from the `github` MCP server.               |

Note

Wildcards are only supported for `shell` to match all subcommands of a specified tool, and for `url` at the start of the host name to match any subdomain, or at the end of a path to match any path suffix-as shown in the preceding table.

### Environment variables

You can use environment variables to configure various aspects of the CLI's behavior when running programmatically. This is particularly useful for setting configuration in CI/CD workflows or other automated environments where you may not want to specify certain options directly in the command line.

| Variable               | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `COPILOT_ALLOW_ALL`    | Set to `true` for full permissions                                          |
| `COPILOT_MODEL`        | Set the model (for example, `gpt-5.2` , `claude-sonnet-4.5` )               |
| `COPILOT_HOME`         | Set the directory for the CLI configuration file ( `~/.copilot` by default) |
| `COPILOT_GITHUB_TOKEN` | Authentication token (highest precedence)                                   |
| `GH_TOKEN`             | Authentication token (second precedence)                                    |
| `GITHUB_TOKEN`         | Authentication token (third precedence)                                     |

For full details of environment variables for Copilot CLI, use the command `copilot help environment` in your terminal.

### Choosing a model

When you send a prompt to Copilot CLI in non-interactive mode, the model that the CLI uses to generate a response is shown in the response output (if the `-s` , or `--silent` , option is not used).

You can use the `--model` option to specify which AI model the CLI should use. This allows you to choose a model that is best suited to your prompt, balancing factors like speed, cost, and capability.

For example, for straightforward tasks, such as explaining some code or generating a summary, you might choose a fast, lower cost model such as a Claude Haiku model:

Bash

```
copilot -p "What does this project do?" -s --model claude-haiku-4.5
```

For more complex tasks that require deeper reasoning-such as debugging or refactoring code-you might choose a more powerful model, such as a GPT Codex model:

Bash

```
copilot -p "Fix the race condition in the worker pool" \
  --model gpt-5.3-codex \
  --allow-tool= 'write, shell'
```

Note

You can find the model strings for all available models in the description of the `--model` option when you enter `copilot help` in your terminal.

Alternatively, you can set the `COPILOT_MODEL` environment variable to specify a model for the duration of the shell session.

To persist a model selection across shell sessions, you can set the `model` key in the CLI configuration file. This file is located at `~/.copilot/config.json` (or `$COPILOT_HOME/.copilot/config.json` if you have set the `COPILOT_HOME` environment variable). Some models also allow you to set a reasoning effort level, which controls how much time the model spends thinking before responding.

JSON

```
{ "model" : "gpt-5.3-codex" , "reasoning_effort" : "low"
}
```

Tip

The easiest way to set a model persistently in the configuration file is with the `/model` slash command in an interactive session. The choice you make with this command is written to the configuration file.

#### Model precedence

When determining which model to use for a given prompt, the CLI checks for model specifications in the following order of precedence (from highest to lowest):

- Where a custom agent is used: the model specified in the custom agent definition (if any).
- The `--model` command line option.
- The `COPILOT_MODEL` environment variable.
- The `model` key in the configuration file ( `~/.copilot/config.json` or `$COPILOT_HOME/.copilot/config.json` ).
- The CLI's default model.

### Using custom agents

You can delegate work to a specialized agent by using the `--agent` option. For more information, see [Creating and using custom agents for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/create-custom-agents-for-cli) .

In this example, the `code-review` agent is used. This requires that a custom agent has been created with this name.

```
copilot -p "Review the latest commit" \
  --allow-tool= 'shell' \
  --agent code-review
```

### Further reading

- [GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli)
- [GitHub Copilot CLI command reference](/en/copilot/reference/copilot-cli-reference/cli-command-reference)
- [GitHub Copilot CLI plugin reference](/en/copilot/reference/copilot-cli-reference/cli-plugin-reference)


### Overview

The Agent Client Protocol (ACP) is a protocol that standardizes communication between clients (such as code editors and IDEs) and agents (such as Copilot CLI). For more details about this protocol, see the [official introduction](https://agentclientprotocol.com/get-started/introduction) .

### Use cases

- **IDE integrations:** Build Copilot support into any editor or development environment.
- **CI/CD pipelines:** Orchestrate agentic coding tasks in automated workflows.
- **Custom frontends:** Create specialized interfaces for specific developer workflows.
- **Multi-agent systems:** Coordinate Copilot with other AI agents using a standard protocol.

### Starting the ACP server

GitHub Copilot CLI can be started as an ACP server using the `--acp` flag. The server supports two modes, `stdio` and `TCP` .

#### stdio mode (recommended for IDE integration)

By default, when providing the `--acp` flag, `stdio` mode will be inferred. The `--stdio` flag can also be provided for disambiguation.

```
copilot --acp --stdio
```

#### TCP mode

If the `--port` flag is provided in combination with the `--acp` flag, the server is started in TCP mode.

```
copilot --acp --port 3000
```

### Integrating with the ACP server

There is a growing ecosystem of libraries to programmatically interact with ACP servers. Given GitHub Copilot CLI is correctly installed and authenticated, the following example demonstrates using the [typescript](https://agentclientprotocol.com/libraries/typescript) client to send a single prompt and print the AI response.

```
import * as acp from "@agentclientprotocol/sdk" ; import { spawn } from "node:child_process" ; import { Readable , Writable } from "node:stream" ; async function main ( ) { const executable = process. env . COPILOT_CLI_PATH ?? "copilot" ; // ACP uses standard input/output (stdin/stdout) for transport; we pipe these for the NDJSON stream. const copilotProcess = spawn (executable, [ "--acp" , "--stdio" ], { stdio : [ "pipe" , "pipe" , "inherit" ],
  }); if (!copilotProcess. stdin || !copilotProcess. stdout ) { throw new Error ( "Failed to start Copilot ACP process with piped stdio." );
  } // Create ACP streams (NDJSON over stdio) const output = Writable . toWeb (copilotProcess. stdin ) as WritableStream < Uint8Array >; const input = Readable . toWeb (copilotProcess. stdout ) as ReadableStream < Uint8Array >; const stream = acp. ndJsonStream (output, input); const client : acp. Client = { async requestPermission ( params ) { // This example should not trigger tool calls; if it does, refuse. return { outcome : { outcome : "cancelled" } };
    }, async sessionUpdate ( params ) { const update = params. update ; if (update. sessionUpdate === "agent_message_chunk" && update. content . type === "text" ) {
        process. stdout . write (update. content . text );
      }
    },
  }; const connection = new acp. ClientSideConnection ( ( _agent ) => client, stream); await connection. initialize ({ protocolVersion : acp. PROTOCOL_VERSION , clientCapabilities : {},
  }); const sessionResult = await connection. newSession ({ cwd : process. cwd (), mcpServers : [],
  });

  process. stdout . write ( "Session started!\n" ); const promptText = "Hello ACP Server!" ;
  process. stdout . write ( `Sending prompt: ' ${promptText} '\n` ); const promptResult = await connection. prompt ({ sessionId : sessionResult. sessionId , prompt : [{ type : "text" , text : promptText }],
  });

  process. stdout . write ( "\n" ); if (promptResult. stopReason !== "end_turn" ) {
    process. stderr . write ( `Prompt finished with stopReason= ${promptResult.stopReason} \n` );
  } // Best-effort cleanup copilotProcess. stdin . end ();
  copilotProcess. kill ( "SIGTERM" ); await new Promise < void >( ( resolve ) => {
    copilotProcess. once ( "exit" , () => resolve ()); setTimeout ( () => resolve (), 2000 );
  });
} main (). catch ( ( error ) => { console . error (error);
  process. exitCode = 1 ;
});
```

### Further reading

- [Official ACP documentation](https://agentclientprotocol.com/protocol/overview)


### Directory overview

The `~/.copilot` directory contains the following top-level items.

| Path                      | Type      | Description                                      |
|---------------------------|-----------|--------------------------------------------------|
| `config.json`             | File      | Your personal configuration settings             |
| `mcp-config.json`         | File      | User-level MCP server definitions                |
| `permissions-config.json` | File      | Saved tool and directory permissions per project |
| `agents/`                 | Directory | Personal custom agent definitions                |
| `skills/`                 | Directory | Personal custom skill definitions                |
| `hooks/`                  | Directory | User-level hook scripts                          |
| `logs/`                   | Directory | Session log files                                |
| `session-state/`          | Directory | Session history and workspace data               |
| `session-store.db`        | File      | SQLite database for cross-session data           |
| `installed-plugins/`      | Directory | Installed plugin files                           |
| `ide/`                    | Directory | IDE integration state                            |

Note

Not all of these items appear immediately. Some are created on demand the first time you use a particular feature-for example, `installed-plugins/` appears only after you install your first plugin.

### User-editable files

The following files are designed to be edited by you directly, or managed through CLI commands.

#### config.json

This is the primary configuration file for Copilot CLI. You can edit it directly in a text editor, or use interactive commands like `/model` and `/theme` to change specific values from within a session. The file supports JSON with comments (JSONC).

Common settings include:

| Key                   | Type     | Description                                                                                                                 |
|-----------------------|----------|-----------------------------------------------------------------------------------------------------------------------------|
| `model`               | string   | AI model to use (e.g., `"gpt-5.2"` , `"claude-sonnet-4.6"` )                                                                |
| `effortLevel`         | string   | Reasoning effort level for models that support it                                                                           |
| `theme`               | string   | Color theme: `"auto"` , `"dark"` , or `"light"`                                                                             |
| `mouse`               | boolean  | Enable mouse support in alt screen mode (default: `true` )                                                                  |
| `banner`              | string   | Animated banner frequency: `"always"` , `"once"` , or `"never"` (default: `"once"` )                                        |
| `renderMarkdown`      | boolean  | Render Markdown in responses (default: `true` )                                                                             |
| `screenReader`        | boolean  | Enable screen reader optimizations (default: `false` )                                                                      |
| `autoUpdate`          | boolean  | Automatically download CLI updates (default: `true` )                                                                       |
| `stream`              | boolean  | Stream responses token by token (default: `true` )                                                                          |
| `includeCoAuthoredBy` | boolean  | Add Co-authored-by to agent-created commits (default: `true` )                                                              |
| `respectGitignore`    | boolean  | Exclude gitignored files from the `@` file picker (default: `true` )                                                        |
| `trusted_folders`     | string[] | Folders where read/execute permission has been granted                                                                      |
| `allowed_urls`        | string[] | URLs or domains allowed without prompting                                                                                   |
| `denied_urls`         | string[] | URLs or domains that are always denied                                                                                      |
| `logLevel`            | string   | Log verbosity: `"none"` , `"error"` , `"warning"` , `"info"` , `"debug"` , `"all"` , or `"default"` (default: `"default"` ) |
| `disableAllHooks`     | boolean  | Disable all hooks (default: `false` )                                                                                       |
| `hooks`               | object   | Inline user-level hook definitions                                                                                          |

For a full list of configuration settings, enter `copilot help config` in your terminal.

Tip

Some settings can also be set using command-line flags. For example, the `/model` slash command writes your model selection to this file so it persists across sessions.

#### mcp-config.json

Defines MCP (Model Context Protocol) servers available at the user level. These servers are available in all your sessions, regardless of which project directory you're in. Project-level MCP configurations (in `.mcp.json` , `.github/mcp.json` , or `.vscode/mcp.json` ) take precedence over user-level definitions when server names conflict.

For more information, see [Adding MCP servers for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/add-mcp-servers) .

#### agents/

Store personal custom agent definitions here as `.agent.md` files. Agents placed in this directory are available in all your sessions. Project-level agents (in `.github/agents/` ) take precedence over personal agents if they share the same name.

For more information, see [Creating and using custom agents for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/create-custom-agents-for-cli) .

#### skills/

Store personal custom skill definitions here. Each skill lives in a subdirectory containing a `SKILL.md` file-for example, `~/.copilot/skills/my-skill/SKILL.md` . Personal skills are available in all your sessions. Project-level skills take precedence over personal skills if they share the same name.

For more information, see [Creating agent skills for GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/create-skills) .

#### hooks/

Store user-level hook scripts here. These hooks apply to all your sessions. You can also define hooks inline in `config.json` using the `hooks` key. Repository-level hooks (in `.github/hooks/` ) are loaded alongside user-level hooks.

For more information, see [Using hooks with GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli/customize-copilot/use-hooks) .

### Automatically managed files

The following items are managed by the CLI. You generally should not edit them manually.

#### permissions-config.json

Stores your saved tool and directory permission decisions, organized by project location. When you approve a tool or grant access to a directory, the CLI records the decision here so you aren't prompted again in the same project.

Note

If you want to reset permissions for a project, you can delete the relevant entry from this file. However, editing the file while a session is running may cause unexpected behavior.

#### session-state/

Contains session history data, organized by session ID in subdirectories. Each session directory stores an event log ( `events.jsonl` ) and workspace artifacts (plans, checkpoints, tracked files). This data enables session resume ( `--resume` or `--continue` ).

#### session-store.db

A SQLite database used by the CLI for cross-session data such as checkpoint indexing and search. This file is automatically managed and should not be edited.

#### logs/

Contains log files for CLI sessions. Each session creates a log file named `process-{timestamp}-{pid}.log` . These files are useful for debugging issues.

Tip

To find the log file for your current session, enter `/session` in an interactive session. The output includes the full path to the log file, along with other session details such as the session ID, duration, and working directory.

#### installed-plugins/

Contains the files for plugins you have installed. Plugins installed from a marketplace are stored under `installed-plugins/{marketplace-name}/{plugin-name}/` . Directly installed plugins are stored under `installed-plugins/_direct/` . Manage plugins using the `copilot plugin` commands rather than editing this directory directly.

For more information, see [GitHub Copilot CLI plugin reference](/en/copilot/reference/copilot-cli-reference/cli-plugin-reference) .

#### ide/

Contains lock files and state for IDE integrations (for example, when Copilot CLI connects to Visual Studio Code). This directory is automatically managed.

### Changing the location of the configuration directory

You can override the default `~/.copilot` location in two ways:

- **Environment variable** : Set `COPILOT_HOME` to the path of the directory you want to use. Bash `export COPILOT_HOME=/path/to/my/copilot-config`
- **Command-line option** : Use `--config-dir` when launching the CLI. Bash `copilot --config-dir /path/to/my/copilot-config`

The `--config-dir` option takes precedence over `COPILOT_HOME` , which in turn takes precedence over the default `~/.copilot` location.

#### Things to be aware of

- `COPILOT_HOME` replaces the entire `~/.copilot` path. The value you set should be the complete path to the directory you want to use for the configuration files and subdirectories.
- Changing the directory means your existing configuration, session history, installed plugins, and saved permissions will not be found in the new location. Copy or move the contents of `~/.copilot` to the new location if you want to preserve them.
- The **cache directory** (used for marketplace caches, auto-update packages, and other ephemeral data) follows platform conventions and is not affected by `COPILOT_HOME` . It is located at: To override the cache directory separately, set `COPILOT_CACHE_HOME` .
    - **macOS** : `~/Library/Caches/copilot`
    - **Linux** : `$XDG_CACHE_HOME/copilot` or `~/.cache/copilot`
    - **Windows** : `%LOCALAPPDATA%/copilot`

### What you can safely delete

| Item                             | Safe to delete?   | Effect                                                                                               |
|----------------------------------|-------------------|------------------------------------------------------------------------------------------------------|
| `logs/`                          | Yes               | Log files are re-created each session. Deleting them has no functional impact.                       |
| `session-state/`                 | With caution      | Deleting removes session history. You will no longer be able to resume past sessions.                |
| `session-store.db`               | With caution      | Deleting removes cross-session data. The file is re-created automatically.                           |
| `config.json`                    | With caution      | Resets all configuration to defaults. You will need to reconfigure your preferences.                 |
| `permissions-config.json`        | With caution      | Resets all saved permissions. The CLI will prompt you again for tool and directory approvals.        |
| `installed-plugins/`             | Not recommended   | Use `copilot plugin uninstall` instead, to ensure plugin metadata in `config.json` stays consistent. |
| `mcp-config.json`                | Not recommended   | You will lose your user-level MCP server definitions. Back up first.                                 |
| `agents/` , `skills/` , `hooks/` | Not recommended   | You will lose your personal customizations. Back up first.                                           |

### Further reading

- [GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli)
- [GitHub Copilot CLI command reference](/en/copilot/reference/copilot-cli-reference/cli-command-reference)
- [GitHub Copilot CLI programmatic reference](/en/copilot/reference/copilot-cli-reference/cli-programmatic-reference)
- [GitHub Copilot CLI plugin reference](/en/copilot/reference/copilot-cli-reference/cli-plugin-reference)


### YAML frontmatter properties

The following table outlines the properties that you can configure for agent profiles in GitHub.com, the Copilot CLI, and supported IDEs (unless otherwise noted). Any environment-specific behavior is noted in the property description. The configuration file's name (minus `.md` or `.agent.md` ) is used for deduplication between levels so that the lowest level configuration takes precedence.

| Property                   | Type                                                     | Purpose                                                                                                                                                                                                                                                                                                                    |
|----------------------------|----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `name`                     | string                                                   | Display name for the custom agent. Optional.                                                                                                                                                                                                                                                                               |
| `description`              | **Required** string                                      | Description of the custom agent's purpose and capabilities                                                                                                                                                                                                                                                                 |
| `target`                   | string                                                   | Target environment or context for the custom agent ( `vscode` or `github-copilot` ). If unset, defaults to both environments.                                                                                                                                                                                              |
| `tools`                    | list of strings, string                                  | List of tool names the custom agent can use. Supports both a comma separated string and yaml string array. If unset, defaults to all tools. See [Tools](#tools) .                                                                                                                                                          |
| `model`                    | string                                                   | Model to use when this custom agent executes. If unset, inherits the default model.                                                                                                                                                                                                                                        |
| `disable-model-invocation` | boolean                                                  | Disables Copilot cloud agent from automatically using this custom agent based on task context. When `true` , the agent must be manually selected. Setting `disable-model-invocation: true` is equivalent to `infer: false` . If both are set, `disable-model-invocation` takes precedence. If unset, defaults to `false` . |
| `user-invocable`           | boolean                                                  | Controls whether this custom agent can be selected by a user. When `false` , the agent cannot be manually selected and can only be accessed programmatically. If unset, defaults to `true` .                                                                                                                               |
| `infer`                    | boolean                                                  | **Retired** . Use `disable-model-invocation` and `user-invocable` instead. Enables Copilot cloud agent to automatically use this custom agent based on task context. When `false` , the agent must be manually selected. If unset, defaults to `true` .                                                                    |
| `mcp-servers`              | object                                                   | Additional MCP servers and tools that should be used by the custom agent. **Not used in VS Code and other IDE custom agents.**                                                                                                                                                                                             |
| `metadata`                 | object consisting of a name and value pair, both strings | Allows annotation of the agent with useful data. **Not used in VS Code and other IDE custom agents.**                                                                                                                                                                                                                      |

Define the agent's behavior, expertise, and instructions in the Markdown content below the YAML frontmatter. The prompt can be a maximum of 30,000 characters.

Note

- The `argument-hint` and `handoffs` properties from VS Code and other IDE custom agents are currently not supported for Copilot cloud agent on GitHub.com. They are ignored to ensure compatibility.
- For more information on custom agent file structure in VS Code, see [Custom agents in VS Code](https://code.visualstudio.com/docs/copilot/customization/custom-agents#_custom-agent-file-structure) in the VS Code documentation.

### Tools

The custom agent `tools` property controls which tools are available to your agent, including those from MCP servers.

Your custom agent will have access to MCP server tools that have been configured in both its agent profile and/or the repository settings. For more information on configuring MCP servers for cloud agent in a repository, see [Extending GitHub Copilot cloud agent with the Model Context Protocol (MCP)](/en/copilot/how-tos/use-copilot-agents/cloud-agent/extend-cloud-agent-with-mcp) .

You can configure `tools` using the following approaches:

- **Enable all available tools** : Omit the `tools` property entirely or use `tools: ["*"]` to enable all available tools. This will include all MCP server tools configured in the agent profile and/or repository settings.
- **Enable specific tools** : Provide a list of specific tool names or aliases (for example, `tools: ["read", "edit", "search"]` ) to enable only those tools. For available tool aliases, see [Tool aliases](#tool-aliases) below.
    - Note that if your repository has MCP servers configured, you can choose to make only specific tools from those servers available to your custom agent. Tool names from specific MCP servers can be prefixed with the server name followed by a `/` . For example, `some-mcp-server/some-tool` .
    - You can also explicitly enable all tools from a specific MCP server using `some-mcp-server/*` .
    - Tools from VS Code extensions can use the extension name as a proxy, like `azure.some-extension/some-tool` .
- **Disable all tools** : Use an empty list ( `tools: []` ) to disable all tools for the agent.

All unrecognized tool names are ignored, which allows product-specific tools to be specified in an agent profile without causing problems.

#### Tool aliases

The following tool aliases are available for custom agents. All aliases are case insensitive:

| Primary alias   | Compatible aliases                              | Cloud agent mapping                                   | Purpose                                                                                                  |
|-----------------|-------------------------------------------------|-------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `execute`       | `shell` , `Bash` , `powershell`                 | Shell tools: `bash` or `powershell`                   | Execute a command in the appropriate shell for the operating system.                                     |
| `read`          | `Read` , `NotebookRead`                         | `view`                                                | Read file contents.                                                                                      |
| `edit`          | `Edit` , `MultiEdit` , `Write` , `NotebookEdit` | Edit tools: e.g. `str_replace` , `str_replace_editor` | Allow LLM to edit. Exact arguments can vary.                                                             |
| `search`        | `Grep` , `Glob`                                 | `search`                                              | Search for files or text in files.                                                                       |
| `agent`         | `custom-agent` , `Task`                         | "Custom agent" tools                                  | Allows a different custom agent to be invoked to accomplish a task.                                      |
| `web`           | `WebSearch` , `WebFetch`                        | Currently not applicable for cloud agent.             | Allows fetching content from URLs and performing a web search                                            |
| `todo`          | `TodoWrite`                                     | Currently not applicable for cloud agent.             | Creates and manages structured task lists. Not supported in cloud agent today, but supported by VS Code. |

#### Tool names for "out-of-the-box" MCP servers

The following MCP servers are available out-of-box for Copilot cloud agent and can be referenced using namespacing:

| MCP server name   | Available tools                                                                                                                                                                                                                                                                                                                           |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `github`          | All read-only tools are available by default, but the token the server receives is scoped to the source repository. `github/*` includes all of them, or you can reference `github/<tool name>` where `<tool name>` is a value from the MCP server documentation.                                                                          |
| `playwright`      | All playwright tools are available by default, but the server is configured to only access localhost. `playwright/*` includes all of them, or you can reference `playwright/<tool name>` where `<tool name>` is a value from the MCP server documentation. By default the token it has access to is scoped to the source code repository. |

### MCP server configuration details

The following sample agent profile shows an agent with an MCP server and a secret configured. Additionally, only one tool from the MCP server has been enabled in the `tools` property in the YAML frontmatter:

```
---
name: my-custom-agent-with-mcp
description: Custom agent description
tools: ['tool-a', 'tool-b', 'custom-mcp/tool-1']
mcp-servers:
  custom-mcp:
    type: 'local'
    command: 'some-command'
    args: ['--arg1', '--arg2']
    tools: ["*"]
    env:
      ENV_VAR_NAME: ${{ secrets.COPILOT_MCP_ENV_VAR_VALUE }}

Prompt with suggestions for behavior and output
```

The `mcp-servers` property in an agent profile is a YAML representation of the JSON configuration format used to configure MCP servers for Copilot cloud agent.

Most sub-properties are the same as the JSON representation. The following sections describe changes from the initial implementation of MCP configuration in Copilot cloud agent that are relevant to custom agents. For more information about the JSON configuration format, see [Extending GitHub Copilot cloud agent with the Model Context Protocol (MCP)](/en/copilot/how-tos/use-copilot-agents/cloud-agent/extend-cloud-agent-with-mcp#writing-a-json-configuration-for-mcp-servers) .

#### MCP server type

For compatibility, the `stdio` type used by Claude Code and VS Code is mapped to cloud agent's `local` type.

#### MCP server environment variables and secrets

Note

If your MCP server requires secrets or environment variables, these must be configured in the Copilot environment in each repository where the custom agent will be used. For more information on setting up environment variables, see [Customizing the development environment for GitHub Copilot cloud agent](/en/copilot/how-tos/use-copilot-agents/cloud-agent/customize-the-agent-environment#setting-environment-variables-in-copilots-environment) .

Custom agent MCP configuration supports the same environment variable and secret replacement capabilities as existing repository-level MCP configurations. Similar to repository-level configurations, secrets and variables can be sourced from the "copilot" environment in the repository's settings. The syntax for referencing these values has been expanded to support common patterns used in GitHub Actions and Claude Code.

Both the repository-level MCP JSON configuration and the custom agent YAML configuration support the following syntax patterns:

- `$COPILOT_MCP_ENV_VAR_VALUE` - Environment variable and header
- `${COPILOT_MCP_ENV_VAR_VALUE}` - Environment variable and header (Claude Code syntax)
- `${COPILOT_MCP_ENV_VAR_VALUE:-default}` - Environment variable and header with default

The custom agent YAML configuration supports the following additional syntax patterns:

- `${{ secrets.COPILOT_MCP_ENV_VAR_VALUE }}` - Environment variable and header
- `${{ vars.COPILOT_MCP_ENV_VAR_VALUE }}` - Environment variable and header

### Example agent profile configurations

The following examples demonstrate what an agent profile could look like for the common tasks of writing tests or planning the implementation of a project. For additional inspiration, see the [Custom agents](/en/copilot/tutorials/customization-library/custom-agents) examples in the customization library. You can also find more specific examples in the [awesome-copilot](https://github.com/github/awesome-copilot/tree/main/agents) community collection.

#### Testing specialist

This example enables all tools by omitting the `tools` property.

Text

```
---
name: test-specialist
description: Focuses on test coverage, quality, and testing best practices without modifying production code

You are a testing specialist focused on improving code quality through comprehensive testing. Your responsibilities:

- Analyze existing tests and identify coverage gaps
- Write unit tests, integration tests, and end-to-end tests following best practices
- Review test quality and suggest improvements for maintainability
- Ensure tests are isolated, deterministic, and well-documented
- Focus only on test files and avoid modifying production code unless specifically requested

Always include clear test descriptions and use appropriate testing patterns for the language and framework.
```

#### Implementation planner

This example only enables a subset of tools.

Text

```
---
name: implementation-planner
description: Creates detailed implementation plans and technical specifications in markdown format
tools: ["read", "search", "edit"]

You are a technical planning specialist focused on creating comprehensive implementation plans. Your responsibilities:

- Analyze requirements and break them down into actionable tasks
- Create detailed technical specifications and architecture documentation
- Generate implementation plans with clear steps, dependencies, and timelines
- Document API designs, data models, and system interactions
- Create markdown files with structured plans that development teams can follow

Always structure your plans with clear headings, task breakdowns, and acceptance criteria. Include considerations for testing, deployment, and potential risks. Focus on creating thorough documentation rather than implementing code.
```

### Processing of custom agents

#### Custom agents names

In the case of naming conflicts, the lowest level configuration overrides higher-level configurations. This means that a repository-level agent would take precedence over an organization-level agent, and the organization-level agent would override an enterprise-level agent.

#### Versioning

Custom agent versioning is based on Git commit SHAs for the agent profile file. This allows you to create branches or tags with different versions of custom agents as needed. When you assign a custom agent to a task, the custom agent will be instantiated using the latest version of the agent profile for that repository and branch. When the agent creates a pull request, interactions within the pull request use the same version of the custom agent for consistency.

#### Tools processing

The `tools` list filters the set of tools that are made available to the agent - whether built-in or sourced from MCP servers. When you configure tools in your agent profile, the behavior depends on what you specify:

- If no tools are specified, all available tools are enabled
- An empty tools list ( `tools: []` ) disables all tools
- A specific list ( `tools: [...]` ) enables only those tools

#### MCP server configurations

For MCP server configurations, there is a specific processing order that ensures proper override behavior: out-of-the-box MCP configurations (like the GitHub MCP) are processed first, followed by the custom agent MCP configuration, and finally MCP configurations specified through repository settings. This allows each level to override settings from the previous level as appropriate.

### Further reading

- [GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli)
- [GitHub Copilot CLI command reference](/en/copilot/reference/copilot-cli-reference/cli-command-reference#custom-agents-reference)


### GitHub.com

| Copilot feature     | Types of custom instructions supported                                                                                                                                                                                                                                                               |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Copilot Chat        | - **Personal** instructions. - **Repository-wide** instructions (using the `.github/copilot-instructions.md` file). - **Organization** instructions.                                                                                                                                                 |
| Copilot cloud agent | - **Repository-wide** instructions (using the `.github/copilot-instructions.md` file). - **Path-specific** instructions (using `.github/instructions/**/*.instructions.md` files). - **Agent** instructions (using `AGENTS.md` , `CLAUDE.md` or `GEMINI.md` files). - **Organization** instructions. |
| Copilot code review | - **Repository-wide** instructions (using the `.github/copilot-instructions.md` file). - **Path-specific** instructions (using `.github/instructions/**/*.instructions.md` files). - **Organization** instructions.                                                                                  |

### Visual Studio Code

| Copilot feature     | Types of custom instructions supported                                                                                                                                                                                                                              |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Copilot Chat        | - **Repository-wide** instructions (using the `.github/copilot-instructions.md` file). - **Path-specific** instructions (using `.github/instructions/**/*.instructions.md` files). - **Agent** instructions (using an `AGENTS.md` file).                            |
| Copilot cloud agent | - **Repository-wide** instructions (using the `.github/copilot-instructions.md` file). - **Path-specific** instructions (using `.github/instructions/**/*.instructions.md` files). - **Agent** instructions (using `AGENTS.md` , `CLAUDE.md` or `GEMINI.md` files). |
| Copilot code review | - **Repository-wide** instructions (using the `.github/copilot-instructions.md` file).                                                                                                                                                                              |

### Visual Studio

| Copilot feature     | Types of custom instructions supported                                                                                                                                             |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Copilot Chat        | - **Repository-wide** instructions (using the `.github/copilot-instructions.md` file). - **Path-specific** instructions (using `.github/instructions/**/*.instructions.md` files). |
| Copilot code review | - **Repository-wide** instructions (using the `.github/copilot-instructions.md` file).                                                                                             |

### JetBrains IDEs

| Copilot feature     | Types of custom instructions supported                                                                                                                                                                                                                              |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Copilot Chat        | - **Repository-wide** instructions (using the `.github/copilot-instructions.md` file). - **Path-specific** instructions (using `.github/instructions/**/*.instructions.md` files).                                                                                  |
| Copilot cloud agent | - **Repository-wide** instructions (using the `.github/copilot-instructions.md` file). - **Path-specific** instructions (using `.github/instructions/**/*.instructions.md` files). - **Agent** instructions (using `AGENTS.md` , `CLAUDE.md` or `GEMINI.md` files). |
| Copilot code review | - **Repository-wide** instructions (using the `.github/copilot-instructions.md` file). - **Path-specific** instructions (using `.github/instructions/**/*.instructions.md` files).                                                                                  |

### Eclipse

| Copilot feature     | Types of custom instructions supported                                                                                                                                                                                                                              |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Copilot Chat        | - **Repository-wide** instructions (using the `.github/copilot-instructions.md` file).                                                                                                                                                                              |
| Copilot cloud agent | - **Repository-wide** instructions (using the `.github/copilot-instructions.md` file). - **Path-specific** instructions (using `.github/instructions/**/*.instructions.md` files). - **Agent** instructions (using `AGENTS.md` , `CLAUDE.md` or `GEMINI.md` files). |
| Copilot code review | Custom instructions are currently not supported.                                                                                                                                                                                                                    |

### Xcode

| Copilot feature     | Types of custom instructions supported                                                                                                                                                                                                                              |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Copilot Chat        | - **Repository-wide** instructions (using the `.github/copilot-instructions.md` file). - **Path-specific** instructions (using `.github/instructions/**/*.instructions.md` files).                                                                                  |
| Copilot cloud agent | - **Repository-wide** instructions (using the `.github/copilot-instructions.md` file). - **Path-specific** instructions (using `.github/instructions/**/*.instructions.md` files). - **Agent** instructions (using `AGENTS.md` , `CLAUDE.md` or `GEMINI.md` files). |
| Copilot code review | - **Repository-wide** instructions (using the `.github/copilot-instructions.md` file). - **Path-specific** instructions (using `.github/instructions/**/*.instructions.md` files).                                                                                  |

### Copilot CLI

- **Repository-wide** instructions (using the `.github/copilot-instructions.md` file).
- **Path-specific** instructions (using `.github/instructions/**/*.instructions.md` files).
- **Agent** instructions (using an `AGENTS.md` file).

### Further reading

- [Adding repository custom instructions for GitHub Copilot](/en/copilot/how-tos/configure-custom-instructions/add-repository-instructions)
- [Adding personal custom instructions for GitHub Copilot](/en/copilot/how-tos/configure-custom-instructions/add-personal-instructions)
- [Adding organization custom instructions for GitHub Copilot](/en/copilot/how-tos/configure-custom-instructions/add-organization-instructions)


### Hook types

#### Session start hook

Executed when a new agent session begins or when resuming an existing session.

**Input JSON:**

JSON

```
{ "timestamp" : 1704614400000 , "cwd" : "/path/to/project" , "source" : "new" , "initialPrompt" : "Create a new feature"
}
```

**Fields:**

- `timestamp` : Unix timestamp in milliseconds
- `cwd` : Current working directory
- `source` : Either `"new"` (new session), `"resume"` (resumed session), or `"startup"`
- `initialPrompt` : The user's initial prompt (if provided)

**Output:** Ignored (no return value processed)

**Example hook:**

JSON

```
{ "type" : "command" , "bash" : "./scripts/session-start.sh" , "powershell" : "./scripts/session-start.ps1" , "cwd" : "scripts" , "timeoutSec" : 30
}
```

**Example script (Bash):**

Shell

```
### !/bin/bash INPUT=$(cat)
SOURCE=$(echo "$INPUT" | jq -r '.source')
TIMESTAMP=$(echo "$INPUT" | jq -r '.timestamp')

echo "Session started from $SOURCE at $TIMESTAMP" >> session.log
```

#### Session end hook

Executed when the agent session completes or is terminated.

**Input JSON:**

JSON

```
{ "timestamp" : 1704618000000 , "cwd" : "/path/to/project" , "reason" : "complete"
}
```

**Fields:**

- `timestamp` : Unix timestamp in milliseconds
- `cwd` : Current working directory
- `reason` : One of `"complete"` , `"error"` , `"abort"` , `"timeout"` , or `"user_exit"`

**Output:** Ignored

**Example script:**

Shell

```
### !/bin/bash INPUT=$(cat)
REASON=$(echo "$INPUT" | jq -r '.reason')

echo "Session ended: $REASON" >> session.log # Cleanup temporary files rm -rf /tmp/session-*
```

#### User prompt submitted hook

Executed when the user submits a prompt to the agent.

**Input JSON:**

JSON

```
{ "timestamp" : 1704614500000 , "cwd" : "/path/to/project" , "prompt" : "Fix the authentication bug"
}
```

**Fields:**

- `timestamp` : Unix timestamp in milliseconds
- `cwd` : Current working directory
- `prompt` : The exact text the user submitted

**Output:** Ignored (prompt modification not currently supported in customer hooks)

**Example script:**

Shell

```
### !/bin/bash INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.prompt')
TIMESTAMP=$(echo "$INPUT" | jq -r '.timestamp') # Log to a structured file echo "$(date -d @$((TIMESTAMP/1000))): $PROMPT" >> prompts.log
```

#### Pre-tool use hook

Executed before the agent uses any tool (such as `bash` , `edit` , `view` ). This is the most powerful hook as it can **approve or deny tool executions** .

**Input JSON:**

JSON

```
{ "timestamp" : 1704614600000 , "cwd" : "/path/to/project" , "toolName" : "bash" , "toolArgs" : "{\"command\":\"rm -rf dist\",\"description\":\"Clean build directory\"}"
}
```

**Fields:**

- `timestamp` : Unix timestamp in milliseconds
- `cwd` : Current working directory
- `toolName` : Name of the tool being invoked (such as "bash", "edit", "view", "create")
- `toolArgs` : JSON string containing the tool's arguments

**Output JSON (optional):**

JSON

```
{ "permissionDecision" : "deny" , "permissionDecisionReason" : "Destructive operations require approval"
}
```

**Output fields:**

- `permissionDecision` : Either `"allow"` , `"deny"` , or `"ask"` (only `"deny"` is currently processed)
- `permissionDecisionReason` : Human-readable explanation for the decision

**Example hook to block dangerous commands:**

Shell

```
### !/bin/bash INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.toolName')
TOOL_ARGS=$(echo "$INPUT" | jq -r '.toolArgs') # Log the tool use echo "$(date): Tool=$TOOL_NAME Args=$TOOL_ARGS" >> tool-usage.log # Check for dangerous patterns if echo "$TOOL_ARGS" | grep -qE "rm -rf /|format|DROP TABLE"; then
  echo '{"permissionDecision":"deny","permissionDecisionReason":"Dangerous command detected"}'
  exit 0
fi # Allow by default (or omit output to allow) echo '{"permissionDecision":"allow"}'
```

**Example hook to enforce file permissions:**

Shell

```
### !/bin/bash INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.toolName') # Only allow editing specific directories if [ "$TOOL_NAME" = "edit" ]; then
  PATH_ARG=$(echo "$INPUT" | jq -r '.toolArgs' | jq -r '.path')

  if [[ ! "$PATH_ARG" =~ ^(src/|test/) ]]; then
    echo '{"permissionDecision":"deny","permissionDecisionReason":"Can only edit files in src/ or test/ directories"}'
    exit 0
  fi
fi # Allow all other tools
```

#### Post-tool use hook

Executed after a tool completes execution (whether successful or failed).

**Example input JSON:**

JSON

```
{ "timestamp" : 1704614700000 , "cwd" : "/path/to/project" , "toolName" : "bash" , "toolArgs" : "{\"command\":\"npm test\"}" , "toolResult" : { "resultType" : "success" , "textResultForLlm" : "All tests passed (15/15)" }
}
```

**Fields:**

- `timestamp` : Unix timestamp in milliseconds
- `cwd` : Current working directory
- `toolName` : Name of the tool that was executed
- `toolArgs` : JSON string containing the tool's arguments
- `toolResult` : Result object containing:
    - `resultType` : Either `"success"` , `"failure"` , or `"denied"`
    - `textResultForLlm` : The result text shown to the agent

**Output:** Ignored (result modification is not currently supported)

**Example script that logs tool execution statistics to a CSV file:**

This script logs tool execution statistics to a CSV file and sends an email alert when a tool fails.

Shell

```
### !/bin/bash INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.toolName')
RESULT_TYPE=$(echo "$INPUT" | jq -r '.toolResult.resultType') # Track statistics echo "$(date),${TOOL_NAME},${RESULT_TYPE}" >> tool-stats.csv # Alert on failures if [ "$RESULT_TYPE" = "failure" ]; then
  RESULT_TEXT=$(echo "$INPUT" | jq -r '.toolResult.textResultForLlm')
  echo "FAILURE: $TOOL_NAME - $RESULT_TEXT" | mail -s "Agent Tool Failed" admin@example.com
fi
```

#### Error occurred hook

Executed when an error occurs during agent execution.

**Example input JSON:**

JSON

```
{ "timestamp" : 1704614800000 , "cwd" : "/path/to/project" , "error" : { "message" : "Network timeout" , "name" : "TimeoutError" , "stack" : "TimeoutError: Network timeout\n    at ..." }
}
```

**Fields:**

- `timestamp` : Unix timestamp in milliseconds
- `cwd` : Current working directory
- `error` : Error object containing:
    - `message` : Error message
    - `name` : Error type/name
    - `stack` : Stack trace (if available)

**Output:** Ignored (error handling modification is not currently supported)

**Example script that extracts error details to a log file:**

Shell

```
### !/bin/bash INPUT=$(cat)
ERROR_MSG=$(echo "$INPUT" | jq -r '.error.message')
ERROR_NAME=$(echo "$INPUT" | jq -r '.error.name')

echo "$(date): [$ERROR_NAME] $ERROR_MSG" >> errors.log
```

### Script best practices

#### Reading input

This example script reads JSON input from stdin into a variable, then uses `jq` to extract the `timestamp` and `cwd` fields.

**Bash:**

Shell

```
### !/bin/bash
### Read JSON from stdin INPUT=$(cat) # Parse with jq TIMESTAMP=$(echo "$INPUT" | jq -r '.timestamp')
CWD=$(echo "$INPUT" | jq -r '.cwd')
```

**PowerShell:**

PowerShell

```
### Read JSON from stdin
$input = [ Console ]::In.ReadToEnd() | ConvertFrom-Json
### Access properties
$timestamp = $input .timestamp $cwd = $input .cwd
```

#### Outputting JSON

This example script shows how to output valid JSON from your hook script. Use `jq -c` in Bash for compact single-line output, or `ConvertTo-Json -Compress` in PowerShell.

**Bash:**

Shell

```
### !/bin/bash
### Use jq to compact the JSON output to a single line echo '{"permissionDecision":"deny","permissionDecisionReason":"Security policy violation"}' | jq -c # Or construct with variables REASON="Too dangerous"
jq -n --arg reason "$REASON" '{permissionDecision: "deny", permissionDecisionReason: $reason}'
```

**PowerShell:**

PowerShell

```
### Use ConvertTo-Json to compact the JSON output to a single line
$output = @ {
    permissionDecision = "deny" permissionDecisionReason = "Security policy violation" } $output | ConvertTo-Json -Compress
```

#### Error handling

This script example demonstrates how to handle errors in hook scripts.

**Bash:**

Shell

```
### !/bin/bash set -e  # Exit on error

INPUT=$(cat) # ... process input ...
### Exit with 0 for success exit 0
```

**PowerShell:**

PowerShell

```
$ErrorActionPreference = "Stop"
try { $input = [ Console ]::In.ReadToEnd() | ConvertFrom-Json # ... process input ... exit 0 } catch { Write-Error $_ .Exception.Message exit 1 }
```

#### Handling timeouts

Hooks have a default timeout of 30 seconds. For longer operations, increase `timeoutSec` :

JSON

```
{ "type" : "command" , "bash" : "./scripts/slow-validation.sh" , "timeoutSec" : 120
}
```

### Advanced patterns

#### Multiple hooks of the same type

You can define multiple hooks for the same event. They execute in order:

JSON

```
{ "version" : 1 , "hooks" : { "preToolUse" : [ { "type" : "command" , "bash" : "./scripts/security-check.sh" , "comment" : "Security validation - runs first" } , { "type" : "command" , "bash" : "./scripts/audit-log.sh" , "comment" : "Audit logging - runs second" } , { "type" : "command" , "bash" : "./scripts/metrics.sh" , "comment" : "Metrics collection - runs third" } ] }
}
```

#### Conditional logic in scripts

**Example: Only block specific tools**

Shell

```
### !/bin/bash INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.toolName') # Only validate bash commands if [ "$TOOL_NAME" != "bash" ]; then
  exit 0  # Allow all non-bash tools
fi # Check bash command for dangerous patterns COMMAND=$(echo "$INPUT" | jq -r '.toolArgs' | jq -r '.command')
if echo "$COMMAND" | grep -qE "rm -rf|sudo|mkfs"; then
  echo '{"permissionDecision":"deny","permissionDecisionReason":"Dangerous system command"}'
fi
```

#### Structured logging

**Example: JSON Lines format**

Shell

```
### !/bin/bash INPUT=$(cat)
TIMESTAMP=$(echo "$INPUT" | jq -r '.timestamp')
TOOL_NAME=$(echo "$INPUT" | jq -r '.toolName')
RESULT_TYPE=$(echo "$INPUT" | jq -r '.toolResult.resultType') # Output structured log entry jq -n \
  --arg ts "$TIMESTAMP" \
  --arg tool "$TOOL_NAME" \
  --arg result "$RESULT_TYPE" \
  '{timestamp: $ts, tool: $tool, result: $result}' >> logs/audit.jsonl
```

#### Integration with external systems

**Example: Send alerts to Slack**

Shell

```
### !/bin/bash INPUT=$(cat)
ERROR_MSG=$(echo "$INPUT" | jq -r '.error.message')

WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

curl -X POST "$WEBHOOK_URL" \
  -H 'Content-Type: application/json' \
  -d "{\"text\":\"Agent Error: $ERROR_MSG\"}"
```

### Example use cases

#### Compliance audit trail

Log all agent actions for compliance requirements by utilizing log scripts:

JSON

```
{ "version" : 1 , "hooks" : { "sessionStart" : [ { "type" : "command" , "bash" : "./audit/log-session-start.sh" } ] , "userPromptSubmitted" : [ { "type" : "command" , "bash" : "./audit/log-prompt.sh" } ] , "preToolUse" : [ { "type" : "command" , "bash" : "./audit/log-tool-use.sh" } ] , "postToolUse" : [ { "type" : "command" , "bash" : "./audit/log-tool-result.sh" } ] , "sessionEnd" : [ { "type" : "command" , "bash" : "./audit/log-session-end.sh" } ] }
}
```

#### Cost tracking

Track tool usage for cost allocation:

Shell

```
### !/bin/bash INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.toolName')
TIMESTAMP=$(echo "$INPUT" | jq -r '.timestamp')
USER=${USER:-unknown}

echo "$TIMESTAMP,$USER,$TOOL_NAME" >> /var/log/copilot/usage.csv
```

#### Code quality enforcement

Prevent commits that violate code standards:

Shell

```
### !/bin/bash INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.toolName')

if [ "$TOOL_NAME" = "edit" ] || [ "$TOOL_NAME" = "create" ]; then # Run linter before allowing edits npm run lint-staged
  if [ $? -ne 0 ]; then
    echo '{"permissionDecision":"deny","permissionDecisionReason":"Code does not pass linting"}'
  fi
fi
```

#### Notification system

Send notifications on important events:

Shell

```
### !/bin/bash INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.prompt') # Notify on production-related prompts if echo "$PROMPT" | grep -iq "production"; then
  echo "ALERT: Production-related prompt: $PROMPT" | mail -s "Agent Alert" team@example.com
fi
```

### Further reading

- [Concepts for GitHub Copilot cloud agent](/en/copilot/concepts/agents/cloud-agent)
- [GitHub Copilot CLI](/en/copilot/how-tos/copilot-cli)
- [GitHub Copilot CLI command reference](/en/copilot/reference/copilot-cli-reference/cli-command-reference)
