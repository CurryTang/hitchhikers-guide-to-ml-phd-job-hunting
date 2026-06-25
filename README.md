# Interview Notes

This repository hosts an interview notes site. MLSYS notes and LeetCode Core Skills notes are published as parallel interview sections.

## Live Site

**GitHub Pages:** [https://currytang.github.io/MLSYS_tutorial/](https://currytang.github.io/MLSYS_tutorial/)

The site publishes the curated MLSYS notes from `notes/Mlsys/` and LeetCode Core Skills notes from `notes/Leetcode/`. The frontend reader supports Chinese and English variants when both exist, and falls back to the available note when only one variant is present.

## Repository Layout

- `notes/Mlsys/`: MLSYS interview note markdown files and local assets
- `notes/Leetcode/`: LeetCode Core Skills note markdown files
- `src/`: React frontend for browsing and rendering interview sections
- `docs/plans/`: design and implementation notes for repo changes

## Practice Blocks

Markdown files can render interactive multiple-choice practice blocks with a fenced code block:

````
```quiz
title: Quick Check
question: CUDA thread blocks are scheduled onto which hardware unit?
answer: B
A. Host compiler
B. GPU SM
C. Browser runtime
explanation: Blocks are assigned to streaming multiprocessors.
```
````

Use `mcq` instead of `quiz` if you prefer that fence name. The rendered block can be hidden, and option clicks show whether the selected answer is correct.

Long-form answers can be kept collapsible with native Markdown/HTML details blocks:

```html
<details class="solution">
<summary>展开解法</summary>

Write the reference solution, pseudocode, and complexity analysis here.

</details>
```

## Local Development

```bash
npm install
npm run dev
```

## Verification

```bash
npm test
npm run lint
npm run build
```

## Deployment Notes

GitHub Pages deployment is handled by `.github/workflows/deploy-pages.yml`.

Vite infers the production base path from `GITHUB_REPOSITORY` in CI. If needed, override it with `VITE_BASE_PATH` during the build.
