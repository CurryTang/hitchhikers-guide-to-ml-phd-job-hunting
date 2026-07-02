import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';

describe('App', () => {
  beforeEach(() => {
    window.history.replaceState(null, '', '/');
    vi.restoreAllMocks();
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const requestUrl = String(input);
      const english = requestUrl.includes('.en.md');
      const chineseContent = [
        '# 中文教程',
        '',
        '这是中文版本。',
        '',
        '```quiz',
        'title: Quick Check',
        'question: CUDA thread blocks are scheduled onto which hardware unit?',
        'answer: B',
        'A. Host compiler',
        'B. GPU SM',
        'C. Browser runtime',
        'explanation: Blocks are assigned to streaming multiprocessors.',
        '```',
        '',
        '```python',
        'def can_jump(nums):',
        '    return True',
        '```',
        '',
        'Inline math $QK^T$ and display math:',
        '',
        '$$',
        '\\sum_i x_i',
        '$$',
      ].join('\n');

      return {
        ok: true,
        text: async () =>
          english
            ? '# English tutorial\n\nThis is the English version.'
            : chineseContent,
      };
    });
  });

  it('starts on a prominent home page with MLSYS navigation', () => {
    render(<App />);

    expect(screen.getByRole('heading', { name: /ML \/ LLM 技术复习笔记/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /start mlsys/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^mlsys/i })).toBeInTheDocument();
  });

  it('shows author contact details in the About section', () => {
    render(<App />);

    expect(screen.getByRole('heading', { name: /关于作者/ })).toBeInTheDocument();
    expect(screen.getByText(/agent memory/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /githubgithub\.com\/currytang/i })).toHaveAttribute(
      'href',
      'https://github.com/CurryTang',
    );
    expect(screen.getByRole('link', { name: /linkedinzhikai chen/i })).toHaveAttribute(
      'href',
      'https://www.linkedin.com/in/zhikai-chen-435252129',
    );
    expect(screen.getByRole('link', { name: /emailchenzh85@msu\.edu/i })).toHaveAttribute(
      'href',
      'mailto:chenzh85@msu.edu',
    );
  });

  it('keeps the same tutorial selected while switching languages in place', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: /start mlsys/i }));
    fireEvent.click(await screen.findByRole('button', { name: /MLSYS1 · GPU 体系结构入门/i }));

    const initialHeading = await screen.findByRole('heading', {
      name: /mlsys1/i,
    });

    expect(initialHeading).toBeInTheDocument();
    expect(await screen.findByText('这是中文版本。')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /english/i }));

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /mlsys1/i })).toBeInTheDocument();
    });

    expect(screen.getByRole('button', { name: /english/i })).toHaveAttribute('aria-pressed', 'true');
    expect(await screen.findByText('This is the English version.')).toBeInTheDocument();
  });

  it('renders interactive multiple-choice practice blocks', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: /start mlsys/i }));

    expect(await screen.findByText('Quick Check')).toBeInTheDocument();
    expect(screen.getByText(/CUDA thread blocks/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /Browser runtime/i }));

    expect(await screen.findByText(/Not quite/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /GPU SM/i }));

    expect(await screen.findByText(/Correct/)).toBeInTheDocument();
    expect(screen.getByText(/streaming multiprocessors/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /hide/i }));

    expect(screen.queryByText(/CUDA thread blocks/)).not.toBeInTheDocument();
  });

  it('renders enhanced code blocks with language labels', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: /start mlsys/i }));

    expect(await screen.findByText('Python')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Copy' })).toBeInTheDocument();
    expect(screen.getByText('def')).toHaveClass('code-token', 'keyword');
  });

  it('renders Markdown math through KaTeX without losing LaTeX commands', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: /start mlsys/i }));

    expect(await screen.findByText(/Inline math/)).toBeInTheDocument();

    const annotations = Array.from(document.querySelectorAll('annotation[encoding="application/x-tex"]')).map(
      (node) => node.textContent,
    );

    expect(annotations).toContain('QK^T');
    expect(annotations).toContain('\\sum_i x_i');
  });

  it('keeps the reader sidebar scoped to the current section', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: /start mlsys/i }));

    expect(await screen.findByRole('heading', { name: /MLSYS1/i })).toBeInTheDocument();

    const sidebar = document.querySelector('.notes-panel');
    expect(sidebar).not.toBeNull();
    expect(within(sidebar).getByRole('heading', { name: 'MLSYS' })).toBeInTheDocument();
    expect(within(sidebar).getByText('18 notes in this section')).toBeInTheDocument();
    expect(within(sidebar).queryByText('LLM八股')).not.toBeInTheDocument();
    expect(within(sidebar).queryByText('LeetCode')).not.toBeInTheDocument();
  });

  it('opens the LeetCode section from the top navigation', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));

    expect(await screen.findByRole('heading', { name: /Core Skills 1/i })).toBeInTheDocument();
    expect(screen.getAllByText('CoreSkills01 Design Dynamic Array.md')).toHaveLength(2);
  });

  it('opens the System Design section with the new overview notes', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'System Design' }));

    expect(await screen.findByRole('heading', { name: /System Design 0/i })).toBeInTheDocument();
    expect(screen.getByText('7 notes in this section')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 1 · 无状态设计范式/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 2 · 数据库基本范式/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 3 · 数据库扩展三件套/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 4 · 存储系统/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 5 · 设计题基本流程/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /System Design 99 · 高频术语整合/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /System Design 2 · 数据库基本范式/i }));

    expect(await screen.findByRole('heading', { name: /数据库基本范式/ })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /System Design 3 · 数据库扩展三件套/i }));

    expect(await screen.findByRole('heading', { name: /Feature Store 分片的代价/ })).toBeInTheDocument();
    expect(screen.getByText(/Push \/ active update/)).toBeInTheDocument();
    expect(screen.getByText('Database Scaling Check 1')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /System Design 4 · 存储系统/i }));

    expect(await screen.findByRole('heading', { name: /中文教程/ })).toBeInTheDocument();
    expect(screen.getAllByText('SystemDesign04 Storage Systems.md')).toHaveLength(2);

    fireEvent.click(screen.getByRole('button', { name: /System Design 5 · 设计题基本流程/i }));

    expect(await screen.findByRole('heading', { name: /设计题基本流程/ })).toBeInTheDocument();
    expect(screen.getAllByText('SystemDesign05 Interview Flow.md')).toHaveLength(2);

    fireEvent.click(screen.getByRole('button', { name: /System Design 99 · 高频术语整合/i }));

    expect(await screen.findByRole('heading', { name: /高频术语整合/ })).toBeInTheDocument();
    expect(screen.getAllByText('SystemDesign99 Glossary.md')).toHaveLength(2);
  });

  it('shows local-only draft notes in development mode', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: '草稿区' }));

    expect(await screen.findByRole('heading', { name: '草稿区' })).toBeInTheDocument();
    expect(screen.getByText('2 notes in this section')).toBeInTheDocument();
    expect(screen.getAllByText(/LLM八股 Overview · JD 高频主题拆解/).length).toBeGreaterThan(0);
    expect(screen.getByRole('button', { name: /Quant 草稿 · 概率基础公式与记忆框架/i })).toBeInTheDocument();
    expect(await screen.findByRole('heading', { name: /Motivation/ })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /7\. RLVR & Agentic RL/ })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /Quant 草稿 · 概率基础公式与记忆框架/i }));
    expect(await screen.findByRole('heading', { name: /先按题型选工具/ })).toBeInTheDocument();
    expect(
      Array.from(document.querySelectorAll('annotation[encoding="application/x-tex"]')).some((node) =>
        node.textContent?.includes('\\mathbb{E}[X]'),
      ),
    ).toBe(true);

    expect(screen.queryByRole('button', { name: /System Design 草稿 · 数据库扩展三件套/i })).not.toBeInTheDocument();
  });
});
