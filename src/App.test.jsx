import { fireEvent, render, screen, waitFor } from '@testing-library/react';
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

    expect(screen.getByRole('heading', { name: /面试复习的系统化笔记库/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /start mlsys/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^mlsys/i })).toBeInTheDocument();
  });

  it('keeps the same tutorial selected while switching languages in place', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: /start mlsys/i }));

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

  it('opens the LeetCode section from the top navigation', async () => {
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: 'LeetCode' }));

    expect(await screen.findByRole('heading', { name: /Core Skills 1/i })).toBeInTheDocument();
    expect(screen.getAllByText('CoreSkills01 Design Dynamic Array.md')).toHaveLength(2);
  });
});
