import { Fragment, useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import 'katex/dist/katex.min.css';
import './App.css';

const markdownModules = import.meta.glob('../notes/**/*.md', {
  eager: true,
  import: 'default',
  query: '?url',
});

const mlsysNoteDefinitions = [
  createTutorialDefinition('MLSYS Overview：学习路径与目录', 'MLSYS Overview.md', null),
  createTutorialDefinition('MLSYS1 · GPU 体系结构入门', 'MLSYS1.md', 'MLSYS1.en.md'),
  createTutorialDefinition('MLSYS2 · CUDA 编程模型与 GPU 组件', 'MLSYS2.md', 'MLSYS2.en.md'),
  createTutorialDefinition('MLSYS3 · Roofline Analysis', 'MLSYS3.md', 'MLSYS3.en.md'),
  createTutorialDefinition('MLSYS4 · CUDA Reduce Kernel 完全指南', 'MLSYS4.md', 'MLSYS4.en.md'),
  createTutorialDefinition('MLSYS5 · CUDA Parallel Primitives: Histogram & Scan', 'MLSYS5.md', 'MLSYS5.en.md'),
  createTutorialDefinition('MLSYS6 · Memory-Bound Kernel 优化', 'MLSYS6.md', 'MLSYS6.en.md'),
  createTutorialDefinition(
    'MLSYS7 · Compute-Bound Kernel (1)',
    'MLSYS7 Compute-Bound Kernel (1).md',
    'MLSYS7 Compute-Bound Kernel (1).en.md',
  ),
  createTutorialDefinition(
    'MLSYS8 · Compute-Bound Kernel (2)',
    'MLSYS8 Compute-Bound Kernel (2).md',
    'MLSYS8 Compute-Bound Kernel (2).en.md',
  ),
  createTutorialDefinition(
    'MLSYS9 · Compute-Bound Kernel (3)',
    'MLSYS9 Compute-bound kernel (3).md',
    'MLSYS9 Compute-bound kernel (3).en.md',
  ),
  createTutorialDefinition('MLSYS10 · 分布式训练并行范式', 'MLSYS10 parallelism.md', 'MLSYS10 parallelism.en.md'),
  createTutorialDefinition('MLSYS11 · nano-vllm 精读 (1)', 'MLSYS11 nano-vllm-1.md', 'MLSYS11 nano-vllm-1.en.md'),
  createTutorialDefinition('MLSYS12 · nano-vllm 精读 (2)', 'MLSYS12 nano-vllm-2.md', 'MLSYS12 nano-vllm-2.en.md'),
  createTutorialDefinition(
    'MLSYS13 · Low-bit Quantization 核心方法详解',
    'MLSYS13 Quantization and precision.md',
    'MLSYS13 Quantization and precision.en.md',
  ),
];

const mlsysNotes = mlsysNoteDefinitions.map((definition) => ({
  ...definition,
  variants: {
    zh: createVariant(definition.zhFileName, definition.directory),
    en: createVariant(definition.enFileName, definition.directory),
  },
}));

const leetcodeNoteDefinitions = [
  createTutorialDefinition(
    'Core Skills 1 · Design Dynamic Array',
    'CoreSkills01 Design Dynamic Array.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Easy' },
  ),
  createTutorialDefinition(
    'Core Skills 2 · Design Singly Linked List',
    'CoreSkills02 Design Singly Linked List.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Easy' },
  ),
  createTutorialDefinition(
    'Core Skills 3 · Design Double-ended Queue',
    'CoreSkills03 Design Double-ended Queue.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Easy' },
  ),
  createTutorialDefinition(
    'Core Skills 4 · Design Binary Search Tree',
    'CoreSkills04 Design Binary Search Tree.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 5 · Design Hash Table',
    'CoreSkills05 Design Hash Table.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 6 · Design Heap',
    'CoreSkills06 Design Heap.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 7 · Design Graph',
    'CoreSkills07 Design Graph.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 8 · Design Disjoint Set',
    'CoreSkills08 Design Disjoint Set Union Find.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 9 · Design Segment Tree',
    'CoreSkills09 Design Segment Tree.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'Core Skills 10 · Insertion Sort',
    'CoreSkills10 Insertion Sort.md',
    null,
    { directory: 'Leetcode', category: 'Sorting', difficulty: 'Easy' },
  ),
  createTutorialDefinition(
    'Core Skills 11 · Merge Sort',
    'CoreSkills11 Merge Sort.md',
    null,
    { directory: 'Leetcode', category: 'Sorting', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 12 · Quick Sort',
    'CoreSkills12 Quick Sort.md',
    null,
    { directory: 'Leetcode', category: 'Sorting', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 13 · Matrix DFS',
    'CoreSkills13 Matrix Depth First Search.md',
    null,
    { directory: 'Leetcode', category: 'Graphs', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 14 · Matrix BFS',
    'CoreSkills14 Matrix Breadth First Search.md',
    null,
    { directory: 'Leetcode', category: 'Graphs', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    "Core Skills 15 · Shortest Path: Dijkstra & Bellman-Ford",
    'CoreSkills15 Dijkstra Algorithm.md',
    null,
    { directory: 'Leetcode', category: 'Graphs', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    "Core Skills 16 · Prim's Algorithm",
    'CoreSkills16 Prim Algorithm.md',
    null,
    { directory: 'Leetcode', category: 'Graphs', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    "Core Skills 17 · Kruskal's Algorithm",
    'CoreSkills17 Kruskal Algorithm.md',
    null,
    { directory: 'Leetcode', category: 'Graphs', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'Core Skills 18 · Topological Sort / Foreign Dictionary',
    'CoreSkills18 Topological Sort.md',
    null,
    { directory: 'Leetcode', category: 'Graphs', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'Core Skills 19 · 0 / 1 Knapsack',
    'CoreSkills19 0-1 Knapsack.md',
    null,
    { directory: 'Leetcode', category: 'Dynamic Programming', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 20 · Unbounded Knapsack',
    'CoreSkills20 Unbounded Knapsack.md',
    null,
    { directory: 'Leetcode', category: 'Dynamic Programming', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 21 · Dynamic Programming',
    'CoreSkills21 Decode Ways Dynamic Programming.md',
    null,
    { directory: 'Leetcode', category: 'Dynamic Programming', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 22 · Rejection Sampling / Rand10',
    'CoreSkills22 Rejection Sampling Rand10.md',
    null,
    { directory: 'Leetcode', category: 'Math & Probability', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 23 · Greedy Algorithms',
    'CoreSkills23 Greedy Algorithms.md',
    null,
    { directory: 'Leetcode', category: 'Greedy', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 24 · Interval Problems',
    'CoreSkills24 Interval Problems.md',
    null,
    { directory: 'Leetcode', category: 'Intervals', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 25 · Math: Fast Power',
    'CoreSkills25 Math Binary Exponentiation.md',
    null,
    { directory: 'Leetcode', category: 'Math', difficulty: 'Medium' },
  ),
];

const leetcodeNotes = leetcodeNoteDefinitions.map((definition) => ({
  ...definition,
  variants: {
    zh: createVariant(definition.zhFileName, definition.directory),
    en: createVariant(definition.enFileName, definition.directory),
  },
}));

const llmNoteDefinitions = [
  createTutorialDefinition(
    'LLM八股 14 · Post-Training Infra：从 TRL 到 Forge',
    'MLSYS14 Post-Training Infra.md',
    'MLSYS14 Post-Training Infra.en.md',
  ),
  createTutorialDefinition(
    'LLM八股 15 · Inference：Speculative Decoding 到 DFlash',
    'MLSYS15 LLM Inference Speculative Decoding DFlash.md',
    null,
  ),
  createTutorialDefinition(
    'LLM八股 16 · Modern MoE：SonicMoE 深入解析',
    'MLSYS16 Modern MoE SonicMoE.md',
    null,
  ),
  createTutorialDefinition(
    'LLM八股 附录 · RL Infra 自测 35 问',
    'MLSYS15 RL Infra 自测 35 问.md',
    'MLSYS15 RL Infra 自测 35 问.en.md',
  ),
];

const llmNotes = llmNoteDefinitions.map((definition) => ({
  ...definition,
  variants: {
    zh: createVariant(definition.zhFileName, definition.directory),
    en: createVariant(definition.enFileName, definition.directory),
  },
}));

const noteSections = [
  {
    id: 'mlsys',
    title: 'MLSYS',
    description: 'GPU kernels, training systems, inference systems, and performance notes',
    notes: mlsysNotes,
  },
  {
    id: 'llm',
    title: 'LLM八股',
    description: 'LLM post-training, RL infra, and framework interview notes',
    notes: llmNotes,
  },
  {
    id: 'leetcode',
    title: 'LeetCode',
    description: 'Core data structure and algorithm interview drills',
    notes: leetcodeNotes,
  },
];

const tutorials = noteSections.flatMap((section) =>
  section.notes.map((note) => ({
    ...note,
    sectionId: section.id,
    sectionTitle: section.title,
  })),
);

const noteIdByAlias = buildNoteAliasMap(tutorials);
const mediaModules = import.meta.glob('../notes/**/assets/**/*.{png,jpg,jpeg,gif,webp,svg,avif,bmp}', {
  eager: true,
  import: 'default',
  query: '?url',
});
const mediaUrlByAlias = buildMediaAliasMap(mediaModules);
const languageOptions = [
  { id: 'zh', label: '中文' },
  { id: 'en', label: 'English' },
];

const homeStats = [
  { value: noteSections.length, label: 'Sections' },
  { value: tutorials.length, label: 'Notes' },
  { value: '2', label: 'Languages' },
];

function createTutorialDefinition(title, zhFileName, enFileName, options = {}) {
  const directory = options.directory ?? 'Mlsys';
  return {
    id: zhFileName,
    title,
    fileName: zhFileName,
    zhFileName,
    enFileName,
    directory,
    category: options.category ?? '',
    difficulty: options.difficulty ?? '',
  };
}

function createVariant(fileName, directory) {
  if (!fileName) {
    return {
      fileName: '',
      url: null,
    };
  }

  const modulePath = `../notes/${directory}/${fileName}`;
  const url = markdownModules[modulePath];
  return {
    fileName,
    url: typeof url === 'string' ? url : null,
  };
}

function normalizePathToken(rawValue) {
  if (!rawValue) {
    return '';
  }

  let value = rawValue.trim().replace(/\\/g, '/');
  try {
    value = decodeURIComponent(value);
  } catch {
    // Ignore malformed URI fragments and keep the original token.
  }

  value = value.replace(/^\.\//, '');
  value = value.replace(/^\//, '');
  value = value.replace(/^notes\//i, '');

  return value.toLowerCase();
}

function buildNoteAliasMap(tutorialList) {
  const map = new Map();

  const addAlias = (alias, id) => {
    const normalized = normalizePathToken(alias);
    if (normalized && !map.has(normalized)) {
      map.set(normalized, id);
    }
  };

  tutorialList.forEach((tutorial) => {
    const fileNames = [tutorial.variants.zh.fileName, tutorial.variants.en.fileName].filter(Boolean);

    addAlias(tutorial.id, tutorial.id);
    addAlias(tutorial.fileName, tutorial.id);
    addAlias(`${tutorial.directory}/${tutorial.fileName}`, tutorial.id);
    addAlias(`notes/${tutorial.directory}/${tutorial.fileName}`, tutorial.id);

    fileNames.forEach((fileName) => {
      const withoutMd = fileName.replace(/\.md$/i, '');
      const withoutLang = withoutMd.replace(/\.en$/i, '');
      addAlias(fileName, tutorial.id);
      addAlias(`${tutorial.directory}/${fileName}`, tutorial.id);
      addAlias(`notes/${tutorial.directory}/${fileName}`, tutorial.id);
      addAlias(withoutMd, tutorial.id);
      addAlias(withoutLang, tutorial.id);
    });
  });

  return map;
}

function buildMediaAliasMap(modules) {
  const map = new Map();

  const addAlias = (alias, url) => {
    const normalized = normalizePathToken(alias);
    if (normalized && !map.has(normalized)) {
      map.set(normalized, url);
    }
  };

  Object.entries(modules).forEach(([modulePath, assetUrl]) => {
    if (typeof assetUrl !== 'string') {
      return;
    }

    const relativePath = modulePath.replace('../notes/', '');
    const fileName = relativePath.split('/').at(-1) ?? relativePath;
    addAlias(relativePath, assetUrl);
    addAlias(`notes/${relativePath}`, assetUrl);
    addAlias(fileName, assetUrl);
    addAlias(`assets/${fileName}`, assetUrl);
    addAlias(`./assets/${fileName}`, assetUrl);
  });

  return map;
}

function splitObsidianTarget(rawContent) {
  const [targetPart, ...aliasParts] = rawContent.split('|');
  const target = targetPart?.trim() ?? '';
  const aliasRaw = aliasParts.join('|').trim();

  if (!aliasRaw || /^\d+$/.test(aliasRaw)) {
    return { target, alias: '' };
  }

  return { target, alias: aliasRaw };
}

function prettyLabel(rawTarget) {
  const [withoutAnchor] = rawTarget.split('#');
  const token = withoutAnchor.split('/').at(-1) ?? withoutAnchor;
  const anchor = rawTarget.includes('#') ? cleanHeadingText(rawTarget.split('#').slice(1).join('#')) : '';
  return token.replace(/\.en\.md$/i, '').replace(/\.md$/i, '').trim() || anchor || rawTarget.trim();
}

function resolveNoteId(rawTarget) {
  const [withoutAnchor] = rawTarget.split('#');
  const normalized = normalizePathToken(withoutAnchor);

  if (!normalized) {
    return null;
  }

  const basename = normalized.split('/').at(-1) ?? normalized;
  const candidates = [
    normalized,
    normalized.endsWith('.md') ? normalized.slice(0, -3) : `${normalized}.md`,
    basename,
    basename.endsWith('.md') ? basename.slice(0, -3) : `${basename}.md`,
  ];

  for (const candidate of candidates) {
    const match = noteIdByAlias.get(candidate);
    if (match) {
      return match;
    }
  }

  return null;
}

function resolveObsidianLink(target, alias) {
  if (target.startsWith('#')) {
    const heading = cleanHeadingText(target.slice(1));
    if (!heading) {
      return alias || '';
    }

    return `[${alias || heading}](#${slugify(heading)})`;
  }

  const noteId = resolveNoteId(target);
  if (!noteId) {
    return null;
  }

  return `[${alias || prettyLabel(target)}](#${encodeURIComponent(noteId)})`;
}

function resolveMediaUrl(rawTarget) {
  const [withoutAnchor] = rawTarget.split('#');
  const normalized = normalizePathToken(withoutAnchor);

  if (!normalized) {
    return null;
  }

  const basename = normalized.split('/').at(-1) ?? normalized;
  const candidates = [normalized, basename, `mlsys/assets/${basename}`, `assets/${basename}`];

  for (const candidate of candidates) {
    const match = mediaUrlByAlias.get(candidate);
    if (match) {
      return match;
    }
  }

  return null;
}

function slugify(text) {
  return String(text)
    .toLowerCase()
    .replace(/\s+/g, '-')
    .replace(/[^\w一-龥-]/g, '')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
}

function HeadingWithAnchor({ level, children }) {
  const Tag = `h${level}`;
  const text = extractPlainText(children);
  const id = slugify(text);
  const scrollToSection = (event) => {
    event.preventDefault();
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <Tag id={id} className="heading-anchor-host">
      {children}
      <a href={`#${id}`} className="heading-anchor" aria-label="Link to section" onClick={scrollToSection}>¶</a>
    </Tag>
  );
}

function extractPlainText(value) {
  if (typeof value === 'string' || typeof value === 'number') {
    return String(value);
  }

  if (Array.isArray(value)) {
    return value.map(extractPlainText).join('');
  }

  if (value?.props?.children) {
    return extractPlainText(value.props.children);
  }

  return '';
}

function extractMarkdownHeadings(markdownText) {
  if (!markdownText) {
    return [];
  }

  return markdownText
    .replace(/```[\s\S]*?```/g, '')
    .split('\n')
    .map((line) => {
      const match = /^(#{1,3})\s+(.+?)\s*$/.exec(line);
      if (!match) {
        return null;
      }

      const text = cleanHeadingText(match[2]);
      if (!text) {
        return null;
      }

      return {
        id: slugify(text),
        level: match[1].length,
        text,
      };
    })
    .filter(Boolean);
}

function cleanHeadingText(text) {
  return text
    .replace(/<[^>]+>/g, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[`*_~]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function normalizeAnswerToken(rawValue) {
  const value = String(rawValue ?? '')
    .trim()
    .replace(/^[([{\s]+/g, '')
    .replace(/[\])}\s.。:：]+$/g, '');
  if (!value) {
    return '';
  }

  if (/^\d+$/.test(value)) {
    return String(Number(value) - 1);
  }

  return value.charAt(0).toUpperCase();
}

function parseQuizSource(rawSource) {
  const lines = String(rawSource ?? '')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);

  const quiz = {
    title: 'Practice',
    question: '',
    answer: '',
    explanation: '',
    options: [],
  };

  lines.forEach((line) => {
    const fieldMatch = line.match(/^(title|question|answer|correct|explanation|解析|答案)\s*[:：]\s*(.+)$/i);
    if (fieldMatch) {
      const [, rawKey, rawValue] = fieldMatch;
      const key = rawKey.toLowerCase();
      if (key === 'correct' || key === 'answer' || rawKey === '答案') {
        quiz.answer = normalizeAnswerToken(rawValue);
      } else if (key === 'explanation' || rawKey === '解析') {
        quiz.explanation = rawValue.trim();
      } else {
        quiz[key] = rawValue.trim();
      }
      return;
    }

    const optionMatch = line.match(/^(?:[-*]\s*)?([A-Ha-h]|\d+)[).、:：]\s+(.+)$/);
    if (optionMatch) {
      const [, rawKey, text] = optionMatch;
      quiz.options.push({
        id: normalizeAnswerToken(rawKey),
        label: /^[A-Ha-h]$/.test(rawKey) ? rawKey.toUpperCase() : String(quiz.options.length + 1),
        text: text.trim(),
      });
      return;
    }

    if (!quiz.question) {
      quiz.question = line;
    }
  });

  if (!quiz.answer && quiz.options.some((option) => /^\*/.test(option.text))) {
    const correctOption = quiz.options.find((option) => /^\*/.test(option.text));
    quiz.answer = correctOption.id;
    quiz.options = quiz.options.map((option) => ({
      ...option,
      text: option.text.replace(/^\*\s*/, ''),
    }));
  }

  return quiz;
}

function QuizBlock({ source }) {
  const quiz = useMemo(() => parseQuizSource(source), [source]);
  const [collapsed, setCollapsed] = useState(false);
  const [selectedOption, setSelectedOption] = useState('');

  const isAnswered = Boolean(selectedOption);
  const isCorrect = selectedOption === quiz.answer;

  if (!quiz.question || quiz.options.length === 0 || !quiz.answer) {
    return (
      <pre>
        <code>{source}</code>
      </pre>
    );
  }

  return (
    <section className={`practice-card ${collapsed ? 'collapsed' : ''}`}>
      <button
        className="practice-card-toggle"
        type="button"
        onClick={() => setCollapsed((current) => !current)}
        aria-expanded={!collapsed}
        aria-label={`${collapsed ? 'Show' : 'Hide'} ${quiz.title}`}
      >
        <span>{quiz.title}</span>
        <span aria-hidden="true">{collapsed ? 'Show' : 'Hide'}</span>
      </button>

      {!collapsed && (
        <div className="practice-card-body">
          <p className="practice-question">{quiz.question}</p>
          <div className="practice-options" role="group" aria-label={quiz.question}>
            {quiz.options.map((option) => {
              const optionSelected = selectedOption === option.id;
              const optionCorrect = option.id === quiz.answer;
              const stateClass = isAnswered && optionSelected
                ? isCorrect
                  ? 'correct'
                  : 'incorrect'
                : isAnswered && optionCorrect
                  ? 'correct'
                  : '';

              return (
                <button
                  key={option.id}
                  className={`practice-option ${stateClass}`}
                  type="button"
                  onClick={() => setSelectedOption(option.id)}
                  aria-pressed={optionSelected}
                >
                  <span className="practice-option-key">{option.label}</span>
                  <span>{option.text}</span>
                </button>
              );
            })}
          </div>
          {isAnswered && (
            <p className={`practice-feedback ${isCorrect ? 'correct' : 'incorrect'}`} role="status">
              {isCorrect ? 'Correct.' : 'Not quite.'}
              {quiz.explanation ? ` ${quiz.explanation}` : ''}
            </p>
          )}
        </div>
      )}
    </section>
  );
}

function ForeignDictionaryTopoVisual() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackKey, setPlaybackKey] = useState(0);
  const comparisons = [
    ['hrn', 'hrf', 'n -> f', 'pair-1'],
    ['hrf', 'er', 'h -> e', 'pair-2'],
    ['er', 'enn', 'r -> n', 'pair-3'],
    ['enn', 'rfnn', 'e -> r', 'pair-4'],
  ];

  const nodes = [
    ['h', '0', 'node-h'],
    ['e', '1', 'node-e'],
    ['r', '1', 'node-r'],
    ['n', '1', 'node-n'],
    ['f', '1', 'node-f'],
  ];

  const playAnimation = () => {
    setIsPlaying(false);
    window.requestAnimationFrame(() => {
      setPlaybackKey((current) => current + 1);
      setIsPlaying(true);
    });
  };

  return (
    <section
      className={`topo-visual ${isPlaying ? 'is-playing' : ''}`}
      aria-label="Foreign Dictionary topological sorting visualization"
    >
      <div className="topo-visual-copy">
        <div>
          <p className="topo-kicker">Animated walkthrough</p>
          <h2>从相邻单词比较，到 Kahn 拓扑序</h2>
          <p>每一组相邻单词只看第一个不同字符；这个字符对就是一条有向边。边建完后，入度为 0 的字符先进入队列。</p>
        </div>
        <button className="topo-play-button" type="button" onClick={playAnimation}>
          {isPlaying ? 'Replay' : 'Play'}
        </button>
      </div>

      <div className="topo-stage" key={playbackKey}>
        <div className="topo-words" aria-label="Adjacent word comparisons">
          {comparisons.map(([first, second, edge, pairClass]) => (
            <span className={`topo-word-pair ${pairClass}`} key={edge}>
              <b>{first}</b>
              <b>{second}</b>
              <em>{edge}</em>
            </span>
          ))}
        </div>

        <div className="topo-graph-board" aria-label="Directed graph h to e to r to n to f">
          <div className="topo-chain">
            {nodes.map(([label, indegree, nodeClass], index) => (
              <Fragment key={label}>
                <span className={`topo-node ${nodeClass}`}>
                  {label}
                  <small>{indegree}</small>
                </span>
                {index < nodes.length - 1 && (
                  <span
                    className={`topo-edge-link ${['edge-he', 'edge-er', 'edge-rn', 'edge-nf'][index]}`}
                    aria-hidden="true"
                  />
                )}
              </Fragment>
            ))}
          </div>
        </div>

        <div className="topo-output" aria-label="Topological output order">
          {['h', 'e', 'r', 'n', 'f'].map((char, index) => (
            <span className={`out-${index + 1}`} key={char}>{char}</span>
          ))}
        </div>

        <ol className="topo-timeline">
          <li className="step-1">比较 <code>hrn</code> 和 <code>hrf</code>，第一个不同字符是 <code>n/f</code>，得到 <code>n -&gt; f</code>。</li>
          <li className="step-2">比较 <code>hrf</code> 和 <code>er</code>，第一个不同字符是 <code>h/e</code>，得到 <code>h -&gt; e</code>。</li>
          <li className="step-3">比较 <code>er</code> 和 <code>enn</code>，第一个不同字符是 <code>r/n</code>，得到 <code>r -&gt; n</code>。</li>
          <li className="step-4">比较 <code>enn</code> 和 <code>rfnn</code>，第一个不同字符是 <code>e/r</code>，得到 <code>e -&gt; r</code>。</li>
          <li className="step-5">Kahn 算法从入度为 0 的 <code>h</code> 开始，依次释放 <code>e</code>、<code>r</code>、<code>n</code>、<code>f</code>，输出 <code>hernf</code>。</li>
        </ol>
      </div>
    </section>
  );
}

function CheapestFlightsBellmanVisual() {
  const [activeRound, setActiveRound] = useState(0);
  const rounds = [
    {
      label: 'init',
      title: 'Round 0 / source only',
      prices: ['0', '∞', '∞', '∞'],
      activeEdges: [],
      note: 'Only src=0 is reachable before taking any flight.',
    },
    {
      label: '1 edge',
      title: 'Round 1 / at most 1 flight',
      prices: ['0', '100', '∞', '∞'],
      activeEdges: ['flight-0-1'],
      note: 'Use the previous prices array. Flight 0 -> 1 relaxes city 1 to 100.',
    },
    {
      label: '2 edges',
      title: 'Round 2 / at most 2 flights',
      prices: ['0', '100', '200', '700'],
      activeEdges: ['flight-1-2', 'flight-1-3'],
      note: 'Copy before relaxing, so 1 -> 2 and 1 -> 3 are allowed, but 2 -> 3 cannot chain inside this same round.',
    },
  ];
  const cities = [
    ['0', 'src', 'city-0'],
    ['1', '', 'city-1'],
    ['2', '', 'city-2'],
    ['3', 'dst', 'city-3'],
  ];
  const flights = [
    ['flight-0-1', '0 -> 1', '$100'],
    ['flight-1-2', '1 -> 2', '$100'],
    ['flight-2-0', '2 -> 0', '$100'],
    ['flight-1-3', '1 -> 3', '$600'],
    ['flight-2-3', '2 -> 3', '$200'],
  ];
  const round = rounds[activeRound];

  const nextRound = () => {
    setActiveRound((current) => Math.min(current + 1, rounds.length - 1));
  };

  const previousRound = () => {
    setActiveRound((current) => Math.max(current - 1, 0));
  };

  return (
    <section className="bf-visual" aria-label="Optimized Bellman-Ford visualization for Cheapest Flights Within K Stops">
      <div className="bf-header">
        <div>
          <p className="bf-kicker">Bellman-Ford with edge budget</p>
          <h2>Cheapest Flights Within K Stops</h2>
          <p>Example: <code>n=4</code>, <code>src=0</code>, <code>dst=3</code>, <code>k=1</code>. We may use at most <code>k + 1 = 2</code> flights.</p>
        </div>
        <div className="bf-controls" aria-label="Bellman-Ford round controls">
          <button type="button" onClick={previousRound} disabled={activeRound === 0} aria-label="Previous round">Prev</button>
          <span>{round.title}</span>
          <button type="button" onClick={nextRound} disabled={activeRound === rounds.length - 1} aria-label="Next round">Next</button>
        </div>
      </div>

      <div className="bf-stage">
        <div className="bf-round-tabs" role="tablist" aria-label="Bellman-Ford rounds">
          {rounds.map((candidate, index) => (
            <button
              key={candidate.label}
              className={index === activeRound ? 'active' : ''}
              type="button"
              onClick={() => setActiveRound(index)}
              role="tab"
              aria-selected={index === activeRound}
            >
              {candidate.label}
            </button>
          ))}
        </div>

        <div className="bf-layout">
          <div className="bf-graph" aria-label="Weighted directed flights">
            {cities.map(([id, tag, className]) => (
              <div className={`bf-city ${className}`} key={id}>
                <strong>{id}</strong>
                {tag && <small>{tag}</small>}
              </div>
            ))}
            {flights.map(([className, route, price]) => (
              <div
                className={`bf-flight ${className} ${round.activeEdges.includes(className) ? 'active' : ''}`}
                key={className}
              >
                <span>{route}</span>
                <em>{price}</em>
              </div>
            ))}
          </div>

          <div className="bf-prices" aria-label="Prices array">
            <div className="bf-prices-title">
              <span>prices</span>
              <small>from previous round only</small>
            </div>
            <div className="bf-price-grid">
              {round.prices.map((price, index) => (
                <div className={`bf-price ${price !== '∞' ? 'reachable' : ''}`} key={`${round.label}-${index}`}>
                  <span>city {index}</span>
                  <strong>{price}</strong>
                </div>
              ))}
            </div>
            <p>{round.note}</p>
          </div>
        </div>
      </div>
    </section>
  );
}

function SegmentTreeLISVisual() {
  const values = [2, 3, 5, 7, 9, 10, 18, 101];
  const steps = [
    { input: 10, rank: 5, query: '0..4', beforeBest: 0, current: 1, lis: 1, after: [0, 0, 0, 0, 0, 1, 0, 0] },
    { input: 9, rank: 4, query: '0..3', beforeBest: 0, current: 1, lis: 1, after: [0, 0, 0, 0, 1, 1, 0, 0] },
    { input: 2, rank: 0, query: 'empty', beforeBest: 0, current: 1, lis: 1, after: [1, 0, 0, 0, 1, 1, 0, 0] },
    { input: 5, rank: 2, query: '0..1', beforeBest: 1, current: 2, lis: 2, after: [1, 0, 2, 0, 1, 1, 0, 0] },
    { input: 3, rank: 1, query: '0..0', beforeBest: 1, current: 2, lis: 2, after: [1, 2, 2, 0, 1, 1, 0, 0] },
    { input: 7, rank: 3, query: '0..2', beforeBest: 2, current: 3, lis: 3, after: [1, 2, 2, 3, 1, 1, 0, 0] },
    { input: 101, rank: 7, query: '0..6', beforeBest: 3, current: 4, lis: 4, after: [1, 2, 2, 3, 1, 1, 0, 4] },
    { input: 18, rank: 6, query: '0..5', beforeBest: 3, current: 4, lis: 4, after: [1, 2, 2, 3, 1, 1, 4, 4] },
  ];
  const [activeStep, setActiveStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const step = steps[activeStep];

  useEffect(() => {
    if (!isPlaying) {
      return undefined;
    }

    const timer = window.setInterval(() => {
      setActiveStep((current) => (current + 1) % steps.length);
    }, 1900);

    return () => window.clearInterval(timer);
  }, [isPlaying, steps.length]);

  const previousStep = () => {
    setIsPlaying(false);
    setActiveStep((current) => (current === 0 ? steps.length - 1 : current - 1));
  };

  const nextStep = () => {
    setIsPlaying(false);
    setActiveStep((current) => (current + 1) % steps.length);
  };

  const treeLevels = buildSegmentTreeLevels(step.after);
  const smallerValues = values.slice(0, step.rank);
  const queryDescription = smallerValues.length > 0
    ? `rank 0..${step.rank - 1}，也就是值 ${smallerValues.join(', ')}`
    : '没有更小的压缩值';

  return (
    <section className="seg-visual" aria-label="Segment tree visualization for longest increasing subsequence">
      <div className="seg-header">
        <div>
          <p className="seg-kicker">Segment tree walkthrough</p>
          <h2>LIS: 先查更小值的最好结果，再更新当前值</h2>
          <p>
            例子输入 <code>[10, 9, 2, 5, 3, 7, 101, 18]</code>。坐标压缩后，
            每个叶子存 <strong>以这个值结尾的最长递增子序列长度</strong>。
          </p>
        </div>

        <div className="seg-controls" aria-label="Segment tree animation controls">
          <button type="button" onClick={previousStep} aria-label="Previous LIS step">Prev</button>
          <button type="button" onClick={() => setIsPlaying((current) => !current)} aria-label="Play segment tree animation">
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          <button type="button" onClick={nextStep} aria-label="Next LIS step">Next</button>
        </div>
      </div>

      <div className="seg-stage">
        <div className="seg-explainer">
          <div>
            <span>这一帧怎么看</span>
            <p>
              现在处理输入里的第 <strong>{activeStep + 1}</strong> 个数：<strong>{step.input}</strong>。
              因为 LIS 要严格递增，它只能接在比 {step.input} 更小的值后面。
            </p>
          </div>
          <ol>
            <li>蓝色叶子是本轮查询范围：<code>{queryDescription}</code>。</li>
            <li>线段树返回这些更小值里的最大 LIS 长度：<code>{step.beforeBest}</code>。</li>
            <li>当前数自己的长度就是 <code>{step.beforeBest} + 1 = {step.current}</code>，写到绿色叶子。</li>
          </ol>
        </div>

        <div className="seg-step-summary">
          <div>
            <span>current num</span>
            <strong>{step.input}</strong>
            <small>rank {step.rank}</small>
          </div>
          <div>
            <span>query</span>
            <strong>{step.query}</strong>
            <small>best smaller = {step.beforeBest}</small>
          </div>
          <div>
            <span>update</span>
            <strong>{step.current}</strong>
            <small>tree[{step.rank}] = {step.current}</small>
          </div>
          <div>
            <span>LIS so far</span>
            <strong>{step.lis}</strong>
            <small>global answer</small>
          </div>
        </div>

        <div className="seg-board-title">
          <span>压缩后的叶子</span>
          <small>叶子里的数字 = 以该值结尾的最佳 LIS 长度</small>
        </div>
        <div className="seg-rank-board" aria-label="Compressed value leaves">
          {values.map((value, index) => {
            const inQuery = step.rank > 0 && index < step.rank;
            const isUpdated = index === step.rank;

            return (
              <div
                className={`seg-leaf ${inQuery ? 'in-query' : ''} ${isUpdated ? 'updated' : ''}`}
                key={value}
              >
                <span>rank {index}</span>
                <strong>{value}</strong>
                <em>{step.after[index]}</em>
              </div>
            );
          })}
        </div>

        <div className="seg-board-title">
          <span>线段树缓存</span>
          <small>每个内部节点保存自己区间里的最大叶子值</small>
        </div>
        <div className="seg-tree-board" aria-label="Segment tree max values">
          {treeLevels.map((level, levelIndex) => (
            <div className="seg-tree-level" key={`level-${levelIndex}`}>
              {level.map((node) => {
                const intersectsQuery = step.rank > 0 && node.left < step.rank;
                const containsUpdate = node.left <= step.rank && step.rank <= node.right;

                return (
                  <div
                    className={`seg-tree-node ${intersectsQuery ? 'touches-query' : ''} ${containsUpdate ? 'update-path' : ''}`}
                    key={`${node.left}-${node.right}`}
                  >
                    <span>[{node.left}, {node.right}]</span>
                    <strong>{node.value}</strong>
                  </div>
                );
              })}
            </div>
          ))}
        </div>

        <div className="seg-board-title">
          <span>输入顺序</span>
          <small>点击任意一步，观察一个数如何改变整棵树</small>
        </div>
        <ol className="seg-timeline">
          {steps.map((candidate, index) => (
            <li className={index === activeStep ? 'active' : ''} key={`${candidate.input}-${index}`}>
              <button type="button" onClick={() => { setIsPlaying(false); setActiveStep(index); }}>
                <span>{index + 1}</span>
                <strong>{candidate.input}</strong>
                <em>LIS {candidate.lis}</em>
              </button>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}

function buildSegmentTreeLevels(leaves) {
  const levels = [
    leaves.map((value, index) => ({ left: index, right: index, value })),
  ];

  while (levels[0].length > 1) {
    const previous = levels[0];
    const next = [];
    for (let index = 0; index < previous.length; index += 2) {
      const left = previous[index];
      const right = previous[index + 1] ?? left;
      next.push({
        left: left.left,
        right: right.right,
        value: Math.max(left.value, right.value),
      });
    }
    levels.unshift(next);
  }

  return levels;
}

const INTERVAL_VISUALS = {
  'interval-merge-demo': {
    title: 'Merge Intervals',
    subtitle: 'Sort by start, then keep extending the current merged interval.',
    domain: [0, 18],
    intervals: [
      { id: 'a', label: '[1, 3]', start: 1, end: 3 },
      { id: 'b', label: '[2, 6]', start: 2, end: 6 },
      { id: 'c', label: '[8, 10]', start: 8, end: 10 },
      { id: 'd', label: '[15, 18]', start: 15, end: 18 },
    ],
    steps: [
      {
        title: 'Step 1 · sort by start',
        note: '先按 start 排序，保证只需要和当前 merged interval 比较。',
        active: ['a'],
        result: [{ id: 'm1', label: 'current [1, 3]', start: 1, end: 3 }],
        stats: [['current', '[1, 3]'], ['output', '[]']],
      },
      {
        title: 'Step 2 · overlap, extend end',
        note: '[2, 6] 的 start <= current end 3，所以合并成 [1, 6]。',
        active: ['a', 'b'],
        muted: ['a'],
        result: [{ id: 'm1', label: 'merged [1, 6]', start: 1, end: 6 }],
        stats: [['condition', '2 <= 3'], ['current', '[1, 6]']],
      },
      {
        title: 'Step 3 · gap, flush current',
        note: '[8, 10] 的 start > current end 6，说明前一段结束，输出 [1, 6]。',
        active: ['c'],
        result: [
          { id: 'm1', label: 'output [1, 6]', start: 1, end: 6 },
          { id: 'm2', label: 'current [8, 10]', start: 8, end: 10 },
        ],
        stats: [['condition', '8 > 6'], ['output', '[[1, 6]]']],
      },
      {
        title: 'Step 4 · finish',
        note: '最后没有重叠，依次输出剩余 current。',
        active: ['d'],
        result: [
          { id: 'm1', label: '[1, 6]', start: 1, end: 6 },
          { id: 'm2', label: '[8, 10]', start: 8, end: 10 },
          { id: 'm3', label: '[15, 18]', start: 15, end: 18 },
        ],
        stats: [['answer', '[[1,6],[8,10],[15,18]]']],
      },
    ],
  },
  'interval-insert-demo': {
    title: 'Insert Interval',
    subtitle: 'Three zones: before newInterval, overlapping block, after newInterval.',
    domain: [0, 17],
    intervals: [
      { id: 'a', label: '[1, 2]', start: 1, end: 2 },
      { id: 'b', label: '[3, 5]', start: 3, end: 5 },
      { id: 'c', label: '[6, 7]', start: 6, end: 7 },
      { id: 'd', label: '[8, 10]', start: 8, end: 10 },
      { id: 'e', label: '[12, 16]', start: 12, end: 16 },
      { id: 'new', label: 'new [4, 8]', start: 4, end: 8, kind: 'new' },
    ],
    steps: [
      {
        title: 'Step 1 · append before zone',
        note: '[1, 2] 完全在 newInterval 左边，直接进 output。',
        active: ['a', 'new'],
        result: [{ id: 'o1', label: 'output [1, 2]', start: 1, end: 2 }],
        stats: [['rule', 'end < new.start'], ['output', '[[1,2]]']],
      },
      {
        title: 'Step 2 · merge overlap block',
        note: '[3,5], [6,7], [8,10] 都和 [4,8] 有交集，持续扩张 newInterval。',
        active: ['b', 'c', 'd', 'new'],
        muted: ['b', 'c', 'd'],
        result: [
          { id: 'o1', label: 'output [1, 2]', start: 1, end: 2 },
          { id: 'm1', label: 'merged [3, 10]', start: 3, end: 10 },
        ],
        stats: [['merged start', 'min(4,3)=3'], ['merged end', 'max(8,10)=10']],
      },
      {
        title: 'Step 3 · append after zone',
        note: '[12,16] 完全在合并结果右边，先放入 [3,10]，再追加剩余区间。',
        active: ['e'],
        result: [
          { id: 'o1', label: '[1, 2]', start: 1, end: 2 },
          { id: 'm1', label: '[3, 10]', start: 3, end: 10 },
          { id: 'o2', label: '[12, 16]', start: 12, end: 16 },
        ],
        stats: [['answer', '[[1,2],[3,10],[12,16]]']],
      },
    ],
  },
  'interval-rooms-demo': {
    title: 'Meeting Rooms II',
    subtitle: 'Sweep starts and ends; the answer is max active meetings.',
    domain: [0, 30],
    intervals: [
      { id: 'a', label: '[0, 30]', start: 0, end: 30 },
      { id: 'b', label: '[5, 10]', start: 5, end: 10 },
      { id: 'c', label: '[15, 20]', start: 15, end: 20 },
    ],
    events: [
      { time: 0, label: '+1' },
      { time: 5, label: '+1' },
      { time: 10, label: '-1' },
      { time: 15, label: '+1' },
      { time: 20, label: '-1' },
      { time: 30, label: '-1' },
    ],
    steps: [
      {
        title: 't = 0 · first meeting starts',
        note: 'active 从 0 变成 1，需要 1 个房间。',
        active: ['a'],
        marker: 0,
        stats: [['active', '1'], ['max rooms', '1']],
      },
      {
        title: 't = 5 · overlap appears',
        note: '[5,10] 开始时 [0,30] 还没结束，active = 2。',
        active: ['a', 'b'],
        marker: 5,
        stats: [['active', '2'], ['max rooms', '2']],
      },
      {
        title: 't = 10 · one room freed',
        note: '[5,10] 结束，active 回到 1。',
        active: ['a'],
        marker: 10,
        stats: [['active', '1'], ['max rooms', '2']],
      },
      {
        title: 't = 15 · another overlap',
        note: '[15,20] 开始时 [0,30] 仍在进行，max rooms 仍然是 2。',
        active: ['a', 'c'],
        marker: 15,
        stats: [['active', '2'], ['answer', '2']],
      },
    ],
  },
  'interval-query-demo': {
    title: 'Minimum Interval to Include Each Query',
    subtitle: 'Sort queries; push candidate intervals into a min heap by length.',
    domain: [0, 7],
    intervals: [
      { id: 'a', label: '[1, 4] len 4', start: 1, end: 4 },
      { id: 'b', label: '[2, 4] len 3', start: 2, end: 4 },
      { id: 'c', label: '[3, 6] len 4', start: 3, end: 6 },
      { id: 'd', label: '[4, 4] len 1', start: 4, end: 4 },
    ],
    queries: [2, 3, 4, 5],
    steps: [
      {
        title: 'query = 2',
        note: '加入 start <= 2 的区间：[1,4], [2,4]。最短覆盖区间是 [2,4]，长度 3。',
        active: ['a', 'b'],
        marker: 2,
        result: [{ id: 'best', label: 'best [2,4]', start: 2, end: 4 }],
        stats: [['heap top', 'len 3 [2,4]'], ['ans[2]', '3']],
      },
      {
        title: 'query = 3',
        note: '加入 [3,6]。heap 顶仍是 [2,4]，它覆盖 3。',
        active: ['b', 'c'],
        marker: 3,
        result: [{ id: 'best', label: 'best [2,4]', start: 2, end: 4 }],
        stats: [['heap top', 'len 3 [2,4]'], ['ans[3]', '3']],
      },
      {
        title: 'query = 4',
        note: '加入 [4,4]，长度 1，立刻成为最优答案。',
        active: ['a', 'b', 'c', 'd'],
        marker: 4,
        result: [{ id: 'best', label: 'best [4,4]', start: 4, end: 4 }],
        stats: [['heap top', 'len 1 [4,4]'], ['ans[4]', '1']],
      },
      {
        title: 'query = 5',
        note: '弹掉 end < 5 的区间，剩下 [3,6] 覆盖 5，长度 4。',
        active: ['c'],
        marker: 5,
        result: [{ id: 'best', label: 'best [3,6]', start: 3, end: 6 }],
        stats: [['removed', 'end < 5'], ['ans[5]', '4']],
      },
    ],
  },
};

function IntervalPatternVisual({ kind }) {
  const visual = INTERVAL_VISUALS[kind];
  const [activeStep, setActiveStep] = useState(0);
  const step = visual.steps[activeStep];
  const active = new Set(step.active ?? []);
  const muted = new Set(step.muted ?? []);
  const domain = visual.domain;
  const ticks = buildIntervalTicks(domain);

  return (
    <section className="interval-visual">
      <header className="interval-visual-header">
        <div>
          <p className="eyebrow">Interval visual</p>
          <h2>{visual.title}</h2>
          <p>{visual.subtitle}</p>
        </div>
        <div className="interval-step-counter">
          {activeStep + 1}<span>/ {visual.steps.length}</span>
        </div>
      </header>

      <div className="interval-step-note">
        <strong>{step.title}</strong>
        <span>{step.note}</span>
      </div>

      <div className="interval-axis" aria-label={`${visual.title} timeline`}>
        <div className="interval-axis-line">
          {ticks.map((tick) => (
            <span
              className="interval-tick"
              key={tick}
              style={{ left: `${intervalPercent(tick, domain)}%` }}
            >
              {tick}
            </span>
          ))}
          {step.marker !== undefined && (
            <span
              className="interval-marker"
              style={{ left: `${intervalPercent(step.marker, domain)}%` }}
            >
              q={step.marker}
            </span>
          )}
        </div>

        <div className="interval-lanes">
          {visual.intervals.map((interval) => (
            <IntervalBar
              domain={domain}
              interval={interval}
              isActive={active.has(interval.id)}
              isMuted={muted.has(interval.id)}
              key={interval.id}
            />
          ))}
        </div>

        {visual.queries && (
          <div className="interval-query-row">
            {visual.queries.map((query) => (
              <span
                className={query === step.marker ? 'active' : ''}
                key={query}
                style={{ left: `${intervalPercent(query, domain)}%` }}
              >
                {query}
              </span>
            ))}
          </div>
        )}

        {step.result && (
          <div className="interval-result-lanes">
            {step.result.map((interval) => (
              <IntervalBar
                domain={domain}
                interval={{ ...interval, kind: 'result' }}
                isActive
                key={interval.id}
              />
            ))}
          </div>
        )}

        {visual.events && (
          <div className="interval-events">
            {visual.events.map((event) => (
              <span
                className={event.time === step.marker ? 'active' : ''}
                key={`${event.time}-${event.label}`}
                style={{ left: `${intervalPercent(event.time, domain)}%` }}
              >
                {event.label}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="interval-stat-grid">
        {(step.stats ?? []).map(([label, value]) => (
          <div key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>

      <ol className="interval-step-list">
        {visual.steps.map((candidate, index) => (
          <li className={index === activeStep ? 'active' : ''} key={candidate.title}>
            <button type="button" onClick={() => setActiveStep(index)}>
              <span>{index + 1}</span>
              {candidate.title.replace(/^Step \d+ · /, '')}
            </button>
          </li>
        ))}
      </ol>
    </section>
  );
}

const POW_STEPS = [
  {
    title: 'Init',
    power: 10,
    base: 2,
    res: 1,
    bit: 0,
    action: 'power = 10, binary = 1010. Lowest bit is 0, so this round does not contribute to the answer yet.',
    next: 'Square base to 4, shift power right to 5.',
  },
  {
    title: 'Read bit 1',
    power: 5,
    base: 4,
    res: 1,
    bit: 1,
    action: 'power is odd. The current base represents x^2, so multiply it into res.',
    next: 'res = 1 * 4 = 4. Square base to 16, shift power right to 2.',
  },
  {
    title: 'Read bit 0',
    power: 2,
    base: 16,
    res: 4,
    bit: 0,
    action: 'Lowest bit is 0. x^4 is not needed for n = 10, so res stays unchanged.',
    next: 'Square base to 256, shift power right to 1.',
  },
  {
    title: 'Read bit 1',
    power: 1,
    base: 256,
    res: 4,
    bit: 1,
    action: 'power is odd again. The current base represents x^8, and n = 10 includes this bit.',
    next: 'res = 4 * 256 = 1024. Shift power to 0, stop.',
  },
  {
    title: 'Done',
    power: 0,
    base: 65536,
    res: 1024,
    bit: null,
    action: 'All bits have been consumed from right to left: 10 = 8 + 2.',
    next: 'Return 1024 for pow(2, 10).',
  },
];

function BinaryPowVisual() {
  const [activeStep, setActiveStep] = useState(0);
  const step = POW_STEPS[activeStep];
  const binaryBits = ['1', '0', '1', '0'];
  const consumedFromRight = Math.min(activeStep, binaryBits.length);

  return (
    <section className="pow-visual" aria-label="Binary exponentiation walkthrough">
      <header className="pow-header">
        <div>
          <p className="eyebrow">Math visual</p>
          <h2>Binary Exponentiation: pow(2, 10)</h2>
          <p>每一轮只看 `power` 的最低位：bit 为 1 才把当前 `base` 乘进 `res`。</p>
        </div>
        <div className="pow-counter">{activeStep + 1}<span>/ {POW_STEPS.length}</span></div>
      </header>

      <div className="pow-board">
        <div className="pow-bits" aria-label="Binary bits of exponent 10">
          {binaryBits.map((bit, index) => {
            const fromRight = binaryBits.length - 1 - index;
            const isCurrent = fromRight === consumedFromRight && activeStep < binaryBits.length;
            const isConsumed = fromRight < consumedFromRight;
            return (
              <span
                className={`${isCurrent ? 'current' : ''} ${isConsumed ? 'consumed' : ''}`}
                key={`${bit}-${index}`}
              >
                {bit}
                <small>{[8, 4, 2, 1][index]}</small>
              </span>
            );
          })}
        </div>

        <div className="pow-state-grid">
          <div>
            <span>x / base</span>
            <strong>{step.base}</strong>
          </div>
          <div>
            <span>power</span>
            <strong>{step.power}</strong>
          </div>
          <div>
            <span>power & 1</span>
            <strong>{step.bit === null ? '-' : step.bit}</strong>
          </div>
          <div>
            <span>res</span>
            <strong>{step.res}</strong>
          </div>
        </div>

        <div className="pow-explain">
          <strong>{step.title}</strong>
          <p>{step.action}</p>
          <p>{step.next}</p>
        </div>
      </div>

      <ol className="pow-timeline">
        {POW_STEPS.map((candidate, index) => (
          <li className={index === activeStep ? 'active' : ''} key={candidate.title}>
            <button type="button" onClick={() => setActiveStep(index)}>
              <span>{index + 1}</span>
              {candidate.title}
            </button>
          </li>
        ))}
      </ol>
    </section>
  );
}

function IntervalBar({ domain, interval, isActive = false, isMuted = false }) {
  const left = intervalPercent(interval.start, domain);
  const right = intervalPercent(interval.end, domain);
  const width = Math.max(right - left, 1.4);

  return (
    <div
      className={`interval-bar ${interval.kind ?? ''} ${isActive ? 'active' : ''} ${isMuted ? 'muted' : ''}`}
      style={{ left: `${left}%`, width: `${width}%` }}
    >
      <span>{interval.label}</span>
    </div>
  );
}

function intervalPercent(value, [min, max]) {
  if (max === min) {
    return 0;
  }

  return ((value - min) / (max - min)) * 100;
}

function buildIntervalTicks([min, max]) {
  const width = max - min;
  const step = width <= 8 ? 1 : Math.ceil(width / 6);
  const ticks = [];
  for (let value = min; value <= max; value += step) {
    ticks.push(value);
  }
  if (ticks[ticks.length - 1] !== max) {
    ticks.push(max);
  }
  return ticks;
}

function MarkdownPre({ children, ...props }) {
  const child = Array.isArray(children) ? children[0] : children;
  const className = child?.props?.className ?? '';
  const match = /language-(quiz|mcq|topo-demo|bellman-demo|segment-tree-demo|interval-merge-demo|interval-insert-demo|interval-rooms-demo|interval-query-demo|pow-demo)/.exec(className);

  if (match?.[1] === 'topo-demo') {
    return <ForeignDictionaryTopoVisual />;
  }

  if (match?.[1] === 'bellman-demo') {
    return <CheapestFlightsBellmanVisual />;
  }

  if (match?.[1] === 'segment-tree-demo') {
    return <SegmentTreeLISVisual />;
  }

  if (match?.[1]?.startsWith('interval-')) {
    return <IntervalPatternVisual kind={match[1]} />;
  }

  if (match?.[1] === 'pow-demo') {
    return <BinaryPowVisual />;
  }

  if (match) {
    return <QuizBlock source={extractPlainText(child.props.children).replace(/\n$/, '')} />;
  }

  return <CodeBlock className={className} source={extractPlainText(child?.props?.children)} {...props} />;
}

function CodeBlock({ className = '', source = '' }) {
  const [copied, setCopied] = useState(false);
  const language = className.match(/language-([\w-]+)/)?.[1] ?? 'text';
  const label = formatCodeLanguage(language);
  const code = source.replace(/\n$/, '');

  const copyCode = async () => {
    if (!navigator?.clipboard) {
      return;
    }

    await navigator.clipboard.writeText(code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  };

  return (
    <figure className="code-frame">
      <figcaption className="code-frame-header">
        <span>{label}</span>
        <button type="button" onClick={copyCode}>
          {copied ? 'Copied' : 'Copy'}
        </button>
      </figcaption>
      <pre>
        <code className={className}>
          <HighlightedCode code={code} language={language} />
        </code>
      </pre>
    </figure>
  );
}

function HighlightedCode({ code, language }) {
  const tokens = tokenizeCode(code, language);
  return tokens.map((token, index) => (
    token.type === 'text'
      ? <Fragment key={index}>{token.value}</Fragment>
      : <span className={`code-token ${token.type}`} key={index}>{token.value}</span>
  ));
}

function formatCodeLanguage(language) {
  const labels = {
    js: 'JavaScript',
    jsx: 'React JSX',
    py: 'Python',
    python: 'Python',
    text: 'Text',
  };

  return labels[language] ?? language.toUpperCase();
}

function tokenizeCode(code, language) {
  if (!['py', 'python', 'js', 'jsx'].includes(language)) {
    return [{ type: 'text', value: code }];
  }

  const keywordPattern = language === 'python'
    || language === 'py'
    ? 'False|None|True|and|as|break|class|continue|def|elif|else|for|from|if|import|in|is|not|or|return|while|with'
    : 'const|let|var|function|return|if|else|for|while|import|from|export|class|new|true|false|null|undefined|await|async';
  const builtinPattern = language === 'python' || language === 'py'
    ? 'Counter|List|abs|bool|dict|enumerate|float|heapify|heappop|int|len|list|max|min|range|set|sorted|sum'
    : 'Array|Boolean|Map|Math|Number|Object|Promise|Set|String|console';
  const tokenPattern = new RegExp(
    `(#.*|//.*|"""[\\s\\S]*?"""|'''[\\s\\S]*?'''|"(?:\\\\.|[^"\\\\])*"|'(?:\\\\.|[^'\\\\])*'|\\b(?:${keywordPattern})\\b|\\b(?:${builtinPattern})\\b|\\b\\d+(?:\\.\\d+)?\\b)`,
    'g',
  );

  const tokens = [];
  let cursor = 0;
  for (const match of code.matchAll(tokenPattern)) {
    if (match.index > cursor) {
      tokens.push({ type: 'text', value: code.slice(cursor, match.index) });
    }

    const value = match[0];
    let type = 'number';
    if (value.startsWith('#') || value.startsWith('//')) {
      type = 'comment';
    } else if (value.startsWith('"') || value.startsWith("'")) {
      type = 'string';
    } else if (new RegExp(`^(?:${keywordPattern})$`).test(value)) {
      type = 'keyword';
    } else if (new RegExp(`^(?:${builtinPattern})$`).test(value)) {
      type = 'builtin';
    }

    tokens.push({ type, value });
    cursor = match.index + value.length;
  }

  if (cursor < code.length) {
    tokens.push({ type: 'text', value: code.slice(cursor) });
  }

  return tokens;
}

function parseHashRoute(rawHash) {
  const hashValue = decodeURIComponent(String(rawHash ?? '').replace(/^#/, '')).replace(/^\/+/, '');

  if (!hashValue || hashValue === 'home') {
    return { view: 'home', noteId: null, sectionId: null };
  }

  const noteMatch = tutorials.find((tutorial) => tutorial.id === hashValue);
  if (noteMatch) {
    return { view: 'reader', noteId: noteMatch.id, sectionId: noteMatch.sectionId };
  }

  const sectionMatch = noteSections.find((section) => section.id === hashValue);
  if (sectionMatch) {
    return { view: 'reader', noteId: sectionMatch.notes[0]?.id ?? null, sectionId: sectionMatch.id };
  }

  return null;
}

function normalizeObsidianMarkdown(markdownText) {
  if (!markdownText) {
    return '';
  }

  let normalized = markdownText;

  normalized = normalized.replace(/%%[\s\S]*?%%/g, '');

  normalized = normalized.replace(/^>\s*\[!([^\]\n+-]+)(?:[+-])?\](.*)$/gim, (_, type, rawTitle) => {
    const label = type.trim();
    const title = rawTitle.trim().replace(/^[-:\s]+/, '');
    const heading = title || (label.charAt(0).toUpperCase() + label.slice(1).toLowerCase());
    return `> **${heading}:**`;
  });

  normalized = normalized.replace(/!\[\[([^\]\n]+)\]\]/g, (_, body) => {
    const { target, alias } = splitObsidianTarget(body);
    if (!target) {
      return '';
    }

    const mediaUrl = resolveMediaUrl(target);
    if (mediaUrl) {
      return `![${alias || prettyLabel(target)}](${mediaUrl})`;
    }

    const noteId = resolveNoteId(target);
    if (noteId) {
      return `[Embedded note: ${alias || prettyLabel(target)}](#${encodeURIComponent(noteId)})`;
    }

    return `*Embedded asset not found: ${alias || prettyLabel(target)}*`;
  });

  normalized = normalized.replace(/\[\[([^\]\n]+)\]\]/g, (_, body) => {
    const { target, alias } = splitObsidianTarget(body);
    if (!target) {
      return '';
    }

    const resolvedLink = resolveObsidianLink(target, alias);
    if (resolvedLink) {
      return resolvedLink;
    }

    if (/^https?:\/\//i.test(target)) {
      return `[${alias || target}](${target})`;
    }

    return alias || prettyLabel(target);
  });

  normalized = normalized.replace(/==([^=\n][^=\n]*?)==/g, '<mark>$1</mark>');

  return normalized;
}

function App() {
  const initialRoute = parseHashRoute(window.location.hash) ?? { view: 'home', noteId: null, sectionId: null };
  const initialId = initialRoute.noteId ?? tutorials[0]?.id ?? '';

  const [currentView, setCurrentView] = useState(initialRoute.view);
  const [selectedTutorialId, setSelectedTutorialId] = useState(initialId);
  const [language, setLanguage] = useState('zh');
  const [query, setQuery] = useState('');
  const [contentByKey, setContentByKey] = useState({});
  const [errorByKey, setErrorByKey] = useState({});
  const inFlightRef = useRef(new Set());
  const selectedSection = noteSections.find((section) =>
    section.notes.some((note) => note.id === selectedTutorialId),
  ) ?? noteSections[0];
  const activeSectionNotes = tutorials.filter((tutorial) => tutorial.sectionId === selectedSection?.id);

  const filteredTutorials = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    if (!normalizedQuery) {
      return activeSectionNotes;
    }

    return activeSectionNotes.filter((tutorial) =>
      [tutorial.title, tutorial.fileName].some((field) => field.toLowerCase().includes(normalizedQuery)),
    );
  }, [activeSectionNotes, query]);

  const selectedTutorial =
    tutorials.find((tutorial) => tutorial.id === selectedTutorialId) ?? filteredTutorials[0] ?? tutorials[0] ?? null;

  const activeLanguage =
    selectedTutorial?.variants[language]?.url ? language : 'zh';
  const selectedVariant = selectedTutorial?.variants[activeLanguage] ?? null;
  const contentKey =
    selectedTutorial && selectedVariant?.url ? `${selectedTutorial.id}:${activeLanguage}` : '';

  useEffect(() => {
    if (!selectedVariant?.url || !contentKey) {
      return;
    }

    const isLoaded = Object.prototype.hasOwnProperty.call(contentByKey, contentKey);
    if (isLoaded || errorByKey[contentKey] || inFlightRef.current.has(contentKey)) {
      return;
    }

    inFlightRef.current.add(contentKey);

    fetch(selectedVariant.url)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Unable to load markdown (${response.status})`);
        }
        return response.text();
      })
      .then((content) => {
        setContentByKey((prev) => ({
          ...prev,
          [contentKey]: content,
        }));
      })
      .catch((error) => {
        setErrorByKey((prev) => ({
          ...prev,
          [contentKey]: error.message,
        }));
      })
      .finally(() => {
        inFlightRef.current.delete(contentKey);
      });
  }, [contentByKey, contentKey, errorByKey, selectedVariant]);

  const hasSelectedContent = contentKey
    ? Object.prototype.hasOwnProperty.call(contentByKey, contentKey)
    : false;
  const selectedContent = hasSelectedContent ? contentByKey[contentKey] : '';
  const selectedError = contentKey ? errorByKey[contentKey] : '';
  const selectedIsLoading = Boolean(selectedTutorial && selectedVariant?.url && !hasSelectedContent && !selectedError);

  const normalizedSelectedContent = useMemo(
    () => normalizeObsidianMarkdown(selectedContent),
    [selectedContent],
  );
  const sectionHeadings = useMemo(
    () => extractMarkdownHeadings(normalizedSelectedContent).filter((heading) => heading.level <= 3),
    [normalizedSelectedContent],
  );

  const scrollToHeading = (headingId) => {
    const target = document.getElementById(headingId);
    target?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const navigateHome = () => {
    setCurrentView('home');
    setQuery('');
    if (window.location.hash) {
      window.history.replaceState(null, '', window.location.pathname + window.location.search);
    }
  };

  const navigateToSection = (sectionId) => {
    const section = noteSections.find((candidate) => candidate.id === sectionId);
    const nextId = section?.notes[0]?.id ?? tutorials[0]?.id ?? '';
    setCurrentView('reader');
    setQuery('');
    setSelectedTutorialId(nextId);
  };

  const navigateToTutorial = (tutorialId) => {
    setCurrentView('reader');
    setSelectedTutorialId(tutorialId);
  };

  useEffect(() => {
    if (currentView !== 'reader' || !selectedTutorial) {
      return;
    }

    const encoded = `#${encodeURIComponent(selectedTutorial.id)}`;
    if (window.location.hash !== encoded) {
      window.history.replaceState(null, '', encoded);
    }
  }, [currentView, selectedTutorial]);

  useEffect(() => {
    const handleHashChange = () => {
      const route = parseHashRoute(window.location.hash);
      if (!route) {
        return;
      }

      setCurrentView(route.view);
      if (route.noteId) {
        setSelectedTutorialId(route.noteId);
      }
    };

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  return (
    <div className={`site-shell ${currentView === 'home' ? 'home-view' : 'reader-view'}`}>
      <header className="top-nav">
        <button className="brand-lockup" type="button" onClick={navigateHome}>
          <span className="brand-mark">IN</span>
          <span>
            <strong>Interview Notes</strong>
            <small>systems · infra · practice</small>
          </span>
        </button>

        <nav className="top-nav-links" aria-label="Main navigation">
          <button
            className={`top-nav-link ${currentView === 'home' ? 'active' : ''}`}
            type="button"
            onClick={navigateHome}
          >
            Home
          </button>
          {noteSections.map((section) => (
            <button
              key={section.id}
              className={`top-nav-link ${currentView === 'reader' && selectedSection?.id === section.id ? 'active' : ''}`}
              type="button"
              onClick={() => navigateToSection(section.id)}
            >
              {section.title}
            </button>
          ))}
        </nav>
      </header>

      {currentView === 'home' ? (
        <main className="home-page">
          <section className="home-hero">
            <div className="home-hero-copy">
              <p className="eyebrow">Interview Notes</p>
              <h1>面试复习的系统化笔记库</h1>
              <p>
                用一个入口组织机器学习系统、推理基础设施、CUDA kernel 和 post-training infra。
                每个板块都可以沉淀笔记、双语内容和可交互练习题。
              </p>
              <div className="home-actions">
                <button className="primary-action" type="button" onClick={() => navigateToSection('mlsys')}>
                  Start MLSYS
                </button>
                <button className="secondary-action" type="button" onClick={() => navigateToTutorial('MLSYS1.md')}>
                  Try Practice
                </button>
              </div>
            </div>

            <div className="home-hero-panel" aria-label="Site summary">
              {homeStats.map((stat) => (
                <div className="home-stat" key={stat.label}>
                  <strong>{stat.value}</strong>
                  <span>{stat.label}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="home-sections" aria-label="Interview note sections">
            <div className="section-heading">
              <p className="eyebrow">Sections</p>
              <h2>当前板块</h2>
            </div>

            <div className="section-card-grid">
              {noteSections.map((section) => (
                <button
                  key={section.id}
                  className="home-section-card"
                  type="button"
                  onClick={() => navigateToSection(section.id)}
                >
                  <span className="section-card-kicker">{section.notes.length} notes</span>
                  <strong>{section.title}</strong>
                  <span>{section.description}</span>
                </button>
              ))}
            </div>
          </section>
        </main>
      ) : (
        <div className="app-shell">
      <aside className="notes-panel">
        <header className="panel-header">
          <p className="eyebrow">Interview Notes</p>
          <h1>Interview Notebook</h1>
          <p className="panel-meta">{noteSections.length} section · {tutorials.length} notes</p>
        </header>

        <nav className="section-tabs" aria-label="Interview note sections">
          {noteSections.map((section) => (
            <button
              key={section.id}
              className={`section-tab ${selectedSection?.id === section.id ? 'active' : ''}`}
              type="button"
              onClick={() => navigateToSection(section.id)}
            >
              <span>{section.title}</span>
              <small>{section.notes.length}</small>
            </button>
          ))}
        </nav>

        <label className="search">
          <span>Search {selectedSection?.title ?? 'Notes'}</span>
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Type note title or filename"
          />
        </label>

        <div className="note-list">
          {filteredTutorials.map((tutorial) => (
            <button
              key={tutorial.id}
              className={`note-button ${selectedTutorial?.id === tutorial.id ? 'active' : ''}`}
              onClick={() => navigateToTutorial(tutorial.id)}
              type="button"
            >
              <span className="note-title">{tutorial.title}</span>
              <span className="note-subtitle">{tutorial.fileName}</span>
            </button>
          ))}
          {filteredTutorials.length === 0 && (
            <p className="list-empty">No notes matched your search.</p>
          )}
        </div>
      </aside>

      <main className="reader-panel">
        {selectedTutorial ? (
          <>
            <header className="reader-header">
              <div className="reader-header-top">
                <div>
                  <p className="reader-label">{selectedTutorial.sectionTitle} / Interview Notes</p>
                  <h2>{selectedTutorial.title}</h2>
                  <p>{selectedVariant?.fileName ?? selectedTutorial.fileName}</p>
                </div>

                <div className="reader-controls">
                  <div className="language-toggle" aria-label="Language selector" role="group">
                    {languageOptions.map((option) => (
                      <button
                        key={option.id}
                        className={`language-button ${activeLanguage === option.id ? 'active' : ''}`}
                        onClick={() => setLanguage(option.id)}
                        type="button"
                        aria-pressed={activeLanguage === option.id}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </header>

            <div className="reader-content-grid">
              <article className="markdown-body">
                {selectedError && <p className="empty-note">Load failed: {selectedError}</p>}
                {selectedIsLoading && !selectedError && <p className="empty-note">Loading markdown...</p>}
                {!selectedIsLoading && !selectedError && normalizedSelectedContent?.trim() && (
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm, remarkMath]}
                    rehypePlugins={[rehypeRaw, rehypeKatex]}
                    components={{
                      a: ({ href, children, ...props }) => {
                        const external = href?.startsWith('http');
                        return (
                          <a
                            href={href}
                            target={external ? '_blank' : undefined}
                            rel={external ? 'noreferrer' : undefined}
                            {...props}
                          >
                            {children}
                          </a>
                        );
                      },
                      h1: ({ children }) => <HeadingWithAnchor level={1}>{children}</HeadingWithAnchor>,
                      h2: ({ children }) => <HeadingWithAnchor level={2}>{children}</HeadingWithAnchor>,
                      h3: ({ children }) => <HeadingWithAnchor level={3}>{children}</HeadingWithAnchor>,
                      h4: ({ children }) => <HeadingWithAnchor level={4}>{children}</HeadingWithAnchor>,
                      pre: MarkdownPre,
                      code: ({ className, children, ...props }) => (
                        <code className={className} {...props}>
                          {children}
                        </code>
                      ),
                    }}
                  >
                    {normalizedSelectedContent}
                  </ReactMarkdown>
                )}
                {!selectedIsLoading && !selectedError && selectedContent !== undefined && !selectedContent.trim() && (
                  <p className="empty-note">This file is empty and ready for future notes.</p>
                )}
              </article>

              {sectionHeadings.length > 0 && (
                <aside className="section-toc" aria-label="Section navigation">
                  <div className="section-toc-inner">
                    <div className="section-toc-heading">
                      <p className="eyebrow">Sections</p>
                      <span>{sectionHeadings.length}</span>
                    </div>
                    <nav>
                      {sectionHeadings.map((heading, index) => (
                        <a
                          className={`toc-link level-${heading.level}`}
                          href={`#${heading.id}`}
                          key={`${heading.id}-${index}`}
                          onClick={(event) => {
                            event.preventDefault();
                            scrollToHeading(heading.id);
                          }}
                        >
                          {heading.text}
                        </a>
                      ))}
                    </nav>
                  </div>
                </aside>
              )}
            </div>
          </>
        ) : (
          <section className="reader-empty">
            <h2>No published Markdown files found</h2>
            <p>Add ready notes to an interview section and refresh.</p>
          </section>
        )}
      </main>
    </div>
      )}
    </div>
  );
}

export default App;
