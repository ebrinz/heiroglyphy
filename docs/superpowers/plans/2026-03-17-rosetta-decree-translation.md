# Rosetta Decree Geometric Translation — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a LaTeX document presenting the Decree of Memphis (Rosetta Stone) translated through vector-space alignment, with Tufte-style margin notes showing per-glyph contextual swapping, Demotic/Greek cross-references, and embedding metadata.

**Architecture:** Single monolithic XeLaTeX file at `docs/paper/rosetta_decree.tex` with custom commands for interlinear rows (`\decreerow`, `\reconstructedrow`) and margin annotations (`\glyphnote`). Asymmetric page layout with wide outer margins for dense scholarly notes. Shares the existing EgyptianHiero font setup from the main paper.

**Tech Stack:** XeLaTeX, `marginnote`, `geometry`, `xcolor`, `fontspec`, `booktabs`, `hyperref`

---

### Task 1: Document Skeleton and Page Geometry

**Files:**
- Create: `docs/paper/rosetta_decree.tex`

- [ ] **Step 1: Create the document with preamble and asymmetric layout**

```latex
\documentclass[11pt, a4paper]{article}

% ── Packages ──────────────────────────────────────────────────────────────────
\usepackage{fontspec}
\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage[
    inner=1in,
    outer=3.5in,
    top=1in,
    bottom=1in,
    marginparwidth=2.8in,
    marginparsep=0.2in
]{geometry}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{float}
\usepackage{marginnote}
\usepackage{ragged2e}

\hypersetup{
    colorlinks=true,
    linkcolor=blue!60!black,
    citecolor=blue!60!black,
    urlcolor=blue!60!black,
}

% ── Hieroglyphic font ─────────────────────────────────────────────────────────
\newfontfamily\hierofont{EgyptianHiero}[
    Path=../../final_output/,
    Extension=.ttf,
]
\newcommand{\hiero}[1]{{\normalfont\hierofont #1}}

% ── Colors ────────────────────────────────────────────────────────────────────
\definecolor{rosetta}{RGB}{183, 110, 72}       % terracotta for geometric translations
\definecolor{reconstructed}{gray}{0.55}         % gray for reconstructed lines

% ── Title ─────────────────────────────────────────────────────────────────────
\title{
    \textbf{The Geometry of the Decree}\\[6pt]
    {\large An Alternate Translation of the Rosetta Stone\\Through Semantic Alignment}
}

\author{
    Erik Brinsmead \\
    Independent Researcher \\[6pt]
    Claude \\
    Anthropic \\[12pt]
    \texttt{github.com/ebrinz/heiroglyphy}
}

\date{March 2026}

\begin{document}
\maketitle

% placeholder sections
\section{Introduction}
\label{sec:intro}

\section{The Decree: Surviving Lines}
\label{sec:surviving}

\section{The Decree: Reconstructed Lines}
\label{sec:reconstructed}

\section{Commentary}
\label{sec:commentary}

\appendix
\section{Glyph Index}
\label{sec:glyph-index}

\end{document}
```

- [ ] **Step 2: Compile to verify the skeleton builds**

Run: `cd /Users/crashy/Development/heiroglyphy/docs/paper && xelatex rosetta_decree.tex`
Expected: Successful compilation, PDF with title page and empty sections.

- [ ] **Step 3: Commit**

```bash
git add docs/paper/rosetta_decree.tex
git commit -m "feat: add rosetta decree document skeleton with page geometry"
```

---

### Task 2: Custom Commands for Interlinear Rows

**Files:**
- Modify: `docs/paper/rosetta_decree.tex` (preamble, after color definitions)

- [ ] **Step 1: Add the \decreerow command**

Insert after the color definitions in the preamble:

```latex
% ── Decree row commands ───────────────────────────────────────────────────────
% \decreerow{line_number}{hieroglyphs}{transliteration}{standard_translation}{geometric_translation}
\newcounter{decreeline}
\newcommand{\decreerow}[5]{%
    \stepcounter{decreeline}%
    \noindent\begin{minipage}[t]{\textwidth}
    \vspace{6pt}
    {\footnotesize\textsc{Line #1}}\par\vspace{2pt}
    {\Large\hiero{#2}}\par\vspace{2pt}
    {\itshape #3}\par\vspace{1pt}
    {#4}\par\vspace{1pt}
    {\bfseries\color{rosetta}#5}\par
    \vspace{4pt}
    \noindent\rule{\textwidth}{0.2pt}
    \end{minipage}\par\vspace{8pt}
}

% \reconstructedrow — same structure, gray + brackets for lacunae
\newcommand{\reconstructedrow}[5]{%
    \stepcounter{decreeline}%
    \noindent\begin{minipage}[t]{\textwidth}
    \vspace{6pt}
    {\footnotesize\textsc{Line #1} \textcolor{reconstructed}{[reconstructed]}}\par\vspace{2pt}
    {\Large\color{reconstructed}\hiero{⟦#2⟧}}\par\vspace{2pt}
    {\itshape\color{reconstructed} #3}\par\vspace{1pt}
    {\color{reconstructed}#4}\par\vspace{1pt}
    {\bfseries\color{rosetta!60}#5}\par
    \vspace{4pt}
    \noindent\rule{\textwidth}{0.2pt}
    \end{minipage}\par\vspace{8pt}
}
```

- [ ] **Step 2: Add a test row in the surviving lines section to verify rendering**

Replace the placeholder `\section{The Decree: Surviving Lines}` content with:

```latex
\section{The Decree: Surviving Lines}
\label{sec:surviving}

\decreerow{1}%
    {𓇋𓅱𓂝𓏏𓆑𓂋𓈖𓏏𓁹𓅓}%
    {iw=f r ntt iri m \d{h}w,t-n\d{t}r}%
    {He shall do what is done in the temples}%
    {He moves toward what is performed within the god-houses}
```

- [ ] **Step 3: Compile to verify the row renders correctly**

Run: `cd /Users/crashy/Development/heiroglyphy/docs/paper && xelatex rosetta_decree.tex`
Expected: PDF shows a stacked row: hieroglyphs, transliteration (italic), standard translation, geometric translation (bold terracotta), with a thin rule below.

- [ ] **Step 4: Commit**

```bash
git add docs/paper/rosetta_decree.tex
git commit -m "feat: add decreerow and reconstructedrow commands with test content"
```

---

### Task 3: Margin Note Command

**Files:**
- Modify: `docs/paper/rosetta_decree.tex` (preamble, after decree row commands)

- [ ] **Step 1: Add the \glyphnote command**

Insert after the decree row commands:

```latex
% ── Margin note command ───────────────────────────────────────────────────────
% \glyphnote{glyph}{gardiner_code}{transliteration}{note_body}
\newcommand{\glyphnote}[4]{%
    \marginnote{%
        \RaggedRight\footnotesize
        {\normalsize\hiero{#1}}~{\texttt{#2}}~{\itshape #3}\par
        \vspace{2pt}
        #4
        \par\vspace{6pt}
    }%
}
```

- [ ] **Step 2: Add a test margin note to the existing test row**

Update the test row in `sec:surviving` to include a margin note:

```latex
\decreerow{1}%
    {𓇋𓅱𓂝𓏏𓆑𓂋𓈖𓏏𓁹𓅓}%
    {iw=f r ntt iri m \d{h}w,t-n\d{t}r}%
    {He shall do what is done in the temples}%
    {He moves toward what is performed within the god-houses}%
\glyphnote{𓂋}{D21}{r}{%
    \textbf{standard:} to, at\\
    \textbf{geometric:} toward, facing\\
    \textbf{cos\_sim:} 0.847\\
    \textbf{context:} [iw=f, ntt, iri]\\
    \textbf{neighbors:} n\,(0.91), m\,(0.88), \d{H}r\,(0.83)\\[3pt]
    \textbf{Demotic:} \textit{r}\\
    \textbf{Greek:} \textit{πρός}%
}
```

- [ ] **Step 3: Compile to verify margin note appears in the wide outer margin**

Run: `cd /Users/crashy/Development/heiroglyphy/docs/paper && xelatex rosetta_decree.tex`
Expected: Margin note renders in the 2.8in outer margin, showing the glyph header, all metadata fields, and cross-script references.

- [ ] **Step 4: Commit**

```bash
git add docs/paper/rosetta_decree.tex
git commit -m "feat: add glyphnote margin annotation command"
```

---

### Task 4: Introduction Section

**Files:**
- Modify: `docs/paper/rosetta_decree.tex` (Introduction section)

- [ ] **Step 1: Write the introduction**

Replace the placeholder introduction with:

```latex
\section{Introduction}
\label{sec:intro}

This document presents the Decree of Memphis---the text inscribed on the Rosetta Stone in 196 BCE---translated through a method that no ancient scribe could have imagined: vector space alignment of word embeddings trained on 100,729 sentences of Ancient Egyptian text.

The standard translation, refined over two centuries of Egyptological scholarship, renders each hieroglyph into its nearest English equivalent. Our \textit{geometric} translation does something different. It asks: given the statistical neighborhood of this word in the Egyptian textual record, what English word occupies the corresponding position in English semantic space? Where the two translations agree, the conventional reading is confirmed. Where they diverge, the geometry reveals something that literal translation compresses away.

This is a companion document to \textit{The Geometry of Meaning} \cite{brinsmead2026geometry}, which describes the alignment methodology in full. The present work applies that methodology to a single, celebrated text.

\subsection{How to Read This Document}

Each line of the Decree is presented as an interlinear stack:

\begin{enumerate}[nosep]
    \item \textbf{Hieroglyphs} --- rendered in the EgyptianHiero font.
    \item \textbf{Transliteration} --- standard Egyptological romanization (\textit{italic}).
    \item \textbf{Standard translation} --- the scholarly consensus reading.
    \item {\bfseries\color{rosetta}Geometric translation} --- our vector-space reading (\textbf{bold terracotta}).
\end{enumerate}

Margin notes annotate individual glyphs where the geometric reading diverges from the standard, or where the embedding space reveals unexpected contextual behavior. Each note includes:

\begin{itemize}[nosep]
    \item The glyph, its Gardiner code, and transliteration
    \item The standard and geometric readings with cosine similarity scores
    \item The context window and nearest neighbors in the embedding space
    \item Cross-references to the Demotic and Greek versions of the same passage
\end{itemize}

Lines marked \textcolor{reconstructed}{in gray with ⟦brackets⟧} are reconstructions. The hieroglyphic section of the Rosetta Stone is the most damaged---roughly half of the original $\sim$29 lines are lost or heavily fragmented. These have been reconstructed from the better-preserved Demotic and Greek sections, following standard Egyptological practice.

\subsection{Source Text}

The hieroglyphic transcription follows Quirke and Andrews (\textit{The Rosetta Stone}, British Museum Press, 1988) with reference to Budge's earlier edition. Line numbers correspond to the physical stone. Reconstructed passages are drawn from the Demotic and Greek sections as mediated by standard scholarship.
```

- [ ] **Step 2: Compile**

Run: `cd /Users/crashy/Development/heiroglyphy/docs/paper && xelatex rosetta_decree.tex`
Expected: Introduction renders cleanly with the colored geometric translation sample and reading guide.

- [ ] **Step 3: Commit**

```bash
git add docs/paper/rosetta_decree.tex
git commit -m "feat: add introduction and reading guide for rosetta decree"
```

---

### Task 5: Surviving Decree Lines (Lines 1–14)

**Files:**
- Modify: `docs/paper/rosetta_decree.tex` (Surviving Lines section)

This is the content-heavy task. Each line needs: hieroglyphic text, transliteration, standard translation, geometric translation, and selective margin notes.

**Note to implementer:** The hieroglyphic text, transliterations, and standard translations should be sourced from Quirke & Andrews. The geometric translations and margin note metadata (cosine similarities, neighbors, context windows) are illustrative placeholders — they demonstrate the format and will be refined with actual embedding lookups. The Demotic and Greek cross-references should reflect the known parallel text.

- [ ] **Step 1: Write surviving lines 1–5 with margin notes**

Replace the test content in `sec:surviving` with the first five lines of the decree. Each line uses `\decreerow` with appropriate margin notes on key glyphs. Example pattern for each line:

```latex
\decreerow{1}%
    {𓇋𓅱𓂝𓏏𓆑𓂋𓈖𓏏𓁹𓅓}%
    {iw=f r ntt iri m \d{h}w,t-n\d{t}r}%
    {He shall do what is done in the temples}%
    {He moves toward what is performed within the god-houses}%
\glyphnote{𓂋}{D21}{r}{%
    \textbf{standard:} to, at\\
    \textbf{geometric:} toward, facing\\
    \textbf{cos\_sim:} 0.847\\
    \textbf{context:} [iw=f, ntt, iri]\\
    \textbf{neighbors:} n\,(0.91), m\,(0.88), \d{H}r\,(0.83)\\[3pt]
    \textbf{Demotic:} \textit{r}\\
    \textbf{Greek:} \textit{πρός}%
}
```

Apply margin notes selectively per the design criteria: meaningful divergence, unexpected neighbors, cross-script insight, or contextual swap demonstration.

- [ ] **Step 2: Compile and verify lines 1–5**

Run: `cd /Users/crashy/Development/heiroglyphy/docs/paper && xelatex rosetta_decree.tex`
Expected: Five decree lines render with interlinear stacks and margin notes. Check for margin overflow — if notes collide, add `\vspace` adjustments between `\glyphnote` calls.

- [ ] **Step 3: Commit lines 1–5**

```bash
git add docs/paper/rosetta_decree.tex
git commit -m "feat: add surviving decree lines 1-5 with margin annotations"
```

- [ ] **Step 4: Write surviving lines 6–10 with margin notes**

Continue the pattern for lines 6–10.

- [ ] **Step 5: Compile and verify lines 6–10**

Run: `cd /Users/crashy/Development/heiroglyphy/docs/paper && xelatex rosetta_decree.tex`
Expected: Clean compilation, no margin collisions.

- [ ] **Step 6: Commit lines 6–10**

```bash
git add docs/paper/rosetta_decree.tex
git commit -m "feat: add surviving decree lines 6-10 with margin annotations"
```

- [ ] **Step 7: Write surviving lines 11–14 with margin notes**

Continue the pattern for the remaining surviving lines.

- [ ] **Step 8: Compile and verify lines 11–14**

Run: `cd /Users/crashy/Development/heiroglyphy/docs/paper && xelatex rosetta_decree.tex`
Expected: Full surviving section complete, all lines render correctly.

- [ ] **Step 9: Commit lines 11–14**

```bash
git add docs/paper/rosetta_decree.tex
git commit -m "feat: add surviving decree lines 11-14 with margin annotations"
```

---

### Task 6: Reconstructed Decree Lines (Lines 15–29)

**Files:**
- Modify: `docs/paper/rosetta_decree.tex` (Reconstructed Lines section)

- [ ] **Step 1: Write reconstructed lines 15–20 using \reconstructedrow**

Replace the placeholder `sec:reconstructed` content. Each line uses `\reconstructedrow` instead of `\decreerow`. Margin notes are lighter — fewer per line, and noted as less certain where applicable.

```latex
\section{The Decree: Reconstructed Lines}
\label{sec:reconstructed}

\reconstructedrow{15}%
    {𓈖𓏏𓂋𓅱...}%
    {n\d{t}r.w nb.w ...}%
    {All the gods and goddesses...}%
    {The divine ones, the totality...}%
\glyphnote{𓈖}{N35}{n\d{t}r.w}{%
    \textbf{standard:} gods\\
    \textbf{geometric:} divine ones, radiant\\
    \textbf{cos\_sim:} 0.793\\
    \textbf{context:} [nb.w, \d{h}w,t-n\d{t}r]\\
    \textbf{neighbors:} nbw\,(0.88), nswt\,(0.81)\\[3pt]
    \textbf{Demotic:} \textit{ntr.w}\\
    \textbf{Greek:} \textit{θεοί}\\[3pt]
    \textit{Reconstructed from Greek parallel.}%
}
```

- [ ] **Step 2: Compile and verify**

Run: `cd /Users/crashy/Development/heiroglyphy/docs/paper && xelatex rosetta_decree.tex`
Expected: Reconstructed lines render in gray with brackets, visually distinct from surviving lines.

- [ ] **Step 3: Commit lines 15–20**

```bash
git add docs/paper/rosetta_decree.tex
git commit -m "feat: add reconstructed decree lines 15-20"
```

- [ ] **Step 4: Write reconstructed lines 21–29**

Continue the pattern for the remaining reconstructed lines.

- [ ] **Step 5: Compile and verify**

Run: `cd /Users/crashy/Development/heiroglyphy/docs/paper && xelatex rosetta_decree.tex`
Expected: Full reconstructed section renders cleanly.

- [ ] **Step 6: Commit lines 21–29**

```bash
git add docs/paper/rosetta_decree.tex
git commit -m "feat: add reconstructed decree lines 21-29"
```

---

### Task 7: Commentary Section

**Files:**
- Modify: `docs/paper/rosetta_decree.tex` (Commentary section)

- [ ] **Step 1: Write the commentary**

Replace the placeholder commentary with thematic analysis covering:

```latex
\section{Commentary}
\label{sec:commentary}

\subsection{Where the Geometry Diverges}

% 2-3 paragraphs discussing the most significant places where the geometric
% translation reads differently from the standard. Reference specific lines
% and margin notes. Connect to findings from the main paper where relevant
% (e.g., if ntr appears near nbw in the decree context, link to the
% gold/divinity discovery).

\subsection{Contextual Swapping in Practice}

% 2-3 paragraphs analyzing glyphs that appear multiple times in the decree
% with different geometric readings. This is the core showcase: the same
% glyph, in different contexts, maps to different regions of English
% semantic space. Show the range of readings and what drives the shift.

\subsection{Cross-Script Convergence and Divergence}

% 1-2 paragraphs discussing where the Demotic/Greek readings align with
% the geometric translation (vs. the standard), and where all three
% diverge. This triangulation is unique to the Rosetta Stone — no other
% Egyptian text offers three parallel versions to compare against.

\subsection{Connections to the Embedding Space}

% 1-2 paragraphs linking decree-specific findings back to the broader
% discoveries documented in "The Geometry of Meaning" — e.g., the
% priest-as-administrator finding may manifest in how the decree describes
% priestly duties, the gold/divinity cluster may appear in honorific
% passages about the king.
```

- [ ] **Step 2: Compile**

Run: `cd /Users/crashy/Development/heiroglyphy/docs/paper && xelatex rosetta_decree.tex`
Expected: Commentary section renders with subsections.

- [ ] **Step 3: Commit**

```bash
git add docs/paper/rosetta_decree.tex
git commit -m "feat: add commentary section for rosetta decree"
```

---

### Task 8: Glyph Index Appendix

**Files:**
- Modify: `docs/paper/rosetta_decree.tex` (Glyph Index appendix)

- [ ] **Step 1: Write the glyph index**

Create a tabular appendix sorted by Gardiner code. Each row shows every contextual reading a glyph received across the decree:

```latex
\appendix
\section{Glyph Index}
\label{sec:glyph-index}

This index lists every glyph appearing in the Decree, sorted by Gardiner code. For glyphs with multiple occurrences, all contextual readings are shown, demonstrating the range of geometric translations produced by different surrounding contexts.

\vspace{8pt}

\begin{longtable}{@{}lllp{1.2in}p{1.2in}@{}}
\toprule
\textbf{Glyph} & \textbf{Code} & \textbf{Translit.} & \textbf{Standard Readings} & \textbf{Geometric Readings} \\
\midrule
\endhead
\hiero{𓀀} & A1 & i & I, me, my & I, me, my \\
\hiero{𓂋} & D21 & r & to, at, concerning & toward, facing; concerning; surpassing \\
\hiero{𓅓} & G17 & m & in, with, from & within, among; by means of \\
\hiero{𓈖} & N35 & n & to, for, belonging to & unto, for; of \\
% ... additional entries sorted by Gardiner code
\bottomrule
\end{longtable}
```

**Note:** This requires adding `\usepackage{longtable}` to the preamble.

- [ ] **Step 2: Add longtable to preamble**

Add `\usepackage{longtable}` to the packages section.

- [ ] **Step 3: Compile**

Run: `cd /Users/crashy/Development/heiroglyphy/docs/paper && xelatex rosetta_decree.tex`
Expected: Appendix renders as a multi-page table if needed.

- [ ] **Step 4: Commit**

```bash
git add docs/paper/rosetta_decree.tex
git commit -m "feat: add glyph index appendix with contextual readings"
```

---

### Task 9: Bibliography and Final Polish

**Files:**
- Modify: `docs/paper/rosetta_decree.tex` (bibliography, final adjustments)

- [ ] **Step 1: Add bibliography**

Add before `\end{document}`:

```latex
\bibliographystyle{plainnat}
\begin{thebibliography}{99}

\bibitem[Brinsmead and Claude, 2026]{brinsmead2026geometry}
Brinsmead, E. and Claude (Anthropic) (2026).
\newblock The Geometry of Meaning: What Vector Space Alignment Reveals About How Ancient Egyptians Thought.
\newblock \url{https://github.com/ebrinz/heiroglyphy}

\bibitem[Quirke and Andrews, 1988]{quirke1988rosetta}
Quirke, S. and Andrews, C. (1988).
\newblock \textit{The Rosetta Stone}.
\newblock British Museum Press.

\bibitem[Budge, 1913]{budge1913rosetta}
Budge, E.~A.~W. (1913).
\newblock \textit{The Rosetta Stone}.
\newblock British Museum.

\bibitem[Simpson, 2003]{simpson2003literature}
Simpson, W.~K., editor (2003).
\newblock \textit{The Literature of Ancient Egypt}.
\newblock Yale University Press, 3rd edition.

\bibitem[BBAW, 2023]{bbaw_tla}
Berlin-Brandenburg Academy of Sciences and Humanities (2023).
\newblock Thesaurus Linguae Aegyptiae (TLA): Digital corpus of Egyptian texts.
\newblock \url{https://aaew.bbaw.de/tla/}

\end{thebibliography}
```

- [ ] **Step 2: Add \usepackage{natbib} to preamble if not already present**

Verify natbib is in the package list. Add if missing.

- [ ] **Step 3: Final compilation — run XeLaTeX twice for cross-references**

Run: `cd /Users/crashy/Development/heiroglyphy/docs/paper && xelatex rosetta_decree.tex && xelatex rosetta_decree.tex`
Expected: Clean compilation with resolved cross-references, no warnings about undefined references.

- [ ] **Step 4: Commit**

```bash
git add docs/paper/rosetta_decree.tex
git commit -m "feat: add bibliography and finalize rosetta decree document"
```
