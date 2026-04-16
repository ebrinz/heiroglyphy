# Publishing Roadmap

**Status:** Parked pending Egyptologist co-author.
**Goal:** Move `docs/paper/heiroglyphy.tex` from "strong preprint" to "peer-reviewed publication" in a Digital Humanities / Cultural Analytics venue.
**Owner:** Erik
**Last updated:** 2026-04-16

---

## Where we are

- ✅ V15 SOTA stable at 32.35% Top-1 (`heiro_v15/`)
- ✅ Full paper drafted (`docs/paper/heiroglyphy.tex`, 405 lines, 7 sections)
- ✅ Discoveries documented (`DISCOVERIES.md`)
- ✅ Repo open-source, reproducible
- ✅ YouTube mini-doc published
- ⛔ No domain-expert validation of Egyptological claims
- ⛔ Discoveries presented as anecdote, not evidence
- ⛔ Paper framed as NLP result; submitting to NLP venues will fail
- ⛔ Function-word ceiling not addressed in evaluation

---

## Target venues (in priority order)

| Rank | Venue | Type | Why | Open access |
|------|-------|------|-----|-------------|
| 1 | **Journal of Cultural Analytics** | Journal | Best fit — values method + interpretive payoff equally | Yes |
| 2 | **Digital Scholarship in the Humanities** (Oxford) | Journal | More traditional, higher prestige | Hybrid |
| 3 | **Computational Humanities Research (CHR)** | Conference | Peer-reviewed, friendly to method+findings | Yes |
| 4 | **LaTeCH-CLfL workshop** (ACL/EMNLP-adjacent) | Workshop | Right NLP venue if going that route | Yes |

**Pre-submission action for each:** read 3 recent papers, mirror structure.

---

## Phases

### Phase 0 — Park-time prep (no Egyptologist needed) ⏸️

Work that can happen now in spare cycles. None of it is wasted regardless of which way the project goes.

- [ ] **Stratified evaluation script.** Add `heiro_v16_evaluation/` (or extend V15) computing Top-1/5/10 separately for:
  - Function words (English target ∈ NLTK stopwords)
  - Content words (everything else)
  - Per-POS bins if achievable
  - Per concept domain (use `concept_categories.json`)
  - **Why:** content-word Top-1 is almost certainly 50%+ — your real headline number.
- [ ] **VecMap baseline.** Run [`artetxe/vecmap`](https://github.com/artetxe/vecmap) on the same anchor split. Report regardless of outcome — reviewers will demand this comparison.
- [ ] **Corpus bias quantification.** Compute % of corpus by genre (funerary, religious, administrative, narrative). Bake into Discussion section so reviewers can't surprise you with it.
- [ ] **Generate the validation probe set BLINDLY.** Pre-register 40–50 semantic probes (analogies, midpoints, cluster centroids, cross-domain bridges) before any Egyptologist sees outputs. Commit the list with a timestamped commit. **Critical:** this list cannot be edited later or the validation is worthless.
- [ ] **Citation cleanup.** Every Egyptological claim in `DISCOVERIES.md` needs a source (Allen *Middle Egyptian*, Loprieno *Ancient Egyptian: A Linguistic Introduction*, Vycichl *Dictionnaire étymologique*, Erman-Grapow *Wörterbuch*). Mark unsourced claims with `[CITE]` so co-author can fill in.

### Phase 1 — Recruit Egyptologist co-author 🎯 (GATING)

This is the unblocking step. Without it, nothing past here happens.

**Outreach channels:**
- [ ] **Egyptological Forum / EEF mailing list** — post a short pitch
- [ ] **Academia.edu** — find faculty whose papers cite distributional semantics, computational philology, or Egyptian lexicography
- [ ] **Cold email faculty at:** Brown (Egyptology + Digital Humanities), UCLA, Chicago Oriental Institute (now ISAC), Oxford, Leiden, Heidelberg, Göttingen, Berlin (TLA team — already a data source!)
- [ ] **TLA team direct contact** — they made the corpus you used; natural fit, already invested in computational Egyptology
- [ ] **Twitter/Bluesky #Egyptology #DigitalHumanities** — soft outreach
- [ ] **Conferences:** ICE (International Congress of Egyptologists), DH annual conference

**Pitch template (one paragraph):**
> I'm an independent researcher who built an unsupervised cross-lingual alignment system for Ancient Egyptian (32% Top-1, 100K BBAW sentences). The system surfaces semantic structures invisible to literal translation — *gold colocated with divinity*, *silence as the defining quality of death*, *the snake as divine rather than wise*. I have a draft paper but need an Egyptologist co-author to validate the claims against the literature and bring disciplinary credibility before submitting to *Journal of Cultural Analytics* or similar. The repo is open: github.com/ebrinz/heiroglyphy. Would you be interested in a 15-min call?

**What to offer:** equal authorship, full data/code transparency, you do all engineering and writing scaffolding, they do validation + Egyptological framing + literature situating.

**Realistic timeline:** 2–6 months to land the right person. Junior faculty / postdocs more responsive than senior.

### Phase 2 — Discovery validation 🔬 (needs co-author)

Once co-author is on board:

- [ ] Co-author scores pre-registered probe set blind on rubric: `plausible / surprising-but-defensible / wrong / unevaluable`
- [ ] Compute and report distribution: e.g., "32/47 probes returned plausible Egyptological readings"
- [ ] Each "plausible" result gets one sentence of literature situating ("This matches Allen's reading of...")
- [ ] Each "surprising-but-defensible" result becomes a featured discovery
- [ ] Each "wrong" result is reported honestly as a limitation

This is what converts the Discoveries section from anecdote → evidence.

### Phase 3 — Methodological additions

- [ ] Add VecMap + stratified eval results to paper
- [ ] Add corpus genre breakdown
- [ ] Promote BERT catastrophe and Coptic trap to "Negative Results" subsection — they are real contributions
- [ ] Add a **Limitations** section reviewers can't ambush you with: corpus bias, function-word ceiling, no living validators, transliteration variance, sign-vs-word granularity

### Phase 4 — Restructure paper for venue

If targeting JCA:
- [ ] Lead with the cultural-analytic question, not the alignment task
- [ ] Demote accuracy to "method validation" subsection
- [ ] Promote validated discoveries to the spine of the paper
- [ ] Cut anything that reads as "look how cool this is"; convert to "here is evidence that..."
- [ ] Length-check against JCA norms (typically 6–10K words)

### Phase 5 — Submit

- [ ] Final co-author review
- [ ] Submit to venue #1
- [ ] Post arXiv preprint same day
- [ ] If rejected → revise based on reviews → submit to venue #2 (don't sit on it)

---

## Decisions to revisit when unparking

- **Author byline.** Currently "Erik Brinsmead + Claude (Anthropic)." Most journals will not accept an LLM as a named author. Acceptable framings: "with computational assistance from Claude," or move to acknowledgments. Confirm policy with target venue.
- **Whether to retrain V15 with `cc.en.300` target.** Easy +2-3pp possible; not blocking publication; do only if reviewers ask.
- **Whether to do LLM-rerank experiment.** Genuinely interesting, possibly +5pp Top-1, but adds a whole methodological section. Defer unless reviewers want a stronger SOTA.
- **Whether to make a separate "tools paper" out of `final_output/`.** The aligned vectors + lookup utility could be a JOSS submission independent of the main paper. Low effort if main paper lands.

---

## Success criteria for unparking

Resume active work when **any one** of:
1. Egyptologist co-author confirmed (primary trigger)
2. Phase 0 backlog complete and you want momentum on Phase 1 outreach
3. A relevant venue announces a special issue / themed call you'd be a fit for

---

## Notes

- Don't chase Top-1 accuracy further. The 32.35% number is fine; the metric is partly broken (60% function words). Effort goes into validation, not optimization.
- The DISCOVERIES.md content is the actual asset. Protect it: don't submit anywhere that requires assigning copyright that would block reuse.
- Keep the YouTube doc and repo public — they're discovery surface for finding the co-author.
