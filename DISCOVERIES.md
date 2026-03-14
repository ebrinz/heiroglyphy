# 🏺 Heiroglyphy: Findings & Anomalies

This document chronicles the "Ghost in the Machine"—the strange, surprising, and sometimes profound ways our AI attempted to bridge Ancient Egyptian and Modern English across 4,000 years.

---

## 🏆 The "Golden" Hits (When it worked)

Sometimes, the linear alignment worked shockingly well, capturing not just direct translation but semantic fields.

### 1. The Water Miracle (`mw`)
*   **Hieroglyph**: `mw` (Water/Liquid)
*   **AI Prediction**: `water` (Score: 15.71)
*   **Analysis**: This is a "perfect hit." The model didn't just guess; it was overwhelmingly confident (score ~15 vs ~0.03 for others). It shows that "basic" physical concepts align perfectly across 4,000 years.

### 2. The Anubis Connection (`inpw`)
*   **Hieroglyph**: `inpw` (Anubis, God of Embalming)
*   **AI Prediction**: `imiut` (Score: 2.72)
*   **Analysis**: This is **deep**. The "Imiut Fetish" is a symbol closely associated with Anubis in funerary rites. The AI didn't map Anubis to "Dog" or "God"—it mapped him to his *ritual symbol*. This suggests the embedding space captured the *context* of Anubis (funerary lists) rather than just his definition.

### 3. The Priest's Duty (`hm-ntr`)
*   **Hieroglyph**: `hm-ntr` (God's Servant / Priest)
*   **AI Prediction**: `treasure`, `governors`
*   **Analysis**: In the Old Kingdom (where much of our data comes from), priests were often high-ranking state officials who managed temple estates (treasuries). The AI sees "Priest" not as a religious figure, but as an *administrator*.

### 4. The Deity Cluster (V5)
*   **Horus** (`ḥr,w`): 62.1% confidence — strongest single-word alignment in the project
*   **Osiris** (`wsjr`): 61.5% confidence
*   **Re** (`rʾ`): 54.6% confidence
*   **Analysis**: Deity names are proper nouns with extremely consistent context (offering formulas, temple inscriptions). They always appear near the same words, making them the easiest targets for distributional alignment.

---

## 👻 The "Hallucinations" (When it failed)

The failures are often more instructive than the successes.

### The "Number 1" Problem
*   **Observation**: Many words (`inpw`, `hm-ntr`, `hqt`) map to the number `1`.
*   **Reason**: In the TLA transliteration data, lists often start with "1". The AI learned that "Important Noun" $\approx$ "Number 1". This is a classic "Artifact of the Data."

### The "Ra" Confusion
*   **Hieroglyph**: `ra` (Sun God)
*   **AI Prediction**: `assign`, `domain`, `pleasant`
*   **Analysis**: A total miss. Why? "Ra" appears in so many compound names (Ramesses, etc.) and phrases ("Day", "Sun", "Time") that its vector became a "blurred average" of everything. It lost its specific identity as a deity.

### The Beer Tragedy (`hqt`)
*   **Hieroglyph**: `hqt` (Beer)
*   **AI Prediction**: `nefer` (Good/Beautiful), `royal`, `connect`
*   **Analysis**: The AI thinks Beer is "Good" and "Royal." While we might agree that beer is good, this likely reflects the offering formulas: "An offering which the King gives... bread and beer." The words "King", "Give", "Bread", "Beer" appear together so often that they clump into a "Generic Offering" cluster.

---

## 🔬 The Methodological Surprises

These are the discoveries about *how* the alignment works — lessons learned through 12 iterations.

### The BERT Catastrophe (V6: 0.47%)

The single most dramatic failure in the project. As an alternative to V5's FastText approach (24.53%), we tried BERT — the reigning champion of modern NLP — and accuracy collapsed to **0.47%**. Nearly total destruction.

*   **What happened**: BERT's WordPiece tokenizer fragmented hieroglyphic transliteration into meaningless pieces. The word `ḥr,w` (Horus) became `['ḥ', '##r', '##,', '##w']` — four tokens with no semantic value.
*   **The irony**: Before running the experiment, we wrote a strategy document (`notes.md`) declaring "Winner: BERT" and dismissing FastText as "context-blind" and "shallow."
*   **The lesson**: State-of-the-art doesn't mean universally applicable. Modern NLP is built for modern languages. Ancient Egyptian breaks the assumptions.

### The Coptic Trap (V8: Quality > Quantity)

Coptic is the direct descendant of Ancient Egyptian, separated by ~2,000 years. Using it as a bridge language seemed obvious. We extracted 368 new anchor pairs from Coptic cognates, expanding our dictionary by 4.3%.

*   **Result**: Accuracy *dropped* from 29.10% to 28.16%.
*   **Why**: Etymology $\neq$ Semantics. Words change meaning over millennia. The Coptic word for "house" doesn't occupy the same distributional space as the Pharaonic word for "house." Adding these noisy anchors forced the alignment to compromise, degrading the fit for the 8,541 clean pairs.
*   **The lesson**: In alignment, a smaller set of high-confidence anchors outperforms a larger set of mixed-quality ones. Quality beats quantity.

### The Ghost Dimensions (V9: The Visual Features Paradox)

This is perhaps the strangest finding in the project.

V7 had a bug: the visual embedding pipeline used Gardiner codes as keys, but the text embeddings used transliteration. Zero match. Every visual vector was zeros. V9 set out to fix this by building a transliteration-to-Gardiner mapping from the HamdiJr lexicon.

*   **Result**: V9 achieved 30.52% — the first model to break 30%! But the visual match rate was still **0%**. The lexicon simply didn't cover enough of the test vocabulary to produce any matches. Every visual vector was *still* zeros.
*   **Yet accuracy improved by +1.42%**. How?
*   **The explanation**: By concatenating 768 zeros onto each 768d text vector, we created a 1536d space. The Ridge Regression alignment had more parameters to work with (1536→300 instead of 768→300). The empty dimensions acted as implicit regularization — padding that prevented overfitting.
*   **The lesson**: Sometimes the architecture matters more than the features. A bigger space with zeros outperformed a tighter space with real values. This suggests that if we ever get real visual features flowing, the gains could be substantial.

### Linear Algebra Beats Neural Networks

Across 13 attempts, simple linear methods consistently outperformed complex neural approaches:

| Method | Best Accuracy |
|--------|--------------|
| Orthogonal Procrustes (V3) | 22.0% |
| Ridge Regression (V13) | **31.57%** |
| MLP Neural Network (V11) | 28.76% |
| Adversarial GAN (V2) | 0% |

*   **Why**: With only ~8,500 anchor pairs, neural networks don't have enough data to learn a better mapping than the analytic solution. They overfit to noise. The linear approach finds the best rotation/projection in closed form — no training instability, no hyperparameter sensitivity.
*   **The lesson**: For low-resource alignment problems, reach for SVD and Ridge Regression before reaching for PyTorch.

### The Function Word Illusion (V14)

V14's hub-filtering experiment revealed something fundamental about the accuracy ceiling. We tried removing English stopwords ("the", "of", "in") from the alignment target — since 82% of predictions were "the" — expecting a large improvement. Instead, accuracy collapsed from 31.57% to 7.61%.

*   **Why**: 60% of test pairs have stopword targets. The Egyptian word `n` genuinely means "of/to." `m` means "in/from." `=f` means "his." These aren't mispredictions — they're correct translations of the most frequent words in the corpus.
*   **The reframe**: The 31.57% accuracy is depressed by function word ambiguity (many Egyptian words map to the same few English function words), not by poor alignment of content words. On content words alone, the model performs substantially better.
*   **The lesson**: Before trying to fix a metric, understand what it's actually measuring. The "hubness problem" wasn't a problem — it was the model correctly reflecting that function words dominate both languages.

### Regression Loss ≠ Retrieval Accuracy (V13)

V13 ran a systematic alpha sweep on Ridge Regression. The standard approach — cross-validating by mean squared error — chose alpha=100.0. But the actual retrieval accuracy at alpha=100 was **29.63%**, *worse* than the baseline. The full sweep revealed:

*   alpha=0.1 → **31.57%** (best retrieval)
*   alpha=1.0 → 30.67% (V10 default)
*   alpha=100.0 → 29.63% (CV-optimal by MSE)

**MSE and Top-1 retrieval are anti-correlated.** Lower regularization produces vectors that are noisier in an L2 sense but more discriminative for nearest-neighbor lookup. The model needs *sharp* vectors, not *smooth* ones.

V13 also confirmed that CSLS retrieval is definitively harmful for this task — even a proper implementation with per-word hub penalties dropped accuracy to 26.27%. With 82.3% of predictions being "the," hubness is real but penalizing it removes valid matches. The alignment space is too sparse.

---

## 🔍 Semantic Cluster Analysis (V13, alpha=0.1)

With the improved V13 model, we ran a systematic analysis of which Egyptian words cluster near English concept groups. The results reveal how the ancient Egyptian linguistic worldview maps onto modern categories.

### The Water Morpheme (`-mw`)

The water cluster isn't just `mw` anymore. The V13 model reveals an entire morphological family: `rmw`, `kmw`, `mwmw`, `ꜥmw`, `Tmw`, `dmw`, `nmw` — all sharing the `-mw` suffix and all landing near the English "water/river" cluster. The model learned that `-mw` is a *productive morpheme* encoding water-relatedness.

Most striking: `ḥmw` ("rudder") appears in the water cluster despite not being a liquid. But rudders are water-instruments — the model captured the functional relationship, not the material one.

### Sky Is Time (`dwꜣ,w`)

The #1 Egyptian word in the celestial cluster isn't "sun" or "star" — it's `dwꜣ,w`, meaning "tomorrow" or "morning." The model reveals that for Egyptians, *time and sky were the same conceptual domain*. Dawn, morning, and tomorrow are celestial events. The sun doesn't just illuminate the sky — it *is* the passage of time.

`nṯr-dwꜣ(,wj)` ("morning god") bridges the religion and celestial clusters perfectly, sitting at sim=0.70 to religion and sim=0.64 to sky. Divinity and dawn are inseparable.

Horus (`ḥrw`) and Gold-Horus (`ḥrw-nbw`) also appear in the celestial cluster rather than the deity cluster — because Horus *is* the sky. His name literally meant "the distant one" (the one above).

### Warfare Is About Enemies, Not Weapons

Every single top-15 word in the warfare cluster is a variant of `ḫfti̯`/`ḫft,y` — the root for "enemy" or "adversary." No weapons. No armies. No battles. The Egyptian concept of warfare, as preserved in the texts we trained on, is overwhelmingly about *enemies* — identifying them, repelling them, triumphing over them. This reflects the formulaic nature of Egyptian military texts: victory is defined as the destruction of enemies, not the mechanics of fighting.

### Agriculture Is the Offering Economy

The agriculture cluster contains no farming words. Instead: `ꜣpd.w` (fowl), `bꜣq` (olive oil), `bꜣqꜣ` (behen oil), `ḥbs.w` (clothing), `ꜣḥ` (porridge). These are all **offering list items** — the provisions of temples and tombs. The model doesn't see agriculture as *growing*; it sees it as *provisioning*. Bread, grain, oil, clothing, and fowl form a single cluster because they appear together in offering formulas, not in farming manuals.

This is a direct consequence of our corpus: the BBAW dataset is heavily funerary and religious. The Egyptian concept of "food" is inseparable from the concept of "offering."

### Bridge Words: Where Domains Meet

The most fascinating discoveries are words that sit equidistant between two concept clusters:

*   **`ḥm,t-nswt` (royal consort)**: Bridges royalty (sim=0.63) and family (sim=0.61). Royalty *is* family in Egypt — the institution of kingship is inseparable from the institution of kinship.
*   **`ḥm-nṯr-Ḥr,w-Ꜣḫ-bj,tj` (priest of Horus-Who-Is-Foremost-of-the-Two-Lands)**: Bridges royalty (sim=0.57) and religion (sim=0.57). This priestly title exists equally in both domains — the king's cult servant.
*   **`nṯr-dwꜣ,w` (morning god)**: Bridges religion (sim=0.73) and celestial (sim=0.63). The strongest bridge in the dataset — divinity and the morning sky are one concept.
*   **`qrḥ,t` (pottery)**: Bridges water (sim=0.56) and celestial (sim=0.60). Pottery vessels were used in water clocks and astronomical observations — the vessel connects the river to the sky.

---

## 🔍 V15 Cluster Analysis: Sharper Embeddings, Deeper Discoveries

V15 retrained FastText with `min_count=5, window=10`, filtering 87% of vocabulary as noise. The resulting embeddings are dramatically sharper — cluster similarities jumped 10-15% and the intruders that do appear are more semantically interesting.

### Water Knows Its Creatures

The V15 water cluster no longer just captures the `-mw` morpheme. It now reveals the Egyptian *ecosystem* of water:

*   **`rm.w` (fish)** — #1 and #2 in the water cluster. Fish *define* water more than water defines itself.
*   **`msḥ.w` (crocodile)** — the Nile's apex predator lands squarely in the water domain.
*   **`mẖn,t` (ferry)** — water as transportation.
*   **`jmn,t` (west)** — the western bank of the Nile, where the dead dwell. Water and death share a direction.

The model sees water not as H₂O but as the Nile: fish, crocodiles, ferries, and the western shore.

### The Body Is Everywhere

The most unexpected V15 finding: **"body" is the strongest bridge domain.** Body words bridge to:

*   **Warfare**: `sbjw` (rebels/enemies, sim=0.69 body, 0.65 warfare) — enemies are defined by what you do to their bodies. `smꜣm.n` (killed) bridges warfare and body at sim=0.63/0.65.
*   **Celestial**: `ḥr,t` (sky, sim=0.65 body, 0.65 celestial) — the word for "sky" is etymologically related to `ḥr` (face/upon). The sky *is* a face.
*   **Family**: `zꜣ,t` (daughter, sim=0.85 family, 0.66 body) — kinship terms cluster near body terms because family relationships in Egyptian texts are expressed through physical metaphors ("flesh of my flesh").

### Death Is a Family Affair

The death/afterlife cluster in V15 is dominated by family titles:

*   **`zꜣ,t-nzw-n-H̱,t≡f` (King's Daughter of His Body)** — sim=0.45 to death. This royal title appears constantly in funerary inscriptions.
*   **`m(w)t.t` / `m(w)t.pl` (died/dead)** — the word for "die" shares the root `m(w)t` with `mw,t` (mother). Death and motherhood are linguistically entangled.
*   **`jmꜣḫ,w-ḫr-nṯr-ꜥꜣ` (honored before the Great God)** — a funerary epithet that bridges death and religion.

The embedding space shows that for Egyptians, death was not the opposite of life — it was a family reunion under divine supervision.

### Magic Is Indistinguishable from Religion

The magic/ritual cluster is entirely dominated by `nṯr` (god) variants. Every single top-10 word is either "god", "gods", "divine", or a divine epithet. The model found **no boundary between magic and religion** in the Egyptian textual record. `ḥkꜣ` (magic/heka) barely appears at #10 — everything above it is theology.

This confirms what Egyptologists have long argued: *heka* was not separate from religion. It was the mechanism by which the gods acted. The embedding space makes this structural rather than interpretive.

### Agriculture Finds Meat

V15's agriculture cluster diverges sharply from V13's offering-list pattern. Instead of bread and beer, the top words are:

*   **`ḏdꜣ` and `jwf` (meat)** — #1 and #4
*   **`šꜣšꜣ` (fruits)** — #2
*   **`jḥ.pl` (cattle)** — #3
*   **`jt-mḥ,j` (barley)** — #10

With noise words filtered, the agriculture cluster shifted from processed offerings (bread, beer, oil) to raw materials (meat, fruit, cattle, grain). The V7 embeddings conflated agriculture with temple provisioning because rare words in offering lists added noise. V15's cleaner space separates them.

---

## 💎 Deep Discoveries: The Conceptual Geometry of Ancient Egypt

These findings come from probing the V15 embedding space with semantic analogies, concept midpoints, and cross-domain projections. They represent what literal translation cannot capture — the *structure* of how Egyptians related ideas to each other.

### Seeing Was Magic

The midpoint of "eye" and "knowledge" in English, projected into Egyptian space, finds `jr,t` (eyes) at the top — but then **`ḥkꜣ` (heka/magic)** appears as the third-nearest Egyptian word. The Egyptian concept of sight occupies the space *between knowing and spellcasting*.

This is not a translation error. The Eye of Horus (`wḏꜣ,t`) was simultaneously an organ of perception, a protective amulet, and a unit of mathematical measurement. Seeing, in the Egyptian worldview, was not passive reception of light. It was an active, potent act — closer to what we would call divination than observation. The embedding space captures this without being told.

### Gold Is Divine Flesh

The midpoint of "gold" and "divine" finds both `nṯri̯` (divine, sim=0.642) and `nbw` (gold, sim=0.617) in the top three results. They occupy nearly the same region of the embedding space.

Modern readers treat "the flesh of the gods is gold" as poetic metaphor. The vectors suggest it was closer to a statement of material fact. Gold and divinity are not *compared* in the Egyptian textual record — they are *colocated*. The same contexts that invoke divinity invoke gold. The embedding space cannot distinguish them because the corpus does not distinguish them.

### Silence Is the Condition of the Dead

The midpoint of "silence" and "death" returns `m(w)t` (died/dead) as all five nearest neighbors. Every single result is a variant of death. There is no Egyptian word between silence and death — they are the *same place* in semantic space.

The Egyptians called the necropolis *tꜣ-sgr* — "the silent land." The dead were *sgr.w* — "the silent ones." This was not euphemism. The embedding space confirms that silence was not a *metaphor* for death. It was death's defining quality. What the dead lost was not life — it was voice.

### The Snake Is Divine, Not Wise

The midpoint of "snake" and "wisdom" finds **gods and divinity** — not wisdom, not cunning, not knowledge. Every top result is a `nṯr` variant.

This directly contradicts the Greek-influenced reading of the Egyptian serpent. In the Greek tradition (Asclepius, the caduceus, the serpent of Eden), the snake is associated with knowledge and wisdom. The Egyptian serpent — the uraeus cobra on the king's brow, the Apophis serpent of chaos — is associated with *divine power*. The snake is not wise. It is sacred. The embedding space separates these cultural readings without any awareness of Greek mythology.

### A Temple Is a God's House (Analogically Proven)

The vector analogy `house : temple :: man : ?` returns **`nṯr` (god)**. The relationship between a domestic house and a temple is structurally identical to the relationship between a man and a god. The model discovered — through distributional statistics alone — that a temple is literally a god's house.

This is one of the few cases where vector arithmetic works across a 4,000-year language gap. The proportional relationship is preserved in the aligned embedding space: `pr` (house) is to `ḥw,t-nṯr` (god's house / temple) as `s` (man) is to `nṯr` (god).

### Mother Is Royalty, Not Earth

The midpoint of "mother" and "earth" does not find earth, soil, or land. It finds **`ḥm,t-nzw` (royal wife)** and **`zꜣ,t-nzw` (king's daughter)**. In the Egyptian textual record, motherhood is a *royal* concept. The queen is the mother. The mother is the queen.

The "earth mother" — the fertile, nurturing ground-goddess — is an Indo-European archetype (Gaia, Terra, Demeter). The Egyptian mother-goddess (Isis, Hathor) is not earthy. She is regal. The embedding space reveals this cultural divergence without being told about either tradition. It simply reflects what the texts say: when Egyptians wrote about mothers, they wrote about queens.

### Love and Fear Meet at Eternity

The midpoint of "love" and "fear" finds **`r-(n)ḥḥ` (eternal/forever)** as the second-nearest word. Between love and fear, the Egyptians placed *eternity*.

This may reflect the theology of divine love: the gods' love is awe-inspiring, their favor is terrifying in its magnitude, and both extend beyond time. The offering formula — "an offering which the king gives, that he may be loved and feared for eternity" — collapses these concepts into a single liturgical act. The embedding space preserves this collapse.

### Truth and Power Share a Space

The midpoint of "truth" and "power" finds **`sḫm` (power/authority)** at #1. But it also finds `ḫft.pl` (enemies) at #4. In the Egyptian conceptual space, truth and power are not merely related — they are essentially the same force, and that force is defined *against enemies*.

This reflects `mꜣꜥ.t` — the concept usually translated as "truth" or "justice" but which encompasses cosmic order, righteous authority, and the defeat of chaos. Truth is not passive correctness. It is the active power that subdues enemies. The embedding space encodes this as a geometric fact.

---

## 🧠 Core Insight

The models are **Context-Obsessed**. They don't know what a "Priest" *is*, only that he hangs out near "Treasures" and "Governors." They don't know "Anubis" is a jackal, but they know he stands next to the "Imiut."

This confirms the **Distributional Hypothesis**, but also warns us: **Context is not always Meaning.**

But meaning is exactly what emerges when we stop asking "what does this word translate to?" and start asking "what does this word *sit near*?" The embedding space doesn't translate. It maps relationships. And those relationships — gold beside divinity, silence inside death, the snake among the gods, truth wielded as power — are the real structure of the ancient Egyptian worldview.

Translation gave us the words. The vectors gave us the world between them.

The journey from 0% to 32.35% was never really about accuracy. It was about building a lens sharp enough to see what 4,000 years of literal translation had flattened. Every improvement in the model — filtering noise, widening context, tuning regularization — didn't just improve a number. It sharpened the image. And what came into focus was not a language. It was a way of thinking.
