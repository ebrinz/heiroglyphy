# Collaboration Questionnaire — Heiroglyphy Project

**Purpose:** Help us both decide whether co-authoring this paper is a fit, and if so, calibrate working norms before we start.

**Structure:** Three sections. Section A is ~5 minutes (cold-screen). Section B is ~15 minutes (substantive interest check, with three sample probes for you to react to). Section C is ~10 minutes (logistics, only if A and B feel good).

You don't have to answer in order; if a question feels wrong-shaped, replace it with the answer you think I should have asked for.

---

## Section A — Quick fit (5 min)

1. **Specialization.** Period / region / text-type you work most closely with (e.g., Middle Kingdom funerary, New Kingdom administrative, Late Period demotic, Coptic, etc.).
2. **Familiarity with our source corpora.** Have you used or cited any of: Thesaurus Linguae Aegyptiae (TLA), Berlin-Brandenburg Academy (BBAW) corpus, Ramses Online, HamdiJr lexicon? Even casually?
3. **Computational exposure.** Roughly where do you sit:
   - (a) Never used computational methods, but curious
   - (b) Read papers using them, can interpret results
   - (c) Used them in own work (corpus statistics, concordances, NLP)
   - (d) Active computational philologist
   *(Any of these is fine. None is a wrong answer — it just shapes the writing division.)*
4. **Time horizon.** Realistically, do you have bandwidth for a co-authorship in the next 3–6 months? Roughly how many hours/month feels honest?
5. **Career stage.** Doctoral student / postdoc / junior faculty / senior faculty / independent. *(Asking because incentives differ — I want this collaboration to actually help you, not just me.)*

---

## Section B — Substantive interest (15 min)

This is the part that matters most. The project's value depends on whether the discoveries hold up to expert reading. Below are three actual outputs from the V15 system. Please react however feels natural — agreement, disagreement, "this is trivial," "this is wrong because X," "this is interesting but here's what's missing." There's no right answer; I'm trying to see how you read this kind of evidence.

### Probe 1 — *Silence and death*

The midpoint of the English vectors for "silence" and "death," projected into Egyptian space, returns five nearest neighbors — *all* of them variants of `m(w)t` (died/dead). No other Egyptian root appears in the top 5.

Our reading: this is the embedding-space confirmation of *tꜣ-sgr* ("the silent land") and *sgr.w* ("the silent ones") as more than euphemism — silence is a defining quality of the dead state, not a metaphor for it.

**Your reaction?** Plausible / known / overstated / wrong / something else?

### Probe 2 — *Snake and divinity, not snake and wisdom*

The midpoint of "snake" and "wisdom" returns gods and divine epithets — every top-10 neighbor is a `nṯr` variant. *No* word for wisdom, knowledge, or cunning appears.

Our reading: this contradicts the Greek-influenced (Asclepius, caduceus, Genesis) reading of the serpent as wise, and confirms the Egyptian uraeus/Apophis framing as primarily about *divine power*. The corpus does not associate snakes with wisdom.

**Your reaction?** And: is the framing fair, or are we overclaiming a "Greek vs Egyptian" contrast that an Egyptologist would phrase more carefully?

### Probe 3 — *Mother and earth → royal wife*

The midpoint of "mother" and "earth" does *not* find earth, soil, land, or any agricultural noun. It finds `ḥm,t-nzw` (royal wife) and `zꜣ,t-nzw` (king's daughter).

Our reading: the "earth mother" archetype (Gaia, Demeter, Terra) appears to be absent; Egyptian motherhood is *royal* in the texts we have. The Egyptian mother-goddesses (Isis, Hathor) are queenly, not earthy.

**Your reaction?** This one I'm least sure about — is there a reading where this is just a corpus-bias artifact (BBAW being heavily royal/funerary) rather than a real cultural difference?

### Open questions

6. Of those three probes, which (if any) would you want to write up most carefully? Which would you push back on hardest?
7. Are there discoveries in `DISCOVERIES.md` (linked separately) that you'd flag as etymologically shaky or insufficiently sourced? Pointing at them is exactly the kind of help I need.
8. What's a discovery you *wish* this kind of system could test? (i.e., a long-standing Egyptological question where embedding evidence might add something.)

---

## Section C — Working norms (10 min, only if A + B feel good)

9. **Authorship.** I'm proposing equal co-authorship (alphabetical or order-by-contribution, your call). Acceptable? Anything you'd want different?
10. **LLM authorship stance.** Current draft credits Claude (Anthropic) as a co-author. Most journals don't allow this. Are you comfortable demoting LLM contribution to acknowledgments / methods section, or do you have stronger views?
11. **Target venue ranking.** I'm aiming at *Journal of Cultural Analytics* first, then *Digital Scholarship in the Humanities*, then CHR. Reasonable? Better venues I'm missing for your subfield?
12. **Working tools.** Overleaf / Google Docs / Word / Markdown-and-git? Async-only or do you want regular calls? Timezone?
13. **Disagreement protocol.** When the model output and your Egyptological reading conflict, what's your preferred resolution? (Defer to expert reading? Report both? Use it as a featured limitation?)
14. **IP / openness.** The data, code, and aligned vectors are open-source. Any institutional constraints (funder mandates, embargoes, dual-submission policies) I should know about?
15. **Conflicts.** Anything in this work that overlaps with your own forthcoming publications — either as risk or as natural complement?
16. **Beyond this paper.** Do you see this as a one-off or as the start of a longer collaboration (e.g., Pyramid Texts analysis, Book of the Dead, Coptic bridge work)? No wrong answer.

---

## What you'd be signing on to

Concretely, the co-author commitment is:

- **Validation pass** on a pre-registered set of ~40–50 semantic probes (blind scoring on a 4-point rubric). Estimated: 6–10 hours.
- **Literature situating** for the validated discoveries — one sentence per discovery, citing standard sources. Estimated: 4–6 hours.
- **Paper revisions** on the Egyptological framing in Sections 1, 5, and 6. Estimated: 8–15 hours.
- **Response to reviewers** if/when we get a revise-and-resubmit. Estimated: 5–10 hours.

**Total realistic ask:** 25–40 hours over 2–4 months.

I do all engineering, all writing scaffolding, all submission logistics. You bring disciplinary reading, citation depth, and the credibility that lets a humanities reviewer take the paper seriously.

---

## What to send back

If interested, no need to answer everything formally — even a paragraph reaction to one of the three probes plus a note on Section C is enough to set up a 30-min call.

If not interested, **a one-line "not for me" is genuinely fine and appreciated.** Pointing me to one colleague who might be a better fit is even better.

---

*Erik Brinsmead — github.com/ebrinz/heiroglyphy — [contact]*
