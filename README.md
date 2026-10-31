# Séance — a local medium that claims to read a brain, and an audit of whether it can

A séance console: a linear probe reads one second of EEG and reports a single
concept; a local **Gemma 3 4B IT** turns that impression into a message from
beyond. Then we check whether the ghost can actually see.

The spooky part is the product. The scientific part is that we run the same
pipeline three times -- with the sitter's real brain, with somebody else's
brain, and with no brain at all -- and measure how much of the ghost's speech
is actually about the person sitting in front of it.

Everything runs on one consumer laptop. No API key, no account, no data
leaving the machine.

> **Live demo:** [huggingface.co/spaces/redmingwei/seance](https://huggingface.co/spaces/redmingwei/seance)
> **Source:** [github.com/Microbiosis/seance](https://github.com/Microbiosis/seance)

---

## The idea

A séance is a machine for talking to the dead. This one talks to a brain.

The device ("the medium") is deliberately boring: a ridge probe on 630
dimensions of band power and time-windowed amplitude, the same feature
convention the published dataset card uses. It reports one impression --
one of 200 concepts -- and how clear that impression is. The ghost is a local
open-weight language model with a séance persona, and it speaks from that
impression.

The interesting question is not "can a spooky bot be built" (it can, in an
afternoon) but:

> **When the ghost says something true, is it because of the brain?**

That question needs controls, and the controls are the project.

## The audit

For every sitting we run three conditions through the identical pipeline:

| condition | what the medium reports | what it isolates |
|---|---|---|
| `real` | what the probe saw in **this** sitter's EEG | the system as built |
| `swapped` | what the probe saw in **somebody else's** EEG | the effect of the impression word alone |
| `blind` | nothing at all | the model's prior, with no signal |

`real` and `swapped` differ in exactly one variable: the impression. The EEG
is identical in both and the clarity band is held fixed on purpose, so any
difference in the prose is attributable to the sighted image and nothing else.

We run the whole thing at two evidence levels, `K = 1` and `K = 16`: how many
presentations of the same image the medium is allowed to average. It turns
out the séance has a dial.

Three numbers come out:

* **M1 -- can it see?** probe top-1 accuracy on 200 concepts against chance
  and against a shuffled-label control.
* **M2 -- does it say what it saw?** how often the message names the concept
  the sitter actually looked at, per condition.
* **M3 -- is it talking about *you*?** content-word overlap between messages,
  for different sitters, for a swapped impression, and for a dropped
  impression.

## Results

<!-- RESULTS:BEGIN -->

One subject (`sub-01`), one local Gemma 3 4B IT, 60 sittings per
condition, greedy decoding. Everything below is regenerated from
`results/audit.json` by `scripts/fill_readme.py`.

### M1 — can it see?

Top-1 accuracy of the probe on 200 concepts (chance = 0.5%).

| presentations averaged | accuracy | × chance | shuffled-label control |
|---|---|---|---|
| 1 | 4.5% | 9× | 0.5% |
| 2 | 7.5% | 15× | 1.0% |
| 4 | 10.0% | 20× | 0.0% |
| 8 | 21.5% | 43× | 0.0% |
| 16 | 39.5% | 79× | 2.0% |
| 20 | 41.0% | 82× | 1.5% |

A single second of EEG already carries the signal: **4.5%** (9× chance) against a shuffled-label control of **2.0%** — the control sits at chance, so the signal is real and the pipeline
does not leak the labels. Averaging more presentations sharpens the
medium fast: the best level reaches **41.0%**.

### M2 — does it say what the sitter actually saw?

The ghost almost never says the concept out loud — it evokes it
("the heat, copper, burnt sugar" for a blowtorch). So each message is
matched against the **same local model's own 200 concept descriptions**
by content-word overlap: the ghost's own atlas, in the ghost's own
register. Both sides are spoken by the same 4B model, which is what
makes the overlap mean something.

At **K=16** (16 presentations of the same image averaged):

| condition | message matches the concept the sitter saw | 95% CI |
|---|---|---|
| `real` EEG | **16.7%** | 8.3–26.7% |
| `swapped` EEG | 0.0% | 0.0–0.0% |
| `blind` | 0.0% | 0.0–0.0% |

The ghost talks about **the impression it was handed** in 55.0% of sittings — it is faithful. The medium was right about
the concept in 26.7% of sittings (16 of 60).

In 56.2% of *those* sittings the ghost actually spoke about
the impression it was given, which is 15.0% of all
sittings. We measured 16.7%. The gap is one sitting in sixty
where the ghost landed on the true concept even though the medium had
named a different one — luck, not signal.

Swap the brain and the channel is gone; blind the medium and it never
opens. The difference between `real` and `swapped` is the whole
brain-to-prose channel.

### M3 — is it talking about *you*?

Mean content-word overlap between messages (Jaccard):

| pair | overlap |
|---|---|
| two different sitters | 0.107 |
| same sitter, impression swapped | 0.099 |
| same sitter, impression dropped | 0.081 |
| blind to blind | 1.000 |

This is the number we did not expect. Messages from different sitters
are no more alike than messages from the same sitter with a different
impression, and both are close to the impression-dropped baseline: the
prose is essentially a function of the one word the medium handed over.
Blind messages are identical to each other (1.000) because there is
nothing to vary.

**The blunt version: the ghost tells you one word of your brain, and
then it tells you itself.** The EEG contributes one word out of about
37; the rest
is the model's own prior.

### The séance has a dial

Both levels of the same experiment:

| | K=1 (one presentation) | K=16 (sixteen presentations) |
|---|---|---|
| probe top-1 | 4.5% | 39.5% |
| message matches the truth | 1.7% | 16.7% |
| ghost talks about the impression | 61.7% | 55.0% |

More evidence sharpens the *sighting*, not the *talking*. The ghost's
fidelity barely moves; what moves is how often the medium was right.

<!-- RESULTS:END -->

## Why open matters here

* **It has to run offline to be audited.** Swapping the brain signal out and
  re-running the model is only cheap when inference is local. A closed API
  would make the control conditions expensive, rate-limited, and hard to
  reproduce -- the ghost's words would have to be produced by someone else's
  server, on someone else's schedule, from a model version we cannot pin.
* **The open-weight model is the artifact.** Gemma 3 4B IT runs 4-bit NF4 on
  an 8 GB consumer GPU. We use it greedily, deterministically, so every number
  in the report is reproducible.
* **The data never leaves the machine.** EEG recordings are the kind of data
  you do not want shipping to a third party. The whole rig runs on data that
  is already public (THINGS-EEG, CC BY 4.0) and stays local.
* **The probe is inspectable.** It is a closed-form ridge solution with a
  published protocol -- no hidden weights, no training run to reproduce.

## Reproduce

```bash
pip install -r requirements.txt

# the medium only (no language model needed): ~1 minute on CPU
python scripts/run_audit.py --decode-only

# the full audit: downloads ~3 GB of weights, ~30 minutes on one GPU
hf download unsloth/gemma-3-4b-it-bnb-4bit --local-dir models/gemma-3-4b-it-bnb-4bit
python scripts/run_audit.py --model models/gemma-3-4b-it-bnb-4bit --sittings 60

# the semantic metric, then every artefact the console and the post need
python scripts/atlas_metric.py
python scripts/rebuild.py

# the console, locally
streamlit run app/app.py
```

Data comes from `illusion-brain/brain-domains` (THINGS-EEG test split) and is
fetched on first run; set `SEANCE_DATA_DIR` to use a local mirror instead.

### How the assets ship, and why

`app/assets/` holds the probe, the sitting features, the transcripts, the
figures and the full audit result — but **gzipped, base64-encoded and split
into parts**, with `manifest.json` saying how to put them back together.

That is not a stylistic choice. The network this project was uploaded from
drops large request bodies, and it does so unevenly: on the day of the upload a
12 KB body failed while a 67 KB body passed moments later, and a 1 KB body
always passed. Re-measured later the same link carried 250 KB without a
hiccup, so the ceiling moves with the load rather than sitting at one number.
It also mangles git's own push protocol outright, so the repository went up
through the REST API instead of `git push`. The Hub's LFS route (a second,
presigned upload) is unreachable from it too. The alternative was not "ship
the probe", it was "ship no demo".

Two consequences worth knowing:

* The numeric arrays ship as **float16**. That was checked against the
  audited probe, not assumed: float16 reproduces all 60 audited impressions,
  int8 does not (it flips three of them — some top-1/top-2 margins are razor
  thin).
* The packed parts and the originals are both in the repository:
  `results/audit.json` and `results/atlas.json` are committed as plain files
  so a reader can open the numbers the post quotes, and the same two are
  inside `app/assets/audit.p*` and `app/assets/atlas.p*` because the demo
  Space can only read what its own manifest reassembles.
  `scripts/rebuild.py` regenerates the lot.

## Data and credits

* **EEG:** [THINGS-EEG](https://doi.org/10.1016/j.neuroimage.2022.119754)
  (Gifford, Dwivedi & Roig, *NeuroImage* 2022), CC BY 4.0, mirrored at
  [`illusion-brain/brain-domains`](https://huggingface.co/datasets/illusion-brain/brain-domains).
  Ten subjects, 63 electrodes, 250 Hz, 200 test concepts.
* **Model:** [Gemma 3 4B IT](https://ai.google.dev/gemma/docs/core/model_card_3),
  Google's open-weight release. This repository uses the community 4-bit NF4
  quantization (`unsloth/gemma-3-4b-it-bnb-4bit`) so it fits on a consumer GPU.
* **Protocol:** the feature extraction, the repetition split, and the
  decode-versus-chance framing follow the published dataset card for the
  dataset, so the numbers here are comparable with them.

## Honest limitations

* One subject (`sub-01`) for the ghost experiment; the decoder numbers were
  measured on that subject. The evidence curve is not a population statistic.
* The probe is trained on the same 200 images it is tested on (same images,
  new repetitions), so it measures *can the signal carry this concept*, not
  generalization to unseen concepts.
* "Matches the true concept" is matched against the ghost's own atlas of 200
  concept descriptions (content-word overlap). The atlas is written by the same
  4B model, so the two sides share a voice — but the metric still measures
  shared vocabulary, not meaning. A generic sentence encoder was tried and
  rejected: the true concept landed at median rank 110/200.
* The similarity metric is content-word Jaccard. It is a proxy for "same
  words", not a semantic model.

## License

MIT for the code. The data keeps its own CC BY 4.0 license; the model keeps
Google's terms.
