# thumbwalk

A [Gymnasium](https://gymnasium.farama.org/) environment where the learnable agent is the
**viewer**, not the recommender.

## The inversion

[RecSim](https://github.com/google-research/recsim) (Google, 2019) and
[SARDINE](https://github.com/naver/sardine) (Naver, 2024) simulate a *user*, and the learnable
agent is the *recommender*: it proposes a slate of items to that simulated user and learns to
maximize engagement.

`thumbwalk` flips this. The recommender is a **fixed, non-learnable two-stage funnel** — part
of the environment's dynamics, like physics in a robotics sim. The learnable agent is the
**viewer**: on every step it sees one "playing" thumbnail (the current seed) plus a slate of
related thumbnails, and takes exactly one of **5 x slate_size** actions (see "Action space"
below). You train agents that model *how people watch*, not how to recommend to them.

The recommender ([`thumbwalk/recommender.py`](thumbwalk/recommender.py)) is modeled on
Covington, Adams & Sargin, *"Deep Neural Networks for YouTube Recommendations"* (2016):

1. **candidate generation** — with watch history, retrieve the nearest neighbors of a
   *viewing profile* (a watch-fraction-weighted average of up to the last 50 watched items,
   Covington et al.'s history window) in embedding space (optimizes recall). With no history
   yet (cold start, e.g. the very first slate of an episode), falls back to
   popularity-weighted sampling, the demographic-prior stand-in Covington et al. use for new
   users; this is also what anchors the episode's initial slate.
2. **ranking** — scores candidates by profile affinity and popularity, demotes items that have
   been shown repeatedly without being watched (sec. 4.1 of Covington et al.), applies the two
   negative-feedback suppressions below, and injects 1-2 exploration items sampled from
   farther away (optimizes precision, adds serendipity/freshness).

Two explicit negative-feedback actions follow Ricks & McCrosky, *"Does This Button Work?"*
(2022), the only published measurement found of YouTube's own controls' effect:

- **"No me interesa"** soft-suppresses the marked item's k-NN neighborhood (~11% drop in
  similar recommendations, `not_interested_factor` default `0.89`).
- **"No recomendar canal"** soft-suppresses every item sharing the marked item's *channel*
  (~43% drop, the more effective control, `channel_suppression_factor` default `0.57`).

Both are multiplicative and floor-clamped (never reach exact zero), matching the report:
neither control ever achieves full suppression.

## Package layout

```
thumbwalk/
  corpus.py        # loads a preprocessed corpus (embeddings, clusters, channels, popularity)
  recommender.py   # the fixed two-stage funnel: candidate_generation() + rank()
  click_model.py   # configurable "simulated viewer" click models for baselines
  env.py            # the Gymnasium environment: ThumbwalkEnv
scripts/
  preprocess.py     # folder of images -> corpus + variety/noise analysis
examples/
  random_agent.py   # random-acting agent, prints a trajectory
tests/
```

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .
```

### 1. Preprocess a folder of images into a corpus

```bash
python scripts/preprocess.py --input path/to/thumbnails --output corpus/ --embedder clip --k 15
```

This embeds every image, clusters the embeddings (KMeans with an auto-chosen k, plus HDBSCAN
for outlier detection), builds a k-NN index, infers a **channel** id per image from its
top-level subfolder (one folder per YouTube channel, e.g. `thumbnails/MrBeast/...`; images with
no subfolder each get their own singleton channel), and writes everything
`thumbwalk.corpus.Corpus` needs to `corpus/`. It also writes `corpus/analysis/`: a 2D
projection colored by cluster, an entropy histogram, an entropy-vs-isolation scatter with
outliers flagged, per-cluster contact sheets, a "noisiest images" montage, and a
`summary.json` — enough to visually confirm the corpus has genuine variety (distinct clusters)
*and* some noisy/outlier images.

Popularity defaults to uniform (no view-count metadata is read from a folder of images alone);
pass a real popularity signal into `Corpus(..., popularity=...)` directly if your dataset
carries one, to get a non-trivial cold-start / demographic-prior effect.

`--embedder` accepts `clip` (default, needs `open_clip_torch` or `transformers`+`torch`),
`resnet` (needs `torchvision`), or `autoencoder` (a zero-extra-dependency PCA embedder that
always works, useful without a deep-learning stack installed). If the requested embedder's
optional dependencies aren't installed, preprocessing automatically falls back down that list
with a warning, so the script always runs.

### 2. Run an agent against it

```bash
python examples/random_agent.py --corpus-dir corpus/
```

Or skip preprocessing entirely and try the environment with a small in-memory synthetic corpus:

```bash
python examples/random_agent.py
```

This runs a uniform-random VIEWER policy for one episode and prints each step: the action
taken (watch with a time fraction, "No me interesa", or "No recomendar canal", and which slate
position), the resulting seed, the running count of suppressed items, and the reward.

### 3. Use it in your own training loop

```python
from thumbwalk.corpus import Corpus
from thumbwalk.env import ThumbwalkEnv, ThumbwalkEnvConfig

corpus = Corpus.load("corpus/")

def my_reward_fn(context: dict) -> float:
    # context: seed_id_before, action, action_type, position, clicked_id,
    #          watch_fraction, is_watch, is_not_interested, is_no_channel,
    #          invalid_action, masked_count
    return 1.0 if context["is_watch"] else 0.0

env = ThumbwalkEnv(corpus, ThumbwalkEnvConfig(
    slate_size=10,
    k=15,
    exploration_slots=1,
    history_window=50,              # Covington et al.'s last-50-watches window
    not_interested_factor=0.89,     # ~11% drop (Ricks & McCrosky 2022)
    channel_suppression_factor=0.57,  # ~43% drop, the more effective control
    horizon=50,
    reward_fn=my_reward_fn,
))

obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

`thumbwalk` ships **no** reward logic of its own beyond a `0.0` placeholder — `reward_fn` is
required for anything but the placeholder, by design, so the environment stays useful for
whatever your research actually needs to measure (entropy/diversity, engagement, regret,
fairness...).

## Action space

On every step the viewer takes exactly **one** decision among `Discrete(5 * slate_size)`
actions, encoding a slate position `k` and an action type `j` as `a = j * slate_size + k`:

| `j` | Action | Changes seed? | Effect |
|---|---|---|---|
| 0 | watch, abandon | yes -> `slate[k]` | implicit signal, watch_fraction=0.0 |
| 1 | watch, half | yes -> `slate[k]` | implicit signal, watch_fraction=0.5 |
| 2 | watch, full | yes -> `slate[k]` | implicit signal, watch_fraction=1.0 |
| 3 | "No me interesa" on `slate[k]` | no | suppresses `slate[k]`'s k-NN neighborhood |
| 4 | "No recomendar canal" on `slate[k]` | no | suppresses `slate[k]`'s whole channel |

Watching and picking a watch-time fraction are a single step on purpose: between clicking a
thumbnail and starting to watch, the agent gains no new observation (it already saw the
thumbnail in the slate), so splitting it into two steps would just lengthen episodes for free.
Negative-feedback actions are, for the same reason, an *alternative* to watching rather than a
separate pre-step: marking consumes one of the episode's fixed number of decisions exactly like
watching does, so it has a real opportunity cost instead of being free to spam.

## Observation space

- **Observation** (`gymnasium.spaces.Dict`): `seed_embedding`, `seed_id`, `slate_embeddings`
  (shape `(N, D)`), `slate_ids`, and `slate_mask` (1.0 = real item, 0.0 = padding if the
  recommender's reachable pool has shrunk below the slate size). Only embeddings and integer
  ids are in the observation, so the env is fully usable headless; image paths and string ids
  are available in `info` for a renderer.
- **Action** (`Discrete(5 * N)`): see table above.

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

Tests run against a small synthetic corpus (planted Gaussian clusters + outlier points, grouped
into multi-item channels) and require only `numpy`/`scikit-learn`/`gymnasium` — none of the
optional embedder/clustering/projection dependencies.

## License

MIT — see [LICENSE](LICENSE).
