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
**viewer**: on every step it sees one "playing" thumbnail (the *seed*) plus a slate of related
thumbnails, and chooses to click one (which becomes the new seed) or mark one "not interested"
(which suppresses its neighborhood from future recommendations). You train agents that model
*how people watch*, not how to recommend to them.

The recommender ([`thumbwalk/recommender.py`](thumbwalk/recommender.py)) is modeled on
Covington, Adams & Sargin, *"Deep Neural Networks for YouTube Recommendations"* (2016):

1. **candidate generation** — retrieve the K nearest neighbors of the seed in embedding space
   (optimizes recall).
2. **ranking** — score and order those candidates into the final slate, injecting 1-2
   exploration items sampled from farther away (optimizes precision, adds serendipity).

The "not interested" mechanic follows Liu, Wu & Resnick (2024), who found that the real
YouTube "Not interested" button suppresses a *neighborhood* around the disliked item, not just
the item itself — and that suppression is never total. By default `thumbwalk` down-weights
(rather than deletes) a disliked cluster or k-NN neighborhood; pass `hard_mask=True` for full
exclusion instead.

## Package layout

```
thumbwalk/
  corpus.py        # loads a preprocessed corpus (embeddings, clusters, metadata)
  recommender.py   # the fixed two-stage funnel: candidate_generation() + rank()
  click_model.py   # configurable "simulated viewer" click models for baselines
  env.py            # the Gymnasium environment: ThumbwalkEnv
scripts/
  preprocess.py     # folder of images -> corpus + variety/noise analysis
examples/
  random_agent.py   # random-clicking agent, prints a trajectory
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
for outlier detection), builds a k-NN index, and writes everything `thumbwalk.corpus.Corpus`
needs to `corpus/`. It also writes `corpus/analysis/`: a 2D projection colored by cluster, an
entropy histogram, an entropy-vs-isolation scatter with outliers flagged, per-cluster contact
sheets, a "noisiest images" montage, and a `summary.json` — enough to visually confirm the
corpus has genuine variety (distinct clusters) *and* some noisy/outlier images.

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

This runs a uniform-random VIEWER policy for one episode and prints each step: the action taken
(click vs. not-interested, and which slate position), the resulting seed, the running count of
suppressed items, and the reward.

### 3. Use it in your own training loop

```python
from thumbwalk.corpus import Corpus
from thumbwalk.env import ThumbwalkEnv, ThumbwalkEnvConfig

corpus = Corpus.load("corpus/")

def my_reward_fn(context: dict) -> float:
    # context: seed_id_before, action, clicked_id, is_click,
    #          is_not_interested, invalid_action, masked_count
    return 1.0 if context["is_click"] else 0.0

env = ThumbwalkEnv(corpus, ThumbwalkEnvConfig(
    slate_size=10,
    k=15,
    exploration_slots=1,
    hard_mask=False,       # soft, partial suppression by default
    mask_scope="cluster",  # or "knn"
    horizon=50,
    reward_fn=my_reward_fn,
))

obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

`thumbwalk` ships **no** reward logic of its own beyond a `0.0` placeholder — `reward_fn` is
required for anything but the placeholder, by design, so the environment stays useful for
whatever your research actually needs to measure (engagement, diversity, regret, fairness...).

## Observation / action spaces

- **Observation** (`gymnasium.spaces.Dict`): `seed_embedding`, `seed_id`, `slate_embeddings`
  (shape `(N, D)`), `slate_ids`, and `slate_mask` (1.0 = real item, 0.0 = padding if the
  recommender's reachable pool has shrunk below the slate size). Only embeddings and integer
  ids are in the observation, so the env is fully usable headless; image paths and string ids
  are available in `info` for a renderer.
- **Action** (`Discrete(2 * N)`): actions `0..N-1` click the slate item at that position;
  actions `N..2N-1` mark the item at position `action - N` "not interested".

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

Tests run against a small synthetic corpus (planted Gaussian clusters + outlier points) and
require only `numpy`/`scikit-learn`/`gymnasium` — none of the optional embedder/clustering/
projection dependencies.

## License

MIT — see [LICENSE](LICENSE).
