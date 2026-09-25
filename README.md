# SDF-SP

`SDF_SP.ipynb` contains the semantic dual-order base forecaster, training-only
retrieval memory, global and node-level simplicial projections, and statistical
adaptive fusion. The notebook contains code only, with no comments, saved cell
outputs, or print statements. Artifacts are saved to a separate output directory
for each run.

## Setup

Install Python 3.10 or later and the dependencies in `requirements.txt`:

```sh
pip install -r requirements.txt
jupyter lab SDF_SP.ipynb
```

Place a locally loadable Hugging Face GPT-2 model (configuration and weights) in
`pretrained/gpt2/`, or change `GPT2_PATH` in the first cell. Loading is local-only.
The intended backbone is GPT-2 with its native causal attention and frozen parameters.
No datasets or pretrained weights are included.

Run the notebook from this directory, from the first cell to the last:

```text
data/
  train.npz
  val.npz
  test.npz
  adjacency.npy
  node_embeddings.npy
  semantic_adjacency.npy
pretrained/
  gpt2/
    config.json
    model.safetensors (or pytorch_model.bin)
SDF_SP.ipynb
```

## Prepared input datasets

Each `.npz` file must contain the following arrays. `S` can differ between files;
node order, node count, embedding dimensions, and forecast horizon must agree.

| Key | Shape | Contents |
| --- | --- | --- |
| `x` | `[S, N, T]` | Finite historical observations in original measurement units |
| `y` | `[S, N, H]` | Finite future targets in the same units |
| `event` | `[S, 4096]` | Context embedding aligned to each sample's forecast origin |
| `future_time` | `[S, H]` | Known forecast timestamps, stored as Unicode ISO strings or `datetime64` |

The default configuration is `N=79`, `T=24`, `H=8`. Change the first cell if needed.
The loader reads these arrays directly; it does not perform custom input validation.
The projection memory requires at least six training samples. Object arrays and
pickled dictionaries are not supported. Save already prepared arrays as follows:

```python
np.savez_compressed(
    "data/train.npz",
    x=x_train,
    y=y_train,
    event=context_train,
    future_time=forecast_timestamps_train.astype("U32"),
)
```

Use the same keys for `val.npz` and `test.npz`. Supply per-sample embeddings directly;
the notebook does not repeat daily embeddings or look them up by original-series indices.

The notebook performs no partitioning, day slicing, or sliding-window generation.
Prepare train, validation, and test datasets upstream using the intended temporal
protocol. Targets must remain within their assigned partition; any preceding
historical context must predate the forecast origin. Supplying three separate files
does not itself prove that the datasets are chronological or disjoint. No original
series identifiers are assumed and no identifier-based overlap mask is applied.
Validation selects the base checkpoint; test labels are used only for evaluation.

## Graph and semantic inputs

| File | Shape | Contents |
| --- | --- | --- |
| `adjacency.npy` | `[N,N]` | Finite nonnegative base graph weights; self-loops and symmetric degree normalization are applied in the model |
| `node_embeddings.npy` | `[N,4096]` | Finite node semantic vectors, aligned to the node axis in every dataset |
| `semantic_adjacency.npy` | `[N,N]` | Cosine similarities of the original node semantic vectors; diagonal is set to zero by the model |

Context embeddings must contain only information available at forecast time.
Construct any data-derived graph or preprocessing statistics without future labels.

## Retained method and settings

- Shared temporal/spatial operators with TS and ST branches, semantic feature/edge
  modulation, and context-conditioned branch fusion.
- Temporal/spatial tokenization and a frozen GPT-2 backbone.
- Huber loss, Adam at `1e-3`, batch size 24, at most 2500 epochs, and early stopping
  after 50 epochs without validation-loss improvement.
- Training-only residual memory `y - x_last`, eight historical statistics per node
  plus eight calendar features per forecast step, and training-memory standardization.
- Six retrieved candidates, mutual 3-nearest-neighbor edges, radius factor 2, and
  local vertex/edge/triangle projection at global and node levels.
- For each sample and node, `dg = mean(abs(global - base))` and
  `dn = mean(abs(node - base))` over the forecast horizon. Global weight is
  `dn / (dg + dn)`; node weight is its complement. A denominator at most `1e-20`
  uses equal weights. There are no label-fitted fusion coefficients.

Calendar features retain the source implementation's 15-minute daytime convention:
36 slots per day with a 09:00 origin. For datasets with another cadence, adapt
`make_time_features` consistently before running experiments. MAE and RMSE use all
targets. The retained WMAPE convention excludes zero-target errors from its numerator;
its denominator requires a nonzero sum of targets. This is not a universal metric
definition for signed or zero-sum datasets.

## Saved artifacts

Each `outputs/<timestamp>/` directory contains the best base checkpoint, training
memory, checkpoint/memory hashes, predictions, targets, per-sample/per-node fusion
weights, and metrics. `Original_Base_New_RAG` is the adaptive-fusion result;
`Original_SP` is the retained unscaled global-only comparison. Artifact key names
are preserved for compatibility. The notebook does not print training logs or scores.

Only code and documentation are included in this release folder. A full training
run with the intended data and pretrained checkpoint is required to reproduce results.
