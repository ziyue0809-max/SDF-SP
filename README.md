# SDF-SP

SDF-SP combines a semantically conditioned dual-order spatiotemporal encoder with
retrieval-based forecast correction. A frozen GPT-2 backbone produces the base
forecast, which is refined through global and node-level simplicial projections
and adaptive fusion.

## Setup

```sh
pip install -r requirements.txt
jupyter lab SDF_SP.ipynb
```

Place the GPT-2 configuration and weights in `pretrained/gpt2/`. Set the data and
model paths in the first cell, then run the notebook in order.

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
    model.safetensors
SDF_SP.ipynb
```

GPT-2 weights in `pytorch_model.bin` format can also be used.

## Data

The notebook reads prepared training, validation, and test files with the same
array structure:

| Key | Shape | Description |
| --- | --- | --- |
| `x` | `[S, N, T]` | Historical observations |
| `y` | `[S, N, H]` | Forecast targets |
| `event` | `[S, 4096]` | Context embedding for each sample |
| `future_time` | `[S, H]` | Forecast timestamps as Unicode strings or `datetime64` |

Here, `S` is the number of samples, `N` the number of nodes, `T` the input length,
and `H` the forecast horizon. The defaults are `N=79`, `T=24`, and `H=8`.
Observations and targets use their original measurement units.

```python
np.savez_compressed(
    "data/train.npz",
    x=x_train,
    y=y_train,
    event=context_train,
    future_time=forecast_timestamps_train.astype("U32"),
)
```

Save validation and test data in the same format. Keep the node order consistent
across the data, graph, and semantic embeddings.

| File | Shape | Description |
| --- | --- | --- |
| `adjacency.npy` | `[N, N]` | Nonnegative graph adjacency weights |
| `node_embeddings.npy` | `[N, 4096]` | Node semantic embeddings |
| `semantic_adjacency.npy` | `[N, N]` | Pairwise cosine similarities of the node embeddings |

The model adds self-loops and normalizes the graph. The diagonal of the semantic
similarity matrix is set to zero.

Calendar features in `make_time_features` use 15-minute intervals, 36 daily slots,
and a 09:00 start. Adjust this function for a different sampling schedule.

## Training and inference

The base model uses Huber loss and Adam with a learning rate of `1e-3` and batch
size 24. Training runs for up to 2500 epochs, with early stopping after 50 epochs
without improvement in validation loss.

After training, the notebook builds a residual memory from the training data and
retrieves six candidates per query. Global and node-level projections use local
vertices, mutual 3-nearest-neighbor edges, and triangular faces, with a radius
factor of 2.

Fusion weights depend on each projection's mean absolute change from the base
forecast. For each sample and node, the global weight is `dn / (dg + dn)`, where
`dg` and `dn` are the global and node-level correction magnitudes. The node weight
is its complement. Both weights are 0.5 when the sum is at most `1e-20` and are
shared across forecast steps.

## Results

Each run saves its checkpoint, retrieval memory, predictions, fusion weights, and
metrics under `outputs/<timestamp>/`.

In `metrics.json`, `Original_Base` is the base forecast and `Original_Base_New_RAG`
is the final adaptive-fusion forecast. `Original_SP`, `Weighted_global_hard`, and
`Node_hard` report the unscaled global, scaled global, and node-only projections.
Metrics are reported for the first four steps and all eight steps. WMAPE is
computed as `100 * sum(abs(pred - y)[y != 0]) / sum(y)`.
