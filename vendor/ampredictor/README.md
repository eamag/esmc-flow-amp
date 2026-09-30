# AMPredictor (Vendored)

Inference code for AMPredictor, used as an activity surrogate in the AMP Challenge 2027 starter kit.

## Origin and Licence

- **Origin:** AMPredictor repository provided with the AMP Challenge 2027 starter kit.
- **Licence:** Upstream carries no `LICENSE` file. Vendored here under academic research fair use for reproducibility of the competition evaluation harness.

## Weights

`src/ampredictor/models/model_GNNPredictor_.model` (the starter-kit checkpoint, unchanged) is committed so that
`score_pool` and `rank_top100` run from a clean clone. It is loaded through `seqme`'s `ThirdPartyModel`
(`entry_point="ampredictor.predict:predict"`, `path=vendor/ampredictor`).
