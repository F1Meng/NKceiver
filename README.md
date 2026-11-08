# NKceiver

NKceiver predicts disease class, COVID-19 severity and ICI response from single-cell NK-cell expression profiles, with state-level contributions for each prediction.

## Installation

Use Python 3.11.

```bash
git clone https://github.com/F1Meng/NKceiver.git
cd NKceiver
python -m pip install -r requirements.txt
python -m pip install .
```

## Checkpoints

Trained NKceiver model weights will be made publicly available on Hugging Face upon publication.

Place the model bundle in `resources/weights/`:

```text
resources/weights/
├── annotator.pt
├── disease/
│   ├── bundle.json
│   └── model.ckpt
├── covid/
│   ├── bundle.json
│   └── model.ckpt
└── ici/
    ├── bundle.json
    └── model.ckpt
```

## Prediction

Provide an H5AD file containing one donor's NK cells, raw counts in `X` and gene symbols in `var_names`. The input must contain the bundle's gene panel.

```bash
nkceiver-predict --input donor_NK.h5ad --bundle resources/weights \
  --task ici --counts-key X --output results/donor
```

Choose `disease`, `covid` or `ici` with `--task`. For a count layer, use `--counts-key counts`. For multiple samples from one donor, add `--sample-key sample_id --patient-key donor_id`.

## Citation

If you use NKceiver in your research, please cite our preprint:

> Meng, F., Lin, P., Zhang, S., Li, T., Liu, Z., Ye, P., Zhong, C. & Zhou, Y. **Disease-associated NK-cell state remodeling reveals transcriptional profiles predictive of clinical phenotypes.** *bioRxiv* (2026). [doi:10.64898/2026.10.22.760478](https://doi.org/10.64898/2026.10.22.760478).

<details>
<summary>BibTeX</summary>

```bibtex
@article{meng2026hnka,
  author = {Meng, Fanyi and Lin, Peiran and Zhang, Shunjie and Li, Tianhao and Liu, Zili and Ye, Peng and Zhong, Chao and Zhou, Yuan},
  title = {Disease-associated {NK}-cell state remodeling reveals transcriptional profiles predictive of clinical phenotypes},
  journal = {bioRxiv},
  year = {2026},
  doi = {10.64898/2026.10.22.760478},
  url = {https://doi.org/10.64898/2026.10.22.760478}
}
```

</details>

## License

[MIT](LICENSE).
