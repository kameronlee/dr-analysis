# Retinal vessel density, graph fragility, and endpoint stability

Kameron Lee, Thomas Jefferson High School for Science and Technology

This study uses 295 OCTA-500 retinal angiograms to ask whether simulated breaks in the vessel network help distinguish diabetic retinopathy from healthy eyes, and how reliably different tracings preserve apparent vessel endings. Adding graph fragility to the baseline model changed cross-validated AUROC from 0.901 to 0.898. Endpoint density was sensitive to how much vessel the segmentation retained.

[Read the manuscript (PDF)](paper_materials/W3PHIAI-27/paper.pdf) · [Manuscript source](paper_materials/W3PHIAI-27/paper.tex) · [Figure inputs](paper_materials/W3PHIAI-27/figures/plot_data)

## Reproduce the figures

From the repository root, install the project in a Python 3.11 environment and run the two plotting scripts. The four figures can be regenerated directly from the included inputs.

```bash
python -m pip install -e '.[dev]'
python paper_materials/W3PHIAI-27/figures/make_fragility.py
python paper_materials/W3PHIAI-27/figures/build_figures.py
python -m pytest tests/
```

## Build the manuscript

With pdfLaTeX, BibTeX, latexmk and the newtx fonts installed, run the following from the repository root.

```bash
cd paper_materials/W3PHIAI-27
latexmk -pdf -halt-on-error paper.tex
```

## Analysis and data

The Python package implements segmentation, graph extraction, segment removal and the statistical models. The scripts directory contains the endpoint, image-quality and perturbation analyses. The manuscript’s analysis directory includes the paired estimates, distance-matched dilation results, and the code used to calculate them. The crosscheck files contain subsequent exploratory comparisons of interior endpoint rankings across tracers and discrimination by scan field.

Rerunning the scan-level analyses requires the separately distributed OCTA-500 scans and vessel labels. The verification and recount scripts read the original project manifest, saved graphs and intermediate measurements under the paths listed in the source index. [Numerical source index](paper_materials/W3PHIAI-27/analysis/quantity_sources.csv) · [OCTA-500 dataset](https://ieee-dataport.org/open-access/octa-500)

Figure 2 uses a cropped and enlarged OCTA-500 angiogram with added vessel boundaries, centerlines, endpoints and scale bars. The original angiogram and labels are credited to Mingchao Li, Songtao Yuan and Qiang Chen under CC BY 4.0. [Dataset publication](https://doi.org/10.1016/j.media.2024.103092) · [License](https://creativecommons.org/licenses/by/4.0/) · [Figure provenance](paper_materials/W3PHIAI-27/figures/plot_data/retinal_tracing_provenance.json)
