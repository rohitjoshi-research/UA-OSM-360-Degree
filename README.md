# UA-OSM: Uncertainty-Aware Object-Shaped Merging

**Reducing object loss in 360° surround-view stitching — simulation study, code and data**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GITHUB_USER/UA-OSM/blob/main/notebooks/UA-OSM_validation.ipynb)
[![DOI](https://doi.org/ZENODO_DOI](https://zenodo.org/records/22992593)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> Between two stools, one falls to the ground. Between two cameras, a child can fall off the screen.

## The problem
A surround-view system stitches four fisheye cameras into one top-down image. Upright objects are stretched radially away from each camera, in a different direction per camera. In the overlap at each vehicle corner, a merge seam that falls between the two projections crops both copies, so an object seen by two cameras can be shown by neither.

## What this repository contains
* **Feasibility analysis** of straight seams (closed-form condition, infeasible-zone maps and sensitivity sweeps).
* **UA-OSM**: each object's complete footprint is assigned to one camera that detects it, dilated by tracker covariance, with joint camera assignment for all objects in a corner and a five-level escalation ladder ending in a driver warning.
* **Baselines and ablations**: fixed seam, dynamic straight seam, curved mask-avoiding seam, one-camera-per-object, and UA-OSM without joint assignment / visibility / dilation / prediction.
* **Model-independent evaluation** on ray-traced pixels, with bootstrap confidence intervals, Holm-corrected paired tests and effect sizes.
* **Scenario catalog**: 4 times of day × 3 weather states × 1–5 objects.
* **Validation notebook** for Google Colab reproducing every result, and the scripts that generated the demo video.

## Key results (simulation)
| Method | Frames with a lost object (pixel-level) |
|---|---|
| Fixed 45° seam | 63.0 % |
| One camera per object (naive) | 27.6 % |
| Curved mask-avoiding seam | 24.0 % |
| Dynamic straight seam | 19.9 % |
| **UA-OSM** | **7.0 %** |

* Five objects at one corner: 45.7 % → 20.0 %.
* Better than the dynamic straight seam in all 12 time-of-day and weather conditions (4.5–13.0 % vs. 22.5–27.0 %).
* With seam angles limited to 15–75°, 91 % of positions within 0.5 m of the corner cannot be preserved by any straight seam.

**Limitations, stated up front:** all merging results are simulated (flat-ground top view, articulated-cylinder pedestrians); weather and lighting act through assumed perception degradation; the driver warning fires too often (up to 40 % of frames at night in fog); validation on real synchronized four-camera recordings is still required.

## Quick start
**Colab (recommended):** click the badge above, then *Runtime → Run all*. `QUICK = True` runs in about 15–20 minutes; `QUICK = False` uses the paper's scene counts (several hours).

**Local:**
```bash
pip install -r requirements.txt
cd src
python -c "import e6; print(e6.run_px(methods=['dynamic','uaosm+vis'], sc0=0, nsc=5, kset=[1,2,3,4,5], sigma=0.08, lat=0.1, pdrop=0.1).keys())"
```

## Repository layout
| Path | Content |
|---|---|
| `src/sim2.py` | Camera model, ray tracer, top-view projection, footprint model, merging methods, ladder, metrics |
| `src/e2.py`, `src/e5.py`, `src/e6.py` | Perception model, random scenes, pixel-level benchmark harness |
| `src/e9.py`, `src/cond.py` | Robustness harness and scenario catalog |
| `src/e3.py`, `src/scen.py` | Static rendering helpers and pedestrian appearance |
| `notebooks/` | Colab validation notebook |
| `results/` | Raw result files behind every table and figure (see `results/README.md`) |
| `figures/` | Paper figures |
| `video/` | Scripts that render the demo video |

## Citation
If you use this work, please cite the paper and this software (see `CITATION.cff`):
```bibtex
@misc{joshi2026uaosm,
  author = {Joshi, Rohit},
  title  = {{UA-OSM}: Uncertainty-Aware Object-Shaped Merging to Reduce Object Loss in 360° Surround-View Stitching},
  year   = {2026},
  doi    = {ZENODO_DOI},
  url    = {https://github.com/GITHUB_USER/UA-OSM}
}
```

## Licence
Code: MIT (see `LICENSE`). Result data and figures: CC BY 4.0.

## Acknowledgements
Built entirely on open-source software: Python, NumPy, SciPy, OpenCV, Matplotlib, Pillow and FFmpeg. Generative AI tools assisted with drafting, simulation code and analysis; all content was reviewed and verified by the author.
