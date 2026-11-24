# Learned turbulence closure

| File | Format and use |
| --- | --- |
| `closure.mp4` | H.264, 1440 × 1560, 24 fps, 10.71 s, fast start |
| `closure.webm` | VP9, same resolution, rate and duration |
| `closure_poster.png` | 1440 × 1560 PNG for the video poster |
| `closure_still.png` | 2880 × 3120 PNG for print or PDF |

## Reading the panels

- **Top left, filtered DNS.** Vorticity from the DNS, sharply filtered to the coarse grid after the current step.
- **Top right, hybrid rollout.** Vorticity at the same time from a free rollout started at the filtered DNS initial condition. The CNN corrects the coarse solver after every step.
- **Bottom left, DNS-implied correction.** The vorticity increment that would bring a single coarse step from the previous filtered DNS state to the next filtered DNS state.
- **Bottom right, CNN correction.** The vorticity increment the CNN predicts for that same coarse step. Red adds positive vorticity and blue adds negative vorticity in both correction panels.

For a filtered DNS transition `wₙ → wₙ₊₁`, let `b = S(wₙ)` be the output of one locked coarse solver step. The correction panels show `wₙ₊₁ − b` and `dt × CNN(b)`. Both start from the same field, so differences between the correction panels come from the one-step prediction. The DNS-implied increment is the residual of the discrete coarse solver and includes any discretisation differences between the coarse and fine steps. The top-right rollout feeds its own state back into the solver.

The two vorticity maps show the result of the step illustrated below them. The animation samples the forward rollout through step 500, pauses briefly at the endpoints and fades through white to restart. The poster and print still show step 150.

## Sources and scales

- **Data.** Locked validation seed `10000`, the first validation seed and the same one used in the existing browser walkthrough. Its provenance is in `MAIN/notebooks/public/rollout_seed10000.json`. The longer trajectory is generated on demand with [the repository DNS driver](../src/tesseract_hybrid_closure/data.py), using the settings saved in the selected checkpoint.
- **Checkpoint.** `MAIN/runs/w2-calibrated-a20-dt002-100x3/stage-unroll-30-updates-700.pkl`, selected in `MAIN/runs/final-submission/selection.json`. The renderer verifies SHA-256 `8ed5b36fac902e61fcd3a1749f727f6928c93f3b7303e5781f5f66ce86c2e9b7` against the checked-in [integrity manifest](../docs/results/final-metrics.json) before loading. The complete selection and test protocol is in [the verification record](../docs/VERIFICATION.md).
- **Numbers on the frames.** The vorticity colourbar runs from −11 to +11. The per-step increment colourbar runs from −0.04 to +0.04. For each row, the script pools absolute coarse-grid values across its two fields and all 232 displayed steps, takes the 99th percentile and rounds up to the next 1 for vorticity or 0.005 for the increment. Both maps in a row use the same symmetric zero-centred `RdBu_r` scale throughout playback and in both PNGs. Values beyond the endpoints use the end colours.

## Regenerate

Run from the root of this worktree. `MAIN` is the sibling clone containing the saved checkpoint. FFmpeg must provide `libx264` and `libvpx-vp9`. The renderer downloads the Regular and Medium weights of IBM Plex Sans at a pinned source revision into a temporary directory. [IBM Plex source](https://github.com/IBM/plex/tree/ba3013d72a8419a22c758cc51c44e04ac7ea565b/packages/plex-sans)

```sh
uv sync --locked
uv run --locked python scripts/portfolio_media/closure.py \
  --main "$PWD/../tesseract-hybrid-closure-agent"
```

## Suggested website caption

This validation rollout compares the learned and DNS-implied vorticity corrections for a hybrid that reduced 500-step rollout MSE across 32 test trajectories by 71.7% relative to the coarse solver and 15.7% relative to a matched a-priori CNN.

The caption's aggregate results come from the sealed test evaluation in [the verification record](../docs/VERIFICATION.md).

## Last session (2026-11-24)

Regenerated on CPU from base revision `e690f11` with the renderer added in this branch. The final render took 69 seconds. The 30-step hybrid prefix matched the saved walkthrough arrays exactly, and the largest absolute filtered DNS difference was `1.15e-5` or less. The poster and still were inspected at native size and at 360 or 720 px wide. Both videos decoded fully, passed the size limit and returned to near-white at the loop boundary. MP4 fast start was verified from the atom order.

Repository-wide Ruff lint and format checks passed, along with 40 tests from `test_data.py`, `test_configs.py`, `test_notebook_demo.py` and `test_submission_assets.py`. The training and container smokes were outside the render scope. The media are ready for site integration. No jobs are in flight.
