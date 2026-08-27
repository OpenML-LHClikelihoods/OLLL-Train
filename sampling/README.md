# Sampling

Generates training data for surrogate models of LHC full statistical models.

Given an ATLAS `pyhf` workspace, the sampler walks a random-walk Metropolis
chain over the **per-bin signal yields** and, at every point it visits, records
the yields together with eight likelihood values. One row of the output is
therefore one (yields → likelihood) example: exactly what a surrogate has to
learn.

---

## Contents

| path | what it is |
|---|---|
| `sample.py` | The main script. Run this. |
| `parameters.yaml` | Example input file: global settings plus one document per analysis. |
| `cards/` | One card per analysis — workspace files, channel roles, published background yields, precomputed scan limits. |
| `default_params.py` | Every scan parameter with its default value. |
| `likelihood.py` | The scan itself: model building, the MCMC proposal, the lower-limit probe. |
| `utils.py` | Scan limits, starting points, result merging, metadata. |
| `misc.py` | Logger setup. |
| `name_dict.py` | arXiv number → ATLAS analysis name. |

Workspaces live outside this directory, under `input_folder` (`../data/` by
default), in a subdirectory named after the analysis.

---

## Running the simplest analysis

From this directory:

```bash
python sample.py parameters.yaml --log_dir logs
```

`parameters.yaml` is a multi-document YAML file: the first document holds
settings that apply to every analysis, and each one after it describes a single
analysis. The minimal version is

```yaml
---
# global settings, applied to every analysis document below
analyses      : ['1911.06660']
input_folder  : '../data/'
output_folder : '../tables/'
processes     : 1
scans         : 1
points        : 100
---
# one document per analysis
analysis : '1911.06660'
include  : cards/1911.06660.yaml
```

That is the whole input. `include` pulls in the card, which supplies the
workspace files, the channel roles and the background yields; anything not set
anywhere falls back to `default_params.py`. `analyses` selects which of the
documents actually run — use `'all'` for every one of them, or list the ones you
want, so a file can carry many analyses and you enable them a few at a time.

Both arguments are optional: `sample.py` reads `parameters.yaml` and logs to
`logs/` when they are omitted.

### What you get

Under `output_folder`, one directory per scan (`1911.06660-0`, `-1`, …), each
holding a CSV and its metadata JSON:

```
QCR1cut_cuts-0,QCR2cut_cuts-0,SR1cut_cuts-0,...,nLL_exp_mu0,nLL_exp_mu1,nLL_obs_mu0,nLL_obs_mu1,nLLA_...
+69.47934011,  +29.37824469,  +22.64258016, ...,+107.44972418,+115.00416987,+108.53965958,+111.00750596,...
```

One column per bin, holding the **total yield** (background + sampled signal),
followed by the eight likelihood columns: negative log-likelihood at `mu=0` and
`mu=1`, expected (`exp`, Asimov data) and observed (`obs`), each also in its
asymptotic form (`nLLA_*`). The JSON beside it records the settings, the scan
box and the per-column extrema.

---

## The parameters worth knowing

Everything has a default; these are the ones that change what you get.

| parameter | meaning |
|---|---|
| `scans`, `points` | Number of chains and points per chain. `processes` runs chains in parallel. |
| `SR_sigma`, `CR_sigma`, `VR_sigma` | Proposal step size, as a fraction of each bin's range. |
| `start_method` | `'random'`, `'gauss'`, `'edges'` or `'default'` (zero signal). |
| `scan_criterion` | Which likelihood the chain accepts on; `'mu1'` picks one of the `nLL_*_mu1` at random per scan. |
| `sig_rel_unc` | Relative uncertainty on the injected signal. Set it in the **card**, not here — the card wins. |
| `signal_leakage_CR/VR` | Whether signal may appear in control/validation regions; `*_spread` sets how much. |
| `removeCRsVRs`, `remove_channels` | Drop channels from the fit entirely. Requires signal leakage off. |
| `low_lim_samples` | Bisection steps in the lower-limit probe. See below. |
| `bkg_unc_samples` | Background-uncertainty samples used when building the model. |

---

## Scan limits, and why the first run can be slow

Each bin is scanned over a box. The upper end is a closed-form estimate. The
lower end starts from a closed-form candidate and is then **probed**: signal is
injected and the likelihood evaluated, because a background subtraction that is
too aggressive makes the model undefined. That probe is by far the most
expensive part of a scan — hours on a large workspace.

The cards ship with the answer precomputed, so a normal run just loads it:

```
Scan limits mode: LOADED from the parameter card (5 bins, computed for sig_rel_unc=0.0,
CR/VR spread 0.1/0.1). Skipping the lower-limit probe.
```

The card records what those limits were computed with in `scan_limits_context`,
and the sampler checks your run against it. Change `sig_rel_unc`, or widen a
leakage spread, and the stored limits no longer describe your run, so it works
them out itself:

```
Not using the card's scan limits for patchset.json (card->run: sig_rel_unc 0.0->0.2).
Computing them now, which can be slow. See README to store them in the card.
```

Two settings are reported but do **not** discard the stored limits:

- `low_lim_samples` — how finely the lower limit is searched for. Asking for
  more than the card was built with only warns: a coarser search returns a limit
  that is conservative, never unsafe.
- `removeCRsVRs` / `remove_channels` — the lower-limit search always evaluates
  the full model, so recomputing after a removal returns the same numbers. Note
  that the scan itself *does* drop the channels, so the limits were established
  on a slightly larger model than the one being sampled.

### Keeping limits for a configuration you use often

The recomputed values are printed as the `Scan limits` table and saved in the
run's metadata JSON as `lower_limits` / `upper_limits`. To stop paying for them
every time, copy them into the card:

1. Run once with your settings and let the sampler compute the limits.
2. Open the run's `metadata.json` and take `lower_limits` and `upper_limits` —
   they are TOTAL yields, in the card's bin order, exactly the form the card
   wants.
3. Replace the `scan_limits` entry for that patchset with `[min, max]` pairs
   built from them, one per bin.
4. Update `scan_limits_context` to the settings you just ran with, so the
   sampler can confirm the two match.

Steps 3 and 4 go together: a `scan_limits` block that does not agree with its
context is the one thing the sampler cannot detect.

---

## Cards

A card describes one analysis. The essentials:

```yaml
bkgfiles  :                       # background-only workspace, one per patchset
    - Region-combined/BkgOnly.json
patchsets :                       # signal patchsets
    - Region-combined/patchset.json
channels  :                       # channel -> role, in WORKSPACE order
    -   QCR1cut_cuts : 'CR'
        SR1cut_cuts  : 'SR'
bkg_yields :                      # published background, one entry per BIN
    - 73.0
bkg_unc :                         # its uncertainty, same order
    - 8.0
```

Two things to watch:

- `bkg_yields` and `bkg_unc` are indexed **per bin**, not per channel. A channel
  split into three bins needs three entries.
- The order is the **workspace's** channel order, which is often not the order
  the paper's table uses. Reorder when copying published numbers; getting this
  wrong puts one region's uncertainty on another and silently distorts its scan
  box.

Control and validation regions are centred on their *observed* count with a
range of `± obs * spread`, so their `bkg_yields` entries never reach the scan
limits — they appear only in the printed table and the metadata.

Everything below the `# --- scan limits` marker is the precomputed scan box and
the settings it belongs to. Edit it only as described above — the two blocks
have to stay consistent with each other.

### A card always wins

The card is merged **on top of** the analysis document, so a value set in both
places takes the card's. The sampler warns when a document sets a key the card then overrides.

---

## Author

Rafał Maselek
[rafal.maselek@ijs.si](mailto:rafal.maselek@ijs.si)
ORCID [0000-0002-5558-8249](https://orcid.org/0000-0002-5558-8249)

