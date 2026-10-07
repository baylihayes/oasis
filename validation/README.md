# OASIS regression checks

Tools to check that the current OASIS code still reproduces the reference
version (`oasis-reference` tag, the code before the MDPL2 refactor), and that
the tiled pipeline reproduces the normal mini-box pipeline. See plan §15.2–15.3.

## Quick start

From the repository root:

```bash
bash validation/run_regression.sh small ~/oasis_regression            # under a minute
bash validation/run_regression.sh small ~/oasis_regression --tiled    # + tiled check
bash validation/run_regression.sh dense ~/oasis_regression --tiled    # ~10 minutes, several GB
```

The script exports the reference code from the `oasis-reference` tag (no second
clone needed), generates the regression box once, runs the reference and the
current code on it, and compares them. It ends with `REGRESSION: PASS` or
`REGRESSION: FAIL` (exit code 0 or 1). Set `THREADS=16` to use more workers.

It makes three comparisons:

1. **reference vs current**, with the current code run with
   `seed_tie_break='load_order'` (the reference's order for seeds with equal
   M200b). Must pass in `--mode reference`.
2. **(with `--tiled`) current mini-box run vs current tiled run**, both with
   the default `seed_tie_break='halo_id'`. Must pass in `--mode tiled`.
3. **Informational:** how many haloes the default tie-break changes compared
   with the reference order. Never fails; it records the effect of that
   deliberate change.

Run it after every change to the `oasis` package, before committing.

## Files

| File | What it does |
|---|---|
| `make_regression_box.py` | Generates the synthetic 100 Mpc/h box (`small` or `dense`). Fixed random seeds: always the same box, byte for byte. |
| `run_pipeline.py` | Runs one OASIS version end to end on a box: mini-box split, self-calibration data, catalogue (optionally tiled with `--tiles N`). |
| `compare_runs.py` | Compares two runs, halo by halo, and prints a verdict. |
| `run_regression.sh` | All of the above in one command. |
| `padding_convergence.py` | Runs one box with increasing padding (= ribbon width) and reports which haloes still change (plan §6, §15.4, Milestone E.2). |

## What the box contains

NFW-like haloes with infall shells, subhaloes that should and should not be
absorbed (6D ball), close pairs that compete in percolation, fake seeds with
almost no particles, quantized masses (so M200b ties occur), haloes on
mini-box and periodic edges, and a uniform background.

| Box | Particles | Seeds | Haloes found (n_orb_min = 20) |
|---|---|---|---|
| small | 4,108,512 | 1,375 | 858 |
| dense | 59,136,085 | 21,246 | 15,472 |

## Reading the comparison

Each check prints `OK`, `ACCEPTED` or `FAIL`:

- **OK**: identical.
- **ACCEPTED**: a known, intended difference (listed below). Values are still
  checked, e.g. member lists are compared per halo.
- **FAIL**: an unexplained difference. Do not merge until it is understood.

### Accepted differences, reference vs current (`--mode reference`)

| Difference | Reason |
|---|---|
| `Morb` within 1e-6 relative | The reference summed N float32 copies of the particle mass; the current code computes N × m_p directly (Milestone B). |
| `Halo_ID`, `PID`, `LIDX`, `RIDX`, `SLIDX`, `SRIDX` stored as int64 | 32-bit types would overflow for MDPL2-sized catalogues (fixes 1–2). Values are identical. |
| `LIDX`/`RIDX`/`SLIDX`/`SRIDX` values not compared | The reference merged mini-box files in filesystem order, so member positions differ. The members of every halo are compared instead. |
| `memb/Halo_ID` integer instead of float64 | The reference produced float64 by accident (concatenating empty lists). Values are identical. |

### Accepted differences, mini-box vs tiled run (`--mode tiled`)

| Difference | Reason |
|---|---|
| Member order within a halo | Follows the order particles were loaded (mini-box files vs tile cells), which is not science. Members are compared per halo as sets. |
| Member ID dtypes | Tiles keep the input ID type; mini-box files use the smallest unsigned type. Values are identical. |

### Ties in mass (`seed_tie_break`)

The processing order decides which halo gets first claim on shared particles,
and in two places that order needs a rule for exact ties:

1. **Seeds with equal M200b.** Seeds are classified from most to least
   massive; equal M200b is common when M200b is a whole number of particle
   masses.
2. **Haloes with equal Morb.** Haloes enter percolation from most to least
   orbiting mass; with constant particle mass Morb = Norb × m_p, so equal Morb
   is common.

- `seed_tie_break='halo_id'` (default): equal M200b by increasing Halo_ID,
  and equal Morb kept in that seed order (stable sort). The catalogue then
  does not depend on how the data is stored or how much is loaded around each
  region (mini-box files, mini-box size, tiles, padding).
- `seed_tie_break='load_order'`: equal M200b in the order the data source
  returns them, and pandas' default (unstable) sort for Morb. This is the
  reference code's behaviour; for mini-box files it reproduces the reference
  exactly, which is why check 1 uses it.

Why this matters:
- With the reference's seed order, the dense box gave 78 of 15,472 haloes
  (2 haloes swapped, 74 with a different Norb) that differed between a
  mini-box run and a tiled run, even though both read identical seeds and
  particles. Running the mini-box data in the tile's seed order reproduced the
  tiled result exactly, so seed tie order was the only cause.
- With the unstable Morb sort, the order of two haloes with equal Morb
  depended on how many other haloes the region held, so 2-12 haloes changed
  between paddings on Quijote even deep inside mini-boxes (e.g. a pair 1.5
  Mpc/h apart swapping 10 shared particles). With the stable sort, paddings
  5 to 20 give identical catalogues (see `padding_convergence.py`).

Check 3 reports how many haloes the default rule changes compared with the
reference's order.

Everything else (all other catalogue columns, halo set, member sets,
substructure lists, mini-box files, calibration data) must be identical.

## Running pieces by hand

```bash
python validation/make_regression_box.py small /path/box.npz
python validation/run_pipeline.py <package_dir> /path/box.npz /path/run_a
python validation/compare_runs.py /path/run_a /path/run_b --mode exact
```

`<package_dir>` is the folder that contains the `oasis` package: the
repository root for the current code, or an exported copy of any version, e.g.
`git archive <commit> oasis | tar -x -C <dir>`. Each run also writes
`times.json`, and current versions of OASIS write `run_reg/timings.json` with
per-stage time and memory.

## Adding a new accepted difference

If a change is meant to alter outputs, add the difference and its reason to
`compare_runs.py` (docstring and the matching check) and to the table above in
the same commit. That keeps every intended change to the science documented.
