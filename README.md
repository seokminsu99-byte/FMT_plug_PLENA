# FMT_plug_PLENA

[![Build and smoke test](https://github.com/seokminsu99-byte/FMT_plug_PLENA/actions/workflows/build.yml/badge.svg)](https://github.com/seokminsu99-byte/FMT_plug_PLENA/actions/workflows/build.yml)

PLENA (Program for Large-scale Evaluation of Network Analysis) is a C++17
implementation of the one-parameter Gibbs model for stochastic structural
analysis of drainage networks. It generates feasible drainage-tree states,
calculates outlet-referenced width functions, and supports beta-class
estimation using the Nash-Sutcliffe efficiency (NSE).

**Repository release:** `v0.9.3`<br>
**Synchronized research implementation:** `PLENA_FINAL v0.9`

## Scientific scope

PLENA provides structural comparisons of outlet-referenced flow-distance
patterns. A beta class or width function does not independently represent
rainfall forcing, conduit capacity, storage, pump operation, downstream water
level, inundation depth, or flood probability. Event-based hydrologic and
hydraulic analysis is required for design or rehabilitation decisions.

## Repository contents

| Path | Purpose |
| --- | --- |
| [`src/main_EN_ver.cpp`](src/main_EN_ver.cpp) | Canonical source and public build target |
| [`PROVENANCE.md`](PROVENANCE.md) | Software lineage, baseline hash, third-party sources, and AI-use disclosure |
| [`CHANGELOG.md`](CHANGELOG.md) | Release history and scope of the v0.9.3 synchronization |
| [`example.txt`](example.txt) | Synthetic 51 x 51 input |
| [`example_explain.txt`](example_explain.txt) | Input contract and execution guide |
| [`supplementary/ESM_1.pdf`](supplementary/ESM_1.pdf) | SERRA Online Resource 1 |
| [`supplementary/ESM_2.xlsx`](supplementary/ESM_2.xlsx) | SERRA Online Resource 2 |

Only one C++ source is designated as canonical. Generated executables, build
directories, temporary files, and manuscript working files are not committed.

## Build

A C++17 compiler is required.

```bash
g++ -O2 -std=c++17 src/main_EN_ver.cpp -o PLENA
```

On Windows with MinGW, use `-o PLENA.exe`. The GitHub Actions workflow builds
the same source on Ubuntu and exercises both the single-run and batch/NSE paths.

## Run

```bash
# Linux or macOS
./PLENA example.txt

# Windows
PLENA.exe example.txt
```

If no input path is supplied, PLENA lists eligible `.txt` files in the working
directory. It then requests the beta exponent and, when selected, the
width-function and batch-analysis settings.

## Input contract and FMT boundary

The plain-text input contains:

1. the number of rows and columns;
2. the one-based outlet row and column; and
3. the drainage-direction matrix.

Direction codes are `0` = inactive/no pipe, `1` = east, `2` = south, `3` =
west, and `4` = north. The input contract requires an active outlet and every
active cell to reach that outlet.

In the study workflow, FMT prepares and validates this input before PLENA is
run. PLENA checks dimensions, matrix completeness, outlet bounds, and whether
the outlet is active, but the synchronized `PLENA_FINAL v0.9` loader does not
repeat FMT's direction-range or initial full-network reachability validation.
During stochastic generation, PLENA does check each proposed state and rejects
any proposal in which an active cell cannot reach the outlet.

## Analysis configuration

Each generated network uses the scale-proportional update count
`I = alpha * n * m`, with `alpha = 10` in this release. The batch interface
initially offers candidate exponents `k = -4, ..., 3` and `100` stochastic
realizations per candidate; both the exponent range and realization count can
be changed interactively. A thread cap of `0` requests automatic selection from
`hardware_concurrency()`.

The base random seed is derived from the system clock. Consequently, individual
generated networks are stochastic and are not expected to be byte-for-byte
identical across executions. Beta estimation is based on ensemble width-function
summaries rather than one generated network.

## Outputs

The primary result text file contains the generated direction matrix,
outlet-distance matrix, flow-accumulation matrix, and `q(t)`. Optional
width-function analysis writes FD and LS matrices, a width-functions CSV file,
and an NSE report with candidate-class summaries.

## SERRA online resources

This repository supports the article *Scalable Stochastic Characterization of
Urban Drainage Networks: Applying PLENA to 239 Seoul Catchments*, submitted to
*Stochastic Environmental Research and Risk Assessment*.

- [`ESM_1.pdf`](supplementary/ESM_1.pdf) documents the PLENA implementation,
  workflow, supplementary figures, and additional performance results.
- [`ESM_2.xlsx`](supplementary/ESM_2.xlsx) separates the best beta result for
  each of 239 Seoul drainage catchments from all available candidate-specific
  NSE summaries and compares PLENA results with Seo et al. (2024).

Municipal drainage-network inputs are not redistributed because of applicable
data-use restrictions. See
[`SUPPLEMENTARY_INFORMATION.md`](SUPPLEMENTARY_INFORMATION.md) for the online
resource captions and inventory.

## Provenance, AI use, citation, and license

The C++ implementation was ported with permission from the MATLAB reference
implementation provided by Yongwon Seo and A. R. Schmidt. Yongwon Seo is the
third author of the accompanying study; Minsoo Seok led the C++ implementation.
Conversational AI tools, including OpenAI Codex, assisted with the NSE routine,
debugging suggestions, language polishing, readability-oriented cleanup, and
repository documentation. The human authors reviewed the released work and
retain full responsibility. Detailed boundaries and third-party technical
sources are recorded in [`PROVENANCE.md`](PROVENANCE.md).

Use [`CITATION.cff`](CITATION.cff) when citing the software. Repository content
is distributed under the [MIT License](LICENSE).
