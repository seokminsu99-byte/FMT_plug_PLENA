# FMT_plug_PLENA

[![Build and smoke test](https://github.com/seokminsu99-byte/FMT_plug_PLENA/actions/workflows/build.yml/badge.svg)](https://github.com/seokminsu99-byte/FMT_plug_PLENA/actions/workflows/build.yml)

PLENA (Program for Large-scale Evaluation of Network Analysis) is a C++17
implementation of the one-parameter Gibbs model for stochastic structural
analysis of drainage networks. It generates feasible drainage-tree states,
calculates outlet-referenced width functions, and supports beta-class
estimation using the Nash-Sutcliffe efficiency (NSE).

**Repository release:** `v0.9.4`<br>
**Research lineage:** `PLENA_FINAL v0.9` and the Seo-Schmidt MATLAB reference implementation

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
| [`example.txt`](example.txt) | Synthetic 51 x 51 input |
| [`example_explain.txt`](example_explain.txt) | Input contract and execution guide |
| [`supplementary/ESM_1.pdf`](supplementary/ESM_1.pdf) | SERRA Online Resource 1 |
| [`supplementary/ESM_2.xlsx`](supplementary/ESM_2.xlsx) | SERRA Online Resource 2 |

Only one C++ source is designated as canonical. Generated executables, build
directories, temporary files, and manuscript working files are not committed.

## Build

A C++17 compiler is required.

```bash
g++ -O2 -std=c++17 -pthread src/main_EN_ver.cpp -o PLENA
```

On Windows with MinGW, use `-o PLENA.exe`. The GitHub Actions workflow builds
the same source on Ubuntu and exercises both the single-run and batch/NSE paths.

The source also includes a small transition-kernel diagnostic:

```bash
./PLENA --self-test
```

## Run

```bash
# Linux or macOS
./PLENA example.txt

# Windows
PLENA.exe example.txt
```

If no input path is supplied, PLENA lists eligible `.txt` files in the working
directory. It then requests the beta exponent, the update coefficient `alpha`,
and, when selected, the width-function and batch-analysis settings. A fixed
seed may be supplied as a second command-line argument for an exactly
repeatable diagnostic run.

## Input contract and FMT boundary

The plain-text input contains:

1. the number of rows and columns;
2. the one-based outlet row and column; and
3. the drainage-direction matrix.

Direction codes are `0` = inactive/no pipe, `1` = east, `2` = south, `3` =
west, and `4` = north. The input contract requires an active outlet and every
active cell to reach that outlet.

In the study workflow, FMT prepares and validates this input before PLENA is
run. The public source also checks matrix dimensions and completeness,
direction codes, outlet bounds, outlet activity, and initial outlet
reachability. During stochastic generation, PLENA checks each proposed state
and rejects any proposal in which an active cell cannot reach the outlet.

## Analysis configuration

Each generated network uses the scale-proportional update count
`I = alpha * n * m`. Entering `0` at the prompt selects the study setting
`alpha = 10`. The batch interface initially offers candidate exponents
`k = -4, ..., 3` and `100` stochastic realizations per candidate; both the
exponent range and realization count can be changed interactively. A thread
cap of `0` requests automatic selection from `hardware_concurrency()`.

At each update, PLENA uniformly selects one non-outlet active cell and one of
the three directions other than its current direction. Boundary, inactive-cell,
immediate two-cycle, and outlet-unreachable proposals are rejected as
self-transitions, and every proposal attempt counts toward `I`. An
outlet-reachable candidate is accepted with the conditional probability
`min(1, exp(-beta * DeltaH))`; the cell-direction selection supplies the
proposal-probability factor `1/r` in the total transition rule, where
`r = 3*N_u` and `N_u` is the number of non-outlet active cells.

The default seed combines hardware entropy with clock-based values. Individual
generated networks are therefore stochastic and are not expected to be
byte-for-byte identical across executions. Beta estimation is based on ensemble
width-function summaries rather than one generated network.

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
