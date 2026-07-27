# FMT_plug_PLENA

[![Build and smoke test](https://github.com/seokminsu99-byte/FMT_plug_PLENA/actions/workflows/build.yml/badge.svg)](https://github.com/seokminsu99-byte/FMT_plug_PLENA/actions/workflows/build.yml)

PLENA (Program for Large-scale Evaluation of Network Analysis) is a C++17 implementation of the one-parameter Gibbs model for stochastic drainage-network analysis. It generates outlet-connected drainage trees, calculates outlet-referenced width functions, and supports beta-class estimation using Nash-Sutcliffe efficiency (NSE).

Current release: `v0.9.2`

## Scientific scope

PLENA provides a structural comparison of drainage networks. The estimated beta class and width function do not independently represent rainfall forcing, conduit capacity, storage, pump operation, downstream water level, inundation depth, or flood probability. Hydraulic interpretation requires separate event-based hydrologic and hydraulic analysis.

## Repository contents

| Path | Purpose |
| --- | --- |
| [`src/main_EN_ver.cpp`](src/main_EN_ver.cpp) | Canonical English source and public build target |
| [`src/main.cpp`](src/main.cpp) | Korean-console variant with different diagnostic and output text |
| [`example.txt`](example.txt) | Synthetic 51 x 51 example input |
| [`example_explain.txt`](example_explain.txt) | Input-field and execution explanation |
| [`supplementary/ESM_1.pdf`](supplementary/ESM_1.pdf) | SERRA Online Resource 1 |
| [`supplementary/ESM_2.xlsx`](supplementary/ESM_2.xlsx) | SERRA Online Resource 2 |

The documented release workflow and automated checks use `src/main_EN_ver.cpp`. The two source variants should not be mixed when comparing diagnostic text or generated filenames.

## Build

A C++17 compiler is required.

```bash
g++ -O2 -std=c++17 src/main_EN_ver.cpp -o PLENA
```

On Windows with MinGW, use `-o PLENA.exe`. The repository workflow compiles and smoke-tests the canonical source on Ubuntu.

## Run

```bash
# Linux or macOS
./PLENA example.txt

# Windows
PLENA.exe example.txt
```

If no path is supplied, PLENA lists eligible text files in the working directory. The beta exponent, optional width-function analysis, candidate range, number of stochastic realizations, and thread cap are entered interactively.

## Input format and validation

The plain-text input contains:

1. the number of rows and columns;
2. the one-based outlet row and column; and
3. the drainage-direction matrix.

Direction codes are `0` = inactive/no pipe, `1` = east, `2` = south, `3` = west, and `4` = north. PLENA rejects values outside `0`-`4`, an inactive outlet, and any input network in which an active cell cannot reach the specified outlet. See [`example.txt`](example.txt) and [`example_explain.txt`](example_explain.txt).

## Analysis configuration

The release source uses the scale-proportional stochastic update budget `I = alpha * n * m` with `alpha = 10`, matching the study configuration. Because Gibbs-model networks are stochastic, beta estimation is based on ensemble mean width-function summaries rather than a single generated network.

## Outputs

The primary result text file contains the generated direction matrix, outlet-distance matrix, flow-accumulation matrix, and `q(t)`. Optional width-function analysis writes FD and LS matrices, a width-functions CSV file, and an NSE report containing candidate-class summaries.

## SERRA online resources

The repository supports the article *Scalable Stochastic Characterization of Urban Drainage Networks: Applying PLENA to 239 Seoul Catchments* submitted to *Stochastic Environmental Research and Risk Assessment*:

- [`ESM_1.pdf`](supplementary/ESM_1.pdf), Online Resource 1: PLENA implementation details and reproducible computational workflow, including Figures S1-S3 and additional computational-performance results.
- [`ESM_2.xlsx`](supplementary/ESM_2.xlsx), Online Resource 2: best beta estimates by catchment and all candidate-specific NSE summaries for 239 Seoul drainage catchments, including comparison with Seo et al. (2024).

The municipal drainage-network inputs are not redistributed because of applicable data-use restrictions. The public workbook contains the catchment-level results and candidate summaries described in the manuscript. See [`SUPPLEMENTARY_INFORMATION.md`](SUPPLEMENTARY_INFORMATION.md) for the exact captions and article metadata.

## Citation and license

Use the metadata in [`CITATION.cff`](CITATION.cff). The repository is distributed under the [MIT License](LICENSE).

## Provenance and attribution

**Implementation author:** Minsoo Seok (`seokminsu10@yu.ac.kr`)

- The C++ implementation was ported and extended from a MATLAB reference implementation supplied by Seo and Schmidt with permission. Minsoo Seok performed the C++ porting, experiments, analysis, and verification and takes responsibility for the repository content.
- The pseudorandom number generator follows the `xoshiro256**` reference implementation by David Blackman and Sebastiano Vigna: <https://prng.di.unimi.it/xoshiro256starstar.c>.
- Seed mixing uses an `fmix64`-style finalizer pattern associated with MurmurHash3: <https://github.com/rurban/smhasher/blob/master/MurmurHash3.cpp>.
- A conversational AI tool assisted with the NSE function, debugging suggestions, language polishing, and readability-oriented cleanup. The author reviewed and integrated the resulting code and retains responsibility for it.
