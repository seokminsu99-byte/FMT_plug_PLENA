# FMT_plug_PLENA

PLENA (Program for Large-scale Evaluation of Network Analysis) is a C++17 implementation of the one-parameter Gibbs model for drainage-network analysis. It generates stochastic drainage-direction networks, evaluates outlet-referenced width functions, and supports beta-class estimation with Nash-Sutcliffe efficiency (NSE).

Current version: `v0.9.1`

## Build

Compile the checked source with a C++17 compiler:

```bash
g++ -O2 -std=c++17 src/main_EN_ver.cpp -o PLENA
```

## Run

Pass an input file on the command line:

```bash
./PLENA example.txt
```

If no path is supplied, PLENA lists eligible text files in the working directory and prompts for a selection. The remaining calculation settings are entered interactively.

## Input format

The plain-text input contains:

1. the number of rows and columns;
2. the one-based outlet row and column; and
3. the drainage-direction matrix.

Direction codes are:

- `0`: inactive cell / no pipe
- `1`: east
- `2`: south
- `3`: west
- `4`: north

See [`example.txt`](example.txt) and [`example_explain.txt`](example_explain.txt).

## Outputs

The primary result text file contains the generated direction matrix, travel-time matrix, flow-accumulation matrix, and `q(t)`. Optional width-function analysis also writes FD and LS matrices, width-function CSV data, and an NSE report for the candidate beta classes.

## SERRA online resources

The repository contains the online resources prepared for *Stochastic Environmental Research and Risk Assessment* (SERRA):

- [`supplementary/ESM_1.pdf`](supplementary/ESM_1.pdf): PLENA implementation details, reproducible workflow, supplementary figures, and computational-performance results.
- [`supplementary/ESM_2.xlsx`](supplementary/ESM_2.xlsx): catchment-level beta classifications for 239 Seoul drainage catchments, comparison with Seo et al. (2024), and candidate beta-specific NSE summaries from the supplied result files.

See [`SUPPLEMENTARY_INFORMATION.md`](SUPPLEMENTARY_INFORMATION.md) for article metadata, captions, data-coverage notes, and the SERRA compliance checklist.

## Citation

Please use the repository metadata in [`CITATION.cff`](CITATION.cff).

## License

This repository is distributed under the [MIT License](LICENSE).

## Provenance and attribution

**Implementation author:** Minsoo Seok (`seokminsu10@yu.ac.kr`)

- The C++ implementation was ported and extended from a MATLAB reference implementation supplied by Seo and Schmidt with permission. Minsoo Seok performed the C++ porting, experiments, analysis, and verification and takes responsibility for the final repository content.
- The pseudorandom number generator follows the `xoshiro256**` reference implementation by David Blackman and Sebastiano Vigna: <https://prng.di.unimi.it/xoshiro256starstar.c>.
- Seed mixing uses an `fmix64`-style finalizer pattern associated with MurmurHash3: <https://github.com/rurban/smhasher/blob/master/MurmurHash3.cpp>.
- A conversational AI tool assisted with the NSE function, debugging suggestions, language polishing, and readability-oriented cleanup. The author reviewed and integrated the resulting code and retains responsibility for it.
