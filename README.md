# FMT_plug_PLENA

PLENA is a C++ implementation of the one-parameter Gibbs model for drainage-network analysis. It generates a new drainage-direction/network matrix from an input matrix and supports downstream width-function estimation and hydrologic-response analysis. The program can be used as a standalone tool and can also be integrated as a plug-in module for the FMT workflow.

Current version: v0.9

## Build

Compile the program with:

```bash
g++ -O2 -std=c++17 src/main_EN_ver.cpp -o PLENA
```

## Usage

The example below assumes that the program reads a single plain-text input file.

```bash
./PLENA example.txt
```

If your current implementation uses a different command-line syntax, adjust the command accordingly.

## Input format

The input must be provided as a plain-text (`.txt`) file.

The input file contains:
1. the number of rows and columns,
2. the outlet row and column, and
3. the drainage-direction matrix.

Direction coding:
- `0` = no pipe
- `1` = East
- `2` = South
- `3` = West
- `4` = North

See `example.txt` for a sample input and `example_explain.txt` for additional explanation.

## Output

The program produces output files in TXT and CSV formats.

The outputs may include:
- a generated drainage-direction/network matrix (`.txt`), and
- analysis-ready output data for downstream processing (`.csv`).

These outputs can be used for width-function computation and hydrologic-response analysis.

## Example

Sample files are provided in the repository root:
- `example.txt`
- `example_explain.txt`

A typical run is:

```bash
./PLENA example.txt
```

## Citation

Please cite this repository using the metadata provided in `CITATION.cff`.

## License

MIT License (see `LICENSE`).

## Provenance / Attribution

**Author:** Seok Minsoo / seokminsu10@yu.ac.kr

### Author responsibility
- The author performed the C++ porting/implementation, experiments, analysis, and verification.
- The author takes full responsibility for the final code, results, interpretations, and manuscript.

### 1) MATLAB -> C++ port (used with permission)
- Core workflow (e.g., `changedirect2`, `loopcheck2`, Gibbs update loop, etc.) was ported to C++ based on a MATLAB reference implementation provided by Seo & Schmidt.
- C++ porting and modifications: Seok Minsoo (seokminsu10@yu.ac.kr).
- This repository includes a derivative implementation based on the provided reference code, used within the scope of the granted permission.

### 2) PRNG (third-party component)
- Uses the `xoshiro256**` reference implementation by David Blackman & Sebastiano Vigna.
- The upstream reference includes a public-domain-style dedication / broad permission notice and an “AS IS” (no-warranty) disclaimer.
- Reference: <https://prng.di.unimi.it/xoshiro256starstar.c>

### 3) Seed mixing (hash finalizer pattern / constants)
- Uses an `fmix64`-style finalizer pattern/constants commonly used with MurmurHash3.
- MurmurHash3 is widely distributed with a public-domain disclaimer in the upstream source header.
- Reference example: <https://github.com/rurban/smhasher/blob/master/MurmurHash3.cpp>

### 4) AI assistance (transparent disclosure)
- The NSE (Nash–Sutcliffe Efficiency) function was written by a conversational AI tool.
- Multithreading: the author proposed the idea/direction. Unless explicitly stated otherwise, implementation/coding is considered integrated and reviewed under the author’s responsibility.
- In addition, limited assistance from a conversational AI tool was used for debugging suggestions, language polishing of text/comments, and code cleanup (readability-oriented refactoring).
