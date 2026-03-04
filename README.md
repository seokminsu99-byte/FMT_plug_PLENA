# FMT_plug_PLENA
This program takes an input matrix and applies a one-parameter Gibbs model to generate and return a new (direction/network) matrix. The output can be used for downstream width-function computation and hydrologic-response analysis, and it can also be integrated as an extension module for the (currently unpublished) FMT tool.
/*
FMT_plug tool - PLENA
ver. 0.9

PROVENANCE / ATTRIBUTION

Author: Seok Minsoo / seokminsu10@yu.ac.kr

Author responsibility
- The author performed the C++ porting/implementation, experiments, analysis, and verification.
- The author takes full responsibility for the final code, results, interpretations, and manuscript.

1) MATLAB -> C++ port (used with permission)
- Core workflow (e.g., changedirect2 / loopcheck2 / Gibbs update loop, etc.) was ported to C++ based on a MATLAB reference implementation provided by Seo & Schmidt.
- C++ porting and modifications: Seok Minsoo (seokminsu10@yu.ac.kr).
- This repository includes a derivative implementation based on the provided reference code, used within the scope of the granted permission.

2) PRNG (third-party component)
- Uses the xoshiro256** reference implementation by David Blackman & Sebastiano Vigna.
- The upstream reference includes a public-domain-style dedication / broad permission notice and an “AS IS” (no-warranty) disclaimer.
- Reference: https://prng.di.unimi.it/xoshiro256starstar.c

3) Seed mixing (hash finalizer pattern / constants)
- Uses an fmix64-style finalizer pattern/constants commonly used with MurmurHash3.
- MurmurHash3 is widely distributed with a public domain disclaimer in the upstream source header.
- Reference example: https://github.com/rurban/smhasher/blob/master/MurmurHash3.cpp

4) AI assistance (transparent disclosure)
- The NSE (Nash–Sutcliffe Efficiency) function was written by a conversational AI tool.
- Multithreading: the author proposed the idea/direction. (Unless explicitly stated otherwise, implementation/coding is considered integrated and reviewed under the author’s responsibility.)
- In addition, limited assistance from a conversational AI tool was used for debugging suggestions, language polishing of text/comments, and code cleanup (readability-oriented refactoring).

5) Additional modifications by the author (examples)
- Batch execution (task scheduling, progress reporting), file I/O and file-selection UI, robust failure handling (e.g., CHANGE_MAX_TRIES), result writing (txt/csv), and reproducibility-related logging.



BUILD: g++ -O2 -std=c++17 src/main.cpp -o PLENA
- 

