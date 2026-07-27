# Software provenance and AI-use disclosure

## Public implementation

The canonical public source is [`src/main_EN_ver.cpp`](src/main_EN_ver.cpp),
released as repository version `v0.9.4`. It retains the authors'
`PLENA_FINAL v0.9` and Seo-Schmidt MATLAB lineage while expressing the
transition workflow documented in the manuscript and algorithm figure.

```text
SHA-256: 5BDB1776ECB3795BCBED4BA79CD076A34567AFBF648A8574CD97B2B673F70723
```

For each stochastic update, the public source uniformly selects one non-outlet
active cell and one of the three flow directions other than its current
direction. Boundary, inactive-cell, immediate two-cycle, and outlet-unreachable
proposals leave the current state unchanged as self-transitions. Every proposal
attempt contributes to `I = alpha * n * m`. For an outlet-reachable proposal,
conditional acceptance uses `min(1, exp(-beta * DeltaH))`; the preceding
cell-direction selection supplies the proposal-probability factor `1/r` in the
total transition rule, where `r = 3*N_u` and `N_u` is the number of non-outlet
active cells.

The source includes `--self-test`, which enumerates the valid states of a small
2 x 2 network and checks transition-row sums, preservation of the Gibbs target
distribution, and detailed balance to a numerical tolerance of `1e-12`.

## Scientific and software lineage

PLENA implements the one-parameter Gibbs drainage-network model described by
Troutman and Karlinger (1992) and follows the urban-drainage application and
width-function workflow developed in Seo and Schmidt (2014) and Seo et al.
(2015).

The core `changedirect2`, `loopcheck2`, and Gibbs-update workflow was ported
with permission from a MATLAB reference implementation provided by Yongwon Seo
and A. R. Schmidt. Yongwon Seo is the third author of the accompanying study.
Minsoo Seok performed the C++ porting and implementation and led the software
experiments, analysis, and verification. Changmin Park, Minsoo Seok, and
Yongwon Seo retain responsibility for the released software and the associated
scientific claims.

Relevant publications:

- Troutman BM, Karlinger MR (1992) Gibbs' distribution on drainage networks.
  *Water Resources Research* 28:563-577.
  <https://doi.org/10.1029/91WR02648>
- Seo Y, Schmidt AR (2014) Application of Gibbs' model to urban drainage
  networks: a case study in southwestern Chicago, USA. *Hydrological
  Processes* 28:1148-1158. <https://doi.org/10.1002/hyp.9657>
- Seo Y, Hwang J, Noh SJ (2015) Analysis of urban drainage networks using
  Gibbs' Model: a case study in Seoul, South Korea. *Water* 7:4129-4143.
  <https://doi.org/10.3390/w7084129>

## Third-party technical components

- The pseudorandom-number generator follows the `xoshiro256**` reference
  implementation by David Blackman and Sebastiano Vigna:
  <https://prng.di.unimi.it/xoshiro256starstar.c>.
- Seed mixing uses an `fmix64`-style finalizer pattern and constants associated
  with MurmurHash3. A reference implementation is available at
  <https://github.com/rurban/smhasher/blob/master/MurmurHash3.cpp>.

## AI-assisted development

Conversational AI tools, including OpenAI Codex, were used in a supporting
role. The NSE routine was initially drafted with conversational-AI assistance.
AI tools also provided debugging suggestions, English-language polishing,
readability-oriented refactoring suggestions, repository documentation, and
release-quality checks. The multithreading concept and development direction
were specified by the author.

All AI-assisted material was reviewed, tested, and integrated by the authors.
AI tools did not qualify for authorship, are not cited as scientific evidence,
and do not bear responsibility for the software, data, analysis, or
interpretation. Those responsibilities remain with the human authors.

## Input responsibility

The research workflow uses FMT to prepare and validate PLENA inputs. The
released PLENA source also checks the documented `0`-to-`4` direction range,
outlet activity, and initial outlet reachability before stochastic generation.
During generation, PLENA independently checks every proposed network state and
rejects proposals that break outlet reachability.
