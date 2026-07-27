# Software provenance and AI-use disclosure

## Research implementation

The canonical public source is [`src/main_EN_ver.cpp`](src/main_EN_ver.cpp).
Its computational body is synchronized with the authors' `PLENA_FINAL v0.9`
research implementation. Two independently stored local copies of that source
were compared before this release and were byte-identical:

```text
SHA-256: 47D6D69204D2DD895759CCF1823FB4D687C84E8369BC74155EA6F34DDC4AE87A
```

Relative to that baseline, the public source changes only the attribution
header, adds explicit standard-library includes for portability, normalizes one
comment to ASCII, and terminates the file with a newline. The Gibbs update, reachability test, energy
calculation, scale-proportional update count, random-number workflow, width
function, NSE calculation, batch execution, and output logic are unchanged.

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
released PLENA source assumes that direction codes follow the documented
`0`-to-`4` convention and that every active input cell reaches the active
outlet. During stochastic generation, PLENA independently checks every proposed
network state and rejects proposals that break outlet reachability.
