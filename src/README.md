# PLENA source

[`main_EN_ver.cpp`](main_EN_ver.cpp) is the sole canonical source and public
build target for repository release `v0.9.4`.

The source retains the `PLENA_FINAL v0.9` and Seo-Schmidt MATLAB lineage while
implementing the manuscript's public transition workflow: uniform random-scan
cell-direction proposals, candidate screening, outlet-reachability checks,
self-transitions for rejected proposals, conditional energy-based acceptance,
and the scale-proportional update budget `I = alpha * n * m`. See
[`../PROVENANCE.md`](../PROVENANCE.md) for software lineage, third-party
sources, author responsibilities, and the existing AI-use disclosure.
