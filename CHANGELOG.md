# Changelog

## v0.9.4 - 2026-07-27

This release retains the one-parameter Gibbs model and the Seo-Schmidt MATLAB
lineage while aligning the public C++ transition workflow, documentation, and
SERRA Online Resource 1.

- Uniformly selects one non-outlet active cell and one of its three alternative
  directions before candidate screening.
- Treats locally screened and outlet-unreachable proposals as self-transitions
  and counts every proposal attempt toward `I = alpha * n * m`.
- Applies `min(1, exp(-beta * DeltaH))` as the conditional acceptance
  probability after the candidate has been selected.
- Adds an automatic default seed, optional fixed diagnostic seed, input
  validation, sampler metadata, and a small transition-kernel self-test.
- Updates the build workflow, source documentation, provenance description,
  and Online Resource 1 to match the released implementation.
- Retains the existing author responsibility, software lineage, third-party
  source, and AI-use disclosures.

## v0.9.3 - 2026-07-27

This is a repository-packaging and provenance release. It does not introduce a
new scientific model or change the study's `PLENA_FINAL v0.9` computational
workflow.

- Synchronized the canonical public source with the verified research
  implementation used as the release baseline.
- Restored the research implementation's direct travel-time energy calculation
  and its original input-responsibility boundary with FMT.
- Removed the damaged, duplicate Korean-console source so that the repository
  has one unambiguous build target.
- Added the baseline source hash, software lineage, third-party technical
  sources, author responsibilities, and AI-use disclosure.
- Updated documentation and CI to test the single-run, width-function, and NSE
  batch paths of the canonical source.
- Retained the finalized SERRA online resources without changing their data.
