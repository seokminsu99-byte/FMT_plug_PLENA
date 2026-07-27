# SERRA Online Resources

Official journal guidance: [Stochastic Environmental Research and Risk Assessment - Submission guidelines](https://link.springer.com/journal/477/submission-guidelines)

## Article metadata

- **Article title:** Scalable Stochastic Characterization of Urban Drainage Networks: Applying PLENA to 239 Seoul Catchments
- **Journal:** Stochastic Environmental Research and Risk Assessment
- **Authors:** Changmin Park, Minsoo Seok, Yongwon Seo
- **Affiliation:** Department of Civil Engineering, Yeungnam University, Gyeongsan, Republic of Korea
- **Corresponding author:** Yongwon Seo (`yongwon.seo@yu.ac.kr`)

## Prepared files

| File | Manuscript label | Concise caption |
| --- | --- | --- |
| [`supplementary/ESM_1.pdf`](supplementary/ESM_1.pdf) | Online Resource 1 | PLENA implementation details and reproducible computational workflow. This PDF contains supplementary figures, descriptions of the input-file structure and flow-direction coding, compilation and execution procedures, candidate generation and acceptance rules, output-file descriptions, implementation notes, and additional computational-performance results. |
| [`supplementary/ESM_2.xlsx`](supplementary/ESM_2.xlsx) | Online Resource 2 | Catchment-level beta estimates and NSE results for the 239 drainage catchments in Seoul. This spreadsheet separates the best beta estimate for each drainage catchment from the NSE summaries for all candidate beta classes and includes comparisons between PLENA-based estimates and previous MATLAB-based estimates. |

## Code identity and provenance

- [`src/main_EN_ver.cpp`](src/main_EN_ver.cpp) is the sole canonical source for
  repository release `v0.9.4` and retains the authors' `PLENA_FINAL v0.9` and
  Seo-Schmidt MATLAB lineage.
- FMT prepares and validates the drainage-direction inputs used in the study;
  the public PLENA source also validates the input and checks outlet
  reachability for every proposed stochastic state.
- The MATLAB-to-C++ lineage, third-party technical components, author
  responsibilities, and existing AI-assisted development boundaries are
  documented in [`PROVENANCE.md`](PROVENANCE.md).

## Online Resource 1 figure inventory

- **Figure S1:** contiguous matrix storage and cache locality.
- **Figure S2:** parallel PLENA execution and processor utilization.
- **Figure S3:** generation-time comparison between the MATLAB-based Gibbs-model implementation and PLENA across the tested network sizes.

## Online Resource 1 implementation coverage

- Each stochastic update uniformly selects one non-outlet active cell and one
  of the three alternative flow directions.
- Boundary, inactive-cell, immediate two-cycle, and outlet-unreachable
  proposals are retained as self-transitions; every proposal attempt counts
  toward `I = alpha * n * m`.
- For an outlet-reachable proposal, the conditional acceptance probability is
  `min(1, exp(-beta * DeltaH))`. The proposal selection supplies the separate
  `1/r` factor in the general transition rule, where `r = 3*N_u` and `N_u` is
  the number of non-outlet active cells.
- Independent beta-class and replicate tasks are parallelized; updates within
  an individual stochastic chain remain sequential.

## Online Resource 2 contents

- `Best Beta Results` contains all 239 catchments in ID order, including 237 valid PLENA classifications and 2 inputs without a valid classification.
- The final PLENA distribution is: `10^-4` = 11, `10^-3` = 84, `10^-2` = 122, `10^-1` = 17, `10^0` = 1, `10^1` = 2, and not classified = 2.
- `All Beta NSE` reports `k`, `beta`, mean NSE, and `NSE(mean_q)` for every available candidate class.
- The comparison field is labeled `Seo et al. (2024)` throughout.

Comparison reference:

> Seo Y, Kim KJ, Jeong WC (2024) Network characteristics and its impact on flooding in urban catchments: A case study in Seoul, South Korea. *Journal of Korea Water Resources Association* 57:1221-1230. <https://doi.org/10.3741/JKWRA.2024.57.S-1.1221>

## SERRA compliance checklist

| Requirement | Status |
| --- | --- |
| Supplementary files use consecutive `ESM_n` names and are cited as `Online Resource n`. | `ESM_1.pdf` and `ESM_2.xlsx` correspond to Online Resources 1 and 2. |
| Text and presentation-style supplementary content is supplied as PDF. | Implementation notes and supplementary figures are combined in `ESM_1.pdf`. |
| Spreadsheet data uses an accepted spreadsheet format. | Catchment and NSE data are supplied in `ESM_2.xlsx`. |
| Each file includes the article title, journal, authors, affiliation, and corresponding-author email. | Metadata appear on page 1 of `ESM_1.pdf` and in the `Metadata` sheet of `ESM_2.xlsx`. |
| Each file has a concise manuscript caption. | Captions are recorded above and match the intended Online Resource numbering. |
| Supplementary figures have descriptive captions. | Figures S1-S3 have captions that explain the visual content and its relationship to PLENA. |
| Published files must be self-contained because supplementary information is published as received. | Definitions, workflow notes, source links, and column meanings are included in the files. |

## Interpretation boundary

PLENA beta classes and width functions are structural indicators of outlet-referenced flow-distance organization. They do not independently represent rainfall forcing, pipe capacity, pumps, downstream water levels, surface storage, inundation depth, or flood probability.
