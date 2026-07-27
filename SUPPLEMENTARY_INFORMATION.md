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
| [`supplementary/ESM_1.pdf`](supplementary/ESM_1.pdf) | Online Resource 1 | PLENA implementation details and reproducible computational workflow. The PDF contains supplementary figures, input-file and flow-direction descriptions, compilation and execution procedures, output-file descriptions, implementation notes, and computational-performance results. |
| [`supplementary/ESM_2.xlsx`](supplementary/ESM_2.xlsx) | Online Resource 2 | Catchment-level beta estimates and NSE results for the 239 drainage catchments in Seoul. The workbook contains final PLENA beta classes, comparison values from Seo et al. (2024), and candidate beta-specific NSE summaries from the supplied result files. |

## Online Resource 2 data scope

- `Catchment Results` contains all 239 catchments in ID order, including 237 valid PLENA classifications and 2 inputs without a valid classification.
- The final PLENA distribution is: `10^-4` = 11, `10^-3` = 84, `10^-2` = 122, `10^-1` = 17, `10^0` = 1, `10^1` = 2, and not classified = 2.
- `Candidate NSE` contains eight candidate-class summaries from each of 231 supplied result files, for 1,848 rows in total.
- The 231 result files cover 227 unique catchments because four catchment IDs have two supplied files. All are retained; the validated file is marked `Selected source = Yes`.
- Candidate-level values are absent where no corresponding result file was supplied. Missing NSE values were not estimated or imputed.
- `Candidate NSE` reports `k`, `beta`, mean NSE, and `NSE(mean_q)`, matching the checked PLENA summary structure. Mean NSE summarizes the stochastic realizations for each candidate class, while `NSE(mean_q)` evaluates the corresponding mean simulated width function.
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
| Published files must be self-contained because supplementary information is published as received. | Definitions, workflow notes, source links, column meanings, and coverage limitations are included in the files. |

## Interpretation boundary

PLENA beta classes and width functions are structural indicators of outlet-referenced flow-distance organization. They do not independently represent rainfall forcing, pipe capacity, pumps, downstream water levels, surface storage, inundation depth, or flood probability.
