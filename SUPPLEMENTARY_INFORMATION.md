# Supplementary Information Preparation Notes

Official guideline source checked: Natural Hazards submission guidelines  
https://link.springer.com/journal/11069/submission-guidelines

## Article Metadata

- Article title: City-Wide Assessment of Structural Complexity in Urban Drainage Networks Using Gibbs’ Model for Flood Risk Management
- Journal: Natural Hazards
- Authors: Changmin Park, Minsoo Seok, Yongwon Seo
- Affiliation: Department of Civil Engineering, Yeungnam University, Gyeongsan, Republic of Korea
- Corresponding author: Yongwon Seo (yongwon.seo@yu.ac.kr)

## Prepared Supplementary Files

| File | Suggested label | Content |
| --- | --- | --- |
| `supplementary/ESM_1.pdf` | Online Resource 1 | Supplementary PLENA implementation notes and captions for Figure S1, Figure S2, and Figure S3. |
| `supplementary/ESM_2.xlsx` | Online Resource 2 | English workbook summarizing the Seoul catchment beta-class comparison between Kim et al., 2017 and This Study. |

## Guideline Alignment Checklist

| Requirement checked against Natural Hazards / Springer guidance | Prepared status |
| --- | --- |
| Supplementary files are separate from the manuscript. | Prepared as separate PDF and XLSX files. |
| Supplementary files should use consecutive file numbering. | Files are named `ESM_1.pdf` and `ESM_2.xlsx`. |
| Supplementary files should be cited in the manuscript text using consecutive Online Resource numbering. | `ESM_1.pdf` is Online Resource 1; `ESM_2.xlsx` is Online Resource 2. |
| Each supplementary file should include article title, journal name, author names, affiliation, and corresponding-author e-mail. | `ESM_1.pdf` includes the metadata on page 1; `ESM_2.xlsx` includes the metadata in the `Metadata` sheet. |
| Supplementary figures should have captions or descriptive text. | Figure S1, Figure S2, and Figure S3 each have captions in `ESM_1.pdf`. |
| Spreadsheet supplementary data should use a standard spreadsheet format. | The result table is prepared as `.xlsx`. |
| Supplementary material should not introduce unsupported claims beyond the manuscript. | The PDF and workbook are restricted to PLENA. FMT is not presented as a completed or validated method. |

## Content Scope

This supplementary package is limited to PLENA, the C++ implementation used for Gibbs' Model-based drainage-network generation, width-function computation, Nash-Sutcliffe Efficiency comparison, and beta-class estimation.

The manuscript source indicates that FMT is not the focus of the current paper. Therefore, this supplementary package does not describe FMT as a developed method and does not make validation claims about FMT.

The beta classes and width functions are structural indicators. They are not direct hydraulic-risk, pipe-capacity, slope, roughness, pump-operation, downstream water-level, inundation-depth, or flood-probability indicators.

## Excel Workbook Summary

Workbook: `supplementary/ESM_2.xlsx`

- Metadata sheet: `Metadata`
- Main sheet: `PLENA Results`
- Summary sheet: `Summary`
- Rows: 239 catchments
- Main columns: `Catchment ID`, `Catchment Name`, `Kim et al., 2017`, `This Study`
- Row order: same as the source workbook, Catchment ID 1-239
- Source workbook labels converted:
  - source comparison column -> `Kim et al., 2017`
  - current-study result column -> `This Study`
