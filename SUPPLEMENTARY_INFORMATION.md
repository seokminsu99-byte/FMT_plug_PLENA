# Supplementary Information Preparation Notes

Official guideline source checked: Natural Hazards submission guidelines  
https://link.springer.com/journal/11069/submission-guidelines

## Article Metadata

- Article title: High-Performance C++ Reimplementation and Optimization of the Gibbs' Model for Analyzing Structural Complexity in Urban Drainage Networks
- Journal: Natural Hazards
- Authors: Changmin Park, Minsoo Seok, Yongwon Seo
- Affiliation: Department of Civil Engineering, Yeungnam University, Gyeongsan, Republic of Korea
- Corresponding author: Yongwon Seo (yongwon.seo@yu.ac.kr)

## Prepared Supplementary Files

| File | Suggested label | Content |
| --- | --- | --- |
| `supplementary/online_resource_1_revised.pdf` | Online Resource 1 | Supplementary PLENA implementation notes and captions for Figure S1, Figure S2, and Figure S3. |
| `supplementary/PLENA_beta_results_english.xlsx` | Online Resource 2 | English workbook summarizing the Seoul catchment beta-class comparison between Kim et al., 2017 and This Study. |

## Guideline Alignment Checklist

| Requirement checked against Natural Hazards / Springer guidance | Prepared status |
| --- | --- |
| Supplementary files are separate from the manuscript. | Prepared as separate PDF and XLSX files. |
| Supplementary files should be cited in the manuscript text using consecutive Online Resource numbering. | PDF is prepared as Online Resource 1; workbook can be cited as Online Resource 2 if submitted. |
| Supplementary figures should have captions or descriptive text. | Figure S1, Figure S2, and Figure S3 each have captions in `online_resource_1_revised.pdf`. |
| Spreadsheet supplementary data should use a standard spreadsheet format. | The result table is prepared as `.xlsx`. |
| Supplementary material should not introduce unsupported claims beyond the manuscript. | The PDF and workbook are restricted to PLENA. FMT is not presented as a completed or validated method. |

## Content Scope

This supplementary package is limited to PLENA, the C++ implementation used for Gibbs' Model-based drainage-network generation, width-function computation, Nash-Sutcliffe Efficiency comparison, and beta-class estimation.

The manuscript source indicates that FMT is not the focus of the current paper. Therefore, this supplementary package does not describe FMT as a developed method and does not make validation claims about FMT.

The beta classes and width functions are structural indicators. They are not direct hydraulic-risk, pipe-capacity, slope, roughness, pump-operation, downstream water-level, inundation-depth, or flood-probability indicators.

## Excel Workbook QA

Workbook: `supplementary/PLENA_beta_results_english.xlsx`

- Main sheet: `PLENA Results`
- Rows: 239 catchments
- Main columns: `Catchment ID`, `Catchment Name`, `Kim et al., 2017`, `This Study`
- Row order: same as the source workbook, Catchment ID 1-239
- Source workbook labels converted:
  - source comparison column -> `Kim et al., 2017`
  - current-study result column -> `This Study`
- Catchment-name traceability is retained in the workbook's `Name Audit` sheet.
- Verification:
  - The output workbook was reopened after export.
  - `PLENA Results` contains 239 data rows and 4 main columns.
  - Catchment IDs are ordered 1-239.
  - Formula-error scan returned zero matches.
  - Rendered previews of the main sheet, audit sheet, and summary sheet were visually checked.
