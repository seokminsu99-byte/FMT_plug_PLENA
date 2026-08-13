# PLENA

[![Build and smoke test](https://github.com/seokminsu99-byte/FMT_plug_PLENA/actions/workflows/build.yml/badge.svg)](https://github.com/seokminsu99-byte/FMT_plug_PLENA/actions/workflows/build.yml)

PLENA (Program for Large-scale Evaluation of Network Analysis) is a C++17
implementation of a one-parameter Gibbs model for stochastic structural
analysis of drainage networks. It generates feasible drainage-tree states,
calculates outlet-referenced width functions, and estimates beta classes using
the Nash-Sutcliffe efficiency (NSE).

This repository accompanies the manuscript *Sinuosity–Inundation Trade-Offs in
Urban Drainage Networks: A Scale-Proportional Gibbs’ Model Analysis of Seoul*,
prepared for *Hydrological Processes*.

## Repository contents

| Path | Description |
| --- | --- |
| [`src/plena.cpp`](src/plena.cpp) | PLENA C++17 source code |
| [`example.txt`](example.txt) | Synthetic 51 x 51 input network |
| [`example_explain.txt`](example_explain.txt) | Input format and execution guide |
| [`supplementary/Supporting_Information.docx`](supplementary/Supporting_Information.docx) | Supporting Information with Figures S1-S3 |
| [`supplementary/Data_S2.xlsx`](supplementary/Data_S2.xlsx) | Catchment-level beta estimates, NSE results, and inundation metrics |
| [`PROVENANCE.md`](PROVENANCE.md) | Scientific and software lineage |

Generated executables, program outputs, editor files, and temporary working
files are intentionally excluded.

## Build and test

A C++17 compiler with standard thread support is required.

```bash
g++ -O2 -std=c++17 -pthread -Wall -Wextra -pedantic src/plena.cpp -o PLENA
./PLENA --self-test
```

On Windows with MinGW, use `-o PLENA.exe`. The GitHub Actions workflow builds
the same source and runs the transition-kernel and input/output smoke tests.

## Run

```bash
./PLENA example.txt
```

If no input path is supplied, PLENA lists eligible `.txt` files in the working
directory. A fixed random seed may be supplied as the second command-line
argument for an exactly repeatable diagnostic run.

The plain-text input contains the grid dimensions, the one-based outlet row and
column, and the drainage-direction matrix. Direction codes are `0` = inactive
or no pipe, `1` = east, `2` = south, `3` = west, and `4` = north. The outlet
must be active, and every active cell must reach it. See
[`example_explain.txt`](example_explain.txt) for the full input contract.

## Method and outputs

The update budget is `I = alpha * n * m`; entering `0` for `alpha` selects the
study setting of 10. At each update, PLENA selects one non-outlet active cell
and proposes one of the three directions different from its current direction.
Invalid or outlet-unreachable proposals are rejected as self-transitions.

The primary result file contains the generated direction matrix,
outlet-distance matrix, flow-accumulation matrix, and outlet-referenced `q(t)`.
Optional batch analysis writes flow-direction and link-state matrices,
width-function values, and NSE summaries.

PLENA provides a structural comparison of drainage-network configurations. A
beta class or width function does not by itself represent rainfall forcing,
conduit capacity, storage, pump operation, downstream water level, inundation
depth, or flood probability.

## Supporting data and access

[`Data_S2.xlsx`](supplementary/Data_S2.xlsx) contains the catchment-level data
and formula-linked checks used in the manuscript. Municipal drainage-network
and catchment GIS files are not redistributed because access and reuse are
controlled by the Seoul Metropolitan Government.

Annual inundation-footprint data are available from Seoul Open Data Plaza
dataset OA-15636: <https://data.seoul.go.kr/dataList/OA-15636/F/1/datasetView.do>.
Requests for the underlying drainage GIS data should be directed to the Seoul
Metropolitan Government, Water Circulation Safety Bureau, Water Reclamation
Planning Division: <https://news.seoul.go.kr/env/archives/747>.

## Citation and license

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The repository
is distributed under the [MIT License](LICENSE).
