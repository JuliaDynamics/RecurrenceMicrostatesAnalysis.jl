# RecurrenceMicrostatesAnalysis.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliadynamics.github.io/RecurrenceMicrostatesAnalysis.jl/stable/)
[![Publication](https://img.shields.io/badge/publication-Chaos-blue.svg)](https://doi.org/10.1063/5.0293708)
[![CI](https://github.com/JuliaDynamics/RecurrenceMicrostatesAnalysis.jl/workflows/CI/badge.svg)](https://github.com/JuliaDynamics/RecurrenceMicrostatesAnalysis.jl/actions)
[![codecov](https://codecov.io/gh/JuliaDynamics/RecurrenceMicrostatesAnalysis.jl/graph/badge.svg?token=NR3S4JOE4R)](https://codecov.io/gh/JuliaDynamics/RecurrenceMicrostatesAnalysis.jl)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Package Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FRecurrenceMicrostatesAnalysis&query=total_requests&label=Downloads)](http://juliapkgstats.com/pkg/RecurrenceMicrostatesAnalysis)

**RecurrenceMicrostatesAnalysis.jl** is a simple and fast Julia-based package for recurrence microstates analysis.
It implements the computation of Recurrence Microstates Analysis (RMA) distributions, specific quantifiers — such as
[disorder](https://doi.org/10.1103/1y98-x33s) — and the estimation of typical RQA quantifiers, including 
determinism and laminarity.

RMA is a subfield of Recurrence Analysis and is a powerful tool for analyzing large time series or large datasets
using statistical methods, offering high performance and avoiding memory issues. Although the field is still
relatively new, it has shown promising applications, including in [Machine Learning](https://doi.org/10.1063/5.0203801).

**RecurrenceMicrostatesAnalysis.jl** is part of [DynamicalSystems.jl](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/dynamicalsystems/stable/),
and can be used either as a standalone package or in combination with other packages in the ecosystem. It is also
integrated with [ComplexityMeasures.jl](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/complexitymeasures/stable/),
providing an easy-to-use interface.

To install the package, run:
```julia
import Pkg
Pkg.add("RecurrenceMicrostatesAnalysis")
```

## Citation

If you use this package in a publication, or simply want to refer to it, 
please cite the paper below:

```
@article{RecurrenceMicrostatesAnalysis.jl,
    author = {Vinicius Ferreira, Gabriel and Lopes da Cruz, Felipe Eduardo and Marghoti, Gabriel and de Lima Prado, Thiago and Roberto Lopes, Sergio and Marwan, Norbert and Kurths, Jürgen},
    title = {RecurrenceMicrostatesAnalysis.jl: A Julia library for analyzing dynamical systems with recurrence microstates},
    journal = {Chaos: An Interdisciplinary Journal of Nonlinear Science},
    volume = {35},
    number = {11},
    pages = {113123},
    year = {2025},
    month = {11},
    issn = {1054-1500},
    doi = {10.1063/5.0293708},
    url = {https://doi.org/10.1063/5.0293708}
}
```
