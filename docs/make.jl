cd(@__DIR__)
using ComplexityMeasures
using Documenter
using DocumenterCitations
using RecurrenceMicrostatesAnalysis
using StateSpaceSets

# Convert tutorial file to markdown
import Literate
Literate.markdown("src/tutorial.jl", "src"; credit = false)

pages = [
    "Welcome" => "index.md",
    "Tutorial" => "tutorial.md",
    "API" => "api.md",
    # "API" => "api.md",
    # "Tutorial" => [
    #     "Distributions" => "tutorial/distributions.md",
    #     "Quantifiers" => "tutorial/quantifiers.md",
    #     "Recurrence Functions" => "tutorial/recurrences.md",
    #     "Shapes and Sampling" => "tutorial/shapes_and_sampling.md",
    #     "GPU" => "tutorial/gpu.md",
    #     "Utils" => "tutorial/utils.md",
    # ],
    "Examples" => "examples.md",
    # "Ecosystem Integration" => [
    #         "Machine Learning" => "examples/ml.md",
    # ],
    "Developers docs" => "dev.md",
    "References" => "refs.md",
]

# Apply JuliaDynamics theme, choosing a specific branch (easier debugging)
github_user = "JuliaDynamics"
branch = "master"
download_path = "https://raw.githubusercontent.com/$github_user/doctheme/$branch/"

import Downloads
Downloads.download(
    "$download_path/build_docs_with_style.jl",
    joinpath(@__DIR__, "build_docs_with_style.jl")
)
include("build_docs_with_style.jl")

using DocumenterCitations

bib = CitationBibliography(
    joinpath(@__DIR__, "refs.bib");
    style=:authoryear
)

build_docs_with_style(pages, RecurrenceMicrostatesAnalysis, ComplexityMeasures, StateSpaceSets;
    expandfirst = ["index.md"], bib,
)
