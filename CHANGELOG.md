#   CHANGELOG

This changelog is maintained starting from version 0.5.

##  0.5
- Complete redesign of the backend using abstract types, structs, and function overloading, replacing the previous function-based implementation.
- The input type of the `distribution` function has been changed to `StateSpaceSet` for time series, improving interoperability with [DynamicalSystems.jl](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/dynamicalsystems/stable/).
- The output type of the `distributions` function has been changed to `Probabilities`, improving interoperability with [ComplexityMeasures.jl](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/complexitymeasures/stable/)
- Added a new GPU backend based on [KernelAbstractions.jl](https://juliagpu.github.io/KernelAbstractions.jl/stable/).
- Implemented operations for computing microstate permutations.

### BREAKING CHANGES

Version 0.5 is not backward compatible with previous releases of RecurrenceMicrostateAnalysis.jl, the notes above discuss the key types and their inputs.
We recommend going through the brand new tutorial to learn the software.
