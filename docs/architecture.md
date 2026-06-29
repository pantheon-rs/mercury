# Mercury Architecture

Mercury is the math substrate for Pantheon.

The crate should remain useful outside aerospace. Aerospace-specific physics
belongs in `vulcan`; plant simulation belongs in `icarus`.

## Responsibilities

- scalar traits and math operations
- numeric, AD, and symbolic execution contracts
- vector and matrix facade
- dense derivative APIs
- symbolic expression IR
- sparsity patterns and graph coloring
- root finding and optimization interfaces
- derivative testing utilities

## Design Direction

Mercury owns the public math contract. External crates such as `ad_trait`,
`nalgebra`, `faer`, sparse matrix crates, IPOPT bindings, or Enzyme integrations
are backend choices.

Model code should eventually be able to run through:

- `f64` numeric evaluation
- forward AD
- reverse AD
- symbolic tracing
- finite-difference diagnostics
- optional generated or Enzyme-backed hot paths

The initial crate is intentionally small. The first milestone is a stable
repository/tooling foundation and a narrow scalar surface that can expand under
test.
