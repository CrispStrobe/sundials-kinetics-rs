# Sundials Kinetics RS (Prototype)

This repository is currently a foundational work-in-progress aiming to provide high-performance, memory-safe Rust bindings to the **Sundials** suite (version 6+), specifically targeting stiff ODE and DAE integration (CVODE/IDA) with Sparse Jacobian (KLU) support. 

While the ultimate, long-term goal of this project is to achieve feature parity with Python-based chemical kinetics frameworks like [ChemPy](https://github.com/bjodah/chempy), the immediate priority is building a complete and robust native Rust interface to Sundials. 

## Current Capabilities

At present, this workspace contains:
1. **`sundials-sys` & `sundials`**:
   - Zero-cost Rust FFI bindings to Sundials CVODE and IDA.
   - Safe Rust wrappers for `SUNContext`, `NVector`, `DenseMatrix`, and `SparseMatrix` (CSC/CSR).
   - Linear solver wrappers for `DenseLinearSolver` and the `SuiteSparse KLU` direct sparse solver (`SUNLinSol_KLU`).
   - A highly optimized C-trampoline that allows passing pure Rust closures (with zero-cost panic-unwind protection) directly to the CVODE right-hand-side evaluator.
2. **`symengine-sys` & `symengine`**:
   - Native Rust bindings to the SymEngine C++ library.
   - Safe AST construction and analytical differentiation for building symbolic rate laws and Jacobians.
3. **`chem_core`**:
   - A prototype kinetic evaluator demonstrating how to leverage the above wrappers.
   - Supports defining a `ReactionSystem` with simplified mass-action kinetics or completely arbitrary rate-law closures.
   - Supports fixed-concentration species modeling (e.g. for heterogeneous catalysts).
   - Includes a basic mechanism parser for Chemkin-like `.inp` text definitions.

## Roadmap

### Phase 1: Full Sundials Parity
Before expanding the chemistry engine, we need to expose the full power of Sundials to Rust:
- [x] CVODE and IDA core wrappers
- [x] Dense and Sparse (KLU) Linear Solvers
- [ ] Add support for Iterative Solvers (SPGMR, SPBCGS, SPTFQMR)
- [ ] Add support for Preconditioner Setup and Solve callbacks
- [ ] Implement `SUNMatrix_Band` and Banded Linear Solvers
- [ ] Expose Sundials sensitivities and adjoint sensitivity analysis (CVODES/IDAS)
- [ ] Robust cross-platform compilation targets (WASM, iOS) via CMake configurations in `build.rs`.

### Phase 2: C-API and FFI
- [ ] Write an `extern "C"` wrapper over the kinetics engine.
- [ ] Expose to Python and Dart (Flutter) environments, eliminating Python/C++ interpreter overhead from numerical integration loops.

### Phase 3: ChemPy Parity
- [ ] Full Chemkin `.inp` and Cantera `.cti` / `.yaml` format parsers.
- [ ] Automatic analytical Jacobian generation via SymEngine bridging.
- [ ] Support for complex pressure-dependent reactions, third-body efficiencies, and falloff curves.
- [ ] Seamless integration with thermodynamic databases (e.g. NASA polynomials).

## License

This project is licensed under the [MIT License](LICENSE).
