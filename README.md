# Sundials Kinetics RS (Prototype)

This repository is currently a foundational work-in-progress aiming to provide high-performance, memory-safe Rust bindings to the **Sundials** suite (version 6+), specifically targeting stiff ODE and DAE integration (CVODE/IDA) with Sparse Jacobian (KLU) support. 

While the ultimate, long-term goal of this project is to achieve feature parity with Python-based chemical kinetics frameworks like [ChemPy](https://github.com/bjodah/chempy), the immediate priority is building a complete and robust native Rust interface to Sundials. 

## Current Capabilities

At present, this workspace contains:
1. **`sundials-sys` & `sundials`**:
   - Zero-cost Rust FFI bindings to Sundials 7.x (CVODE, IDA, ARKode, KINSOL).
   - Safe Rust wrappers for `SUNContext`, `NVector`, `DenseMatrix`, `BandMatrix`, and `SparseMatrix` (CSC/CSR).
   - Linear solver wrappers: `DenseLinearSolver`, `BandLinearSolver`, `SparseLinearSolver` (KLU), and iterative solvers (`SpgmrSolver`, `SpbcgsSolver`, `SptfqmrSolver`).
   - Preconditioner setup/solve callbacks for CVODE, IDA, and ARKode via boxed closures with `catch_unwind` safety.
   - Forward sensitivity analysis (CVODES) and adjoint sensitivity analysis (CVODES/IDAS).
   - Cross-platform build support: feature-gated KLU, CMake toolchain files for WASM and iOS.
   - A highly optimized C-trampoline that allows passing pure Rust closures (with zero-cost panic-unwind protection) directly to the solver right-hand-side evaluator.
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
- [x] Add support for Iterative Solvers (SPGMR, SPBCGS, SPTFQMR)
- [x] Add support for Preconditioner Setup and Solve callbacks
- [x] Implement `SUNMatrix_Band` and Banded Linear Solvers
- [x] Expose Sundials sensitivities and adjoint sensitivity analysis (CVODES/IDAS)
- [x] Robust cross-platform compilation targets (WASM, iOS) via CMake configurations in `build.rs`.

### Phase 2: C-API and FFI
- [x] Write an `extern "C"` wrapper over the kinetics engine (type-erased vtable dispatch for generic CVODE solver).
- [x] Add solver statistics FFI (`chem_evaluator_get_num_steps`, `chem_evaluator_get_num_rhs_evals`).
- [ ] Expose to Python and Dart (Flutter) environments, eliminating Python/C++ interpreter overhead from numerical integration loops.

### Phase 3: ChemPy Parity
- [x] Cantera `.yaml` mechanism parser with species, NASA-7 thermodynamics, elementary/three-body/falloff reactions.
- [x] Chemkin-like `.inp` parser (simplified format).
- [x] NASA-7 polynomial evaluation (cp/R, h/RT, s/R, g/RT).
- [x] Automatic analytical Jacobian generation via SymEngine bridging (MassAction and Arrhenius; pressure-dependent falls back to numerical).
- [x] Arrhenius temperature-dependent rate evaluation: k(T) = A·T^b·exp(-Ea/RT).
- [x] Runtime pressure-dependent rate evaluation: third-body [M], Lindemann falloff, Troe broadening factor.
- [ ] Full Cantera `.cti` format support.
- [ ] Python and Dart (Flutter) binding generation.

## License

This project is licensed under the [MIT License](LICENSE).
