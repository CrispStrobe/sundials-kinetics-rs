use crate::context::Context;
use crate::matrix::DenseMatrix;
use crate::nvector::NVector;
use sundials_sys::{SUNLinSolFree, SUNLinSol_Dense, SUNLinearSolver};

pub struct DenseLinearSolver {
    inner: SUNLinearSolver,
}

impl DenseLinearSolver {
    pub fn new(y: &NVector, mat: &DenseMatrix, ctx: &Context) -> Self {
        // SUNLinSol_Dense(y, A, ctx)
        let inner = unsafe { SUNLinSol_Dense(y.as_raw(), mat.as_raw(), ctx.as_raw()) };
        if inner.is_null() {
            panic!("Failed to allocate SUNLinSol_Dense");
        }
        Self { inner }
    }

    pub fn as_raw(&self) -> SUNLinearSolver {
        self.inner
    }
}

impl Drop for DenseLinearSolver {
    fn drop(&mut self) {
        unsafe {
            SUNLinSolFree(self.inner);
        }
    }
}

// ---------------------------------------------------------------------------
// Banded linear solver
// ---------------------------------------------------------------------------

pub struct BandLinearSolver {
    inner: SUNLinearSolver,
}

impl BandLinearSolver {
    pub fn new(y: &NVector, mat: &crate::matrix::BandMatrix, ctx: &Context) -> Self {
        let inner = unsafe { sundials_sys::SUNLinSol_Band(y.as_raw(), mat.as_raw(), ctx.as_raw()) };
        if inner.is_null() {
            panic!("Failed to allocate SUNLinSol_Band");
        }
        Self { inner }
    }

    pub fn as_raw(&self) -> SUNLinearSolver {
        self.inner
    }
}

impl Drop for BandLinearSolver {
    fn drop(&mut self) {
        unsafe {
            SUNLinSolFree(self.inner);
        }
    }
}

// ---------------------------------------------------------------------------
// Iterative solvers — SPGMR, SPBCGS, SPTFQMR
// ---------------------------------------------------------------------------

/// Preconditioner type for iterative solvers.
#[derive(Clone, Copy)]
pub enum PrecType {
    None = 0,
    Left = 1,
    Right = 2,
    Both = 3,
}

/// SPGMR (Scaled Preconditioned GMRES) iterative linear solver.
pub struct SpgmrSolver {
    inner: SUNLinearSolver,
}

impl SpgmrSolver {
    /// Create an SPGMR solver. `maxl` is the maximum Krylov dimension (0 = default = 5).
    pub fn new(y: &NVector, prec_type: PrecType, maxl: i32, ctx: &Context) -> Self {
        let inner = unsafe {
            sundials_sys::SUNLinSol_SPGMR(
                y.as_raw(),
                prec_type as i32,
                maxl,
                ctx.as_raw(),
            )
        };
        if inner.is_null() {
            panic!("Failed to allocate SUNLinSol_SPGMR");
        }
        Self { inner }
    }

    pub fn as_raw(&self) -> SUNLinearSolver {
        self.inner
    }
}

impl Drop for SpgmrSolver {
    fn drop(&mut self) {
        unsafe {
            SUNLinSolFree(self.inner);
        }
    }
}

/// SPBCGS (Scaled Preconditioned Bi-CGStab) iterative linear solver.
pub struct SpbcgsSolver {
    inner: SUNLinearSolver,
}

impl SpbcgsSolver {
    /// Create an SPBCGS solver. `maxl` is the maximum iterations (0 = default = 5).
    pub fn new(y: &NVector, prec_type: PrecType, maxl: i32, ctx: &Context) -> Self {
        let inner = unsafe {
            sundials_sys::SUNLinSol_SPBCGS(
                y.as_raw(),
                prec_type as i32,
                maxl,
                ctx.as_raw(),
            )
        };
        if inner.is_null() {
            panic!("Failed to allocate SUNLinSol_SPBCGS");
        }
        Self { inner }
    }

    pub fn as_raw(&self) -> SUNLinearSolver {
        self.inner
    }
}

impl Drop for SpbcgsSolver {
    fn drop(&mut self) {
        unsafe {
            SUNLinSolFree(self.inner);
        }
    }
}

/// SPTFQMR (Scaled Preconditioned TFQMR) iterative linear solver.
pub struct SptfqmrSolver {
    inner: SUNLinearSolver,
}

impl SptfqmrSolver {
    /// Create an SPTFQMR solver. `maxl` is the maximum iterations (0 = default = 5).
    pub fn new(y: &NVector, prec_type: PrecType, maxl: i32, ctx: &Context) -> Self {
        let inner = unsafe {
            sundials_sys::SUNLinSol_SPTFQMR(
                y.as_raw(),
                prec_type as i32,
                maxl,
                ctx.as_raw(),
            )
        };
        if inner.is_null() {
            panic!("Failed to allocate SUNLinSol_SPTFQMR");
        }
        Self { inner }
    }

    pub fn as_raw(&self) -> SUNLinearSolver {
        self.inner
    }
}

impl Drop for SptfqmrSolver {
    fn drop(&mut self) {
        unsafe {
            SUNLinSolFree(self.inner);
        }
    }
}

/// Trait implemented by all linear solvers so solvers (CVODE, IDA, ARKode)
/// can accept any of them generically.
pub trait LinearSolver {
    fn as_raw(&self) -> SUNLinearSolver;
}

impl LinearSolver for DenseLinearSolver {
    fn as_raw(&self) -> SUNLinearSolver {
        self.inner
    }
}
impl LinearSolver for BandLinearSolver {
    fn as_raw(&self) -> SUNLinearSolver {
        self.inner
    }
}
impl LinearSolver for SpgmrSolver {
    fn as_raw(&self) -> SUNLinearSolver {
        self.inner
    }
}
impl LinearSolver for SpbcgsSolver {
    fn as_raw(&self) -> SUNLinearSolver {
        self.inner
    }
}
impl LinearSolver for SptfqmrSolver {
    fn as_raw(&self) -> SUNLinearSolver {
        self.inner
    }
}
impl LinearSolver for crate::sparse::SparseLinearSolver {
    fn as_raw(&self) -> SUNLinearSolver {
        crate::sparse::SparseLinearSolver::as_raw(self)
    }
}

/// Trait implemented by all matrix types.
pub trait SunMatrix {
    fn as_raw(&self) -> sundials_sys::SUNMatrix;
}

impl SunMatrix for DenseMatrix {
    fn as_raw(&self) -> sundials_sys::SUNMatrix {
        DenseMatrix::as_raw(self)
    }
}
impl SunMatrix for crate::matrix::BandMatrix {
    fn as_raw(&self) -> sundials_sys::SUNMatrix {
        crate::matrix::BandMatrix::as_raw(self)
    }
}
impl SunMatrix for crate::sparse::SparseMatrix {
    fn as_raw(&self) -> sundials_sys::SUNMatrix {
        crate::sparse::SparseMatrix::as_raw(self)
    }
}

/// Marker trait for iterative solvers that can accept preconditioners.
pub trait IterativeSolver: LinearSolver {}
impl IterativeSolver for SpgmrSolver {}
impl IterativeSolver for SpbcgsSolver {}
impl IterativeSolver for SptfqmrSolver {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;
    use crate::matrix::{BandMatrix, DenseMatrix};
    use crate::nvector::NVector;

    #[test]
    fn test_dense_linsol_creation() {
        let ctx = Context::new();
        let vec = NVector::new_serial(3, &ctx);
        let mat = DenseMatrix::new(3, 3, &ctx);
        let linsol = DenseLinearSolver::new(&vec, &mat, &ctx);

        assert!(!linsol.as_raw().is_null());
    }

    #[test]
    fn test_band_linsol_creation() {
        let ctx = Context::new();
        let vec = NVector::new_serial(5, &ctx);
        let mat = BandMatrix::new(5, 1, 1, &ctx);
        let linsol = BandLinearSolver::new(&vec, &mat, &ctx);
        assert!(!LinearSolver::as_raw(&linsol).is_null());
    }

    #[test]
    fn test_spgmr_creation() {
        let ctx = Context::new();
        let vec = NVector::new_serial(5, &ctx);
        let ls = SpgmrSolver::new(&vec, PrecType::None, 0, &ctx);
        assert!(!LinearSolver::as_raw(&ls).is_null());
    }

    #[test]
    fn test_spbcgs_creation() {
        let ctx = Context::new();
        let vec = NVector::new_serial(5, &ctx);
        let ls = SpbcgsSolver::new(&vec, PrecType::None, 0, &ctx);
        assert!(!LinearSolver::as_raw(&ls).is_null());
    }

    #[test]
    fn test_sptfqmr_creation() {
        let ctx = Context::new();
        let vec = NVector::new_serial(5, &ctx);
        let ls = SptfqmrSolver::new(&vec, PrecType::None, 0, &ctx);
        assert!(!LinearSolver::as_raw(&ls).is_null());
    }
}
