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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;
    use crate::matrix::DenseMatrix;
    use crate::nvector::NVector;

    #[test]
    fn test_dense_linsol_creation() {
        let ctx = Context::new();
        let vec = NVector::new_serial(3, &ctx);
        let mat = DenseMatrix::new(3, 3, &ctx);
        let linsol = DenseLinearSolver::new(&vec, &mat, &ctx);

        assert!(!linsol.as_raw().is_null());
    }
}
