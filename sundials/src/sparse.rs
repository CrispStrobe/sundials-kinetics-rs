use crate::{context::Context, nvector::NVector};
use sundials_sys::{
    SUNMatDestroy, SUNMatrix, SUNSparseMatrix, CSC_MAT, CSR_MAT,
};

#[derive(Clone, Copy)]
pub enum SparseType {
    Csc = CSC_MAT as isize,
    Csr = CSR_MAT as isize,
}

pub struct SparseMatrix {
    inner: SUNMatrix,
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
    pub sparse_type: SparseType,
}

impl SparseMatrix {
    pub fn new(
        rows: usize,
        cols: usize,
        nnz: usize,
        sparse_type: SparseType,
        ctx: &Context,
    ) -> Self {
        let inner = unsafe {
            SUNSparseMatrix(
                rows as i64,
                cols as i64,
                nnz as i64,
                sparse_type as i32,
                ctx.as_raw(),
            )
        };
        if inner.is_null() {
            panic!("Failed to allocate SUNSparseMatrix");
        }

        Self {
            inner,
            rows,
            cols,
            nnz,
            sparse_type,
        }
    }

    pub fn as_raw(&self) -> SUNMatrix {
        self.inner
    }
}

impl Drop for SparseMatrix {
    fn drop(&mut self) {
        unsafe {
            SUNMatDestroy(self.inner);
        }
    }
}

#[cfg(feature = "klu")]
pub struct SparseLinearSolver {
    inner: sundials_sys::SUNLinearSolver,
}

#[cfg(feature = "klu")]
impl SparseLinearSolver {
    pub fn new(y: &NVector, mat: &SparseMatrix, ctx: &Context) -> Self {
        let inner = unsafe { sundials_sys::SUNLinSol_KLU(y.as_raw(), mat.as_raw(), ctx.as_raw()) };
        if inner.is_null() {
            panic!("Failed to allocate SUNLinSol_KLU");
        }
        Self { inner }
    }

    pub fn as_raw(&self) -> sundials_sys::SUNLinearSolver {
        self.inner
    }
}

#[cfg(feature = "klu")]
impl Drop for SparseLinearSolver {
    fn drop(&mut self) {
        unsafe {
            sundials_sys::SUNLinSolFree(self.inner);
        }
    }
}
