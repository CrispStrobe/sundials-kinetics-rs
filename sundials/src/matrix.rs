use crate::context::Context;
use sundials_sys::{SUNDenseMatrix, SUNMatDestroy, SUNMatrix};

pub struct DenseMatrix {
    inner: SUNMatrix,
    rows: usize,
    cols: usize,
}

impl DenseMatrix {
    pub fn new(rows: usize, cols: usize, ctx: &Context) -> Self {
        let inner = unsafe { SUNDenseMatrix(rows as i64, cols as i64, ctx.as_raw()) };
        if inner.is_null() {
            panic!("Failed to allocate SUNDenseMatrix");
        }
        Self { inner, rows, cols }
    }

    pub fn as_raw(&self) -> SUNMatrix {
        self.inner
    }
}

impl Drop for DenseMatrix {
    fn drop(&mut self) {
        unsafe {
            SUNMatDestroy(self.inner);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;

    #[test]
    fn test_dense_matrix_creation() {
        let ctx = Context::new();
        let mat = DenseMatrix::new(3, 3, &ctx);
        assert!(!mat.as_raw().is_null());
    }
}
