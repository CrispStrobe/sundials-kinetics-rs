use crate::context::Context;
use sundials_sys::{SUNBandMatrix, SUNDenseMatrix, SUNMatDestroy, SUNMatrix};

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

/// Banded matrix wrapper around SUNBandMatrix.
///
/// `n` is the matrix dimension, `mu` is the upper half-bandwidth,
/// `ml` is the lower half-bandwidth, and `smu` is the storage upper
/// bandwidth (≥ mu, typically mu+ml for LU factorization).
pub struct BandMatrix {
    inner: SUNMatrix,
    n: usize,
    mu: usize,
    ml: usize,
}

impl BandMatrix {
    pub fn new(n: usize, mu: usize, ml: usize, ctx: &Context) -> Self {
        let inner =
            unsafe { SUNBandMatrix(n as i64, mu as i64, ml as i64, ctx.as_raw()) };
        if inner.is_null() {
            panic!("Failed to allocate SUNBandMatrix");
        }
        Self { inner, n, mu, ml }
    }

    pub fn as_raw(&self) -> SUNMatrix {
        self.inner
    }

    pub fn n(&self) -> usize {
        self.n
    }
    pub fn upper_bandwidth(&self) -> usize {
        self.mu
    }
    pub fn lower_bandwidth(&self) -> usize {
        self.ml
    }
}

impl Drop for BandMatrix {
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

    #[test]
    fn test_band_matrix_creation() {
        let ctx = Context::new();
        let mat = BandMatrix::new(5, 1, 1, &ctx);
        assert!(!mat.as_raw().is_null());
        assert_eq!(mat.n(), 5);
        assert_eq!(mat.upper_bandwidth(), 1);
        assert_eq!(mat.lower_bandwidth(), 1);
    }
}
