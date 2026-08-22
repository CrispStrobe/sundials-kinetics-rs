use sundials_sys::{N_Vector, N_VNew_Serial, N_VDestroy_Serial, N_VGetArrayPointer_Serial};
use crate::context::Context;

/// A wrapper around Sundials N_Vector (Serial for now)
pub struct NVector {
    inner: N_Vector,
    length: usize,
}

impl NVector {
    /// Create a new serial NVector of the given length
    pub fn new_serial(length: usize, ctx: &Context) -> Self {
        let inner = unsafe { N_VNew_Serial(length as i64, ctx.as_raw()) };
        if inner.is_null() {
            panic!("Failed to allocate N_Vector");
        }
        Self { inner, length }
    }

    /// Access the underlying raw N_Vector
    pub fn as_raw(&self) -> N_Vector {
        self.inner
    }

    /// Access the raw array as a slice
    pub fn as_slice(&self) -> &[f64] {
        unsafe {
            let ptr = N_VGetArrayPointer_Serial(self.inner);
            std::slice::from_raw_parts(ptr, self.length)
        }
    }

    /// Access the raw array as a mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        unsafe {
            let ptr = N_VGetArrayPointer_Serial(self.inner);
            std::slice::from_raw_parts_mut(ptr, self.length)
        }
    }
}

impl Drop for NVector {
    fn drop(&mut self) {
        unsafe {
            N_VDestroy_Serial(self.inner);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;

    #[test]
    fn test_nvector_creation_and_slices() {
        let ctx = Context::new();
        let mut vec = NVector::new_serial(10, &ctx);
        
        assert_eq!(vec.as_slice().len(), 10);
        
        // Mutate through as_mut_slice
        {
            let slice = vec.as_mut_slice();
            for i in 0..10 {
                slice[i] = i as f64 * 2.0;
            }
        }
        
        // Read through as_slice
        let slice = vec.as_slice();
        for i in 0..10 {
            assert_eq!(slice[i], i as f64 * 2.0);
        }
    }
}
