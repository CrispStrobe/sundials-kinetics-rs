use crate::context::Context;
use crate::linsol::DenseLinearSolver;
use crate::matrix::DenseMatrix;
use crate::nvector::NVector;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use sundials_sys::{
    KINCreate, KINFree, KINInit, KINSetLinearSolver, KINSetUserData, KINSol, N_Vector, KIN_SUCCESS,
};

pub struct KinsolSolver<'a, F> {
    inner: *mut c_void,
    _ctx: &'a Context,
    sys_fn: *mut c_void,
    _marker: std::marker::PhantomData<F>,
}

pub struct KinsolBuilder<'a> {
    inner: *mut c_void,
    _ctx: &'a Context,
}

extern "C" fn sys_trampoline<F>(u: N_Vector, fval: N_Vector, user_data: *mut c_void) -> i32
where
    F: FnMut(&[f64], &mut [f64]) -> Result<(), ()>,
{
    unsafe {
        let closure = &mut *(user_data as *mut F);
        let u_slice = std::slice::from_raw_parts(
            sundials_sys::N_VGetArrayPointer_Serial(u),
            sundials_sys::N_VGetLength_Serial(u) as usize,
        );
        let f_slice = std::slice::from_raw_parts_mut(
            sundials_sys::N_VGetArrayPointer_Serial(fval),
            sundials_sys::N_VGetLength_Serial(fval) as usize,
        );

        let result = catch_unwind(AssertUnwindSafe(|| closure(u_slice, f_slice)));

        match result {
            Ok(Ok(())) => 0,
            _ => -1, // Unrecoverable error
        }
    }
}

impl<'a> KinsolBuilder<'a> {
    pub fn new(ctx: &'a Context) -> Self {
        let inner = unsafe { KINCreate(ctx.as_raw()) };
        if inner.is_null() {
            panic!("Failed to create KINSOL solver");
        }
        Self { inner, _ctx: ctx }
    }

    pub fn init<F>(mut self, tmpl: &NVector, sys_fn: F) -> KinsolSolver<'a, F>
    where
        F: FnMut(&[f64], &mut [f64]) -> Result<(), ()> + 'a,
    {
        let sys_box = Box::new(sys_fn);
        let user_data_ptr = Box::into_raw(sys_box) as *mut c_void;
        let inner = self.inner;
        self.inner = std::ptr::null_mut();

        unsafe {
            let flag = KINInit(inner, Some(sys_trampoline::<F>), tmpl.as_raw());
            if flag != 0 {
                panic!("KINInit failed with code {}", flag);
            }
            KINSetUserData(inner, user_data_ptr);
        }

        KinsolSolver {
            inner,
            _ctx: self._ctx,
            sys_fn: user_data_ptr,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a> Drop for KinsolBuilder<'a> {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                KINFree(&mut self.inner);
            }
        }
    }
}

impl<'a, F> KinsolSolver<'a, F> {
    pub fn set_linear_solver(&mut self, linsol: &DenseLinearSolver, mat: &DenseMatrix) {
        unsafe {
            let flag = KINSetLinearSolver(self.inner, linsol.as_raw(), mat.as_raw());
            if flag != 0 {
                panic!("KINSetLinearSolver failed with code {}", flag);
            }
        }
    }

    pub fn solve(&mut self, u: &mut NVector) -> i32 {
        let strategy = 0; // KIN_NONE
        let mut u_scale = NVector::new_serial(u.as_slice().len(), self._ctx);
        let mut f_scale = NVector::new_serial(u.as_slice().len(), self._ctx);
        u_scale.as_mut_slice().fill(1.0);
        f_scale.as_mut_slice().fill(1.0);

        unsafe {
            KINSol(
                self.inner,
                u.as_raw(),
                strategy,
                u_scale.as_raw(),
                f_scale.as_raw(),
            )
        }
    }
}

impl<'a, F> Drop for KinsolSolver<'a, F> {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                KINFree(&mut self.inner);
                let _ = Box::from_raw(self.sys_fn as *mut F);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;
    use crate::linsol::DenseLinearSolver;
    use crate::matrix::DenseMatrix;
    use crate::nvector::NVector;

    #[test]
    fn test_kinsol_solve() {
        let ctx = Context::new();
        // Solve x^2 - 4 = 0
        let mut u = NVector::new_serial(1, &ctx);
        u.as_mut_slice()[0] = 1.0; // Initial guess

        let mut solver = KinsolBuilder::new(&ctx).init(&u, |u_val, fval| {
            fval[0] = u_val[0] * u_val[0] - 4.0;
            Ok(())
        });

        let mat = DenseMatrix::new(1, 1, &ctx);
        let linsol = DenseLinearSolver::new(&u, &mat, &ctx);
        solver.set_linear_solver(&linsol, &mat);

        let flag = solver.solve(&mut u);
        assert_eq!(flag, KIN_SUCCESS as i32);

        let actual = u.as_slice()[0];
        assert!(
            (actual - 2.0).abs() < 1e-4 || (actual + 2.0).abs() < 1e-4,
            "Expected 2 or -2, got {}",
            actual
        );
    }
}
