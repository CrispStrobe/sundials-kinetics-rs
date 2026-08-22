use crate::context::Context;
use crate::linsol::DenseLinearSolver;
use crate::matrix::DenseMatrix;
use crate::nvector::NVector;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use sundials_sys::{
    sunrealtype, IDACreate, IDAFree, IDAInit, IDASStolerances, IDASetLinearSolver, IDASolve,
    N_Vector, IDA_NORMAL, IDA_SUCCESS,
};

pub struct IdaSolver<'a, F> {
    inner: *mut c_void,
    _ctx: &'a Context,
    res: *mut c_void,
    _marker: std::marker::PhantomData<F>,
}

pub struct IdaBuilder<'a> {
    inner: *mut c_void,
    _ctx: &'a Context,
}

extern "C" fn res_trampoline<F>(
    t: sunrealtype,
    y: N_Vector,
    yp: N_Vector,
    resval: N_Vector,
    user_data: *mut c_void,
) -> i32
where
    F: FnMut(f64, &[f64], &[f64], &mut [f64]) -> Result<(), ()>,
{
    let closure = unsafe { &mut *(user_data as *mut F) };

    let len = unsafe { sundials_sys::N_VGetLength_Serial(y) } as usize;
    let y_slice =
        unsafe { std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(y), len) };
    let yp_slice =
        unsafe { std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(yp), len) };
    let resval_slice = unsafe {
        std::slice::from_raw_parts_mut(sundials_sys::N_VGetArrayPointer_Serial(resval), len)
    };

    let result = catch_unwind(AssertUnwindSafe(|| {
        closure(t, y_slice, yp_slice, resval_slice)
    }));

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(())) => 1,
        Err(_) => -1,
    }
}

impl<'a> IdaBuilder<'a> {
    pub fn new(ctx: &'a Context) -> Self {
        let inner = unsafe { IDACreate(ctx.as_raw()) };
        if inner.is_null() {
            panic!("Failed to create IDA solver");
        }
        Self { inner, _ctx: ctx }
    }

    pub fn init<F>(mut self, t0: f64, y0: &NVector, yp0: &NVector, res: F) -> IdaSolver<'a, F>
    where
        F: FnMut(f64, &[f64], &[f64], &mut [f64]) -> Result<(), ()> + 'a,
    {
        let res_box = Box::new(res);
        let user_data_ptr = Box::into_raw(res_box) as *mut c_void;
        let inner = self.inner;
        self.inner = std::ptr::null_mut();

        unsafe {
            sundials_sys::IDASetUserData(inner, user_data_ptr);
            let flag = IDAInit(
                inner,
                Some(res_trampoline::<F>),
                t0,
                y0.as_raw(),
                yp0.as_raw(),
            );
            assert_eq!(flag, IDA_SUCCESS as i32);
        }

        IdaSolver {
            inner,
            _ctx: self._ctx,
            res: user_data_ptr,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a> Drop for IdaBuilder<'a> {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                IDAFree(&mut self.inner);
            }
        }
    }
}

impl<'a, F> IdaSolver<'a, F> {
    pub fn set_ss_tolerances(&self, reltol: f64, abstol: f64) {
        unsafe {
            let flag = IDASStolerances(self.inner, reltol, abstol);
            assert_eq!(flag, IDA_SUCCESS as i32);
        }
    }

    pub fn set_linear_solver(&self, linsol: &DenseLinearSolver, mat: &DenseMatrix) {
        unsafe {
            let flag = IDASetLinearSolver(self.inner, linsol.as_raw(), mat.as_raw());
            assert_eq!(flag, IDA_SUCCESS as i32);
        }
    }

    pub fn solve(
        &mut self,
        tout: f64,
        yout: &mut NVector,
        ypout: &mut NVector,
        tret: &mut f64,
    ) -> i32 {
        unsafe {
            IDASolve(
                self.inner,
                tout,
                tret,
                yout.as_raw(),
                ypout.as_raw(),
                IDA_NORMAL as i32,
            )
        }
    }
}

impl<'a, F> Drop for IdaSolver<'a, F> {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                IDAFree(&mut self.inner);
                let _ = Box::from_raw(self.res as *mut F);
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
    fn test_ida_integration() {
        let ctx = Context::new();
        let mut y = NVector::new_serial(1, &ctx);
        let mut yp = NVector::new_serial(1, &ctx);
        y.as_mut_slice()[0] = 1.0; // y(0) = 1.0
        yp.as_mut_slice()[0] = -0.5; // y'(0) = -0.5

        let mut solver = IdaBuilder::new(&ctx).init(0.0, &y, &yp, |_t, y_val, yp_val, res| {
            res[0] = yp_val[0] + 0.5 * y_val[0];
            Ok(())
        });

        solver.set_ss_tolerances(1e-4, 1e-4);

        let mat = DenseMatrix::new(1, 1, &ctx);
        let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
        solver.set_linear_solver(&linsol, &mat);

        let mut tret = 0.0;
        let flag = solver.solve(1.0, &mut y, &mut yp, &mut tret);

        assert_eq!(flag, IDA_SUCCESS as i32);
        let expected = (-0.5_f64).exp();
        let actual = y.as_slice()[0];
        assert!(
            (actual - expected).abs() < 1e-3,
            "Expected {}, got {}",
            expected,
            actual
        );
    }
}
