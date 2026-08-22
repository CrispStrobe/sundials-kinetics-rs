use sundials_sys::{
    IDACreate, IDAInit, IDASStolerances, IDASolve, IDAFree,
    IDASetLinearSolver,
    N_Vector, sunrealtype, IDA_SUCCESS, IDA_NORMAL
};
use crate::context::Context;
use crate::nvector::NVector;
use crate::linsol::DenseLinearSolver;
use crate::matrix::DenseMatrix;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};

pub struct IdaSolver<'a> {
    inner: *mut c_void,
    _ctx: &'a Context,
    res: Box<dyn FnMut(f64, &[f64], &[f64], &mut [f64]) -> Result<(), ()> + 'a>,
}

extern "C" fn res_trampoline(
    t: sunrealtype,
    y: N_Vector,
    yp: N_Vector,
    resval: N_Vector,
    user_data: *mut c_void,
) -> i32 {
    let closure: &mut Box<dyn FnMut(f64, &[f64], &[f64], &mut [f64]) -> Result<(), ()>> =
        unsafe { &mut *(user_data as *mut _) };

    let len = unsafe { sundials_sys::N_VGetLength_Serial(y) } as usize;
    let y_slice = unsafe { std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(y), len) };
    let yp_slice = unsafe { std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(yp), len) };
    let resval_slice = unsafe { std::slice::from_raw_parts_mut(sundials_sys::N_VGetArrayPointer_Serial(resval), len) };

    let result = catch_unwind(AssertUnwindSafe(|| closure(t, y_slice, yp_slice, resval_slice)));
    
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(())) => 1,
        Err(_) => -1,
    }
}

impl<'a> IdaSolver<'a> {
    pub fn new(ctx: &'a Context) -> Self {
        let inner = unsafe { IDACreate(ctx.as_raw()) };
        if inner.is_null() {
            panic!("Failed to create IDA solver");
        }
        Self {
            inner,
            _ctx: ctx,
            res: Box::new(|_, _, _, _| Ok(())),
        }
    }

    pub fn init<F>(&mut self, t0: f64, y0: &NVector, yp0: &NVector, res: F)
    where
        F: FnMut(f64, &[f64], &[f64], &mut [f64]) -> Result<(), ()> + 'a,
    {
        self.res = Box::new(res);
        let user_data_ptr = &mut self.res as *mut _ as *mut c_void;
        
        unsafe {
            sundials_sys::IDASetUserData(self.inner, user_data_ptr);
            let flag = IDAInit(self.inner, Some(res_trampoline), t0, y0.as_raw(), yp0.as_raw());
            assert_eq!(flag, IDA_SUCCESS as i32);
        }
    }

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

    pub fn solve(&mut self, tout: f64, yout: &mut NVector, ypout: &mut NVector, tret: &mut f64) -> i32 {
        unsafe {
            IDASolve(self.inner, tout, tret, yout.as_raw(), ypout.as_raw(), IDA_NORMAL as i32)
        }
    }
}

impl<'a> Drop for IdaSolver<'a> {
    fn drop(&mut self) {
        unsafe {
            IDAFree(&mut self.inner);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;
    use crate::nvector::NVector;
    use crate::matrix::DenseMatrix;
    use crate::linsol::DenseLinearSolver;

    #[test]
    fn test_ida_integration() {
        let ctx = Context::new();
        let mut y = NVector::new_serial(1, &ctx);
        let mut yp = NVector::new_serial(1, &ctx);
        y.as_mut_slice()[0] = 1.0; // y(0) = 1.0
        yp.as_mut_slice()[0] = -0.5; // y'(0) = -0.5

        let mut solver = IdaSolver::new(&ctx);
        // F(t, y, y') = y' + 0.5 * y = 0
        solver.init(0.0, &y, &yp, |_t, y_val, yp_val, res| {
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
        assert!((actual - expected).abs() < 1e-3, "Expected {}, got {}", expected, actual);
    }
}
