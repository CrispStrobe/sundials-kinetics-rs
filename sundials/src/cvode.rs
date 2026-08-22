use sundials_sys::{
    CVodeCreate, CVodeInit, CVodeSStolerances, CVode, CVodeFree,
    CVodeSetLinearSolver, CV_ADAMS, CV_BDF,
    N_Vector, sunrealtype, CV_SUCCESS
};
use crate::context::Context;
use crate::nvector::NVector;
use crate::linsol::DenseLinearSolver;
use crate::matrix::DenseMatrix;
use std::ffi::c_void;
use std::ptr;
use std::panic::{catch_unwind, AssertUnwindSafe};

pub enum Lmm {
    Adams = CV_ADAMS as isize,
    Bdf = CV_BDF as isize,
}

pub struct CvodeSolver<'a> {
    inner: *mut c_void,
    // Keep reference to context so it isn't dropped
    _ctx: &'a Context,
    // Store user closure
    rhs: Box<dyn FnMut(f64, &[f64], &mut [f64]) -> Result<(), ()> + 'a>,
}

extern "C" fn rhs_trampoline(
    t: sunrealtype,
    y: N_Vector,
    ydot: N_Vector,
    user_data: *mut c_void,
) -> i32 {
    let closure: &mut Box<dyn FnMut(f64, &[f64], &mut [f64]) -> Result<(), ()>> =
        unsafe { &mut *(user_data as *mut _) };

    let len = unsafe { sundials_sys::N_VGetLength_Serial(y) } as usize;
    let y_slice = unsafe { std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(y), len) };
    let ydot_slice = unsafe { std::slice::from_raw_parts_mut(sundials_sys::N_VGetArrayPointer_Serial(ydot), len) };

    let result = catch_unwind(AssertUnwindSafe(|| closure(t, y_slice, ydot_slice)));
    
    match result {
        Ok(Ok(())) => 0, // success
        Ok(Err(())) => 1, // recoverable error
        Err(_) => -1, // unrecoverable error (panic)
    }
}

impl<'a> CvodeSolver<'a> {
    pub fn new(lmm: Lmm, ctx: &'a Context) -> Self {
        let inner = unsafe { CVodeCreate(lmm as i32, ctx.as_raw()) };
        if inner.is_null() {
            panic!("Failed to create CVode solver");
        }
        Self {
            inner,
            _ctx: ctx,
            rhs: Box::new(|_, _, _| Ok(())),
        }
    }

    pub fn init<F>(&mut self, t0: f64, y0: &NVector, rhs: F)
    where
        F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), ()> + 'a,
    {
        self.rhs = Box::new(rhs);
        let user_data_ptr = &mut self.rhs as *mut _ as *mut c_void;
        
        unsafe {
            sundials_sys::CVodeSetUserData(self.inner, user_data_ptr);
            let flag = CVodeInit(self.inner, Some(rhs_trampoline), t0, y0.as_raw());
            assert_eq!(flag, CV_SUCCESS as i32);
        }
    }

    pub fn set_ss_tolerances(&self, reltol: f64, abstol: f64) {
        unsafe {
            let flag = CVodeSStolerances(self.inner, reltol, abstol);
            assert_eq!(flag, CV_SUCCESS as i32);
        }
    }

    pub fn set_linear_solver(
        &mut self,
        linsol: &crate::linsol::DenseLinearSolver,
        mat: &crate::matrix::DenseMatrix,
    ) {
        let flag = unsafe { CVodeSetLinearSolver(self.inner, linsol.as_raw(), mat.as_raw()) };
        if flag != CV_SUCCESS as i32 {
            panic!("CVodeSetLinearSolver failed with code {}", flag);
        }
    }

    pub fn set_sparse_linear_solver(
        &mut self,
        linsol: &crate::sparse::SparseLinearSolver,
        mat: &crate::sparse::SparseMatrix,
    ) {
        let flag = unsafe { CVodeSetLinearSolver(self.inner, linsol.as_raw(), mat.as_raw()) };
        if flag != CV_SUCCESS as i32 {
            panic!("CVodeSetLinearSolver (sparse) failed with code {}", flag);
        }
    }

    pub fn step(&mut self, tout: f64, yout: &mut NVector, tret: &mut f64) -> i32 {
        unsafe {
            // CV_NORMAL is 1
            CVode(self.inner, tout, yout.as_raw(), tret, 1)
        }
    }

    pub fn get_num_steps(&self) -> Result<i64, i32> {
        let mut nsteps = 0;
        let flag = unsafe { sundials_sys::CVodeGetNumSteps(self.inner, &mut nsteps) };
        if flag == CV_SUCCESS as i32 { Ok(nsteps as i64) } else { Err(flag) }
    }

    pub fn get_num_rhs_evals(&self) -> Result<i64, i32> {
        let mut nfevals = 0;
        let flag = unsafe { sundials_sys::CVodeGetNumRhsEvals(self.inner, &mut nfevals) };
        if flag == CV_SUCCESS as i32 { Ok(nfevals as i64) } else { Err(flag) }
    }

    pub fn get_num_lin_solv_setups(&self) -> Result<i64, i32> {
        let mut nlinsetups = 0;
        let flag = unsafe { sundials_sys::CVodeGetNumLinSolvSetups(self.inner, &mut nlinsetups) };
        if flag == CV_SUCCESS as i32 { Ok(nlinsetups as i64) } else { Err(flag) }
    }

    pub fn get_num_err_test_fails(&self) -> Result<i64, i32> {
        let mut netfails = 0;
        let flag = unsafe { sundials_sys::CVodeGetNumErrTestFails(self.inner, &mut netfails) };
        if flag == CV_SUCCESS as i32 { Ok(netfails as i64) } else { Err(flag) }
    }
}

impl<'a> Drop for CvodeSolver<'a> {
    fn drop(&mut self) {
        unsafe {
            CVodeFree(&mut self.inner);
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
    fn test_cvode_integration() {
        let ctx = Context::new();
        let mut y = NVector::new_serial(1, &ctx);
        y.as_mut_slice()[0] = 1.0; // y(0) = 1.0

        let mut solver = CvodeSolver::new(Lmm::Adams, &ctx);
        solver.init(0.0, &y, |t, y_val, ydot| {
            // dy/dt = -0.5 * y
            ydot[0] = -0.5 * y_val[0];
            Ok(())
        });

        solver.set_ss_tolerances(1e-4, 1e-4);

        let mat = DenseMatrix::new(1, 1, &ctx);
        let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
        solver.set_linear_solver(&linsol, &mat);

        let mut tret = 0.0;
        // Integrate to t = 1.0
        let flag = solver.step(1.0, &mut y, &mut tret);
        
        assert_eq!(flag, CV_SUCCESS as i32);
        let expected = (-0.5_f64).exp(); // e^(-0.5)
        let actual = y.as_slice()[0];
        assert!((actual - expected).abs() < 1e-3, "Expected {}, got {}", expected, actual);

        // Verify stats are recorded
        let steps = solver.get_num_steps().unwrap();
        assert!(steps > 0);
        let rhs_evals = solver.get_num_rhs_evals().unwrap();
        assert!(rhs_evals > 0);
    }
}
