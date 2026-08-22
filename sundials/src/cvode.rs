use crate::context::Context;
use crate::linsol::DenseLinearSolver;
use crate::matrix::DenseMatrix;
use crate::nvector::NVector;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use sundials_sys::{
    sunrealtype, CVode, CVodeCreate, CVodeFree, CVodeGetSens, CVodeInit, CVodeSStolerances,
    CVodeSensEEtolerances, CVodeSensInit, CVodeSetLinearSolver, CVodeSetSensParams, N_Vector,
    CV_ADAMS, CV_BDF, CV_SIMULTANEOUS, CV_STAGGERED, CV_SUCCESS,
};

pub enum Lmm {
    Adams = CV_ADAMS as isize,
    Bdf = CV_BDF as isize,
}

pub enum SensMethod {
    Simultaneous = CV_SIMULTANEOUS as isize,
    Staggered = CV_STAGGERED as isize,
}

struct UserData<F, G> {
    rhs: F,
    sens_rhs: Option<G>,
}

pub struct CvodeSolver<'a, F, G> {
    inner: *mut c_void,
    _ctx: &'a Context,
    user_data: *mut c_void,
    _marker: std::marker::PhantomData<(F, G)>,
}

pub struct CvodeBuilder<'a> {
    inner: *mut c_void,
    _ctx: &'a Context,
}

extern "C" fn rhs_trampoline<F, G>(
    t: sunrealtype,
    y: N_Vector,
    ydot: N_Vector,
    user_data: *mut c_void,
) -> i32
where
    F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), ()>,
{
    let ud = unsafe { &mut *(user_data as *mut UserData<F, G>) };

    let len = unsafe { sundials_sys::N_VGetLength_Serial(y) } as usize;
    let y_slice =
        unsafe { std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(y), len) };
    let ydot_slice = unsafe {
        std::slice::from_raw_parts_mut(sundials_sys::N_VGetArrayPointer_Serial(ydot), len)
    };

    let result = catch_unwind(AssertUnwindSafe(|| (ud.rhs)(t, y_slice, ydot_slice)));

    match result {
        Ok(Ok(())) => 0,  // success
        Ok(Err(())) => 1, // recoverable error
        Err(_) => -1,     // unrecoverable error (panic)
    }
}

extern "C" fn sens_rhs_trampoline<F, G>(
    ns: i32,
    t: sunrealtype,
    y: N_Vector,
    ydot: N_Vector,
    ys_1d: *mut N_Vector,
    ysdot_1d: *mut N_Vector,
    user_data: *mut c_void,
    _tmp1: N_Vector,
    _tmp2: N_Vector,
) -> i32
where
    F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), ()>,
    G: FnMut(f64, &[f64], &[f64], &mut [&mut [f64]]) -> Result<(), ()>,
{
    let ud = unsafe { &mut *(user_data as *mut UserData<F, G>) };

    if let Some(ref mut sens_closure) = ud.sens_rhs {
        let len = unsafe { sundials_sys::N_VGetLength_Serial(y) } as usize;
        let y_slice =
            unsafe { std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(y), len) };
        let ydot_slice = unsafe {
            std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(ydot), len)
        };

        let mut ysdot_slices = Vec::with_capacity(ns as usize);
        unsafe {
            let ys_array = std::slice::from_raw_parts(ysdot_1d, ns as usize);
            for i in 0..ns as usize {
                ysdot_slices.push(std::slice::from_raw_parts_mut(
                    sundials_sys::N_VGetArrayPointer_Serial(ys_array[i]),
                    len,
                ));
            }
        }

        let result = catch_unwind(AssertUnwindSafe(|| {
            sens_closure(t, y_slice, ydot_slice, &mut ysdot_slices)
        }));

        match result {
            Ok(Ok(())) => 0,
            Ok(Err(())) => 1,
            Err(_) => -1,
        }
    } else {
        -1 // Should not be called if None
    }
}

impl<'a> CvodeBuilder<'a> {
    pub fn new(lmm: Lmm, ctx: &'a Context) -> Self {
        let inner = unsafe { CVodeCreate(lmm as i32, ctx.as_raw()) };
        if inner.is_null() {
            panic!("Failed to create CVode solver");
        }
        Self { inner, _ctx: ctx }
    }

    pub fn init<F>(mut self, t0: f64, y0: &NVector, rhs: F) -> CvodeSolver<'a, F, ()>
    where
        F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), ()> + 'a,
    {
        let ud = Box::new(UserData::<F, ()> {
            rhs,
            sens_rhs: None,
        });
        let user_data = Box::into_raw(ud) as *mut c_void;
        let inner = self.inner;
        self.inner = std::ptr::null_mut();

        unsafe {
            sundials_sys::CVodeSetUserData(inner, user_data);
            let flag = CVodeInit(inner, Some(rhs_trampoline::<F, ()>), t0, y0.as_raw());
            assert_eq!(flag, CV_SUCCESS as i32);
        }

        CvodeSolver {
            inner,
            _ctx: self._ctx,
            user_data,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a> Drop for CvodeBuilder<'a> {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                CVodeFree(&mut self.inner);
            }
        }
    }
}

impl<'a, F, G> CvodeSolver<'a, F, G> {
    pub fn init_sensitivities<G2>(
        mut self,
        method: SensMethod,
        y_s0: &[NVector],
        sens_rhs: Option<G2>,
    ) -> CvodeSolver<'a, F, G2>
    where
        F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), ()> + 'a,
        G2: FnMut(f64, &[f64], &[f64], &mut [&mut [f64]]) -> Result<(), ()> + 'a,
    {
        let inner = self.inner;
        self.inner = std::ptr::null_mut();

        let old_ud = unsafe { Box::from_raw(self.user_data as *mut UserData<F, G>) };
        let new_ud = Box::new(UserData::<F, G2> {
            rhs: old_ud.rhs,
            sens_rhs,
        });
        let user_data_ptr = Box::into_raw(new_ud) as *mut c_void;

        let ns = y_s0.len() as i32;
        let mut raw_ys0: Vec<N_Vector> = y_s0.iter().map(|v| v.as_raw()).collect();

        unsafe {
            sundials_sys::CVodeSetUserData(inner, user_data_ptr);

            let cb = if (*(user_data_ptr as *mut UserData<F, G2>))
                .sens_rhs
                .is_some()
            {
                Some(sens_rhs_trampoline::<F, G2> as _)
            } else {
                None
            };

            let flag = CVodeSensInit(inner, ns, method as i32, cb, raw_ys0.as_mut_ptr());
            assert_eq!(flag, CV_SUCCESS as i32, "CVodeSensInit failed");

            let flag = CVodeSensEEtolerances(inner);
            assert_eq!(flag, CV_SUCCESS as i32, "CVodeSensEEtolerances failed");
        }

        CvodeSolver {
            inner,
            _ctx: self._ctx,
            user_data: user_data_ptr,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn set_sens_params(
        &mut self,
        p: &mut [f64],
        pbar: Option<&mut [f64]>,
        plist: Option<&mut [i32]>,
    ) {
        unsafe {
            let pbar_ptr = pbar.map_or(ptr::null_mut(), |s| s.as_mut_ptr());
            let plist_ptr = plist.map_or(ptr::null_mut(), |s| s.as_mut_ptr());

            let flag = CVodeSetSensParams(self.inner, p.as_mut_ptr(), pbar_ptr, plist_ptr);
            assert_eq!(flag, CV_SUCCESS as i32, "CVodeSetSensParams failed");
        }
    }

    pub fn get_sens(&self, tret: &mut f64, y_s: &mut [NVector]) -> i32 {
        let mut raw_ys: Vec<N_Vector> = y_s.iter_mut().map(|v| v.as_raw()).collect();
        unsafe { sundials_sys::CVodeGetSens(self.inner, tret, raw_ys.as_mut_ptr()) }
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
        if flag == CV_SUCCESS as i32 {
            Ok(nsteps as i64)
        } else {
            Err(flag)
        }
    }

    pub fn get_num_rhs_evals(&self) -> Result<i64, i32> {
        let mut nfevals = 0;
        let flag = unsafe { sundials_sys::CVodeGetNumRhsEvals(self.inner, &mut nfevals) };
        if flag == CV_SUCCESS as i32 {
            Ok(nfevals as i64)
        } else {
            Err(flag)
        }
    }

    pub fn get_num_lin_solv_setups(&self) -> Result<i64, i32> {
        let mut nlinsetups = 0;
        let flag = unsafe { sundials_sys::CVodeGetNumLinSolvSetups(self.inner, &mut nlinsetups) };
        if flag == CV_SUCCESS as i32 {
            Ok(nlinsetups as i64)
        } else {
            Err(flag)
        }
    }

    pub fn get_num_err_test_fails(&self) -> Result<i64, i32> {
        let mut netfails = 0;
        let flag = unsafe { sundials_sys::CVodeGetNumErrTestFails(self.inner, &mut netfails) };
        if flag == CV_SUCCESS as i32 {
            Ok(netfails as i64)
        } else {
            Err(flag)
        }
    }
}

impl<'a, F, G> Drop for CvodeSolver<'a, F, G> {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                CVodeFree(&mut self.inner);
                let _ = Box::from_raw(self.user_data as *mut UserData<F, G>);
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
    fn test_cvode_integration() {
        let ctx = Context::new();
        let mut y = NVector::new_serial(1, &ctx);
        y.as_mut_slice()[0] = 1.0;

        let mut solver = CvodeBuilder::new(Lmm::Adams, &ctx).init(0.0, &y, |_t, y_val, ydot| {
            ydot[0] = -0.5 * y_val[0];
            Ok(())
        });

        solver.set_ss_tolerances(1e-4, 1e-4);

        let mat = DenseMatrix::new(1, 1, &ctx);
        let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
        solver.set_linear_solver(&linsol, &mat);

        let mut tret = 0.0;
        let flag = solver.step(1.0, &mut y, &mut tret);

        assert_eq!(flag, CV_SUCCESS as i32);
        let expected = (-0.5_f64).exp();
        let actual = y.as_slice()[0];
        assert!(
            (actual - expected).abs() < 1e-3,
            "Expected {}, got {}",
            expected,
            actual
        );
    }

    #[test]
    fn test_robertson_stiff_kinetics() {
        let ctx = Context::new();
        let mut y = NVector::new_serial(3, &ctx);

        let slice = y.as_mut_slice();
        slice[0] = 1.0;
        slice[1] = 0.0;
        slice[2] = 0.0;

        let mut solver = CvodeBuilder::new(Lmm::Bdf, &ctx).init(0.0, &y, |_t, y, ydot| {
            ydot[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
            ydot[2] = 3.0e7 * y[1] * y[1];
            ydot[1] = -ydot[0] - ydot[2]; // Mass conservation
            Ok(())
        });

        solver.set_ss_tolerances(1e-4, 1e-8);

        let mat = DenseMatrix::new(3, 3, &ctx);
        let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
        solver.set_linear_solver(&linsol, &mat);

        let mut tret = 0.0;
        let flag = solver.step(0.4, &mut y, &mut tret);

        assert_eq!(flag, CV_SUCCESS as i32);

        let out = y.as_slice();
        assert!((out[0] - 0.9851712).abs() < 1e-4, "y1 was {}", out[0]);
        assert!((out[1] - 0.0000338).abs() < 1e-5, "y2 was {}", out[1]);
        assert!((out[2] - 0.0147949).abs() < 1e-4, "y3 was {}", out[2]);
    }

    #[test]
    fn test_adams_simple() {
        let ctx = Context::new();
        let mut y = NVector::new_serial(1, &ctx);
        y.as_mut_slice()[0] = 0.0;

        let mut solver = CvodeBuilder::new(Lmm::Adams, &ctx).init(0.0, &y, |_t, _u, du| {
            du[0] = 1.0;
            Ok(())
        });

        solver.set_ss_tolerances(1e-4, 1e-8);
        let mat = DenseMatrix::new(1, 1, &ctx);
        let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
        solver.set_linear_solver(&linsol, &mat);

        let mut tret = 0.0;
        solver.step(2.0, &mut y, &mut tret);

        assert!((y.as_slice()[0] - 2.0).abs() < 1e-4);
    }
}
