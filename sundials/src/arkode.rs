use crate::context::Context;
use crate::nvector::NVector;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use sundials_sys::{
    sunrealtype, N_Vector,
    ARKStepCreate, ARKStepEvolve, ARKStepFree, ARKStepSetUserData, ARKStepSetLinearSolver,
    ARKStepSStolerances, ARK_SUCCESS, ARK_NORMAL,
};

struct UserData<F, G> {
    f_e: Option<F>,
    f_i: Option<G>,
}

pub struct ArkodeBuilder<'a> {
    _ctx: &'a Context,
}

pub struct ArkodeSolver<'a, F, G> {
    inner: *mut c_void,
    _ctx: &'a Context,
    user_data: *mut c_void,
    _marker: std::marker::PhantomData<(F, G)>,
}

extern "C" fn fe_trampoline<F, G>(
    t: sunrealtype,
    y: N_Vector,
    ydot: N_Vector,
    user_data: *mut c_void,
) -> i32
where
    F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), ()>,
{
    let ud = unsafe { &mut *(user_data as *mut UserData<F, G>) };
    if let Some(ref mut fe) = ud.f_e {
        let len = unsafe { sundials_sys::N_VGetLength_Serial(y) } as usize;
        let y_slice = unsafe { std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(y), len) };
        let ydot_slice = unsafe { std::slice::from_raw_parts_mut(sundials_sys::N_VGetArrayPointer_Serial(ydot), len) };

        let result = catch_unwind(AssertUnwindSafe(|| (fe)(t, y_slice, ydot_slice)));
        match result {
            Ok(Ok(())) => 0,
            Ok(Err(())) => 1,
            Err(_) => -1,
        }
    } else {
        -1
    }
}

extern "C" fn fi_trampoline<F, G>(
    t: sunrealtype,
    y: N_Vector,
    ydot: N_Vector,
    user_data: *mut c_void,
) -> i32
where
    G: FnMut(f64, &[f64], &mut [f64]) -> Result<(), ()>,
{
    let ud = unsafe { &mut *(user_data as *mut UserData<F, G>) };
    if let Some(ref mut fi) = ud.f_i {
        let len = unsafe { sundials_sys::N_VGetLength_Serial(y) } as usize;
        let y_slice = unsafe { std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(y), len) };
        let ydot_slice = unsafe { std::slice::from_raw_parts_mut(sundials_sys::N_VGetArrayPointer_Serial(ydot), len) };

        let result = catch_unwind(AssertUnwindSafe(|| (fi)(t, y_slice, ydot_slice)));
        match result {
            Ok(Ok(())) => 0,
            Ok(Err(())) => 1,
            Err(_) => -1,
        }
    } else {
        -1
    }
}

impl<'a> ArkodeBuilder<'a> {
    pub fn new(ctx: &'a Context) -> Self {
        Self { _ctx: ctx }
    }

    pub fn init_explicit<F>(
        self,
        t0: f64,
        y0: &NVector,
        fe: F,
    ) -> ArkodeSolver<'a, F, ()>
    where
        F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), ()> + 'a,
    {
        let ud = Box::new(UserData::<F, ()> {
            f_e: Some(fe),
            f_i: None,
        });
        let user_data = Box::into_raw(ud) as *mut c_void;

        let inner = unsafe {
            ARKStepCreate(Some(fe_trampoline::<F, ()>), None, t0, y0.as_raw(), self._ctx.as_raw())
        };
        if inner.is_null() {
            panic!("Failed to create ARKStep solver");
        }

        unsafe {
            let flag = ARKStepSetUserData(inner, user_data);
            assert_eq!(flag, ARK_SUCCESS as i32);
        }

        ArkodeSolver {
            inner,
            _ctx: self._ctx,
            user_data,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn init_implicit<G>(
        self,
        t0: f64,
        y0: &NVector,
        fi: G,
    ) -> ArkodeSolver<'a, (), G>
    where
        G: FnMut(f64, &[f64], &mut [f64]) -> Result<(), ()> + 'a,
    {
        let ud = Box::new(UserData::<(), G> {
            f_e: None,
            f_i: Some(fi),
        });
        let user_data = Box::into_raw(ud) as *mut c_void;

        let inner = unsafe {
            ARKStepCreate(None, Some(fi_trampoline::<(), G>), t0, y0.as_raw(), self._ctx.as_raw())
        };
        if inner.is_null() {
            panic!("Failed to create ARKStep solver");
        }

        unsafe {
            let flag = ARKStepSetUserData(inner, user_data);
            assert_eq!(flag, ARK_SUCCESS as i32);
        }

        ArkodeSolver {
            inner,
            _ctx: self._ctx,
            user_data,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn init_imex<F, G>(
        self,
        t0: f64,
        y0: &NVector,
        fe: F,
        fi: G,
    ) -> ArkodeSolver<'a, F, G>
    where
        F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), ()> + 'a,
        G: FnMut(f64, &[f64], &mut [f64]) -> Result<(), ()> + 'a,
    {
        let ud = Box::new(UserData::<F, G> {
            f_e: Some(fe),
            f_i: Some(fi),
        });
        let user_data = Box::into_raw(ud) as *mut c_void;

        let inner = unsafe {
            ARKStepCreate(Some(fe_trampoline::<F, G>), Some(fi_trampoline::<F, G>), t0, y0.as_raw(), self._ctx.as_raw())
        };
        if inner.is_null() {
            panic!("Failed to create ARKStep solver");
        }

        unsafe {
            let flag = ARKStepSetUserData(inner, user_data);
            assert_eq!(flag, ARK_SUCCESS as i32);
        }

        ArkodeSolver {
            inner,
            _ctx: self._ctx,
            user_data,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a, F, G> ArkodeSolver<'a, F, G> {
    pub fn set_ss_tolerances(&self, reltol: f64, abstol: f64) {
        unsafe {
            let flag = ARKStepSStolerances(self.inner, reltol, abstol);
            assert_eq!(flag, ARK_SUCCESS as i32);
        }
    }

    pub fn set_linear_solver(
        &mut self,
        linsol: &crate::linsol::DenseLinearSolver,
        mat: &crate::matrix::DenseMatrix,
    ) {
        let flag = unsafe { ARKStepSetLinearSolver(self.inner, linsol.as_raw(), mat.as_raw()) };
        if flag != ARK_SUCCESS as i32 {
            panic!("ARKStepSetLinearSolver failed with code {}", flag);
        }
    }

    pub fn step(&mut self, tout: f64, yout: &mut NVector, tret: &mut f64) -> i32 {
        unsafe {
            ARKStepEvolve(self.inner, tout, yout.as_raw(), tret, ARK_NORMAL as i32)
        }
    }

    pub fn get_num_steps(&self) -> Result<i64, i32> {
        let mut nsteps = 0;
        let flag = unsafe { sundials_sys::ARKStepGetNumSteps(self.inner, &mut nsteps) };
        if flag == ARK_SUCCESS as i32 {
            Ok(nsteps as i64)
        } else {
            Err(flag)
        }
    }

    pub fn get_num_rhs_evals(&self) -> Result<(i64, i64), i32> {
        let mut nfevals = 0;
        let mut nfievals = 0;
        let flag = unsafe { sundials_sys::ARKStepGetNumRhsEvals(self.inner, &mut nfevals, &mut nfievals) };
        if flag == ARK_SUCCESS as i32 {
            Ok((nfevals as i64, nfievals as i64))
        } else {
            Err(flag)
        }
    }

    pub fn get_num_lin_solv_setups(&self) -> Result<i64, i32> {
        let mut nlinsetups = 0;
        let flag = unsafe { sundials_sys::ARKStepGetNumLinSolvSetups(self.inner, &mut nlinsetups) };
        if flag == ARK_SUCCESS as i32 {
            Ok(nlinsetups as i64)
        } else {
            Err(flag)
        }
    }

    pub fn get_num_err_test_fails(&self) -> Result<i64, i32> {
        let mut netfails = 0;
        let flag = unsafe { sundials_sys::ARKStepGetNumErrTestFails(self.inner, &mut netfails) };
        if flag == ARK_SUCCESS as i32 {
            Ok(netfails as i64)
        } else {
            Err(flag)
        }
    }
}

impl<'a, F, G> Drop for ArkodeSolver<'a, F, G> {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ARKStepFree(&mut self.inner);
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
    fn test_arkode_explicit() {
        let ctx = Context::new();
        let mut y = NVector::new_serial(1, &ctx);
        y.as_mut_slice()[0] = 1.0;

        let mut solver = ArkodeBuilder::new(&ctx).init_explicit(0.0, &y, |_t, y_val, ydot| {
            ydot[0] = -0.5 * y_val[0];
            Ok(())
        });

        solver.set_ss_tolerances(1e-4, 1e-4);

        let mut tret = 0.0;
        let flag = solver.step(1.0, &mut y, &mut tret);

        assert_eq!(flag, ARK_SUCCESS as i32);
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
    fn test_arkode_implicit() {
        let ctx = Context::new();
        let mut y = NVector::new_serial(1, &ctx);
        y.as_mut_slice()[0] = 1.0;

        let mut solver = ArkodeBuilder::new(&ctx).init_implicit(0.0, &y, |_t, y_val, ydot| {
            ydot[0] = -0.5 * y_val[0];
            Ok(())
        });

        solver.set_ss_tolerances(1e-4, 1e-4);

        let mat = DenseMatrix::new(1, 1, &ctx);
        let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
        solver.set_linear_solver(&linsol, &mat);

        let mut tret = 0.0;
        let flag = solver.step(1.0, &mut y, &mut tret);

        assert_eq!(flag, ARK_SUCCESS as i32);
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
