use crate::context::Context;
use crate::nvector::NVector;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use sundials_sys::{sunrealtype, N_Vector};

// SUNDIALS 7.x: ARKStepCreate still exists, but the evolve/set/get helpers
// moved to unified ARKode* names.

struct UserData<F, G> {
    f_e: Option<F>,
    f_i: Option<G>,
    psetup: Option<Box<dyn FnMut(f64, &[f64], &[f64], bool, f64) -> Result<bool, ()>>>,
    psolve: Option<
        Box<dyn FnMut(f64, &[f64], &[f64], &[f64], &mut [f64], f64, f64, i32) -> Result<(), ()>>,
    >,
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
        let y_slice = unsafe {
            std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(y), len)
        };
        let ydot_slice = unsafe {
            std::slice::from_raw_parts_mut(sundials_sys::N_VGetArrayPointer_Serial(ydot), len)
        };

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
        let y_slice = unsafe {
            std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(y), len)
        };
        let ydot_slice = unsafe {
            std::slice::from_raw_parts_mut(sundials_sys::N_VGetArrayPointer_Serial(ydot), len)
        };

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

// Preconditioner trampolines for ARKode
extern "C" fn ark_psetup_trampoline<F, G>(
    t: sunrealtype,
    y: N_Vector,
    fy: N_Vector,
    jok: i32,
    jcur_ptr: *mut i32,
    gamma: sunrealtype,
    user_data: *mut c_void,
) -> i32 {
    let ud = unsafe { &mut *(user_data as *mut UserData<F, G>) };
    if let Some(ref mut psetup) = ud.psetup {
        let len = unsafe { sundials_sys::N_VGetLength_Serial(y) } as usize;
        let y_s = unsafe {
            std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(y), len)
        };
        let fy_s = unsafe {
            std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(fy), len)
        };
        let result =
            catch_unwind(AssertUnwindSafe(|| psetup(t, y_s, fy_s, jok != 0, gamma)));
        match result {
            Ok(Ok(jcur)) => {
                unsafe { *jcur_ptr = if jcur { 1 } else { 0 } };
                0
            }
            Ok(Err(())) => 1,
            Err(_) => -1,
        }
    } else {
        0
    }
}

extern "C" fn ark_psolve_trampoline<F, G>(
    t: sunrealtype,
    y: N_Vector,
    fy: N_Vector,
    r: N_Vector,
    z: N_Vector,
    gamma: sunrealtype,
    delta: sunrealtype,
    lr: i32,
    user_data: *mut c_void,
) -> i32 {
    let ud = unsafe { &mut *(user_data as *mut UserData<F, G>) };
    if let Some(ref mut psolve) = ud.psolve {
        let len = unsafe { sundials_sys::N_VGetLength_Serial(y) } as usize;
        let y_s = unsafe {
            std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(y), len)
        };
        let fy_s = unsafe {
            std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(fy), len)
        };
        let r_s = unsafe {
            std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(r), len)
        };
        let z_s = unsafe {
            std::slice::from_raw_parts_mut(sundials_sys::N_VGetArrayPointer_Serial(z), len)
        };
        let result = catch_unwind(AssertUnwindSafe(|| {
            psolve(t, y_s, fy_s, r_s, z_s, gamma, delta, lr)
        }));
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

    pub fn init_explicit<F>(self, t0: f64, y0: &NVector, fe: F) -> ArkodeSolver<'a, F, ()>
    where
        F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), ()> + 'a,
    {
        let ud = Box::new(UserData::<F, ()> {
            f_e: Some(fe),
            f_i: None,
            psetup: None,
            psolve: None,
        });
        let user_data = Box::into_raw(ud) as *mut c_void;

        let inner = unsafe {
            sundials_sys::ARKStepCreate(
                Some(fe_trampoline::<F, ()>),
                None,
                t0,
                y0.as_raw(),
                self._ctx.as_raw(),
            )
        };
        if inner.is_null() {
            panic!("Failed to create ARKStep solver");
        }

        unsafe {
            let flag = sundials_sys::ARKodeSetUserData(inner, user_data);
            assert_eq!(flag, sundials_sys::ARK_SUCCESS as i32);
        }

        ArkodeSolver {
            inner,
            _ctx: self._ctx,
            user_data,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn init_implicit<G>(self, t0: f64, y0: &NVector, fi: G) -> ArkodeSolver<'a, (), G>
    where
        G: FnMut(f64, &[f64], &mut [f64]) -> Result<(), ()> + 'a,
    {
        let ud = Box::new(UserData::<(), G> {
            f_e: None,
            f_i: Some(fi),
            psetup: None,
            psolve: None,
        });
        let user_data = Box::into_raw(ud) as *mut c_void;

        let inner = unsafe {
            sundials_sys::ARKStepCreate(
                None,
                Some(fi_trampoline::<(), G>),
                t0,
                y0.as_raw(),
                self._ctx.as_raw(),
            )
        };
        if inner.is_null() {
            panic!("Failed to create ARKStep solver");
        }

        unsafe {
            let flag = sundials_sys::ARKodeSetUserData(inner, user_data);
            assert_eq!(flag, sundials_sys::ARK_SUCCESS as i32);
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
            psetup: None,
            psolve: None,
        });
        let user_data = Box::into_raw(ud) as *mut c_void;

        let inner = unsafe {
            sundials_sys::ARKStepCreate(
                Some(fe_trampoline::<F, G>),
                Some(fi_trampoline::<F, G>),
                t0,
                y0.as_raw(),
                self._ctx.as_raw(),
            )
        };
        if inner.is_null() {
            panic!("Failed to create ARKStep solver");
        }

        unsafe {
            let flag = sundials_sys::ARKodeSetUserData(inner, user_data);
            assert_eq!(flag, sundials_sys::ARK_SUCCESS as i32);
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
            let flag = sundials_sys::ARKodeSStolerances(self.inner, reltol, abstol);
            assert_eq!(flag, sundials_sys::ARK_SUCCESS as i32);
        }
    }

    pub fn set_linear_solver(
        &mut self,
        linsol: &crate::linsol::DenseLinearSolver,
        mat: &crate::matrix::DenseMatrix,
    ) {
        let flag = unsafe {
            sundials_sys::ARKodeSetLinearSolver(self.inner, linsol.as_raw(), mat.as_raw())
        };
        if flag != sundials_sys::ARK_SUCCESS as i32 {
            panic!("ARKodeSetLinearSolver failed with code {}", flag);
        }
    }

    /// Set an iterative linear solver (matrix-free).
    pub fn set_iterative_linear_solver<LS: crate::linsol::IterativeSolver>(
        &mut self,
        linsol: &LS,
    ) {
        let flag = unsafe {
            sundials_sys::ARKodeSetLinearSolver(self.inner, linsol.as_raw(), ptr::null_mut())
        };
        if flag != sundials_sys::ARK_SUCCESS as i32 {
            panic!("ARKodeSetLinearSolver (iterative) failed with code {}", flag);
        }
    }

    /// Set preconditioner for use with an iterative solver.
    pub fn set_preconditioner(
        &mut self,
        psetup: impl FnMut(f64, &[f64], &[f64], bool, f64) -> Result<bool, ()> + 'static,
        psolve: impl FnMut(f64, &[f64], &[f64], &[f64], &mut [f64], f64, f64, i32) -> Result<(), ()>
            + 'static,
    ) {
        let ud = unsafe { &mut *(self.user_data as *mut UserData<F, G>) };
        ud.psetup = Some(Box::new(psetup));
        ud.psolve = Some(Box::new(psolve));

        unsafe {
            let flag = sundials_sys::ARKodeSetPreconditioner(
                self.inner,
                Some(ark_psetup_trampoline::<F, G>),
                Some(ark_psolve_trampoline::<F, G>),
            );
            assert_eq!(
                flag,
                sundials_sys::ARK_SUCCESS as i32,
                "ARKodeSetPreconditioner failed"
            );
        }
    }

    pub fn step(&mut self, tout: f64, yout: &mut NVector, tret: &mut f64) -> i32 {
        unsafe {
            sundials_sys::ARKodeEvolve(
                self.inner,
                tout,
                yout.as_raw(),
                tret,
                sundials_sys::ARK_NORMAL as i32,
            )
        }
    }

    pub fn get_num_steps(&self) -> Result<i64, i32> {
        let mut nsteps = 0;
        let flag = unsafe { sundials_sys::ARKodeGetNumSteps(self.inner, &mut nsteps) };
        if flag == sundials_sys::ARK_SUCCESS as i32 {
            Ok(nsteps as i64)
        } else {
            Err(flag)
        }
    }

    pub fn get_num_rhs_evals_explicit(&self) -> Result<i64, i32> {
        let mut nfevals = 0;
        let flag =
            unsafe { sundials_sys::ARKodeGetNumRhsEvals(self.inner, 0, &mut nfevals) };
        if flag == sundials_sys::ARK_SUCCESS as i32 {
            Ok(nfevals as i64)
        } else {
            Err(flag)
        }
    }

    pub fn get_num_rhs_evals_implicit(&self) -> Result<i64, i32> {
        let mut nfevals = 0;
        let flag =
            unsafe { sundials_sys::ARKodeGetNumRhsEvals(self.inner, 1, &mut nfevals) };
        if flag == sundials_sys::ARK_SUCCESS as i32 {
            Ok(nfevals as i64)
        } else {
            Err(flag)
        }
    }

    pub fn get_num_lin_solv_setups(&self) -> Result<i64, i32> {
        let mut nlinsetups = 0;
        let flag =
            unsafe { sundials_sys::ARKodeGetNumLinSolvSetups(self.inner, &mut nlinsetups) };
        if flag == sundials_sys::ARK_SUCCESS as i32 {
            Ok(nlinsetups as i64)
        } else {
            Err(flag)
        }
    }

    pub fn get_num_err_test_fails(&self) -> Result<i64, i32> {
        let mut netfails = 0;
        let flag =
            unsafe { sundials_sys::ARKodeGetNumErrTestFails(self.inner, &mut netfails) };
        if flag == sundials_sys::ARK_SUCCESS as i32 {
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
                sundials_sys::ARKStepFree(&mut self.inner);
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

        assert_eq!(flag, sundials_sys::ARK_SUCCESS as i32);
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

        assert_eq!(flag, sundials_sys::ARK_SUCCESS as i32);
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
