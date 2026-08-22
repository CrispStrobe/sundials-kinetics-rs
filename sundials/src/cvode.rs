use crate::context::Context;
use crate::nvector::NVector;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use sundials_sys::{
    sunrealtype, CVode, CVodeCreate, CVodeFree, CVodeInit, CVodeSStolerances,
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

/// Interpolation type for adjoint checkpointing.
pub enum AdjInterp {
    Hermite = 1,    // CV_HERMITE
    Polynomial = 2, // CV_POLYNOMIAL
}

struct UserData<F, G> {
    rhs: F,
    sens_rhs: Option<G>,
    // Preconditioner closures (only used with iterative solvers)
    psetup: Option<Box<dyn FnMut(f64, &[f64], &[f64], bool, f64) -> Result<bool, ()>>>,
    psolve: Option<
        Box<dyn FnMut(f64, &[f64], &[f64], &[f64], &mut [f64], f64, f64, i32) -> Result<(), ()>>,
    >,
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
    _ys_1d: *mut N_Vector,
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

// ---------------------------------------------------------------------------
// Preconditioner trampolines
// ---------------------------------------------------------------------------

extern "C" fn psetup_trampoline<F, G>(
    t: sunrealtype,
    y: N_Vector,
    fy: N_Vector,
    jok: i32,
    jcur_ptr: *mut i32,
    gamma: sunrealtype,
    user_data: *mut c_void,
) -> i32
where
    F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), ()>,
{
    let ud = unsafe { &mut *(user_data as *mut UserData<F, G>) };
    if let Some(ref mut psetup) = ud.psetup {
        let len = unsafe { sundials_sys::N_VGetLength_Serial(y) } as usize;
        let y_slice =
            unsafe { std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(y), len) };
        let fy_slice =
            unsafe { std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(fy), len) };

        let result =
            catch_unwind(AssertUnwindSafe(|| psetup(t, y_slice, fy_slice, jok != 0, gamma)));
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

extern "C" fn psolve_trampoline<F, G>(
    t: sunrealtype,
    y: N_Vector,
    fy: N_Vector,
    r: N_Vector,
    z: N_Vector,
    gamma: sunrealtype,
    delta: sunrealtype,
    lr: i32,
    user_data: *mut c_void,
) -> i32
where
    F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), ()>,
{
    let ud = unsafe { &mut *(user_data as *mut UserData<F, G>) };
    if let Some(ref mut psolve) = ud.psolve {
        let len = unsafe { sundials_sys::N_VGetLength_Serial(y) } as usize;
        let y_slice =
            unsafe { std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(y), len) };
        let fy_slice =
            unsafe { std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(fy), len) };
        let r_slice =
            unsafe { std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(r), len) };
        let z_slice = unsafe {
            std::slice::from_raw_parts_mut(sundials_sys::N_VGetArrayPointer_Serial(z), len)
        };

        let result = catch_unwind(AssertUnwindSafe(|| {
            psolve(t, y_slice, fy_slice, r_slice, z_slice, gamma, delta, lr)
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
            psetup: None,
            psolve: None,
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
            psetup: old_ud.psetup,
            psolve: old_ud.psolve,
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

    /// Set a banded linear solver.
    pub fn set_band_linear_solver(
        &mut self,
        linsol: &crate::linsol::BandLinearSolver,
        mat: &crate::matrix::BandMatrix,
    ) {
        let flag = unsafe {
            CVodeSetLinearSolver(
                self.inner,
                crate::linsol::LinearSolver::as_raw(linsol),
                crate::linsol::SunMatrix::as_raw(mat),
            )
        };
        if flag != CV_SUCCESS as i32 {
            panic!("CVodeSetLinearSolver (band) failed with code {}", flag);
        }
    }

    /// Set an iterative linear solver (SPGMR, SPBCGS, or SPTFQMR).
    /// Pass `None` for the matrix to use a matrix-free Newton-Krylov method.
    pub fn set_iterative_linear_solver<LS: crate::linsol::IterativeSolver>(
        &mut self,
        linsol: &LS,
    ) {
        let flag = unsafe {
            CVodeSetLinearSolver(self.inner, linsol.as_raw(), ptr::null_mut())
        };
        if flag != CV_SUCCESS as i32 {
            panic!("CVodeSetLinearSolver (iterative) failed with code {}", flag);
        }
    }

    /// Set preconditioner callbacks for use with an iterative linear solver.
    ///
    /// `psetup(t, y, fy, jok, gamma) -> Result<jcur, ()>`:
    ///   - `jok`: true if the Jacobian data is still current
    ///   - Returns `Ok(jcur)` where `jcur` = true if the Jacobian was recomputed
    ///
    /// `psolve(t, y, fy, r, z, gamma, delta, lr) -> Result<(), ()>`:
    ///   - Solves P*z = r where P approximates I - gamma*J
    ///   - `lr` = 1 for left, 2 for right preconditioning
    pub fn set_preconditioner(
        &mut self,
        psetup: impl FnMut(f64, &[f64], &[f64], bool, f64) -> Result<bool, ()> + 'static,
        psolve: impl FnMut(f64, &[f64], &[f64], &[f64], &mut [f64], f64, f64, i32) -> Result<(), ()>
            + 'static,
    ) where
        F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), ()>,
    {
        let ud = unsafe { &mut *(self.user_data as *mut UserData<F, G>) };
        ud.psetup = Some(Box::new(psetup));
        ud.psolve = Some(Box::new(psolve));

        unsafe {
            let flag = sundials_sys::CVodeSetPreconditioner(
                self.inner,
                Some(psetup_trampoline::<F, G>),
                Some(psolve_trampoline::<F, G>),
            );
            assert_eq!(flag, CV_SUCCESS as i32, "CVodeSetPreconditioner failed");
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

    // -----------------------------------------------------------------------
    // Adjoint sensitivity analysis (CVODES)
    // -----------------------------------------------------------------------

    /// Initialize adjoint sensitivity computation.
    /// `ncheck` is the number of checkpointing steps between check points.
    pub fn adj_init(&mut self, ncheck: i64, interp: AdjInterp) {
        let flag =
            unsafe { sundials_sys::CVodeAdjInit(self.inner, ncheck, interp as i32) };
        assert_eq!(flag, CV_SUCCESS as i32, "CVodeAdjInit failed");
    }

    /// Forward integration with checkpointing for adjoint computation.
    /// Returns `(flag, ncheck)` where `ncheck` is the number of checkpoints stored.
    pub fn forward(
        &mut self,
        tout: f64,
        yout: &mut NVector,
        tret: &mut f64,
    ) -> (i32, i32) {
        let mut ncheck: i32 = 0;
        let flag = unsafe {
            sundials_sys::CVodeF(self.inner, tout, yout.as_raw(), tret, 1, &mut ncheck)
        };
        (flag, ncheck)
    }

    /// Create a backward problem. Returns the `which` index.
    pub fn create_backward(&mut self, lmm: Lmm) -> i32 {
        let mut which: i32 = 0;
        let flag = unsafe {
            sundials_sys::CVodeCreateB(self.inner, lmm as i32, &mut which)
        };
        assert_eq!(flag, CV_SUCCESS as i32, "CVodeCreateB failed");
        which
    }

    /// Initialize a backward problem with a RHS that does NOT depend on
    /// forward sensitivities.
    ///
    /// # Safety
    /// The `rhs_b` closure is stored as a raw pointer; the caller must ensure
    /// it lives until `backward()` completes.
    pub fn init_backward<FB>(
        &mut self,
        which: i32,
        tb0: f64,
        yb0: &NVector,
        rhs_b: &mut FB,
    ) where
        FB: FnMut(f64, &[f64], &[f64], &mut [f64]) -> Result<(), ()>,
    {
        extern "C" fn adj_rhs_trampoline<FB>(
            t: sunrealtype,
            y: N_Vector,
            yb: N_Vector,
            ybdot: N_Vector,
            user_data: *mut c_void,
        ) -> i32
        where
            FB: FnMut(f64, &[f64], &[f64], &mut [f64]) -> Result<(), ()>,
        {
            let closure = unsafe { &mut *(user_data as *mut FB) };
            let len = unsafe { sundials_sys::N_VGetLength_Serial(y) } as usize;
            let lenb = unsafe { sundials_sys::N_VGetLength_Serial(yb) } as usize;
            let y_s = unsafe {
                std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(y), len)
            };
            let yb_s = unsafe {
                std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(yb), lenb)
            };
            let ybdot_s = unsafe {
                std::slice::from_raw_parts_mut(
                    sundials_sys::N_VGetArrayPointer_Serial(ybdot),
                    lenb,
                )
            };
            let result = catch_unwind(AssertUnwindSafe(|| closure(t, y_s, yb_s, ybdot_s)));
            match result {
                Ok(Ok(())) => 0,
                Ok(Err(())) => 1,
                Err(_) => -1,
            }
        }

        let user_b = rhs_b as *mut FB as *mut c_void;
        unsafe {
            let flag = sundials_sys::CVodeInitB(
                self.inner,
                which,
                Some(adj_rhs_trampoline::<FB>),
                tb0,
                yb0.as_raw(),
            );
            assert_eq!(flag, CV_SUCCESS as i32, "CVodeInitB failed");

            sundials_sys::CVodeSetUserDataB(self.inner, which, user_b);
        }
    }

    /// Set scalar tolerances for a backward problem.
    pub fn set_ss_tolerances_b(&mut self, which: i32, reltol: f64, abstol: f64) {
        let flag =
            unsafe { sundials_sys::CVodeSStolerancesB(self.inner, which, reltol, abstol) };
        assert_eq!(flag, CV_SUCCESS as i32, "CVodeSStolerancesB failed");
    }

    /// Set linear solver for a backward problem.
    pub fn set_linear_solver_b(
        &mut self,
        which: i32,
        linsol: &crate::linsol::DenseLinearSolver,
        mat: &crate::matrix::DenseMatrix,
    ) {
        let flag = unsafe {
            sundials_sys::CVodeSetLinearSolverB(
                self.inner,
                which,
                linsol.as_raw(),
                mat.as_raw(),
            )
        };
        assert_eq!(
            flag, CV_SUCCESS as i32,
            "CVodeSetLinearSolverB failed"
        );
    }

    /// Integrate all backward problems from `tbout` backward in time.
    pub fn backward(&mut self, tbout: f64) -> i32 {
        // CV_NORMAL = 1
        unsafe { sundials_sys::CVodeB(self.inner, tbout, 1) }
    }

    /// Get the backward solution for a given backward problem index.
    pub fn get_backward(&self, which: i32, tret: &mut f64, yb: &mut NVector) -> i32 {
        unsafe { sundials_sys::CVodeGetB(self.inner, which, tret, yb.as_raw()) }
    }

    /// Get the forward solution at a given time (interpolated from checkpoints).
    pub fn get_adj_y(&self, t: f64, y: &mut NVector) -> i32 {
        unsafe { sundials_sys::CVodeGetAdjY(self.inner, t, y.as_raw()) }
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

    #[test]
    fn test_cvode_with_spgmr() {
        use crate::linsol::{PrecType, SpgmrSolver};

        let ctx = Context::new();
        let mut y = NVector::new_serial(1, &ctx);
        y.as_mut_slice()[0] = 1.0;

        let mut solver = CvodeBuilder::new(Lmm::Bdf, &ctx).init(0.0, &y, |_t, y_val, ydot| {
            ydot[0] = -0.5 * y_val[0];
            Ok(())
        });

        solver.set_ss_tolerances(1e-6, 1e-6);

        let ls = SpgmrSolver::new(&y, PrecType::None, 0, &ctx);
        solver.set_iterative_linear_solver(&ls);

        let mut tret = 0.0;
        let flag = solver.step(1.0, &mut y, &mut tret);
        assert_eq!(flag, CV_SUCCESS as i32);
        let expected = (-0.5_f64).exp();
        assert!(
            (y.as_slice()[0] - expected).abs() < 1e-4,
            "Got {}",
            y.as_slice()[0]
        );
    }

    #[test]
    fn test_cvode_adjoint_simple() {
        // Solve dy/dt = -y forward, then compute dg/dy0 where g = y(T)
        // Analytical: y(T) = y0 * exp(-T), so dg/dy0 = exp(-T)
        let ctx = Context::new();
        let mut y = NVector::new_serial(1, &ctx);
        y.as_mut_slice()[0] = 1.0;

        let mut solver = CvodeBuilder::new(Lmm::Bdf, &ctx).init(0.0, &y, |_t, yv, ydot| {
            ydot[0] = -yv[0];
            Ok(())
        });
        solver.set_ss_tolerances(1e-8, 1e-10);
        let mat = DenseMatrix::new(1, 1, &ctx);
        let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
        solver.set_linear_solver(&linsol, &mat);

        // Initialize adjoint with 100 checkpoint steps
        solver.adj_init(100, AdjInterp::Hermite);

        // Forward integration to T=1
        let mut tret = 0.0;
        let (flag, _ncheck) = solver.forward(1.0, &mut y, &mut tret);
        assert_eq!(flag, CV_SUCCESS as i32);

        // Create backward problem: dλ/dt = λ (adjoint of dy/dt = -y)
        let which = solver.create_backward(Lmm::Bdf);

        let mut yb = NVector::new_serial(1, &ctx);
        yb.as_mut_slice()[0] = 1.0; // dg/dy(T) = 1

        let mut rhs_b =
            |_t: f64, _y: &[f64], yb: &[f64], ybdot: &mut [f64]| -> Result<(), ()> {
                // Adjoint RHS: dλ/dt = -∂f/∂y^T * λ = -(-1) * λ = λ
                ybdot[0] = yb[0];
                Ok(())
            };

        solver.init_backward(which, 1.0, &yb, &mut rhs_b);
        solver.set_ss_tolerances_b(which, 1e-8, 1e-10);

        let mat_b = DenseMatrix::new(1, 1, &ctx);
        let linsol_b = DenseLinearSolver::new(&yb, &mat_b, &ctx);
        solver.set_linear_solver_b(which, &linsol_b, &mat_b);

        // Backward integration to t=0
        let flag = solver.backward(0.0);
        assert!(
            flag == CV_SUCCESS as i32 || flag == 1, // CV_TSTOP_RETURN = 1
            "CVodeB returned error code {}",
            flag
        );

        let mut tb_ret = 0.0;
        solver.get_backward(which, &mut tb_ret, &mut yb);

        // dg/dy0 = exp(-T) * exp(T) = 1... wait, let me reconsider.
        // Actually: λ(0) should equal exp(-1) since:
        // y(t) = y0*exp(-t), g = y(1) = y0*exp(-1), dg/dy0 = exp(-1)
        // The adjoint equation is dλ/dt = -(-1)*λ = λ with λ(1) = 1
        // Integrating backward: λ(t) = exp(-(1-t)) = exp(t-1)
        // So λ(0) = exp(-1) ≈ 0.3679
        let expected = (-1.0_f64).exp();
        let actual = yb.as_slice()[0];
        assert!(
            (actual - expected).abs() < 1e-4,
            "Adjoint: expected {}, got {}",
            expected,
            actual
        );
    }
}
