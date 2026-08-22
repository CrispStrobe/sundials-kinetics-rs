use crate::context::Context;
use crate::linsol::DenseLinearSolver;
use crate::matrix::DenseMatrix;
use crate::nvector::NVector;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use sundials_sys::{
    sunrealtype, IDACreate, IDAFree, IDAInit, IDASStolerances, IDASetLinearSolver, IDASolve,
    N_Vector, IDA_NORMAL, IDA_SUCCESS,
};

/// Forward sensitivity method for IDA.
pub enum IdaSensMethod {
    Simultaneous = 1, // IDA_SIMULTANEOUS
    Staggered = 2,    // IDA_STAGGERED
}

/// Interpolation type for adjoint checkpointing.
pub enum IdaAdjInterp {
    Hermite = 1,    // IDA_HERMITE
    Polynomial = 2, // IDA_POLYNOMIAL
}

struct UserData<F> {
    res: F,
    psetup: Option<Box<dyn FnMut(f64, &[f64], &[f64], &[f64], f64) -> Result<(), ()>>>,
    psolve: Option<
        Box<dyn FnMut(f64, &[f64], &[f64], &[f64], &[f64], &mut [f64], f64, f64) -> Result<(), ()>>,
    >,
}

pub struct IdaSolver<'a, F> {
    inner: *mut c_void,
    _ctx: &'a Context,
    user_data: *mut c_void,
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
    let ud = unsafe { &mut *(user_data as *mut UserData<F>) };

    let len = unsafe { sundials_sys::N_VGetLength_Serial(y) } as usize;
    let y_slice =
        unsafe { std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(y), len) };
    let yp_slice =
        unsafe { std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(yp), len) };
    let resval_slice = unsafe {
        std::slice::from_raw_parts_mut(sundials_sys::N_VGetArrayPointer_Serial(resval), len)
    };

    let result = catch_unwind(AssertUnwindSafe(|| {
        (ud.res)(t, y_slice, yp_slice, resval_slice)
    }));

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(())) => 1,
        Err(_) => -1,
    }
}

// ---------------------------------------------------------------------------
// IDA Preconditioner trampolines
// ---------------------------------------------------------------------------

extern "C" fn ida_psetup_trampoline<F>(
    t: sunrealtype,
    y: N_Vector,
    yp: N_Vector,
    rr: N_Vector,
    cj: sunrealtype,
    user_data: *mut c_void,
) -> i32
where
    F: FnMut(f64, &[f64], &[f64], &mut [f64]) -> Result<(), ()>,
{
    let ud = unsafe { &mut *(user_data as *mut UserData<F>) };
    if let Some(ref mut psetup) = ud.psetup {
        let len = unsafe { sundials_sys::N_VGetLength_Serial(y) } as usize;
        let y_s =
            unsafe { std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(y), len) };
        let yp_s =
            unsafe { std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(yp), len) };
        let rr_s = unsafe {
            std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(rr), len)
        };
        let result = catch_unwind(AssertUnwindSafe(|| psetup(t, y_s, yp_s, rr_s, cj)));
        match result {
            Ok(Ok(())) => 0,
            Ok(Err(())) => 1,
            Err(_) => -1,
        }
    } else {
        0
    }
}

extern "C" fn ida_psolve_trampoline<F>(
    t: sunrealtype,
    y: N_Vector,
    yp: N_Vector,
    rr: N_Vector,
    rvec: N_Vector,
    zvec: N_Vector,
    cj: sunrealtype,
    delta: sunrealtype,
    user_data: *mut c_void,
) -> i32
where
    F: FnMut(f64, &[f64], &[f64], &mut [f64]) -> Result<(), ()>,
{
    let ud = unsafe { &mut *(user_data as *mut UserData<F>) };
    if let Some(ref mut psolve) = ud.psolve {
        let len = unsafe { sundials_sys::N_VGetLength_Serial(y) } as usize;
        let y_s =
            unsafe { std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(y), len) };
        let yp_s =
            unsafe { std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(yp), len) };
        let rr_s = unsafe {
            std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(rr), len)
        };
        let r_s = unsafe {
            std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(rvec), len)
        };
        let z_s = unsafe {
            std::slice::from_raw_parts_mut(sundials_sys::N_VGetArrayPointer_Serial(zvec), len)
        };
        let result =
            catch_unwind(AssertUnwindSafe(|| psolve(t, y_s, yp_s, rr_s, r_s, z_s, cj, delta)));
        match result {
            Ok(Ok(())) => 0,
            Ok(Err(())) => 1,
            Err(_) => -1,
        }
    } else {
        -1
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
        let ud = Box::new(UserData {
            res,
            psetup: None,
            psolve: None,
        });
        let user_data_ptr = Box::into_raw(ud) as *mut c_void;
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
            user_data: user_data_ptr,
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

    /// Set a banded linear solver.
    pub fn set_band_linear_solver(
        &self,
        linsol: &crate::linsol::BandLinearSolver,
        mat: &crate::matrix::BandMatrix,
    ) {
        let flag = unsafe {
            IDASetLinearSolver(
                self.inner,
                crate::linsol::LinearSolver::as_raw(linsol),
                crate::linsol::SunMatrix::as_raw(mat),
            )
        };
        assert_eq!(flag, IDA_SUCCESS as i32);
    }

    /// Set an iterative linear solver (matrix-free).
    pub fn set_iterative_linear_solver<LS: crate::linsol::IterativeSolver>(&self, linsol: &LS) {
        let flag = unsafe {
            IDASetLinearSolver(self.inner, linsol.as_raw(), ptr::null_mut())
        };
        assert_eq!(flag, IDA_SUCCESS as i32);
    }

    /// Set preconditioner callbacks for use with an iterative linear solver.
    ///
    /// `psetup(t, y, yp, rr, cj) -> Result<(), ()>`:
    ///   Set up the preconditioner P ≈ ∂F/∂y + cj * ∂F/∂y'
    ///
    /// `psolve(t, y, yp, rr, r, z, cj, delta) -> Result<(), ()>`:
    ///   Solve P*z = r
    pub fn set_preconditioner(
        &mut self,
        psetup: impl FnMut(f64, &[f64], &[f64], &[f64], f64) -> Result<(), ()> + 'static,
        psolve: impl FnMut(f64, &[f64], &[f64], &[f64], &[f64], &mut [f64], f64, f64) -> Result<(), ()>
            + 'static,
    ) where
        F: FnMut(f64, &[f64], &[f64], &mut [f64]) -> Result<(), ()>,
    {
        let ud = unsafe { &mut *(self.user_data as *mut UserData<F>) };
        ud.psetup = Some(Box::new(psetup));
        ud.psolve = Some(Box::new(psolve));

        unsafe {
            let flag = sundials_sys::IDASetPreconditioner(
                self.inner,
                Some(ida_psetup_trampoline::<F>),
                Some(ida_psolve_trampoline::<F>),
            );
            assert_eq!(flag, IDA_SUCCESS as i32, "IDASetPreconditioner failed");
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

    // -----------------------------------------------------------------------
    // Forward Sensitivity Analysis (IDAS)
    // -----------------------------------------------------------------------

    /// Initialize forward sensitivity analysis.
    /// `ys0` and `yps0` are the initial conditions for each sensitivity.
    /// Pass `None` for the sensitivity residual to use internal difference quotients.
    pub fn init_sensitivities(
        &mut self,
        method: IdaSensMethod,
        ys0: &[NVector],
        yps0: &[NVector],
    ) {
        let ns = ys0.len() as i32;
        let mut raw_ys0: Vec<N_Vector> = ys0.iter().map(|v| v.as_raw()).collect();
        let mut raw_yps0: Vec<N_Vector> = yps0.iter().map(|v| v.as_raw()).collect();

        unsafe {
            let flag = sundials_sys::IDASensInit(
                self.inner,
                ns,
                method as i32,
                None, // use internal DQ
                raw_ys0.as_mut_ptr(),
                raw_yps0.as_mut_ptr(),
            );
            assert_eq!(flag, IDA_SUCCESS as i32, "IDASensInit failed");

            let flag = sundials_sys::IDASensEEtolerances(self.inner);
            assert_eq!(flag, IDA_SUCCESS as i32, "IDASensEEtolerances failed");
        }
    }

    /// Set sensitivity parameters.
    pub fn set_sens_params(
        &mut self,
        p: &mut [f64],
        pbar: Option<&mut [f64]>,
        plist: Option<&mut [i32]>,
    ) {
        unsafe {
            let pbar_ptr = pbar.map_or(ptr::null_mut(), |s| s.as_mut_ptr());
            let plist_ptr = plist.map_or(ptr::null_mut(), |s| s.as_mut_ptr());
            let flag =
                sundials_sys::IDASetSensParams(self.inner, p.as_mut_ptr(), pbar_ptr, plist_ptr);
            assert_eq!(flag, IDA_SUCCESS as i32, "IDASetSensParams failed");
        }
    }

    /// Get sensitivity solutions.
    pub fn get_sens(&self, tret: &mut f64, ys: &mut [NVector]) -> i32 {
        let mut raw: Vec<N_Vector> = ys.iter_mut().map(|v| v.as_raw()).collect();
        unsafe { sundials_sys::IDAGetSens(self.inner, tret, raw.as_mut_ptr()) }
    }

    // -----------------------------------------------------------------------
    // Adjoint Sensitivity Analysis (IDAS)
    // -----------------------------------------------------------------------

    /// Initialize adjoint sensitivity computation.
    pub fn adj_init(&mut self, ncheck: i64, interp: IdaAdjInterp) {
        let flag =
            unsafe { sundials_sys::IDAAdjInit(self.inner, ncheck, interp as i32) };
        assert_eq!(flag, IDA_SUCCESS as i32, "IDAAdjInit failed");
    }

    /// Forward integration with checkpointing.
    pub fn forward(
        &mut self,
        tout: f64,
        yout: &mut NVector,
        ypout: &mut NVector,
        tret: &mut f64,
    ) -> (i32, i32) {
        let mut ncheck: i32 = 0;
        let flag = unsafe {
            sundials_sys::IDACalcIC(self.inner, 1, 0.001); // IDA_YA_YDP_INIT = 1
            let _ = 0; // ignore CalcIC result for now
            sundials_sys::IDASolveF(
                self.inner,
                tout,
                tret,
                yout.as_raw(),
                ypout.as_raw(),
                IDA_NORMAL as i32,
                &mut ncheck,
            )
        };
        (flag, ncheck)
    }

    /// Create a backward problem. Returns the `which` index.
    pub fn create_backward(&mut self) -> i32 {
        let mut which: i32 = 0;
        let flag = unsafe { sundials_sys::IDACreateB(self.inner, &mut which) };
        assert_eq!(flag, IDA_SUCCESS as i32, "IDACreateB failed");
        which
    }

    /// Initialize a backward problem.
    pub fn init_backward<FB>(
        &mut self,
        which: i32,
        tb0: f64,
        yb0: &NVector,
        ypb0: &NVector,
        res_b: &mut FB,
    ) where
        FB: FnMut(f64, &[f64], &[f64], &[f64], &[f64], &mut [f64]) -> Result<(), ()>,
    {
        extern "C" fn adj_res_trampoline<FB>(
            t: sunrealtype,
            y: N_Vector,
            yp: N_Vector,
            yb: N_Vector,
            ypb: N_Vector,
            resb: N_Vector,
            user_data: *mut c_void,
        ) -> i32
        where
            FB: FnMut(f64, &[f64], &[f64], &[f64], &[f64], &mut [f64]) -> Result<(), ()>,
        {
            let closure = unsafe { &mut *(user_data as *mut FB) };
            let len = unsafe { sundials_sys::N_VGetLength_Serial(y) } as usize;
            let lenb = unsafe { sundials_sys::N_VGetLength_Serial(yb) } as usize;
            let y_s = unsafe {
                std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(y), len)
            };
            let yp_s = unsafe {
                std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(yp), len)
            };
            let yb_s = unsafe {
                std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(yb), lenb)
            };
            let ypb_s = unsafe {
                std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer_Serial(ypb), lenb)
            };
            let resb_s = unsafe {
                std::slice::from_raw_parts_mut(
                    sundials_sys::N_VGetArrayPointer_Serial(resb),
                    lenb,
                )
            };
            let result =
                catch_unwind(AssertUnwindSafe(|| closure(t, y_s, yp_s, yb_s, ypb_s, resb_s)));
            match result {
                Ok(Ok(())) => 0,
                Ok(Err(())) => 1,
                Err(_) => -1,
            }
        }

        let user_b = res_b as *mut FB as *mut c_void;
        unsafe {
            let flag = sundials_sys::IDAInitB(
                self.inner,
                which,
                Some(adj_res_trampoline::<FB>),
                tb0,
                yb0.as_raw(),
                ypb0.as_raw(),
            );
            assert_eq!(flag, IDA_SUCCESS as i32, "IDAInitB failed");

            sundials_sys::IDASetUserDataB(self.inner, which, user_b);
        }
    }

    /// Set scalar tolerances for a backward problem.
    pub fn set_ss_tolerances_b(&mut self, which: i32, reltol: f64, abstol: f64) {
        let flag =
            unsafe { sundials_sys::IDASStolerancesB(self.inner, which, reltol, abstol) };
        assert_eq!(flag, IDA_SUCCESS as i32, "IDASStolerancesB failed");
    }

    /// Set linear solver for a backward problem.
    pub fn set_linear_solver_b(
        &mut self,
        which: i32,
        linsol: &DenseLinearSolver,
        mat: &DenseMatrix,
    ) {
        let flag = unsafe {
            sundials_sys::IDASetLinearSolverB(
                self.inner,
                which,
                linsol.as_raw(),
                mat.as_raw(),
            )
        };
        assert_eq!(flag, IDA_SUCCESS as i32, "IDASetLinearSolverB failed");
    }

    /// Integrate backward.
    pub fn backward(&mut self, tbout: f64) -> i32 {
        unsafe { sundials_sys::IDASolveB(self.inner, tbout, IDA_NORMAL as i32) }
    }

    /// Get backward solution.
    pub fn get_backward(
        &self,
        which: i32,
        tret: &mut f64,
        yb: &mut NVector,
        ypb: &mut NVector,
    ) -> i32 {
        unsafe {
            sundials_sys::IDAGetB(self.inner, which, tret, yb.as_raw(), ypb.as_raw())
        }
    }
}

impl<'a, F> Drop for IdaSolver<'a, F> {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                IDAFree(&mut self.inner);
                let _ = Box::from_raw(self.user_data as *mut UserData<F>);
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

    #[test]
    fn test_ida_with_spgmr() {
        use crate::linsol::{PrecType, SpgmrSolver};

        let ctx = Context::new();
        let mut y = NVector::new_serial(1, &ctx);
        let mut yp = NVector::new_serial(1, &ctx);
        y.as_mut_slice()[0] = 1.0;
        yp.as_mut_slice()[0] = -0.5;

        let mut solver = IdaBuilder::new(&ctx).init(0.0, &y, &yp, |_t, y_val, yp_val, res| {
            res[0] = yp_val[0] + 0.5 * y_val[0];
            Ok(())
        });

        solver.set_ss_tolerances(1e-4, 1e-4);

        let ls = SpgmrSolver::new(&y, PrecType::None, 0, &ctx);
        solver.set_iterative_linear_solver(&ls);

        let mut tret = 0.0;
        let flag = solver.solve(1.0, &mut y, &mut yp, &mut tret);
        assert_eq!(flag, IDA_SUCCESS as i32);
        let expected = (-0.5_f64).exp();
        assert!(
            (y.as_slice()[0] - expected).abs() < 1e-3,
            "Got {}",
            y.as_slice()[0]
        );
    }
}
