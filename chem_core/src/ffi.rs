use crate::evaluator::KineticEvaluator;
use crate::{Phase, Reaction, ReactionSystem, Species};
use std::ffi::{c_void, CStr};
use std::os::raw::{c_char, c_double, c_int};
use sundials::cvode::{CvodeBuilder, Lmm};
use sundials::linsol::DenseLinearSolver;
use sundials::matrix::DenseMatrix;
use sundials::nvector::NVector;
use sundials::Context;

#[repr(C)]
pub struct ChemSystem {
    sys: Box<ReactionSystem>,
}

/// Opaque evaluator handle. Internally stores type-erased solver resources.
pub struct ChemEvaluator {
    // The solver and its closure are type-erased behind a vtable.
    // step_fn calls CvodeSolver::step on the inner solver.
    inner: *mut c_void,
    step_fn: unsafe fn(*mut c_void, f64, *mut NVector, *mut f64) -> i32,
    num_steps_fn: unsafe fn(*mut c_void) -> i64,
    num_rhs_fn: unsafe fn(*mut c_void) -> i64,
    drop_fn: unsafe fn(*mut c_void),
    y: *mut NVector,
    n: usize,
    _ctx: *mut Context,
    _eval: *mut KineticEvaluator,
    _mat: *mut DenseMatrix,
    _linsol: *mut DenseLinearSolver,
}


#[unsafe(no_mangle)]
pub unsafe extern "C" fn chem_system_new() -> *mut ChemSystem {
    let sys = Box::new(ReactionSystem::new());
    Box::into_raw(Box::new(ChemSystem { sys }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn chem_system_free(ptr: *mut ChemSystem) {
    if !ptr.is_null() {
        unsafe {
            drop(Box::from_raw(ptr));
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn chem_system_add_species(
    ptr: *mut ChemSystem,
    name: *const c_char,
    mass: c_double,
    charge: c_int,
    phase: c_int,
    fixed: c_int,
) -> usize {
    let sys = unsafe { &mut (*ptr).sys };
    let name_str = unsafe { CStr::from_ptr(name) }.to_str().unwrap();
    let p = match phase {
        1 => Phase::Gas,
        2 => Phase::Solid,
        _ => Phase::Aqueous,
    };
    let mut sp = Species::new(name_str, mass, charge, p);
    if fixed != 0 {
        sp = sp.set_fixed(true);
    }
    sys.add_species(sp)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn chem_system_add_reaction(
    ptr: *mut ChemSystem,
    rate_constant: c_double,
    num_reactants: usize,
    reactant_indices: *const usize,
    reactant_coeffs: *const c_double,
    num_products: usize,
    product_indices: *const usize,
    product_coeffs: *const c_double,
) {
    let sys = unsafe { &mut (*ptr).sys };
    let mut rxn = Reaction::new(rate_constant);

    let r_idx = unsafe { std::slice::from_raw_parts(reactant_indices, num_reactants) };
    let r_coeff = unsafe { std::slice::from_raw_parts(reactant_coeffs, num_reactants) };
    for i in 0..num_reactants {
        rxn = rxn.add_reactant(r_idx[i], r_coeff[i]);
    }

    let p_idx = unsafe { std::slice::from_raw_parts(product_indices, num_products) };
    let p_coeff = unsafe { std::slice::from_raw_parts(product_coeffs, num_products) };
    for i in 0..num_products {
        rxn = rxn.add_product(p_idx[i], p_coeff[i]);
    }

    sys.add_reaction(rxn);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn chem_evaluator_new(
    sys_ptr: *mut ChemSystem,
    initial_conditions: *const c_double,
    len: usize,
    reltol: c_double,
    abstol: c_double,
) -> *mut ChemEvaluator {
    let sys = unsafe { &(*sys_ptr).sys };
    let ctx = Box::into_raw(Box::new(Context::new()));

    let eval = Box::into_raw(Box::new(KineticEvaluator::new(sys.as_ref().clone())));
    let slice = unsafe { std::slice::from_raw_parts(initial_conditions, len) };
    let mut y_box = Box::new(NVector::new_serial(len, unsafe { &*ctx }));
    y_box.as_mut_slice().copy_from_slice(slice);
    let y = Box::into_raw(y_box);

    // Transmute the eval reference to 'static. Safe because we own eval and
    // will drop the solver before eval (controlled by drop order in evaluator_free).
    let eval_ref: &'static KineticEvaluator = unsafe { std::mem::transmute(&*eval) };

    let rhs_closure = move |_t: f64, y_slice: &[f64], ydot_slice: &mut [f64]| -> Result<(), ()> {
        eval_ref.evaluate_rhs(y_slice, ydot_slice);
        Ok(())
    };

    let mut solver =
        CvodeBuilder::new(Lmm::Bdf, unsafe { &*ctx }).init(0.0, unsafe { &*y }, rhs_closure);

    solver.set_ss_tolerances(reltol, abstol);

    let mat = Box::into_raw(Box::new(DenseMatrix::new(len, len, unsafe { &*ctx })));
    let linsol = Box::into_raw(Box::new(DenseLinearSolver::new(
        unsafe { &*y },
        unsafe { &*mat },
        unsafe { &*ctx },
    )));
    solver.set_linear_solver(unsafe { &*linsol }, unsafe { &*mat });

    fn create_evaluator_inner<'a, F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), ()>>(
        solver: sundials::cvode::CvodeSolver<'a, F, ()>,
        y: *mut NVector,
        n: usize,
        ctx: *mut Context,
        eval: *mut KineticEvaluator,
        mat: *mut DenseMatrix,
        linsol: *mut DenseLinearSolver,
    ) -> *mut ChemEvaluator {
        unsafe fn step_fn<F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), ()>>(
            ptr: *mut c_void, tout: f64, y: *mut NVector, tret: *mut f64,
        ) -> i32 {
            let s = unsafe { &mut *(ptr as *mut sundials::cvode::CvodeSolver<'static, F, ()>) };
            s.step(tout, unsafe { &mut *y }, unsafe { &mut *tret })
        }
        unsafe fn num_steps_fn<F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), ()>>(
            ptr: *mut c_void,
        ) -> i64 {
            let s = unsafe { &*(ptr as *const sundials::cvode::CvodeSolver<'static, F, ()>) };
            s.get_num_steps().unwrap_or(-1)
        }
        unsafe fn num_rhs_fn<F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), ()>>(
            ptr: *mut c_void,
        ) -> i64 {
            let s = unsafe { &*(ptr as *const sundials::cvode::CvodeSolver<'static, F, ()>) };
            s.get_num_rhs_evals().unwrap_or(-1)
        }
        unsafe fn drop_fn<F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), ()>>(
            ptr: *mut c_void,
        ) {
            unsafe { drop(Box::from_raw(ptr as *mut sundials::cvode::CvodeSolver<'static, F, ()>)) }
        }

        // Transmute solver lifetime to 'static for FFI storage.
        // Safe because we control the drop order in evaluator_free.
        let solver: sundials::cvode::CvodeSolver<'static, F, ()> =
            unsafe { std::mem::transmute(solver) };
        let inner = Box::into_raw(Box::new(solver)) as *mut c_void;
        Box::into_raw(Box::new(ChemEvaluator {
            inner,
            step_fn: step_fn::<F>,
            num_steps_fn: num_steps_fn::<F>,
            num_rhs_fn: num_rhs_fn::<F>,
            drop_fn: drop_fn::<F>,
            y,
            n,
            _ctx: ctx,
            _eval: eval,
            _mat: mat,
            _linsol: linsol,
        }))
    }

    create_evaluator_inner::<_>(solver, y, len, ctx, eval, mat, linsol)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn chem_evaluator_free(ptr: *mut ChemEvaluator) {
    if !ptr.is_null() {
        unsafe {
            let ev = Box::from_raw(ptr);
            // Drop solver first (it borrows ctx/y/mat/linsol)
            (ev.drop_fn)(ev.inner);
            drop(Box::from_raw(ev._linsol));
            drop(Box::from_raw(ev._mat));
            drop(Box::from_raw(ev.y));
            drop(Box::from_raw(ev._eval));
            drop(Box::from_raw(ev._ctx));
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn chem_evaluator_step(
    ptr: *mut ChemEvaluator,
    t_out: c_double,
    out_y: *mut c_double,
) -> c_double {
    let ev = unsafe { &mut (*ptr) };
    let mut t_ret: f64 = 0.0;

    unsafe { (ev.step_fn)(ev.inner, t_out, ev.y, &mut t_ret) };

    let y = unsafe { &*ev.y };
    let slice = unsafe { std::slice::from_raw_parts_mut(out_y, ev.n) };
    slice.copy_from_slice(y.as_slice());
    t_ret
}

/// Get the number of internal steps taken by the solver.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chem_evaluator_get_num_steps(ptr: *const ChemEvaluator) -> i64 {
    let ev = unsafe { &*ptr };
    unsafe { (ev.num_steps_fn)(ev.inner) }
}

/// Get the number of RHS evaluations.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn chem_evaluator_get_num_rhs_evals(ptr: *const ChemEvaluator) -> i64 {
    let ev = unsafe { &*ptr };
    unsafe { (ev.num_rhs_fn)(ev.inner) }
}
