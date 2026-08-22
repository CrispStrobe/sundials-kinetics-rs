use std::os::raw::{c_char, c_double, c_int};
use std::ffi::CStr;
use crate::{ReactionSystem, Species, Phase, Reaction};
use crate::evaluator::KineticEvaluator;
use sundials::Context;
use sundials::cvode::{CvodeSolver, Lmm};
use sundials::linsol::DenseLinearSolver;
use sundials::matrix::DenseMatrix;
use sundials::nvector::NVector;

#[repr(C)]
pub struct ChemSystem {
    sys: Box<ReactionSystem>,
}

#[repr(C)]
pub struct ChemEvaluator {
    ctx: *mut Context,
    eval: *mut KineticEvaluator,
    y: *mut NVector,
    cvode: *mut CvodeSolver<'static>,
    mat: *mut DenseMatrix,
    linsol: *mut DenseLinearSolver,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn chem_system_new() -> *mut ChemSystem {
    let sys = Box::new(ReactionSystem::new());
    Box::into_raw(Box::new(ChemSystem { sys }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn chem_system_free(ptr: *mut ChemSystem) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)); }
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
    
    let mut cvode_box = Box::new(CvodeSolver::new(Lmm::Bdf, unsafe { &*ctx }));
    
    // We transmute the lifetime of the eval reference to 'static to store it in cvode
    // This is safe because we own eval and cvode, and will drop cvode before eval.
    let eval_ref: &'static KineticEvaluator = unsafe { std::mem::transmute(&*eval) };
    
    // Create the closure that cvode expects
    cvode_box.init(0.0, unsafe { &*y }, move |_t, y_slice, ydot_slice| {
        eval_ref.evaluate_rhs(y_slice, ydot_slice);
        Ok(())
    });
    
    cvode_box.set_ss_tolerances(reltol, abstol);
    
    let mat = Box::into_raw(Box::new(DenseMatrix::new(len, len, unsafe { &*ctx })));
    let linsol = Box::into_raw(Box::new(DenseLinearSolver::new(unsafe { &*y }, unsafe { &*mat }, unsafe { &*ctx })));
    cvode_box.set_linear_solver(unsafe { &*linsol }, unsafe { &*mat });
    
    let cvode = Box::into_raw(cvode_box);
    let cvode_static = unsafe { std::mem::transmute::<*mut CvodeSolver<'_>, *mut CvodeSolver<'static>>(cvode) };
    
    Box::into_raw(Box::new(ChemEvaluator {
        ctx,
        eval,
        y,
        cvode: cvode_static,
        mat,
        linsol,
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn chem_evaluator_free(ptr: *mut ChemEvaluator) {
    if !ptr.is_null() {
        unsafe {
            let ev = Box::from_raw(ptr);
            // Drop in reverse dependency order
            drop(Box::from_raw(ev.cvode));
            drop(Box::from_raw(ev.linsol));
            drop(Box::from_raw(ev.mat));
            drop(Box::from_raw(ev.y));
            drop(Box::from_raw(ev.eval));
            drop(Box::from_raw(ev.ctx));
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
    
    let cvode = unsafe { &mut *ev.cvode };
    let y = unsafe { &mut *ev.y };
    
    cvode.step(t_out, y, &mut t_ret);
    
    let slice = unsafe { std::slice::from_raw_parts_mut(out_y, y.as_slice().len()) };
    slice.copy_from_slice(y.as_slice());
    t_ret
}
