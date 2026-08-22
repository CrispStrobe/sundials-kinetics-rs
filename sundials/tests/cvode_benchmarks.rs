use sundials::{Context, NVector, CvodeSolver, Lmm, DenseMatrix, DenseLinearSolver};
use sundials_sys::CV_SUCCESS;

#[test]
fn test_exponential_decay() {
    let ctx = Context::new();
    let mut y = NVector::new_serial(1, &ctx);
    y.as_mut_slice().copy_from_slice(&[1.0]);
    let mut solver = CvodeSolver::new(Lmm::Adams, &ctx);
    solver.init(0.0, &y, |_t, y, ydot| { ydot[0] = -0.5 * y[0]; Ok(()) });
    solver.set_ss_tolerances(1e-4, 1e-8);
    let mat = DenseMatrix::new(1, 1, &ctx);
    let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
    solver.set_linear_solver(&linsol, &mat);
    let mut tret = 0.0;
    assert_eq!(solver.step(1.0, &mut y, &mut tret), CV_SUCCESS as i32);
}

#[test]
fn test_harmonic_oscillator() {
    let ctx = Context::new();
    let mut y = NVector::new_serial(2, &ctx);
    y.as_mut_slice().copy_from_slice(&[1.0, 0.0]);
    let mut solver = CvodeSolver::new(Lmm::Adams, &ctx);
    solver.init(0.0, &y, |_t, y, ydot| { 
        ydot[0] = y[1];
        ydot[1] = -y[0];
        Ok(()) 
    });
    solver.set_ss_tolerances(1e-4, 1e-8);
    let mat = DenseMatrix::new(2, 2, &ctx);
    let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
    solver.set_linear_solver(&linsol, &mat);
    let mut tret = 0.0;
    assert_eq!(solver.step(1.0, &mut y, &mut tret), CV_SUCCESS as i32);
}

#[test]
fn test_van_der_pol() {
    let ctx = Context::new();
    let mut y = NVector::new_serial(2, &ctx);
    y.as_mut_slice().copy_from_slice(&[2.0, 0.0]);
    let mut solver = CvodeSolver::new(Lmm::Bdf, &ctx);
    solver.init(0.0, &y, |_t, y, ydot| { 
        ydot[0] = y[1];
        ydot[1] = 1.0 * (1.0 - y[0]*y[0])*y[1] - y[0];
        Ok(()) 
    });
    solver.set_ss_tolerances(1e-4, 1e-8);
    let mat = DenseMatrix::new(2, 2, &ctx);
    let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
    solver.set_linear_solver(&linsol, &mat);
    let mut tret = 0.0;
    assert_eq!(solver.step(1.0, &mut y, &mut tret), CV_SUCCESS as i32);
}

#[test]
fn test_lotka_volterra() {
    let ctx = Context::new();
    let mut y = NVector::new_serial(2, &ctx);
    y.as_mut_slice().copy_from_slice(&[10.0, 10.0]);
    let mut solver = CvodeSolver::new(Lmm::Bdf, &ctx);
    solver.init(0.0, &y, |_t, y, ydot| { 
        ydot[0] = 1.5 * y[0] - 0.1 * y[0] * y[1];
        ydot[1] = 0.1 * y[0] * y[1] - 1.5 * y[1];
        Ok(()) 
    });
    solver.set_ss_tolerances(1e-4, 1e-8);
    let mat = DenseMatrix::new(2, 2, &ctx);
    let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
    solver.set_linear_solver(&linsol, &mat);
    let mut tret = 0.0;
    assert_eq!(solver.step(1.0, &mut y, &mut tret), CV_SUCCESS as i32);
}

#[test]
fn test_brusselator() {
    let ctx = Context::new();
    let mut y = NVector::new_serial(2, &ctx);
    y.as_mut_slice().copy_from_slice(&[1.5, 3.0]);
    let mut solver = CvodeSolver::new(Lmm::Bdf, &ctx);
    solver.init(0.0, &y, |_t, y, ydot| { 
        ydot[0] = 1.0 - 4.0 * y[0] + y[0]*y[0]*y[1];
        ydot[1] = 3.0 * y[0] - y[0]*y[0]*y[1];
        Ok(()) 
    });
    solver.set_ss_tolerances(1e-4, 1e-8);
    let mat = DenseMatrix::new(2, 2, &ctx);
    let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
    solver.set_linear_solver(&linsol, &mat);
    let mut tret = 0.0;
    assert_eq!(solver.step(1.0, &mut y, &mut tret), CV_SUCCESS as i32);
}

#[test]
fn test_oregonator() {
    let ctx = Context::new();
    let mut y = NVector::new_serial(3, &ctx);
    y.as_mut_slice().copy_from_slice(&[1.0, 2.0, 3.0]);
    let mut solver = CvodeSolver::new(Lmm::Bdf, &ctx);
    solver.init(0.0, &y, |_t, y, ydot| { 
        ydot[0] = 77.27 * (y[1] + y[0] * (1.0 - 8.375e-6 * y[0] - y[1]));
        ydot[1] = 1.0/77.27 * (y[2] - (1.0 + y[0]) * y[1]);
        ydot[2] = 0.161 * (y[0] - y[2]);
        Ok(()) 
    });
    solver.set_ss_tolerances(1e-4, 1e-8);
    let mat = DenseMatrix::new(3, 3, &ctx);
    let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
    solver.set_linear_solver(&linsol, &mat);
    let mut tret = 0.0;
    assert_eq!(solver.step(1.0, &mut y, &mut tret), CV_SUCCESS as i32);
}

#[test]
fn test_lorenz_system() {
    let ctx = Context::new();
    let mut y = NVector::new_serial(3, &ctx);
    y.as_mut_slice().copy_from_slice(&[1.0, 1.0, 1.0]);
    let mut solver = CvodeSolver::new(Lmm::Bdf, &ctx);
    solver.init(0.0, &y, |_t, y, ydot| { 
        ydot[0] = 10.0 * (y[1] - y[0]);
        ydot[1] = y[0] * (28.0 - y[2]) - y[1];
        ydot[2] = y[0] * y[1] - 8.0/3.0 * y[2];
        Ok(()) 
    });
    solver.set_ss_tolerances(1e-4, 1e-8);
    let mat = DenseMatrix::new(3, 3, &ctx);
    let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
    solver.set_linear_solver(&linsol, &mat);
    let mut tret = 0.0;
    assert_eq!(solver.step(1.0, &mut y, &mut tret), CV_SUCCESS as i32);
}

#[test]
fn test_sir_model() {
    let ctx = Context::new();
    let mut y = NVector::new_serial(3, &ctx);
    y.as_mut_slice().copy_from_slice(&[0.99, 0.01, 0.0]);
    let mut solver = CvodeSolver::new(Lmm::Bdf, &ctx);
    solver.init(0.0, &y, |_t, y, ydot| { 
        ydot[0] = -0.3 * y[0] * y[1];
        ydot[1] = 0.3 * y[0] * y[1] - 0.1 * y[1];
        ydot[2] = 0.1 * y[1];
        Ok(()) 
    });
    solver.set_ss_tolerances(1e-4, 1e-8);
    let mat = DenseMatrix::new(3, 3, &ctx);
    let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
    solver.set_linear_solver(&linsol, &mat);
    let mut tret = 0.0;
    assert_eq!(solver.step(1.0, &mut y, &mut tret), CV_SUCCESS as i32);
}

#[test]
fn test_robertson_ode() {
    let ctx = Context::new();
    let mut y = NVector::new_serial(3, &ctx);
    y.as_mut_slice().copy_from_slice(&[1.0, 0.0, 0.0]);
    let mut solver = CvodeSolver::new(Lmm::Bdf, &ctx);
    solver.init(0.0, &y, |_t, y, ydot| { 
        ydot[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
        ydot[1] = 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1];
        ydot[2] = 3e7 * y[1] * y[1];
        Ok(()) 
    });
    solver.set_ss_tolerances(1e-4, 1e-8);
    let mat = DenseMatrix::new(3, 3, &ctx);
    let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
    solver.set_linear_solver(&linsol, &mat);
    let mut tret = 0.0;
    assert_eq!(solver.step(1.0, &mut y, &mut tret), CV_SUCCESS as i32);
}

#[test]
fn test_pendulum_ode() {
    let ctx = Context::new();
    let mut y = NVector::new_serial(2, &ctx);
    y.as_mut_slice().copy_from_slice(&[3.14/4.0, 0.0]);
    let mut solver = CvodeSolver::new(Lmm::Bdf, &ctx);
    solver.init(0.0, &y, |_t, y, ydot| { 
        ydot[0] = y[1];
        ydot[1] = -9.81 * y[0].sin();
        Ok(()) 
    });
    solver.set_ss_tolerances(1e-4, 1e-8);
    let mat = DenseMatrix::new(2, 2, &ctx);
    let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
    solver.set_linear_solver(&linsol, &mat);
    let mut tret = 0.0;
    assert_eq!(solver.step(1.0, &mut y, &mut tret), CV_SUCCESS as i32);
}

#[test]
fn test_fitzhugh_nagumo() {
    let ctx = Context::new();
    let mut y = NVector::new_serial(2, &ctx);
    y.as_mut_slice().copy_from_slice(&[1.0, 0.0]);
    let mut solver = CvodeSolver::new(Lmm::Bdf, &ctx);
    solver.init(0.0, &y, |_t, y, ydot| { 
        ydot[0] = y[0] - y[0]*y[0]*y[0]/3.0 - y[1] + 0.5;
        ydot[1] = 0.08 * (y[0] + 0.7 - 0.8 * y[1]);
        Ok(()) 
    });
    solver.set_ss_tolerances(1e-4, 1e-8);
    let mat = DenseMatrix::new(2, 2, &ctx);
    let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
    solver.set_linear_solver(&linsol, &mat);
    let mut tret = 0.0;
    assert_eq!(solver.step(1.0, &mut y, &mut tret), CV_SUCCESS as i32);
}

#[test]
fn test_roessler_attractor() {
    let ctx = Context::new();
    let mut y = NVector::new_serial(3, &ctx);
    y.as_mut_slice().copy_from_slice(&[1.0, 1.0, 1.0]);
    let mut solver = CvodeSolver::new(Lmm::Bdf, &ctx);
    solver.init(0.0, &y, |_t, y, ydot| { 
        ydot[0] = -y[1] - y[2];
        ydot[1] = y[0] + 0.2 * y[1];
        ydot[2] = 0.2 + y[2] * (y[0] - 5.7);
        Ok(()) 
    });
    solver.set_ss_tolerances(1e-4, 1e-8);
    let mat = DenseMatrix::new(3, 3, &ctx);
    let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
    solver.set_linear_solver(&linsol, &mat);
    let mut tret = 0.0;
    assert_eq!(solver.step(1.0, &mut y, &mut tret), CV_SUCCESS as i32);
}

#[test]
fn test_seir_model() {
    let ctx = Context::new();
    let mut y = NVector::new_serial(4, &ctx);
    y.as_mut_slice().copy_from_slice(&[0.9, 0.09, 0.01, 0.0]);
    let mut solver = CvodeSolver::new(Lmm::Bdf, &ctx);
    solver.init(0.0, &y, |_t, y, ydot| { 
        ydot[0] = -0.3 * y[0] * y[2];
        ydot[1] = 0.3 * y[0] * y[2] - 0.2 * y[1];
        ydot[2] = 0.2 * y[1] - 0.1 * y[2];
        ydot[3] = 0.1 * y[2];
        Ok(()) 
    });
    solver.set_ss_tolerances(1e-4, 1e-8);
    let mat = DenseMatrix::new(4, 4, &ctx);
    let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
    solver.set_linear_solver(&linsol, &mat);
    let mut tret = 0.0;
    assert_eq!(solver.step(1.0, &mut y, &mut tret), CV_SUCCESS as i32);
}

#[test]
fn test_chemical_kinetics() {
    let ctx = Context::new();
    let mut y = NVector::new_serial(2, &ctx);
    y.as_mut_slice().copy_from_slice(&[1.0, 0.0]);
    let mut solver = CvodeSolver::new(Lmm::Bdf, &ctx);
    solver.init(0.0, &y, |_t, y, ydot| { 
        ydot[0] = -0.1 * y[0] + 0.05 * y[1];
        ydot[1] = 0.1 * y[0] - 0.05 * y[1];
        Ok(()) 
    });
    solver.set_ss_tolerances(1e-4, 1e-8);
    let mat = DenseMatrix::new(2, 2, &ctx);
    let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
    solver.set_linear_solver(&linsol, &mat);
    let mut tret = 0.0;
    assert_eq!(solver.step(1.0, &mut y, &mut tret), CV_SUCCESS as i32);
}

#[test]
fn test_free_fall_with_drag() {
    let ctx = Context::new();
    let mut y = NVector::new_serial(2, &ctx);
    y.as_mut_slice().copy_from_slice(&[1000.0, 0.0]);
    let mut solver = CvodeSolver::new(Lmm::Adams, &ctx);
    solver.init(0.0, &y, |_t, y, ydot| { 
        ydot[0] = y[1];
        ydot[1] = -9.81 + 0.01 * y[1] * y[1] * if y[1] > 0.0 { -1.0 } else { 1.0 };
        Ok(()) 
    });
    solver.set_ss_tolerances(1e-4, 1e-8);
    let mat = DenseMatrix::new(2, 2, &ctx);
    let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
    solver.set_linear_solver(&linsol, &mat);
    let mut tret = 0.0;
    assert_eq!(solver.step(1.0, &mut y, &mut tret), CV_SUCCESS as i32);
}
