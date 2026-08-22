use sundials::{Context, DenseLinearSolver, DenseMatrix, IdaBuilder, NVector};
use sundials_sys::IDA_SUCCESS;

#[test]
fn test_simple_dae() {
    let ctx = Context::new();
    let mut y = NVector::new_serial(2, &ctx);
    let mut yp = NVector::new_serial(2, &ctx);
    y.as_mut_slice().copy_from_slice(&[1.0, 1.0]);
    yp.as_mut_slice().copy_from_slice(&[-1.0, 0.0]);
    let mut solver = IdaBuilder::new(&ctx).init(0.0, &y, &yp, |_t, y, yp, res| {
        res[0] = yp[0] + y[0];
        res[1] = y[1] - y[0] * y[0];
        Ok(())
    });
    solver.set_ss_tolerances(1e-4, 1e-8);
    let mat = DenseMatrix::new(2, 2, &ctx);
    let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
    solver.set_linear_solver(&linsol, &mat);
    let mut tret = 0.0;
    assert_eq!(
        solver.solve(1.0, &mut y, &mut yp, &mut tret),
        IDA_SUCCESS as i32
    );
}

#[test]
fn test_robertson_dae() {
    let ctx = Context::new();
    let mut y = NVector::new_serial(3, &ctx);
    let mut yp = NVector::new_serial(3, &ctx);
    y.as_mut_slice().copy_from_slice(&[1.0, 0.0, 0.0]);
    yp.as_mut_slice().copy_from_slice(&[-0.04, 0.04, 0.0]);
    let mut solver = IdaBuilder::new(&ctx).init(0.0, &y, &yp, |_t, y, yp, res| {
        res[0] = yp[0] + 0.04 * y[0] - 1e4 * y[1] * y[2];
        res[1] = yp[1] - 0.04 * y[0] + 1e4 * y[1] * y[2] + 3e7 * y[1] * y[1];
        res[2] = y[0] + y[1] + y[2] - 1.0;
        Ok(())
    });
    solver.set_ss_tolerances(1e-4, 1e-8);
    let mat = DenseMatrix::new(3, 3, &ctx);
    let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
    solver.set_linear_solver(&linsol, &mat);
    let mut tret = 0.0;
    assert_eq!(
        solver.solve(1.0, &mut y, &mut yp, &mut tret),
        IDA_SUCCESS as i32
    );
}

#[test]
#[test]
fn test_index1_dae() {
    let ctx = Context::new();
    let mut y = NVector::new_serial(2, &ctx);
    let mut yp = NVector::new_serial(2, &ctx);
    y.as_mut_slice().copy_from_slice(&[1.0, 2.0]);
    yp.as_mut_slice().copy_from_slice(&[1.0, 0.0]);
    let mut solver = IdaBuilder::new(&ctx).init(0.0, &y, &yp, |_t, y, yp, res| {
        res[0] = yp[0] - y[1];
        res[1] = y[0] + y[1] - 3.0;
        Ok(())
    });
    solver.set_ss_tolerances(1e-4, 1e-8);
    let mat = DenseMatrix::new(2, 2, &ctx);
    let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
    solver.set_linear_solver(&linsol, &mat);
    let mut tret = 0.0;
    assert_eq!(
        solver.solve(1.0, &mut y, &mut yp, &mut tret),
        IDA_SUCCESS as i32
    );
}

#[test]
fn test_chemical_equilibrium_dae() {
    let ctx = Context::new();
    let mut y = NVector::new_serial(3, &ctx);
    let mut yp = NVector::new_serial(3, &ctx);
    y.as_mut_slice().copy_from_slice(&[1.0, 1.0, 1.0]);
    yp.as_mut_slice().copy_from_slice(&[-0.1, -0.1, 0.0]);
    let mut solver = IdaBuilder::new(&ctx).init(0.0, &y, &yp, |_t, y, yp, res| {
        res[0] = yp[0] + 0.1 * y[0];
        res[1] = yp[1] + 0.1 * y[1];
        res[2] = y[0] * y[1] - y[2];
        Ok(())
    });
    solver.set_ss_tolerances(1e-4, 1e-8);
    let mat = DenseMatrix::new(3, 3, &ctx);
    let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
    solver.set_linear_solver(&linsol, &mat);
    let mut tret = 0.0;
    assert_eq!(
        solver.solve(1.0, &mut y, &mut yp, &mut tret),
        IDA_SUCCESS as i32
    );
}

#[test]
#[test]
#[test]
fn test_reactor_dae() {
    let ctx = Context::new();
    let mut y = NVector::new_serial(2, &ctx);
    let mut yp = NVector::new_serial(2, &ctx);
    y.as_mut_slice().copy_from_slice(&[1.0, 0.5]);
    yp.as_mut_slice().copy_from_slice(&[-0.1, 0.0]);
    let mut solver = IdaBuilder::new(&ctx).init(0.0, &y, &yp, |_t, y, yp, res| {
        res[0] = yp[0] + 0.1 * y[0];
        res[1] = y[1] - y[0] * 0.5;
        Ok(())
    });
    solver.set_ss_tolerances(1e-4, 1e-8);
    let mat = DenseMatrix::new(2, 2, &ctx);
    let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
    solver.set_linear_solver(&linsol, &mat);
    let mut tret = 0.0;
    assert_eq!(
        solver.solve(1.0, &mut y, &mut yp, &mut tret),
        IDA_SUCCESS as i32
    );
}

#[test]
fn test_transistor_amplifier_dae() {
    let ctx = Context::new();
    let mut y = NVector::new_serial(3, &ctx);
    let mut yp = NVector::new_serial(3, &ctx);
    y.as_mut_slice().copy_from_slice(&[0.0, 1.0, 1.0]);
    yp.as_mut_slice().copy_from_slice(&[1.0, -1.0, 0.0]);
    let mut solver = IdaBuilder::new(&ctx).init(0.0, &y, &yp, |_t, y, yp, res| {
        res[0] = yp[0] + y[0] - 1.0;
        res[1] = yp[1] + y[1] + y[0];
        res[2] = y[2] - y[1] * y[1];
        Ok(())
    });
    solver.set_ss_tolerances(1e-4, 1e-8);
    let mat = DenseMatrix::new(3, 3, &ctx);
    let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
    solver.set_linear_solver(&linsol, &mat);
    let mut tret = 0.0;
    assert_eq!(
        solver.solve(1.0, &mut y, &mut yp, &mut tret),
        IDA_SUCCESS as i32
    );
}

#[test]
fn test_heat_transfer_dae() {
    let ctx = Context::new();
    let mut y = NVector::new_serial(2, &ctx);
    let mut yp = NVector::new_serial(2, &ctx);
    y.as_mut_slice().copy_from_slice(&[100.0, 20.0]);
    yp.as_mut_slice().copy_from_slice(&[-1.0, 0.0]);
    let mut solver = IdaBuilder::new(&ctx).init(0.0, &y, &yp, |_t, y, yp, res| {
        res[0] = yp[0] + 0.1 * (y[0] - y[1]);
        res[1] = y[1] - 20.0;
        Ok(())
    });
    solver.set_ss_tolerances(1e-4, 1e-8);
    let mat = DenseMatrix::new(2, 2, &ctx);
    let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
    solver.set_linear_solver(&linsol, &mat);
    let mut tret = 0.0;
    assert_eq!(
        solver.solve(1.0, &mut y, &mut yp, &mut tret),
        IDA_SUCCESS as i32
    );
}
