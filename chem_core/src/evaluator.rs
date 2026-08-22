use crate::{ReactionSystem};

pub struct KineticEvaluator {
    sys: ReactionSystem,
    s_matrix: Vec<f64>,
    num_species: usize,
    num_reactions: usize,
}

impl KineticEvaluator {
    pub fn new(sys: ReactionSystem) -> Self {
        let s_matrix = sys.stoichiometric_matrix();
        let num_species = sys.species.len();
        let num_reactions = sys.reactions.len();
        Self { sys, s_matrix, num_species, num_reactions }
    }

    /// Evaluates the rate of progress for each reaction r_i
    pub fn evaluate_rates(&self, c: &[f64], rates: &mut [f64]) {
        for (i, reaction) in self.sys.reactions.iter().enumerate() {
            match &reaction.rate_law {
                crate::RateLaw::MassAction(k) => {
                    let mut r = *k;
                    for react in &reaction.reactants {
                        r *= c[react.species_idx].powf(react.coefficient);
                    }
                    rates[i] = r;
                },
                crate::RateLaw::Custom(f) => {
                    rates[i] = f(c);
                }
            }
        }
    }

    /// Evaluates the RHS: dc/dt = S * r
    pub fn evaluate_rhs(&self, c: &[f64], ydot: &mut [f64]) {
        let mut rates = vec![0.0; self.num_reactions];
        self.evaluate_rates(c, &mut rates);

        // ydot = S * rates
        for j in 0..self.num_species {
            if self.sys.species[j].fixed_concentration {
                ydot[j] = 0.0;
                continue;
            }
            let mut sum = 0.0;
            for i in 0..self.num_reactions {
                // S_matrix is row-major: S[j * num_reactions + i]
                sum += self.s_matrix[j * self.num_reactions + i] * rates[i];
            }
            ydot[j] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ReactionSystem, Species, Reaction, Phase};

    #[test]
    fn test_rhs_evaluation() {
        let mut sys = ReactionSystem::new();
        let a = sys.add_species(Species::new("A", 1.0, 0, Phase::Aqueous));
        let b = sys.add_species(Species::new("B", 1.0, 0, Phase::Aqueous));
        
        // A -> B, k=0.5
        sys.add_reaction(Reaction::new(0.5).add_reactant(a, 1.0).add_product(b, 1.0));

        let eval = KineticEvaluator::new(sys);
        
        let c = vec![2.0, 0.0];
        let mut ydot = vec![0.0, 0.0];
        
        eval.evaluate_rhs(&c, &mut ydot);
        
        // r = 0.5 * 2.0 = 1.0
        // A: -1.0, B: +1.0
        assert_eq!(ydot[0], -1.0);
        assert_eq!(ydot[1], 1.0);
    }

    #[test]
    fn test_cvode_live_integration() {
        use sundials::{Context, NVector, DenseMatrix, DenseLinearSolver, CvodeSolver, Lmm};

        let mut sys = ReactionSystem::new();
        let a = sys.add_species(Species::new("A", 1.0, 0, Phase::Aqueous));
        let b = sys.add_species(Species::new("B", 1.0, 0, Phase::Aqueous));
        
        // A -> B, rate = 0.5 * [A]
        sys.add_reaction(Reaction::new(0.5).add_reactant(a, 1.0).add_product(b, 1.0));

        let eval = KineticEvaluator::new(sys);

        let ctx = Context::new();
        let mut y = NVector::new_serial(2, &ctx);
        y.as_mut_slice()[0] = 1.0; // [A]_0 = 1.0
        y.as_mut_slice()[1] = 0.0; // [B]_0 = 0.0

        let mut solver = CvodeSolver::new(Lmm::Bdf, &ctx);
        solver.init(0.0, &y, |t, y_val, ydot| {
            eval.evaluate_rhs(y_val, ydot);
            Ok(())
        });

        solver.set_ss_tolerances(1e-6, 1e-8);

        let mat = DenseMatrix::new(2, 2, &ctx);
        let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
        solver.set_linear_solver(&linsol, &mat);

        let mut tret = 0.0;
        let flag = solver.step(1.0, &mut y, &mut tret);
        
        // Expected [A](1.0) = exp(-0.5) ~ 0.60653
        // Expected [B](1.0) = 1.0 - exp(-0.5) ~ 0.39346
        let expected_a = (-0.5_f64).exp();
        let actual_a = y.as_slice()[0];
        let actual_b = y.as_slice()[1];
        
        assert!((actual_a - expected_a).abs() < 1e-4);
        assert!((actual_b - (1.0 - expected_a)).abs() < 1e-4);
    }

    #[test]
    fn test_heterogeneous_catalysis_fixed_concentration() {
        use sundials::{Context, NVector, DenseMatrix, DenseLinearSolver, CvodeSolver, Lmm};

        let mut sys = ReactionSystem::new();
        // H2O2 -> H2O + 0.5 O2
        let h2o2 = sys.add_species(Species::new("H2O2", 34.01, 0, Phase::Aqueous));
        let h2o = sys.add_species(Species::new("H2O", 18.01, 0, Phase::Aqueous));
        let o2 = sys.add_species(Species::new("O2", 32.0, 0, Phase::Gas));
        // MnO2 catalyst, marked as fixed_concentration!
        let mno2 = sys.add_species(Species::new("MnO2", 86.94, 0, Phase::Solid).set_fixed(true));
        
        // Rate = k * [H2O2]^1 * [MnO2]^1
        // We use [MnO2] as mass or surface area directly in the rate equation.
        sys.add_reaction(Reaction::new(10.0)
            .add_reactant(h2o2, 1.0)
            .add_reactant(mno2, 1.0)  // Catalyst drives the rate!
            .add_product(h2o, 1.0)
            .add_product(o2, 0.5)
        );

        let eval = KineticEvaluator::new(sys);

        let ctx = Context::new();
        let mut y = NVector::new_serial(4, &ctx);
        y.as_mut_slice()[h2o2] = 0.1; // 0.1 mol H2O2
        y.as_mut_slice()[h2o] = 0.0;
        y.as_mut_slice()[o2] = 0.0;
        y.as_mut_slice()[mno2] = 0.5; // 0.5g MnO2

        let mut solver = CvodeSolver::new(Lmm::Bdf, &ctx);
        solver.init(0.0, &y, |t, y_val, ydot| {
            eval.evaluate_rhs(y_val, ydot);
            Ok(())
        });
        solver.set_ss_tolerances(1e-6, 1e-8);
        let mat = DenseMatrix::new(4, 4, &ctx);
        let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
        solver.set_linear_solver(&linsol, &mat);

        let mut tret = 0.0;
        let _flag = solver.step(0.1, &mut y, &mut tret);
        
        let actual_h2o2 = y.as_slice()[h2o2];
        let actual_mno2 = y.as_slice()[mno2];
        
        // H2O2 should be depleted
        assert!(actual_h2o2 < 0.1);
        
        // MnO2 MUST be exactly 0.5 still!
        assert_eq!(actual_mno2, 0.5);
    }

    #[test]
    fn test_custom_rate_expression() {
        use sundials::{Context, NVector, DenseMatrix, DenseLinearSolver, CvodeSolver, Lmm};

        let mut sys = ReactionSystem::new();
        let s1 = sys.add_species(Species::new("S1", 10.0, 0, Phase::Aqueous));
        let s2 = sys.add_species(Species::new("S2", 10.0, 0, Phase::Aqueous));
        
        // Arbitrary rate law: rate = 5.0 * sin([S1])
        sys.add_reaction(Reaction::with_custom_rate(move |c| {
            5.0 * c[s1].sin()
        })
        .add_reactant(s1, 1.0)
        .add_product(s2, 1.0));

        let eval = KineticEvaluator::new(sys);

        let ctx = Context::new();
        let mut y = NVector::new_serial(2, &ctx);
        y.as_mut_slice()[s1] = std::f64::consts::PI / 2.0; // sin(PI/2) = 1.0
        y.as_mut_slice()[s2] = 0.0;

        let mut solver = CvodeSolver::new(Lmm::Bdf, &ctx);
        solver.init(0.0, &y, |t, y_val, ydot| {
            eval.evaluate_rhs(y_val, ydot);
            Ok(())
        });
        solver.set_ss_tolerances(1e-6, 1e-8);
        let mat = DenseMatrix::new(2, 2, &ctx);
        let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
        solver.set_linear_solver(&linsol, &mat);

        let mut tret = 0.0;
        let _flag = solver.step(0.01, &mut y, &mut tret);
        
        let actual_s1 = y.as_slice()[s1];
        let actual_s2 = y.as_slice()[s2];
        
        assert!(actual_s1 < std::f64::consts::PI / 2.0 - 0.04);
        assert!(actual_s2 > 0.04);
    }
}
