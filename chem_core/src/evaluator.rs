use crate::ReactionSystem;

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
        Self {
            sys,
            s_matrix,
            num_species,
            num_reactions,
        }
    }

    pub fn temperature(&self) -> f64 {
        self.sys.temperature
    }

    pub fn set_temperature(&mut self, t: f64) {
        self.sys.temperature = t;
    }

    /// Evaluates the rate of progress for each reaction r_i
    pub fn evaluate_rates(&self, c: &[f64], rates: &mut [f64]) {
        let t = self.sys.temperature;
        for (i, reaction) in self.sys.reactions.iter().enumerate() {
            match &reaction.rate_law {
                crate::RateLaw::MassAction(k) => {
                    let mut r = *k;
                    for react in &reaction.reactants {
                        r *= c[react.species_idx].max(0.0).powf(react.coefficient);
                    }
                    rates[i] = r;
                }
                crate::RateLaw::ArrheniusLaw(arr) => {
                    let mut r = arr.k(t);
                    for react in &reaction.reactants {
                        r *= c[react.species_idx].max(0.0).powf(react.coefficient);
                    }
                    rates[i] = r;
                }
                crate::RateLaw::PressureDependent { arrhenius, pressure } => {
                    let k_base = arrhenius.k(t);
                    let k_pressure =
                        pressure.rate_constant(t, c, self.num_species);
                    let mut r = k_base * k_pressure;
                    for react in &reaction.reactants {
                        r *= c[react.species_idx].max(0.0).powf(react.coefficient);
                    }
                    rates[i] = r;
                }
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

        for j in 0..self.num_species {
            if self.sys.species[j].fixed_concentration {
                ydot[j] = 0.0;
                continue;
            }
            let mut sum = 0.0;
            for i in 0..self.num_reactions {
                sum += self.s_matrix[j * self.num_reactions + i] * rates[i];
            }
            ydot[j] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Arrhenius, Phase, Reaction, ReactionSystem, Species};

    #[test]
    fn test_rhs_evaluation() {
        let mut sys = ReactionSystem::new();
        let a = sys.add_species(Species::new("A", 1.0, 0, Phase::Aqueous));
        let b = sys.add_species(Species::new("B", 1.0, 0, Phase::Aqueous));

        sys.add_reaction(
            Reaction::new(0.5)
                .add_reactant(a, 1.0)
                .add_product(b, 1.0),
        );

        let eval = KineticEvaluator::new(sys);
        let c = vec![2.0, 0.0];
        let mut ydot = vec![0.0, 0.0];
        eval.evaluate_rhs(&c, &mut ydot);
        assert_eq!(ydot[0], -1.0);
        assert_eq!(ydot[1], 1.0);
    }

    #[test]
    fn test_arrhenius_rate_evaluation() {
        let mut sys = ReactionSystem::new();
        sys.set_temperature(1000.0);
        let a = sys.add_species(Species::new("A", 1.0, 0, Phase::Gas));
        let b = sys.add_species(Species::new("B", 1.0, 0, Phase::Gas));

        sys.add_reaction(
            Reaction::with_arrhenius(1e13, 0.0, 40000.0)
                .add_reactant(a, 1.0)
                .add_product(b, 1.0),
        );

        let eval = KineticEvaluator::new(sys);
        let c = vec![1.0, 0.0];
        let mut ydot = vec![0.0, 0.0];
        eval.evaluate_rhs(&c, &mut ydot);

        let expected_k = Arrhenius::new(1e13, 0.0, 40000.0).k(1000.0);
        assert!(
            (ydot[0] + expected_k).abs() < expected_k * 1e-10,
            "ydot[0] = {}, expected -k = {}",
            ydot[0],
            -expected_k
        );
    }

    #[test]
    fn test_cvode_live_integration() {
        use sundials::{Context, CvodeBuilder, DenseLinearSolver, DenseMatrix, Lmm, NVector};

        let mut sys = ReactionSystem::new();
        let a = sys.add_species(Species::new("A", 1.0, 0, Phase::Aqueous));
        let _b = sys.add_species(Species::new("B", 1.0, 0, Phase::Aqueous));
        sys.add_reaction(
            Reaction::new(0.5)
                .add_reactant(a, 1.0)
                .add_product(1, 1.0),
        );

        let eval = KineticEvaluator::new(sys);
        let ctx = Context::new();
        let mut y = NVector::new_serial(2, &ctx);
        y.as_mut_slice()[0] = 1.0;
        y.as_mut_slice()[1] = 0.0;

        let mut solver =
            CvodeBuilder::new(Lmm::Bdf, &ctx).init(0.0, &y, move |_t, y_val, ydot| {
                eval.evaluate_rhs(y_val, ydot);
                Ok(())
            });
        solver.set_ss_tolerances(1e-6, 1e-8);
        let mat = DenseMatrix::new(2, 2, &ctx);
        let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
        solver.set_linear_solver(&linsol, &mat);

        let mut tret = 0.0;
        solver.step(1.0, &mut y, &mut tret);
        let expected_a = (-0.5_f64).exp();
        assert!((y.as_slice()[0] - expected_a).abs() < 1e-4);
        assert!((y.as_slice()[1] - (1.0 - expected_a)).abs() < 1e-4);
    }

    #[test]
    fn test_heterogeneous_catalysis_fixed_concentration() {
        use sundials::{Context, CvodeBuilder, DenseLinearSolver, DenseMatrix, Lmm, NVector};

        let mut sys = ReactionSystem::new();
        let h2o2 = sys.add_species(Species::new("H2O2", 34.01, 0, Phase::Aqueous));
        let _h2o = sys.add_species(Species::new("H2O", 18.01, 0, Phase::Aqueous));
        let _o2 = sys.add_species(Species::new("O2", 32.0, 0, Phase::Gas));
        let mno2 = sys.add_species(Species::new("MnO2", 86.94, 0, Phase::Solid).set_fixed(true));
        sys.add_reaction(
            Reaction::new(10.0)
                .add_reactant(h2o2, 1.0)
                .add_reactant(mno2, 1.0)
                .add_product(1, 1.0)
                .add_product(2, 0.5),
        );

        let eval = KineticEvaluator::new(sys);
        let ctx = Context::new();
        let mut y = NVector::new_serial(4, &ctx);
        y.as_mut_slice()[h2o2] = 0.1;
        y.as_mut_slice()[1] = 0.0;
        y.as_mut_slice()[2] = 0.0;
        y.as_mut_slice()[mno2] = 0.5;

        let mut solver =
            CvodeBuilder::new(Lmm::Bdf, &ctx).init(0.0, &y, move |_t, y_val, ydot| {
                eval.evaluate_rhs(y_val, ydot);
                Ok(())
            });
        solver.set_ss_tolerances(1e-6, 1e-8);
        let mat = DenseMatrix::new(4, 4, &ctx);
        let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
        solver.set_linear_solver(&linsol, &mat);

        let mut tret = 0.0;
        solver.step(0.1, &mut y, &mut tret);
        assert!(y.as_slice()[h2o2] < 0.1);
        assert_eq!(y.as_slice()[mno2], 0.5);
    }

    #[test]
    fn test_custom_rate_expression() {
        use sundials::{Context, CvodeBuilder, DenseLinearSolver, DenseMatrix, Lmm, NVector};

        let mut sys = ReactionSystem::new();
        let s1 = sys.add_species(Species::new("S1", 10.0, 0, Phase::Aqueous));
        let _s2 = sys.add_species(Species::new("S2", 10.0, 0, Phase::Aqueous));
        sys.add_reaction(
            Reaction::with_custom_rate(move |c| 5.0 * c[s1].sin())
                .add_reactant(s1, 1.0)
                .add_product(1, 1.0),
        );

        let eval = KineticEvaluator::new(sys);
        let ctx = Context::new();
        let mut y = NVector::new_serial(2, &ctx);
        y.as_mut_slice()[s1] = std::f64::consts::PI / 2.0;
        y.as_mut_slice()[1] = 0.0;

        let mut solver =
            CvodeBuilder::new(Lmm::Bdf, &ctx).init(0.0, &y, move |_t, y_val, ydot| {
                eval.evaluate_rhs(y_val, ydot);
                Ok(())
            });
        solver.set_ss_tolerances(1e-6, 1e-8);
        let mat = DenseMatrix::new(2, 2, &ctx);
        let linsol = DenseLinearSolver::new(&y, &mat, &ctx);
        solver.set_linear_solver(&linsol, &mat);

        let mut tret = 0.0;
        solver.step(0.01, &mut y, &mut tret);
        assert!(y.as_slice()[s1] < std::f64::consts::PI / 2.0 - 0.04);
        assert!(y.as_slice()[1] > 0.04);
    }
}
