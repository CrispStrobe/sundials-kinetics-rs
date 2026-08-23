//! Bridge between SymEngine-generated analytical Jacobians and SUNDIALS CVODE.
//!
//! 1. SymEngine symbolically differentiates the ODE right-hand side once
//! 2. The resulting expressions are stored as pre-differentiated formulas
//! 3. At each CVODE Newton step, `evaluate()` substitutes current
//!    concentrations via `SubstitutionMap` and `eval_double()`
//! 4. CVODE gets exact derivatives — no finite-difference approximation

use crate::ReactionSystem;
use crate::symengine_generator::SymEngineGenerator;
use symengine::{Expression, SubstitutionMap};

/// A compiled Jacobian: symbolic expressions differentiated once at setup,
/// evaluated numerically at each CVODE step.
pub struct CompiledJacobian {
    pub n: usize,
    /// J[j * n + k] = ∂f_j/∂c_k as a SymEngine expression.
    jac_exprs: Vec<Expression>,
    /// The symbolic concentration variables c_0, c_1, ...
    symbols: Vec<Expression>,
}

impl CompiledJacobian {
    /// Compile the analytical Jacobian. This is the expensive symbolic
    /// differentiation — done once at problem setup, never in the hot loop.
    pub fn compile(sys: &ReactionSystem) -> Self {
        let generator = SymEngineGenerator::new(sys);
        let n = sys.species.len();
        let symbols = generator.generate_symbols();
        let jac_matrix = generator.generate_jacobian();

        let mut jac_exprs = Vec::with_capacity(n * n);
        for row in jac_matrix {
            for entry in row {
                jac_exprs.push(entry);
            }
        }

        Self { n, jac_exprs, symbols }
    }

    /// Evaluate the Jacobian at current concentrations.
    ///
    /// Writes into `jac_out` in column-major order (SUNDIALS convention):
    /// jac_out[j + k * n] = J[j][k] = ∂f_j/∂c_k
    ///
    /// Returns true if evaluation succeeded for all entries.
    pub fn evaluate(&self, concentrations: &[f64], jac_out: &mut [f64]) -> bool {
        assert_eq!(concentrations.len(), self.n);
        assert!(jac_out.len() >= self.n * self.n);

        let map = SubstitutionMap::from_values(&self.symbols, concentrations);

        let mut ok = true;
        for j in 0..self.n {
            for k in 0..self.n {
                let expr = &self.jac_exprs[j * self.n + k];
                let substituted = expr.subs(&map);
                match substituted.eval_double() {
                    Some(val) => {
                        jac_out[j + k * self.n] = val;
                    }
                    None => {
                        jac_out[j + k * self.n] = 0.0;
                        ok = false;
                    }
                }
            }
        }
        ok
    }

    /// Sparsity pattern: which J[j][k] entries are structurally nonzero.
    pub fn sparsity_pattern(&self) -> Vec<(usize, usize)> {
        let mut nonzero = Vec::new();
        for j in 0..self.n {
            for k in 0..self.n {
                let s = self.jac_exprs[j * self.n + k].to_string();
                if s != "0" && s != "0.0" && s != "0.00000000000000000" {
                    nonzero.push((j, k));
                }
            }
        }
        nonzero
    }

    pub fn nnz(&self) -> usize {
        self.sparsity_pattern().len()
    }

    pub fn density(&self) -> f64 {
        self.nnz() as f64 / (self.n * self.n) as f64
    }

    pub fn print_symbolic(&self) {
        for j in 0..self.n {
            for k in 0..self.n {
                eprintln!("  J[{j},{k}] = {}", self.jac_exprs[j * self.n + k]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Species, Reaction, Phase};

    fn simple_decay() -> ReactionSystem {
        let mut sys = ReactionSystem::new();
        let a = sys.add_species(Species::new("A", 1.0, 0, Phase::Aqueous));
        let b = sys.add_species(Species::new("B", 1.0, 0, Phase::Aqueous));
        sys.add_reaction(
            Reaction::new(0.1).add_reactant(a, 1.0).add_product(b, 1.0),
        );
        sys
    }

    fn consecutive() -> ReactionSystem {
        let mut sys = ReactionSystem::new();
        let a = sys.add_species(Species::new("A", 1.0, 0, Phase::Aqueous));
        let b = sys.add_species(Species::new("B", 1.0, 0, Phase::Aqueous));
        let c = sys.add_species(Species::new("C", 1.0, 0, Phase::Aqueous));
        sys.add_reaction(
            Reaction::new(0.1).add_reactant(a, 1.0).add_product(b, 1.0),
        );
        sys.add_reaction(
            Reaction::new(0.2).add_reactant(b, 1.0).add_product(c, 1.0),
        );
        sys
    }

    #[test]
    fn compile_and_evaluate_simple_jacobian() {
        let sys = simple_decay();
        let jac = CompiledJacobian::compile(&sys);
        assert_eq!(jac.n, 2);

        let mut out = vec![0.0; 4];
        let ok = jac.evaluate(&[1.0, 0.0], &mut out);
        assert!(ok);

        // Column-major: J[j][k] at j + k*n
        let j00 = out[0]; // ∂f_A/∂A = -0.1
        let j10 = out[1]; // ∂f_B/∂A = +0.1
        let j01 = out[2]; // ∂f_A/∂B = 0
        let j11 = out[3]; // ∂f_B/∂B = 0

        assert!((j00 - (-0.1)).abs() < 1e-10, "J[0,0] = {j00}");
        assert!((j10 - 0.1).abs() < 1e-10, "J[1,0] = {j10}");
        assert!(j01.abs() < 1e-10, "J[0,1] = {j01}");
        assert!(j11.abs() < 1e-10, "J[1,1] = {j11}");
    }

    #[test]
    fn sparsity_detects_zeros_in_consecutive() {
        let sys = consecutive();
        let jac = CompiledJacobian::compile(&sys);
        let pattern = jac.sparsity_pattern();
        assert!(!pattern.contains(&(0, 2)), "J[0,2] should be zero");
        assert!(pattern.contains(&(0, 0)), "J[0,0] should be nonzero");
    }
}
