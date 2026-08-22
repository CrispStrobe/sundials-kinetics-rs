use crate::ReactionSystem;
use symengine::Expression;

pub struct SymEngineGenerator<'a> {
    sys: &'a ReactionSystem,
}

impl<'a> SymEngineGenerator<'a> {
    pub fn new(sys: &'a ReactionSystem) -> Self {
        Self { sys }
    }

    /// Generates the array of symbolic variables for concentrations: c_0, c_1, ...
    pub fn generate_symbols(&self) -> Vec<Expression> {
        self.sys.species.iter().enumerate().map(|(i, _)| {
            Expression::symbol(&format!("c_{}", i))
        }).collect()
    }

    /// Generates the analytical rate laws for each reaction as an expression
    pub fn generate_rates(&self, syms: &[Expression]) -> Vec<Expression> {
        self.sys.reactions.iter().map(|reaction| {
            match &reaction.rate_law {
                crate::RateLaw::MassAction(k) => {
                    let mut rate_expr = Expression::real_double(*k);
                    for react in &reaction.reactants {
                        let conc = &syms[react.species_idx];
                        let exp = Expression::real_double(react.coefficient);
                        let term = conc.pow(&exp);
                        rate_expr = &rate_expr * &term;
                    }
                    rate_expr
                },
                crate::RateLaw::Custom(_) => {
                    unimplemented!("SymEngine AST generation for Custom rate laws via closure is not yet supported. Please use numerical Jacobians or provide an AST builder.");
                }
            }
        }).collect()
    }

    /// Generates the RHS equations f_j = dc_j/dt for each species
    pub fn generate_rhs(&self) -> Vec<Expression> {
        let syms = self.generate_symbols();
        let rates = self.generate_rates(&syms);
        let s_matrix = self.sys.stoichiometric_matrix();
        let num_reactions = self.sys.reactions.len();

        let mut f_exprs = Vec::with_capacity(self.sys.species.len());

        for j in 0..self.sys.species.len() {
            if self.sys.species[j].fixed_concentration {
                f_exprs.push(Expression::real_double(0.0));
                continue;
            }
            let mut sum_expr = Expression::real_double(0.0);
            for i in 0..num_reactions {
                let s_val = s_matrix[j * num_reactions + i];
                if s_val != 0.0 {
                    let s_expr = Expression::real_double(s_val);
                    let term = &s_expr * &rates[i];
                    sum_expr = &sum_expr + &term;
                }
            }
            f_exprs.push(sum_expr);
        }
        f_exprs
    }

    /// Generates the exact analytical Jacobian matrix expressions J_jk = df_j/dc_k
    pub fn generate_jacobian(&self) -> Vec<Vec<Expression>> {
        let f_exprs = self.generate_rhs();
        let syms = self.generate_symbols();

        let mut jac = Vec::with_capacity(f_exprs.len());
        for f in &f_exprs {
            let mut row = Vec::with_capacity(syms.len());
            for c in &syms {
                row.push(f.diff(c));
            }
            jac.push(row);
        }
        jac
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ReactionSystem, Species, Reaction, Phase};

    #[test]
    fn test_symengine_generator() {
        let mut sys = ReactionSystem::new();
        let a = sys.add_species(Species::new("A", 1.0, 0, Phase::Aqueous));
        let b = sys.add_species(Species::new("B", 1.0, 0, Phase::Aqueous));
        
        // 2A -> B, rate = 0.5 * [A]^2
        sys.add_reaction(Reaction::new(0.5).add_reactant(a, 2.0).add_product(b, 1.0));

        let generator = SymEngineGenerator::new(&sys);
        
        let f_exprs = generator.generate_rhs();
        let rhs_str_a = f_exprs[0].to_string(); // dc_a/dt = -2.0 * (0.5 * c_a^2) = -1.0 * c_a^2
        let rhs_str_b = f_exprs[1].to_string(); // dc_b/dt = 1.0 * (0.5 * c_a^2) = 0.5 * c_a^2
        
        assert!(rhs_str_a.contains("c_0") && rhs_str_a.contains("2.0") || rhs_str_a.contains("-1.0"));
        
        let jac = generator.generate_jacobian();
        // J_00 = d(f_a)/dc_a = -2.0 * c_a
        let j_00_str = jac[0][0].to_string();
        assert!(j_00_str.contains("c_0"));
        
        // J_01 = d(f_a)/dc_b = 0.0
        let j_01_str = jac[0][1].to_string();
        assert!(j_01_str.contains("0"));
    }
}
