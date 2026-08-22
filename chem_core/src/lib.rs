pub mod evaluator;
pub mod symengine_generator;
pub mod parser;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Phase {
    Aqueous,
    Gas,
    Solid,
}

#[derive(Debug, Clone)]
pub struct Species {
    pub name: String,
    pub mass: f64,    // kg/mol
    pub charge: i32,  // elementary charge
    pub phase: Phase,
    pub fixed_concentration: bool, // If true, ydot is always 0
}

impl Species {
    pub fn new(name: &str, mass: f64, charge: i32, phase: Phase) -> Self {
        Self {
            name: name.to_string(),
            mass,
            charge,
            phase,
            fixed_concentration: false,
        }
    }

    pub fn set_fixed(mut self, fixed: bool) -> Self {
        self.fixed_concentration = fixed;
        self
    }
}

/// Defines the stoichiometry of a species in a reaction
#[derive(Debug, Clone)]
pub struct Stoichiometry {
    pub species_idx: usize,
    pub coefficient: f64,
}

pub enum RateLaw {
    /// Standard Mass-Action kinetics: rate = k * prod(c_i^coeff)
    MassAction(f64),
    /// Custom rate law evaluated via a user-provided closure.
    /// Takes the current concentration slice and returns the rate.
    Custom(std::sync::Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>),
}

impl std::fmt::Debug for RateLaw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RateLaw::MassAction(k) => write!(f, "MassAction({})", k),
            RateLaw::Custom(_) => write!(f, "Custom(<closure>)"),
        }
    }
}

impl Clone for RateLaw {
    fn clone(&self) -> Self {
        match self {
            RateLaw::MassAction(k) => RateLaw::MassAction(*k),
            RateLaw::Custom(c) => RateLaw::Custom(std::sync::Arc::clone(c)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Reaction {
    pub reactants: Vec<Stoichiometry>,
    pub products: Vec<Stoichiometry>,
    pub rate_law: RateLaw,
}

impl Reaction {
    pub fn new(rate_constant: f64) -> Self {
        Self {
            reactants: Vec::new(),
            products: Vec::new(),
            rate_law: RateLaw::MassAction(rate_constant),
        }
    }

    pub fn with_custom_rate<F>(rate_fn: F) -> Self 
    where F: Fn(&[f64]) -> f64 + Send + Sync + 'static 
    {
        Self {
            reactants: Vec::new(),
            products: Vec::new(),
            rate_law: RateLaw::Custom(std::sync::Arc::new(rate_fn)),
        }
    }

    pub fn add_reactant(mut self, species_idx: usize, coefficient: f64) -> Self {
        self.reactants.push(Stoichiometry { species_idx, coefficient });
        self
    }

    pub fn add_product(mut self, species_idx: usize, coefficient: f64) -> Self {
        self.products.push(Stoichiometry { species_idx, coefficient });
        self
    }
}

#[derive(Debug, Clone)]
pub struct ReactionSystem {
    pub species: Vec<Species>,
    pub reactions: Vec<Reaction>,
}

impl ReactionSystem {
    pub fn new() -> Self {
        Self {
            species: Vec::new(),
            reactions: Vec::new(),
        }
    }

    pub fn add_species(&mut self, species: Species) -> usize {
        let idx = self.species.len();
        self.species.push(species);
        idx
    }

    pub fn add_reaction(&mut self, reaction: Reaction) {
        self.reactions.push(reaction);
    }

    /// Computes the net stoichiometry coefficient for a given species in a given reaction.
    /// S_ij = (coeff as product) - (coeff as reactant)
    pub fn net_stoichiometry(&self, reaction_idx: usize, species_idx: usize) -> f64 {
        let reaction = &self.reactions[reaction_idx];
        
        let prod_sum: f64 = reaction.products.iter()
            .filter(|p| p.species_idx == species_idx)
            .map(|p| p.coefficient)
            .sum();
            
        let react_sum: f64 = reaction.reactants.iter()
            .filter(|r| r.species_idx == species_idx)
            .map(|r| r.coefficient)
            .sum();
            
        prod_sum - react_sum
    }

    /// Returns the dense stoichiometric matrix S as a 1D vector (row-major).
    /// Rows = species, Columns = reactions.
    pub fn stoichiometric_matrix(&self) -> Vec<f64> {
        let mut s_matrix = Vec::with_capacity(self.species.len() * self.reactions.len());
        
        for i in 0..self.species.len() {
            for j in 0..self.reactions.len() {
                s_matrix.push(self.net_stoichiometry(j, i));
            }
        }
        
        s_matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stoichiometric_matrix() {
        let mut sys = ReactionSystem::new();
        
        // 0: A, 1: B, 2: C
        let idx_a = sys.add_species(Species::new("A", 1.0, 0, Phase::Aqueous));
        let idx_b = sys.add_species(Species::new("B", 1.0, 0, Phase::Aqueous));
        let idx_c = sys.add_species(Species::new("C", 2.0, 0, Phase::Aqueous));
        
        // R0: 2A + B -> C
        sys.add_reaction(Reaction::new(1.5)
            .add_reactant(idx_a, 2.0)
            .add_reactant(idx_b, 1.0)
            .add_product(idx_c, 1.0)
        );
        
        let s_matrix = sys.stoichiometric_matrix();
        
        // Rows: A, B, C; Cols: R0
        // A: -2.0
        // B: -1.0
        // C:  1.0
        assert_eq!(s_matrix, vec![-2.0, -1.0, 1.0]);
    }
}
