pub mod cantera;
pub mod evaluator;
pub mod ffi;
pub mod parser;
pub mod symengine_generator;

/// Gas constant in J·mol⁻¹·K⁻¹ (same as kerotakis-core).
pub const R_GAS: f64 = 8.314_462_618;

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

/// Arrhenius rate parameters: k(T) = A * T^b * exp(-Ea / RT)
#[derive(Debug, Clone, Copy)]
pub struct Arrhenius {
    /// Pre-exponential factor (units depend on reaction order)
    pub a: f64,
    /// Temperature exponent (dimensionless)
    pub b: f64,
    /// Activation energy in J/mol (NOT cal/mol — Cantera uses J/mol)
    pub ea: f64,
}

impl Arrhenius {
    pub fn new(a: f64, b: f64, ea: f64) -> Self {
        Self { a, b, ea }
    }

    /// Evaluate the rate constant at temperature T (Kelvin).
    pub fn k(&self, t: f64) -> f64 {
        self.a * t.powf(self.b) * (-self.ea / (R_GAS * t)).exp()
    }
}

/// Troe falloff broadening parameters.
#[derive(Debug, Clone, Copy)]
pub struct TroeParams {
    pub a: f64,
    pub t3: f64,
    pub t1: f64,
    pub t2: Option<f64>,
}

impl TroeParams {
    /// Compute the Troe broadening factor F at temperature T and reduced pressure Pr.
    pub fn broadening_factor(&self, t: f64, pr: f64) -> f64 {
        let f_cent = (1.0 - self.a) * (-t / self.t3).exp()
            + self.a * (-t / self.t1).exp()
            + self.t2.map_or(0.0, |t2| (-t2 / t).exp());
        let log_f_cent = f_cent.max(1e-300).log10();
        let c = -0.4 - 0.67 * log_f_cent;
        let n = 0.75 - 1.27 * log_f_cent;
        let d = 0.14;
        let log_pr = pr.max(1e-300).log10();
        let f1 = (log_pr + c) / (n - d * (log_pr + c));
        10.0_f64.powf(log_f_cent / (1.0 + f1 * f1))
    }
}

/// Third-body collision efficiency for a specific species.
#[derive(Debug, Clone)]
pub struct ThirdBodyEfficiency {
    pub species_idx: usize,
    pub efficiency: f64,
}

/// Pressure dependence model for a reaction.
#[derive(Debug, Clone)]
pub enum PressureDependence {
    /// Simple third-body: rate = k * [M] where [M] = Σ ε_i * c_i
    ThirdBody {
        efficiencies: Vec<ThirdBodyEfficiency>,
        default_efficiency: f64,
    },
    /// Lindemann falloff: k = k_inf * (Pr / (1 + Pr))
    Lindemann {
        high_pressure: Arrhenius,
        low_pressure: Arrhenius,
        efficiencies: Vec<ThirdBodyEfficiency>,
        default_efficiency: f64,
    },
    /// Troe falloff: k = k_inf * (Pr / (1 + Pr)) * F(T, Pr)
    Troe {
        high_pressure: Arrhenius,
        low_pressure: Arrhenius,
        troe: TroeParams,
        efficiencies: Vec<ThirdBodyEfficiency>,
        default_efficiency: f64,
    },
}

impl PressureDependence {
    /// Compute the effective rate constant given temperature and concentrations.
    pub fn rate_constant(&self, t: f64, concentrations: &[f64], n_species: usize) -> f64 {
        match self {
            PressureDependence::ThirdBody {
                efficiencies,
                default_efficiency,
            } => {
                let m = third_body_concentration(concentrations, n_species, efficiencies, *default_efficiency);
                m
            }
            PressureDependence::Lindemann {
                high_pressure,
                low_pressure,
                efficiencies,
                default_efficiency,
            } => {
                let k_inf = high_pressure.k(t);
                let k_0 = low_pressure.k(t);
                let m = third_body_concentration(concentrations, n_species, efficiencies, *default_efficiency);
                let pr = k_0 * m / k_inf.max(1e-300);
                k_inf * pr / (1.0 + pr)
            }
            PressureDependence::Troe {
                high_pressure,
                low_pressure,
                troe,
                efficiencies,
                default_efficiency,
            } => {
                let k_inf = high_pressure.k(t);
                let k_0 = low_pressure.k(t);
                let m = third_body_concentration(concentrations, n_species, efficiencies, *default_efficiency);
                let pr = k_0 * m / k_inf.max(1e-300);
                let f = troe.broadening_factor(t, pr);
                k_inf * pr / (1.0 + pr) * f
            }
        }
    }
}

fn third_body_concentration(
    concentrations: &[f64],
    n_species: usize,
    efficiencies: &[ThirdBodyEfficiency],
    default_efficiency: f64,
) -> f64 {
    let mut m = 0.0;
    for i in 0..n_species.min(concentrations.len()) {
        let eff = efficiencies
            .iter()
            .find(|e| e.species_idx == i)
            .map_or(default_efficiency, |e| e.efficiency);
        m += eff * concentrations[i];
    }
    m
}

pub enum RateLaw {
    /// Constant rate: rate = k * prod(c_i^coeff)
    MassAction(f64),
    /// Temperature-dependent Arrhenius: rate = k(T) * prod(c_i^coeff)
    ArrheniusLaw(Arrhenius),
    /// Temperature + pressure dependent reaction
    PressureDependent {
        arrhenius: Arrhenius,
        pressure: PressureDependence,
    },
    /// Custom rate law evaluated via a user-provided closure.
    /// Takes the current concentration slice and returns the rate.
    Custom(std::sync::Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>),
}

impl std::fmt::Debug for RateLaw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RateLaw::MassAction(k) => write!(f, "MassAction({})", k),
            RateLaw::ArrheniusLaw(arr) => write!(f, "Arrhenius(A={}, b={}, Ea={})", arr.a, arr.b, arr.ea),
            RateLaw::PressureDependent { arrhenius, .. } => {
                write!(f, "PressureDependent(A={}, b={}, Ea={})", arrhenius.a, arrhenius.b, arrhenius.ea)
            }
            RateLaw::Custom(_) => write!(f, "Custom(<closure>)"),
        }
    }
}

impl Clone for RateLaw {
    fn clone(&self) -> Self {
        match self {
            RateLaw::MassAction(k) => RateLaw::MassAction(*k),
            RateLaw::ArrheniusLaw(arr) => RateLaw::ArrheniusLaw(*arr),
            RateLaw::PressureDependent { arrhenius, pressure } => RateLaw::PressureDependent {
                arrhenius: *arrhenius,
                pressure: pressure.clone(),
            },
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

    pub fn with_arrhenius(a: f64, b: f64, ea: f64) -> Self {
        Self {
            reactants: Vec::new(),
            products: Vec::new(),
            rate_law: RateLaw::ArrheniusLaw(Arrhenius::new(a, b, ea)),
        }
    }

    pub fn with_custom_rate<F>(rate_fn: F) -> Self
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        Self {
            reactants: Vec::new(),
            products: Vec::new(),
            rate_law: RateLaw::Custom(std::sync::Arc::new(rate_fn)),
        }
    }

    pub fn add_reactant(mut self, species_idx: usize, coefficient: f64) -> Self {
        self.reactants
            .push(Stoichiometry { species_idx, coefficient });
        self
    }

    pub fn add_product(mut self, species_idx: usize, coefficient: f64) -> Self {
        self.products
            .push(Stoichiometry { species_idx, coefficient });
        self
    }
}

#[derive(Debug, Clone)]
pub struct ReactionSystem {
    pub species: Vec<Species>,
    pub reactions: Vec<Reaction>,
    /// Temperature in Kelvin (used for Arrhenius rate evaluation).
    pub temperature: f64,
}

impl ReactionSystem {
    pub fn new() -> Self {
        Self {
            species: Vec::new(),
            reactions: Vec::new(),
            temperature: 298.15, // default: 25°C
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

    pub fn set_temperature(&mut self, t: f64) {
        self.temperature = t;
    }

    /// Computes the net stoichiometry coefficient for a given species in a given reaction.
    pub fn net_stoichiometry(&self, reaction_idx: usize, species_idx: usize) -> f64 {
        let reaction = &self.reactions[reaction_idx];

        let prod_sum: f64 = reaction
            .products
            .iter()
            .filter(|p| p.species_idx == species_idx)
            .map(|p| p.coefficient)
            .sum();

        let react_sum: f64 = reaction
            .reactants
            .iter()
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

        let idx_a = sys.add_species(Species::new("A", 1.0, 0, Phase::Aqueous));
        let idx_b = sys.add_species(Species::new("B", 1.0, 0, Phase::Aqueous));
        let idx_c = sys.add_species(Species::new("C", 2.0, 0, Phase::Aqueous));

        // R0: 2A + B -> C
        sys.add_reaction(
            Reaction::new(1.5)
                .add_reactant(idx_a, 2.0)
                .add_reactant(idx_b, 1.0)
                .add_product(idx_c, 1.0),
        );

        let s_matrix = sys.stoichiometric_matrix();
        assert_eq!(s_matrix, vec![-2.0, -1.0, 1.0]);
    }

    #[test]
    fn test_arrhenius_rate() {
        let arr = Arrhenius::new(1e13, 0.0, 40000.0);
        let k_300 = arr.k(300.0);
        let k_1000 = arr.k(1000.0);
        // Higher temperature -> higher rate
        assert!(k_1000 > k_300);
        // At 300K with Ea=40kJ/mol, k should be much less than A
        assert!(k_300 < 1e13);
        assert!(k_300 > 0.0);
    }

    #[test]
    fn test_troe_broadening() {
        let troe = TroeParams {
            a: 0.5,
            t3: 1e-30,
            t1: 1e30,
            t2: None,
        };
        let f = troe.broadening_factor(1000.0, 1.0);
        // F should be between 0 and 1 for typical parameters
        assert!(f > 0.0 && f <= 1.0, "Troe F = {}", f);
    }
}
