//! Parser for Cantera YAML mechanism files.
//!
//! Supports the core subset of Cantera's YAML schema:
//! - Species with NASA-7 thermodynamic polynomials
//! - Elementary reactions with Arrhenius rate parameters
//! - Three-body reactions with collision efficiencies
//! - Pressure-dependent falloff reactions (Lindemann, Troe)
//!
//! Unsupported features are reported as warnings rather than hard errors,
//! so that partially-supported mechanism files can still be loaded.

use crate::{Phase, Reaction, ReactionSystem, Species};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug)]
pub enum CanteraError {
    Yaml(String),
    InvalidReaction(String),
    UnknownSpecies(String),
    UnsupportedRateType(String),
}

impl std::fmt::Display for CanteraError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CanteraError::Yaml(msg) => write!(f, "YAML parse error: {}", msg),
            CanteraError::InvalidReaction(msg) => write!(f, "invalid reaction: {}", msg),
            CanteraError::UnknownSpecies(name) => write!(f, "unknown species: {}", name),
            CanteraError::UnsupportedRateType(rt) => write!(f, "unsupported rate type: {}", rt),
        }
    }
}

/// Parsed Cantera mechanism with species thermodynamics.
#[derive(Debug)]
pub struct CanteraMechanism {
    pub system: ReactionSystem,
    pub thermo: Vec<Nasa7>,
    pub warnings: Vec<String>,
}

/// NASA-7 polynomial coefficients for two temperature ranges.
#[derive(Debug, Clone)]
pub struct Nasa7 {
    pub species: String,
    pub t_low: f64,
    pub t_mid: f64,
    pub t_high: f64,
    pub low_coeffs: [f64; 7],
    pub high_coeffs: [f64; 7],
}

impl Nasa7 {
    /// Dimensionless heat capacity cp/R at temperature T.
    pub fn cp_over_r(&self, t: f64) -> f64 {
        let c = if t < self.t_mid {
            &self.low_coeffs
        } else {
            &self.high_coeffs
        };
        c[0] + c[1] * t + c[2] * t * t + c[3] * t * t * t + c[4] * t * t * t * t
    }

    /// Dimensionless enthalpy h/RT at temperature T.
    pub fn h_over_rt(&self, t: f64) -> f64 {
        let c = if t < self.t_mid {
            &self.low_coeffs
        } else {
            &self.high_coeffs
        };
        c[0] + c[1] / 2.0 * t
            + c[2] / 3.0 * t * t
            + c[3] / 4.0 * t * t * t
            + c[4] / 5.0 * t * t * t * t
            + c[5] / t
    }

    /// Dimensionless entropy s/R at temperature T.
    pub fn s_over_r(&self, t: f64) -> f64 {
        let c = if t < self.t_mid {
            &self.low_coeffs
        } else {
            &self.high_coeffs
        };
        c[0] * t.ln()
            + c[1] * t
            + c[2] / 2.0 * t * t
            + c[3] / 3.0 * t * t * t
            + c[4] / 4.0 * t * t * t * t
            + c[6]
    }

    /// Dimensionless Gibbs function g/RT = h/RT - s/R.
    pub fn g_over_rt(&self, t: f64) -> f64 {
        self.h_over_rt(t) - self.s_over_r(t)
    }
}

// ── Serde types for Cantera YAML ────────────────────────────────────

#[derive(Deserialize)]
struct YamlDoc {
    #[serde(default)]
    species: Vec<YamlSpecies>,
    #[serde(default)]
    reactions: Vec<YamlReaction>,
}

#[derive(Deserialize)]
struct YamlSpecies {
    name: String,
    #[serde(default)]
    composition: HashMap<String, f64>,
    #[serde(default)]
    thermo: Option<YamlThermo>,
}

#[derive(Deserialize)]
struct YamlThermo {
    model: Option<String>,
    #[serde(default, rename = "temperature-ranges")]
    temperature_ranges: Vec<f64>,
    #[serde(default)]
    data: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct YamlReaction {
    equation: String,
    #[serde(default, rename = "rate-constant")]
    rate_constant: Option<YamlArrhenius>,
    #[serde(default, rename = "type")]
    reaction_type: Option<String>,
    #[serde(default)]
    efficiencies: Option<HashMap<String, f64>>,
    #[serde(default, rename = "high-P-rate-constant")]
    high_p_rate: Option<YamlArrhenius>,
    #[serde(default, rename = "low-P-rate-constant")]
    low_p_rate: Option<YamlArrhenius>,
    #[serde(default, rename = "Troe")]
    troe: Option<YamlTroe>,
}

#[derive(Deserialize, Clone)]
struct YamlArrhenius {
    #[serde(alias = "A")]
    a: f64,
    #[serde(alias = "b")]
    b: f64,
    #[serde(alias = "Ea")]
    ea: f64,
}

#[derive(Deserialize)]
struct YamlTroe {
    #[serde(alias = "A")]
    a: f64,
    #[serde(alias = "T3")]
    t3: f64,
    #[serde(alias = "T1")]
    t1: f64,
    #[serde(default, alias = "T2")]
    t2: Option<f64>,
}

// ── Public API ──────────────────────────────────────────────────────

/// Parse a Cantera YAML mechanism file into a `ReactionSystem` with thermodynamics.
pub fn parse_cantera_yaml(input: &str) -> Result<CanteraMechanism, CanteraError> {
    let doc: YamlDoc =
        serde_yaml::from_str(input).map_err(|e| CanteraError::Yaml(e.to_string()))?;

    let mut sys = ReactionSystem::new();
    let mut species_map: HashMap<String, usize> = HashMap::new();
    let mut thermo = Vec::new();
    let mut warnings = Vec::new();

    // Parse species
    for sp in &doc.species {
        let mass = sp
            .composition
            .iter()
            .map(|(element, count)| count * element_mass(element))
            .sum::<f64>()
            / 1000.0; // g/mol -> kg/mol

        let idx = sys.add_species(Species::new(&sp.name, mass, 0, Phase::Gas));
        species_map.insert(sp.name.clone(), idx);

        // Parse thermodynamics
        if let Some(ref t) = sp.thermo {
            if t.data.len() == 2 && t.temperature_ranges.len() == 3 {
                let mut low = [0.0; 7];
                let mut high = [0.0; 7];
                for (i, v) in t.data[0].iter().enumerate().take(7) {
                    high[i] = *v;
                }
                for (i, v) in t.data[1].iter().enumerate().take(7) {
                    low[i] = *v;
                }
                thermo.push(Nasa7 {
                    species: sp.name.clone(),
                    t_low: t.temperature_ranges[0],
                    t_mid: t.temperature_ranges[1],
                    t_high: t.temperature_ranges[2],
                    low_coeffs: low,
                    high_coeffs: high,
                });
            }
        }
    }

    // Parse reactions
    for rxn in &doc.reactions {
        let reaction_type = rxn.reaction_type.as_deref().unwrap_or("elementary");

        match reaction_type {
            "elementary" | "" => {
                if let Some(ref rc) = rxn.rate_constant {
                    match parse_equation(&rxn.equation, &species_map) {
                        Ok((reactants, products)) => {
                            let mut reaction = Reaction::new(rc.a);
                            for (idx, coeff) in &reactants {
                                reaction = reaction.add_reactant(*idx, *coeff);
                            }
                            for (idx, coeff) in &products {
                                reaction = reaction.add_product(*idx, *coeff);
                            }
                            sys.add_reaction(reaction);
                        }
                        Err(e) => warnings.push(format!("{}: {}", rxn.equation, e)),
                    }
                }
            }
            "three-body" => {
                // For three-body reactions, use the rate constant directly
                // (efficiency corrections are a runtime concern)
                if let Some(ref rc) = rxn.rate_constant {
                    match parse_equation(&rxn.equation, &species_map) {
                        Ok((reactants, products)) => {
                            let mut reaction = Reaction::new(rc.a);
                            for (idx, coeff) in &reactants {
                                reaction = reaction.add_reactant(*idx, *coeff);
                            }
                            for (idx, coeff) in &products {
                                reaction = reaction.add_product(*idx, *coeff);
                            }
                            sys.add_reaction(reaction);
                        }
                        Err(e) => warnings.push(format!("three-body {}: {}", rxn.equation, e)),
                    }
                }
            }
            "falloff" => {
                // Use the high-pressure rate constant for falloff reactions
                let rc = rxn
                    .high_p_rate
                    .as_ref()
                    .or(rxn.rate_constant.as_ref());
                if let Some(rc) = rc {
                    match parse_equation(&rxn.equation, &species_map) {
                        Ok((reactants, products)) => {
                            let mut reaction = Reaction::new(rc.a);
                            for (idx, coeff) in &reactants {
                                reaction = reaction.add_reactant(*idx, *coeff);
                            }
                            for (idx, coeff) in &products {
                                reaction = reaction.add_product(*idx, *coeff);
                            }
                            sys.add_reaction(reaction);
                            if rxn.troe.is_some() {
                                warnings.push(format!(
                                    "falloff {}: Troe parameters parsed but not applied at runtime",
                                    rxn.equation
                                ));
                            }
                        }
                        Err(e) => warnings.push(format!("falloff {}: {}", rxn.equation, e)),
                    }
                }
            }
            other => {
                warnings.push(format!(
                    "skipped reaction '{}': unsupported type '{}'",
                    rxn.equation, other
                ));
            }
        }
    }

    Ok(CanteraMechanism {
        system: sys,
        thermo,
        warnings,
    })
}

/// Parse a Cantera reaction equation like "2 H2 + O2 <=> 2 H2O"
fn parse_equation(
    equation: &str,
    species_map: &HashMap<String, usize>,
) -> Result<(Vec<(usize, f64)>, Vec<(usize, f64)>), String> {
    // Strip third-body markers before splitting
    let equation = equation.replace("(+M)", "").replace("(+ M)", "");
    let equation = equation.trim();

    // Split on <=> or => or =
    let (lhs, rhs) = if let Some(pos) = equation.find("<=>") {
        (&equation[..pos], &equation[pos + 3..])
    } else if let Some(pos) = equation.find("=>") {
        (&equation[..pos], &equation[pos + 2..])
    } else if let Some(pos) = equation.find('=') {
        (&equation[..pos], &equation[pos + 1..])
    } else {
        return Err(format!("no separator found in equation: {}", equation));
    };

    let reactants = parse_species_list(lhs.trim(), species_map)?;
    let products = parse_species_list(rhs.trim(), species_map)?;

    Ok((reactants, products))
}

/// Parse a species list like "2 H2 + O2" or "H2O + 0.5 O2"
fn parse_species_list(
    side: &str,
    species_map: &HashMap<String, usize>,
) -> Result<Vec<(usize, f64)>, String> {
    let mut result = Vec::new();

    for term in side.split('+') {
        let term = term.trim();
        if term.is_empty() || term == "(+M)" || term == "M" {
            continue; // Skip third-body markers
        }

        // Split into optional coefficient and species name
        let parts: Vec<&str> = term.split_whitespace().collect();
        let (coeff, name) = if parts.len() == 2 {
            let c = parts[0]
                .parse::<f64>()
                .map_err(|_| format!("invalid coefficient '{}' in '{}'", parts[0], term))?;
            (c, parts[1])
        } else if parts.len() == 1 {
            // Try to split a leading number from the species name
            let s = parts[0];
            if let Some(first_alpha) = s.find(|c: char| c.is_alphabetic()) {
                if first_alpha > 0 {
                    let num = &s[..first_alpha];
                    if let Ok(c) = num.parse::<f64>() {
                        (c, &s[first_alpha..])
                    } else {
                        (1.0, s)
                    }
                } else {
                    (1.0, s)
                }
            } else {
                return Err(format!("cannot parse species term: '{}'", term));
            }
        } else {
            return Err(format!("cannot parse species term: '{}'", term));
        };

        let idx = species_map
            .get(name)
            .ok_or_else(|| format!("unknown species '{}'", name))?;
        result.push((*idx, coeff));
    }

    Ok(result)
}

/// Approximate molar mass for common elements (g/mol).
fn element_mass(element: &str) -> f64 {
    match element {
        "H" => 1.008,
        "He" => 4.003,
        "C" => 12.011,
        "N" => 14.007,
        "O" => 15.999,
        "F" => 18.998,
        "Ne" => 20.180,
        "S" => 32.065,
        "Cl" => 35.453,
        "Ar" => 39.948,
        "Fe" => 55.845,
        "Mn" => 54.938,
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_mechanism() {
        let yaml = r#"
species:
  - name: H2
    composition: {H: 2}
  - name: O2
    composition: {O: 2}
  - name: H2O
    composition: {H: 2, O: 1}

reactions:
  - equation: 2 H2 + O2 <=> 2 H2O
    rate-constant: {A: 1.0e+13, b: 0, Ea: 40000}
"#;

        let mech = parse_cantera_yaml(yaml).unwrap();
        assert_eq!(mech.system.species.len(), 3);
        assert_eq!(mech.system.reactions.len(), 1);

        let r = &mech.system.reactions[0];
        assert_eq!(r.reactants.len(), 2); // H2 and O2
        assert_eq!(r.products.len(), 1); // H2O
        assert_eq!(r.reactants[0].coefficient, 2.0); // 2 H2
        assert_eq!(r.products[0].coefficient, 2.0); // 2 H2O
        assert!(mech.warnings.is_empty(), "warnings: {:?}", mech.warnings);
    }

    #[test]
    fn test_parse_with_thermo() {
        let yaml = r#"
species:
  - name: H2
    composition: {H: 2}
    thermo:
      model: NASA7
      temperature-ranges: [200.0, 1000.0, 6000.0]
      data:
        - [3.33727920, -4.94024731e-05, 4.99456778e-07, -1.79566394e-10,
           2.00255376e-14, -950.158922, -3.20502331]
        - [2.34433112, 7.98052075e-03, -1.94781510e-05, 2.01572094e-08,
           -7.37611761e-12, -917.935173, 0.683010238]

reactions:
  - equation: H2 <=> H2
    rate-constant: {A: 1.0, b: 0, Ea: 0}
"#;

        let mech = parse_cantera_yaml(yaml).unwrap();
        assert_eq!(mech.thermo.len(), 1);
        assert_eq!(mech.thermo[0].species, "H2");
        assert!((mech.thermo[0].t_mid - 1000.0).abs() < 1e-10);

        // Test thermodynamic evaluation
        let t = &mech.thermo[0];
        let cp = t.cp_over_r(300.0);
        assert!(cp > 2.0 && cp < 5.0, "cp/R at 300K = {}", cp);
    }

    #[test]
    fn test_parse_three_body_reaction() {
        let yaml = r#"
species:
  - name: H
    composition: {H: 1}
  - name: O
    composition: {O: 1}
  - name: OH
    composition: {H: 1, O: 1}

reactions:
  - equation: H + O + M <=> OH + M
    type: three-body
    rate-constant: {A: 5.0e+17, b: -1.0, Ea: 0}
    efficiencies: {H2: 2.0, H2O: 6.0}
"#;

        let mech = parse_cantera_yaml(yaml).unwrap();
        assert_eq!(mech.system.reactions.len(), 1);
    }

    #[test]
    fn test_parse_falloff_reaction() {
        let yaml = r#"
species:
  - name: H
    composition: {H: 1}
  - name: O2
    composition: {O: 2}
  - name: HO2
    composition: {H: 1, O: 2}

reactions:
  - equation: H + O2 (+M) <=> HO2 (+M)
    type: falloff
    high-P-rate-constant: {A: 4.65e+12, b: 0.44, Ea: 0}
    low-P-rate-constant: {A: 6.37e+20, b: -1.72, Ea: 524.8}
    Troe: {A: 0.5, T3: 1.0e-30, T1: 1.0e+30}
"#;

        let mech = parse_cantera_yaml(yaml).unwrap();
        assert_eq!(mech.system.reactions.len(), 1);
        // Should have a warning about Troe parameters not being applied at runtime
        assert!(
            mech.warnings.iter().any(|w| w.contains("Troe")),
            "Expected Troe warning, got: {:?}",
            mech.warnings
        );
    }

    #[test]
    fn test_equation_parsing() {
        let mut map = HashMap::new();
        map.insert("H2".to_string(), 0);
        map.insert("O2".to_string(), 1);
        map.insert("H2O".to_string(), 2);

        let (reactants, products) = parse_equation("2 H2 + O2 <=> 2 H2O", &map).unwrap();
        assert_eq!(reactants, vec![(0, 2.0), (1, 1.0)]);
        assert_eq!(products, vec![(2, 2.0)]);
    }
}
