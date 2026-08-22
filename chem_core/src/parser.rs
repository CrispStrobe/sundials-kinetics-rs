use crate::{ReactionSystem, Species, Reaction, Phase};
use std::collections::HashMap;

#[derive(Debug, PartialEq)]
pub enum ParserError {
    InvalidFormat(String),
    UnknownSpecies(String),
}

pub struct MechanismParser;

impl MechanismParser {
    /// Parses a simplified Chemkin-like mechanism file.
    /// Format expected:
    /// SPECIES
    /// H2O2
    /// H2O
    /// O2
    /// END
    /// REACTIONS
    /// H2O2 => H2O + 0.5O2  10.0
    /// END
    pub fn parse(input: &str) -> Result<ReactionSystem, ParserError> {
        let mut sys = ReactionSystem::new();
        let mut species_map = HashMap::new();
        
        let mut section = "";
        
        for line in input.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('!') || line.starts_with('#') {
                continue;
            }
            
            if line == "SPECIES" {
                section = "SPECIES";
                continue;
            } else if line == "REACTIONS" {
                section = "REACTIONS";
                continue;
            } else if line == "END" {
                section = "";
                continue;
            }
            
            if section == "SPECIES" {
                // For simplified parsing, we assume default properties (mass 0, Aqueous)
                let name = line.to_string();
                let idx = sys.add_species(Species::new(&name, 0.0, 0, Phase::Aqueous));
                species_map.insert(name, idx);
            } else if section == "REACTIONS" {
                Self::parse_reaction_line(line, &mut sys, &species_map)?;
            }
        }
        
        Ok(sys)
    }

    fn parse_reaction_line(
        line: &str, 
        sys: &mut ReactionSystem, 
        species_map: &HashMap<String, usize>
    ) -> Result<(), ParserError> {
        // Example: "H2O2 => H2O + 0.5O2  10.0"
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() { return Ok(()); }
        
        // The last token is the rate constant
        let rate_str = parts.last().unwrap();
        let rate_constant = rate_str.parse::<f64>().map_err(|_| {
            ParserError::InvalidFormat(format!("Invalid rate constant: {}", rate_str))
        })?;
        
        let eqn_str = line[..line.len() - rate_str.len()].trim();
        let sides: Vec<&str> = eqn_str.split("=>").collect();
        if sides.len() != 2 {
            // Also try '='
            let sides_eq: Vec<&str> = eqn_str.split('=').collect();
            if sides_eq.len() != 2 {
                return Err(ParserError::InvalidFormat(format!("Reaction must have => or = : {}", eqn_str)));
            }
        }
        let sides = if sides.len() == 2 { sides } else { eqn_str.split('=').collect() };
        
        let mut reaction = Reaction::new(rate_constant);
        
        // Parse reactants
        Self::parse_side(sides[0], species_map, |idx, coeff| {
            reaction = reaction.clone().add_reactant(idx, coeff);
        })?;
        
        // Parse products
        Self::parse_side(sides[1], species_map, |idx, coeff| {
            reaction = reaction.clone().add_product(idx, coeff);
        })?;
        
        sys.add_reaction(reaction);
        Ok(())
    }

    fn parse_side<F>(
        side_str: &str, 
        species_map: &HashMap<String, usize>,
        mut add_fn: F
    ) -> Result<(), ParserError> 
    where F: FnMut(usize, f64) 
    {
        let side_str = side_str.trim();
        if side_str.is_empty() { return Ok(()); }
        
        let tokens = side_str.split('+');
        for token in tokens {
            let token = token.trim();
            if token.is_empty() { continue; }
            
            // Extract optional coefficient. E.g. "0.5O2", "2H2O", "H2O2"
            let mut coeff = 1.0;
            let mut name = token;
            
            // Find where alphabetic chars start
            if let Some(first_alpha) = token.find(|c: char| c.is_alphabetic()) {
                if first_alpha > 0 {
                    let num_str = &token[..first_alpha].trim();
                    if !num_str.is_empty() {
                        coeff = num_str.parse::<f64>().map_err(|_| {
                            ParserError::InvalidFormat(format!("Invalid coefficient: {}", num_str))
                        })?;
                    }
                    name = &token[first_alpha..];
                }
            }
            
            let name = name.trim();
            let idx = species_map.get(name).ok_or_else(|| {
                ParserError::UnknownSpecies(name.to_string())
            })?;
            
            add_fn(*idx, coeff);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mechanism_parser() {
        let input = "
SPECIES
H2O2
H2O
O2
END
REACTIONS
H2O2 => H2O + 0.5O2  10.0
2H2O = H2O2 + H2O2   0.1
END
";
        let sys = MechanismParser::parse(input).unwrap();
        
        assert_eq!(sys.species.len(), 3);
        assert_eq!(sys.species[0].name, "H2O2");
        assert_eq!(sys.species[1].name, "H2O");
        assert_eq!(sys.species[2].name, "O2");
        
        assert_eq!(sys.reactions.len(), 2);
        
        let r1 = &sys.reactions[0];
        assert_eq!(r1.reactants.len(), 1);
        assert_eq!(r1.reactants[0].coefficient, 1.0);
        assert_eq!(r1.products.len(), 2);
        assert_eq!(r1.products[1].coefficient, 0.5);
        
        let r2 = &sys.reactions[1];
        assert_eq!(r2.reactants.len(), 1);
        assert_eq!(r2.reactants[0].coefficient, 2.0);
        assert_eq!(r2.products.len(), 2);
    }
}
