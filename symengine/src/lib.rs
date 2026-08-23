use symengine_sys::*;
use std::ffi::{CString, CStr};
use std::os::raw::c_char;
use std::ptr;
use std::fmt;

/// A wrapper around SymEngine's basic_struct pointer
pub struct Expression {
    inner: *mut basic_struct,
}

impl Expression {
    fn new() -> Self {
        unsafe {
            let inner = basic_new_heap();
            if inner.is_null() {
                panic!("Failed to allocate SymEngine basic_struct");
            }
            Self { inner }
        }
    }

    pub fn symbol(name: &str) -> Self {
        let mut expr = Self::new();
        let c_name = CString::new(name).unwrap();
        unsafe {
            symbol_set(expr.inner, c_name.as_ptr());
        }
        expr
    }

    pub fn real_double(val: f64) -> Self {
        let mut expr = Self::new();
        unsafe {
            real_double_set_d(expr.inner, val);
        }
        expr
    }

    pub fn integer(val: i64) -> Self {
        let mut expr = Self::new();
        unsafe {
            integer_set_si(expr.inner, val as std::os::raw::c_long);
        }
        expr
    }

    pub fn add(&self, other: &Expression) -> Self {
        let mut result = Self::new();
        unsafe {
            basic_add(result.inner, self.inner, other.inner);
        }
        result
    }

    pub fn mul(&self, other: &Expression) -> Self {
        let mut result = Self::new();
        unsafe {
            basic_mul(result.inner, self.inner, other.inner);
        }
        result
    }

    pub fn pow(&self, other: &Expression) -> Self {
        let mut result = Self::new();
        unsafe {
            basic_pow(result.inner, self.inner, other.inner);
        }
        result
    }

    pub fn diff(&self, symbol: &Expression) -> Self {
        let mut result = Self::new();
        unsafe {
            basic_diff(result.inner, self.inner, symbol.inner);
        }
        result
    }

    /// Substitute a single symbol with another expression: self[old → new].
    pub fn subs2(&self, old: &Expression, new: &Expression) -> Self {
        let mut result = Self::new();
        unsafe {
            basic_subs2(result.inner, self.inner, old.inner, new.inner);
        }
        result
    }

    /// Substitute all symbols in the map: self[k₁→v₁, k₂→v₂, ...].
    pub fn subs(&self, map: &SubstitutionMap) -> Self {
        let mut result = Self::new();
        unsafe {
            basic_subs(result.inner, self.inner, map.inner);
        }
        result
    }

    /// Evaluate to a floating-point number (53-bit double precision).
    /// Returns None if the expression still contains free symbols.
    pub fn eval_double(&self) -> Option<f64> {
        let mut result = Self::new();
        unsafe {
            let status = basic_evalf(result.inner, self.inner, 53, 0);
            if status != 0 {
                return None;
            }
            let val = real_double_get_d(result.inner);
            if val.is_nan() {
                return None;
            }
            Some(val)
        }
    }

    /// Check if the expression is numerically zero.
    pub fn is_zero(&self) -> bool {
        unsafe { number_is_zero(self.inner) != 0 }
    }

    /// Raw pointer access for FFI.
    pub fn as_raw(&self) -> *mut basic_struct {
        self.inner
    }
}

/// A map from symbols to expressions for bulk substitution.
pub struct SubstitutionMap {
    inner: *mut CMapBasicBasic,
}

impl SubstitutionMap {
    pub fn new() -> Self {
        unsafe {
            Self {
                inner: mapbasicbasic_new(),
            }
        }
    }

    /// Insert a symbol → value mapping.
    pub fn insert(&mut self, key: &Expression, value: &Expression) {
        unsafe {
            mapbasicbasic_insert(self.inner, key.inner, value.inner);
        }
    }

    /// Build a substitution map from symbol names and f64 values.
    pub fn from_values(symbols: &[Expression], values: &[f64]) -> Self {
        let mut map = Self::new();
        for (sym, &val) in symbols.iter().zip(values) {
            let val_expr = Expression::real_double(val);
            map.insert(sym, &val_expr);
        }
        map
    }
}

impl Drop for SubstitutionMap {
    fn drop(&mut self) {
        unsafe {
            mapbasicbasic_free(self.inner);
        }
    }
}

impl Drop for Expression {
    fn drop(&mut self) {
        unsafe {
            basic_free_heap(self.inner);
        }
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unsafe {
            let c_str = basic_str(self.inner);
            if c_str.is_null() {
                return Err(fmt::Error);
            }
            let s = CStr::from_ptr(c_str).to_string_lossy().into_owned();
            basic_str_free(c_str);
            write!(f, "{}", s)
        }
    }
}

// Implement standard operators for convenience
impl std::ops::Add for &Expression {
    type Output = Expression;
    fn add(self, other: Self) -> Expression {
        self.add(other)
    }
}

impl std::ops::Mul for &Expression {
    type Output = Expression;
    fn mul(self, other: Self) -> Expression {
        self.mul(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symengine_math() {
        let x = Expression::symbol("x");
        let y = Expression::symbol("y");
        
        // z = x * y + x^2
        let two = Expression::integer(2);
        let x_squared = x.pow(&two);
        let x_times_y = &x * &y;
        let z = &x_times_y + &x_squared;
        
        let dz_dx = z.diff(&x);
        
        // Expected derivative: y + 2*x
        let expected = &y + &(&Expression::integer(2) * &x);
        
        // Basic string validation (internal ordering might vary slightly but usually predictable)
        let dz_dx_str = dz_dx.to_string();
        assert!(dz_dx_str.contains("2*x") || dz_dx_str.contains("x*2"));
        assert!(dz_dx_str.contains("y"));
    }
}
