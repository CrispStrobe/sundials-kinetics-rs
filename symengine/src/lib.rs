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
