use sundials_sys::{SUNContext, SUNContext_Create, SUNContext_Free};
use std::ptr;

pub struct Context {
    inner: SUNContext,
}

impl Context {
    pub fn new() -> Self {
        let mut inner: SUNContext = ptr::null_mut();
        unsafe {
            // SUNContext_Create takes a comm argument, usually 0 for serial when MPI is not enabled.
            // On versions 7+, the signature is SUNContext_Create(comm, &ctx) where comm is an i32 (int).
            SUNContext_Create(0, &mut inner);
        }
        Self { inner }
    }

    pub fn as_raw(&self) -> SUNContext {
        self.inner
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            SUNContext_Free(&mut self.inner);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = Context::new();
        assert!(!ctx.as_raw().is_null(), "Context pointer should not be null");
    }
}
