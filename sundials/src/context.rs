use sundials_sys::{SUNContext, SUNContext_Create, SUNContext_Free};
use std::ptr;

pub struct Context {
    inner: SUNContext,
}

impl Context {
    pub fn new() -> Self {
        let mut inner: SUNContext = ptr::null_mut();
        unsafe {
            // SUNContext_Create takes a comm argument, usually NULL for serial, and a pointer to context.
            // On versions 6+, the signature is SUNContext_Create(comm, &ctx)
            // comm is of type *mut c_void (void*)
            SUNContext_Create(ptr::null_mut(), &mut inner);
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
