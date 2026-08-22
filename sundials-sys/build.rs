use std::env;
use std::path::PathBuf;

fn main() {
    // Try to find sundials using pkg-config first (useful for local development).
    // In a fully cross-platform setup, we would fallback to compiling Sundials from source via the `cmake` crate.
    let library = pkg_config::Config::new()
        .atleast_version("6.0.0")
        .probe("sundials_cvode");
    
    let include_paths = match library {
        Ok(lib) => lib.include_paths,
        Err(_) => {
            // Fallback: This is where we would use the `cmake` crate to build Sundials from source
            // for cross-compiling to iOS, WASM, etc.
            // For now, we assume it's installed in standard locations.
            println!("cargo:rustc-link-lib=sundials_cvode");
            println!("cargo:rustc-link-lib=sundials_ida");
            println!("cargo:rustc-link-lib=sundials_nvecserial");
            println!("cargo:rustc-link-lib=sundials_sunlinsoldense");
            println!("cargo:rustc-link-lib=sundials_sunmatrixdense");
            println!("cargo:rustc-link-lib=sundials_sunlinsolklu");
            println!("cargo:rustc-link-lib=sundials_sunmatrixsparse");
            // SuiteSparse KLU also requires klu and amd
            println!("cargo:rustc-link-lib=klu");
            println!("cargo:rustc-link-lib=amd");
            vec![PathBuf::from("/usr/include"), PathBuf::from("/usr/include/suitesparse")]
        }
    };

    // Generate bindings
    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .blocklist_item("FP_NAN")
        .blocklist_item("FP_INFINITE")
        .blocklist_item("FP_ZERO")
        .blocklist_item("FP_SUBNORMAL")
        .blocklist_item("FP_NORMAL")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    for path in include_paths {
        builder = builder.clang_arg(format!("-I{}", path.display()));
    }

    let bindings = builder
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
