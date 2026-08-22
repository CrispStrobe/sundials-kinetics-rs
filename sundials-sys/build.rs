use std::env;
use std::path::PathBuf;
use cmake::Config;

fn main() {
    let dst = Config::new("sundials_src")
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("BUILD_STATIC_LIBS", "ON")
        .define("EXAMPLES_ENABLE_C", "OFF")
        .define("EXAMPLES_INSTALL", "OFF")
        .define("BUILD_TESTING", "OFF")
        .define("ENABLE_KLU", "ON")
        // In Ubuntu, SuiteSparse headers are usually in /usr/include/suitesparse
        .define("KLU_INCLUDE_DIR", "/usr/include/suitesparse")
        .define("KLU_LIBRARY_DIR", "/usr/lib/x86_64-linux-gnu")
        .define("AMD_INCLUDE_DIR", "/usr/include/suitesparse")
        .define("AMD_LIBRARY_DIR", "/usr/lib/x86_64-linux-gnu")
        .define("SUNDIALS_INDEX_SIZE", "64") // 64-bit sunindextype
        .build();

    let lib_dir = dst.join("lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Link all Sundials libraries we need (statically)
    // Core
    println!("cargo:rustc-link-lib=static=sundials_core");
    // Solvers (sensitivity-capable variants)
    println!("cargo:rustc-link-lib=static=sundials_cvodes");
    println!("cargo:rustc-link-lib=static=sundials_idas");
    println!("cargo:rustc-link-lib=static=sundials_kinsol");
    println!("cargo:rustc-link-lib=static=sundials_arkode");
    // NVector
    println!("cargo:rustc-link-lib=static=sundials_nvecserial");
    // Matrices
    println!("cargo:rustc-link-lib=static=sundials_sunmatrixdense");
    println!("cargo:rustc-link-lib=static=sundials_sunmatrixband");
    println!("cargo:rustc-link-lib=static=sundials_sunmatrixsparse");
    // Direct linear solvers
    println!("cargo:rustc-link-lib=static=sundials_sunlinsoldense");
    println!("cargo:rustc-link-lib=static=sundials_sunlinsolband");
    println!("cargo:rustc-link-lib=static=sundials_sunlinsolklu");
    // Iterative linear solvers
    println!("cargo:rustc-link-lib=static=sundials_sunlinsolspgmr");
    println!("cargo:rustc-link-lib=static=sundials_sunlinsolspbcgs");
    println!("cargo:rustc-link-lib=static=sundials_sunlinsolsptfqmr");
    // Nonlinear solvers
    println!("cargo:rustc-link-lib=static=sundials_sunnonlinsolnewton");
    println!("cargo:rustc-link-lib=static=sundials_sunnonlinsolfixedpoint");

    // We also need SuiteSparse
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-lib=klu");
    println!("cargo:rustc-link-lib=amd");
    println!("cargo:rustc-link-lib=colamd");
    println!("cargo:rustc-link-lib=btf");
    println!("cargo:rustc-link-lib=suitesparseconfig");
    
    // C math library
    println!("cargo:rustc-link-lib=m");

    // Generate bindings
    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}/include", dst.display()))
        .clang_arg("-I/usr/include/suitesparse")
        .blocklist_item("FP_NAN")
        .blocklist_item("FP_INFINITE")
        .blocklist_item("FP_ZERO")
        .blocklist_item("FP_SUBNORMAL")
        .blocklist_item("FP_NORMAL")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    let bindings = builder
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
