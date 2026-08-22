use std::env;
use std::path::PathBuf;

use cmake::Config;

fn main() {
    let target = env::var("TARGET").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    let use_klu = env::var("CARGO_FEATURE_KLU").is_ok();
    let is_wasm = target.contains("wasm");
    let is_ios = target_os == "ios";

    // ── CMake configuration ────────────────────────────────────────────
    let mut cmake = Config::new("sundials_src");
    cmake
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("BUILD_STATIC_LIBS", "ON")
        .define("EXAMPLES_ENABLE_C", "OFF")
        .define("EXAMPLES_INSTALL", "OFF")
        .define("BUILD_TESTING", "OFF")
        .define("SUNDIALS_INDEX_SIZE", "64");

    // Disable pthreads for single-threaded targets
    if is_wasm {
        cmake
            .define("SUNDIALS_BUILD_WITH_MONITORING", "OFF")
            .define("ENABLE_PTHREAD", "OFF")
            .define("ENABLE_OPENMP", "OFF");
    }

    // Cross-compilation toolchain file support
    if let Ok(toolchain_file) = env::var("SUNDIALS_CMAKE_TOOLCHAIN_FILE") {
        cmake.define("CMAKE_TOOLCHAIN_FILE", &toolchain_file);
    }

    // iOS-specific CMake settings
    if is_ios {
        let sdk = if target.contains("sim") {
            "iphonesimulator"
        } else {
            "iphoneos"
        };
        cmake
            .define("CMAKE_SYSTEM_NAME", "iOS")
            .define("CMAKE_OSX_SYSROOT", sdk);
        match target_arch.as_str() {
            "aarch64" => {
                cmake.define("CMAKE_OSX_ARCHITECTURES", "arm64");
            }
            "x86_64" => {
                cmake.define("CMAKE_OSX_ARCHITECTURES", "x86_64");
            }
            _ => {}
        }
    }

    // KLU / SuiteSparse configuration
    if use_klu && !is_wasm && !is_ios {
        cmake.define("ENABLE_KLU", "ON");

        // Allow overriding SuiteSparse paths via environment variables
        let klu_include = env::var("KLU_INCLUDE_DIR")
            .unwrap_or_else(|_| "/usr/include/suitesparse".to_string());
        let klu_lib = env::var("KLU_LIBRARY_DIR").unwrap_or_else(|_| {
            format!("/usr/lib/{}-linux-gnu", target_arch)
        });
        let amd_include = env::var("AMD_INCLUDE_DIR").unwrap_or_else(|_| klu_include.clone());
        let amd_lib = env::var("AMD_LIBRARY_DIR").unwrap_or_else(|_| klu_lib.clone());

        cmake
            .define("KLU_INCLUDE_DIR", &klu_include)
            .define("KLU_LIBRARY_DIR", &klu_lib)
            .define("AMD_INCLUDE_DIR", &amd_include)
            .define("AMD_LIBRARY_DIR", &amd_lib);
    } else {
        cmake.define("ENABLE_KLU", "OFF");
    }

    let dst = cmake.build();

    let lib_dir = dst.join("lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // ── Link Sundials static libraries ─────────────────────────────────
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
    // Direct linear solvers
    println!("cargo:rustc-link-lib=static=sundials_sunlinsoldense");
    println!("cargo:rustc-link-lib=static=sundials_sunlinsolband");
    // Iterative linear solvers
    println!("cargo:rustc-link-lib=static=sundials_sunlinsolspgmr");
    println!("cargo:rustc-link-lib=static=sundials_sunlinsolspbcgs");
    println!("cargo:rustc-link-lib=static=sundials_sunlinsolsptfqmr");
    // Nonlinear solvers
    println!("cargo:rustc-link-lib=static=sundials_sunnonlinsolnewton");
    println!("cargo:rustc-link-lib=static=sundials_sunnonlinsolfixedpoint");

    // Sparse matrix is always built; KLU solver only with feature
    println!("cargo:rustc-link-lib=static=sundials_sunmatrixsparse");

    if use_klu && !is_wasm && !is_ios {
        println!("cargo:rustc-link-lib=static=sundials_sunlinsolklu");

        // Link SuiteSparse libraries
        let suitesparse_lib = env::var("KLU_LIBRARY_DIR").unwrap_or_else(|_| {
            format!("/usr/lib/{}-linux-gnu", target_arch)
        });
        println!("cargo:rustc-link-search=native={}", suitesparse_lib);
        println!("cargo:rustc-link-lib=klu");
        println!("cargo:rustc-link-lib=amd");
        println!("cargo:rustc-link-lib=colamd");
        println!("cargo:rustc-link-lib=btf");
        println!("cargo:rustc-link-lib=suitesparseconfig");
    }

    // C math library (not needed on WASM)
    if !is_wasm {
        println!("cargo:rustc-link-lib=m");
    }

    // ── Generate bindings ──────────────────────────────────────────────
    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}/include", dst.display()))
        .blocklist_item("FP_NAN")
        .blocklist_item("FP_INFINITE")
        .blocklist_item("FP_ZERO")
        .blocklist_item("FP_SUBNORMAL")
        .blocklist_item("FP_NORMAL")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    if use_klu && !is_wasm && !is_ios {
        let klu_include = env::var("KLU_INCLUDE_DIR")
            .unwrap_or_else(|_| "/usr/include/suitesparse".to_string());
        builder = builder
            .clang_arg(format!("-I{}", klu_include))
            .clang_arg("-DSUNDIALS_KLU_ENABLED");
    }

    let bindings = builder.generate().expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
