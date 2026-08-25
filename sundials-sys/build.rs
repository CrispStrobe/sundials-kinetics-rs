use std::env;
use std::path::PathBuf;

use cmake::Config;

fn has_tool(name: &str) -> bool {
    std::process::Command::new(name)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok()
}

struct KluPaths {
    include: String,
    lib: String,
}

fn probe_suitesparse() -> Option<KluPaths> {
    if let Ok(lib) = pkg_config::Config::new().probe("klu") {
        let include = lib
            .include_paths
            .first()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        let libdir = lib
            .link_paths
            .first()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        if !include.is_empty() && !libdir.is_empty() {
            return Some(KluPaths {
                include,
                lib: libdir,
            });
        }
    }

    let include_dirs = [
        "/usr/include/suitesparse",
        "/usr/local/include/suitesparse",
        "/usr/include",
        "/usr/local/include",
        "/opt/homebrew/include/suitesparse",
        "/opt/homebrew/include",
    ];
    let lib_dirs = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
        "/usr/local/lib",
        "/opt/homebrew/lib",
    ];

    let inc = include_dirs
        .iter()
        .find(|d| PathBuf::from(d).join("klu.h").exists())?;

    let lib = lib_dirs.iter().find(|d| {
        PathBuf::from(d).join("libklu.so").exists()
            || PathBuf::from(d).join("libklu.a").exists()
            || PathBuf::from(d).join("libklu.dylib").exists()
    })?;

    Some(KluPaths {
        include: inc.to_string(),
        lib: lib.to_string(),
    })
}

fn main() {
    let target = env::var("TARGET").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    let is_wasm = target.contains("wasm");
    let is_ios = target_os == "ios";

    // KLU: off by default. Enabled by the cargo feature, KEROTAKIS_KLU=1,
    // or auto-detected suitesparse (both header AND library present).
    // Never on wasm/iOS.
    let (use_klu, klu_paths) = if is_wasm || is_ios {
        (false, None)
    } else if env::var("KEROTAKIS_KLU").as_deref() == Ok("1")
        || env::var("CARGO_FEATURE_KLU").is_ok()
    {
        // Explicit request — probe for paths but build even if not found
        // (cmake may still locate them via its own search).
        (true, probe_suitesparse())
    } else {
        // Auto-detect: require both header and library.
        match probe_suitesparse() {
            Some(paths) => (true, Some(paths)),
            None => (false, None),
        }
    };

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

    if is_wasm {
        cmake
            .define("SUNDIALS_BUILD_WITH_MONITORING", "OFF")
            .define("ENABLE_PTHREAD", "OFF")
            .define("ENABLE_OPENMP", "OFF");
    }

    if let Ok(toolchain_file) = env::var("SUNDIALS_CMAKE_TOOLCHAIN_FILE") {
        cmake.define("CMAKE_TOOLCHAIN_FILE", &toolchain_file);
    }

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

    // KLU / SuiteSparse — pass only env-override or probed paths.
    if use_klu {
        cmake.define("ENABLE_KLU", "ON");

        let inc = env::var("KLU_INCLUDE_DIR")
            .ok()
            .or_else(|| klu_paths.as_ref().map(|p| p.include.clone()));
        let lib = env::var("KLU_LIBRARY_DIR")
            .ok()
            .or_else(|| klu_paths.as_ref().map(|p| p.lib.clone()));
        let amd_inc = env::var("AMD_INCLUDE_DIR").ok().or_else(|| inc.clone());
        let amd_lib = env::var("AMD_LIBRARY_DIR").ok().or_else(|| lib.clone());

        if let Some(d) = &inc {
            cmake.define("KLU_INCLUDE_DIR", d);
        }
        if let Some(d) = &lib {
            cmake.define("KLU_LIBRARY_DIR", d);
        }
        if let Some(d) = &amd_inc {
            cmake.define("AMD_INCLUDE_DIR", d);
        }
        if let Some(d) = &amd_lib {
            cmake.define("AMD_LIBRARY_DIR", d);
        }
    } else {
        cmake.define("ENABLE_KLU", "OFF");
    }

    if has_tool("ninja") {
        cmake.generator("Ninja");
    }
    if has_tool("ccache") {
        cmake.define("CMAKE_C_COMPILER_LAUNCHER", "ccache");
        cmake.define("CMAKE_CXX_COMPILER_LAUNCHER", "ccache");
    }

    let dst = cmake.build();

    let lib_dir = dst.join("lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // ── Link Sundials static libraries ─────────────────────────────────
    println!("cargo:rustc-link-lib=static=sundials_core");
    println!("cargo:rustc-link-lib=static=sundials_cvodes");
    println!("cargo:rustc-link-lib=static=sundials_idas");
    println!("cargo:rustc-link-lib=static=sundials_kinsol");
    println!("cargo:rustc-link-lib=static=sundials_arkode");
    println!("cargo:rustc-link-lib=static=sundials_nvecserial");
    println!("cargo:rustc-link-lib=static=sundials_sunmatrixdense");
    println!("cargo:rustc-link-lib=static=sundials_sunmatrixband");
    println!("cargo:rustc-link-lib=static=sundials_sunlinsoldense");
    println!("cargo:rustc-link-lib=static=sundials_sunlinsolband");
    println!("cargo:rustc-link-lib=static=sundials_sunlinsolspgmr");
    println!("cargo:rustc-link-lib=static=sundials_sunlinsolspbcgs");
    println!("cargo:rustc-link-lib=static=sundials_sunlinsolsptfqmr");
    println!("cargo:rustc-link-lib=static=sundials_sunnonlinsolnewton");
    println!("cargo:rustc-link-lib=static=sundials_sunnonlinsolfixedpoint");
    println!("cargo:rustc-link-lib=static=sundials_sunmatrixsparse");

    if use_klu {
        println!("cargo:rustc-link-lib=static=sundials_sunlinsolklu");

        if let Some(paths) = &klu_paths {
            if !paths.lib.is_empty() {
                println!("cargo:rustc-link-search=native={}", paths.lib);
            }
        } else if let Ok(dir) = env::var("KLU_LIBRARY_DIR") {
            println!("cargo:rustc-link-search=native={}", dir);
        }
        println!("cargo:rustc-link-lib=klu");
        println!("cargo:rustc-link-lib=amd");
        println!("cargo:rustc-link-lib=colamd");
        println!("cargo:rustc-link-lib=btf");
        println!("cargo:rustc-link-lib=suitesparseconfig");
    }

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

    if use_klu {
        let klu_inc = env::var("KLU_INCLUDE_DIR")
            .ok()
            .or_else(|| klu_paths.as_ref().map(|p| p.include.clone()));
        if let Some(d) = &klu_inc {
            if !d.is_empty() {
                builder = builder.clang_arg(format!("-I{}", d));
            }
        }
        builder = builder.clang_arg("-DSUNDIALS_KLU_ENABLED");
    }

    let bindings = builder.generate().expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
