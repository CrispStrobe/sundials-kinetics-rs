use std::env;
use std::path::PathBuf;
use cmake::Config;

fn main() {
    let dst = Config::new("symengine_src")
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("BUILD_TESTS", "OFF")
        .define("BUILD_BENCHMARKS", "OFF")
        .define("WITH_SYMENGINE_THREAD_SAFE", "OFF")
        .define("INTEGER_CLASS", "gmp")
        .define("WITH_GMP", "ON")
        .build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=symengine");
    
    // We also need to link GMP and C++ stdlib
    println!("cargo:rustc-link-lib=gmp");
    println!("cargo:rustc-link-lib=stdc++");

    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}/include", dst.display()))
        // Blacklist everything else to only get symengine C API
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
