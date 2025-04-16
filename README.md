# MCS Algorithm Rust Implementation

A fast Rust implementation of the Multilevel Coordinate Search (MCS) algorithm for derivative-free optimization, originally developed by Huyer and
Neumaier. This implementation provides significant performance improvements over the original MATLAB and existing Python versions.

## Overview

MCS is a global optimization algorithm that combines global and local search strategies to find the minimum of an N-dimensional function without
requiring derivatives. The algorithm works by:

- Splitting boxes with large unexplored territory (global search)
- Splitting boxes with promising function values (local search)
- Using a sophisticated balance between these two strategies

For detailed information about the algorithm, refer to the [original paper](https://arnold-neumaier.at/ms/mcs.pdf).

## Features

- Written in pure Rust for maximum performance
- Uses recently added const generics for compile-time dimension checking
- Supports arbitrary N-dimensional optimization problems
- Includes 400+ tests
- Includes benchmarking
- Provides performance analysis tools

## GUI interface:

If you are familiar with `Docker`, download user-friendly browser interface:

[Rust_MCS_web](https://github.com/SergeiGL/Rust_MCS_web)

## Manual Setup:

- Add the crate to your `Cargo.toml`

```toml
[dependencies]
Rust_MCS = { git = "https://github.com/SergeiGL/Rust_MCS" }
```

- Example usage in `main.rs` file:

```rust
use nalgebra::{SVector, SMatrix};
use Rust_MCS::*;

fn main() {
    const SMAX: usize = 1_000; // number of levels used
    const N: usize = 6; // number of dimensions

    // Optimization Bounds:
    let u = SVector::<f64, N>::from_row_slice(&[0.0; N]); // lower bound
    let v = SVector::<f64, N>::from_row_slice(&[1.0; N]); // upper bound

    let nsweeps = 100;   // maximum number of sweeps
    let nf = 100_000; // maximum number of function evaluations

    let local = 50;    // local search level
    let gamma = 2e-14; // acceptable relative accuracy for local search

    let hess = SMatrix::<f64, N, N>::repeat(1.); // sparsity pattern of Hessian

    #[inline]
    fn func<const N: usize>(x: &SVector<f64, N>) -> f64 {
        // Implement your function here
    }

    let (xbest, fbest, _, _, _, _, exitflag) = mcs::<SMAX, N>(func, &u, &v, nsweeps, nf, local, gamma, &hess).unwrap();
}
```

## API Reference

### Main Function

```rust
// ONLY SIMPLE INITIALIZATION IS SUPPORTED (IinitEnum::Zero / iinit=0 in Matlab)
pub fn mcs<const SMAX: usize, const N: usize>(
    func: fn(&SVector<f64, N>) -> f64, // Function to optimize
    u: &SVector<f64, N>, // Lower bounds
    v: &SVector<f64, N>, // Upper bounds
    nsweeps: usize, // max number of sweeps; should be > 1
    nf: usize, // max number of function calls
    local: usize, // local search level
    gamma: f64,  // acceptable relative accuracy for local search
    hess: &SMatrix<f64, N, N>, // sparsity pattern of Hessian
) ->
    Result<(
        SVector<f64, N>,       // xbest; Best solution found
        f64,                   // fbest; Corresponding best function value
        Vec<SVector<f64, N>>,  // xmin; Local minima coordinates
        Vec<f64>,              // fmi; Corresponding minima values
        usize,                 // ncall; Number of function calls
        usize,                 // ncloc; Number of local searches
        ExitFlagEnum,          // ExitFlag; Exit status
    ), String>
where
    Const<N>: DimMin<Const<N>, Output=Const<N>>,
{ ... }
```

### Types

```rust
pub enum ExitFlagEnum {
    NormalShutdown,         // Normal termination
    StopNfExceeded,         // Maximum function evaluations exceeded
    StopNsweepsExceeded,    // Maximum sweeps without improvement exceeded
}
```

## Testing

```bash
cargo test
```

## Benchmarking

#### Run [criterion](https://github.com/bheisler/criterion.rs) `benchmarks`:

```bash
cargo bench
```

#### Run `flamegraph`  profiling tool:

- Install [flamegraph](https://github.com/flamegraph-rs/flamegraph)
- Ensure that `Cargo.toml` file contains:

```
[profile.release]
debug = true
```

- Run terminal `as admin`
- Navigate to the Rust_MCS directory:

```
cd [*your_path*]\Rust_MCS
```

- Run:

```bash
cargo flamegraph --bench benchmarks
```

The flame graph will be saved as `flamegraph.svg` in your project directory.

My flamegraph: [flamegraph.svg](flamegraph.svg) file.

## Credits

Original MCS algorithm by [W. Huyer and A. Neumaier](https://arnold-neumaier.at/software/mcs/index.html)