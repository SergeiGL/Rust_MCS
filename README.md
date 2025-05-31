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
    const N: usize = 6; // number of dimensions

    // Optimization Bounds:
    let u = SVector::<f64, N>::from_row_slice(&[0.0; N]); // lower bound
    let v = SVector::<f64, N>::from_row_slice(&[1.0; N]); // upper bound

    let nsweeps = 100; // maximum number of sweeps
    let nf = 20_000; // maximum number of function evaluations

    let local = 50;    // local search level
    let gamma = 2e-14; // acceptable relative accuracy for local search
    let smax = 1_000; // number of levels used

    let hess = SMatrix::<f64, N, N>::repeat(1.); // sparsity pattern of Hessian

    #[inline]
    fn func<const N: usize>(x: &SVector<f64, N>) -> f64 {
        let mut sum = 0.0;
        for i in 0..N {
            sum += (x[i] - 0.12345).powi(2);
        }
        sum
    }

    let (xbest, fbest, xmin, fmi, ncall, ncloc, ExitFlag) = mcs::<N>(func, &u, &v, nsweeps, nf, local, gamma, smax, &hess).unwrap();
}
```

## API Reference

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

### Built-in tests

#### Run [criterion](https://github.com/bheisler/criterion.rs) `benchmarks`:

```bash
cargo bench
```

### Profiling (stack stace graph)

#### Run `flamegraph` profiling tool:

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