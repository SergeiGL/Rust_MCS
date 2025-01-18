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
- Includes comprehensive tests
- Includes benchmarking suite
- Provides performance analysis tools

## Implementing Your Own Function

Modify the release_func in `feval.rs`:

```rust
#[inline]
fn release_func<const N: usize>(_x: &SVector<f64, N>) -> f64 {
    1. // Implement your function here
}
```

## Usage

Do not forget to update the `feval.rs` file with your own function or compile in a test mode to use the default one.

Read the [Implementing Your Own Function](#implementing-your-own-function) section for more info.

Usage example:

```rust
use nalgebra::{SVector, SMatrix};
use Rust_MCS::{mcs, StopStruct, IinitEnum, ExitFlagEnum};

// Define your optimization bounds
let u = SVector::<f64, 6 >::from_row_slice(& [0.; 6]); // lower bounds
let v = SVector::<f64, 6 >::from_row_slice(& [1.0; 6]); // upper bounds

// Configure stopping criteria
let stop = StopStruct {
nsweeps: 70,                // maximum number of sweeps
freach: f64::NEG_INFINITY,  // target function value
nf: 1_000_000,              // maximum number of function evaluations
};

// Additional parameters
const SMAX: usize = 1_000;                      // number of levels used
let iinit = IinitEnum::Zero;                    // choice of init procedure
let local = 20;                                 // local search level
let gamma = 2e-7;                               // acceptable relative accuracy for local search
let hess = SMatrix::<f64, 6, 6 >::repeat(1.);   // sparsity pattern of Hessian

// Run the optimization
let (xbest, fbest, xmin, fmi, ncall, ncloc, flag) = mcs::<SMAX, 6 > ( & u, & v, & stop, & iinit, local, gamma, & hess);
```

## API Reference

### Main Function

```rust
pub fn mcs<const SMAX: usize, const N: usize>(
    u: &SVector<f64, N>,           // Lower bounds
    v: &SVector<f64, N>,           // Upper bounds
    stop_struct: &StopStruct,      // Stopping criteria
    iinit: &IinitEnum,             // Initialization method
    local: usize,                  // Local search level
    gamma: f64,                    // Acceptable relative accuracy for local search
    hess: &SMatrix<f64, N, N>,     // Sparsity pattern of Hessian
) -> (
    SVector<f64, N>,              // Best solution found
    f64,                          // Corresponding best function value
    Vec<SVector<f64, N>>,         // Local minima coordinates
    Vec<f64>,                     // Corresponding minima values
    usize,                        // Number of function calls
    usize,                        // Number of local searches
    ExitFlagEnum,                 // Exit status
)
```

### Types

```rust
pub enum IinitEnum {
    Zero,              // Simple initialization list
    One,               // Not implemented
    Two,               // Not implemented
    Three,             // Not implemented
}

pub enum ExitFlagEnum {
    NormalShutdown,         // Normal termination
    StopNfExceeded,         // Maximum function evaluations exceeded
    StopNsweepsExceeded,    // Maximum sweeps without improvement exceeded
}
```

## Testing

Using built-in Rust tool:

```bash
cargo test
```

## Benchmarking

Run benchmarks using Criterion:

```bash
cargo bench
```

Generate a flame graph for performance analysis:

```bash
cargo flamegraph --unit-test -- tests::test_for_flamegraph_0
```

The flame graph will be saved as `flamegraph.svg` in your project directory.

To see my flamegraph open the [flamegraph.svg](flamegraph.svg) file.

## Credits

- Original MCS algorithm by [W. Huyer and A. Neumaier](https://arnold-neumaier.at/software/mcs/index.html)