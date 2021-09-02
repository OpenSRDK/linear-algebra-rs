# linear-algebra-rs

## Usage

```toml
[dependencies]
opensrdk-linear-algebra = "0.7.1"
blas-src = { version = "0.8", features = ["openblas"] }
lapack-src = { version = "0.8", features = ["openblas"] }
```

```rust
extern crate opensrdk_linear_algebra;
extern crate blas_src;
extern crate lapack_src;
```

You can also use accelerate, intel-mkl and so on.
See

- [blas-src](https://github.com/blas-lapack-rs/blas-src)
- [lapack-src](https://github.com/blas-lapack-rs/lapack-src)

```rust
use opensrdk_linear_algebra::*;
```

## Examples

- [macros test code](src/macros/mod.rs)
- [operators test code](src/matrix/operators/mul.rs)
