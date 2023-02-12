# linear-algebra-rs

## Usage

```toml
[dependencies]
opensrdk-linear-algebra = "0.8.8"
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

## Routine Naming

### Prefix

- `bd`: **B**i**d**iagonal
- `ci`: **Ci**rculant
- `di`: **Di**agonal
- `ge`: **Ge**neral
  - `sy_he`: **Sy**mmetric, **He**rmitian
    - `po`: **Po**sitive definite
  - `tr`: **Tr**iangle
- `gt`: **G**eneral **t**ridiagonal
- `kr`: **Kr**onecker
<!--- `or_un`: **Or**thogonal, **Un**itary-->
- `sp_hp`: **S**ymmetric **p**acked, **H**ermite **p**acked
  - `pp`: **P**ositive definite **p**acked
- `ss`: **S**par**s**e
- `st`: **S**ymmetric **t**ridiagonal
  - `pt`: **P**ositive definite **t**ridiagonal
- `to`: **To**eplitz

### Suffix

- `sv`: **S**ol**v**e
- `trf`: **Tr**iangle **f**actorization
- `tri`: **Tr**iangle **i**nversion
- `trs`: **Tr**iangle **s**olution
- `svd`: **S**ingular **v**alue **d**ecomposition
- `ev`: **E**igen**v**alues
- `evd`: **E**igen**v**alue **d**ecomposition
- `trd`: **Tr**idiagonal **d**ecomposition
- `det`: **Det**erminant
