# linear-algebra-rs

## Usage

```toml
[dependencies]
opensrdk-linear-algebra = "0.1.0"
blas-src = { version = "0.6", features = ["openblas"] }
lapack-src = { version = "0.6", features = ["openblas"] }
```

```rs
extern crate opensrdk_linear_algebra;
extern crate blas_src;
extern crate lapack_src;
```

You can also use accelerate, intel-mkl, or netlib instead.
See [here](https://github.com/blas-lapack-rs/blas-lapack-rs.github.io/wiki).

## Macro

```rs
#[cfg(test)]
mod tests {
    use crate::prelude::*;
    #[test]
    fn it_works() {
        let a = mat!(
            1.0, 0.0;
            0.0, 1.0
        );
        assert_eq!(a[0][0], 1.0);
        assert_eq!(a[0][1], 0.0);
        assert_eq!(a[1][0], 0.0);
        assert_eq!(a[1][1], 1.0);

        let b = mat!(
            &a, zeros!(2, 2);
            zeros!(2, 2), &a
        );

        assert_eq!(b[0][0], 1.0);
        assert_eq!(b[0][1], 0.0);
        assert_eq!(b[3][0], 0.0);
        assert_eq!(b[3][3], 1.0);
    }
}
```

```rs
#[cfg(test)]
mod tests {
    use crate::prelude::*;
    #[test]
    fn it_works() {
        let a = mat!(
            1.0, 2.0;
            3.0, 4.0
        ) * mat!(
            5.0, 6.0;
            7.0, 8.0
        );
        assert_eq!(a[0][0], 19.0)
    }
}
```
