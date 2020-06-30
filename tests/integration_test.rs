extern crate blas;
extern crate lapack;
extern crate num_complex;
extern crate openblas_src;
extern crate rayon;

use opensrdk_linear_algebra::prelude::*;

#[test]
fn test() {
    let a = mat!(
        1.0, 0.0;
        0.0, 1.0
    );
    let b = &a * &a;

    assert_eq!(b[0][0], 1.0)
}
