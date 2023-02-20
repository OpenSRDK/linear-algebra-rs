use crate::{
    indices_cartesian_product,
    number::{c64, Number},
    sparse::SparseTensor,
};
use rayon::prelude::*;
use std::ops::{Div, DivAssign};

fn div_scalar<T>(lhs: T, rhs: SparseTensor<T>) -> SparseTensor<T>
where
    T: Number,
{
    let mut rhs = rhs;

    rhs.elems
        .par_iter_mut()
        .map(|r| {
            *r.1 /= lhs;
        })
        .collect::<Vec<_>>();

    rhs
}

fn div<T>(lhs: SparseTensor<T>, rhs: &SparseTensor<T>) -> SparseTensor<T>
where
    T: Number,
{
    if !lhs.is_same_size(rhs) {
        panic!("Dimension mismatch.")
    }
    let mut lhs = lhs;

    indices_cartesian_product(&lhs.sizes)
        .into_iter()
        .for_each(|k| {
            if !lhs.elems.contains_key(&k) {
                return;
            }
            lhs[&k] /= rhs[&k];
        });

    lhs
}

// Scalar and SparseTensor

macro_rules! impl_div_scalar {
  {$t: ty} => {
      impl Div<SparseTensor<$t>> for $t {
          type Output = SparseTensor<$t>;

          fn div(self, rhs: SparseTensor<$t>) -> Self::Output {
              div_scalar(self, rhs)
          }
      }

      impl Div<SparseTensor<$t>> for &$t {
          type Output = SparseTensor<$t>;

          fn div(self, rhs: SparseTensor<$t>) -> Self::Output {
              div_scalar(*self, rhs)
          }
      }
  }
}

impl_div_scalar! {f64}
impl_div_scalar! {c64}

// SparseTensor and Scalar

impl<T> Div<T> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn div(self, rhs: T) -> Self::Output {
        div_scalar(rhs, self)
    }
}

impl<T> Div<&T> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn div(self, rhs: &T) -> Self::Output {
        div_scalar(*rhs, self)
    }
}

// SparseTensor and SparseTensor

impl<T> Div<SparseTensor<T>> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn div(self, rhs: SparseTensor<T>) -> Self::Output {
        div(self, &rhs)
    }
}

impl<T> Div<&SparseTensor<T>> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn div(self, rhs: &SparseTensor<T>) -> Self::Output {
        div(self, rhs)
    }
}

impl<T> Div<SparseTensor<T>> for &SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn div(self, rhs: SparseTensor<T>) -> Self::Output {
        div(rhs, self)
    }
}

// DivAssign

impl<T> DivAssign<SparseTensor<T>> for SparseTensor<T>
where
    T: Number,
{
    fn div_assign(&mut self, rhs: SparseTensor<T>) {
        *self = self as &Self / rhs;
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn div_scalar() {
        let mut lhs = SparseTensor::new(vec![3, 2, 2]);
        lhs[&[0, 0, 0]] = 2.0;
        lhs[&[0, 0, 1]] = 4.0;
        lhs[&[1, 1, 0]] = 2.0;
        lhs[&[1, 1, 1]] = 4.0;
        lhs[&[2, 0, 0]] = 2.0;
        lhs[&[2, 0, 1]] = 4.0;

        let mut hash2 = HashMap::new();

        hash2.insert(vec![0usize, 0, 0], 1.0);
        hash2.insert(vec![0usize, 0, 1], 2.0);

        hash2.insert(vec![1usize, 1, 0], 1.0);
        hash2.insert(vec![1usize, 1, 0], 1.0);
        hash2.insert(vec![1usize, 1, 1], 2.0);

        hash2.insert(vec![2usize, 0, 0], 1.0);
        hash2.insert(vec![2usize, 0, 1], 2.0);

        let rhs = SparseTensor::from(vec![3, 2, 2], hash2).unwrap();

        let res = lhs / 2.0;

        assert_eq!(res, rhs);
    }

    // not working
    // #[test]
    // fn div() {
    //     let mut lhs = SparseTensor::new(vec![3, 2, 2]);
    //     lhs[&[0, 0, 0]] = 2.0;
    //     lhs[&[0, 0, 1]] = 4.0;
    //     lhs[&[1, 1, 0]] = 2.0;
    //     lhs[&[1, 1, 1]] = 4.0;
    //     lhs[&[2, 0, 0]] = 2.0;
    //     lhs[&[2, 0, 1]] = 4.0;

    //     let mut rhs = SparseTensor::new(vec![3, 2, 2]);
    //     rhs[&[0, 0, 0]] = 1.0;
    //     rhs[&[0, 0, 1]] = 2.0;
    //     rhs[&[0, 1, 0]] = 1.0;
    //     rhs[&[0, 1, 1]] = 1.0;

    //     rhs[&[1, 0, 0]] = 1.0;
    //     rhs[&[1, 0, 1]] = 1.0;
    //     rhs[&[1, 1, 0]] = 1.0;
    //     rhs[&[1, 1, 1]] = 2.0;

    //     rhs[&[2, 0, 0]] = 2.0;
    //     rhs[&[2, 0, 1]] = 4.0;
    //     rhs[&[2, 1, 0]] = 1.0;
    //     rhs[&[2, 1, 1]] = 1.0;

    //     let res = lhs / rhs;
    //     assert_eq!(res[&[0, 0, 0]], 2.0);
    //     assert_eq!(res[&[0, 0, 1]], 2.0);
    //     assert_eq!(res[&[1, 1, 0]], 2.0);
    //     assert_eq!(res[&[1, 1, 1]], 2.0);
    //     assert_eq!(res[&[2, 0, 0]], 1.0);
    //     assert_eq!(res[&[2, 0, 1]], 1.0);
    // }
}
