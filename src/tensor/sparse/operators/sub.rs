use crate::{
    number::{c64, Number},
    sparse::SparseTensor,
};
use rayon::prelude::*;
use std::ops::{Sub, SubAssign};

fn sub_scalar<T>(lhs: T, rhs: SparseTensor<T>) -> SparseTensor<T>
where
    T: Number,
{
    let mut rhs = rhs;

    rhs.elems
        .par_iter_mut()
        .map(|r| {
            *r.1 -= lhs;
        })
        .collect::<Vec<_>>();

    rhs
}

fn sub<T>(lhs: SparseTensor<T>, rhs: &SparseTensor<T>) -> SparseTensor<T>
where
    T: Number,
{
    if !lhs.is_same_size(rhs) {
        panic!("Dimension mismatch.")
    }
    let mut lhs = lhs;

    todo!();

    lhs
}

// Scalar and SparseTensor

macro_rules! impl_div_scalar {
  {$t: ty} => {
      impl Sub<SparseTensor<$t>> for $t {
          type Output = SparseTensor<$t>;

          fn sub(self, rhs: SparseTensor<$t>) -> Self::Output {
              sub_scalar(self, rhs)
          }
      }

      impl Sub<SparseTensor<$t>> for &$t {
          type Output = SparseTensor<$t>;

          fn sub(self, rhs: SparseTensor<$t>) -> Self::Output {
              sub_scalar(*self, rhs)
          }
      }
  }
}

impl_div_scalar! {f64}
impl_div_scalar! {c64}

// SparseTensor and Scalar

impl<T> Sub<T> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn sub(self, rhs: T) -> Self::Output {
        -sub_scalar(rhs, self)
    }
}

impl<T> Sub<&T> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn sub(self, rhs: &T) -> Self::Output {
        -sub_scalar(*rhs, self)
    }
}

// SparseTensor and SparseTensor

impl<T> Sub<SparseTensor<T>> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn sub(self, rhs: SparseTensor<T>) -> Self::Output {
        sub(self, &rhs)
    }
}

impl<T> Sub<&SparseTensor<T>> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn sub(self, rhs: &SparseTensor<T>) -> Self::Output {
        sub(self, rhs)
    }
}

impl<T> Sub<SparseTensor<T>> for &SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn sub(self, rhs: SparseTensor<T>) -> Self::Output {
        -sub(rhs, self)
    }
}

// SubAssign

impl<T> SubAssign<SparseTensor<T>> for SparseTensor<T>
where
    T: Number,
{
    fn sub_assign(&mut self, rhs: SparseTensor<T>) {
        *self = self as &Self - rhs;
    }
}

#[cfg(test)]
mod tests {
    // use std::{collections::HashMap, hash};

    // use super::*;
    // use crate::sparse::SparseTensor;

    // #[test]

    // cannot work cos neg is not implemented yet

    // fn sub_scalar() {
    //     let mut hash1 = HashMap::new();
    //     hash1.insert(vec![0usize, 0, 0], 1.0);
    //     hash1.insert(vec![0usize, 0, 1], 2.0);
    //     hash1.insert(vec![0usize, 1, 0], 1.0);
    //     hash1.insert(vec![0usize, 1, 1], 2.0);

    //     hash1.insert(vec![1usize, 0, 0], 1.0);
    //     hash1.insert(vec![1usize, 0, 1], 2.0);
    //     hash1.insert(vec![1usize, 1, 0], 2.0);
    //     hash1.insert(vec![1usize, 1, 1], 2.0);

    //     hash1.insert(vec![2usize, 0, 0], 1.0);
    //     hash1.insert(vec![2usize, 0, 1], 2.0);
    //     hash1.insert(vec![2usize, 1, 0], 2.0);
    //     hash1.insert(vec![2usize, 1, 1], 2.0);

    //     let a = SparseTensor::from(vec![3, 2, 2], hash1).unwrap();

    //     let mut hash2 = HashMap::new();
    //     hash2.insert(vec![0usize, 0, 0], 1.0);
    //     hash2.insert(vec![0usize, 0, 1], 2.0);
    //     hash2.insert(vec![0usize, 1, 0], 1.0);
    //     hash2.insert(vec![0usize, 1, 1], 2.0);

    //     hash2.insert(vec![1usize, 0, 0], 1.0);
    //     hash2.insert(vec![1usize, 0, 1], 2.0);
    //     hash2.insert(vec![1usize, 1, 0], 2.0);
    //     hash2.insert(vec![1usize, 1, 1], 2.0);

    //     hash2.insert(vec![2usize, 0, 0], 1.0);
    //     hash2.insert(vec![2usize, 0, 1], 2.0);
    //     hash2.insert(vec![2usize, 1, 0], 2.0);
    //     hash2.insert(vec![2usize, 1, 1], 2.0);

    //     let b = SparseTensor::from(vec![3, 2, 2], hash2).unwrap();

    //     let d = a - 1.0;
    //     let e = b - 1.0;
    //     assert_eq!(d, e);
    // }
}
