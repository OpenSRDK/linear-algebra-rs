use crate::{
    number::{c64, Number},
    sparse::SparseTensor,
};
use rayon::prelude::*;
use std::ops::{Add, AddAssign};

fn add_scalar<T>(lhs: T, rhs: SparseTensor<T>) -> SparseTensor<T>
where
    T: Number,
{
    let mut rhs = rhs;

    rhs.elems
        .par_iter_mut()
        .map(|r| {
            *r.1 += lhs;
        })
        .collect::<Vec<_>>();

    rhs
}

fn add<T>(lhs: SparseTensor<T>, rhs: &SparseTensor<T>) -> SparseTensor<T>
where
    T: Number,
{
    if !lhs.is_same_size(rhs) {
        panic!("Dimension mismatch.")
    }
    let mut lhs = lhs;

    rhs.elems.iter().for_each(|(k, v)| {
        lhs[k] += *v;
    });

    lhs
}

// Scalar and SparseTensor

macro_rules! impl_div_scalar {
    {$t: ty} => {
        impl Add<SparseTensor<$t>> for $t {
            type Output = SparseTensor<$t>;

            fn add(self, rhs: SparseTensor<$t>) -> Self::Output {
                add_scalar(self, rhs)
            }
        }

        impl Add<SparseTensor<$t>> for &$t {
            type Output = SparseTensor<$t>;

            fn add(self, rhs: SparseTensor<$t>) -> Self::Output {
                add_scalar(*self, rhs)
            }
        }
    }
}

impl_div_scalar! {f64}
impl_div_scalar! {c64}

// SparseTensor and Scalar

impl<T> Add<T> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn add(self, rhs: T) -> Self::Output {
        add_scalar(rhs, self)
    }
}

impl<T> Add<&T> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn add(self, rhs: &T) -> Self::Output {
        add_scalar(*rhs, self)
    }
}

// SparseTensor and SparseTensor

impl<T> Add<SparseTensor<T>> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn add(self, rhs: SparseTensor<T>) -> Self::Output {
        add(self, &rhs)
    }
}

impl<T> Add<&SparseTensor<T>> for SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn add(self, rhs: &SparseTensor<T>) -> Self::Output {
        add(self, rhs)
    }
}

impl<T> Add<SparseTensor<T>> for &SparseTensor<T>
where
    T: Number,
{
    type Output = SparseTensor<T>;

    fn add(self, rhs: SparseTensor<T>) -> Self::Output {
        add(rhs, self)
    }
}

// AddAssign

impl<T> AddAssign<SparseTensor<T>> for SparseTensor<T>
where
    T: Number,
{
    fn add_assign(&mut self, rhs: SparseTensor<T>) {
        *self = self as &Self + rhs;
    }
}
