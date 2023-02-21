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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_scalar() {
        let mut a = SparseTensor::new(vec![3, 2, 2]);
        a[&[0, 0, 0]] = 2.0;
        a[&[0, 0, 1]] = 4.0;
        a[&[1, 1, 0]] = 2.0;
        a[&[1, 1, 1]] = 4.0;
        a[&[2, 0, 0]] = 2.0;
        a[&[2, 0, 1]] = 4.0;

        let mut b = SparseTensor::new(vec![3, 2, 2]);
        b[&[0, 0, 0]] = 4.0;
        b[&[0, 0, 1]] = 6.0;
        b[&[1, 1, 0]] = 4.0;
        b[&[1, 1, 1]] = 6.0;
        b[&[2, 0, 0]] = 4.0;
        b[&[2, 0, 1]] = 6.0;

        assert_eq!(a.clone() + 2.0, b);
        assert_eq!(2.0 + a.clone(), b);
        assert_eq!(&2.0 + a.clone(), b);
        assert_eq!(a + &2.0, b);
    }

    #[test]
    fn add() {
        let mut a = SparseTensor::new(vec![3, 2, 2]);
        a[&[0, 0, 0]] = 2.0;
        a[&[0, 0, 1]] = 4.0;
        a[&[1, 1, 0]] = 2.0;
        a[&[1, 1, 1]] = 4.0;
        a[&[2, 0, 0]] = 2.0;
        a[&[2, 0, 1]] = 4.0;

        let mut b = SparseTensor::new(vec![3, 2, 2]);
        b[&[0, 0, 0]] = 4.0;
        b[&[0, 0, 1]] = 6.0;
        b[&[1, 1, 0]] = 4.0;
        b[&[1, 1, 1]] = 6.0;
        b[&[2, 0, 0]] = 4.0;
        b[&[2, 0, 1]] = 6.0;

        let mut c = SparseTensor::new(vec![3, 2, 2]);
        c[&[0, 0, 0]] = 6.0;
        c[&[0, 0, 1]] = 10.0;
        c[&[1, 1, 0]] = 6.0;
        c[&[1, 1, 1]] = 10.0;
        c[&[2, 0, 0]] = 6.0;
        c[&[2, 0, 1]] = 10.0;

        assert_eq!(a.clone() + b.clone(), b.clone() + a.clone());
        assert_eq!(a.clone() + b.clone(), c.clone());
        assert_eq!(b.clone() + a.clone(), a.clone() + b.clone());
        assert_eq!(a + b, c);
    }
}
