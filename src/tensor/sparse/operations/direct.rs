use crate::sparse::RankIndex;
use crate::tensor::Tensor;
use crate::{sparse::SparseTensor, Number};
use rand::prelude::*;
use std::collections::HashMap;

pub trait DirectProduct<T>
where
    T: Number,
{
    fn direct_product(self) -> SparseTensor<T>;
}

impl<'a, I, T> DirectProduct<T> for I
where
    I: Iterator<Item = &'a SparseTensor<T>>,
    T: Number + 'a,
{
    fn direct_product(self) -> SparseTensor<T> {
        let terms = self.collect::<Vec<_>>();
        let rhs_size = &terms[terms.len() - 1].sizes;
        println!("size0:{:?}", rhs_size);

        let new_sizes = terms.iter().fold(vec![], |mut acc, &next| {
            if acc.len() < next.sizes.len() {
                for i in 0..acc.len() {
                    acc[i] *= next.size(i);
                }
                acc.extend(next.sizes[acc.len()..].iter());
            } else {
                for i in 0..next.sizes.len() {
                    acc[i] *= next.size(i);
                }
            }
            acc
        });

        let new_elems = terms
            .iter()
            .enumerate()
            .fold(
                Vec::<Vec<(usize, &Vec<usize>)>>::new(),
                |accum, (term_index, &next_term)| {
                    if accum.is_empty() {
                        return next_term
                            .elems
                            .keys()
                            .map(|indices| vec![(term_index, indices)])
                            .collect::<Vec<_>>();
                    };
                    accum
                        .into_iter()
                        .flat_map(|acc| {
                            next_term
                                .elems
                                .keys()
                                .map(|indices| [&acc[..], &[(term_index, indices)]].concat())
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                },
            )
            .into_iter()
            .map(|combination| {
                // println!("{:?}", combination);
                combination.into_iter().fold(
                    (Vec::<usize>::new(), T::default()),
                    |(mut accum_indices, mut accum_value), (term_index, indices)| {
                        // println!(
                        //     "accum_indices:{:?}, accum_value{:?}",
                        //     accum_indices, accum_value
                        // );
                        // println!("term_index{:?}", term_index);

                        if accum_indices.is_empty() {
                            return (indices.clone(), terms[term_index].elem(&indices).clone());
                        }
                        println!("accum_indices1:{:?}", accum_indices);
                        println!("indices:{:?}", indices);
                        // println!("term_index:{:?}", term_index);

                        if accum_indices.len() < indices.len() {
                            for i in 0..accum_indices.len() {
                                accum_indices[i] = (accum_indices[i] + 1) * (indices[i] + 1) - 1;
                            }
                            accum_indices.extend(indices[accum_indices.len()..].iter());
                        } else {
                            for i in 0..indices.len() {
                                accum_indices[i] = accum_indices[i] * rhs_size[i] + indices[i];
                                // if accum_indices[i] == 0 {
                                //     accum_indices[i] = 1;
                                // }
                            }
                        }
                        println!("accum_indices2:{:?}", accum_indices);
                        println!();

                        accum_value *= terms[term_index].elem(&indices).clone();

                        (accum_indices, accum_value)
                    },
                )
            })
            .collect();

        SparseTensor::<T>::from(new_sizes, new_elems).unwrap()
    }
}

impl<T> SparseTensor<T>
where
    T: Number,
{
    pub fn direct(&self, rhs: &Self) -> Self {
        vec![self, rhs].into_iter().direct_product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct() {
        let mut a = SparseTensor::<f64>::new(vec![2, 2]);
        a[&[0, 0]] = 1.0;
        a[&[0, 1]] = 2.0;
        a[&[1, 0]] = 3.0;
        a[&[1, 1]] = 4.0;

        let mut b = SparseTensor::<f64>::new(vec![2, 2]);
        b[&[0, 0]] = 5.0;
        b[&[0, 1]] = 6.0;
        b[&[1, 0]] = 7.0;
        b[&[1, 1]] = 8.0;

        let mut c = SparseTensor::<f64>::new(vec![4, 4]);

        c[&[0, 0]] = 5.0;
        c[&[0, 1]] = 6.0;
        c[&[1, 0]] = 7.0;
        c[&[1, 1]] = 8.0;
        c[&[0, 2]] = 10.0;
        c[&[0, 3]] = 12.0;
        c[&[1, 2]] = 14.0;
        c[&[1, 3]] = 16.0;
        c[&[2, 0]] = 15.0;
        c[&[2, 1]] = 18.0;
        c[&[3, 0]] = 21.0;
        c[&[3, 1]] = 24.0;
        c[&[2, 2]] = 20.0;
        c[&[2, 3]] = 24.0;
        c[&[3, 2]] = 28.0;
        c[&[3, 3]] = 32.0;

        assert_eq!(a.direct(&b), c);
    }
}
