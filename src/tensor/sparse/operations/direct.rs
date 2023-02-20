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
                combination.into_iter().fold(
                    (Vec::<usize>::new(), T::default()),
                    |(mut accum_indices, mut accum_value), (term_index, indices)| {
                        if accum_indices.is_empty() {
                            return (indices.clone(), terms[term_index].elem(&indices).clone());
                        }

                        if accum_indices.len() < indices.len() {
                            for i in 0..accum_indices.len() {
                                accum_indices[i] = (accum_indices[i] + 1) * (indices[i] + 1) - 1;
                            }
                            accum_indices.extend(indices[accum_indices.len()..].iter());
                        } else {
                            for i in 0..indices.len() {
                                accum_indices[i] = (accum_indices[i] + 1) * (indices[i] + 1) - 1;
                            }
                        }
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
