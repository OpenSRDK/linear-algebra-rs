use crate::sparse::RankIndex;
use crate::tensor::Tensor;
use crate::{generate_rank_combinations, RankCombinationId};
use crate::{sparse::SparseTensor, Number};
use std::collections::HashMap;

pub trait DotProduct<T>
where
    T: Number,
{
    fn dot_product(
        self,
        rank_combinations: &[HashMap<RankIndex, RankCombinationId>],
    ) -> SparseTensor<T>;
}

impl<'a, I, T> DotProduct<T> for I
where
    I: Iterator<Item = &'a SparseTensor<T>>,
    T: Number + 'a,
{
    fn dot_product(
        self,
        rank_combinations: &[HashMap<RankIndex, RankCombinationId>],
    ) -> SparseTensor<T> {
        let terms = self.collect::<Vec<_>>();
        let max_rank = terms.iter().map(|t| t.rank()).max().unwrap();
        let mut new_sizes = vec![1; max_rank];
        let mut _rank_combination0 = 0;
        let mut _rank_combination1 = 0;

        for (i, t) in terms.iter().enumerate() {
            for (j, &dim) in t.sizes.iter().enumerate() {
                if rank_combinations[i].get(&j).is_none() && dim > 1 {
                    if new_sizes[j] == 1 {
                        new_sizes[j] = dim;
                    } else {
                        panic!("The tensor whose a rank that is not aggregated and has a dimension greater than 1 can't be included.")
                    }
                } else if i == 0 && rank_combinations[i].get(&j).is_some() {
                    _rank_combination0 = j;
                } else if i == 1 && rank_combinations[i].get(&j).is_some() {
                    _rank_combination1 = j;
                }
            }
        }

        let mut result = SparseTensor::<T>::new(new_sizes.clone());

        fn create_indices(dimensions: &[usize]) -> Vec<Vec<usize>> {
            let mut indices = Vec::new();
            if dimensions.len() == 1 {
                for i in 0..dimensions[0] {
                    indices.push(vec![i]);
                }
            } else {
                for i in 0..dimensions[0] {
                    let sub_array = create_indices(&dimensions[1..]);
                    for j in 0..sub_array.len() {
                        let mut elem = sub_array[j].clone();
                        elem.insert(0, i);
                        indices.push(elem);
                    }
                }
            }
            indices
        }

        let indices = create_indices(&new_sizes);

        if terms[0].sizes[_rank_combination0] != terms[1].sizes[_rank_combination1] {
            panic!("The dimensions of the rank to be aggregated must be the same.");
        }
        for index in indices.iter() {
            for k in 0..terms[0].sizes[_rank_combination0] {
                let mut first_index = index.clone();
                first_index[_rank_combination0] = k;
                let mut second_index = index.clone();
                second_index[_rank_combination1] = k;

                result[&index] += terms[0][&first_index] * terms[1][&second_index];
            }
        }
        result
    }
}

impl<T> SparseTensor<T>
where
    T: Number,
{
    pub fn dot(&self, rhs: &Self, rank_pairs: &[[RankIndex; 2]]) -> Self {
        let rank_combinations = generate_rank_combinations(rank_pairs);

        vec![self, rhs].into_iter().dot_product(&rank_combinations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::SparseTensor;
    #[test]
    fn test_dot_product() {
        let mut a = SparseTensor::<f64>::new(vec![2, 2]);
        a[&[0, 0]] = 1.0;
        a[&[0, 1]] = 2.0;
        a[&[1, 0]] = 3.0;
        a[&[1, 1]] = 4.0;

        let mut b = SparseTensor::<f64>::new(vec![2, 2]);
        b[&[0, 0]] = 2.0;
        b[&[0, 1]] = 4.0;
        b[&[1, 0]] = 6.0;
        b[&[1, 1]] = 8.0;

        let mut c = SparseTensor::<f64>::new(vec![2, 2]);
        c[&[0, 0]] = 14.0;
        c[&[0, 1]] = 20.0;
        c[&[1, 0]] = 30.0;
        c[&[1, 1]] = 44.0;

        let mut d = SparseTensor::<f64>::new(vec![2, 2, 2]);
        d[&[0, 0, 0]] = 1.0;
        d[&[0, 0, 1]] = 2.0;
        d[&[0, 1, 0]] = 3.0;
        d[&[0, 1, 1]] = 4.0;
        d[&[1, 0, 0]] = 5.0;
        d[&[1, 0, 1]] = 6.0;
        d[&[1, 1, 0]] = 7.0;
        d[&[1, 1, 1]] = 8.0;

        let mut e = SparseTensor::<f64>::new(vec![2, 2, 2]);
        e[&[0, 0, 0]] = 2.0;
        e[&[0, 0, 1]] = 4.0;
        e[&[0, 1, 0]] = 6.0;
        e[&[0, 1, 1]] = 8.0;
        e[&[1, 0, 0]] = 10.0;
        e[&[1, 0, 1]] = 12.0;
        e[&[1, 1, 0]] = 14.0;
        e[&[1, 1, 1]] = 16.0;

        let rank_pairs = [[1, 0]];
        let rank_combinations = generate_rank_combinations(&rank_pairs);

        let result = vec![&a, &b].into_iter().dot_product(&rank_combinations);
        assert_eq!(result, c);
    }
}
