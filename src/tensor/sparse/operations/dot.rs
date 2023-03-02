use crate::sparse::RankIndex;
use crate::tensor::Tensor;
use crate::{generate_rank_combinations, RankCombinationId, TensorError};
use crate::{sparse::SparseTensor, Number};
use rand::prelude::*;
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

        for (i, t) in terms.iter().enumerate() {
            for (j, &dim) in t.sizes.iter().enumerate() {
                if rank_combinations[i].get(&j).is_none() && dim > 1 {
                    if new_sizes[j] == 1 {
                        new_sizes[j] = dim;
                    } else {
                        panic!("The tensor whose a rank that is not aggregated and has a dimension greater than 1 can't be included.")
                    }
                }
            }
        }

        let mut result = SparseTensor::<T>::new(new_sizes);

        todo!();

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

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::sparse::SparseTensor;
//     use crate::tensor::Tensor;
//     use crate::Number;

//     #[test]
//     fn test_dot_product() {
//         let mut a = SparseTensor::<f64>::new(vec![2, 2]);
//         a[&[0, 0]] = 1.0;
//         a[&[0, 1]] = 2.0;
//         a[&[1, 0]] = 3.0;
//         a[&[1, 1]] = 4.0;

//         let mut b = SparseTensor::<f64>::new(vec![2, 2]);
//         b[&[0, 0]] = 1.0;
//         b[&[0, 1]] = 2.0;
//         b[&[1, 0]] = 3.0;
//         b[&[1, 1]] = 4.0;

//         let mut c = SparseTensor::<f64>::new(vec![2, 2]);
//         c[&[0, 0]] = 1.0;
//         c[&[0, 1]] = 2.0;
//         c[&[1, 0]] = 3.0;
//         c[&[1, 1]] = 4.0;

//         let mut d = SparseTensor::<f64>::new(vec![2, 2]);
//         d[&[0, 0]] = 1.0;
//         d[&[0, 1]] = 2.0;
//         d[&[1, 0]] = 3.0;
//         d[&[1, 1]] = 4.0;

//         let rank_pairs = [[0, 0], [1, 1]];

//         let result = vec![&a, &b, &c, &d].into_iter().dot_product(&rank_pairs);

//         let expected = SparseTensor::<f64>::from_vec(
//             vec![
//                 1., 2., 3., 4., 2., 4., 6., 8., 3., 6., 9., 12., 4., 8., 12., 16.,
//             ],
//             vec![2, 2, 2, 2],
//         );

//         assert_eq!(result, expected);
//     }
// }
