pub mod gv;

use crate::number::Number;

#[derive(Clone, Debug, Default, Hash)]
pub struct CirculantMatrix<T = f64>
where
  T: Number,
{
  col_elems: Vec<T>,
}

impl<T> CirculantMatrix<T>
where
  T: Number,
{
  pub fn new(col_elems: Vec<T>) -> Self {
    Self { col_elems }
  }

  pub fn from(row_elems: &[T]) -> Self {
    let col_elems = if row_elems.len() <= 1 {
      row_elems.to_vec()
    } else {
      row_elems[0..1]
        .iter()
        .chain(row_elems[1..].iter().rev())
        .map(|&e| e)
        .collect::<Vec<_>>()
    };

    Self::new(col_elems)
  }

  pub fn col_elems(&self) -> &[T] {
    &self.col_elems
  }

  pub fn row_elems(&self) -> Vec<T> {
    if self.col_elems.len() <= 1 {
      return self.col_elems.clone();
    }

    self.col_elems[0..1]
      .iter()
      .chain(self.col_elems[1..].iter().rev())
      .map(|&e| e)
      .collect::<Vec<_>>()
  }
}
