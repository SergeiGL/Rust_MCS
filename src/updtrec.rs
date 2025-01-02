use nalgebra::{Dyn, MatrixView, U1, U2};

pub fn updtrec<const SMAX: usize>(j: usize, s: usize, f: MatrixView<f64, U1, Dyn, U1, U2>, record: &mut [usize; SMAX]) {
    if record.len() < s {
        println!("updtrec: VERY CAREFUL record.len() < s");
    } else if record.len() < s || record[s] == 0 || f[j] < f[record[s]] {
        record[s] = j;
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix2xX;

    #[test]
    fn test_0() {
        let j = 3;
        let s = 3;
        let f = Matrix2xX::<f64>::from_row_slice(&[-0.5, -0.6, -0.7, -2.0, -4.0, -0.5, -0.6, -0.7, -2.0, -4.0]);
        let mut record = [0, 1, 2, 3, 4];

        updtrec(j, s, f.row(0), &mut record);
        assert_eq!(record, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_1() {
        let j = 3;
        let s = 2;
        let f = Matrix2xX::<f64>::from_row_slice(&[-0.5, -0.6, -0.7, -2.0, -4.0, -0.5, -0.6, -0.7, -2.0, -4.0]);
        let mut record = [0, 0, 0, 0, 0];

        updtrec(j, s, f.row(0), &mut record);
        assert_eq!(record, [0, 0, 3, 0, 0]);
    }

    #[test]
    fn test_2() {
        let j = 4;
        let s = 2;
        let f = Matrix2xX::<f64>::from_row_slice(&[-0.5, -0.6, -0.7, -2.0, -1.0, -0.5, -0.6, -0.7, -2.0, -1.0]);
        let mut record = [0, 0, 2, 0, 0];

        updtrec(j, s, f.row(0), &mut record);
        assert_eq!(record, [0, 0, 4, 0, 0]);
    }

    #[test]
    fn test_3() {
        let j = 3;
        let s = 2;
        let f = Matrix2xX::<f64>::from_row_slice(&[-0.5, -0.6, -0.7, -2.0, -4.0, -0.5, -0.6, -0.7, -2.0, -4.0]);
        let mut record = [0, 0, 2, 0, 0];

        updtrec(j, s, f.row(0), &mut record);
        assert_eq!(record, [0, 0, 3, 0, 0]);
    }

    #[test]
    fn test_4() {
        let j = 4;
        let s = 3;
        let f = Matrix2xX::<f64>::from_row_slice(&[-0.5, -0.6, -0.7, -2.0, -1.0, -0.5, -0.6, -0.7, -2.0, -1.0]);
        let mut record = [0, 0, 2, 0, 0];

        updtrec(j, s, f.row(0), &mut record);
        assert_eq!(record, [0, 0, 2, 4, 0]);
    }
}