use nalgebra::{Dyn, MatrixView, U1, U2};

pub fn strtsw<const SMAX: usize>(
    level: &Vec<usize>,
    f: MatrixView<f64, U1, Dyn, U1, U2>,
    nboxes: usize,
) -> (
    usize,          // s
    [usize; SMAX]   // record
) {
    let mut record = [0; SMAX];
    let mut s = SMAX;
    for j in 0..=nboxes {
        if level[j] > 0 {
            if level[j] < s {
                s = level[j];
            }
            if record[level[j]] == 0 || f[j] < f[record[level[j]]] {
                record[level[j]] = j;
            }
        }
    }
    (s, record)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix2xX;

    #[test]
    fn test_single_element() {
        let level = vec![5];
        let f = Matrix2xX::from_row_slice(&[0.5, 0.5]);
        let nboxes = 0;

        let (result_s, result_record) = strtsw::<10>(&level, f.row(0), nboxes);
        assert_eq!((result_s, result_record), (5, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
    }

    #[test]
    fn test_all_zero_level() {
        let level = vec![0, 0, 0, 0, 0];
        let f = Matrix2xX::from_row_slice(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.0, 0.1, 0.2, 0.3, 0.4]);
        let nboxes = 4;

        let (result_s, result_record) = strtsw::<5>(&level, f.row(0), nboxes);
        assert_eq!((result_s, result_record), (5, [0, 0, 0, 0, 0]));
    }

    #[test]
    fn test_f_values_order_affect() {
        let level = vec![1, 2, 3, 3, 0, 1];
        let f = Matrix2xX::from_row_slice(&[0.5, 0.2, 0.4, 0.1, 0.0, 0.6, 0.5, 0.2, 0.4, 0.1, 0.0, 0.6]);
        let nboxes = 5;

        let (result_s, result_record) = strtsw::<6>(&level, f.row(0), nboxes);
        assert_eq!((result_s, result_record), (1, [0, 5, 1, 3, 0, 0]));
    }

    #[test]
    fn test_varying_levels() {
        let level = vec![0, 4, 3, 2, 1];
        let f = Matrix2xX::from_row_slice(&[-0.5, 0.25, -0.3, 0.4, 0.1, -0.5, 0.25, -0.3, 0.4, 0.1]);
        let nboxes = 4;

        let (result_s, result_record) = strtsw::<5>(&level, f.row(0), nboxes);
        assert_eq!(result_s, 1);
        assert_eq!(result_record, [0, 4, 3, 2, 1]);
    }

    #[test]
    fn test_negative_f_values() {
        let level = vec![1, 3, 3, 2];
        let f = Matrix2xX::from_row_slice(&[-0.5, -0.9, -0.2, -1.0, -0.5, -0.9, -0.2, -1.0]);
        let nboxes = 3;

        let (result_s, result_record) = strtsw::<7>(&level, f.row(0), nboxes);
        assert_eq!((result_s, result_record), (1, [0, 0, 3, 1, 0, 0, 0]));
    }
}
