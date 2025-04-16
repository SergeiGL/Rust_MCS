use nalgebra::{Dyn, MatrixView, U1, U2};

#[inline]
pub(crate) fn updtrec<const SMAX: usize>(j: usize, s: usize, f: MatrixView<f64, U1, Dyn, U1, U2>, record: &mut [Option<usize>; SMAX]) {
    // s: as in Matlab
    // j: -1 from Matlab
    // record: -1 from Matlab in terms of values; record.len(): +1 from Matlab

    debug_assert!(record.len() >= s + 1); // updtrec: VERY CAREFUL record.len() < s"
    if record.len() < s + 1 || record[s - 1].is_none() || f[j] < f[record[s - 1].expect("Previous check: record[s - 1].is_none()")] {
        record[s - 1] = Some(j);
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix2xX;

    #[test]
    fn test_0() {
        // Matlab test
        //
        // clearvars;
        // clear global;
        // j = 4;
        // s = 3;
        // f = [-0.5, -0.6, -0.7, -2.0, -4.0];
        // global record;
        // record = [1,2,3,4,5];
        //
        // updtrec(j,s,f);
        // disp(record)

        let j = 3;
        let s = 3;
        let f = Matrix2xX::<f64>::from_row_slice(&[-0.5, -0.6, -0.7, -2.0, -4.0, -0.5, -0.6, -0.7, -2.0, -4.0]);
        let mut record = [Some(0), Some(1), Some(2), Some(3), Some(4), None];

        updtrec(j, s, f.row(0), &mut record);
        assert_eq!(record, [Some(0), Some(1), Some(3), Some(3), Some(4), None]);
    }

    #[test]
    fn test_1() {
        // Matlab test
        //
        // clearvars;
        // clear global;
        // j = 3;
        // s = 2;
        // f = [-0.5, -0.6, -0.7, -2.0, -4.0];
        // global record;
        // record = [1,1,1,1,1];
        //
        // updtrec(j,s,f);
        // disp(record)

        let j = 2;
        let s = 2;
        let f = Matrix2xX::<f64>::from_row_slice(&[-0.5, -0.6, -0.7, -2.0, -4.0, -0.5, -0.6, -0.7, -2.0, -4.0]);
        let mut record = [Some(0), Some(0), Some(0), Some(0), Some(0), None];

        updtrec(j, s, f.row(0), &mut record);
        assert_eq!(record, [Some(0), Some(2), Some(0), Some(0), Some(0), None]);
    }

    #[test]
    fn test_2() {
        // Matlab test
        //
        // clearvars;
        // clear global;
        // j = 4;
        // s = 2;
        // f = [-0.5, -0.6, -0.7, -2.0, -1.0];
        // global record;
        // record = [1,1,3,1,1];
        //
        // updtrec(j,s,f);
        // disp(record)

        let j = 3;
        let s = 2;
        let f = Matrix2xX::<f64>::from_row_slice(&[-0.5, -0.6, -0.7, -2.0, -1.0, -0.5, -0.6, -0.7, -2.0, -1.0]);
        let mut record = [Some(0), Some(0), Some(2), Some(0), Some(0), None];

        updtrec(j, s, f.row(0), &mut record);
        assert_eq!(record, [Some(0), Some(3), Some(2), Some(0), Some(0), None]);
    }

    #[test]
    fn test_3() {
        // Matlab test
        //
        // clearvars;
        // clear global;
        // j = 3;
        // s = 2;
        // f = [-0.5, -0.6, -0.7, -2.0, -4.0];
        // global record;
        // record = [1,1,3,1,1];
        //
        // updtrec(j,s,f);
        // disp(record)

        let j = 2;
        let s = 2;
        let f = Matrix2xX::<f64>::from_row_slice(&[-0.5, -0.6, -0.7, -2.0, -4.0, -0.5, -0.6, -0.7, -2.0, -4.0]);
        let mut record = [Some(0), Some(0), Some(2), Some(0), Some(0), None];

        updtrec(j, s, f.row(0), &mut record);
        assert_eq!(record, [Some(0), Some(2), Some(2), Some(0), Some(0), None]);
    }

    #[test]
    fn test_4() {
        // Matlab test
        //
        // clearvars;
        // clear global;
        // j = 4;
        // s = 3;
        // f = [-0.5, -0.6, -0.7, -2.0, -1.0];
        // global record;
        // record = [1,1,3,1,1];
        //
        // updtrec(j,s,f);
        // disp(record)

        let j = 3;
        let s = 3;
        let f = Matrix2xX::<f64>::from_row_slice(&[-0.5, -0.6, -0.7, -2.0, -1.0, -0.5, -0.6, -0.7, -2.0, -1.0]);
        let mut record = [Some(0), Some(0), Some(2), Some(0), Some(0), None];

        updtrec(j, s, f.row(0), &mut record);
        assert_eq!(record, [Some(0), Some(0), Some(3), Some(0), Some(0), None]);
    }

    #[test]
    fn test_5() {
        // Matlab test
        //
        // clearvars;
        // clear global;
        // j = 4;
        // s = 3;
        // f = [-0.5, -0.6, -0.7, -2.0, -1.0];
        // global record;
        // record = [0,0,0,0,0];
        //
        // updtrec(j,s,f);
        // disp(record)

        let j = 3;
        let s = 3;
        let f = Matrix2xX::<f64>::from_row_slice(&[-0.5, -0.6, -0.7, -2.0, -1.0, -0.5, -0.6, -0.7, -2.0, -1.0]);
        let mut record = [None; 6];

        updtrec(j, s, f.row(0), &mut record);
        assert_eq!(record, [None, None, Some(3), None, None, None]);
    }
}