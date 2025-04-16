use nalgebra::{Dyn, MatrixView, U1, U2};

#[inline]
pub(crate) fn strtsw<const SMAX: usize>(
    record: &mut [Option<usize>; SMAX],  // -1 from Matlab
    level: &Vec<usize>, // as in Matlab
    f: MatrixView<f64, U1, Dyn, U1, U2>,
    nboxes: usize, // as from Matlab
) ->
    usize  // s
{
    // SMAX: as in Matlab
    // record: -1 from Matlab (0 -> None)
    // level: as in Matlab
    // nboxes: as from Matlab
    // s: as in Matlab

    // Not SMAX-1 as it'll be hard for generic_const_exprs. Will account for +1 len() later
    record.fill(None);
    let mut s = SMAX;

    // Matlab: 1:nboxes takes nboxes elements
    for j in 0..nboxes {
        let level_val = level[j];
        if level_val != 0 {
            if level_val < s { // both s and level are as in Matlab
                s = level_val;
            };

            let record_el = &mut record[level_val - 1];
            *record_el = Some(match *record_el {
                None => j,
                Some(val) if f[j] < f[val] => j,
                Some(val) => val,
            });
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix2xX;

    #[test]
    fn test_Matlab() {
        // Equivalent Matlab test
        //
        // clearvars;
        // clear global;
        // smax = 10; % as in Rust
        // level = [0, 1, 2, 3, 4, 5]; % same in Rust
        // f = [0.5, 0.1, -1., -10., 3., 0.,]; % x2 in Rust as only first column will be passed
        //
        // global nboxes record;
        // nboxes = 5; % as in Rust
        //
        // s = strtsw(smax, level, f);
        // disp(s);
        // disp(record); % nboxes will not change

        const SMAX: usize = 10;
        let level = vec![0, 1, 2, 3, 4, 5];
        let f = Matrix2xX::from_row_slice(&[0.5, 0.1, -1., -10., 3., 0., 0.5, 0.1, -1., -10., 3., 0.]);
        let nboxes = 5;
        let mut record = [Some(1); SMAX];

        let s = strtsw::<SMAX>(&mut record, &level, f.row(0), nboxes);
        assert_eq!((s, record), (1, [Some(1), Some(2), Some(3), Some(4), None, None, None, None, None, None, ]));
    }

    #[test]
    fn test_empty_level_array() {
        // Equivalent Matlab test
        //
        // clearvars;
        // clear global;
        // smax = 10; % as in Rust
        // level = []; % empty array
        // f = []; % empty array
        //
        // global nboxes record;
        // nboxes = 0;
        //
        // s = strtsw(smax, level, f);
        // disp(s);
        // disp(record);

        const SMAX: usize = 10;
        let level = vec![];
        let f = Matrix2xX::from_row_slice(&[]);
        let nboxes = 0;
        let mut record = [Some(1); SMAX];

        let s = strtsw::<SMAX>(&mut record, &level, f.row(0), nboxes);
        assert_eq!((s, record), (10, [None, None, None, None, None, None, None, None, None, None]));
    }

    #[test]
    fn test_all_zeros_level() {
        // Equivalent Matlab test
        //
        // clearvars;
        // clear global;
        // smax = 10; % as in Rust
        // level = [0, 0, 0, 0, 0]; % all zeros
        // f = [1.0, 2.0, 3.0, 4.0, 5.0]; % values don't matter when all levels are 0
        //
        // global nboxes record;
        // nboxes = 5;
        //
        // s = strtsw(smax, level, f);
        // disp(s);
        // disp(record);

        const SMAX: usize = 10;
        let level = vec![0, 0, 0, 0, 0];
        let f = Matrix2xX::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let nboxes = 5;
        let mut record = [Some(1); SMAX];

        let s = strtsw::<SMAX>(&mut record, &level, f.row(0), nboxes);
        assert_eq!((s, record), (10, [None, None, None, None, None, None, None, None, None, None]));
    }

    #[test]
    fn test_same_level_different_f() {
        // Equivalent Matlab test
        //
        // clearvars;
        // clear global;
        // smax = 10; % as in Rust
        // level = [0, 3, 3, 3, 3]; % multiple entries with same level
        // f = [0.0, 4.0, -2.0, 1.0, 3.0]; % different values
        //
        // global nboxes record;
        // nboxes = 5;
        //
        // s = strtsw(smax, level, f);
        // disp(s);
        // disp(record);

        const SMAX: usize = 10;
        let level = vec![0, 3, 3, 3, 3];
        let f = Matrix2xX::from_row_slice(&[0.0, 4.0, -2.0, 1.0, 3.0, 0.0, 4.0, -2.0, 1.0, 3.0]);
        let nboxes = 5;
        let mut record = [Some(1); SMAX];

        let s = strtsw::<SMAX>(&mut record, &level, f.row(0), nboxes);
        assert_eq!((s, record), (3, [None, None, Some(2), None, None, None, None, None, None, None]));
    }

    #[test]
    fn test_nboxes_less_than_array_length() {
        // Equivalent Matlab test
        //
        // clearvars;
        // clear global;
        // smax = 10; % as in Rust
        // level = [0, 1, 2, 3, 4, 5, 6, 7]; % longer array
        // f = [0.5, 0.1, -1.0, -10.0, 3.0, 0.1, -5.0, 2.0]; % longer array
        //
        // global nboxes record;
        // nboxes = 4;
        //
        // s = strtsw(smax, level, f);
        // disp(s);
        // disp(record);

        const SMAX: usize = 10;
        let level = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let f = Matrix2xX::from_row_slice(&[0.5, 0.1, -1.0, -10.0, 3.0, 0.1, -5.0, 2.0,
            0.5, 0.1, -1.0, -10.0, 3.0, 0.1, -5.0, 2.0]);
        let nboxes = 4;
        let mut record = [Some(1); SMAX];

        let s = strtsw::<SMAX>(&mut record, &level, f.row(0), nboxes);
        assert_eq!((s, record), (1, [Some(1), Some(2), Some(3), None, None, None, None, None, None, None]));
    }

    #[test]
    fn test_level_exceeds_smax() {
        // Equivalent Matlab test
        //
        // clearvars;
        // clear global;
        // smax = 5; % as in Rust
        // level = [0, 1, 2, 3, 10, 6]; % level exceeds smax
        // f = [0.5, 0.1, -1.0, -10.0, 3.0, 0.1];
        //
        // global nboxes record;
        // nboxes = 4;
        //
        // s = strtsw(smax, level, f);
        // disp(s);
        // disp(record);

        const SMAX: usize = 5;
        let level = vec![0, 1, 2, 3, 10, 6];
        let f = Matrix2xX::from_row_slice(&[0.5, 0.1, -1.0, -10.0, 3.0, 0.1,
            0.5, 0.1, -1.0, -10.0, 3.0, 0.1]);
        let nboxes = 4;
        let mut record = [Some(1); SMAX];

        let s = strtsw::<SMAX>(&mut record, &level, f.row(0), nboxes);
        // Note: levels exceeding SMAX will be ignored for record purposes
        assert_eq!((s, record), (1, [Some(1), Some(2), Some(3), None, None]));
    }

    #[test]
    fn test_negative_f_values() {
        // Equivalent Matlab test
        //
        // clearvars;
        // clear global;
        // smax = 10; % as in Rust
        // level = [0, 2, 2, 2, 2]; % same levels
        // f = [0.0, -10.0, -5.0, -20.0, -1.0]; % all negative values
        //
        // global nboxes record;
        // nboxes = 5;
        //
        // s = strtsw(smax, level, f);
        // disp(s);
        // disp(record);

        const SMAX: usize = 10;
        let level = vec![0, 2, 2, 2, 2];
        let f = Matrix2xX::from_row_slice(&[0.0, -10.0, -5.0, -20.0, -1.0,
            0.0, -10.0, -5.0, -20.0, -1.0]);
        let nboxes = 5;
        let mut record = [Some(1); SMAX];

        let s = strtsw::<SMAX>(&mut record, &level, f.row(0), nboxes);
        assert_eq!((s, record), (2, [None, Some(3), None, None, None, None, None, None, None, None]));
    }

    #[test]
    fn test_mixed_levels_and_values() {
        // Equivalent Matlab test
        //
        // clearvars;
        // clear global;
        // smax = 10; % as in Rust
        // level = [0, 4, 0, 2, 4, 2, 1]; % mixed levels
        // f = [0.0, 3.1, 0.0, -1.5, 2.8, -2.7, 0.5]; % mixed values
        //
        // global nboxes record;
        // nboxes = 7;
        //
        // s = strtsw(smax, level, f);
        // disp(s);
        // disp(record);

        const SMAX: usize = 10;
        let level = vec![0, 4, 0, 2, 4, 2, 1];
        let f = Matrix2xX::from_row_slice(&[0.0, 3.1, 0.0, -1.5, 2.8, -2.7, 0.5,
            0.0, 3.1, 0.0, -1.5, 2.8, -2.7, 0.5]);
        let nboxes = 7;
        let mut record = [Some(13434); SMAX];

        let s = strtsw::<SMAX>(&mut record, &level, f.row(0), nboxes);
        assert_eq!((s, record), (1, [Some(6), Some(5), None, Some(4), None, None, None, None, None, None]));
    }

    #[test]
    fn test_single_level() {
        // Equivalent Matlab test
        //
        // clearvars;
        // clear global;
        // smax = 10; % as in Rust
        // level = [0, 5, 0, 0, 0, 0]; % only one non-zero level
        // f = [0.0, 3.1, 0.0, 0.0, 0.0, 0.0]; % values don't matter except for level 5
        //
        // global nboxes record;
        // nboxes = 6;
        //
        // s = strtsw(smax, level, f);
        // disp(s);
        // disp(record);

        const SMAX: usize = 10;
        let level = vec![0, 5, 0, 0, 0, 0];
        let f = Matrix2xX::from_row_slice(&[0.0, 3.1, 0.0, 0.0, 0.0, 0.0,
            0.0, 3.1, 0.0, 0.0, 0.0, 0.0]);
        let nboxes = 6;
        let mut record = [Some(13434); SMAX];

        let s = strtsw::<SMAX>(&mut record, &level, f.row(0), nboxes);
        assert_eq!((s, record), (5, [None, None, None, None, Some(1), None, None, None, None, None]));
    }

    #[test]
    fn test_large_smax() {
        // Equivalent Matlab test
        //
        // clearvars;
        // clear global;
        // smax = 76; % as in Rust
        // level = [0, 50, 25, 75]; % large level values
        // f = [0.0, 1.0, 2.0, 3.0];
        //
        // global nboxes record;
        // nboxes = 4;
        //
        // s = strtsw(smax, level, f);
        // disp(s);
        // disp(record);

        const SMAX: usize = 76;
        let level = vec![0, 50, 25, 75];
        let f = Matrix2xX::from_row_slice(&[0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);
        let nboxes = 4;
        let mut record = [Some(13434); SMAX];

        let s = strtsw::<SMAX>(&mut record, &level, f.row(0), nboxes);

        let mut expected_record = [None; SMAX];
        expected_record[24] = Some(2);  // level 25, index 2
        expected_record[49] = Some(1);  // level 50, index 1
        expected_record[74] = Some(3);  // level 75, index 3

        assert_eq!(s, 25);  // Minimum non-zero level is 25
        assert_eq!(record, expected_record);
    }
}