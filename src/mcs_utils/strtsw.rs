use nalgebra::{Dyn, MatrixView, U1, U2};

pub fn strtsw<const SMAX: usize>(
    level: &Vec<usize>, // as in Matlab
    f: MatrixView<f64, U1, Dyn, U1, U2>,
    nboxes: usize,     // -1 from Matlab
) -> (
    usize,          // s // as in Matlab
    [usize; SMAX]   // record // -1 from Matlab
) {
    // SMAX: -1 from Matlab
    let mut record = [0; SMAX];
    let mut s = SMAX + 1;

    for (j, &level_val) in level.iter().take(nboxes + 1).enumerate() {
        if level_val != 0 {
            if level_val < s {
                s = level_val;
            };

            if record[level_val] == 0 || f[j] < f[record[level_val]] {
                record[level_val] = j;
            }
        }
    }

    (s, record)
}

#[cfg(test)]
mod tests {
    // use super::*;
    // use nalgebra::Matrix2xX;
    //
    // #[test]
    // fn test_Matlab() {
    //     // Equivalent Matlab test
    //     //
    //     // smax = 10; % for Rust -1
    //     // level = [0, 1, 2, 3, 4]; % same in Rust
    //     // f = [0.5, 0.1, -1., -10., 3., 0.,]; % x2 in Rust as only forst column will be passed
    //     //
    //     // global nboxes record;
    //     // nboxes = 5; % -1 for Rust
    //     //
    //     // s = strtsw(smax, level, f);
    //     // disp(s);
    //     // disp(record); % nboxes will not change
    //
    //     const SMAX: usize = 9;
    //     let level = vec![0, 1, 2, 3, 4];
    //     let f = Matrix2xX::from_row_slice(&[0.5, 0.1, -1., -10., 3., 0., 0.5, 0.1, -1., -10., 3., 0.]);
    //     let nboxes = 4;
    //
    //     let (s, record) = strtsw::<SMAX>(&level, f.row(0), nboxes);
    //     assert_eq!((s, record), (1, [
    //              2,
    //  3,
    //  4,
    //  5,
    //  0
    //  0
    //  0
    //  0
    //  0
    //
    //     ]));
    // }
}
