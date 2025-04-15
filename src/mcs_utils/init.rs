use nalgebra::{Matrix3xX, SMatrix, SVector};

pub(crate) fn init<const N: usize>(
    func: fn(&SVector<f64, N>) -> f64,
    x0: &SMatrix<f64, N, 3>,
) -> (
    Matrix3xX<f64>, // f0 = [[f64; N];3]
    [usize; N],    //  istar // -1 from Matlab
    usize          //  ncall
) {
    let mut ncall = 0_usize;

    let mut x: SVector<f64, N> = x0.column(1).into_owned();

    let mut f1 = func(&x);
    ncall += 1;
    let mut f0 = Matrix3xX::<f64>::repeat(N, 0.0); // L[0] is always = 3
    f0[(1, 0)] = f1;

    let mut istar = [1_usize; N];
    for i in 0..N {
        for j in 0..3 {
            if j == 1 {
                if i != 0 { f0[(j, i)] = f0[(istar[i - 1], i - 1)]; }
            } else {
                x[i] = x0[(i, j)];
                f0[(j, i)] = func(&x);
                ncall += 1;
                if f0[(j, i)] < f1 {
                    f1 = f0[(j, i)];
                    istar[i] = j;
                }
            }
        }

        // Update x[i] to the best found value
        x[i] = x0[(i, istar[i])];
    }

    (f0, istar, ncall)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::hm6;

    #[test]
    fn init_test_matlab_0() {
        // Matlab equivalent test
        //
        // fcn="hm6";
        // data = "hm6";
        // x0 = [
        //     [1.0, 2.0, 3.0];
        //     [-1.1, -1.0, -2.0];
        //     [1.4, 1.3, 1.1];
        //     [11.0, 31.0, 41.0];
        //     [-12.0, 0.23, -101.0];
        //     [4.0, 3.0, -5.0];
        // ];
        // l = [2, 2, 2, 2, 2, 2]; % always full of 2
        // L = [3,3,3,3,3,3]; % always full of 3
        // n = 6; % Always 6 as Maxtix should have 6 rows to fit hm6() function
        //
        // [f0_out,istar_out,ncall_out] = init(fcn,data,x0,l,L,n);
        //
        // format long g;
        //
        // disp(f0_out);
        // disp(istar_out);
        // disp(ncall_out);

        let x0 = SMatrix::<f64, 6, 3>::from_row_slice(&[
            1.0, 2.0, 3.0,
            -1.1, -1.0, -2.0,
            1.4, 1.3, 1.1,
            11.0, 31.0, 41.0,
            -12.0, 0.23, -101.0,
            4.0, 3.0, -5.0,
        ]);

        let (f0, istar, ncall) = init(hm6, &x0);

        let expected_f0 = Matrix3xX::<f64>::from_column_slice(&[-4.233391131618164e-76, -3.7295715378230396e-76, -2.973035122710528e-76, -2.2672971762723364e-77, -4.233391131618164e-76, -1.014468515950181e-92, -7.24251448627543e-77, -4.233391131618164e-76, -5.215621541933745e-75, -3.540241400533224e-39, -5.215621541933745e-75, -5.923906705182671e-106, -0.0, -3.540241400533224e-39, -0.0, -1.372381712728418e-69, -3.540241400533224e-39, -1.1821123413448022e-233]);
        let expected_istar = [0, 1, 2, 0, 1, 1]; // == [1, 2, 3, 1, 2, 2]

        assert_eq!(f0, expected_f0);
        assert_eq!(istar, expected_istar);
        assert_eq!(ncall, 13);
    }

    #[test]
    fn init_test_base_case() {
        let x0 = SMatrix::from_row_slice(&[
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
        ]);

        let (f0, istar, ncall) = init(hm6, &x0);

        let expected_f0 = Matrix3xX::<f64>::from_row_slice(&[-0.6232314675898112, -0.8603825505022568, -0.5152637963551447, -0.9883412202327723, -0.3791401895175917, -0.05313547352279423, -0.5053149917022333, -0.6232314675898112, -0.8603825505022568, -0.8603825505022568, -0.9883412202327723, -0.9883412202327723, -0.08793206431638863, -0.09355142624190396, -0.3213921800858628, -0.025134295008083094, -0.6527318901582629, -0.37750674173452126]);
        let expected_istar = [0, 0, 1, 0, 1, 1];

        assert_eq!(f0, expected_f0);
        assert_eq!(istar, expected_istar);
        assert_eq!(ncall, 13);
    }

    #[test]
    fn init_test_same_x0_values() {
        let x0 = SMatrix::from_row_slice(&[
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ]);

        let (f0, istar, ncall) = init(hm6, &x0);

        let expected_f0 = Matrix3xX::<f64>::from_row_slice(&[-3.408539273427753e-05; 18]);
        let expected_istar = [1, 1, 1, 1, 1, 1];

        assert_eq!(f0, expected_f0);
        assert_eq!(istar, expected_istar);
        assert_eq!(ncall, 13);
    }
}

