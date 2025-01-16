use crate::minq::ldlrk1::ldlrk1;
use nalgebra::{SMatrix, SVector};


pub fn ldldown<const N: usize>(
    L: &mut SMatrix<f64, N, N>,
    d: &mut SVector<f64, N>,
    j: usize,
) {
    let dj = d[j];

    let mut LKj = L.view_range((j + 1)..N, j).into_owned();

    // Saves LKI allocation; but due to this there is a code duplication
    match j {
        0 => {
            let mut LKK = L.view_range_mut((j + 1)..N, (j + 1)..N);
            let mut dK = d.view_range_mut((j + 1)..N, 0);

            ldlrk1(&mut LKK.as_view_mut(), &mut dK.as_view_mut(), dj, &mut LKj);

            L.fill_row(j, 0.0);
            L.fill_column(j, 0.0)
        }
        _ => {
            let LKI = L.view_range((j + 1)..N, 0..j).into_owned();

            let mut LKK = L.view_range_mut((j + 1)..N, (j + 1)..N);
            let mut dK = d.view_range_mut((j + 1)..N, 0);

            ldlrk1(&mut LKK.as_view_mut(), &mut dK.as_view_mut(), dj, &mut LKj);

            L.fill_row(j, 0.0);
            L.view_range_mut(j.., j).fill(0.0);

            L.view_range_mut((j + 1)..N, 0..j).copy_from(&LKI);
        }
    }

    L[(j, j)] = 1.0;
    d[j] = 1.;
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_mistake() {
        let mut L = SMatrix::<f64, 6, 6>::from_row_slice(&[
            1., 0., 0., 0., 0., 0.,
            0.018635196087984494, 1., 0., 0., 0., 0.,
            -0.00006744289208153283, -0.002280552111200885, 1., 0., 0., 0.,
            0.048606475490027994, -0.07510298414917427, -0.46396921208814274, 1., 0., 0.,
            0.00014656637804684393, 0.004372396897708983, 0.18136372644709786, 0.005054161476720696, 1., 0.,
            0.00934128801419616, -0.0016648858565015735, 0.4674067707226544, -0.017343086953306597, -0.5222650783107669, 1.
        ]);
        let mut d = SVector::<f64, 6>::from_row_slice(&[
            95.7784634229683, 44.981003582969294, 0.30827844591150716, 49.5399548476696, 0.6927734092514332, 79.93996295955547
        ]);
        let j = 2;

        let expected_L = SMatrix::<f64, 6, 6>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.018635196087984494, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.048606475490027994, -0.07510298414917427, 0.0, 1.0, 0.0, 0.0, 0.00014656637804684393, 0.004372396897708983, 0.0, 0.004524467461310853, 1.0, 0.0, 0.00934128801419616, -0.0016648858565015735, 0.0, -0.018667576757453973, -0.477600159127822, 1.0
        ]);
        let expected_d = SVector::<f64, 6>::from_row_slice(&[95.7784634229683, 44.981003582969294, 1.0, 49.60631715637313, 0.703163545389539, 80.03349478791684]);

        ldldown(&mut L, &mut d, j);

        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_0() {
        let mut L = SMatrix::<f64, 3, 3>::from_row_slice(&[-1.0; 9]);
        let mut d = SVector::<f64, 3>::from_row_slice(&[2.0, 3.0, 4.0]);
        let j = 2;

        let expected_L = SMatrix::<f64, 3, 3>::from_row_slice(&[
            -1., -1., -1.,
            -1., -1., -1.,
            0., 0., 1.
        ]);
        let expected_d = SVector::<f64, 3>::from_row_slice(&[2.0, 3.0, 1.0]);

        ldldown(&mut L, &mut d, j);

        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_general_case() {
        let mut L = SMatrix::<f64, 3, 3>::from_row_slice(&[
            1.0, 0.5, 0.3,
            0.0, 1.0, 0.2,
            0.0, 0.0, 1.0
        ]);
        let mut d = SVector::<f64, 3>::from_row_slice(&[2.0, 3.0, 4.0]);
        let j = 1;

        let expected_L = SMatrix::<f64, 3, 3>::from_row_slice(&[
            1.0, 0.5, 0.3,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]);
        let expected_d = SVector::<f64, 3>::from_row_slice(&[2.0, 1.0, 4.0]);

        ldldown(&mut L, &mut d, j);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_min_j() {
        let mut L = SMatrix::<f64, 2, 2>::from_row_slice(&[
            1.0, 0.5,
            0.0, 1.0
        ]);
        let mut d = SVector::<f64, 2>::from_row_slice(&[2.0, 3.0]);
        let j = 0;

        let expected_L = SMatrix::<f64, 2, 2>::from_row_slice(&[
            1.0, 0.0,
            0.0, 1.0
        ]);
        let expected_d = SVector::<f64, 2>::from_row_slice(&[1.0, 3.0]);

        ldldown(&mut L, &mut d, j);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_max_j() {
        let mut L = SMatrix::<f64, 2, 2>::from_row_slice(&[
            1.0, 0.5,
            0.0, 1.0
        ]);
        let mut d = SVector::<f64, 2>::from_row_slice(&[2.0, 3.0]);
        let j = 1;

        let expected_L = SMatrix::<f64, 2, 2>::from_row_slice(&[
            1.0, 0.5,
            0.0, 1.0
        ]);
        let expected_d = SVector::<f64, 2>::from_row_slice(&[2.0, 1.0]);

        ldldown(&mut L, &mut d, j);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_large_values() {
        let mut L = SMatrix::<f64, 2, 2>::from_row_slice(&[
            1.0, 0.5,
            0.0, 1.0
        ]);
        let mut d = SVector::<f64, 2>::from_row_slice(&[1e10, 1e12]);
        let j = 0;

        let expected_L = SMatrix::<f64, 2, 2>::from_row_slice(&[
            1.0, 0.0,
            0.0, 1.0
        ]);
        let expected_d = SVector::<f64, 2>::from_row_slice(&[1.0, 1e12]);

        ldldown(&mut L, &mut d, j);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }

    #[test]
    fn test_fractional_values() {
        let mut L = SMatrix::<f64, 2, 2>::from_row_slice(&[
            1.0, 0.5,
            0.0, 1.0
        ]);
        let mut d = SVector::<f64, 2>::from_row_slice(&[0.2, 0.5]);
        let j = 1;

        let expected_L = SMatrix::<f64, 2, 2>::from_row_slice(&[
            1.0, 0.5,
            0.0, 1.0
        ]);
        let expected_d = SVector::<f64, 2>::from_row_slice(&[0.2, 1.0]);

        ldldown(&mut L, &mut d, j);
        assert_eq!(L, expected_L);
        assert_eq!(d, expected_d);
    }
}