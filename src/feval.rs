use std::process;

pub fn feval(x: &[f64]) -> f64 {
    hm6(x)
}

const HM6_A: [[f64; 4]; 6] = [
    [10.00, 0.05, 3.00, 17.00],
    [3.00, 10.00, 3.50, 8.00],
    [17.00, 17.00, 1.70, 0.05],
    [3.50, 0.10, 10.00, 10.00],
    [1.70, 8.00, 17.00, 0.10],
    [8.00, 14.00, 8.00, 14.00]
];

const HM6_P: [[f64; 4]; 6] = [
    [0.1312, 0.2329, 0.2348, 0.4047],
    [0.1696, 0.4135, 0.1451, 0.8828],
    [0.5569, 0.8307, 0.3522, 0.8732],
    [0.0124, 0.3736, 0.2883, 0.5743],
    [0.8283, 0.1004, 0.3047, 0.1091],
    [0.5886, 0.9991, 0.6650, 0.0381]
];


fn hm6(x: &[f64]) -> f64 {
    if x.len() != 6 {
        eprintln!("Hartman6 function takes only a vector length 6");
        process::exit(1);
    }

    let c = [1.0, 1.2, 3.0, 3.2];

    let mut d = [0.0; 4];
    for i in 0..4 {
        d[i] = (0..6).map(|j| HM6_A[j][i] * (x[j] - HM6_P[j][i]).powi(2)).sum();
    }
    -c.iter().zip(d.iter()).map(|(ci, di)| ci * (-di).exp()).sum::<f64>()
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hm6_1() {
        let x = vec![0.20601133, 0.20601133, 0.45913871, 0.15954294, 0.29398867, 0.62112999];
        let result = hm6(&x);
        assert_relative_eq!(result, -2.872724123715199, epsilon = 1e-10);
    }

    #[test]
    fn test_hm6_3() {
        let x = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let result = hm6(&x);
        assert_relative_eq!(result, -1.4069105761385299, epsilon = 1e-10);
    }


    #[test]
    fn test_hm6_2() {
        let x = vec![0.1213, 0.2414, 0.1243, 0.345680344, 0.1237595, 0.1354856796];
        let result = hm6(&x);
        assert_relative_eq!(result, -0.16821471453083264, epsilon = 1e-10);
    }
}

