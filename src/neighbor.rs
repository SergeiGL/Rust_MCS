pub fn neighbor(x: &[f64], delta: &[f64], u: &[f64], v: &[f64]) -> (
    Vec<f64>, //x1
    Vec<f64>  //x2
) {
    let mut x1 = x.iter().enumerate()
        .map(|(i, &xi)| u[i].max(xi - delta[i]))
        .collect::<Vec<f64>>();

    let mut x2 = x.iter().enumerate()
        .map(|(i, &xi)| v[i].min(xi + delta[i]))
        .collect::<Vec<f64>>();

    for (i, &xi) in x.iter().enumerate() {
        if xi == u[i] {
            x1[i] = xi + 2.0 * delta[i];
        }
        if xi == v[i] {
            x2[i] = xi - 2.0 * delta[i];
        }
    }
    (x1, x2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0() {
        let x = vec![0.2, 0.15, 0.47, 0.27, 0.31, 0.65];
        let delta = vec![0.001, 0.001, 0.002, 0.001, 0.001, 0.002];
        let u = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let (x1, x2) = neighbor(&x, &delta, &u, &v);
        assert_eq!((x1, x2), (vec![0.199, 0.149, 0.46799999999999997, 0.269, 0.309, 0.648], vec![0.201, 0.151, 0.472, 0.271, 0.311, 0.652]));
    }

    #[test]
    fn test_1() {
        let x = vec![0.0, 0.1, 0.0, 0.2, 0.0, 0.6];
        let delta = vec![0.001, 0.001, 0.002, 0.001, 0.001, 0.002];
        let u = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let (x1, x2) = neighbor(&x, &delta, &u, &v);
        assert_eq!((x1, x2), (vec![0.002, 0.099, 0.004, 0.199, 0.002, 0.598], vec![0.001, 0.101, 0.002, 0.201, 0.001, 0.602]));
    }

    #[test]
    fn test_2() {
        let x = vec![1.0, 0.15, 0.55, 1.0, 0.3, 1.0];
        let delta = vec![0.001, 0.001, 0.002, 0.001, 0.001, 0.002];
        let u = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let (x1, x2) = neighbor(&x, &delta, &u, &v);
        assert_eq!((x1, x2), (vec![0.999, 0.149, 0.548, 0.999, 0.299, 0.998], vec![0.998, 0.151, 0.552, 0.998, 0.301, 0.996]));
    }

    #[test]
    fn test_3() {
        let x = vec![0.0, 0.0, 0.5, 0.0, 0.0, 0.0];
        let delta = vec![0.001, 0.001, 0.002, 0.001, 0.001, 0.002];
        let u = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0];

        let (x1, x2) = neighbor(&x, &delta, &u, &v);
        assert_eq!((x1, x2), (vec![0.002, 0.002, 0.498, 0.002, 0.002, 0.004], vec![-0.002, -0.002, 0.502, -0.002, -0.002, -0.004]));
    }

    #[test]
    fn test_4() {
        let x = vec![0.0001, 0.9999, 0.5001, 0.2001, 0.2999, 0.6501];
        let delta = vec![0.001, 0.001, 0.002, 0.001, 0.001, 0.002];
        let u = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let (x1, x2) = neighbor(&x, &delta, &u, &v);
        assert_eq!((x1, x2), (
            vec![0.0, 0.9989, 0.4981, 0.1991, 0.2989, 0.6481],
            vec![0.0011, 1.0, 0.5021, 0.2011, 0.3009, 0.6521]
        ));
    }

    #[test]
    fn test_5() {
        let x: Vec<f64> = vec![];
        let delta: Vec<f64> = vec![];
        let u: Vec<f64> = vec![];
        let v: Vec<f64> = vec![];

        let (x1, x2) = neighbor(&x, &delta, &u, &v);
        assert_eq!(x1, vec![]);
        assert_eq!(x2, vec![]);
    }
}
