pub fn strtsw(smax: usize, level: &Vec<usize>, f: &Vec<f64>, nboxes: usize) -> (usize, Vec<usize>) {
    let mut record = vec![0; smax];
    let mut s = smax;
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

    #[test]
    fn test_single_element() {
        let smax = 10;
        let level = vec![5];
        let f = vec![0.5];
        let nboxes = 0;

        let (result_s, result_record) = strtsw(smax, &level, &f, nboxes);
        assert_eq!((result_s, result_record), (5, vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
    }

    #[test]
    fn test_all_zero_level() {
        let smax = 5;
        let level = vec![0, 0, 0, 0, 0];
        let f = vec![0.0, 0.1, 0.2, 0.3, 0.4];
        let nboxes = 4;

        let (result_s, result_record) = strtsw(smax, &level, &f, nboxes);
        assert_eq!((result_s, result_record), (5, vec![0, 0, 0, 0, 0]));
    }

    #[test]
    fn test_f_values_order_affect() {
        let smax = 6;
        let level = vec![1, 2, 3, 3, 0, 1];
        let f = vec![0.5, 0.2, 0.4, 0.1, 0.0, 0.6];
        let nboxes = 5;

        let (result_s, result_record) = strtsw(smax, &level, &f, nboxes);
        assert_eq!((result_s, result_record), (1, vec![0, 5, 1, 3, 0, 0]));
    }

    #[test]
    fn test_varying_levels() {
        let smax = 5;
        let level = vec![0, 4, 3, 2, 1];
        let f = vec![-0.5, 0.25, -0.3, 0.4, 0.1];
        let nboxes = 4;

        let (result_s, result_record) = strtsw(smax, &level, &f, nboxes);
        assert_eq!(result_s, 1);
        assert_eq!(result_record, vec![0, 4, 3, 2, 1]);
    }

    #[test]
    fn test_negative_f_values() {
        let smax = 7;
        let level = vec![1, 3, 3, 2];
        let f = vec![-0.5, -0.9, -0.2, -1.0];
        let nboxes = 3;

        let (result_s, result_record) = strtsw(smax, &level, &f, nboxes);
        assert_eq!((result_s, result_record), (1, vec![0, 0, 3, 1, 0, 0, 0]));
    }
}
