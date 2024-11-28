pub fn updtrec(j: usize, s: usize, f: &[f64], record: &mut Vec<usize>) {
    if record.len() < s { //TODO: strange stuff
        for _ in record.len()..s {
            record.push(0);
        }
        record.push(j);
    } else if record[s] == 0 {
        record[s] = j
    } else if f[j] < f[record[s]] {
        record[s] = j
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_if() {
        let j = 3;
        let s = 5;
        let f = vec![-0.5, -0.6];
        let mut record = vec![1, 1];

        updtrec(j, s, &f, &mut record);
        assert_eq!(record, vec![1, 1, 0, 0, 0, 3]);
    }

    #[test]
    fn test_0() {
        let j = 3;
        let s = 3;
        let f = vec![-0.5, -0.6, -0.7, -2.0, -4.0];
        let mut record = vec![0, 1, 2, 3, 4];

        updtrec(j, s, &f, &mut record);
        assert_eq!(record, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_1() {
        let j = 3;
        let s = 2;
        let f = vec![-0.5, -0.6, -0.7, -2.0, -4.0];
        let mut record = vec![0, 0, 0, 0, 0];

        updtrec(j, s, &f, &mut record);
        assert_eq!(record, vec![0, 0, 3, 0, 0]);
    }

    #[test]
    fn test_2() {
        let j = 4;
        let s = 2;
        let f = vec![-0.5, -0.6, -0.7, -2.0, -1.0];
        let mut record = vec![0, 0, 2, 0, 0];

        updtrec(j, s, &f, &mut record);
        assert_eq!(record, vec![0, 0, 4, 0, 0]);
    }

    #[test]
    fn test_3() {
        let j = 3;
        let s = 2;
        let f = vec![-0.5, -0.6, -0.7, -2.0, -4.0];
        let mut record = vec![0, 0, 2, 0, 0];

        updtrec(j, s, &f, &mut record);
        assert_eq!(record, vec![0, 0, 3, 0, 0]);
    }

    #[test]
    fn test_4() {
        let j = 4;
        let s = 3;
        let f = vec![-0.5, -0.6, -0.7, -2.0, -1.0];
        let mut record = vec![0, 0, 2, 0, 0];

        updtrec(j, s, &f, &mut record);
        assert_eq!(record, vec![0, 0, 2, 4, 0]);
    }

    #[test]
    fn test_5() {
        let j = 1000;
        let s = 5;
        let mut f: Vec<f64> = ((-10..0).collect::<Vec<_>>())
            .into_iter()
            .map(|x| x as f64)
            .chain(std::iter::repeat(0.0).take(1000))
            .collect();

        f.extend(std::iter::repeat(0.0).take(1000));
        let mut record = vec![0; 10];

        updtrec(j, s, &f, &mut record);
        assert_eq!(record, vec![0, 0, 0, 0, 0, 1000, 0, 0, 0, 0]);
    }
}