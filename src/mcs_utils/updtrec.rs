#[inline]
pub(crate) fn updtrec(j: usize, s: usize, f: &[f64], record: &mut [usize]) {
    // s: as in Matlab
    // j: -1 from Matlab
    // record: -1 from Matlab in terms of values; record.len(): +1 from Matlab

    debug_assert!(record.len() >= s + 1); // updtrec: VERY CAREFUL record.len() < s"
    if record.len() < s + 1 || record[s - 1] == usize::MAX || f[j] < f[record[s - 1]] {
        record[s - 1] = j;
    }
}


#[cfg(test)]
mod tests {
    use super::*;

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
        let f = vec![-0.5, -0.6, -0.7, -2.0, -4.0];
        let mut record = vec![0, 1, 2, 3, 4, usize::MAX];

        updtrec(j, s, &f, &mut record);
        assert_eq!(record, [0, 1, 3, 3, 4, usize::MAX]);
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
        let f = vec![-0.5, -0.6, -0.7, -2.0, -4.0];
        let mut record = vec![0, 0, 0, 0, 0, usize::MAX];

        updtrec(j, s, &f, &mut record);
        assert_eq!(record, [0, 2, 0, 0, 0, usize::MAX]);
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
        let f = vec![-0.5, -0.6, -0.7, -2.0, -1.0];
        let mut record = vec![0, 0, 2, 0, 0, usize::MAX];

        updtrec(j, s, &f, &mut record);
        assert_eq!(record, [0, 3, 2, 0, 0, usize::MAX]);
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
        let f = vec![-0.5, -0.6, -0.7, -2.0, -4.0];
        let mut record = vec![0, 0, 2, 0, 0, usize::MAX];

        updtrec(j, s, &f, &mut record);
        assert_eq!(record, [0, 2, 2, 0, 0, usize::MAX]);
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
        let f = vec![-0.5, -0.6, -0.7, -2.0, -1.0];
        let mut record = vec![0, 0, 2, 0, 0, usize::MAX];

        updtrec(j, s, &f, &mut record);
        assert_eq!(record, [0, 0, 3, 0, 0, usize::MAX]);
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
        let f = vec![-0.5, -0.6, -0.7, -2.0, -1.0];
        let mut record = vec![usize::MAX; 6];

        updtrec(j, s, &f, &mut record);
        assert_eq!(record, [usize::MAX, usize::MAX, 3, usize::MAX, usize::MAX, usize::MAX]);
    }
}