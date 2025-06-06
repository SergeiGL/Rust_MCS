pub(crate) fn strtsw(
    record: &mut Vec<usize>,  // -1 from Matlab; 0 -> usize::MAX
    level: &[usize], // as in Matlab
    f: &[f64],
    nboxes: usize, // as from Matlab
) ->
    usize  // s
{
    // SMAX: as in Matlab
    // record: -1 from Matlab; 0 -> usize::MAX
    // level: as in Matlab
    // nboxes: as from Matlab
    // s: as in Matlab

    // Not SMAX-1 as it'll be hard for generic_const_exprs. Will account for +1 len() later
    record.fill(usize::MAX);
    debug_assert_eq!(record.len(), record.capacity());
    let mut s = record.len();

    // Matlab: 1:nboxes takes nboxes elements
    for j in 0..nboxes {
        let level_val = level[j];
        if level_val != 0 {
            s = s.min(level_val); // both s and level are as in Matlab

            let record_el = &mut record[level_val - 1];
            if *record_el == usize::MAX || f[j] < f[*record_el] {
                *record_el = j;
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let f = vec![0.5, 0.1, -1., -10., 3., 0.];
        let nboxes = 5;
        let mut record = vec![1; SMAX];

        let s = strtsw(&mut record, &level, &f, nboxes);
        assert_eq!((s, record), (1, vec![1, 2, 3, 4, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX]));
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
        let f = vec![];
        let nboxes = 0;
        let mut record = vec![1; SMAX];

        let s = strtsw(&mut record, &level, &f, nboxes);
        assert_eq!((s, record), (10, vec![usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX]));
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
        let f = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let nboxes = 5;
        let mut record = vec![1; SMAX];

        let s = strtsw(&mut record, &level, &f, nboxes);
        assert_eq!((s, record), (10, vec![usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX]));
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
        let f = vec![0.0, 4.0, -2.0, 1.0, 3.0];
        let nboxes = 5;
        let mut record = vec![1; SMAX];

        let s = strtsw(&mut record, &level, &f, nboxes);
        assert_eq!((s, record), (3, vec![usize::MAX, usize::MAX, 2, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX]));
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
        let f = vec![0.5, 0.1, -1.0, -10.0, 3.0, 0.1, -5.0, 2.0];
        let nboxes = 4;
        let mut record = vec![1; SMAX];

        let s = strtsw(&mut record, &level, &f, nboxes);
        assert_eq!((s, record), (1, vec![1, 2, 3, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX]));
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
        let f = vec![0.5, 0.1, -1.0, -10.0, 3.0, 0.1];
        let nboxes = 4;
        let mut record = vec![1; SMAX];

        let s = strtsw(&mut record, &level, &f, nboxes);
        // Note: levels exceeding SMAX will be ignored for record purposes
        assert_eq!((s, record), (1, vec![1, 2, 3, usize::MAX, usize::MAX]));
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
        let f = vec![0.0, -10.0, -5.0, -20.0, -1.0];
        let nboxes = 5;
        let mut record = vec![1; SMAX];

        let s = strtsw(&mut record, &level, &f, nboxes);
        assert_eq!((s, record), (2, vec![usize::MAX, 3, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX]));
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
        let f = vec![0.0, 3.1, 0.0, -1.5, 2.8, -2.7, 0.5];
        let nboxes = 7;
        let mut record = vec![13434; SMAX];

        let s = strtsw(&mut record, &level, &f, nboxes);
        assert_eq!((s, record), (1, vec![6, 5, usize::MAX, 4, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX]));
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
        let f = vec![0.0, 3.1, 0.0, 0.0, 0.0, 0.0];
        let nboxes = 6;
        let mut record = vec![13434; SMAX];

        let s = strtsw(&mut record, &level, &f, nboxes);
        assert_eq!((s, record), (5, vec![usize::MAX, usize::MAX, usize::MAX, usize::MAX, 1, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX]));
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
        let f = vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0];
        let nboxes = 4;
        let mut record = vec![13434; SMAX];

        let s = strtsw(&mut record, &level, &f, nboxes);

        let mut expected_record = vec![usize::MAX; SMAX];
        expected_record[24] = 2;  // level 25, index 2
        expected_record[49] = 1;  // level 50, index 1
        expected_record[74] = 3;  // level 75, index 3

        assert_eq!(s, 25);  // Minimum non-zero level is 25
        assert_eq!(record, expected_record);
    }
}