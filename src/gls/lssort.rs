use crate::gls::helpers::clear_and_calc_up_down;

/**
 * Sorts points by increasing alpha values and analyzes the function landscape.
 *
 * This function takes two vectors representing points (alpha values) and their corresponding
 * function values, sorts them by alpha, and performs analysis to identify key characteristics
 * of the function landscape, including:
 * - Best point (minimum function value)
 * - Median function value
 * - Monotonicity detection
 * - Local minima identification
 * - Characteristic length scale calculation
 *
 * The function maintains the relationship between alpha and function values during sorting,
 * and computes directional indicators (up/down) to characterize the function's behavior.
 *
 * # Arguments
 *
 * * `alist` - A mutable reference to a vector of alpha values to be sorted.
 * * `flist` - A mutable reference to a vector of function values corresponding to alpha values.
 *
 * # Returns
 *
 * A tuple containing:
 * * `abest` - The alpha value corresponding to the minimum function value.
 * * `fbest` - The minimum function value found.
 * * `fmed` - The median of all function values.
 * * `up` - Boolean vector indicating where function values are increasing.
 * * `down` - Boolean vector indicating where function values are decreasing.
 * * `monotone` - Boolean indicating if the function is monotonic.
 * * `minima` - Boolean vector marking positions of local minima.
 * * `nmin` - Count of local minima found.
 * * `unitlen` - Characteristic length scale of the function.
 * * `s` - Size of the input lists.
 */
pub fn lssort(
    alist: &mut Vec<f64>,
    flist: &mut Vec<f64>,
) -> (
    f64,         // abest
    f64,         // fbest
    f64,         // fmed
    Vec<bool>,   // up
    Vec<bool>,   // down
    bool,        // monotone
    Vec<bool>,   // minima
    usize,       // nmin
    f64,         // unitlen
    usize        // s
) {
    debug_assert_eq!(alist.len(), flist.len());

    let mut pairs_to_sort: Vec<(f64, f64)> = alist.iter().copied().zip(flist.iter().copied()).collect();
    pairs_to_sort.sort_unstable_by(|(a_i, _), (a_j, _)| a_i.total_cmp(a_j));

    // Unzip the sorted pairs back into the original vectors.
    for (i, &(a, f)) in pairs_to_sort.iter().enumerate() {
        alist[i] = a;
        flist[i] = f;
    }

    let s = alist.len();

    // Calculate up and down vectors
    let mut up = Vec::with_capacity(s - 1);
    let mut down = Vec::with_capacity(s - 1);

    clear_and_calc_up_down(&mut up, &mut down, flist);

    down[s - 2] = flist[s - 1] < flist[s - 2]; // strict < sign

    // Check if sequence is monotone
    let monotone = up.iter().all(|&x| !x) || down.iter().all(|&x| !x);

    // Calculate minima
    let mut minima = Vec::with_capacity(s); // capacity one more than up (or down)
    minima.extend(
        up.iter()
            .chain(std::iter::once(&true))  // Add final true value
            .zip(std::iter::once(&true).chain(down.iter()))  // Create paired iterator
            .map(|(&up, &down)| up && down)
    );

    let nmin = minima.iter().filter(|&&x| x).count();

    let (flist_min_pos, &fbest) = flist.iter().enumerate().min_by(|&(_, f_i), (_, f_j)| f_i.total_cmp(f_j)).unwrap();
    let abest = alist[flist_min_pos];

    // Compute median of flist
    let fmed = {
        let len = flist.len();
        let mid = len / 2;
        let mut partial = flist.clone();

        if len % 2 == 0 {
            // Even length case - only need to partition once
            partial.select_nth_unstable_by(mid - 1, |a, b| a.total_cmp(b));

            // The element at mid-1 is now in the correct place
            // We can find the next element by getting the minimum of the remaining elements
            let (left, right) = partial.split_at_mut(mid);
            let mid_left = left[mid - 1];
            let mid_right = *right.iter().min_by(|a, b| a.total_cmp(b)).unwrap_or(&mid_left);

            (mid_left + mid_right) / 2.0
        } else {
            // Odd length case - only need one partition
            let (_, &mut median, _) = partial.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
            median
        }
    };

    let unitlen = if nmin > 1 {
        alist.iter().enumerate()
            .filter(|&(i, _)| minima[i])
            .fold(None, |acc, (_, x)| {
                let curr_diff = (x - abest).abs();
                if curr_diff > f64::EPSILON {
                    match acc {
                        None => Some(curr_diff),
                        Some(acc_diff) => if curr_diff >= acc_diff { Some(acc_diff) } else { Some(curr_diff) }
                    }
                } else { acc }
            })
            .unwrap_or(0.0)
    } else {
        (abest - alist[0]).max(alist[s - 1] - abest)
    };

    (abest, fbest, fmed, up, down, monotone, minima, nmin, unitlen, s)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0() {
        let mut alist = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let mut flist = vec![10.0, 9.0, 8.0, 7.0, 6.0];
        let (abest, fbest, fmed, up, down, monotone, minima, nmin, unitlen, s) = lssort(&mut alist, &mut flist);

        assert_eq!(alist, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(flist, vec![10.0, 9.0, 8.0, 7.0, 6.0]);
        assert_eq!(abest, 4.0);
        assert_eq!(fbest, 6.0);
        assert_eq!(fmed, 8.0);
        assert_eq!(up, vec![false, false, false, false]);
        assert_eq!(down, vec![true, true, true, true]);
        assert!(monotone); // monotone
        assert_eq!(minima, vec![false, false, false, false, true]);
        assert_eq!(nmin, 1);
        assert_eq!(unitlen, 4.0);
        assert_eq!(s, 5);
    }

    #[test]
    fn test_1() {
        let mut alist = vec![3.0, 1.0, 4.0, 0.0, 2.0];
        let mut flist = vec![8.0, 9.0, 7.0, 10.0, 6.0];
        let (abest, fbest, fmed, up, down, monotone, minima, nmin, unitlen, s) = lssort(&mut alist, &mut flist);

        assert_eq!(alist, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(flist, vec![10.0, 9.0, 6.0, 8.0, 7.0]);
        assert_eq!(abest, 2.0);
        assert_eq!(fbest, 6.0);
        assert_eq!(fmed, 8.0);
        assert_eq!(up, vec![false, false, true, false]);
        assert_eq!(down, vec![true, true, false, true]);
        assert!(!monotone);
        assert_eq!(minima, vec![false, false, true, false, true]);
        assert_eq!(nmin, 2);
        assert_eq!(unitlen, 2.0);
        assert_eq!(s, 5);
    }

    #[test]
    fn test_2() {
        let mut alist = vec![1.0, 1.0, 1.0];
        let mut flist = vec![2.0, 2.0, 2.0];
        let (abest, fbest, fmed, up, down, monotone, minima, nmin, unitlen, s) = lssort(&mut alist, &mut flist);

        assert_eq!(alist, vec![1.0, 1.0, 1.0]);
        assert_eq!(flist, vec![2.0, 2.0, 2.0]);
        assert_eq!(abest, 1.0);
        assert_eq!(fbest, 2.0);
        assert_eq!(fmed, 2.0);
        assert_eq!(up, vec![false, false]);
        assert_eq!(down, vec![true, false]);
        assert!(monotone); // monotone
        assert_eq!(minima, vec![false, false, false]);
        assert_eq!(nmin, 0);
        assert_eq!(unitlen, 0.0);
        assert_eq!(s, 3);
    }

    #[test]
    fn test_3() {
        let mut alist = vec![-5.0, 2.0, -3.0, 9.0, 0.0];
        let mut flist = vec![3.0, 5.0, 1.0, 7.0, 2.0];
        let (abest, fbest, fmed, up, down, monotone, minima, nmin, unitlen, s) = lssort(&mut alist, &mut flist);

        assert_eq!(alist, vec![-5.0, -3.0, 0.0, 2.0, 9.0]);
        assert_eq!(flist, vec![3.0, 1.0, 2.0, 5.0, 7.0]);
        assert_eq!(abest, -3.0);
        assert_eq!(fbest, 1.0);
        assert_eq!(fmed, 3.0);
        assert_eq!(up, vec![false, true, true, true]);
        assert_eq!(down, vec![true, false, false, false]);
        assert!(!monotone);
        assert_eq!(minima, vec![false, true, false, false, false]);
        assert_eq!(nmin, 1);
        assert_eq!(unitlen, 12.0);
        assert_eq!(s, 5);
    }


    #[test]
    fn test_4_unsorted_with_multiple_minima() {
        // Matlab equivalent test
        // alist = [1.0, 5.0, 3.0, 7.0, 9.0];
        // flist = [5.0, 1.0, 3.0, 1.0, 4.0];
        //
        // lssort;
        //
        // disp(alist);
        // disp(flist);
        // disp(abest);
        // disp(fbest);
        // disp(fmed);
        // disp(up);
        // disp(down);
        // disp(monotone);
        // disp(minima);
        // disp(nmin);
        // disp(unitlen);
        // disp(s);

        let mut alist = vec![1.0, 5.0, 3.0, 7.0, 9.0];
        let mut flist = vec![5.0, 1.0, 3.0, 1.0, 4.0];
        let (abest, fbest, fmed, up, down, monotone, minima, nmin, unitlen, s) = lssort(&mut alist, &mut flist);

        assert_eq!(alist, vec![1.0, 3.0, 5.0, 7.0, 9.0]);
        assert_eq!(flist, vec![5.0, 3.0, 1.0, 1.0, 4.0]);
        assert_eq!(abest, 5.0); // First occurrence of minimum
        assert_eq!(fbest, 1.0);
        assert_eq!(fmed, 3.0);
        assert_eq!(up, vec![false, false, false, true]);
        assert_eq!(down, vec![true, true, true, false]);
        assert!(!monotone);
        assert_eq!(minima, vec![false, false, false, true, false]);
        assert_eq!(nmin, 1);
        assert_eq!(unitlen, 4.0);
        assert_eq!(s, 5);
    }

    #[test]
    fn test_5_negative_values() {
        // Matlab equivalent test
        // alist = [-10.0, -5.0, 0.0, 5.0, 10.0];
        // flist = [-3.0, -5.0, -1.0, -4.0, -2.0];
        //
        // lssort;
        //
        // disp(alist);
        // disp(flist);
        // disp(abest);
        // disp(fbest);
        // disp(fmed);
        // disp(up);
        // disp(down);
        // disp(monotone);
        // disp(minima);
        // disp(nmin);
        // disp(unitlen);
        // disp(s);

        let mut alist = vec![-10.0, -5.0, 0.0, 5.0, 10.0];
        let mut flist = vec![-3.0, -5.0, -1.0, -4.0, -2.0];
        let (abest, fbest, fmed, up, down, monotone, minima, nmin, unitlen, s) = lssort(&mut alist, &mut flist);

        assert_eq!(alist, vec![-10.0, -5.0, 0.0, 5.0, 10.0]);
        assert_eq!(flist, vec![-3.0, -5.0, -1.0, -4.0, -2.0]);
        assert_eq!(abest, -5.0);
        assert_eq!(fbest, -5.0);
        assert_eq!(fmed, -3.0);
        assert_eq!(up, vec![false, true, false, true]);
        assert_eq!(down, vec![true, false, true, false]);
        assert!(!monotone);
        assert_eq!(minima, vec![false, true, false, true, false]);
        assert_eq!(nmin, 2);
        assert_eq!(unitlen, 10.0);
        assert_eq!(s, 5);
    }

    #[test]
    fn test_6_single_element() {
        // Matlab equivalent test
        // alist = [42.0, -2.];
        // flist = [99.0, 100.];
        //
        // lssort;
        //
        // disp(alist);
        // disp(flist);
        // disp(abest);
        // disp(fbest);
        // disp(fmed);
        // disp(up);
        // disp(down);
        // disp(monotone);
        // disp(minima);
        // disp(nmin);
        // disp(unitlen);
        // disp(s);

        let mut alist = vec![42.0, -2.];
        let mut flist = vec![99.0, 100.];
        let (abest, fbest, fmed, up, down, monotone, minima, nmin, unitlen, s) = lssort(&mut alist, &mut flist);

        assert_eq!(alist, vec![-2., 42.0]);
        assert_eq!(flist, vec![100., 99.0]);
        assert_eq!(abest, 42.0);
        assert_eq!(fbest, 99.0);
        assert_eq!(fmed, 99.5);
        assert_eq!(up, vec![false]);
        assert_eq!(down, vec![true]);
        assert!(monotone); // Empty vectors are considered monotone
        assert_eq!(minima, vec![false, true]);
        assert_eq!(nmin, 1);
        assert_eq!(unitlen, 44.);
        assert_eq!(s, 2);
    }

    #[test]
    fn test_7_v_shaped() {
        // Matlab equivalent test
        // alist = [1.0, -2.0, 3.0, 4.0, -5.0, 30., -100., 1.];
        // flist = [5.0, -4.0, 1.0, -2.0, 3.0, -40., 10., 6.];
        //
        // lssort;
        //
        // disp(alist);
        // disp(flist);
        // disp(abest);
        // disp(fbest);
        // disp(fmed);
        // disp(up);
        // disp(down);
        // disp(monotone);
        // disp(minima);
        // disp(nmin);
        // disp(unitlen);
        // disp(s);

        let mut alist = vec![1.0, -2.0, 3.0, 4.0, -5.0, 30., -100., 1.];
        let mut flist = vec![5.0, -4.0, 1.0, -2.0, 3.0, -40., 10., 6.];
        let (abest, fbest, fmed, up, down, monotone, minima, nmin, unitlen, s) = lssort(&mut alist, &mut flist);

        assert_eq!(alist, vec![-100., -5., -2., 1., 1., 3., 4., 30.]);
        assert_eq!(flist, vec![10., 3., -4., 5., 6., 1., -2., -40.]);
        assert_eq!(abest, 30.0);
        assert_eq!(fbest, -40.0);
        assert_eq!(fmed, 2.0);
        assert_eq!(up, vec![false, false, true, true, false, false, false]);
        assert_eq!(down, vec![true, true, false, false, true, true, true]);
        assert!(!monotone);
        assert_eq!(minima, vec![false, false, true, false, false, false, false, true]);
        assert_eq!(nmin, 2);
        assert_eq!(unitlen, 32.0);
        assert_eq!(s, 8);
    }

    #[test]
    fn test_8_duplicates() {
        // Matlab equivalent test
        // alist = [1.0, 2.3, 2.2, 2.1, 4.0, -1., 2.];
        // flist = [5.0, 3.0, 3.0, 3.0, 3.0, -10., 1.];
        //
        // lssort;
        //
        // disp(alist);
        // disp(flist);
        // disp(abest);
        // disp(fbest);
        // disp(fmed);
        // disp(up);
        // disp(down);
        // disp(monotone);
        // disp(minima);
        // disp(nmin);
        // disp(unitlen);
        // disp(s);

        let mut alist = vec![1.0, 2.3, 2.2, 2.1, 4.0, -1., 2.];
        let mut flist = vec![5.0, 3.0, 3.0, 3.0, 3.0, -10., 1.];
        let (abest, fbest, fmed, up, down, monotone, minima, nmin, unitlen, s) = lssort(&mut alist, &mut flist);

        assert_eq!(alist, vec![-1., 1., 2., 2.1, 2.2, 2.3, 4.]);
        assert_eq!(flist, vec![-10., 5., 1., 3., 3., 3., 3.]);
        assert_eq!(abest, -1.);
        assert_eq!(fbest, -10.0);
        assert_eq!(fmed, 3.0);
        assert_eq!(up, vec![true, false, true, false, false, false]);
        assert_eq!(down, vec![false, true, false, true, true, false]);
        assert!(!monotone);
        assert_eq!(minima, vec![true, false, true, false, false, false, false]);
        assert_eq!(nmin, 2);
        assert_eq!(unitlen, 3.0);
        assert_eq!(s, 7);
    }

    #[test]
    fn test_9_all_duplicates() {
        // Matlab equivalent test
        // alist = [2.2, 2.2, 2.2, 2.2, 2.2];
        // flist = [5.0, 5.0, 5.0, 5.0, 5.0];
        //
        // lssort;
        //
        // disp(alist);
        // disp(flist);
        // disp(abest);
        // disp(fbest);
        // disp(fmed);
        // disp(up);
        // disp(down);
        // disp(monotone);
        // disp(minima);
        // disp(nmin);
        // disp(unitlen);
        // disp(s);

        let mut alist = vec![2.2, 2.2, 2.2, 2.2, 2.2];
        let mut flist = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let (abest, fbest, fmed, up, down, monotone, minima, nmin, unitlen, s) = lssort(&mut alist, &mut flist);

        assert_eq!(alist, vec![2.2, 2.2, 2.2, 2.2, 2.2]);
        assert_eq!(flist, vec![5.0, 5.0, 5.0, 5.0, 5.0]);
        assert_eq!(abest, 2.2);
        assert_eq!(fbest, 5.0);
        assert_eq!(fmed, 5.0);
        assert_eq!(up, vec![false, false, false, false]);
        assert_eq!(down, vec![true, true, true, false]);
        assert!(monotone);
        assert_eq!(minima, vec![false, false, false, false, false]);
        assert_eq!(nmin, 0);
        assert_eq!(unitlen, 0.0);
        assert_eq!(s, 5);
    }

    #[test]
    fn test_10() {
        // Matlab equivalent test
        // alist = [100., 99., 98., 97., 96.];
        // flist = [10., 9., 8., 7., 6.];
        //
        // lssort;
        //
        // disp(alist);
        // disp(flist);
        // disp(abest);
        // disp(fbest);
        // disp(fmed);
        // disp(up);
        // disp(down);
        // disp(monotone);
        // disp(minima);
        // disp(nmin);
        // disp(unitlen);
        // disp(s);

        let mut alist = vec![100., 99., 98., 97., 96.];
        let mut flist = vec![10., 9., 8., 7., 6.];
        let (abest, fbest, fmed, up, down, monotone, minima, nmin, unitlen, s) = lssort(&mut alist, &mut flist);

        assert_eq!(alist, vec![96., 97., 98., 99., 100.]);
        assert_eq!(flist, vec![6., 7., 8., 9., 10., ]);
        assert_eq!(abest, 96.);
        assert_eq!(fbest, 6.);
        assert_eq!(fmed, 8.);
        assert_eq!(up, vec![true, true, true, true]);
        assert_eq!(down, vec![false, false, false, false]);
        assert!(monotone);
        assert_eq!(minima, vec![true, false, false, false, false]);
        assert_eq!(nmin, 1);
        assert_eq!(unitlen, 4.);
        assert_eq!(s, 5);
    }

    #[test]
    fn test_11() {
        // Matlab equivalent test
        // alist = [-101., 99., -98., 97., -96.];
        // flist = [-10., -9., 8., -7., 6.];
        //
        // lssort;
        //
        // disp(alist);
        // disp(flist);
        // disp(abest);
        // disp(fbest);
        // disp(fmed);
        // disp(up);
        // disp(down);
        // disp(monotone);
        // disp(minima);
        // disp(nmin);
        // disp(unitlen);
        // disp(s);

        let mut alist = vec![-101., 99., -98., 97., -96.];
        let mut flist = vec![-10., -9., 8., -7., 6.];
        let (abest, fbest, fmed, up, down, monotone, minima, nmin, unitlen, s) = lssort(&mut alist, &mut flist);

        assert_eq!(alist, vec![-101., -98., -96., 97., 99.]);
        assert_eq!(flist, vec![-10., 8., 6., -7., -9.]);
        assert_eq!(abest, -101.);
        assert_eq!(fbest, -10.);
        assert_eq!(fmed, -7.);
        assert_eq!(up, vec![true, false, false, false]);
        assert_eq!(down, vec![false, true, true, true]);
        assert!(!monotone);
        assert_eq!(minima, vec![true, false, false, false, true]);
        assert_eq!(nmin, 2);
        assert_eq!(unitlen, 200.);
        assert_eq!(s, 5);
    }
}
