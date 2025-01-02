pub fn lssort(alist: &mut Vec<f64>, flist: &mut Vec<f64>) -> (
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
    // Create a vector of tuples containing (value, index, flist_value)
    let mut paired: Vec<(f64, usize, f64)> = alist.iter()
        .zip(0..alist.len())
        .map(|(a, i)| (*a, i, flist[i]))
        .collect();

    // Sort the paired vector based on the first element (alist values)
    paired.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Update alist and flist in a single pass
    for (i, (a, _, f)) in paired.iter().enumerate() {
        alist[i] = *a;
        flist[i] = *f;
    }

    let s = alist.len();

    // Calculate up and down vectors
    let mut up = Vec::with_capacity(s - 1);
    let mut down = Vec::with_capacity(s - 1);

    for i in 0..s - 1 {
        up.push(flist[i] < flist[i + 1]);
        down.push(flist[i + 1] <= flist[i]);
    }

    if down.len() == 1 {
        down[0] = flist[s - 1] < flist[s - 2];
    } else if !down.is_empty() {
        down[s - 2] = flist[s - 1] < flist[s - 2];
    }

    // Check if sequence is monotone
    let monotone = up.iter().all(|&x| !x) || down.iter().all(|&x| !x);

    // Calculate minima
    let mut minima = Vec::with_capacity(s);
    minima.extend(
        up.iter()
            .chain(std::iter::once(&true))  // Add final true value
            .zip(std::iter::once(&true).chain(down.iter()))  // Create paired iterator
            .take(s)
            .map(|(&up, &down)| up && down)
    );

    let nmin = minima.iter().filter(|&&x| x).count();

    // Find best value and its index
    let (abest, fbest) = flist
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, &value)| (alist[i], value)) // Unpack reference to owned values
        .unwrap();

    // Calculate median
    let fmed = if s % 2 == 0 {
        let mid1 = s / 2 - 1;
        let mid2 = s / 2;

        let (m1, m2) = {
            let mut to_sort = flist.clone();

            let m1 = *to_sort.select_nth_unstable_by(mid1, |a, b| a.partial_cmp(b).unwrap()).1;
            let m2 = *to_sort.select_nth_unstable_by(mid2, |a, b| a.partial_cmp(b).unwrap()).1;

            (m1, m2)
        };
        (m1 + m2) / 2.0
    } else {
        // Find the single middle element for an odd-length vector:
        let mid = s / 2;
        let mut to_sort = flist.clone();
        let m = *to_sort.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap()).1;
        m
    };

    let unitlen = if nmin > 1 {
        let mut al = Vec::with_capacity(minima.len());

        for (i, &is_minimum) in minima.iter().enumerate() {
            if is_minimum {
                al.push(alist[i]);
            }
        }

        if let Some(pos) = al.iter().position(|&x| (x - abest).abs() < f64::EPSILON) {
            // swap_remove is O(1) instead of O(n) for remove
            al.swap_remove(pos);
        }

        al.iter()
            .map(|&x| (x - abest).abs())
            .min_by(|a, b| a.partial_cmp(&b).unwrap()).unwrap()
    } else {
        (alist[s - 1] - abest).max(abest - alist[0])
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
}