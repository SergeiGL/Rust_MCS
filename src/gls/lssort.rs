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
    // Pair and sort alist and flist
    let mut paired: Vec<(f64, f64)> = alist.iter().copied().zip(flist.iter().copied()).collect();
    paired.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Update alist and flist based on the sorted order
    for (i, (a, f)) in paired.iter().enumerate() {
        alist[i] = *a;
        flist[i] = *f;
    }

    let s = alist.len();

    // Calculate up and down vectors
    let mut up = Vec::with_capacity(s);
    let mut down = Vec::with_capacity(s);

    for i in 0..s - 1 {
        let (curr, next) = (flist[i], flist[i + 1]);
        up.push(curr < next);
        down.push(next <= curr);
    }

    // Fix down last element if necessary
    if s > 1 {
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
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, &value)| (alist[i], value)) // Unpack reference to owned values
        .unwrap();

    // Compute median of flist
    let fmed = {
        let mut to_sort = flist.clone();
        let mid = s / 2;
        if s % 2 == 0 {
            let mid1 = *to_sort.select_nth_unstable_by(mid - 1, |a, b| a.total_cmp(b)).1;
            let mid2 = *to_sort.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1;
            (mid1 + mid2) / 2.0
        } else {
            *to_sort.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1
        }
    };

    let unitlen = if nmin > 1 {
        minima.iter()
            .enumerate()
            .filter_map(|(i, &is_min)| if is_min { Some(alist[i]) } else { None })
            .filter(|&x| (x - abest).abs() >= f64::EPSILON)
            .map(|x| (x - abest).abs())
            .min_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.0)
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