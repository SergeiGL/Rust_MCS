use std::cmp::Ordering;

pub fn lssort(alist: &mut Vec<f64>, flist: &mut Vec<f64>) -> (
    Vec<f64>,    // sorted alist
    Vec<f64>,    // permuted flist
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
    // Create permutation indices for sorting
    let mut indices: Vec<usize> = (0..alist.len()).collect();
    indices.sort_by(|&a, &b| alist[a].partial_cmp(&alist[b]).unwrap_or(Ordering::Equal));

    // Sort alist and apply permutation to flist
    let mut sorted_alist = alist.clone();
    sorted_alist.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let permuted_flist: Vec<f64> = indices.iter().map(|&i| flist[i]).collect();

    let s = sorted_alist.len();

    // Calculate up and down vectors
    let mut up = Vec::with_capacity(s - 1);
    let mut down = Vec::with_capacity(s - 1);

    for i in 0..s - 1 {
        up.push(permuted_flist[i] < permuted_flist[i + 1]);
        down.push(permuted_flist[i + 1] <= permuted_flist[i]);
    }

    if down.len() == 1 {
        down[0] = permuted_flist[s - 1] < permuted_flist[s - 2];
    } else if !down.is_empty() {
        down[s - 2] = permuted_flist[s - 1] < permuted_flist[s - 2];
    }

    // Check if sequence is monotone
    let monotone = up.iter().all(|&x| !x) || down.iter().all(|&x| !x);

    // Calculate minima
    let mut minima = Vec::with_capacity(s);
    let mut extended_up = up.clone();
    extended_up.push(true);
    let mut extended_down = vec![true];
    extended_down.extend(down.iter());

    for i in 0..s {
        minima.push(extended_up[i] && extended_down[i]);
    }

    let nmin = minima.iter().filter(|&&x| x).count();

    // Find best value and its index
    let fbest = *permuted_flist.iter().min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal)).unwrap();
    let i = permuted_flist.iter().position(|&x| x == fbest).unwrap();
    let abest = sorted_alist[i];

    // Calculate median - corrected version
    let mut sorted_flist = permuted_flist.clone();
    sorted_flist.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let fmed = if s % 2 == 0 {
        (sorted_flist[s / 2 - 1] + sorted_flist[s / 2]) / 2.0
    } else {
        sorted_flist[s / 2]
    };

    // Calculate unitlen - corrected version
    let unitlen = if nmin > 1 {
        let mut al: Vec<f64> = (0..minima.len())
            .filter(|&i| minima[i])
            .map(|i| sorted_alist[i])
            .collect();

        if let Some(pos) = al.iter().position(|&x| (x - abest).abs() < f64::EPSILON) {
            al.remove(pos);
        }

        al.iter()
            .map(|&x| (x - abest).abs())
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(0.0)
    } else {
        (sorted_alist[s - 1] - abest).max(abest - sorted_alist[0])
    };

    (sorted_alist, permuted_flist, abest, fbest, fmed, up, down, monotone, minima, nmin, unitlen, s)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sorted_list() {
        let mut alist = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let mut flist = vec![10.0, 9.0, 8.0, 7.0, 6.0];
        let result = lssort(&mut alist, &mut flist);

        assert_eq!(result.0, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.1, vec![10.0, 9.0, 8.0, 7.0, 6.0]);
        assert_eq!(result.2, 4.0);  // abest
        assert_eq!(result.3, 6.0);  // fbest
        assert_eq!(result.4, 8.0);  // fmed
        assert_eq!(result.5, vec![false, false, false, false]);  // up
        assert_eq!(result.6, vec![true, true, true, true]);  // down
        assert!(result.7);  // monotone
        assert_eq!(result.8, vec![false, false, false, false, true]);  // minima
        assert_eq!(result.9, 1);  // nmin
        assert_eq!(result.10, 4.0);  // unitlen
        assert_eq!(result.11, 5);  // s
    }

    #[test]
    fn test_unsorted_list() {
        let mut alist = vec![3.0, 1.0, 4.0, 0.0, 2.0];
        let mut flist = vec![8.0, 9.0, 7.0, 10.0, 6.0];
        let result = lssort(&mut alist, &mut flist);

        assert_eq!(result.0, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.1, vec![10.0, 9.0, 6.0, 8.0, 7.0]);
        assert_eq!(result.2, 2.0);  // abest
        assert_eq!(result.3, 6.0);  // fbest
        assert_eq!(result.4, 8.0);  // fmed
        assert_eq!(result.5, vec![false, false, true, false]);  // up
        assert_eq!(result.6, vec![true, true, false, true]);  // down
        assert!(!result.7);  // monotone
        assert_eq!(result.8, vec![false, false, true, false, true]);  // minima
        assert_eq!(result.9, 2);  // nmin
        assert_eq!(result.10, 2.0);  // unitlen
        assert_eq!(result.11, 5);  // s
    }

    #[test]
    fn test_identical_elements() {
        let mut alist = vec![1.0, 1.0, 1.0];
        let mut flist = vec![2.0, 2.0, 2.0];
        let result = lssort(&mut alist, &mut flist);

        assert_eq!(result.0, vec![1.0, 1.0, 1.0]);
        assert_eq!(result.1, vec![2.0, 2.0, 2.0]);
        assert_eq!(result.2, 1.0);  // abest
        assert_eq!(result.3, 2.0);  // fbest
        assert_eq!(result.4, 2.0);  // fmed
        assert_eq!(result.5, vec![false, false]);  // up
        assert_eq!(result.6, vec![true, false]);  // down
        assert!(result.7);  // monotone
        assert_eq!(result.8, vec![false, false, false]);  // minima
        assert_eq!(result.9, 0);  // nmin
        assert_eq!(result.10, 0.0);  // unitlen
        assert_eq!(result.11, 3);  // s
    }

    #[test]
    fn test_all_min_max_elements() {
        let mut alist = vec![-5.0, 2.0, -3.0, 9.0, 0.0];
        let mut flist = vec![3.0, 5.0, 1.0, 7.0, 2.0];
        let result = lssort(&mut alist, &mut flist);

        assert_eq!(result.0, vec![-5.0, -3.0, 0.0, 2.0, 9.0]);
        assert_eq!(result.1, vec![3.0, 1.0, 2.0, 5.0, 7.0]);
        assert_eq!(result.2, -3.0);  // abest
        assert_eq!(result.3, 1.0);  // fbest
        assert_eq!(result.4, 3.0);  // fmed
        assert_eq!(result.5, vec![false, true, true, true]);  // up
        assert_eq!(result.6, vec![true, false, false, false]);  // down
        assert!(!result.7);  // monotone
        assert_eq!(result.8, vec![false, true, false, false, false]);  // minima
        assert_eq!(result.9, 1);  // nmin
        assert_eq!(result.10, 12.0);  // unitlen
        assert_eq!(result.11, 5);  // s
    }
}