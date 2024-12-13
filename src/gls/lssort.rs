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
    // Create permutation indices for sorting
    let mut indices: Vec<usize> = (0..alist.len()).collect();
    indices.sort_by(|&a, &b| alist[a].partial_cmp(&alist[b]).unwrap());

    // Sort alist and apply permutation to flist
    alist.sort_by(|a, b| a.partial_cmp(b).unwrap());

    *flist = indices.iter().map(|&i| flist[i]).collect();

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
    let mut extended_up = up.clone();
    extended_up.push(true);
    let mut extended_down = vec![true];
    extended_down.extend(down.iter());

    for i in 0..s {
        minima.push(extended_up[i] && extended_down[i]);
    }

    let nmin = minima.iter().filter(|&&x| x).count();

    // Find best value and its index
    let fbest = *flist.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let i = flist.iter().position(|&x| x == fbest).unwrap();
    let abest = alist[i];

    // Calculate median - corrected version
    let mut sorted_flist = flist.clone();
    sorted_flist.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let fmed = if s % 2 == 0 {
        (sorted_flist[s / 2 - 1] + sorted_flist[s / 2]) / 2.0
    } else {
        sorted_flist[s / 2]
    };

    // Calculate unitlen - corrected version
    let unitlen = if nmin > 1 {
        let mut al: Vec<f64> = (0..minima.len())
            .filter(|&i| minima[i])
            .map(|i| alist[i])
            .collect();

        if let Some(pos) = al.iter().position(|&x| (x - abest).abs() < f64::EPSILON) {
            al.remove(pos);
        }

        al.iter()
            .map(|&x| (x - abest).abs())
            .min_by(|a, b| a.partial_cmp(b).unwrap())
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