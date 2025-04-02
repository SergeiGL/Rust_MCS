pub fn lsconvex(alist: &Vec<f64>, flist: &Vec<f64>, nmin: usize, s: usize) -> bool {
    if nmin > 1 {
        false
    } else {
        for i in 1..(s - 1) {
            let f12 = (flist[i] - flist[i - 1]) / (alist[i] - alist[i - 1]);
            let f13 = (flist[i] - flist[i + 1]) / (alist[i] - alist[i + 1]);
            let f123 = (f13 - f12) / (alist[i + 1] - alist[i - 1]);

            if f123 < 0.0 {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nmin_rist() {
        let alist = vec![1.0, 2.0, 3.0, 4.0];
        let flist = vec![1.0, 2.0, 3.0, 4.0];
        let s = flist.len();
        let nmin = 2;

        assert_eq!(lsconvex(&alist, &flist, nmin, s), false)
    }

    #[test]
    fn test_nmin_rist_2() {
        let alist = vec![1.0, 2.0, 3.0, 4.0];
        let flist = vec![1.0, 2.0, 3.0, 4.0];
        let s = flist.len();
        let nmin = 1;

        assert_eq!(lsconvex(&alist, &flist, nmin, s), true)
    }

    #[test]
    fn test_3() {
        let alist = vec![1.0, 2.0, 3.0, 4.0, 10.0, 30.0];
        let flist = vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0];
        let s = flist.len();
        let nmin = 0;

        assert_eq!(lsconvex(&alist, &flist, nmin, s), false)
    }

    #[test]
    fn test_4() {
        let alist = vec![1.0, 2.0, 3.0, 4.0, 1.0, -20.0];
        let flist = vec![1.0, 2.0, 3.0, 4.0, 1.0, -2.0];
        let s = flist.len();
        let nmin = 0;

        assert_eq!(lsconvex(&alist, &flist, nmin, s), true)
    }
}