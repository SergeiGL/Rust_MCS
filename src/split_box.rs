pub fn split_box<const N: usize>(u: &[f64; N], v: &[f64; N], n_boxes_to_create: usize) ->
(
    Vec<[f64; N]>,  // u_split
    Vec<[f64; N]>,  // v_split
    usize,          // n_boxes_created
) {
    fn find_max_difference_location<const N: usize>(
        data: &[Vec<f64>; N]
    ) -> (
        usize,  // array_idx
        usize,  // vec_idx
        f64     // global_max
    ) {
        let mut global_max_diff = f64::NEG_INFINITY;
        let mut max_array_idx = 0;
        let mut max_vec_idx = 0;

        for (array_idx, vec) in data.iter().enumerate() {
            let (vec_indx, local_max_diff) = vec.windows(2)
                .enumerate()
                .max_by(|(_i, window1), (_j, window2)|
                    (window1[1] - window1[0]).partial_cmp(&(window2[1] - window2[0])).unwrap()
                )
                .map(|(indx, max_arr)| (indx, max_arr[1] - max_arr[0]))
                .unwrap();

            if let Some(std::cmp::Ordering::Greater) = local_max_diff.partial_cmp(&global_max_diff) {
                global_max_diff = local_max_diff;
                max_array_idx = array_idx;
                max_vec_idx = vec_indx;
            }
        }
        (max_array_idx, max_vec_idx, global_max_diff)
    }

    fn generate_splits<const N: usize>(
        split_array: [Vec<f64>; N],
    ) -> (Vec<[f64; N]>, Vec<[f64; N]>) {
        let interval_lists: Vec<Vec<(f64, f64)>> = split_array
            .iter()
            .map(|v| v.windows(2).map(|w| (w[0], w[1])).collect())
            .collect();

        let mut u_split = Vec::new();
        let mut v_split = Vec::new();

        // Temporary buffers to hold current combination of bounds
        let mut current_lower = [0.0; N];
        let mut current_upper = [0.0; N];

        // Recursive function to generate all combinations
        fn recurse<const N: usize>(
            interval_lists: &Vec<Vec<(f64, f64)>>,
            dim: usize,
            current_lower: &mut [f64; N],
            current_upper: &mut [f64; N],
            u_split: &mut Vec<[f64; N]>,
            v_split: &mut Vec<[f64; N]>,
        ) {
            if dim == interval_lists.len() {
                // Once all dimensions are processed, push the current bounds
                u_split.push(*current_lower);
                v_split.push(*current_upper);
                return;
            }

            // Iterate over all intervals in the current dimension
            for &(lower, upper) in &interval_lists[dim] {
                current_lower[dim] = lower;
                current_upper[dim] = upper;
                recurse(
                    interval_lists,
                    dim + 1,
                    current_lower,
                    current_upper,
                    u_split,
                    v_split,
                );
            }
        }

        // Start recursion from the first dimension
        recurse(
            &interval_lists,
            0,
            &mut current_lower,
            &mut current_upper,
            &mut u_split,
            &mut v_split,
        );

        (u_split, v_split)
    }

    fn validate_split<const N: usize>(
        u_split: &Vec<[f64; N]>,
        v_split: &Vec<[f64; N]>,
        u: &[f64; N],
        v: &[f64; N],
        n_boxes_created: usize,
    ) {
        let n_boxes = u_split.len();
        let mut current_position = v_split[0].clone(); // the initial jump after which only 1 coordinate will change at a time.

        for box_n in 1..n_boxes { // from 1 as we have already accounted for the initial jump
            for n in 0..N {
                if current_position[n] == u_split[box_n][n] && v_split[box_n][n] >= u_split[box_n][n] {
                    current_position[n] = v_split[box_n][n];
                    break; // as only 1 coordinate changes in each split
                }
            }
        }

        assert_eq!(n_boxes_created, u_split.len());
        assert_eq!(u_split.len(), v_split.len());

        assert_eq!(u_split[0], *u);
        assert_eq!(v_split[v_split.len() - 1], *v);

        assert_eq!(current_position, *v);
    }

    if u >= v {
        panic!("Error MCS main: v should be > u.\nu = {u:?}\nv = {v:?}");
    }

    let mut split_array: [Vec<f64>; N] = std::array::from_fn(|i| vec!(u[i], v[i]));

    let mut n_boxes_created: usize = 1;
    while n_boxes_created < n_boxes_to_create {
        let (n, i, length) = find_max_difference_location(&split_array);
        split_array[n].insert(i + 1, split_array[n][i] + (length / 2.));
        n_boxes_created = split_array.iter().fold(1, |acc, vec| acc * (vec.len() - 1));
    }
    let (u_split, v_split) = generate_splits(split_array);

    validate_split(&u_split, &v_split, &u, &v, n_boxes_created);

    (u_split, v_split, n_boxes_created)
}


#[cfg(test)]
mod test_split_box {
    use super::*;

    #[test]
    fn test_0() {
        let u = [-1., -2.];
        let v = [1., 2.];
        let (u_split, v_split, n_boxes_created) = split_box(&u, &v, 1);
        assert_eq!(u_split, vec![[-1., -2.]]);
        assert_eq!(v_split, vec![[1., 2.]]);
        assert_eq!(n_boxes_created, v_split.len());
    }

    #[test]
    fn test_1() {
        let u = [-1., -2., -10.];
        let v = [1., 2., 10.];
        let (u_split, v_split, n_boxes_created) = split_box(&u, &v, 2);
        assert_eq!(u_split, vec![[-1., -2., -10.], [-1., -2., 0.]]);
        assert_eq!(v_split, vec![[1., 2., 0.], [1., 2., 10.]]);
        assert_eq!(n_boxes_created, v_split.len());
    }

    #[test]
    fn test_2() {
        let u = [-5., -4., -3., -2., -1., 0., 1.];
        let v = [-1., 0., -1., 10., 3., 5., 10.];
        let (u_split, v_split, n_boxes_created) = split_box(&u, &v, 2);
        assert_eq!(u_split, vec![[-5., -4., -3., -2., -1., 0., 1.], [-5., -4., -3., 4., -1., 0., 1.]]);
        assert_eq!(v_split, vec![[-1., 0., -1., 4., 3., 5., 10.], [-1., 0., -1., 10., 3., 5., 10.]]);
        assert_eq!(n_boxes_created, v_split.len());
    }

    #[test]
    fn test_3() {
        let u = [-5., -4., -3., -2., -1., 0., 1.];
        let v = [-1., 0., -1., 10., 3., 5., 10.];
        let (u_split, v_split, n_boxes_created) = split_box(&u, &v, 4);
        assert_eq!(u_split, vec![
            [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0],
            [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 5.5],
            [-5.0, -4.0, -3.0, 4.0, -1.0, 0.0, 1.0],
            [-5.0, -4.0, -3.0, 4.0, -1.0, 0.0, 5.5]
        ]);
        assert_eq!(v_split, vec![
            [-1.0, 0.0, -1.0, 4.0, 3.0, 5.0, 5.5],
            [-1.0, 0.0, -1.0, 4.0, 3.0, 5.0, 10.0],
            [-1.0, 0.0, -1.0, 10.0, 3.0, 5.0, 5.5],
            [-1.0, 0.0, -1.0, 10.0, 3.0, 5.0, 10.0]
        ]);
        assert_eq!(n_boxes_created, v_split.len());
    }

    #[test]
    fn test_4() {
        let u = [-100., -2.];
        let v = [100., 2.];
        let (u_split, v_split, n_boxes_created) = split_box(&u, &v, 4);
        assert_eq!(u_split, vec![
            [-100.0, -2.0],
            [-50.0, -2.0],
            [0.0, -2.0],
            [50.0, -2.0]
        ]);
        assert_eq!(v_split, vec![
            [-50.0, 2.0],
            [0.0, 2.0],
            [50.0, 2.0],
            [100.0, 2.0]
        ]);
        assert_eq!(n_boxes_created, v_split.len());
    }

    #[test]
    fn test_5() {
        let u = [1.0, 2.0, 3.0, 4.0];
        let v = [5.0, 6.0, 7.0, 8.0];
        let (u_split, v_split, n_boxes_created) = split_box(&u, &v, 4);
        assert_eq!(
            u_split,
            vec![
                [1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 3.0, 4.0], [3.0, 2.0, 3.0, 4.0], [3.0, 4.0, 3.0, 4.0]
            ]
        );
        assert_eq!(
            v_split,
            vec![
                [3.0, 4.0, 7.0, 8.0], [3.0, 6.0, 7.0, 8.0], [5.0, 4.0, 7.0, 8.0], [5.0, 6.0, 7.0, 8.0]
            ]
        );
        assert_eq!(n_boxes_created, v_split.len());
    }


    #[test]
    fn test_6() {
        let u = [2.0, 4.0, 6.0];
        let v = [10.0, 20.0, 30.0];
        let (u_split, v_split, n_boxes_created) = split_box(&u, &v, 10);
        assert_eq!(
            u_split,
            vec![
                [2.0, 4.0, 6.0], [2.0, 4.0, 12.0], [2.0, 4.0, 18.0], [2.0, 4.0, 24.0], [2.0, 12.0, 6.0], [2.0, 12.0, 12.0], [2.0, 12.0, 18.0], [2.0, 12.0, 24.0], [6.0, 4.0, 6.0], [6.0, 4.0, 12.0], [6.0, 4.0, 18.0], [6.0, 4.0, 24.0], [6.0, 12.0, 6.0], [6.0, 12.0, 12.0], [6.0, 12.0, 18.0], [6.0, 12.0, 24.0]
            ]
        );
        assert_eq!(
            v_split,
            vec![
                [6.0, 12.0, 12.0], [6.0, 12.0, 18.0], [6.0, 12.0, 24.0], [6.0, 12.0, 30.0], [6.0, 20.0, 12.0], [6.0, 20.0, 18.0], [6.0, 20.0, 24.0], [6.0, 20.0, 30.0], [10.0, 12.0, 12.0], [10.0, 12.0, 18.0], [10.0, 12.0, 24.0], [10.0, 12.0, 30.0], [10.0, 20.0, 12.0], [10.0, 20.0, 18.0], [10.0, 20.0, 24.0], [10.0, 20.0, 30.0]
            ]
        );
        assert_eq!(n_boxes_created, v_split.len());
    }
}