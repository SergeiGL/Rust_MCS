use std::cmp::Ordering;

pub fn sign(x: f64) -> f64 {
    match x.partial_cmp(&0.0).unwrap() {
        Ordering::Greater => 1.0,
        Ordering::Equal => 0.0,
        Ordering::Less => -1.0,
    }
}
