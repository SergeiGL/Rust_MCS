use std::cmp::Ordering;

pub fn sign(x: f64) -> f64 {
    match x.partial_cmp(&0.0) {
        Some(Ordering::Greater) => 1.0,
        Some(Ordering::Equal) => 0.0,
        Some(Ordering::Less) => -1.0,
        None => panic!("sign:: comparison between float and float should be valid")
    }
}
