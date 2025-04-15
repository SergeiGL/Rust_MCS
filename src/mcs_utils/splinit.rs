use crate::add_basket;
use crate::mcs_utils::genbox::genbox;
use nalgebra::{Matrix2xX, Matrix3xX, SMatrix, SVector};

#[inline]
pub(crate) fn splinit<const N: usize, const SMAX: usize>(
    func: fn(&SVector<f64, N>) -> f64,
    i: usize, // -1 from Matlab
    s: usize,
    par: usize, // -1 from Matlab as comes from record
    x0: &SMatrix<f64, N, 3>,
    u: &SVector<f64, N>,
    v: &SVector<f64, N>,
    x: &SVector<f64, N>,
    xmin: &mut Vec<SVector<f64, N>>,
    fmi: &mut Vec<f64>,
    ipar: &mut Vec<Option<usize>>,
    level: &mut Vec<usize>,
    ichild: &mut Vec<isize>,
    isplit: &mut Vec<isize>,
    nogain: &mut Vec<bool>,
    f: &mut Matrix2xX<f64>,
    z: &mut Matrix2xX<f64>,
    xbest: &mut SVector<f64, N>,
    fbest: &mut f64,
    record: &mut [Option<usize>; SMAX],
    nboxes: &mut usize,
    nbasket: &mut usize,
    nsweepbest: &mut usize,
    nsweep: &mut usize,
    f0: &mut Matrix3xX<f64>,
) {
    // ncall be exactly 2 for InitEnum::Zero
    // Also l===1 and L===2 for InitEnum::Zero

    let mut x = x.clone(); // as Matlab does not return it => passed in function by cloning

    // As in Matlab we have f0(:,m) = f0[returned from splinit]
    // m is always 1 more than the f0.ncols()
    let f0_col_indx = f0.ncols();
    f0.resize_horizontally_mut(f0_col_indx + 1, 0.0);

    // flag will always be true as nsweeps != 0 => no need

    for j in 0..3 {
        if j != 1 { // j is -1 from Matlab
            x[i] = x0[(i, j)];
            f0[(j, f0_col_indx)] = func(&x);
            // Exactly 2 calls, no need for  ncall += 1
            if f0[(j, f0_col_indx)] < *fbest {
                *fbest = f0[(j, f0_col_indx)];
                *xbest = x;
                *nsweepbest = *nsweep;
            }
        } else { f0[(1, f0_col_indx)] = f[(0, par)] }
    }
    // Useless as splval1 and splval2 will never be used:
    // [fm,i1] = min(f0);
    // if i1 > 1
    // ...
    // else
    //   splval2 = v(i);
    // end
    if s + 1 < SMAX {
        let mut nchild: usize = 0;

        if u[i] < x0[(i, 0)] {
            nchild += 1;
            genbox(nboxes, ipar, level, ichild, isplit, nogain, f, z, par, s + 1, -(nchild as isize), f0[(0, f0_col_indx)], record);
        };
        for j in 0..2 { // j: -1 from Matlab
            nchild += 1;
            if (f0[(j, f0_col_indx)] <= f0[(j + 1, f0_col_indx)]) || (s + 2 < SMAX) {
                let level0 = if f0[(j, f0_col_indx)] <= f0[(j + 1, f0_col_indx)] { s + 1 } else { s + 2 };
                genbox(nboxes, ipar, level, ichild, isplit, nogain, f, z, par, level0, -(nchild as isize), f0[(j, f0_col_indx)], record);
            } else {
                x[i] = x0[(i, j)];
                add_basket(nbasket, xmin, fmi, &x, f0[(j, f0_col_indx)]);
            }
            nchild += 1;
            if (f0[(j + 1, f0_col_indx)] < f0[(j, f0_col_indx)]) || (s + 2 < SMAX) {
                let level0 = if f0[(j + 1, f0_col_indx)] < f0[(j, f0_col_indx)] { s + 1 } else { s + 2 };
                genbox(nboxes, ipar, level, ichild, isplit, nogain, f, z, par, level0, -(nchild as isize), f0[(j + 1, f0_col_indx)], record);
            } else {
                x[i] = x0[(i, j + 1)];
                add_basket(nbasket, xmin, fmi, &x, f0[(j + 1, f0_col_indx)]);
            }
        }

        if x0[(i, 2)] < v[i] {
            nchild += 1;
            genbox(nboxes, ipar, level, ichild, isplit, nogain, f, z, par, s + 1, -(nchild as isize), f0[(2, f0_col_indx)], record);
        }
    } else {
        for j in 0..3 {
            x[i] = x0[(i, j)];
            add_basket(nbasket, xmin, fmi, &x, f0[(j, f0_col_indx)]);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::hm6;

    #[test]
    fn test_splinit_0() {
        // Matlab equivalent test
        //
        // clearvars;
        // clear global;
        // n0 = []; % garbage value, do not touch
        // x1 = 0; % garbage value, do not touch
        // x2 = 0; % garbage value, do not touch
        // prt = 0; % garbage value, do not touch
        // nglob = 0;  % garbage value, do not touch
        // stop = [10000];  % garbage value, do not touch
        // y = []; % garbage value, do not touch
        //
        // fcn = "hm6"; % just rubbish; always the same
        // data = "hm6"; % just rubbish; always the same
        //
        // L = [3,3,3,3,3,3,3,3,3,3];  % just rubbish; always the same
        // l = [2,2,2,2,2,2,2,2,2,2];  % just rubbish; always the same
        // smax = 2; % len of record
        // i = 1; // +1 from Rust
        // s = 2;
        // par = 5; % Rust par is -1 from Matlab
        // x0= [
        //     [0.,0.,0.];
        //     [0.,0.,0.];
        //     [0.,0.,0.];
        //     [0.,0.,0.];
        //     [0.,0.,0.];
        //     [0.,0.,0.];
        // ];
        // u = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0];
        // v = [1.0; 2.0; 3.0; 4.0; 5.0; 6.0];
        // x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // xmin = [[0.0; 0.0; 0.0; 0.0; 0.0; 0.0]];
        // fmi = [0.0];
        // ipar = [0,0,0,0,0,0,0,0,0,0];
        // level = [0,0,0,0,0,0,0,0,0,0];
        // ichild = [0,0,0,0,0,0,0,0,0,0];
        // f = [
        //     [0,0,0,0,0,0,0,0,0,0];
        //     [0,0,0,0,0,0,0,0,0,0];
        // ];
        // xbest = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // fbest = 10.0;
        // record = [0, 0];
        // nboxes = 2;
        // nbasket = 1;
        // nsweepbest = 0;
        // nsweep = 1;
        // f0 = [];
        //
        // % Declare global variables used in the function
        // global nbasket nboxes nglob nsweep nsweepbest record xglob xloc;
        //
        // format long g;
        // [xbest,fbest,f0,xmin,fmi,ipar,level,ichild,f,flag,ncall] = splinit(fcn,data,i,s,smax,par,x0,n0,u,v,x,y,x1,x2,L,l,xmin,fmi,ipar,level,ichild,f,xbest,fbest,stop,prt);
        // disp(xbest);
        // disp(fbest);
        // disp(f0);
        // disp(xmin);
        // disp(fmi);
        // disp(ipar); % same as in Matlab
        // disp(level); % same as in Matlab
        // disp(ichild); % same as in Matlab
        // disp(f);
        // disp(record);
        // disp(nboxes);
        // disp(nbasket);
        // disp(nsweepbest);
        // disp(nsweep);

        let i = 0_usize;
        let s = 2_usize;
        let par = 4_usize;
        let x0: SMatrix<f64, 6, 3> = SMatrix::from_row_slice(&[0.0; 18]);
        let u = SVector::<f64, 6>::from_row_slice(&[0.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut x = SVector::<f64, 6>::from_row_slice(&[0.0; 6]);
        let mut xmin = vec![SVector::<f64, 6>::from_row_slice(&[0.0; 6])];
        let mut fmi = vec![0.0];
        let mut ipar = vec![Some(0); 10];
        let mut level = vec![0; 10];
        let mut ichild = vec![0; 10];
        let mut f = Matrix2xX::<f64>::zeros(10);
        let mut xbest = SVector::<f64, 6>::from_row_slice(&[0.0; 6]);
        let mut fbest = 10.0;
        let mut record = [None; 2];
        let mut nboxes = 2_usize;
        let mut nbasket = 1;
        let mut nsweepbest = 0_usize;
        let mut nsweep = 1_usize;
        let mut f0 = Matrix3xX::<f64>::zeros(0);

        splinit(hm6, i, s, par, &x0, &u, &v, &mut x, &mut xmin,
                &mut fmi, &mut ipar, &mut level, &mut ichild, &mut vec![], &mut vec![], &mut f, &mut Matrix2xX::<f64>::zeros(0), &mut xbest,
                &mut fbest, &mut record, &mut nboxes, &mut nbasket,
                &mut nsweepbest, &mut nsweep, &mut f0);

        assert_eq!(xbest.as_slice(), [0.0; 6]);
        assert_eq!(fbest, -0.00508911288366444);
        assert_eq!(f0.as_slice(), [-0.00508911288366444, 0.0, -0.00508911288366444]);
        assert_eq!(xmin, [SVector::<f64, 6>::from_row_slice(&[0.0; 6]); 4]);
        assert_eq!(fmi, [0.0, -0.00508911288366444, 0.0, -0.00508911288366444]);
        assert_eq!(ipar, [Some(0); 10]); // same as in Matlab
        assert_eq!(level, [0; 10]); // same as in Matlab
        assert_eq!(ichild, [0; 10]); // same as in Matlab
        assert_eq!(f, Matrix2xX::<f64>::zeros(10));
        assert_eq!(record, [None; 2]);
        assert_eq!(nboxes, 2);
        assert_eq!(nbasket, 4);
        assert_eq!(nsweepbest, 1);
        assert_eq!(nsweep, 1);
    }

    #[test]
    fn test_splinit_1() {
        // Matlab equivalent test
        //
        // clearvars;
        // clear global;
        // fcn = "hm6"; % just rubbish; always the same
        // data = "hm6"; % just rubbish; always the same
        // n0 = []; % just rubbish; always the same
        // y = []; % just rubbish; always the same
        // x1 = []; % just rubbish; always the same
        // x2 = []; % just rubbish; always the same
        // prt = 0; % just rubbish; always the same
        // L = [3,3,3,3,3,3,3,3,3,3];  % just rubbish; always the same
        // l = [2,2,2,2,2,2,2,2,2,2];  % just rubbish; always the same
        // stop = [100];  % just rubbish; always the same
        // smax = 2;  % len of record
        // i = 1; // +1 from Rust
        // s = 1;
        // par = 4; % +1 from Rust
        // x0= [
        //     [0.,0.,0.];
        //     [0.,0.,0.];
        //     [0.,0.,0.];
        //     [0.,0.,0.];
        //     [0.,0.,0.];
        //     [0.,0.,0.];
        // ];
        // u = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0];
        // v = [1.0; 2.0; 3.0; 4.0; 5.0; 6.0];
        // x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // xmin = ones(6, 6);
        // fmi = [10.0, 10.0, 10.0];
        // ipar = [2,2,2,2,2,2,2,2,2,2]; % +1 from Rust
        // level = [1,1,1,1,1,1,1,1,1,1];
        // ichild = [1,1,1,1,1,1,1,1,1,1];
        // f = [
        //     [1,1,1,1,1,1,1,1,1,1];
        //     [1,1,1,1,1,1,1,1,1,1];
        // ];
        // xbest = [1,1,1,1,1,1,1,1,1,1];
        // fbest = 0.0;
        // record = [0];
        // nboxes = 3;
        // nbasket = 3;
        // nsweepbest = 4;
        // nsweep = 5;
        // f0 = [];
        //
        // % Declare global variables used in the function
        // global nbasket nboxes nglob nsweep nsweepbest record xglob xloc;
        //
        // format long g;
        // [xbest,fbest,f0,xmin,fmi,ipar,level,ichild,f,flag,ncall] = splinit(fcn,data,i,s,smax,par,x0,n0,u,v,x,y,x1,x2,L,l,xmin,fmi,ipar,level,ichild,f,xbest,fbest,stop,prt);
        // disp(xbest);
        // disp(fbest);
        // disp(f0);
        // disp(xmin);
        // disp(fmi);
        // disp(ipar); % same as in Matlab
        // disp(level); % same as in Matlab
        // disp(ichild); % same as in Matlab
        // disp(f);
        // disp(record);
        // disp(nboxes);
        // disp(nbasket);
        // disp(nsweepbest);
        // disp(nsweep);

        let i = 0_usize;
        let s = 1_usize;
        let par = 3_usize;
        let x0: SMatrix<f64, 6, 3> = SMatrix::from_row_slice(&[0.0; 18]);
        let u = SVector::<f64, 6>::from_row_slice(&[0.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let mut x = SVector::<f64, 6>::from_row_slice(&[0.0; 6]);
        let mut xmin = vec![SVector::<f64, 6>::from_row_slice(&[1.0; 6]); 3];
        let mut fmi = vec![10.0, 10.0, 10.0];
        let mut ipar = vec![Some(1); 10];
        let mut level = vec![1; 10];
        let mut ichild = vec![1; 10];
        let mut f = Matrix2xX::<f64>::repeat(10, 1.0);
        let mut xbest = SVector::<f64, 6>::from_row_slice(&[1.0; 6]);
        let mut fbest = 0.0;
        let mut record = [None; 1];
        let mut nboxes = 3_usize;
        let mut nbasket = 3;
        let mut nsweepbest = 4_usize;
        let mut nsweep = 5_usize;
        let mut f0 = Matrix3xX::<f64>::zeros(0);

        splinit(hm6, i, s, par, &x0, &u, &v, &mut x, &mut xmin,
                &mut fmi, &mut ipar, &mut level, &mut ichild, &mut vec![], &mut vec![], &mut f, &mut Matrix2xX::<f64>::zeros(0), &mut xbest,
                &mut fbest, &mut record, &mut nboxes, &mut nbasket,
                &mut nsweepbest, &mut nsweep, &mut f0);

        assert_eq!(xbest.as_slice(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(fbest, -0.00508911288366444);
        assert_eq!(f0.as_slice(), [-0.00508911288366444, 1.0, -0.00508911288366444]);
        assert_eq!(xmin, [SVector::<f64, 6>::from_row_slice(&[1., 1., 1., 1., 1., 1.]), SVector::<f64, 6>::from_row_slice(&[1., 1., 1., 1., 1., 1.]), SVector::<f64, 6>::from_row_slice(&[1., 1., 1., 1., 1., 1.]), SVector::<f64, 6>::from_row_slice(&[0., 0., 0., 0., 0., 0.]), SVector::<f64, 6>::from_row_slice(&[0., 0., 0., 0., 0., 0.]), SVector::<f64, 6>::from_row_slice(&[0., 0., 0., 0., 0., 0.])]);
        assert_eq!(fmi, [10., 10., 10., -0.00508911288366444, 1.0, -0.00508911288366444]);
        assert_eq!(ipar, [Some(1); 10]); // same as in Rust
        assert_eq!(level, [1; 10]);
        assert_eq!(ichild, [1; 10]);
        assert_eq!(f, Matrix2xX::<f64>::repeat(10, 1.0));
        assert_eq!(record, [None]);
        assert_eq!(nboxes, 3);
        assert_eq!(nbasket, 6);
        assert_eq!(nsweepbest, 5);
        assert_eq!(nsweep, 5);
    }

    #[test]
    fn test_splinit_2() {
        // Matlab equivalent test
        //
        // clearvars;
        // clear global;
        // fcn = "hm6"; % just rubbish; always the same
        // data = "hm6"; % just rubbish; always the same
        // n0 = []; % just rubbish; always the same
        // y = []; % just rubbish; always the same
        // x1 = []; % just rubbish; always the same
        // x2 = []; % just rubbish; always the same
        // prt = 0; % just rubbish; always the same
        // L = [3,3,3,3,3,3,3,3,3,3];  % just rubbish; always the same
        // l = [2,2,2,2,2,2,2,2,2,2];  % just rubbish; always the same
        // stop = [100];  % just rubbish; always the same
        // smax = 9;  % len of record
        // i = 1; // +1 from Rust
        // s = 2;
        // par = 5; % +1 from Rust
        // x0= [
        //     [1.,1.,1.];
        //     [1.,1.,1.];
        //     [1.,1.,1.];
        //     [1.,1.,1.];
        //     [1.,1.,1.];
        //     [1.,1.,1.];
        // ];
        // u = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0];
        // v = [1.0; 2.0; 3.0; 4.0; 5.0; 6.0];
        // x = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
        // xmin = ones(6, 6);
        // fmi = [0.0];
        // ipar = zeros(1, 10); % +1 from Rust
        // level = zeros(1, 10);
        // ichild = ones(1, 10);
        // f = zeros(2, 5)
        // xbest = zeros(1, 6);
        // fbest = 0.0;
        // record = [0,0,0,0,0,0,0,0,0];
        // nboxes = 2;
        // nbasket = 0;
        // nsweepbest = 1;
        // nsweep = 2;
        // f0 = [];
        //
        // % Declare global variables used in the function
        // global nbasket nboxes nglob nsweep nsweepbest record xglob xloc;
        //
        // format long g;
        // [xbest,fbest,f0,xmin,fmi,ipar,level,ichild,f,flag,ncall] = splinit(fcn,data,i,s,smax,par,x0,n0,u,v,x,y,x1,x2,L,l,xmin,fmi,ipar,level,ichild,f,xbest,fbest,stop,prt);
        // disp(xbest);
        // disp(fbest);
        // disp(f0);
        // disp(xmin);
        // disp(fmi);
        // disp(ipar); % same as in Matlab
        // disp(level); % same as in Matlab
        // disp(ichild); % same as in Matlab
        // disp(f);
        // disp(record);
        // disp(nboxes);
        // disp(nbasket);
        // disp(nsweepbest);
        // disp(nsweep);

        let i = 0_usize;
        let s = 2_usize;
        let par = 4_usize;
        let x0: SMatrix<f64, 6, 3> = SMatrix::from_row_slice(&[1.0; 18]);
        let u = SVector::<f64, 6>::from_row_slice(&[0.0; 6]);
        let v = SVector::<f64, 6>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut x = SVector::<f64, 6>::from_row_slice(&[5.0; 6]);
        let mut xmin = vec![SVector::<f64, 6>::from_row_slice(&[1.0; 6])];
        let mut fmi = vec![0.0];
        let mut ipar = vec![Some(0); 10];
        let mut level = vec![0; 10];
        let mut ichild = vec![1; 10];
        let mut f = Matrix2xX::<f64>::zeros(10);
        let mut xbest = SVector::<f64, 6>::from_row_slice(&[0.0; 6]);
        let mut fbest = 0.0;
        let mut record = [None; 9];
        let mut nboxes = 2_usize;
        let mut nbasket = 0;
        let mut nsweepbest = 1_usize;
        let mut nsweep = 2_usize;
        let mut f0 = Matrix3xX::<f64>::zeros(0);

        splinit(hm6, i, s, par, &x0, &u, &v, &mut x, &mut xmin,
                &mut fmi, &mut ipar, &mut level, &mut ichild, &mut vec![], &mut vec![], &mut f, &mut Matrix2xX::<f64>::zeros(0), &mut xbest,
                &mut fbest, &mut record, &mut nboxes, &mut nbasket,
                &mut nsweepbest, &mut nsweep, &mut f0);

        let expected_f = Matrix2xX::<f64>::from_row_slice(&[
            0.0, 0.0, -8.679282323749578e-298, -8.679282323749578e-298, 0.0, 0.0, -8.679282323749578e-298, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);

        assert_eq!(xbest, SVector::<f64, 6>::from_row_slice(&[1., 5., 5., 5., 5., 5., ]));
        assert_eq!(fbest, -8.679282323749578e-298);
        assert_eq!(f0.as_slice(), [-8.679282323749578e-298, 0.0, -8.679282323749578e-298]);
        assert_eq!(xmin, [SVector::<f64, 6>::from_row_slice(&[1., 1., 1., 1., 1., 1.])]);
        assert_eq!(fmi, [0.0]);
        assert_eq!(ipar, [Some(0), Some(0), Some(5), Some(5), Some(5), Some(5), Some(5), Some(0), Some(0), Some(0), ]);
        assert_eq!(level, [0, 0, 3, 3, 4, 4, 3, 0, 0, 0]);
        assert_eq!(ichild, [1, 1, -1, -2, -3, -4, -5, 1, 1, 1]);
        assert_eq!(f, expected_f);
        assert_eq!(record, [None, None, Some(2), Some(4), None, None, None, None, None]); // -1 from Matlab
        assert_eq!(nboxes, 7);
        assert_eq!(nbasket, 0);
        assert_eq!(nsweepbest, 2);
        assert_eq!(nsweep, 2);
    }
}