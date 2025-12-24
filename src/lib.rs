use nalgebra as na;
use std::ops::{Add, Index, Mul};

#[derive(Debug, Clone)]
struct MyVec<T>(Vec<T>);

impl<T> Index<usize> for MyVec<T> {
    type Output = T;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.0[idx]
    }
}

type Float = f64;
impl Add<MyVec<Float>> for MyVec<Float> {
    type Output = MyVec<Float>;
    fn add(self, rhs: MyVec<Float>) -> MyVec<Float> {
        let v: Vec<Float> = self
            .0
            .into_iter()
            .zip(rhs.0.iter())
            .map(|(a, b)| a + b)
            .collect();
        MyVec(v)
    }
}
impl Add<&MyVec<Float>> for MyVec<Float> {
    type Output = MyVec<Float>;
    fn add(self, rhs: &MyVec<Float>) -> MyVec<Float> {
        let v: Vec<Float> = self
            .0
            .into_iter()
            .zip(rhs.0.iter())
            .map(|(a, b)| a + b)
            .collect();
        MyVec(v)
    }
}
impl Add<&MyVec<Float>> for &MyVec<Float> {
    type Output = MyVec<Float>;
    fn add(self, rhs: &MyVec<Float>) -> MyVec<Float> {
        let v: Vec<Float> = self
            .0
            .iter()
            .zip(rhs.0.iter())
            .map(|(a, b)| a + b)
            .collect();
        MyVec(v)
    }
}
impl Add<MyVec<Float>> for &MyVec<Float> {
    type Output = MyVec<Float>;
    fn add(self, rhs: MyVec<Float>) -> MyVec<Float> {
        rhs + self
    }
}
impl Mul<Float> for MyVec<Float> {
    type Output = MyVec<Float>;
    fn mul(self, rhs: Float) -> MyVec<Float> {
        let v: Vec<Float> = self.0.into_iter().map(|a| a * rhs).collect();
        MyVec(v)
    }
}
impl Mul<Float> for &MyVec<Float> {
    type Output = MyVec<Float>;
    fn mul(self, rhs: Float) -> MyVec<Float> {
        let v: Vec<Float> = self.0.iter().map(|a| a * rhs).collect();
        MyVec(v)
    }
}
impl Mul<MyVec<Float>> for Float {
    type Output = MyVec<Float>;
    fn mul(self, rhs: MyVec<Float>) -> MyVec<Float> {
        rhs * self
    }
}
impl Mul<&MyVec<Float>> for Float {
    type Output = MyVec<Float>;
    fn mul(self, rhs: &MyVec<Float>) -> MyVec<Float> {
        rhs * self
    }
}

impl MyVec<Float> {
    fn into_na(self) -> na::DVector<Float> {
        na::DVector::from_vec(self.0)
    }
    fn from_na(v: na::DVector<Float>) -> Self {
        MyVec(v.data.as_vec().clone())
    }
}

struct Jacobian {
    df_dx: MyVec<Float>,
    df_dy: na::DMatrix<Float>,
}

trait DiffEq {
    fn derivative(&self, x: Float, y: &MyVec<Float>) -> MyVec<Float>;
    fn jacobian(&self, x: Float, y: &MyVec<Float>) -> Jacobian;
}

fn deriv_ridders(func: &dyn Fn(Float) -> Float, x: Float, h: Float) -> Float {
    const N_TAB: usize = 10;
    const CON: Float = 1.4;
    const CON2: Float = CON * CON;
    const SAFE: Float = 2.0;
    assert_ne!(h, 0.0);
    let mut a: [[Float; N_TAB]; N_TAB] = [[0.0; N_TAB]; N_TAB];
    let mut hh = h;
    a[0][0] = (func(x + hh) - func(x - hh)) / (2.0 * hh);
    let mut err = Float::MAX;
    let mut ans = a[0][0];
    for i in 1..N_TAB {
        hh /= CON;
        a[0][i] = (func(x + hh) - func(x - hh)) / (2.0 * hh);
        let mut fac = CON2;
        for j in 1..=i {
            a[j][i] = (fac * a[j - 1][i] - a[j - 1][i - 1]) / (fac - 1.0);
            fac *= CON2;
            let errt = (a[j][i] - a[j - 1][i])
                .abs()
                .max((a[j][i] - a[j - 1][i - 1]).abs());
            if errt < err {
                err = errt;
                ans = a[j][i];
            }
        }
        if (a[i][i] - a[i - 1][i - 1]).abs() >= SAFE * err {
            break;
        }
    }
    ans
}

struct ProbStat {
    x: Float,
    y: MyVec<Float>,
    dy_dx: MyVec<Float>,
    jacobian: Jacobian,
}
struct Odeint<S: Stepper> {
    prob_stat: ProbStat,
    n_ok: i32,
    n_bad: i32,
    x1: Float,
    x2: Float,
    y_start: MyVec<Float>,
    h_min: Float,
    dense: bool,
    derivative: Box<dyn DiffEq>,
    stepper: S,
    n_step: i32,
    h: Float,
    output: Output,
}

impl<S: Stepper> Odeint<S> {
    const MAXSTP: usize = 50000;
    //const EPS: Float = Float::EPSILON;
    fn new(
        y_start: MyVec<Float>,
        x1: Float,
        x2: Float,
        a_tol: Float,
        r_tol: Float,
        h: Float,
        h_min: Float,
        output: Output,
        derivative: Box<dyn DiffEq>,
    ) -> Self {
        let dense = output.n_save > 0;
        let mut res = Odeint {
            prob_stat: ProbStat {
                x: x1,
                y: y_start.clone(),
                dy_dx: derivative.derivative(x1, &y_start),
                jacobian: derivative.jacobian(x1, &y_start),
            },
            n_ok: 0,
            n_bad: 0,
            x1,
            x2,
            y_start,
            h_min,
            dense,
            derivative,
            stepper: S::new(a_tol, r_tol, dense),
            n_step: 0,
            h: h * (x2 - x1).signum(),
            output,
        };
        res.output.init(x1, x2);
        res
    }

    fn integrate(&mut self) -> Result<(), &'static str> {
        if self.dense {
            self.output.out(
                -1,
                self.prob_stat.x,
                &self.prob_stat.y,
                &self.stepper,
                self.h,
            )?;
        } else {
            self.output.save(self.prob_stat.x, &self.prob_stat.y);
        }
        for n_step in 0..Self::MAXSTP {
            self.n_step = n_step as i32;
            if (self.prob_stat.x + self.h * 1.0001 - self.x2) * (self.x2 - self.x1) >= 0.0 {
                self.h = self.x2 - self.prob_stat.x;
            }
            self.stepper
                .step(&mut self.prob_stat, self.h, &*self.derivative)?;
            let s_data = self.stepper.data();
            if s_data.h_did == self.h {
                self.n_ok += 1;
            } else {
                self.n_bad += 1;
            }
            if self.dense {
                self.output.out(
                    n_step as i32,
                    self.prob_stat.x,
                    &self.prob_stat.y,
                    &self.stepper,
                    s_data.h_did,
                )?;
            } else {
                self.output.save(self.prob_stat.x, &self.prob_stat.y);
            }
            if (self.prob_stat.x - self.x2) * (self.x2 - self.x1) >= 0.0 {
                self.y_start = self.prob_stat.y.clone();
                self.output.save(self.prob_stat.x, &self.prob_stat.y);
                return Ok(());
            }
            if s_data.h_next <= self.h_min {
                return Err("Step size underflow in Odeint::integrate");
            }
            self.h = s_data.h_next;
        }
        Err("Too many steps in Odeint::integrate")
    }
    fn y_out(&self) -> &Vec<MyVec<Float>> {
        &self.output.y_save
    }
    fn x_out(&self) -> &Vec<Float> {
        &self.output.x_save
    }
}

struct Output {
    n_save: i32,
    dense: bool,
    x1: Float,
    x2: Float,
    x_out: Float,
    dx_out: Float,
    x_save: Vec<Float>,
    y_save: Vec<MyVec<Float>>,
}

impl Output {
    fn new(n_save: i32) -> Self {
        Output {
            n_save,
            dense: n_save > 0,
            x1: 0.0,
            x2: 0.0,
            x_out: 0.0,
            dx_out: 0.0,
            x_save: vec![],
            y_save: vec![],
        }
    }
    fn init(&mut self, x1: Float, x2: Float) {
        if self.dense {
            self.x1 = x1;
            self.x2 = x2;
            self.x_out = x1;
            self.dx_out = (x2 - x1) / (self.n_save as Float);
        }
    }
    fn save_dense<S: Stepper>(&mut self, stepper: &S, x_out: Float, h: Float) {
        self.y_save.push(stepper.dense_out(x_out, h));
        self.x_save.push(x_out);
    }
    fn save(&mut self, x: Float, y: &MyVec<Float>) {
        self.y_save.push(y.clone());
        self.x_save.push(x);
    }
    fn out<S: Stepper>(
        &mut self,
        n_step: i32,
        x: Float,
        y: &MyVec<Float>,
        stepper: &S,
        h: Float,
    ) -> Result<(), &'static str> {
        if !self.dense {
            return Err("Output::out called when dense output is disabled");
        }
        if n_step == -1 {
            self.save(x, y);
            self.x_out += self.dx_out;
        } else {
            while (x - self.x_out) * (self.x2 - self.x1) > 0.0 {
                self.save_dense(stepper, self.x_out, h);
                self.x_out += self.dx_out;
            }
        }
        Ok(())
    }
}

struct StepperData {
    x_old: Float,
    a_tol: Float,
    r_tol: Float,
    dense: bool,
    h_did: Float,
    h_next: Float,
    eps: Float,
    y_out: MyVec<Float>,
    y_err: MyVec<Float>,
}

impl StepperData {
    fn new(a_tol: Float, r_tol: Float, dense: bool) -> Self {
        StepperData {
            x_old: 0.0,
            a_tol,
            r_tol,
            dense,
            h_did: 0.0,
            h_next: 0.0,
            eps: Float::EPSILON,
            y_out: MyVec(vec![]),
            y_err: MyVec(vec![]),
        }
    }
}
trait Stepper {
    fn new(a_tol: Float, r_tol: Float, dense: bool) -> Self;
    fn step(
        &mut self,
        stat: &mut ProbStat,
        h: Float,
        derivative: &dyn DiffEq,
    ) -> Result<(), &'static str>;
    fn dense_out(&self, x: Float, h: Float) -> MyVec<Float>;
    fn data(&self) -> &StepperData;
}

struct Controller853 {
    h_next: Float,
    err_old: Float,
    reject: bool,
}
#[allow(non_upper_case_globals)]
impl Controller853 {
    fn new() -> Self {
        Controller853 {
            h_next: 1.0,
            err_old: 1.0e-4,
            reject: false,
        }
    }
    const beta: Float = 0.0;
    const alpha: Float = 1.0 / 8.0 - Self::beta * 0.2;
    const safe: Float = 0.9;
    const min_scale: Float = 0.333;
    const max_scale: Float = 6.0;
    fn success(&mut self, err: Float, h: Float) -> Option<Float> {
        let mut scale: Float;
        if err <= 1.0 {
            if err == 0.0 {
                scale = Self::max_scale;
            } else {
                scale = Self::safe * err.powf(-Self::alpha);
                scale = scale.max(Self::min_scale);
                scale = scale.min(Self::max_scale);
            }
            if self.reject {
                self.h_next = h * scale.min(1.0);
            } else {
                self.h_next = h * scale;
            }
            self.err_old = err.max(1e-4);
            self.reject = false;
            None
        } else {
            scale = Self::min_scale.max(Self::safe * err.powf(-Self::alpha));
            self.reject = true;
            Some(h * scale)
        }
    }
}

struct StepperEuler {
    data: StepperData,
}
impl Stepper for StepperEuler {
    fn new(a_tol: Float, r_tol: Float, dense: bool) -> Self {
        StepperEuler {
            data: StepperData::new(a_tol, r_tol, dense),
        }
    }
    fn step(
        &mut self,
        stat: &mut ProbStat,
        h: Float,
        derivative: &dyn DiffEq,
    ) -> Result<(), &'static str> {
        let dy_dx_new = derivative.derivative(stat.x + h, &stat.y);
        self.data.y_out = &stat.y + h * &dy_dx_new;
        stat.dy_dx = dy_dx_new;
        stat.y = self.data.y_out.clone();
        self.data.x_old = stat.x;
        stat.x += h;
        self.data.h_did = h;
        self.data.h_next = h;
        Ok(())
    }
    fn dense_out(&self, x: Float, h: Float) -> MyVec<Float> {
        let s = (x - self.data.x_old) / h;
        &self.data.y_out * s + &self.data.y_out * (1.0 - s)
    }
    fn data(&self) -> &StepperData {
        &self.data
    }
}

struct StepperDopr853 {
    data: StepperData,
    y_err2: MyVec<Float>,
    k2: MyVec<Float>,
    k3: MyVec<Float>,
    k4: MyVec<Float>,
    k5: MyVec<Float>,
    k6: MyVec<Float>,
    k7: MyVec<Float>,
    k8: MyVec<Float>,
    k9: MyVec<Float>,
    k10: MyVec<Float>,
    rcont1: MyVec<Float>,
    rcont2: MyVec<Float>,
    rcont3: MyVec<Float>,
    rcont4: MyVec<Float>,
    rcont5: MyVec<Float>,
    rcont6: MyVec<Float>,
    rcont7: MyVec<Float>,
    rcont8: MyVec<Float>,
    con: Controller853,
}

impl Stepper for StepperDopr853 {
    fn new(a_tol: Float, r_tol: Float, dense: bool) -> Self {
        StepperDopr853 {
            data: StepperData::new(a_tol, r_tol, dense),
            y_err2: MyVec(vec![]),
            k2: MyVec(vec![]),
            k3: MyVec(vec![]),
            k4: MyVec(vec![]),
            k5: MyVec(vec![]),
            k6: MyVec(vec![]),
            k7: MyVec(vec![]),
            k8: MyVec(vec![]),
            k9: MyVec(vec![]),
            k10: MyVec(vec![]),
            rcont1: MyVec(vec![]),
            rcont2: MyVec(vec![]),
            rcont3: MyVec(vec![]),
            rcont4: MyVec(vec![]),
            rcont5: MyVec(vec![]),
            rcont6: MyVec(vec![]),
            rcont7: MyVec(vec![]),
            rcont8: MyVec(vec![]),
            con: Controller853::new(),
        }
    }
    fn step(
        &mut self,
        stat: &mut ProbStat,
        h_try: Float,
        derivative: &dyn DiffEq,
    ) -> Result<(), &'static str> {
        let mut h = h_try;
        loop {
            self.prepare_step(stat, h, derivative);
            let err = self.error(stat, h);
            match self.con.success(err, h) {
                Some(h_new) => {
                    h = h_new;
                    if h.abs() <= self.data.eps * stat.x.abs() {
                        return Err("Step size underflow in StepperDopr853");
                    }
                }
                None => break,
            }
        }
        let dy_dx_new = derivative.derivative(stat.x + h, &self.data.y_out);
        if self.data.dense {
            self.prepare_dense(stat, h, &dy_dx_new, derivative);
        }
        stat.dy_dx = dy_dx_new;
        stat.y = self.data.y_out.clone();
        self.data.x_old = stat.x;
        stat.x += h;
        self.data.h_did = h;
        self.data.h_next = self.con.h_next;
        Ok(())
    }
    fn dense_out(&self, x: Float, h: Float) -> MyVec<Float> {
        let s = (x - self.data.x_old) / h;
        let s1 = 1.0 - s;
        &self.rcont1
            + s * (&self.rcont2
            + s1 * (&self.rcont3
            + s * (&self.rcont4
            + s1 * (&self.rcont5
            + s * (&self.rcont6 + s1 * (&self.rcont7 + s * &self.rcont8))))))
    }
    fn data(&self) -> &StepperData {
        &self.data
    }
}

#[allow(non_upper_case_globals)]
impl StepperDopr853 {
    fn prepare_step(&mut self, stat: &ProbStat, h: Float, derivative: &dyn DiffEq) {
        let y_temp = &stat.y + (h * Self::a21) * &stat.dy_dx;
        self.k2 = derivative.derivative(stat.x + Self::c2 * h, &y_temp);
        let y_temp = &stat.y + h * (Self::a31 * &stat.dy_dx + Self::a32 * &self.k2);
        self.k3 = derivative.derivative(stat.x + Self::c3 * h, &y_temp);
        let y_temp = &stat.y + h * (Self::a41 * &stat.dy_dx + Self::a43 * &self.k3);
        self.k4 = derivative.derivative(stat.x + Self::c4 * h, &y_temp);
        let y_temp =
            &stat.y + h * (Self::a51 * &stat.dy_dx + Self::a53 * &self.k3 + Self::a54 * &self.k4);
        self.k5 = derivative.derivative(stat.x + Self::c5 * h, &y_temp);
        let y_temp =
            &stat.y + h * (Self::a61 * &stat.dy_dx + Self::a64 * &self.k4 + Self::a65 * &self.k5);
        self.k6 = derivative.derivative(stat.x + Self::c6 * h, &y_temp);
        let y_temp = &stat.y
            + h * (Self::a71 * &stat.dy_dx
            + Self::a74 * &self.k4
            + Self::a75 * &self.k5
            + Self::a76 * &self.k6);
        self.k7 = derivative.derivative(stat.x + Self::c7 * h, &y_temp);
        let y_temp = &stat.y
            + h * (Self::a81 * &stat.dy_dx
            + Self::a84 * &self.k4
            + Self::a85 * &self.k5
            + Self::a86 * &self.k6
            + Self::a87 * &self.k7);
        self.k8 = derivative.derivative(stat.x + Self::c8 * h, &y_temp);
        let y_temp = &stat.y
            + h * (Self::a91 * &stat.dy_dx
            + Self::a94 * &self.k4
            + Self::a95 * &self.k5
            + Self::a96 * &self.k6
            + Self::a97 * &self.k7
            + Self::a98 * &self.k8);
        self.k9 = derivative.derivative(stat.x + Self::c9 * h, &y_temp);
        let y_temp = &stat.y
            + h * (Self::a101 * &stat.dy_dx
            + Self::a104 * &self.k4
            + Self::a105 * &self.k5
            + Self::a106 * &self.k6
            + Self::a107 * &self.k7
            + Self::a108 * &self.k8
            + Self::a109 * &self.k9);
        self.k10 = derivative.derivative(stat.x + Self::c10 * h, &y_temp);
        let y_temp = &stat.y
            + h * (Self::a111 * &stat.dy_dx
            + Self::a114 * &self.k4
            + Self::a115 * &self.k5
            + Self::a116 * &self.k6
            + Self::a117 * &self.k7
            + Self::a118 * &self.k8
            + Self::a119 * &self.k9
            + Self::a1110 * &self.k10);
        self.k2 = derivative.derivative(stat.x + Self::c11 * h, &y_temp);
        let y_temp = &stat.y
            + h * (Self::a121 * &stat.dy_dx
            + Self::a124 * &self.k4
            + Self::a125 * &self.k5
            + Self::a126 * &self.k6
            + Self::a127 * &self.k7
            + Self::a128 * &self.k8
            + Self::a129 * &self.k9
            + Self::a1210 * &self.k10
            + Self::a1211 * &self.k2);
        self.k3 = derivative.derivative(stat.x + h, &y_temp);
        self.k4 = Self::b1 * &stat.dy_dx
            + Self::b6 * &self.k6
            + Self::b7 * &self.k7
            + Self::b8 * &self.k8
            + Self::b9 * &self.k9
            + Self::b10 * &self.k10
            + Self::b11 * &self.k2
            + Self::b12 * &self.k3;
        self.data.y_out = &stat.y + h * &self.k4;
        self.data.y_err =
            &self.k4 + -Self::bhh1 * &stat.dy_dx + -Self::bhh2 * &self.k9 + -Self::bhh3 * &self.k3;
        self.y_err2 = Self::er1 * &stat.dy_dx
            + Self::er6 * &self.k6
            + Self::er7 * &self.k7
            + Self::er8 * &self.k8
            + Self::er9 * &self.k9
            + Self::er10 * &self.k10
            + Self::er11 * &self.k2
            + Self::er12 * &self.k3;
    }
    fn prepare_dense(
        &mut self,
        stat: &ProbStat,
        h: Float,
        dy_dx_new: &MyVec<Float>,
        derivative: &dyn DiffEq,
    ) {
        self.rcont1 = stat.y.clone();
        let ydiff = &self.data.y_out + &stat.y * -1.0;
        self.rcont2 = ydiff.clone();
        let bspl = h * &stat.dy_dx + &ydiff * -1.0;
        self.rcont3 = bspl.clone();
        self.rcont4 = &ydiff + (-h * dy_dx_new + &bspl * -1.0);
        self.rcont5 = Self::d41 * &stat.dy_dx
            + Self::d46 * &self.k6
            + Self::d47 * &self.k7
            + Self::d48 * &self.k8
            + Self::d49 * &self.k9
            + Self::d410 * &self.k10
            + Self::d411 * &self.k2
            + Self::d412 * &self.k3;
        self.rcont6 = Self::d51 * &stat.dy_dx
            + Self::d56 * &self.k6
            + Self::d57 * &self.k7
            + Self::d58 * &self.k8
            + Self::d59 * &self.k9
            + Self::d510 * &self.k10
            + Self::d511 * &self.k2
            + Self::d512 * &self.k3;
        self.rcont7 = Self::d61 * &stat.dy_dx
            + Self::d66 * &self.k6
            + Self::d67 * &self.k7
            + Self::d68 * &self.k8
            + Self::d69 * &self.k9
            + Self::d610 * &self.k10
            + Self::d611 * &self.k2
            + Self::d612 * &self.k3;
        self.rcont8 = Self::d71 * &stat.dy_dx
            + Self::d76 * &self.k6
            + Self::d77 * &self.k7
            + Self::d78 * &self.k8
            + Self::d79 * &self.k9
            + Self::d710 * &self.k10
            + Self::d711 * &self.k2
            + Self::d712 * &self.k3;
        let y_temp = &stat.y
            + h * (Self::a141 * &stat.dy_dx
            + Self::a147 * &self.k7
            + Self::a148 * &self.k8
            + Self::a149 * &self.k9
            + Self::a1410 * &self.k10
            + Self::a1411 * &self.k2
            + Self::a1412 * &self.k3
            + Self::a1413 * dy_dx_new);
        self.k10 = derivative.derivative(stat.x + h * Self::c14, &y_temp);
        let y_temp = &stat.y
            + h * (Self::a151 * &stat.dy_dx
            + Self::a156 * &self.k6
            + Self::a157 * &self.k7
            + Self::a158 * &self.k8
            + Self::a1511 * &self.k2
            + Self::a1512 * &self.k3
            + Self::a1513 * dy_dx_new
            + Self::a1514 * &self.k10);
        self.k2 = derivative.derivative(stat.x + h * Self::c15, &y_temp);
        let y_temp = &stat.y
            + h * (Self::a161 * &stat.dy_dx
            + Self::a166 * &self.k6
            + Self::a167 * &self.k7
            + Self::a168 * &self.k8
            + Self::a169 * &self.k9
            + Self::a1613 * dy_dx_new
            + Self::a1614 * &self.k10
            + Self::a1615 * &self.k2);
        self.k3 = derivative.derivative(stat.x + h * Self::c16, &y_temp);
        self.rcont5 = h
            * (&self.rcont5
            + Self::d413 * dy_dx_new
            + Self::d414 * &self.k10
            + Self::d415 * &self.k2
            + Self::d416 * &self.k3);
        self.rcont6 = h
            * (&self.rcont6
            + Self::d513 * dy_dx_new
            + Self::d514 * &self.k10
            + Self::d515 * &self.k2
            + Self::d516 * &self.k3);
        self.rcont7 = h
            * (&self.rcont7
            + Self::d613 * dy_dx_new
            + Self::d614 * &self.k10
            + Self::d615 * &self.k2
            + Self::d616 * &self.k3);
        self.rcont8 = h
            * (&self.rcont8
            + Self::d713 * dy_dx_new
            + Self::d714 * &self.k10
            + Self::d715 * &self.k2
            + Self::d716 * &self.k3);
    }
    fn error(&self, stat: &ProbStat, h: Float) -> Float {
        let mut err = 0.0;
        let mut err2 = 0.0;
        let n = stat.y.0.len();
        for i in 0..n {
            let sk =
                self.data.a_tol + self.data.r_tol * stat.y[i].abs().max(self.data.y_out[i].abs());
            err += (self.data.y_err[i] / sk).powi(2);
            err2 += (self.y_err2[i] / sk).powi(2);
        }
        let deno = err + 0.01 * err2;
        let deno = if deno > 0.0 { deno } else { 1.0 };
        h.abs() * err * (1.0 / (n as Float * deno)).sqrt()
    }
    const c2: Float = 0.526001519587677318785587544488e-01;
    const c3: Float = 0.789002279381515978178381316732e-01;
    const c4: Float = 0.118350341907227396726757197510e+00;
    const c5: Float = 0.281649658092772603273242802490e+00;
    const c6: Float = 0.333333333333333333333333333333e+00;
    const c7: Float = 0.25e+00;
    const c8: Float = 0.307692307692307692307692307692e+00;
    const c9: Float = 0.651282051282051282051282051282e+00;
    const c10: Float = 0.6e+00;
    const c11: Float = 0.857142857142857142857142857142e+00;
    const c14: Float = 0.1e+00;
    const c15: Float = 0.2e+00;
    const c16: Float = 0.777777777777777777777777777778e+00;
    const b1: Float = 5.42937341165687622380535766363e-2;
    const b6: Float = 4.45031289275240888144113950566e0;
    const b7: Float = 1.89151789931450038304281599044e0;
    const b8: Float = -5.8012039600105847814672114227e0;
    const b9: Float = 3.1116436695781989440891606237e-1;
    const b10: Float = -1.52160949662516078556178806805e-1;
    const b11: Float = 2.01365400804030348374776537501e-1;
    const b12: Float = 4.47106157277725905176885569043e-2;
    const bhh1: Float = 0.244094488188976377952755905512e+00;
    const bhh2: Float = 0.733846688281611857341361741547e+00;
    const bhh3: Float = 0.220588235294117647058823529412e-01;
    const er1: Float = 0.1312004499419488073250102996e-01;
    const er6: Float = -0.1225156446376204440720569753e+01;
    const er7: Float = -0.4957589496572501915214079952e+00;
    const er8: Float = 0.1664377182454986536961530415e+01;
    const er9: Float = -0.3503288487499736816886487290e+00;
    const er10: Float = 0.3341791187130174790297318841e+00;
    const er11: Float = 0.8192320648511571246570742613e-01;
    const er12: Float = -0.2235530786388629525884427845e-01;
    const a21: Float = 5.26001519587677318785587544488e-2;
    const a31: Float = 1.97250569845378994544595329183e-2;
    const a32: Float = 5.91751709536136983633785987549e-2;
    const a41: Float = 2.95875854768068491816892993775e-2;
    const a43: Float = 8.87627564304205475450678981324e-2;
    const a51: Float = 2.41365134159266685502369798665e-1;
    const a53: Float = -8.84549479328286085344864962717e-1;
    const a54: Float = 9.24834003261792003115737966543e-1;
    const a61: Float = 3.7037037037037037037037037037e-2;
    const a64: Float = 1.70828608729473871279604482173e-1;
    const a65: Float = 1.25467687566822425016691814123e-1;
    const a71: Float = 3.7109375e-2;
    const a74: Float = 1.70252211019544039314978060272e-1;
    const a75: Float = 6.02165389804559606850219397283e-2;
    const a76: Float = -1.7578125e-2;
    const a81: Float = 3.70920001185047927108779319836e-2;
    const a84: Float = 1.70383925712239993810214054705e-1;
    const a85: Float = 1.07262030446373284651809199168e-1;
    const a86: Float = -1.53194377486244017527936158236e-2;
    const a87: Float = 8.27378916381402288758473766002e-3;
    const a91: Float = 6.24110958716075717114429577812e-1;
    const a94: Float = -3.36089262944694129406857109825e0;
    const a95: Float = -8.68219346841726006818189891453e-1;
    const a96: Float = 2.75920996994467083049415600797e1;
    const a97: Float = 2.01540675504778934086186788979e1;
    const a98: Float = -4.34898841810699588477366255144e1;
    const a101: Float = 4.77662536438264365890433908527e-1;
    const a104: Float = -2.48811461997166764192642586468e0;
    const a105: Float = -5.90290826836842996371446475743e-1;
    const a106: Float = 2.12300514481811942347288949897e1;
    const a107: Float = 1.52792336328824235832596922938e1;
    const a108: Float = -3.32882109689848629194453265587e1;
    const a109: Float = -2.03312017085086261358222928593e-2;
    const a111: Float = -9.3714243008598732571704021658e-1;
    const a114: Float = 5.18637242884406370830023853209e0;
    const a115: Float = 1.09143734899672957818500254654e0;
    const a116: Float = -8.14978701074692612513997267357e0;
    const a117: Float = -1.85200656599969598641566180701e1;
    const a118: Float = 2.27394870993505042818970056734e1;
    const a119: Float = 2.49360555267965238987089396762e0;
    const a1110: Float = -3.0467644718982195003823669022e0;
    const a121: Float = 2.27331014751653820792359768449e0;
    const a124: Float = -1.05344954667372501984066689879e1;
    const a125: Float = -2.00087205822486249909675718444e0;
    const a126: Float = -1.79589318631187989172765950534e1;
    const a127: Float = 2.79488845294199600508499808837e1;
    const a128: Float = -2.85899827713502369474065508674e0;
    const a129: Float = -8.87285693353062954433549289258e0;
    const a1210: Float = 1.23605671757943030647266201528e1;
    const a1211: Float = 6.43392746015763530355970484046e-1;
    const a141: Float = 5.61675022830479523392909219681e-2;
    const a147: Float = 2.53500210216624811088794765333e-1;
    const a148: Float = -2.46239037470802489917441475441e-1;
    const a149: Float = -1.24191423263816360469010140626e-1;
    const a1410: Float = 1.5329179827876569731206322685e-1;
    const a1411: Float = 8.20105229563468988491666602057e-3;
    const a1412: Float = 7.56789766054569976138603589584e-3;
    const a1413: Float = -8.298e-3;
    const a151: Float = 3.18346481635021405060768473261e-2;
    const a156: Float = 2.83009096723667755288322961402e-2;
    const a157: Float = 5.35419883074385676223797384372e-2;
    const a158: Float = -5.49237485713909884646569340306e-2;
    const a1511: Float = -1.08347328697249322858509316994e-4;
    const a1512: Float = 3.82571090835658412954920192323e-4;
    const a1513: Float = -3.40465008687404560802977114492e-4;
    const a1514: Float = 1.41312443674632500278074618366e-1;
    const a161: Float = -4.28896301583791923408573538692e-1;
    const a166: Float = -4.69762141536116384314449447206e0;
    const a167: Float = 7.68342119606259904184240953878e0;
    const a168: Float = 4.06898981839711007970213554331e0;
    const a169: Float = 3.56727187455281109270669543021e-1;
    const a1613: Float = -1.39902416515901462129418009734e-3;
    const a1614: Float = 2.9475147891527723389556272149e0;
    const a1615: Float = -9.15095847217987001081870187138e0;
    const d41: Float = -0.84289382761090128651353491142e+01;
    const d46: Float = 0.56671495351937776962531783590e+00;
    const d47: Float = -0.30689499459498916912797304727e+01;
    const d48: Float = 0.23846676565120698287728149680e+01;
    const d49: Float = 0.21170345824450282767155149946e+01;
    const d410: Float = -0.87139158377797299206789907490e+00;
    const d411: Float = 0.22404374302607882758541771650e+01;
    const d412: Float = 0.63157877876946881815570249290e+00;
    const d413: Float = -0.88990336451333310820698117400e-01;
    const d414: Float = 0.18148505520854727256656404962e+02;
    const d415: Float = -0.91946323924783554000451984436e+01;
    const d416: Float = -0.44360363875948939664310572000e+01;
    const d51: Float = 0.10427508642579134603413151009e+02;
    const d56: Float = 0.24228349177525818288430175319e+03;
    const d57: Float = 0.16520045171727028198505394887e+03;
    const d58: Float = -0.37454675472269020279518312152e+03;
    const d59: Float = -0.22113666853125306036270938578e+02;
    const d510: Float = 0.77334326684722638389603898808e+01;
    const d511: Float = -0.30674084731089398182061213626e+02;
    const d512: Float = -0.93321305264302278729567221706e+01;
    const d513: Float = 0.15697238121770843886131091075e+02;
    const d514: Float = -0.31139403219565177677282850411e+02;
    const d515: Float = -0.93529243588444783865713862664e+01;
    const d516: Float = 0.35816841486394083752465898540e+02;
    const d61: Float = 0.19985053242002433820987653617e+02;
    const d66: Float = -0.38703730874935176555105901742e+03;
    const d67: Float = -0.18917813819516756882830838328e+03;
    const d68: Float = 0.52780815920542364900561016686e+03;
    const d69: Float = -0.11573902539959630126141871134e+02;
    const d610: Float = 0.68812326946963000169666922661e+01;
    const d611: Float = -0.10006050966910838403183860980e+01;
    const d612: Float = 0.77771377980534432092869265740e+00;
    const d613: Float = -0.27782057523535084065932004339e+01;
    const d614: Float = -0.60196695231264120758267380846e+02;
    const d615: Float = 0.84320405506677161018159903784e+02;
    const d616: Float = 0.11992291136182789328035130030e+02;
    const d71: Float = -0.25693933462703749003312586129e+02;
    const d76: Float = -0.15418974869023643374053993627e+03;
    const d77: Float = -0.23152937917604549567536039109e+03;
    const d78: Float = 0.35763911791061412378285349910e+03;
    const d79: Float = 0.93405324183624310003907691704e+02;
    const d710: Float = -0.37458323136451633156875139351e+02;
    const d711: Float = 0.10409964950896230045147246184e+03;
    const d712: Float = 0.29840293426660503123344363579e+02;
    const d713: Float = -0.43533456590011143754432175058e+02;
    const d714: Float = 0.96324553959188282948394950600e+02;
    const d715: Float = -0.39177261675615439165231486172e+02;
    const d716: Float = -0.14972683625798562581422125276e+03;
}

struct ControllerRoss {
    h_next: Float,
    reject: bool,
    first_step: bool,
    err_old: Float,
    h_old: Float,
}
impl ControllerRoss {
    fn new() -> Self {
        ControllerRoss {
            h_next: 0.0,
            reject: false,
            first_step: true,
            err_old: 0.0,
            h_old: 0.0,
        }
    }
    fn success(&mut self, err: Float, h: Float) -> Option<Float> {
        const SAFE: Float = 0.9;
        const FAC1: Float = 5.0;
        const FAC2: Float = 1.0 / 6.0;
        let mut fac = (err.powf(0.25) / SAFE).clamp(FAC2, FAC1);
        let mut h_new = h / fac;
        if err <= 1.0 {
            if !self.first_step {
                let fac_pred = (self.h_old / h) * (err * err / self.err_old).powf(0.25) / SAFE;
                let fac_pred = fac_pred.clamp(FAC2, FAC1);
                fac = fac.max(fac_pred);
                h_new = h / fac;
            }
            self.first_step = false;
            self.h_old = h;
            self.err_old = err.max(0.01);
            if self.reject {
                h_new = if h >= 0.0 {
                    h_new.min(h)
                } else {
                    h_new.max(h)
                };
            }
            self.h_next = h_new;
            self.reject = false;
            None
        } else {
            self.reject = true;
            Some(h_new)
        }
    }
}

struct StepperRoss {
    data: StepperData,
    k1: MyVec<Float>,
    k2: MyVec<Float>,
    k3: MyVec<Float>,
    k4: MyVec<Float>,
    k5: MyVec<Float>,
    k6: MyVec<Float>,
    cont1: MyVec<Float>,
    cont2: MyVec<Float>,
    cont3: MyVec<Float>,
    cont4: MyVec<Float>,
    con: ControllerRoss,
}

#[allow(non_upper_case_globals)]
impl StepperRoss {
    fn prepare_step(
        &mut self,
        stat: &ProbStat,
        h: Float,
        derivative: &dyn DiffEq,
    ) -> Result<(), &'static str> {
        let mut a = -stat.jacobian.df_dy.clone();
        for i in 0..a.shape().0 {
            a[(i, i)] += 1.0 / (Self::gam * h);
        }
        let alu = a.lu();
        let y_temp = &stat.dy_dx + h * Self::d1 * &stat.jacobian.df_dx;
        self.k1 = MyVec::from_na(alu.solve(&y_temp.into_na()).ok_or("error in LU solve")?);
        let y_temp = &stat.y + Self::a21 * &self.k1;
        let dydx_new = derivative.derivative(stat.x + Self::c2 * h, &y_temp);
        let y_temp = dydx_new + h * Self::d2 * &stat.jacobian.df_dx + Self::c21 / h * &self.k1;
        self.k2 = MyVec::from_na(alu.solve(&y_temp.into_na()).ok_or("error in LU solve")?);
        let y_temp = &stat.y + (Self::a31 * &self.k1 + Self::a32 * &self.k2);
        let dydx_new = derivative.derivative(stat.x + Self::c3 * h, &y_temp);
        let y_temp = dydx_new
            + h * Self::d3 * &stat.jacobian.df_dx
            + (Self::c31 * &self.k1 + Self::c32 * &self.k2) * (1.0 / h);
        self.k3 = MyVec::from_na(alu.solve(&y_temp.into_na()).ok_or("error in LU solve")?);
        let y_temp = &stat.y + (Self::a41 * &self.k1 + Self::a42 * &self.k2 + Self::a43 * &self.k3);
        let dydx_new = derivative.derivative(stat.x + Self::c4 * h, &y_temp);
        let y_temp = dydx_new
            + h * Self::d4 * &stat.jacobian.df_dx
            + (1.0 / h) * (Self::c41 * &self.k1 + Self::c42 * &self.k2 + Self::c43 * &self.k3);
        self.k4 = MyVec::from_na(alu.solve(&y_temp.into_na()).ok_or("error in LU solve")?);
        let y_temp = &stat.y
            + Self::a51 * &self.k1
            + Self::a52 * &self.k2
            + Self::a53 * &self.k3
            + Self::a54 * &self.k4;
        let xph = stat.x + h;
        let dydx_new = derivative.derivative(xph, &y_temp);
        self.k6 = dydx_new
            + (1.0 / h)
            * (Self::c51 * &self.k1
            + Self::c52 * &self.k2
            + Self::c53 * &self.k3
            + Self::c54 * &self.k4);
        self.k5 = MyVec::from_na(
            alu.solve(&self.k6.clone().into_na())
                .ok_or("error in LU solve")?,
        );
        let y_temp = y_temp + &self.k5;
        let dydx_new = derivative.derivative(xph, &y_temp);
        self.k6 = dydx_new
            + (1.0 / h)
            * (Self::c61 * &self.k1
            + Self::c62 * &self.k2
            + Self::c63 * &self.k3
            + Self::c64 * &self.k4
            + Self::c65 * &self.k5);
        self.data.y_err = MyVec::<Float>::from_na(
            alu.solve(&self.k6.clone().into_na())
                .ok_or("error in LU solve")?,
        );
        self.data.y_out = &self.data.y_err + y_temp;
        Ok(())
    }
    fn prepare_dense(&mut self, stat: &ProbStat) {
        self.cont1 = stat.y.clone();
        self.cont2 = self.data.y_out.clone();
        self.cont3 = Self::d21 * &self.k1
            + Self::d22 * &self.k2
            + Self::d23 * &self.k3
            + Self::d24 * &self.k4
            + Self::d25 * &self.k5;
        self.cont4 = Self::d31 * &self.k1
            + Self::d32 * &self.k2
            + Self::d33 * &self.k3
            + Self::d34 * &self.k4
            + Self::d35 * &self.k5;
    }
    fn error(&self, stat: &ProbStat) -> Float {
        let mut err = 0.0;
        let n = self.data.y_out.0.len();
        for i in 0..n {
            let sk =
                self.data.a_tol + self.data.r_tol * stat.y[i].abs().max(self.data.y_out[i].abs());
            err += (self.data.y_err[i] / sk).powi(2);
        }
        (err / n as Float).sqrt()
    }
    const c2: Float = 0.386;
    const c3: Float = 0.21;
    const c4: Float = 0.63;
    const bet2p: Float = 0.0317;
    const bet3p: Float = 0.0635;
    const bet4p: Float = 0.3438;
    const d1: Float = 0.2500000000000000e+00;
    const d2: Float = -0.1043000000000000e+00;
    const d3: Float = 0.1035000000000000e+00;
    const d4: Float = -0.3620000000000023e-01;
    const a21: Float = 0.1544000000000000e+01;
    const a31: Float = 0.9466785280815826e+00;
    const a32: Float = 0.2557011698983284e+00;
    const a41: Float = 0.3314825187068521e+01;
    const a42: Float = 0.2896124015972201e+01;
    const a43: Float = 0.9986419139977817e+00;
    const a51: Float = 0.1221224509226641e+01;
    const a52: Float = 0.6019134481288629e+01;
    const a53: Float = 0.1253708332932087e+02;
    const a54: Float = -0.6878860361058950e+00;
    const c21: Float = -0.5668800000000000e+01;
    const c31: Float = -0.2430093356833875e+01;
    const c32: Float = -0.2063599157091915e+00;
    const c41: Float = -0.1073529058151375e+00;
    const c42: Float = -0.9594562251023355e+01;
    const c43: Float = -0.2047028614809616e+02;
    const c51: Float = 0.7496443313967647e+01;
    const c52: Float = -0.1024680431464352e+02;
    const c53: Float = -0.3399990352819905e+02;
    const c54: Float = 0.1170890893206160e+02;
    const c61: Float = 0.8083246795921522e+01;
    const c62: Float = -0.7981132988064893e+01;
    const c63: Float = -0.3152159432874371e+02;
    const c64: Float = 0.1631930543123136e+02;
    const c65: Float = -0.6058818238834054e+01;
    const gam: Float = 0.2500000000000000e+00;
    const d21: Float = 0.1012623508344586e+02;
    const d22: Float = -0.7487995877610167e+01;
    const d23: Float = -0.3480091861555747e+02;
    const d24: Float = -0.7992771707568823e+01;
    const d25: Float = 0.1025137723295662e+01;
    const d31: Float = -0.6762803392801253e+00;
    const d32: Float = 0.6087714651680015e+01;
    const d33: Float = 0.1643084320892478e+02;
    const d34: Float = 0.2476722511418386e+02;
    const d35: Float = -0.6594389125716872e+01;
}
impl Stepper for StepperRoss {
    fn new(a_tol: Float, r_tol: Float, dense: bool) -> Self {
        StepperRoss {
            data: StepperData::new(a_tol, r_tol, dense),
            k1: MyVec(vec![]),
            k2: MyVec(vec![]),
            k3: MyVec(vec![]),
            k4: MyVec(vec![]),
            k5: MyVec(vec![]),
            k6: MyVec(vec![]),
            cont1: MyVec(vec![]),
            cont2: MyVec(vec![]),
            cont3: MyVec(vec![]),
            cont4: MyVec(vec![]),
            con: ControllerRoss::new(),
        }
    }
    fn step(
        &mut self,
        stat: &mut ProbStat,
        h: Float,
        derivative: &dyn DiffEq,
    ) -> Result<(), &'static str> {
        let mut h = h;
        stat.jacobian = derivative.jacobian(stat.x, &stat.y);
        loop {
            self.prepare_step(stat, h, derivative)?;
            let err = self.error(stat);
            match self.con.success(err, h) {
                None => break,
                Some(h_new) => {
                    h = h_new;
                    if h.abs() < stat.x.abs() * Float::EPSILON {
                        return Err("step size underflow in StepperRoss");
                    }
                },
            }
        }
        let dydx_new = derivative.derivative(stat.x + h, &self.data.y_out);
        if self.data.dense {
            self.prepare_dense(stat);
        }
        stat.dy_dx = dydx_new;
        stat.y = self.data.y_out.clone();
        self.data.x_old = stat.x;
        stat.x += h;
        self.data.h_did = h;
        self.data.h_next = self.con.h_next;
        Ok(())
    }
    fn dense_out(&self, x: Float, h: Float) -> MyVec<Float> {
        let s = (x - self.data.x_old) / h;
        let s1 = 1.0 - s;
        &self.cont1 * s1 + s * (&self.cont2 + s1 * (&self.cont3 + s * &self.cont4))
    }
    fn data(&self) -> &StepperData {
        &self.data
    }
}
struct PrecursorData {
    beta_eff: Float,
    decay_const: Float,
}
impl PrecursorData {
    fn new(beta_eff: Float, decay_const: Float) -> Self {
        PrecursorData {
            beta_eff,
            decay_const,
        }
    }
}

struct PointReactor {
    rho: Box<dyn Fn(Float, Float) -> Float>,
    precursors: Vec<PrecursorData>,
    avg_life: Float,
    beta_eff: Float,
}

impl PointReactor {
    fn new(
        rho: Box<dyn Fn(Float, Float) -> Float>,
        precursors: Vec<PrecursorData>,
        avg_life: Float,
    ) -> Self {
        let beta_eff = precursors.iter().map(|p| p.beta_eff).sum();
        PointReactor {
            rho,
            precursors,
            avg_life,
            beta_eff,
        }
    }
    fn common_reactor(rho: Box<dyn Fn(Float, Float) -> Float>) -> Self {
        let mut precursors = vec![];
        precursors.push(PrecursorData::new(0.00021, 0.0124));
        precursors.push(PrecursorData::new(0.00142, 0.0305));
        precursors.push(PrecursorData::new(0.00127, 0.1114));
        precursors.push(PrecursorData::new(0.00257, 0.3014));
        precursors.push(PrecursorData::new(0.00075, 1.1363));
        precursors.push(PrecursorData::new(0.00027, 3.0137));
        let avg_life = 0.01;
        PointReactor::new(rho, precursors, avg_life)
    }
    fn init_balance(&self) -> MyVec<Float> {
        let mut y0 = MyVec(vec![]);
        y0.0.push(1.0);
        for p in &self.precursors {
            let c = p.beta_eff / (self.avg_life * p.decay_const);
            y0.0.push(c);
        }
        y0
    }
    fn deriv(&self, t: Float, y: &MyVec<Float>) -> MyVec<Float> {
        let rho = (*self.rho)(t, y[0]);
        let k_eff = 1.0 / (1.0 - rho);
        let gen_age = self.avg_life / k_eff;
        let mut res = MyVec(vec![]);
        let mut dndt = rho / gen_age * y[0];
        for i in 0..self.precursors.len() {
            dndt += self.precursors[i].decay_const * y[i + 1]
                - self.precursors[i].beta_eff / gen_age * y[0];
        }
        res.0.push(dndt);
        for i in 0..self.precursors.len() {
            res.0.push(
                self.precursors[i].beta_eff / gen_age * y[0]
                    - self.precursors[i].decay_const * y[i + 1],
            );
        }
        res
    }
}
impl DiffEq for PointReactor {
    fn derivative(&self, t: Float, y: &MyVec<Float>) -> MyVec<Float> {
        self.deriv(t, y)
    }
    fn jacobian(&self, t: Float, y: &MyVec<Float>) -> Jacobian {
        let rho = (self.rho)(t, y[0]);
        let drho_dt = deriv_ridders(&(|t| (self.rho)(t, y[0])), t, 1e-6);
        let drho_dn = deriv_ridders(&(|n| (self.rho)(t, n)), y[0], 1e-6);
        let mut df_drho = MyVec(vec![
            y[0] * (1.0 - self.beta_eff) / self.avg_life / ((1.0 - rho) * (1.0 - rho)),
        ]);
        for p in &self.precursors {
            df_drho
                .0
                .push(p.beta_eff / self.avg_life / ((1.0 - rho) * (1.0 - rho)));
        }
        let df_dt = drho_dt * &df_drho;
        let mut df_dy: na::DMatrix<Float> = na::DMatrix::zeros(y.0.len(), y.0.len());
        df_dy[(0, 0)] = drho_dn * df_drho[0] + (rho - self.beta_eff) / self.avg_life / (1.0 - rho);
        for i in 0..self.precursors.len() {
            let p = &self.precursors[i];
            df_dy[(i + 1, 0)] = drho_dn * df_drho[i + 1] + p.beta_eff / self.avg_life / (1.0 - rho);
            df_dy[(0, i + 1)] = p.decay_const;
            df_dy[(i + 1, i + 1)] = -p.decay_const;
        }
        Jacobian {
            df_dx: df_dt,
            df_dy,
        }
    }
}

pub fn test_main() {
    let res = calc(
        0.0,
        10.0,
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        0.01,
        vec![0.0124, 0.0305, 0.1114, 0.3014, 1.1363, 3.0137],
        vec![0.00021, 0.00142, 0.00127, 0.00257, 0.00075, 0.00027],
        "0.0005 * t".to_string(),
        100,
        true,
    );
    println!("{:#?}", res);
}

use boa_engine::{Context, Source};
fn eval(s: String) -> f64 {
    let mut context = Context::default();
    let source = Source::from_bytes(&s);
    let val = context.eval(source).unwrap();
    val.to_number(&mut context).unwrap()
}

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn calc(x1: Float, x2: Float, y0_vec: Vec<Float>, l: Float, lambdas: Vec<Float>, betas: Vec<Float>, rho_str: String, n_save: i32, implicit: bool) -> Vec<Float> {
    let a_tol = 1e-4;
    let r_tol = a_tol;
    let h1 = ((x2 - x1) / n_save as Float).abs().min(0.01);
    let h_min = 0.0;
    let rho = Box::new(move |_t: Float, _n: Float| eval(format!("(function(t,n){{ return {rho_str} }})({_t},{_n})")));
    let mut precursors = vec![];
    let mut res = vec![];
    for i in 0..lambdas.len() {
        precursors.push(PrecursorData::new(betas[i], lambdas[i]));
    }
    let reactor = PointReactor::new(rho, precursors, l);
    let y_start = MyVec(y0_vec);
    let output = Output::new(n_save);
    if implicit {
        let mut ode = Odeint::<StepperRoss>::new(
            y_start.clone(),
            x1,
            x2,
            a_tol,
            r_tol,
            h1,
            h_min,
            output,
            Box::new(reactor),
        );
        ode.integrate().unwrap();
        let y_out = ode.y_out();
        let x_out = ode.x_out();
        for i in 0..y_out.len() {
            res.push(x_out[i]);
            res.push(y_out[i][0]);
        }
    } else {
        let mut ode = Odeint::<StepperDopr853>::new(
            y_start.clone(),
            x1,
            x2,
            a_tol,
            r_tol,
            h1,
            h_min,
            output,
            Box::new(reactor),
        );
        ode.integrate().unwrap();
        let y_out = ode.y_out();
        let x_out = ode.x_out();
        for i in 0..y_out.len() {
            res.push(x_out[i]);
            res.push(y_out[i][0]);
        }
    }
    res
}
