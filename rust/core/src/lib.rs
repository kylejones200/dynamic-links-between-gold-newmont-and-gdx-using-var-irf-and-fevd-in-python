//! VAR(1) impulse response (IRF) for a single equation.

pub fn var1_irf(phi: f64, shock: f64, steps: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(steps);
    let mut y = shock;
    for _ in 0..steps {
        out.push(y);
        y *= phi;
    }
    out
}
