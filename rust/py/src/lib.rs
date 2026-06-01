use dynamic_links_between_gold_newmont_and_gdx_using_var_irf_and_fevd_in_python_core::var1_irf;
use numpy::{PyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
fn var1_irf_py<'py>(py: Python<'py>, phi: f64, shock: f64, steps: usize) -> Bound<'py, PyArray1<f64>> {
    var1_irf(phi, shock, steps).into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (phi, shock, steps, iterations=5_000))]
fn bench_kernel_py(phi: f64, shock: f64, steps: usize, iterations: usize) -> PyResult<f64> {
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = var1_irf(phi, shock, steps);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn dynamic_links_between_gold_newmont_and_gdx_using_var_irf_and_fevd_in_python_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(var1_irf_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
