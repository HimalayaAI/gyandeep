use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
fn vectors_to_pg_literals(embeddings: Vec<Vec<f64>>) -> Vec<String> {
    embeddings
        .par_iter()
        .map(|emb| {
            let mut s = String::with_capacity(emb.len() * 10 + 2);
            s.push('[');
            for (i, v) in emb.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                use std::fmt::Write;
                write!(s, "{:.6}", v).unwrap();
            }
            s.push(']');
            s
        })
        .collect()
}

#[pymodule]
fn gyandeep_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(vectors_to_pg_literals, m)?)?;
    Ok(())
}
