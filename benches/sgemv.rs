use criterion::{criterion_group, criterion_main, Criterion};
use nnnoiseless::{Matrix, SubMatrix};

pub fn bench_sgemv(c: &mut Criterion) {
    let weights = nnnoiseless::INPUT_DENSE_WEIGHTS;
    const NB_INPUTS: usize = 42;
    const NB_NEURONS: usize = 24;
    let input = [1.0; NB_INPUTS];
    let mut output = [0.0; NB_NEURONS];
    let m = SubMatrix {
        data: weights.as_ref(),
        stride: NB_NEURONS,
        offset: 0,
    };
    c.bench_function("nnnoiseless segmv/normal", |b| {
        b.iter(|| {
            m.sgemv(&mut output, &input);
        })
    });
}

pub fn bench_sgemv_step(c: &mut Criterion) {
    let weights = nnnoiseless::INPUT_DENSE_WEIGHTS;
    const NB_INPUTS: usize = 42;
    const NB_NEURONS: usize = 24;
    let input = [1.0; NB_INPUTS];
    let mut output = [0.0; NB_NEURONS];
    //let m_const = Matrix::<NB_NEURONS, NB_INPUTS>::from_slice(&weights);
    c.bench_function("nnnoiseless segmv/row", |b| {
        b.iter(|| {
            let m = SubMatrix {
                data: weights.as_ref(),
                stride: NB_NEURONS,
                offset: 0,
            };
            m.sgemv_colwise_step(&mut output, &input);
            //m_const.sgemv::<f32, 24>(&mut output, &input);
        })
    });
}

criterion_group!(benches, bench_sgemv_step, bench_sgemv);
criterion_main!(benches);
