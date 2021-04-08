use std::{borrow::Cow, fmt::Debug};

use crate::util::{relu, sigmoid_approx, tansig_approx, zip3};

const MAX_NEURONS: usize = 128;

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Tanh = 0,
    Sigmoid = 1,
    Relu = 2,
}

const WEIGHTS_SCALE: f32 = 1.0 / 256.0;

#[derive(Clone)]
pub struct DenseLayer {
    /// An array of length `nb_neurons`.
    pub bias: Cow<'static, [i8]>,
    /// An array of length `nb_inputs * nb_neurons`.
    pub input_weights: Cow<'static, [i8]>,
    pub nb_inputs: usize,
    pub nb_neurons: usize,
    pub activation: Activation,
}

#[derive(Clone)]
pub struct GruLayer {
    /// An array of length `3 * nb_neurons`.
    pub bias: Cow<'static, [i8]>,
    /// An array of length `3 * nb_inputs * nb_neurons`.
    pub input_weights: Cow<'static, [i8]>,
    /// An array of length `3 * nb_neurons^2`.
    pub recurrent_weights: Cow<'static, [i8]>,
    pub nb_inputs: usize,
    pub nb_neurons: usize,
    pub activation: Activation,
}

/// An `RnnModel` contains all the model parameters for the denoising algorithm.
/// `nnnoiseless` has a built-in model that should work for most purposes, but if you have
/// specific needs then you might benefit from training a custom model. Scripts for model
/// training are available as part of [`RNNoise`]; once the model is trained, you can load it
/// here.
///
/// [`RNNoise`]: https://github.com/xiph/rnnoise
#[derive(Clone)]
pub struct RnnModel {
    pub(crate) input_dense: DenseLayer,
    pub(crate) vad_gru: GruLayer,
    pub(crate) noise_gru: GruLayer,
    pub(crate) denoise_gru: GruLayer,
    pub(crate) denoise_output: DenseLayer,
    pub(crate) vad_output: DenseLayer,
}

#[derive(Clone)]
pub struct RnnState<'model> {
    model: Cow<'model, RnnModel>,
    vad_gru_state: Vec<f32>,
    noise_gru_state: Vec<f32>,
    denoise_gru_state: Vec<f32>,
}

#[derive(Debug)]
pub enum ReadModelError {
    IntParse(std::num::ParseIntError),
    Io(std::io::Error),
    CorruptFile,
}

impl From<std::num::ParseIntError> for ReadModelError {
    fn from(e: std::num::ParseIntError) -> ReadModelError {
        ReadModelError::IntParse(e)
    }
}

impl From<std::io::Error> for ReadModelError {
    fn from(e: std::io::Error) -> ReadModelError {
        ReadModelError::Io(e)
    }
}

impl std::fmt::Display for ReadModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ReadModelError::IntParse(e) => write!(f, "error parsing i8: {}", e),
            ReadModelError::Io(e) => write!(f, "{}", e),
            ReadModelError::CorruptFile => write!(f, "model file corrupted"),
        }
    }
}

impl std::error::Error for ReadModelError {}

impl RnnModel {
    /// Reads an `RnnModel` from a `std::io::Read`.
    ///
    /// The file format of an `RnnModel` is not specified anywhere; it should have been generated
    /// from the `dump_rnn.py` script in the [`RNNoise` repository].
    ///
    /// [`RNNoise`]: https://github.com/xiph/rnnoise
    pub fn from_read<R: std::io::Read>(mut r: R) -> Result<RnnModel, ReadModelError> {
        let mut data = String::new();
        r.read_to_string(&mut data)?;

        let header = "rnnoise-nu model file version 1";
        if !data.starts_with(header) {
            return Err(ReadModelError::CorruptFile);
        }

        let data = &data[header.len()..];

        // After the header, the model file consists of a giant list of whitespace-separated
        // integers. This list consists of the data for the various layers, one after the other.
        // The format for a single dense layer is:
        // <nb_neurons> <nb_inputs> <activation>
        // <weights...>
        // <bias...>
        // where each of the <?> terms represents a single integer, and each of the <?...> terms
        // represents an array of integers of the appropriate length (as determined by nb_neurons
        // and nb_inputs).
        //
        // The format for GRU layer is similar, but with some extra recurrent weights.
        let mut ints = data
            .split_whitespace()
            .map(|s| s.parse::<i8>().map_err(|e| e.into()));

        let mut read_int = || {
            ints.next()
                .ok_or(ReadModelError::CorruptFile)
                .and_then(|x| x)
        };

        fn read_array(
            int: &mut impl FnMut() -> Result<i8, ReadModelError>,
            rows: usize,
            cols: usize,
        ) -> Result<Vec<i8>, ReadModelError> {
            // The array is stored in column-major format, and we want to convert it to row-major.
            let mut ret = vec![0; rows * cols];
            for j in 0..cols {
                for i in 0..rows {
                    ret[i * cols + j] = int()?;
                }
            }
            Ok(ret)
        }

        fn act(x: i8) -> Result<Activation, ReadModelError> {
            match x {
                0 => Ok(Activation::Tanh),
                1 => Ok(Activation::Sigmoid),
                2 => Ok(Activation::Relu),
                _ => Err(ReadModelError::CorruptFile),
            }
        }

        fn unsigned(x: i8) -> Result<usize, ReadModelError> {
            if x >= 0 {
                Ok(x as usize)
            } else {
                Err(ReadModelError::CorruptFile)
            }
        }

        fn read_dense(
            int: &mut impl FnMut() -> Result<i8, ReadModelError>,
        ) -> Result<DenseLayer, ReadModelError> {
            let nb_inputs = unsigned(int()?)?;
            let nb_neurons = unsigned(int()?)?;
            let activation = act(int()?)?;
            let input_weights = read_array(int, nb_neurons, nb_inputs)?;
            let bias = read_array(int, 1, nb_neurons)?;

            Ok(DenseLayer {
                nb_inputs,
                nb_neurons,
                input_weights: input_weights.into(),
                bias: bias.into(),
                activation,
            })
        }

        fn read_gru(
            int: &mut impl FnMut() -> Result<i8, ReadModelError>,
        ) -> Result<GruLayer, ReadModelError> {
            let nb_inputs = unsigned(int()?)?;
            let nb_neurons = unsigned(int()?)?;
            let activation = act(int()?)?;
            let input_weights = read_array(int, 3 * nb_neurons, nb_inputs)?;
            let recurrent_weights = read_array(int, 3 * nb_neurons, nb_neurons)?;
            let bias = read_array(int, 1, 3 * nb_neurons)?;

            Ok(GruLayer {
                nb_inputs,
                nb_neurons,
                input_weights: input_weights.into(),
                recurrent_weights: recurrent_weights.into(),
                bias: bias.into(),
                activation,
            })
        }

        Ok(RnnModel {
            input_dense: read_dense(&mut read_int)?,
            vad_gru: read_gru(&mut read_int)?,
            noise_gru: read_gru(&mut read_int)?,
            denoise_gru: read_gru(&mut read_int)?,
            denoise_output: read_dense(&mut read_int)?,
            vad_output: read_dense(&mut read_int)?,
        })
    }
}

impl Default for RnnModel {
    fn default() -> RnnModel {
        use crate::model::*;

        let input_dense = DenseLayer {
            bias: Cow::Borrowed(&INPUT_DENSE_BIAS[..]),
            input_weights: Cow::Borrowed(&INPUT_DENSE_WEIGHTS[..]),
            nb_inputs: 42,
            nb_neurons: 24,
            activation: Activation::Tanh,
        };

        let vad_gru = GruLayer {
            bias: Cow::Borrowed(&VAD_GRU_BIAS[..]),
            input_weights: Cow::Borrowed(&VAD_GRU_WEIGHTS[..]),
            recurrent_weights: Cow::Borrowed(&VAD_GRU_RECURRENT_WEIGHTS[..]),
            nb_inputs: 24,
            nb_neurons: 24,
            activation: Activation::Relu,
        };

        let noise_gru = GruLayer {
            bias: Cow::Borrowed(&NOISE_GRU_BIAS[..]),
            input_weights: Cow::Borrowed(&NOISE_GRU_WEIGHTS[..]),
            recurrent_weights: Cow::Borrowed(&NOISE_GRU_RECURRENT_WEIGHTS[..]),
            nb_inputs: 90,
            nb_neurons: 48,
            activation: Activation::Relu,
        };

        let denoise_gru = GruLayer {
            bias: Cow::Borrowed(&DENOISE_GRU_BIAS[..]),
            input_weights: Cow::Borrowed(&DENOISE_GRU_WEIGHTS[..]),
            recurrent_weights: Cow::Borrowed(&DENOISE_GRU_RECURRENT_WEIGHTS[..]),
            nb_inputs: 114,
            nb_neurons: 96,
            activation: Activation::Relu,
        };

        let denoise_output = DenseLayer {
            bias: Cow::Borrowed(&DENOISE_OUTPUT_BIAS[..]),
            input_weights: Cow::Borrowed(&DENOISE_OUTPUT_WEIGHTS[..]),
            nb_inputs: 96,
            nb_neurons: 22,
            activation: Activation::Sigmoid,
        };

        let vad_output = DenseLayer {
            bias: Cow::Borrowed(&VAD_OUTPUT_BIAS[..]),
            input_weights: Cow::Borrowed(&VAD_OUTPUT_WEIGHTS[..]),
            nb_inputs: 24,
            nb_neurons: 1,
            activation: Activation::Sigmoid,
        };

        RnnModel {
            input_dense,
            vad_gru,
            noise_gru,
            denoise_gru,
            denoise_output,
            vad_output,
        }
    }
}

impl DenseLayer {
    fn matrix(&self) -> SubMatrix {
        SubMatrix {
            data: self.input_weights.as_ref(),
            stride: self.nb_neurons,
            offset: 0,
        }
    }

    fn compute(&self, output: &mut [f32], input: &[f32]) {
        copy_i8(output, &self.bias[..]);
        self.matrix().sgemv(output, input);

        match self.activation {
            Activation::Sigmoid => {
                for out in output.iter_mut() {
                    *out = sigmoid_approx(*out * WEIGHTS_SCALE);
                }
            }
            Activation::Tanh => {
                for out in output.iter_mut() {
                    *out = tansig_approx(*out * WEIGHTS_SCALE);
                }
            }
            Activation::Relu => {
                for out in output.iter_mut() {
                    *out = relu(*out * WEIGHTS_SCALE);
                }
            }
        }
    }
}

impl GruLayer {
    fn input_submatrix(&self, offset: usize) -> SubMatrix {
        SubMatrix {
            data: self.input_weights.as_ref(),
            stride: self.nb_neurons * 3,
            offset,
        }
    }

    fn rec_submatrix(&self, offset: usize) -> SubMatrix {
        SubMatrix {
            data: self.recurrent_weights.as_ref(),
            stride: self.nb_neurons * 3,
            offset,
        }
    }

    fn compute(&self, state: &mut [f32], input: &[f32]) {
        let mut z = [0.0; MAX_NEURONS];
        let mut r = [0.0; MAX_NEURONS];
        let mut h = [0.0; MAX_NEURONS];
        let n = self.nb_neurons;

        // Compute update gate.
        copy_i8(&mut z[0..n], &self.bias[0..n]);
        self.input_submatrix(0).sgemv(&mut z[0..n], input);
        self.rec_submatrix(0).sgemv(&mut z[0..n], &state[..]);
        for z in z[0..n].iter_mut() {
            *z = sigmoid_approx(WEIGHTS_SCALE * *z);
        }

        // Compute reset gate.
        copy_i8(&mut r[0..n], &self.bias[n..(2 * n)]);
        self.input_submatrix(n).sgemv(&mut r[0..n], input);
        self.rec_submatrix(n).sgemv(&mut r[0..n], &state[..]);
        for (out, &s) in r[0..n].iter_mut().zip(&state[..]) {
            *out = s * sigmoid_approx(WEIGHTS_SCALE * *out);
        }

        // Compute output.
        copy_i8(&mut h[0..n], &self.bias[(2 * n)..]);
        self.input_submatrix(2 * n).sgemv(&mut h[0..n], input);
        self.rec_submatrix(2 * n).sgemv(&mut h[0..n], &r[0..n]);

        for (s, &z, &h) in zip3(state, &z[0..n], &h[0..n]) {
            let h = match self.activation {
                Activation::Sigmoid => sigmoid_approx(WEIGHTS_SCALE * h),
                Activation::Tanh => tansig_approx(WEIGHTS_SCALE * h),
                Activation::Relu => relu(WEIGHTS_SCALE * h),
            };
            *s = z * *s + (1.0 - z) * h;
        }
    }
}

impl<'model> RnnState<'model> {
    pub(crate) fn new(model: Cow<'model, RnnModel>) -> RnnState<'model> {
        let vad_gru_state = vec![0.0f32; model.vad_gru.nb_neurons];
        let noise_gru_state = vec![0.0f32; model.noise_gru.nb_neurons];
        let denoise_gru_state = vec![0.0f32; model.denoise_gru.nb_neurons];
        RnnState {
            model,
            vad_gru_state,
            noise_gru_state,
            denoise_gru_state,
        }
    }

    pub fn compute(&mut self, gains: &mut [f32], vad: &mut [f32], input: &[f32]) {
        assert_eq!(input.len(), INPUT_SIZE);

        let mut buf = [0.0; MAX_NEURONS * 3];
        let mut denoise_buf = [0.0; MAX_NEURONS * 3];
        let model = &self.model;

        let vad_gru_state = &mut self.vad_gru_state[..];
        let noise_gru_state = &mut self.noise_gru_state[..];
        let denoise_gru_state = &mut self.denoise_gru_state[..];
        model
            .input_dense
            .compute(&mut buf[0..model.input_dense.nb_neurons], input);
        model
            .vad_gru
            .compute(vad_gru_state, &buf[0..model.input_dense.nb_neurons]);
        model.vad_output.compute(vad, vad_gru_state);

        copy(&mut buf[model.input_dense.nb_neurons..], vad_gru_state);
        copy(
            &mut buf[(model.input_dense.nb_neurons + model.vad_gru.nb_neurons)..],
            input,
        );
        model.noise_gru.compute(noise_gru_state, &buf);

        copy(&mut denoise_buf, vad_gru_state);
        copy(
            &mut denoise_buf[model.vad_gru.nb_neurons..],
            noise_gru_state,
        );
        copy(
            &mut denoise_buf[(model.vad_gru.nb_neurons + model.noise_gru.nb_neurons)..],
            input,
        );
        model.denoise_gru.compute(denoise_gru_state, &denoise_buf);
        model.denoise_output.compute(gains, denoise_gru_state);
    }
}

const INPUT_SIZE: usize = 42;

fn copy(dst: &mut [f32], src: &[f32]) {
    for (x, y) in dst.iter_mut().zip(src) {
        *x = *y;
    }
}

fn copy_i8(dst: &mut [f32], src: &[i8]) {
    for (x, y) in dst.iter_mut().zip(src) {
        *x = *y as f32;
    }
}

use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};

/// The requirements for a type to be a Matrix Cell. Numeric types fulfill these
/// requirements, and many of them can be derived as needed
pub trait MatrixCell<T>:
    Add<Output = T> + Mul<Output = T> + Sub<Output = T> + AddAssign + SubAssign + Copy + From<i8>
{
}
impl<
        T: Add<Output = T>
            + Mul<Output = T>
            + Sub<Output = T>
            + AddAssign
            + SubAssign
            + Copy
            + From<i8>,
    > MatrixCell<T> for T
{
}

/// Uses const generics to represent a mathematical matrix
#[derive(Copy, Clone)]
pub struct Matrix<'a, const L: usize> {
    pub inner: &'a [i8; L],
}

impl<'a, const L: usize> Matrix<'a, L> {
    pub fn new(inner: &'a [i8; L]) -> Self {
        Matrix { inner }
    }
    //pub fn from_slice(slc: &'a [i8]) -> Self {
    //    assert!(slc.len() == C * R);
    //    unsafe {
    //        let inner: &[i8; R * C] = &from_raw_parts(slc.as_ptr().cast(), 1)[0];
    //        Matrix { inner }
    //    }
    //}

    //pub fn sgemv<T: MatrixCell<T>>(&self, output: &mut [T; R], input: &[T; C])
    //    where [(), R * C]:
    //{
    //    for c in 0..C {
    //        let mut r = 0;
    //        while r < R {
    //            output[r] += T::from(self.inner[c * R + r]) * input[c];
    //            r += 1;
    //        }
    //    }
    //}

    //pub fn sgemv_colwise_step<T: MatrixCell<T>, const STEP: usize>(
    //    &self,
    //    output: &mut [T; R],
    //    input: &[T; C],
    //) {
    //    assert_eq!(R % STEP, 0);
    //    for (col, &input) in self.inner.chunks_exact(R).zip(input) {
    //        for (col_chunk, out_chunk) in col.chunks_exact(STEP).zip(output.chunks_exact_mut(STEP))
    //        {
    //            for (&x, out) in col_chunk[..].iter().zip(&mut out_chunk[..]) {
    //                *out += T::from(x) * input;
    //            }
    //        }
    //    }
    //}

    //pub fn sgemv_colwise_step<T: MatrixCell<T>, const STEP: usize>(&self, output: &mut [T; R], input: & [T; C]) {
    //    assert_eq!(R % STEP, 0);
    //    for (col, &input) in self.inner.chunks_exact(R).zip(input) {
    //        for (col_chunk, out_chunk) in col.chunks_exact(STEP).zip(output.chunks_exact_mut(STEP)) {
    //            for (&x, out) in col_chunk[..].iter().zip(&mut out_chunk[..]) {
    //                *out += T::from(x) * input;
    //            }
    //        }
    //    }
    //}
}

pub struct SubMatrix<'a> {
    pub data: &'a [i8],
    pub stride: usize,
    pub offset: usize,
}

impl<'a> SubMatrix<'a> {
    pub fn sgemv(&self, output: &mut [f32], input: &[f32]) {
        for (col, input) in self.data.chunks_exact(self.stride).zip(input) {
            for (&x, out) in col[self.offset..].iter().zip(&mut output[..]) {
                *out += x as f32 * input;
            }
        }
    }

    pub fn sgemv_colwise_step(&self, output: &mut [f32], input: &[f32]) {
        const STEP: usize = 24;
        assert_eq!(self.stride % STEP, 0);
        for (col, input) in self.data.chunks_exact(self.stride).zip(input) {
            for (col_chunk, out_chunk) in col[self.offset..]
                .chunks_exact(STEP)
                .zip(output.chunks_exact_mut(STEP))
            {
                for (&x, out) in col_chunk.iter().zip(&mut out_chunk[..]) {
                    *out += x as f32 * input;
                }
            }
        }
    }

    pub fn sgemv_rowwise(&self, output: &mut [f32], input: &[f32]) {
        const STEP: usize = 24;
        let w_size = self.data[self.offset..].len() - self.stride + STEP;
        for (row_offsets, out) in self.data[self.offset..]
            .windows(w_size)
            .zip(output.array_chunks_mut::<STEP>())
        {
            for (w, inp) in row_offsets
                .array_chunks::<STEP>()
                .step_by(self.stride / STEP)
                .zip(input)
            {
                for (&x, y) in w.iter().zip(&mut out[..]) {
                    *y += x as f32 * inp;
                }
                //out[0] += w[0] as f32 * inp;
                //out[1] += w[1] as f32 * inp;
                //out[2] += w[2] as f32 * inp;
                //out[3] += w[3] as f32 * inp;
                //out[4] += w[4] as f32 * inp;
                //out[5] += w[5] as f32 * inp;
                //out[6] += w[6] as f32 * inp;
                //out[7] += w[7] as f32 * inp;

                //out[8] += w[8] as f32 * inp;
                //out[9] += w[9] as f32 * inp;
                //out[10] += w[10] as f32 * inp;
                //out[11] += w[11] as f32 * inp;
                //out[12] += w[12] as f32 * inp;
                //out[13] += w[13] as f32 * inp;
                //out[14] += w[14] as f32 * inp;
                //out[15] += w[15] as f32 * inp;

                //out[16] += w[16] as f32 * inp;
                //out[17] += w[17] as f32 * inp;
                //out[18] += w[18] as f32 * inp;
                //out[19] += w[19] as f32 * inp;
                //out[20] += w[20] as f32 * inp;
                //out[21] += w[21] as f32 * inp;
                //out[22] += w[22] as f32 * inp;
                //out[23] += w[23] as f32 * inp;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    use rand::prelude::*;
    use rand_chacha;

    fn generate_vec(len: usize) -> Vec<f32> {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0u64);
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len {
            vec.push(rng.gen::<f32>());
        }
        return vec;
    }

    #[test]
    fn compare_sgemv() {
        const NB_INPUTS: usize = 42;
        const NB_NEURONS: usize = 24;
        let weights = model::INPUT_DENSE_WEIGHTS;
        let input = generate_vec(NB_INPUTS);
        //let mut out_sgemv = [0.0; NB_NEURONS];
        //let mut out_sgemv_rowwise = [0.0; NB_NEURONS];
        let mut out_sgemv: Vec<f32> = (0..NB_NEURONS).map(|x| x as f32).collect();
        let mut out_sgemv_other: Vec<f32> = (0..NB_NEURONS).map(|x| x as f32).collect();
        let m = SubMatrix {
            data: weights.as_ref(),
            stride: NB_NEURONS,
            offset: 0,
        };
        let tmp = &weights[..NB_NEURONS];
        println!("w ..NB_NEURONS: {:?}", tmp);
        m.sgemv(&mut out_sgemv, &input[..]);
        //let mut out: [f32; NB_NEURONS] =
        //    unsafe { out_sgemv_other.as_chunks_unchecked_mut::<NB_NEURONS>()[0] };
        //let inp: [f32; NB_INPUTS] = unsafe { input.as_chunks_unchecked::<NB_INPUTS>()[0] };
        //let m_const = Matrix::<NB_NEURONS, NB_INPUTS>::new(&weights);
        //println!("{:?}", m_const.inner[0]);
        //m_const.sgemv_colwise_step::<f32, 24>(&mut out, &inp);
        //m.sgemv_colwise_step(&mut out_sgemv_other, &input[..]);
        for (a, b) in out_sgemv.iter().zip(out_sgemv_other.iter()) {
            println!("a: {:?}, b: {:?}", a, b);
            assert!((a - b).abs() < 1e-10);
        }
    }
}
