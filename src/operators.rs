use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    // todo!("实现 rms_norm，计算前做一些必要的检查会帮助你后续调试")
    assert_eq!(x.shape(), y.shape());
    let last_dim = x.shape()[x.shape().len() - 1];
    assert_eq!(w.size(), last_dim);

    let x_data = x.data();
    let w_data = w.data();
    let y_data = unsafe { y.data_mut() };

    // Calculate the number of vectors in the input tensor
    let num_vectors = x.size() / last_dim;
    
    for i in 0..num_vectors {
        let offset = i * last_dim;
        let slice = &x_data[offset..offset + last_dim];
        
        // Calculate mean square
        let mean_sq = slice.iter().map(|&v| v * v).sum::<f32>() / last_dim as f32;
        let rms = (mean_sq + epsilon).sqrt();
        
        // Normalize and scale by weights
        for j in 0..last_dim {
            y_data[offset + j] = (x_data[offset + j] / rms) * w_data[j];
        }
    }
}

// y = silu(x) * y
// hint: this is an element-wise operation
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(len == x.size());

    let _y = unsafe { y.data_mut() };
    let _x = x.data();

    // todo!("实现 silu，这里给了一些前期准备工作的提示，你可以参考")

    for i in 0..len {
        let sigmoid = 1.0 / (1.0 + (-_x[i]).exp());
        let silu = _x[i] * sigmoid;
        _y[i] *= silu;
    }
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    // todo!("实现 matmul_transb，计算前做一些必要的检查会帮助你后续调试");
    use std::iter::repeat;
    
    let a_shape = a.shape();
    let b_shape = b.shape();
    let c_shape = c.shape().to_vec();
    
    assert!(a_shape.len() >= 2 && b_shape.len() >= 2, "Inputs must be at least 2D");
    assert_eq!(a_shape.last().unwrap(), b_shape.last().unwrap(), "Inner dimensions must match");
    assert_eq!(c_shape[c_shape.len() - 2], a_shape[a_shape.len() - 2]);
    assert_eq!(c_shape[c_shape.len() - 1], b_shape[b_shape.len() - 2]);
    
    let mut a_strides: Vec<usize> = vec![1];
    let mut b_strides: Vec<usize> = vec![1];
    let mut c_strides: Vec<usize> = vec![1];
    
    for i in (0..a_shape.len() - 1).rev() {
        a_strides.insert(0, a_strides[0] * a_shape[i + 1]);
    }
    for i in (0..b_shape.len() - 1).rev() {
        b_strides.insert(0, b_strides[0] * b_shape[i + 1]);
    }
    for i in (0..c_shape.len() - 1).rev() {
        c_strides.insert(0, c_strides[0] * c_shape[i + 1]);
    }
    
    let a_data = a.data();
    let b_data = b.data();
    let c_data = unsafe { c.data_mut() };
    
    let batch_dims = c_shape[..c_shape.len() - 2]
        .iter()
        .zip(a_shape[..a_shape.len() - 2].iter().chain(repeat(&1)))
        .zip(b_shape[..b_shape.len() - 2].iter().chain(repeat(&1)))
        .map(|((&c_dim, &a_dim), &b_dim)| c_dim.max(a_dim).max(b_dim))
        .collect::<Vec<_>>();
    
    let m = a_shape[a_shape.len() - 2];
    let k = a_shape[a_shape.len() - 1];
    let n = b_shape[b_shape.len() - 2];
    
    for batch_idx in 0..batch_dims.iter().product::<usize>() {
        let a_batch_offset = batch_idx % a_shape[..a_shape.len() - 2].iter().product::<usize>();
        let b_batch_offset = batch_idx % b_shape[..b_shape.len() - 2].iter().product::<usize>();
        let c_batch_offset = batch_idx;
        
        for j in 0..n {
            for i in 0..m {
                let a_row = &a_data[(a_batch_offset * m + i) * k..(a_batch_offset * m + i + 1) * k];
                let b_row = &b_data[(b_batch_offset * n + j) * k..(b_batch_offset * n + j + 1) * k];
                
                let mut sum = 0.0;
                let mut l = 0;
                while l + 3 < k {
                    sum += a_row[l] * b_row[l]
                         + a_row[l+1] * b_row[l+1]
                         + a_row[l+2] * b_row[l+2]
                         + a_row[l+3] * b_row[l+3];
                    l += 4;
                }
                for l in l..k {
                    sum += a_row[l] * b_row[l];
                }
                
                let idx = (c_batch_offset * m + i) * n + j;
                c_data[idx] = f32::mul_add(alpha, sum, beta * c_data[idx]);
            }
        }
    }
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    swiglu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}
