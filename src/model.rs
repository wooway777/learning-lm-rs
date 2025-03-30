use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;
pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;
    
        // 预分配缓冲区
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores = Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
    
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);
    
        for layer in 0..self.n_layers {
            // 确保形状正确
            hidden_states.reshape(&vec![seq_len, self.d]);
            
            // RMS Norm
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );
        
            // QKV投影
            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]);
            
            // 处理KV缓存的生命周期问题
            let k_cache = &mut cache.k_cache(layer, past_seq_len);
            let k = k_cache.reshape(&vec![seq_len, self.n_kv_h * self.dqkv]);
            
            let v_cache = &mut cache.v_cache(layer, past_seq_len);
            let v = v_cache.reshape(&vec![seq_len, self.n_kv_h * self.dqkv]);
            
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            
            // RoPE
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
    
            // Self-Attention
            let full_k = &mut cache.k_cache(layer, 0);
            let full_v = &mut cache.v_cache(layer, 0);
            
            self_attention(
                &mut hidden_states,
                &mut att_scores,
                q,
                full_k,
                full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv,
            );
    
            // 输出投影
            OP::matmul_transb(
                &mut residual,
                1.0,
                &hidden_states,
                &self.params.wo[layer],
                1.0,
            );
    
            // MLP
            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );
        }
    
        // 最终处理
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![1, self.d]);
    
        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,  // 修正为使用rms_out_w
            self.eps,
        );
    
        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);
    
        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32>{
        // let mut result = Vec::<u32>::new();
        
        // todo!("实现文本生成");
        
        // result
        let mut result = Vec::with_capacity(token_ids.len() + max_len);
        result.extend_from_slice(token_ids);
        
        let mut cache = self.new_cache();
        let mut input = Tensor::<u32>::new(token_ids.to_vec(), &vec![token_ids.len()]);
        
        for _ in 0..max_len {
            // Forward pass with existing cache
            let logits = self.forward(&input, &mut cache);
            
            // Sample next token using provided random_sample function
            let next_token = OP::random_sample(&logits, top_p, top_k, temperature);
            result.push(next_token);
            
            // Early stopping if EOS token is generated
            if next_token == self.eos_token_id {
                break;
            }
            
            // Prepare next input (single token)
            input = Tensor::<u32>::new(vec![next_token], &vec![1]);
        }
        
        result
    }
}

fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    // Step 1: Compute scaling factor for dot products
    let scale = 1.0 / (dqkv as f32).sqrt();
    let q_data = q.data();
    let k_data = k.data();
    let v_data = v.data();

    // Compute attention scores: Q @ K^T * scale
    {
        let att_scores_data = unsafe { att_scores.data_mut() };
        for kv_head in 0..n_kv_h {
            for group in 0..n_groups {
                for query_pos in 0..seq_len {
                    let q_offset = query_pos * n_kv_h * n_groups * dqkv 
                                + kv_head * n_groups * dqkv 
                                + group * dqkv;
                    let query_vector = &q_data[q_offset..q_offset + dqkv];
                    
                    for key_pos in 0..total_seq_len {
                        let k_offset = key_pos * n_kv_h * dqkv + kv_head * dqkv;
                        let key_vector = &k_data[k_offset..k_offset + dqkv];
                        
                        let mut score = 0.0;
                        for dim in 0..dqkv {
                            score += query_vector[dim] * key_vector[dim];
                        }
                        score *= scale;
                        
                        let score_idx = kv_head * n_groups * seq_len * total_seq_len 
                                       + group * seq_len * total_seq_len 
                                       + query_pos * total_seq_len 
                                       + key_pos;
                        att_scores_data[score_idx] = score;
                    }
                }
            }
        }
    }

    // Apply masked softmax to attention scores
    OP::masked_softmax(att_scores);

    // Compute attention-weighted value vectors (attn @ V)
    {
        let att_scores_data = att_scores.data();
        let hidden_states_data = unsafe { hidden_states.data_mut() };
        for kv_head in 0..n_kv_h {
            for group in 0..n_groups {
                for query_pos in 0..seq_len {
                    let output_offset = query_pos * n_kv_h * n_groups * dqkv 
                                      + kv_head * n_groups * dqkv 
                                      + group * dqkv;
                    for dim in 0..dqkv {
                        let mut weighted_sum = 0.0;
                        for key_pos in 0..total_seq_len {
                            let v_offset = key_pos * n_kv_h * dqkv + kv_head * dqkv + dim;
                            let attn_idx = kv_head * n_groups * seq_len * total_seq_len 
                                         + group * seq_len * total_seq_len 
                                         + query_pos * total_seq_len 
                                         + key_pos;
                            weighted_sum += att_scores_data[attn_idx] * v_data[v_offset];
                        }
                        hidden_states_data[output_offset + dim] = weighted_sum;
                    }
                }
            }
        }
    }
}


fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    // todo!("Implement mlp");
    // Step 1: RMS normalization
    OP::rms_norm(hidden_states, residual, rms_w, eps);
    
    // Step 2: Compute gate projection
    OP::matmul_transb(gate, 0.0, hidden_states, w_gate, 1.0);
    
    // Step 3: Compute up projection 
    OP::matmul_transb(up, 0.0, hidden_states, w_up, 1.0);

    // Step 4: Apply SwiGLU activation (gate * sigmoid(gate) * up)
    OP::swiglu(up, gate);  // up = up * silu(gate)
    
    // Step 5: Compute output and add to residual
    OP::matmul_transb(residual, 1.0, up, w_down, 1.0);
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use std::path::PathBuf;
    use crate::tensor::float_eq;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(&model.params.embedding_table.data()[50], &0.14453125, 1e-6));
    assert_eq!(model.params.lm_head.data()[10], model.params.embedding_table.data()[10]);
    assert!(float_eq(&model.params.rms_att_w[0].data()[10], &0.18652344, 1e-6));
    assert!(float_eq(&model.params.rms_ffn_w[1].data()[10], &0.32421875, 1e-6));
    assert!(float_eq(&model.params.rms_out_w.data()[100], &0.73046875, 1e-6));
    assert!(float_eq(&model.params.w_down[0].data()[100], &-0.0625, 1e-6));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(&model.params.w_gate[1].data()[100], &0.296875, 1e-6));
    assert!(float_eq(&model.params.wq[1].data()[100], &0.032226563, 1e-6));
    assert!(float_eq(&model.params.wk[1].data()[100], &-0.21386719, 1e-6));
    assert!(float_eq(&model.params.wv[0].data()[100], &0.041015625, 1e-6));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));

}
