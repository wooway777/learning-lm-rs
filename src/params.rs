use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // todo!("实现从safetensors文件的模型参数加载");
        // let get_tensor: impl Fn(&str) -> Tensor<f32> = |name: &str| {
        // ...    
        // };
        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor = safetensor.tensor(name).expect("Tensor doesn't exist.");
            let data = tensor.data()
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            Tensor::new(data, &tensor.shape().to_vec())
        };

        // Macro to load all layers of a parameter
        macro_rules! load_layers {
            ($pattern:literal) => {
                (0..config.num_hidden_layers)
                    .map(|i| get_tensor(&format!($pattern, i)))
                    .collect()
            };
        }
        
        // LLamaParams {
        //     embedding_table: get_tensor(...),
        //     ...
        // }
        LLamaParams {
            // token_id to embedding lookup table
            embedding_table: get_tensor("lm_head.weight"),
            // decoder layer
            rms_att_w: load_layers!("model.layers.{}.input_layernorm.weight"),
            wq: load_layers!("model.layers.{}.self_attn.q_proj.weight"),
            wk: load_layers!("model.layers.{}.self_attn.k_proj.weight"),
            wv: load_layers!("model.layers.{}.self_attn.v_proj.weight"),
            wo: load_layers!("model.layers.{}.self_attn.o_proj.weight"),
            // ffn layer
            rms_ffn_w: load_layers!("model.layers.{}.post_attention_layernorm.weight"),
            w_up: load_layers!("model.layers.{}.mlp.up_proj.weight"),
            w_gate: load_layers!("model.layers.{}.mlp.gate_proj.weight"),
            w_down: load_layers!("model.layers.{}.mlp.down_proj.weight"),
            // output
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
