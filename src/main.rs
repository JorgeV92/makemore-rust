#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]


extern crate ndarray;
use ndarray::{Array, Array1, Array2, Array3, Array4, ArrayD};
use core::{f64, str};
use std::{f64::consts::PI};
use ndarray::prelude::*;



#[derive(Debug, Clone)]
///
/// ----- Configuration struct -----
/// 
pub struct ModelConfig {
    pub block_size: Option<usize>,
    pub vocab_size: Option<usize>,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_embd2: usize,
    pub n_head: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            block_size: None,
            vocab_size: None,
            n_layer: 4,
            n_embd: 64,
            n_embd2: 64,
            n_head: 4,
        }
    }
}

#[derive(Debug, Clone)]
///
/// ----- GELU Activation -----
/// 
pub struct NewGELU;

impl NewGELU {
    /// Applies the GELU activation function elemetwise on the input tensor.
    ///
    /// The GELU activation function defined as:
    /// 0.5 * x * (1.0 + tanh(sqrt(2.0/PI) * (x + 0.044715 * x^3) ))
    /// [TODO] research if we would need a D-dimensional array or 
    /// we could just use a 3D- a rray always
    pub fn forward(&self, input: &ArrayD<f64>) -> ArrayD<f64> {
        // Calculate the scaling factor: sqrt(2.0/PI)
        let scale = (2.0 / PI).sqrt();

        // Apply the activation elememt-wise using mapv
        input.mapv(|x| {
            0.5 * x * (1.0 + (scale * (x + 0.44715 * x.powi(3))).tanh())
        })
    }
}

// ----------------------------------------------------------------------------------------------------
// This in built without libraries such as burn or any bindigns such as tch-rs (LibTorch bindings for Rust) 
// Plan to incoporate these later [TODO]                                 
// ----------------------------------------------------------------------------------------------------

///
/// A simple linear layer mimicking nn.Linear
/// Applies: y = W * x + b for each (B(Batch Size), T(Sequence Length)) element
/// - weight has shape (out_features, in_features)
/// - bias has shape (out_features)
/// 
#[derive(Debug, Clone)]
///
/// ----- Linear Layer -----
/// 
pub struct Linear {
    pub weight: Array2<f64>,    // shape: (out_features, in_features)
    pub bias: Array1<f64>,      // shape: (out_features)
}

impl Linear {
    // Applies the linear transformation for an input of shape (B, T, in_features)
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let (B, T, in_features) = x.dim();
        let out_features = self.bias.len();
        let mut output = Array3::<f64>::zeros((B, T, out_features));

        // Compute W * x + b
        for b in 0..B {
            for t in 0..T {
                let x_vec = x.slice(s![b, t, ..]);
                for o in 0..out_features {
                    // Dot prod of weight row o with the input vector plus bias.
                    output[[b, t, o]] = self.weight.row(o).dot(&x_vec) + self.bias[o];
                }
            }
        }

        output
    }
}

///
/// Utility function to generate a lower-triangular (causal) mask.
/// Returns a 2D Array with ones on and below the diagnol.
/// 
fn lower_triangular(n: usize) -> Array2<f64> {
    let mut a = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            a[[i,j]] = 1.0;
        }
    }

    a
}

///
/// CausalSelfAttention structure
/// It contains two layer for computing query, key, valye (c_attn) and for output projection (c_proj).
/// The bias field holds the causal mask (shape: (1, 1, block_size, block_size)).
/// 
#[derive(Debug, Clone)]
pub struct CausalSelfAttention {
    pub c_attn: Linear,             // projects input (n_embd) to 3 * n n_embd (query, key, value)
    pub c_proj: Linear,             // outout projection from n_embd to n_embd
    pub bias: Array4<f64>,          //  causal mask, shape (1, 1, block_size, block_size)
    pub n_head: usize, 
    pub n_embd: usize,
    pub block_size: usize,
}

impl CausalSelfAttention {
    /// Constuctor for CausalSelfAttention
    /// It asserts that n_embd is divisble by n_head.
    pub fn new(n_embd: usize, n_head: usize, block_size: usize) -> Self {
        assert!(n_embd & n_head == 0, "n_embd must be divisible by n_head");

         
        // [TODO] 
        // For testing purposes we start with linear layers with zeros.
        // fix to weights randomly initialized. 
        let c_attn = Linear {
            // c_attn maps an input of size n_embd to 3 * n_embd.
            weight: Array2::zeros((3*n_embd, n_embd)),
            bias: Array1::zeros(3 * n_embd),
        };
        let c_proj = Linear {
            weight: Array2::zeros((n_embd, n_embd)),
            bias: Array1::zeros(n_embd),
        };

        // Create the causal mask: a lower triangular matrix of ones, then reshape to (1, 1, block_size, block_size)
        let tril = lower_triangular(block_size);
        let bias = tril.to_shape((1,1,block_size,block_size)).unwrap().to_owned();

        Self {
            c_attn, 
            c_proj,
            bias,
            n_head,
            n_embd,
            block_size,
        }
    }

    //
    // Helper for batched matrix multi.
    // Multiplies q (B, n_head, T, head_size) with the transpose of k (B, n_head, T, head_size)
    // resulting in attention scores of shape (B, n_head, T, T).
    //
    fn batched_matmul(q: &Array4<f64>, k: &Array4<f64>) -> Array4<f64> {
        let (B, n_head, T, _head_size) = q.dim();
        let mut att = Array4::<f64>::zeros((B, n_head, T, T));
        for b in 0.. B {
            for h in 0..n_head {
                // q_matrix has shape (T, head_size)
                let q_matrix = q.slice(s![b, h, .., ..]);
                // k_matrix, transposed, has shape (head_size, T)
                let k_matrix = k.slice(s![b, h, .., ..]).reversed_axes();
                let result = q_matrix.dot(&k_matrix); // (T, T)
                att.slice_mut(s![b, h, .., ..]).assign(&result);
            }
        } 

        att
    }

    // 
    // The forward method mimics the Pytorch forward funciton.
    // Input x shape is assumed to be (B, T, n_embd) where B=batch size, T=sequence length.
    //
    pub fn forward(&self, x: & Array3<f64>) -> Array3<f64> {
        let (B, T ,_) = x.dim();
        let head_size = self.n_embd / self.n_head;

        // compute the projections: output shape becomes (B, T, 3 * n_embd)
        let c_attn_out = self.c_attn.forward(x);

        // Split the output into query, key, value.
        let q = c_attn_out.slice(s![..,..,0..self.n_embd]).to_owned();
        let k = c_attn_out.slice(s![..,..,self.n_embd..2 * self.n_embd]).to_owned();
        let v = c_attn_out.slice(s![..,..,2 * self.n_embd..3 * self.n_embd]).to_owned();

        // Reshape each of q, k, v into (B, T, n_head, head_size) then permute to (B, n_head, head_size)
        let q = q.to_shape((B, T, self.n_head, head_size)).unwrap().permuted_axes([0, 2, 1, 3]).to_owned();
        let k = k.to_shape((B, T, self.n_head, head_size)).unwrap().permuted_axes([0, 2, 1, 3]).to_owned();
        let v = v.to_shape((B, T, self.n_head, head_size)).unwrap().permuted_axes([0, 2, 1, 3]).to_owned();

        // Compute raw attention scores: (B, n_head, T, T) = q @ k^t
        let scale = 1.0 / (head_size as f64).sqrt();
        let mut att = Self::batched_matmul(&q, &k) * scale;

        // Apply the causal mask to ensure that attention only attends to previous positions.
        {
            let mask = self.bias.slice(s![0, 0, 0..T, 0..T]);
            for b in 0..B {
                for h in 0..self.n_head {
                    for i in 0..T {
                        for j in 0..T {
                            if mask[[i, j]] == 0.0 {
                                att[[b, h, i, j]] = f64::NEG_INFINITY;
                            }
                        }
                    }
                }
            }
        }

        // Softmax along the last axis for each attention row.
        let mut att_softmax = att.clone();
        for b in 0..B {
            for h in 0..self.n_head {
                for i in 0..T {
                    let row = att.slice(s![b, h, i, ..]);
                    let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exp_row: Vec<f64> = row.iter().map(|&v| (v - max_val).exp()).collect();
                    let sum_exp: f64 = exp_row.iter().sum();
                    for j in 0..T {
                        att_softmax[[b, h, i, j]] = exp_row[j] / sum_exp;
                    }
                }
            }
        }

        let att = att_softmax;

        // Compute the weight sum of values: y = att @ v.
        // y will have shape (B, n_head, T, head_size)
        let mut y = Array4::<f64>::zeros((B, self.n_head, T, head_size));
        for b in 0..B {
            for h in 0..self.n_head {
                let att_matrix = att.slice(s![b, h, .., ..]);       // shape (T, T)
                let v_matrix = v.slice(s![b, h, .., ..]);           // shape (T, head_size)
                let result = att_matrix.dot(&v_matrix);                  // shape (T, head_size)
                y.slice_mut(s![b, h, .., ..]).assign(&result);
            }
        }

        // Reassemble the multi-head outputs: transpose from (B, n_head, T, head_size)
        // to (B, T, n_embd) where n _embd = n_head * head_size.
        let y = y.permuted_axes([0, 2, 1, 3])
            .to_shape((B, T, self.n_embd))
            .unwrap()
            .to_owned();
        
        self.c_proj.forward(&y)
    }
    
}

///
/// ----- Layer Normalization -----
/// A naïve first step implementation maybe update to follow Pytorch nn.LinearNorm
/// for now on the last axis (the embedding dimesion)
/// 
#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub normalized_shape: usize,
    pub epsilon: f64,
}

impl LayerNorm {
    pub fn new(normalized_shape: usize, epsilon: f64) -> Self {
        Self { 
            normalized_shape,
            epsilon 
        }
    }

    ///
    /// For an input of shape (B, T ,C), normalize across the last dimension 
    /// 
    pub fn forward(&self, input: &Array3<f64>) -> Array3<f64> {
        let (B, T, C) = input.dim();
        let mut output = Array3::<f64>::zeros((B, T, C));
        for b in 0..B {
            for t in 0..T {
                let x_slice = input.slice(s![b, t, ..]);
                let mean = x_slice.mean().unwrap();
                // compute variance maunuallu
                // [TODO] 
                let var = x_slice.mapv(|v| (v - mean).powi(2)).mean().unwrap();
                for c in 0..C {
                    output[[b, t, c]] = (input[[b, t, c]] - mean) / (var + self.epsilon).sqrt();
                }   
            }
        }

        output
    }

}


///
/// MLP module that applies: c_fc -> act -> c_proj
/// 
#[derive(Debug, Clone)]
pub struct MLP {
    pub c_fc: Linear,
    pub c_proj: Linear,
    pub act: NewGELU
}

impl MLP {
    pub fn new(n_embd: usize) -> Self {
        Self { 
            c_fc: Linear {
                weight: Array2::zeros((4 * n_embd, n_embd)),
                bias: Array1::zeros(4 * n_embd),
            },
            c_proj: Linear {
                weight: Array2::zeros((n_embd, 4 * n_embd)),
                bias: Array1::zeros(n_embd),
            },
            act: NewGELU,
        }
    }
    
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        // [D]
        // Implementation for a D-dimensional array 
        let x = self.c_fc.forward(x);
        let x_dyn = x.into_dyn();
        let x_activation = self.act.forward(&x_dyn);
        let x_fixed = x_activation.into_dimensionality::<Ix3>().expect("Expected Array3<f64>");
        self.c_proj.forward(&x_fixed)
    }
}

///
/// The [Transformer] block that combines layer norms, causal attention, and an MLP.
/// 
#[derive(Debug, Clone)]
pub struct  Block {
    pub ln_1: LayerNorm,
    pub attn: CausalSelfAttention,
    pub ln_2: LayerNorm,
    pub mlp: MLP,
}

impl Block {
    pub fn new(n_embd: usize, n_head: usize, block_size: usize) -> Self {
        Self {
            ln_1: LayerNorm::new(n_embd, 1e-5),
            attn: CausalSelfAttention::new(n_embd, n_head, block_size),
            ln_2: LayerNorm::new(n_embd, 1e-5),
            mlp: MLP::new(n_embd),
        }
    }

    ///
    /// Forward pass for the block:
    /// x = x _ attn(ln_1(x))
    /// x = x + mlp(ln_2(x))
    /// 
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        // Apply first layer norm then attention; add residual connection.
        let attn_out = self.attn.forward(&self.ln_1.forward(x));
        let x = x + &attn_out;

        // Apply second layer norm then MLP; add residual connection.
        let mlp_out = self.mlp.forward(&self.ln_2.forward(&x));
        x + &mlp_out
    }
}

///
/// ----- Simple Embedding module -----
/// 
#[derive(Debug, Clone)]
pub struct Embedding {
    pub weight: Array2<f64>,    // weught shape: (num_embeddings, embedding_dim)    
}

impl Embedding {
    // Given a tensor of indices of shape (B, T) return embeddings of shape (B, T embedding_dim)
    pub fn forward(&self, idx: &Array2<usize>) -> Array3<f64> {
        let (b, t) = idx.dim();
        let (_, n_embd) = self.weight.dim();
        let mut out = Array3::<f64>::zeros((b, t, n_embd));
        for bi in 0..b {
            for ti in 0..t {
                let token = idx[[bi, ti]];
                out.slice_mut(s![bi, ti, ..]).assign(&self.weight.row(token));
            }
        }
        out 
    }
}

///
/// ----- Transformer Model -----
/// 
#[derive(Debug)]
pub struct Transformer {
    pub block_size: usize,
    pub wte: Embedding,     // token embeddings
    pub wpe: Embedding,     // position embeddings
    pub h: Vec<Block>,      // stack of transformer blocks
    pub ln_f: LayerNorm,    // final layer normalization
    pub lm_head: Linear,    // language model head
}

impl Transformer {
    pub fn new(config: &ModelConfig) -> Self {
        let block_size = config.block_size.expect("block_size is required");
        let voacb_size = config.vocab_size.expect("voacb_size is required");
        let n_embd = config.n_embd;

        let wte = Embedding { weight: Array2::zeros((voacb_size, n_embd)) };
        let wpe = Embedding { weight: Array2::zeros((block_size, n_embd)) };

        let mut h = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            h.push(Block::new(n_embd, config.n_head, block_size));
        }

        let ln_f = LayerNorm::new(n_embd, 1e-5);

        // For lm_head we tie weights to the embedding matrix in GPT-2
        // (here we create a separate Linear layer with bias disabled.)
        let lm_head = Linear {
            weight: Array2::zeros((voacb_size, n_embd)),
            bias: Array1::zeros(voacb_size),
        };

        // (For demonstration, we skip parameter counting—one could iterate over each module’s parameters.)
        // println!("number of parameters: {}M", 0.0);

        Self {
            block_size,
            wte, 
            wpe,
            h,
            ln_f,
            lm_head,
        }
    }

    pub fn get_block_size(&self) -> usize {
        self.block_size
    }

    ///
    /// Forward pass accepts:
    /// -idx: an Array2 of shape (B, T) (token indices)
    /// - targets: Optional target indices (for computing a loss)
    /// Returns (logits, loss)
    /// 
    pub fn forward(&self, idx: &Array2<usize>, targets: Option<&Array2<usize>>) -> (Array3<f64>, Option<f64>) {
        let (b, t) = idx.dim();
        if t > self.block_size {
            panic!("Cannot forward sequence of length {}, block size is only {}", t, self.block_size);
        }

        // create positional indices of shape (1, t)
        let pos = Array2::from_shape_fn((1, t), |(_, j)| j);
        let tok_emb = self.wte.forward(idx);          // (B, T, n_embd)
        let pos_emb = self.wpe.forward(&pos);       // (1, T, n_embd)
        let shape = tok_emb.dim();
        // Broadcast pos_emb to (B, T, n_embd) and add to token embeddings
        let x = tok_emb + pos_emb.broadcast(shape).unwrap();

        // Pass through each Block
        let mut x = x;
        for block in &self.h {
            x = block.forward(&x);
        }
        // Final layer norm 
        x = self.ln_f.forward(&x);
        // language model head: project embeddings to vocabulary logits
        let logits = self.lm_head.forward(&x);
        
        // compute loss. 
        // [TODO]
        let loss = if targets.is_some() {
            Some(0.0)
        } else {
            None
        };

        (logits, loss)
    }
}


fn main() { 

    println!("
             // ----------------------------------------------------------------------------------------------------
            //                                  Sample the config struct 
            // ---------------------------------------------------------------------------------------------------- 
    ");
    let config = ModelConfig::default();
    println!("{:?}", config);

    println!("
        // ----------------------------------------------------------------------------------------------------
        //                                  Sample the GELU activation 
        // ----------------------------------------------------------------------------------------------------
    ");
    let gelu = NewGELU;
    let input = ArrayD::from_elem(ndarray::IxDyn(&[5]), 0.5);
    let output = gelu.forward(&input);
    println!("Input: {:?}", input);
    println!("Output: {:?}", output);


    println!("
        // ----------------------------------------------------------------------------------------------------
        //                                  Sample the Causal Self Attention
        // ----------------------------------------------------------------------------------------------------
    ");
    let n_embd = 64;
    let n_head = 4;
    let block_size = 16;
    let att_layer = CausalSelfAttention::new(n_embd, n_head, block_size);

    let batch_size = 2;
    let sequencele_length = 10;
    let x = Array3::<f64>::zeros((batch_size, sequencele_length, n_embd));

    let output = att_layer.forward(&x);
    println!("Attention output shape: {:?}", output.dim());

    println!(
        "// ----------------------------------------------------------------------------------------------------
        //                                      Sample Block
        // ----------------------------------------------------------------------------------------------------"
    );
    let transformer_block = Block::new(n_embd, n_head, block_size);
    let mlp = MLP::new(n_embd);
    println!("Initialized Block: {:?}", transformer_block);
    println!("Initialized MLP: {:?}", mlp);

    let block_output = transformer_block.forward(&x);
    println!("Block output shape: {:?}", block_output.dim());

    let mlp_output = mlp.forward(&x);
    println!("MLP output shape: {:?}", mlp_output.dim());

    println!(
        "// ----------------------------------------------------------------------------------------------------
        //                                      Sample Transformer
        // ----------------------------------------------------------------------------------------------------"
    );

    let config = ModelConfig {
        block_size: Some(16),
        vocab_size: Some(50257),
        n_layer: 4,
        n_embd: 65,
        n_embd2: 64,
        n_head: 4,
    };

    let transformer = Transformer::new(&config);
    let idx = Array2::<usize>::zeros((batch_size, sequencele_length));
    println!("Initialized Transformer:\n{:#?}", transformer);

    // // Run forward pass test
    // let (logits, loss) = transformer.forward(&idx, None);
    // println!("Logits shape: {:?}", logits.dim());
    // if let Some(l) = loss {
    //     println!("Loss: {}", l);
    // }
}

