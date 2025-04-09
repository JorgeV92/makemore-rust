extern crate ndarray;
use ndarray::ArrayD;
use std::f64::consts::PI;



#[derive(Debug, Clone)]
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

pub struct NewGELU;

impl NewGELU {
    /// Applies the GELU activation function elemetwise on the input tensor.
    ///
    /// The GELU activation function defined as:
    /// 0.5 * x * (1.0 + tanh(sqrt(2.0/PI) * (x + 0.044715 * x^3) ))
    pub fn forward(&self, input: &ArrayD<f64>) -> ArrayD<f64> {
        // Calculate the scaling factor: sqrt(2.0/PI)
        let scale = (2.0 / PI).sqrt();

        // Apply the activation elememt-wise using mapv
        input.mapv(|x| {
            0.5 * x * (1.0 + (scale * (x + 0.44715 * x.powi(3))).tanh())
        })
    }
}





fn main() { 

    // ----------------------------------------------------------------------------------------------------
    //                                  Sample the config struct 
    // ----------------------------------------------------------------------------------------------------
    println!("
             // ----------------------------------------------------------------------------------------------------
            //                                  Sample the config struct 
            // ---------------------------------------------------------------------------------------------------- 
    ");
    let config = ModelConfig::default();
    println!("{:?}", config);

    // ----------------------------------------------------------------------------------------------------
    //                                  Sample the GELU activation 
    // ----------------------------------------------------------------------------------------------------
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
}

