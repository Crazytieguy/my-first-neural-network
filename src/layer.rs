use itertools::izip;
use rand::Rng;
use std::array;

#[derive(Debug)]
pub(crate) struct Node<const INPUT_SIZE: usize> {
    pub(crate) weights: [f32; INPUT_SIZE],
    pub(crate) bias: f32,
}

pub(crate) struct Layer<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, const RELU: bool> {
    pub(crate) nodes: [Node<INPUT_SIZE>; OUTPUT_SIZE],
}

pub(crate) fn random_param() -> f32 {
    rand::thread_rng().gen_range((-1.)..1.)
}

impl<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize, const RELU: bool>
    Layer<INPUT_SIZE, OUTPUT_SIZE, RELU>
{
    pub(crate) fn zeros() -> Self {
        Self {
            nodes: array::from_fn(|_| Node {
                weights: [0.; INPUT_SIZE],
                bias: 0.,
            }),
        }
    }
    pub(crate) fn random() -> Self {
        Self {
            nodes: array::from_fn(|_| Node {
                weights: array::from_fn(|_| random_param()),
                bias: random_param(),
            }),
        }
    }
    pub(crate) fn compute(&self, input: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        array::from_fn(|i| {
            let node = &self.nodes[i];
            let sum: f32 = izip!(input, &node.weights)
                .map(|(&input, &weight)| input * weight)
                .sum();
            if RELU {
                (sum + node.bias).max(0.)
            } else {
                sum + node.bias
            }
        })
    }
    pub(crate) fn add_gradient(&mut self, input: &[f32; INPUT_SIZE], error: &[f32; OUTPUT_SIZE]) {
        izip!(&mut self.nodes, error).for_each(|(node, &error)| {
            {
                node.bias -= error;
                izip!(input, &mut node.weights).for_each(|(&input, weight)| {
                    *weight -= input * error;
                });
            };
        });
    }
    pub(crate) fn apply_gradient(&mut self, gradient: &Self, learning_rate: f32) {
        izip!(&mut self.nodes, &gradient.nodes).for_each(|(node, node_gradient)| {
            {
                node.bias += node_gradient.bias * learning_rate;
                izip!(&mut node.weights, &node_gradient.weights).for_each(
                    |(weight, weight_gradient)| {
                        *weight += *weight_gradient * learning_rate;
                    },
                );
            };
        });
    }
    pub(crate) fn propagate_errors(
        &self,
        back_errors: &mut [f32; INPUT_SIZE],
        front_errors: &[f32; OUTPUT_SIZE],
        back_activations: &[f32; INPUT_SIZE],
    ) {
        izip!(&self.nodes, front_errors).for_each(|(node, output_error)| {
            izip!(&node.weights, &mut *back_errors).for_each(|(weight, input_error)| {
                *input_error += weight * output_error;
            });
        });
        izip!(back_errors, back_activations).for_each(|(error, activation)| {
            if activation <= &0. {
                *error = 0.;
            }
        });
    }
    pub(crate) fn gradient_size_squared(&self) -> f32 {
        self.nodes
            .iter()
            .map(|node| {
                node.bias.powi(2)
                    + node
                        .weights
                        .iter()
                        .map(|&weight| weight.powi(2))
                        .sum::<f32>()
            })
            .sum()
    }
}
