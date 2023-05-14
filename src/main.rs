#![feature(slice_as_chunks)]
#![feature(generic_const_exprs)]
#![feature(get_many_mut)]
#![warn(clippy::pedantic)]

use std::array;

use indicatif::ProgressBar;
use itertools::{izip, Itertools};
use rand::{seq::SliceRandom, Rng};

const IMAGE_SIZE: usize = 28 * 28;

#[derive(Debug)]
struct Node<const INPUT_SIZE: usize> {
    weights: [f32; INPUT_SIZE],
    bias: f32,
}

struct Layer<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> {
    nodes: [Node<INPUT_SIZE>; OUTPUT_SIZE],
}

fn random_param() -> f32 {
    rand::thread_rng().gen_range((-1.)..1.)
}

impl<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> Layer<INPUT_SIZE, OUTPUT_SIZE> {
    fn zeros() -> Self {
        Self {
            nodes: array::from_fn(|_| Node {
                weights: [0.; INPUT_SIZE],
                bias: 0.,
            }),
        }
    }
    fn random() -> Self {
        Self {
            nodes: array::from_fn(|_| Node {
                weights: array::from_fn(|_| random_param()),
                bias: random_param(),
            }),
        }
    }
    fn compute(&self, input: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        array::from_fn(|i| {
            let node = &self.nodes[i];
            let sum: f32 = izip!(input, &node.weights)
                .map(|(&input, &weight)| input.max(0.) * weight)
                .sum();
            sum + node.bias
        })
    }
    fn add_gradient(&mut self, input: &[f32; INPUT_SIZE], error: &[f32; OUTPUT_SIZE]) {
        izip!(&mut self.nodes, error).for_each(|(node, &error)| {
            {
                node.bias -= error;
                izip!(input, &mut node.weights).for_each(|(&input, weight)| {
                    *weight -= input.max(0.) * error;
                });
            };
        });
    }
    fn apply_gradient(&mut self, gradient: &Self, learning_rate: f32) {
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
    fn propagate_errors(
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
    fn abs_sum(&self) -> f32 {
        self.nodes
            .iter()
            .map(|node| {
                node.bias.abs() + node.weights.iter().map(|&weight| weight.abs()).sum::<f32>()
            })
            .sum()
    }
}

struct NeuralNet<const LAYER_SIZE: usize, const MIDDLE_LAYERS: usize> {
    input_layer: Layer<IMAGE_SIZE, LAYER_SIZE>,
    middle_layers: [Layer<LAYER_SIZE, LAYER_SIZE>; MIDDLE_LAYERS],
    output_layer: Layer<LAYER_SIZE, 10>,
}

#[derive(Debug)]
struct Activations<const LAYER_SIZE: usize, const HIDDEN_LAYERS: usize> {
    hidden_layers: [[f32; LAYER_SIZE]; HIDDEN_LAYERS],
    output_layer: [f32; 10],
}

impl<const LAYER_SIZE: usize, const HIDDEN_LAYERS: usize> Activations<LAYER_SIZE, HIDDEN_LAYERS> {
    fn output_layer_is_correct(&self, label: u8) -> bool {
        self.output_layer
            .iter()
            .position_max_by(|a, b| a.partial_cmp(b).expect("NaN in output layer"))
            .expect("Empty output layer")
            == label as usize
    }
    fn output_errors(&self, label: u8) -> [f32; 10] {
        let layer_size = f32::from(u16::try_from(LAYER_SIZE).unwrap());
        array::from_fn(|i| {
            let act = self.output_layer[i];
            if i == label as usize {
                10. * (act.min(layer_size) - layer_size)
            } else {
                act.max(0.)
            }
        })
    }
}

impl<const LAYER_SIZE: usize, const MIDDLE_LAYERS: usize> NeuralNet<LAYER_SIZE, MIDDLE_LAYERS>
where
    [(); MIDDLE_LAYERS + 1]: Copy,
{
    fn zeros() -> Self {
        Self {
            input_layer: Layer::zeros(),
            middle_layers: array::from_fn(|_| Layer::zeros()),
            output_layer: Layer::zeros(),
        }
    }
    fn random() -> Self {
        Self {
            input_layer: Layer::random(),
            middle_layers: array::from_fn(|_| Layer::random()),
            output_layer: Layer::random(),
        }
    }
    fn train(mut data: Vec<(u8, &[u8; IMAGE_SIZE])>, epochs: usize) -> Self {
        let mut net = Self::random();
        data.shuffle(&mut rand::thread_rng());
        let bar = ProgressBar::new((data.len() * epochs) as u64 / 100);
        for _ in 0..epochs {
            for chunk in data.chunks(100) {
                let mut gradient = Self::zeros();
                for &(label, input) in chunk {
                    let activations = net.compute(input);
                    let errors = net.errors(&activations, label);
                    gradient.add_gradient(input, &activations, &errors);
                }
                net.apply_gradient(&gradient);
                bar.inc(1);
            }
        }
        net
    }
    fn add_gradient(
        &mut self,
        input: &[u8; IMAGE_SIZE],
        activations: &Activations<LAYER_SIZE, { MIDDLE_LAYERS + 1 }>,
        errors: &Activations<LAYER_SIZE, { MIDDLE_LAYERS + 1 }>,
    ) {
        self.input_layer.add_gradient(
            &input.map(|b| f32::from(b) / 256.),
            &errors.hidden_layers[0],
        );
        izip!(
            &mut self.middle_layers,
            &activations.hidden_layers[..MIDDLE_LAYERS],
            &errors.hidden_layers[1..]
        )
        .for_each(|(layer, activations, errors)| {
            layer.add_gradient(activations, errors);
        });
        self.output_layer.add_gradient(
            &activations.hidden_layers[MIDDLE_LAYERS],
            &errors.output_layer,
        );
    }
    fn abs_sum(&self) -> f32 {
        self.input_layer.abs_sum()
            + self.middle_layers.iter().map(Layer::abs_sum).sum::<f32>()
            + self.output_layer.abs_sum()
    }
    fn apply_gradient(&mut self, gradient: &Self) {
        let learning_rate = (10. / gradient.abs_sum()).min(1.);
        self.input_layer
            .apply_gradient(&gradient.input_layer, learning_rate);
        izip!(&mut self.middle_layers, &gradient.middle_layers,).for_each(
            |(layer, layer_gradient)| {
                layer.apply_gradient(layer_gradient, learning_rate);
            },
        );
        self.output_layer
            .apply_gradient(&gradient.output_layer, learning_rate);
    }
    fn compute(&self, input: &[u8; IMAGE_SIZE]) -> Activations<LAYER_SIZE, { MIDDLE_LAYERS + 1 }> {
        let mut hidden_layers = [[0.; LAYER_SIZE]; MIDDLE_LAYERS + 1];
        hidden_layers[0] = self
            .input_layer
            .compute(&input.map(|b| f32::from(b) / 256.));
        for i in 0..MIDDLE_LAYERS {
            hidden_layers[i + 1] = self.middle_layers[i].compute(&hidden_layers[i]);
        }
        let output_layer = self.output_layer.compute(&hidden_layers[MIDDLE_LAYERS]);
        Activations {
            hidden_layers,
            output_layer,
        }
    }
    fn errors(
        &self,
        activations: &Activations<LAYER_SIZE, { MIDDLE_LAYERS + 1 }>,
        label: u8,
    ) -> Activations<LAYER_SIZE, { MIDDLE_LAYERS + 1 }> {
        let output_layer = activations.output_errors(label);
        let mut hidden_layers = [[0.; LAYER_SIZE]; MIDDLE_LAYERS + 1];
        self.output_layer.propagate_errors(
            &mut hidden_layers[MIDDLE_LAYERS],
            &output_layer,
            &activations.hidden_layers[MIDDLE_LAYERS],
        );
        for i in (0..MIDDLE_LAYERS).rev() {
            let [back_errors, front_errors] = hidden_layers.get_many_mut([i, i + 1]).unwrap();
            self.middle_layers[i].propagate_errors(
                back_errors,
                front_errors,
                &activations.hidden_layers[i],
            );
        }
        Activations {
            hidden_layers,
            output_layer,
        }
    }
}

#[allow(dead_code)]
fn print_image(input: &[u8; IMAGE_SIZE]) {
    for i in 0..28 {
        for j in 0..28 {
            let pixel = input[i * 28 + j];
            if pixel == 0 {
                print!(" ");
            } else if pixel < 64 {
                print!(".");
            } else if pixel < 196 {
                print!("o");
            } else {
                print!("@");
            }
        }
        println!();
    }
}

fn main() {
    let train_labels = &include_bytes!("../data/train-labels-idx1-ubyte")[8..];
    let train_images = &include_bytes!("../data/train-images-idx3-ubyte")[16..];
    let test_labels = &include_bytes!("../data/t10k-labels-idx1-ubyte")[8..];
    let test_images = &include_bytes!("../data/t10k-images-idx3-ubyte")[16..];

    let data = izip!(train_labels.iter().copied(), train_images.as_chunks().0).collect::<Vec<_>>();
    let nn = NeuralNet::<64, 1>::train(data, 1);

    let errors = izip!(test_labels, test_images.as_chunks().0)
        .filter(|(&label, image)| !nn.compute(image).output_layer_is_correct(label))
        .count();
    #[allow(clippy::cast_precision_loss)]
    let error_rate = errors as f64 * 100. / test_labels.len() as f64;

    println!("Error rate: {error_rate:.2}%",);
}
