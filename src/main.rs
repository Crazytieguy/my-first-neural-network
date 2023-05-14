#![feature(slice_as_chunks)]
#![feature(generic_const_exprs)]
#![feature(get_many_mut)]
#![warn(clippy::pedantic)]

use std::array;

use indicatif::ProgressBar;
use itertools::Itertools;
use rand::{seq::SliceRandom, Rng};

const IMAGE_SIZE: usize = 28 * 28;

#[derive(Debug)]
struct Node<const INPUT_SIZE: usize> {
    weights: [f32; INPUT_SIZE],
    bias: f32,
}

impl<const INPUT_SIZE: usize> Node<INPUT_SIZE> {
    fn zeros() -> Self {
        Self {
            weights: [0.; INPUT_SIZE],
            bias: 0.,
        }
    }
    fn random() -> Self {
        Self {
            weights: array::from_fn(|_| rand::thread_rng().gen_range((-1.)..1.)),
            bias: rand::thread_rng().gen_range((-1.)..1.),
        }
    }
    fn compute(&self, input: &[f32; INPUT_SIZE]) -> f32 {
        let sum: f32 = input
            .iter()
            .zip_eq(&self.weights)
            .map(|(&input, &weight)| input.max(0.) * weight) // activation on input
            .sum();
        sum + self.bias
    }
    fn update(&mut self, input: &[f32; INPUT_SIZE], error: f32) {
        self.bias -= error;
        self.weights
            .iter_mut()
            .zip_eq(input)
            .filter(|(_, &input)| input > 0.)
            .for_each(|(weight, &input)| {
                *weight -= input * error;
            });
    }
    fn flush_updates(&mut self, updates: &Self) {
        self.bias += updates.bias;
        self.weights
            .iter_mut()
            .zip_eq(&updates.weights)
            .for_each(|(weight, update)| {
                *weight += *update;
            });
    }
}

struct Layer<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> {
    nodes: [Node<INPUT_SIZE>; OUTPUT_SIZE],
}

impl<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> Layer<INPUT_SIZE, OUTPUT_SIZE> {
    fn zeros() -> Self {
        Self {
            nodes: array::from_fn(|_| Node::zeros()),
        }
    }
    fn random() -> Self {
        Self {
            nodes: array::from_fn(|_| Node::random()),
        }
    }
    fn compute(&self, input: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        let mut output = [0.; OUTPUT_SIZE];
        output
            .iter_mut()
            .zip_eq(&self.nodes)
            .for_each(|(out, node)| {
                *out = node.compute(input);
            });
        output
    }
    fn update(&mut self, input: &[f32; INPUT_SIZE], error: &[f32; OUTPUT_SIZE]) {
        self.nodes
            .iter_mut()
            .zip_eq(error)
            .for_each(|(node, error)| {
                node.update(input, *error);
            });
    }
    fn flush_updates(&mut self, updates: &Self) {
        self.nodes
            .iter_mut()
            .zip_eq(&updates.nodes)
            .for_each(|(node, update)| {
                node.flush_updates(update);
            });
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
        let mut updates = Self::zeros();
        data.shuffle(&mut rand::thread_rng());
        let bar = ProgressBar::new((data.len() * epochs) as u64 / 100);
        for _ in 0..epochs {
            for chunk in data.chunks(100) {
                for &(label, input) in chunk {
                    let activations = net.compute(input);
                    let errors = net.errors(&activations, label);
                    updates.update(input, &activations, &errors);
                    if activations.output_layer.iter().copied().any(f32::is_nan)
                        || !activations.output_layer_is_correct(label)
                            && errors.hidden_layers[MIDDLE_LAYERS].iter().all(|&v| v == 0.)
                    {
                        dbg!(
                            label,
                            print_image(input),
                            &activations,
                            &errors,
                            &net.output_layer.nodes[0],
                            &net.middle_layers[0].nodes[0]
                        );
                        panic!()
                    }
                }
                net.flush_updates(&mut updates);
                updates = Self::zeros();
                bar.inc(1);
            }
        }
        net
    }
    fn update(
        &mut self,
        input: &[u8; IMAGE_SIZE],
        activations: &Activations<LAYER_SIZE, { MIDDLE_LAYERS + 1 }>,
        errors: &Activations<LAYER_SIZE, { MIDDLE_LAYERS + 1 }>,
    ) {
        self.output_layer.update(
            &activations.hidden_layers[MIDDLE_LAYERS],
            &errors.output_layer,
        );
        for i in (0..MIDDLE_LAYERS).rev() {
            self.middle_layers[i]
                .update(&activations.hidden_layers[i], &errors.hidden_layers[i + 1]);
        }
        self.input_layer.update(
            &input.map(|b| f32::from(b) / 256.),
            &errors.hidden_layers[0],
        );
    }
    fn flush_updates(&mut self, updates: &mut Self) {
        let sum_gradient = updates
            .input_layer
            .nodes
            .iter()
            .flat_map(|node| node.weights.iter().chain([&node.bias]))
            .copied()
            .map(f32::abs)
            .sum::<f32>()
            + updates
                .middle_layers
                .iter()
                .map(|layer| {
                    layer
                        .nodes
                        .iter()
                        .flat_map(|node| node.weights.iter().chain([&node.bias]))
                        .copied()
                        .map(f32::abs)
                        .sum::<f32>()
                })
                .sum::<f32>()
            + updates
                .output_layer
                .nodes
                .iter()
                .flat_map(|node| node.weights.iter().chain([&node.bias]))
                .copied()
                .map(f32::abs)
                .sum::<f32>();
        let learning_coefficient = (10. / sum_gradient).min(1.);
        updates.input_layer.nodes.iter_mut().for_each(|node| {
            node.weights
                .iter_mut()
                .for_each(|w| *w *= learning_coefficient);
            node.bias *= learning_coefficient;
        });
        updates.middle_layers.iter_mut().for_each(|layer| {
            layer.nodes.iter_mut().for_each(|node| {
                node.weights
                    .iter_mut()
                    .for_each(|w| *w *= learning_coefficient);
                node.bias *= learning_coefficient;
            });
        });
        updates.output_layer.nodes.iter_mut().for_each(|node| {
            node.weights
                .iter_mut()
                .for_each(|w| *w *= learning_coefficient);
            node.bias *= learning_coefficient;
        });
        self.input_layer.flush_updates(&updates.input_layer);
        for i in 0..MIDDLE_LAYERS {
            self.middle_layers[i].flush_updates(&updates.middle_layers[i]);
        }
        self.output_layer.flush_updates(&updates.output_layer);
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
        let mut output_layer = [0.; 10];
        output_layer
            .iter_mut()
            .zip_eq(&activations.output_layer)
            .enumerate()
            .for_each(|(i, (error, &act))| {
                let layer_size = f32::from(u16::try_from(LAYER_SIZE).unwrap());
                *error = if i == label as usize {
                    10. * (act.min(layer_size) - layer_size)
                } else {
                    act.max(0.)
                };
            });
        let mut hidden_layers = [[0.; LAYER_SIZE]; MIDDLE_LAYERS + 1];
        propagate_errors(
            &mut hidden_layers[MIDDLE_LAYERS],
            &output_layer,
            &self.output_layer,
            &activations.hidden_layers[MIDDLE_LAYERS],
        );
        for i in (0..MIDDLE_LAYERS).rev() {
            let [back_errors, front_errors] = hidden_layers
                .get_many_mut([i, i + 1])
                .expect("middle layer out of bounds");
            propagate_errors(
                back_errors,
                front_errors,
                &self.middle_layers[i],
                &activations.hidden_layers[i],
            );
        }
        Activations {
            hidden_layers,
            output_layer,
        }
    }
}

fn propagate_errors<const BACK_LAYER_SIZE: usize, const FRONT_LAYER_SIZE: usize>(
    back_errors: &mut [f32; BACK_LAYER_SIZE],
    front_errors: &[f32; FRONT_LAYER_SIZE],
    nn_layer: &Layer<BACK_LAYER_SIZE, FRONT_LAYER_SIZE>,
    back_activations: &[f32; BACK_LAYER_SIZE],
) {
    nn_layer
        .nodes
        .iter()
        .zip_eq(front_errors)
        .for_each(|(node, output_error)| {
            node.weights
                .iter()
                .zip_eq(back_errors.iter_mut())
                .zip_eq(back_activations)
                .filter(|(_, &activation)| activation > 0.)
                .for_each(|((weight, input_error), _)| {
                    *input_error += weight * output_error;
                });
        });
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

    let data = train_labels
        .iter()
        .copied()
        .zip_eq(train_images.as_chunks().0)
        .collect();
    let nn = NeuralNet::<64, 1>::train(data, 20);

    let errors = test_labels
        .iter()
        .zip_eq(test_images.as_chunks().0)
        .filter(|(&label, image)| !nn.compute(image).output_layer_is_correct(label))
        .count();
    #[allow(clippy::cast_precision_loss)]
    let error_rate = errors as f64 * 100. / test_labels.len() as f64;

    println!(
        "Errors: {errors}\nTotal: {total}\nError rate: {error_rate:.2}%",
        total = test_labels.len(),
    );
}
