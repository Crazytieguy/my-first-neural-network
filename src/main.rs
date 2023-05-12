#![feature(slice_as_chunks)]
#![feature(generic_const_exprs)]
#![feature(get_many_mut)]
#![warn(clippy::pedantic)]

use std::array;

use indicatif::ProgressBar;
use itertools::Itertools;
use rand::{random, seq::SliceRandom};

const IMAGE_SIZE: usize = 28 * 28;
const LEARNING_RATE: f32 = 0.1;

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
            weights: array::from_fn(|_| random()),
            bias: random(),
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
        self.bias -= LEARNING_RATE * error;
        self.weights
            .iter_mut()
            .zip_eq(input)
            .for_each(|(weight, &input)| {
                *weight -= input.max(0.) * LEARNING_RATE * error;
            });
    }
    fn flush_updates(&mut self, updates: &mut Self) {
        self.bias += updates.bias;
        updates.bias = 0.;
        self.weights
            .iter_mut()
            .zip_eq(&mut updates.weights)
            .for_each(|(weight, update)| {
                *weight += *update;
                *update = 0.;
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
    fn flush_updates(&mut self, updates: &mut Self) {
        self.nodes
            .iter_mut()
            .zip_eq(&mut updates.nodes)
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
            for (n, &(label, input)) in data.iter().enumerate() {
                if n % 100 == 99 {
                    net.flush_updates(&mut updates);
                    bar.inc(1);
                }
                let activations = net.compute(input);
                let errors = net.errors(&activations, label);
                updates.update(input, &activations, &errors);
            }
            net.flush_updates(&mut updates);
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
            &input.map(|b| f32::from(b) / 255.),
            &errors.hidden_layers[0],
        );
    }
    fn flush_updates(&mut self, updates: &mut Self) {
        self.input_layer.flush_updates(&mut updates.input_layer);
        for i in 0..MIDDLE_LAYERS {
            self.middle_layers[i].flush_updates(&mut updates.middle_layers[i]);
        }
        self.output_layer.flush_updates(&mut updates.output_layer);
    }
    fn compute(&self, input: &[u8; IMAGE_SIZE]) -> Activations<LAYER_SIZE, { MIDDLE_LAYERS + 1 }> {
        let mut hidden_layers = [[0.; LAYER_SIZE]; MIDDLE_LAYERS + 1];
        hidden_layers[0] = self
            .input_layer
            .compute(&input.map(|b| f32::from(b) / 255.));
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
                *error = if i == label as usize {
                    act.min(256.) - 256.
                } else {
                    act.max(-256.) + 256.
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
    let nn = NeuralNet::<32, 1>::train(data, 10);

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
