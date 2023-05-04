#![feature(array_chunks)]

use std::array;

use itertools::Itertools;
use rand::random;

const IMAGE_SIZE: usize = 28 * 28;

struct Node<const INPUT_SIZE: usize> {
    weights: [i8; INPUT_SIZE],
    bias: i8,
}

impl<const INPUT_SIZE: usize> Node<INPUT_SIZE> {
    fn new() -> Self {
        Self {
            weights: array::from_fn(|_| random()),
            bias: random(),
        }
    }
    fn compute(&self, input: &[u8; INPUT_SIZE]) -> u8 {
        let sum: i32 = input
            .iter()
            .zip(&self.weights)
            .map(|(&input, &weight)| input as i32 * weight as i32)
            .sum();
        let mean = sum / INPUT_SIZE as i32;
        (mean + self.bias as i32).max(0) as u8
    }
}

struct Layer<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> {
    nodes: [Node<INPUT_SIZE>; OUTPUT_SIZE],
}

impl<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> Layer<INPUT_SIZE, OUTPUT_SIZE> {
    fn new() -> Self {
        Self {
            nodes: array::from_fn(|_| Node::new()),
        }
    }
    fn compute(&self, input: &[u8; INPUT_SIZE]) -> [u8; OUTPUT_SIZE] {
        let mut output = [0u8; OUTPUT_SIZE];
        output.iter_mut().zip(&self.nodes).for_each(|(out, node)| {
            *out = node.compute(input);
        });
        output
    }
}

fn output_layer_is_correct(output: &[u8; 10], label: u8) -> bool {
    output.iter().position_max().unwrap() == label as usize
}

struct NN<const LAYER_SIZE: usize, const HIDDEN_LAYERS: usize> {
    input_layer: Layer<IMAGE_SIZE, LAYER_SIZE>,
    hidden_layers: [Layer<LAYER_SIZE, LAYER_SIZE>; HIDDEN_LAYERS],
    output_layer: Layer<LAYER_SIZE, 10>,
}

impl<const LAYER_SIZE: usize, const HIDDEN_LAYERS: usize> NN<LAYER_SIZE, HIDDEN_LAYERS> {
    fn new() -> Self {
        Self {
            input_layer: Layer::new(),
            hidden_layers: array::from_fn(|_| Layer::new()),
            output_layer: Layer::new(),
        }
    }
    fn compute(&self, input: &[u8; IMAGE_SIZE]) -> [u8; 10] {
        let input_layer_activation = self.input_layer.compute(input);
        let last_hidden_layer_activation = self
            .hidden_layers
            .iter()
            .fold(input_layer_activation, |input, layer| layer.compute(&input));
        self.output_layer.compute(&last_hidden_layer_activation)
    }
}

fn main() {
    let train_labels = &include_bytes!("../data/train-labels-idx1-ubyte")[8..];
    let train_images = &include_bytes!("../data/train-images-idx3-ubyte")[16..];
    let test_labels = &include_bytes!("../data/t10k-labels-idx1-ubyte")[8..];
    let test_images = &include_bytes!("../data/t10k-images-idx3-ubyte")[16..];

    let mut nn = NN::<32, 2>::new();
    // TODO: do some training

    let errors = test_labels
        .iter()
        .zip(test_images.array_chunks())
        .filter(|(&label, image)| {
            let output = nn.compute(image);
            !output_layer_is_correct(&output, label)
        })
        .count();
    let error_rate = errors as f64 * 100. / test_labels.len() as f64;

    println!(
        "Errors: {errors}\nTotal: {total}\nError rate: {error_rate:.2}%",
        total = test_labels.len(),
    );
}
