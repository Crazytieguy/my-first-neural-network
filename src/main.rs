#![feature(array_chunks)]

use std::array;

use itertools::Itertools;
use rand::random;

const IMAGE_SIZE: usize = 28 * 28;

#[derive(Debug, Clone, Copy)]
struct Edge {
    weight: u8,
    bias: u8,
}

impl Edge {
    fn new() -> Self {
        Self {
            weight: random(),
            bias: random(),
        }
    }
}

fn compute_node<const INPUT_SIZE: usize>(
    input: &[u8; INPUT_SIZE],
    edges: &[Edge; INPUT_SIZE],
) -> u8 {
    let sum: usize = input
        .iter()
        .zip(edges.iter())
        .map(|(&i, e)| {
            let weighted = i as u16 * e.weight as u16 / 255u16;
            let biased = weighted + e.bias as u16;
            (biased / 2) as u8
        } as usize)
        .sum();
    let mean = (sum / INPUT_SIZE) as u8;
    mean.saturating_sub(127) * 2 // activation
}

fn compute_layer<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize>(
    input: &[u8; INPUT_SIZE],
    edges: &[[Edge; INPUT_SIZE]; OUTPUT_SIZE],
) -> [u8; OUTPUT_SIZE] {
    let mut output = [0u8; OUTPUT_SIZE];
    output.iter_mut().zip(edges).for_each(|(node, edges)| {
        *node = compute_node(input, edges);
    });
    output
}

fn output_layer_is_correct(output: &[u8; 10], label: u8) -> bool {
    output.iter().position_max().unwrap() == label as usize
}

#[derive(Debug, Clone)]
struct NN<const LAYER_SIZE: usize, const HIDDEN_LAYERS: usize> {
    input_layer: [[Edge; IMAGE_SIZE]; LAYER_SIZE],
    hidden_layers: [[[Edge; LAYER_SIZE]; LAYER_SIZE]; HIDDEN_LAYERS],
    output_layer: [[Edge; LAYER_SIZE]; 10],
}

impl<const LAYER_SIZE: usize, const HIDDEN_LAYERS: usize> NN<LAYER_SIZE, HIDDEN_LAYERS> {
    fn new() -> Self {
        Self {
            input_layer: array::from_fn(|_| array::from_fn(|_| Edge::new())),
            hidden_layers: array::from_fn(|_| array::from_fn(|_| array::from_fn(|_| Edge::new()))),
            output_layer: array::from_fn(|_| array::from_fn(|_| Edge::new())),
        }
    }
    fn compute(&self, input: &[u8; IMAGE_SIZE]) -> [u8; 10] {
        let input_layer = compute_layer(input, &self.input_layer);
        let last_hidden_layer = self
            .hidden_layers
            .iter()
            .fold(input_layer, |input, edges| compute_layer(&input, edges));
        compute_layer(&last_hidden_layer, &self.output_layer)
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
