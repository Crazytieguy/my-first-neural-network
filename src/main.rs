#![feature(slice_as_chunks)]
#![feature(generic_const_exprs)]
#![feature(get_many_mut)]
#![warn(clippy::pedantic)]

use itertools::izip;

mod activations;
mod layer;
mod network;

const IMAGE_SIZE: usize = 28 * 28;

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
    let nn = network::NeuralNet::<800, 0>::train(data, 40);

    let errors = izip!(test_labels, test_images.as_chunks().0)
        .filter(|(&label, image)| !nn.compute(image).output_layer_is_correct(label))
        .count();
    #[allow(clippy::cast_precision_loss)]
    let error_rate = errors as f64 * 100. / test_labels.len() as f64;

    println!("Error rate: {error_rate:.2}%",);
}
