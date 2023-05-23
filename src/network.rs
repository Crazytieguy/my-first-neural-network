use indicatif::ProgressBar;
use itertools::izip;
use rand::seq::SliceRandom;
use std::array;

use super::IMAGE_SIZE;
use crate::activations::Activations;
use crate::layer::Layer;

pub(crate) struct NeuralNet<const LAYER_SIZE: usize, const MIDDLE_LAYERS: usize> {
    pub(crate) input_layer: Layer<IMAGE_SIZE, LAYER_SIZE, true>,
    pub(crate) middle_layers: [Layer<LAYER_SIZE, LAYER_SIZE, true>; MIDDLE_LAYERS],
    pub(crate) output_layer: Layer<LAYER_SIZE, 10, false>,
}

impl<const LAYER_SIZE: usize, const MIDDLE_LAYERS: usize> NeuralNet<LAYER_SIZE, MIDDLE_LAYERS>
where
    [(); MIDDLE_LAYERS + 1]: Copy,
{
    pub(crate) fn zeros() -> Self {
        Self {
            input_layer: Layer::zeros(),
            middle_layers: array::from_fn(|_| Layer::zeros()),
            output_layer: Layer::zeros(),
        }
    }
    pub(crate) fn random() -> Self {
        Self {
            input_layer: Layer::random(),
            middle_layers: array::from_fn(|_| Layer::random()),
            output_layer: Layer::random(),
        }
    }
    pub(crate) fn train(mut data: Vec<(u8, &[u8; IMAGE_SIZE])>, epochs: usize) -> Self {
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
    pub(crate) fn add_gradient(
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
    pub(crate) fn gradient_size_squared(&self) -> f32 {
        self.input_layer.gradient_size_squared()
            + self
                .middle_layers
                .iter()
                .map(Layer::gradient_size_squared)
                .sum::<f32>()
            + self.output_layer.gradient_size_squared()
    }
    pub(crate) fn apply_gradient(&mut self, gradient: &Self) {
        let learning_rate = (10. / gradient.gradient_size_squared().sqrt()).min(1.);
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
    pub(crate) fn compute(
        &self,
        input: &[u8; IMAGE_SIZE],
    ) -> Activations<LAYER_SIZE, { MIDDLE_LAYERS + 1 }> {
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
    pub(crate) fn errors(
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
