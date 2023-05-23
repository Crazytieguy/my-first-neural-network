use std::array;

use itertools::Itertools;

#[derive(Debug)]
pub(crate) struct Activations<const LAYER_SIZE: usize, const HIDDEN_LAYERS: usize> {
    pub(crate) hidden_layers: [[f32; LAYER_SIZE]; HIDDEN_LAYERS],
    pub(crate) output_layer: [f32; 10],
}

impl<const LAYER_SIZE: usize, const HIDDEN_LAYERS: usize> Activations<LAYER_SIZE, HIDDEN_LAYERS> {
    pub(crate) fn output_layer_is_correct(&self, label: u8) -> bool {
        self.output_layer
            .iter()
            .position_max_by(|a, b| a.partial_cmp(b).expect("NaN in output layer"))
            .expect("Empty output layer")
            == label as usize
    }
    pub(crate) fn output_errors(&self, label: u8) -> [f32; 10] {
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
