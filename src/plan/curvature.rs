//! Forward tangents followed by differentiated reverse accumulation.

use super::{Linearization, Workspace};
use crate::error::{check_finite, check_len};
use crate::{Error, Result};

impl Linearization<'_, '_> {
    /// Apply the Hessian of a fixed weighted sum of the published outputs.
    /// See the [example](crate::advanced#second-order-products).
    ///
    /// # Errors
    /// Rejects invalid arguments or unsupported derivative orders. A numerical
    /// or operator failure invalidates the linearization and poisons the output.
    pub fn curvature(
        &mut self,
        weights: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<()> {
        self.check_valid()?;
        let plan = self.workspace.plan;
        check_len("curvature weights", weights.len(), plan.outputs.len())?;
        check_len("curvature direction", direction.len(), plan.inputs)?;
        check_len("curvature output", output.len(), plan.inputs)?;
        check_finite("curvature weights", weights)?;
        check_finite("curvature direction", direction)?;
        output.fill(f64::NAN);
        let result = self.workspace.curvature_sweep(weights, direction, output);
        if result.is_err() {
            output.fill(f64::NAN);
            self.workspace.invalidate();
        }
        result
    }
}

impl Workspace<'_> {
    fn curvature_sweep(
        &mut self,
        weights: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<()> {
        let plan = self.plan;
        let mut tangents = vec![0.0; plan.slots];
        tangents[..plan.inputs].copy_from_slice(direction);
        for (node, operator) in plan.nodes.iter().zip(&mut self.operators) {
            let seed: Vec<_> = node
                .sources
                .iter()
                .map(|&source| tangents[source])
                .collect();
            operator
                .jvp(
                    &self.values[node.inputs.clone()],
                    &seed,
                    &mut tangents[node.outputs.clone()],
                )
                .and_then(|()| check_finite("operator tangent", &tangents[node.outputs.clone()]))
                .map_err(|source| Error::Operator {
                    node: node.index,
                    source: Box::new(source),
                })?;
        }
        let mut bars = vec![0.0; plan.slots];
        let mut delta_bars = vec![0.0; plan.slots];
        for (&source, &weight) in plan.outputs.iter().zip(weights) {
            bars[source] += weight;
        }
        check_finite("output cotangents", &bars)?;
        for (node, operator) in plan.nodes.iter().zip(&mut self.operators).rev() {
            let input = &self.values[node.inputs.clone()];
            let direction: Vec<_> = node
                .sources
                .iter()
                .map(|&source| tangents[source])
                .collect();
            let mut bar = vec![0.0; node.inputs.len()];
            let mut delta_bar = vec![0.0; node.inputs.len()];
            let mut curved = vec![0.0; node.inputs.len()];
            let result = (|| {
                operator.vjp(input, &bars[node.outputs.clone()], &mut bar)?;
                operator.vjp(input, &delta_bars[node.outputs.clone()], &mut delta_bar)?;
                operator.curvature(input, &bars[node.outputs.clone()], &direction, &mut curved)?;
                check_finite("operator cotangent", &bar)?;
                check_finite("operator cotangent tangent", &delta_bar)?;
                check_finite("operator curvature", &curved)
            })();
            result.map_err(|source| Error::Operator {
                node: node.index,
                source: Box::new(source),
            })?;
            for (entry, &source) in node.sources.iter().enumerate() {
                bars[source] += bar[entry];
                delta_bars[source] += delta_bar[entry] + curved[entry];
                if !bars[source].is_finite() || !delta_bars[source].is_finite() {
                    return Err(Error::NonFinite("accumulated curvature"));
                }
            }
        }
        output.copy_from_slice(&delta_bars[..plan.inputs]);
        Ok(())
    }
}
