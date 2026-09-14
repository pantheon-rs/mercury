//! Forward tangents followed by differentiated reverse accumulation.

use super::{Linearization, Workspace};
use crate::error::{check_finite, check_len};
use crate::{Error, Result};

/// Scratch grows on the first curvature call, then belongs to this workspace.
#[derive(Default)]
pub(super) struct Scratch {
    tangents: Vec<f64>,
    bars: Vec<f64>,
    delta_bars: Vec<f64>,
    direction: Vec<f64>,
    bar: Vec<f64>,
    delta_bar: Vec<f64>,
    curved: Vec<f64>,
}

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
        let scratch = &mut self.curvature;
        scratch.tangents.resize(plan.slots, 0.0);
        scratch.tangents.fill(0.0);
        scratch.tangents[..plan.inputs].copy_from_slice(direction);
        for (node, operator) in plan.nodes.iter().zip(&mut self.operators) {
            scratch.direction.resize(node.inputs.len(), 0.0);
            for (entry, &source) in scratch.direction.iter_mut().zip(&node.sources) {
                *entry = scratch.tangents[source];
            }
            operator
                .jvp(
                    &self.values[node.inputs.clone()],
                    &scratch.direction,
                    &mut scratch.tangents[node.outputs.clone()],
                )
                .and_then(|()| {
                    check_finite("operator tangent", &scratch.tangents[node.outputs.clone()])
                })
                .map_err(|source| Error::Operator {
                    node: node.index,
                    source: Box::new(source),
                })?;
        }
        scratch.bars.resize(plan.slots, 0.0);
        scratch.bars.fill(0.0);
        scratch.delta_bars.resize(plan.slots, 0.0);
        scratch.delta_bars.fill(0.0);
        for (&source, &weight) in plan.outputs.iter().zip(weights) {
            scratch.bars[source] += weight;
        }
        check_finite("output cotangents", &scratch.bars)?;
        for (node, operator) in plan.nodes.iter().zip(&mut self.operators).rev() {
            let input = &self.values[node.inputs.clone()];
            scratch.direction.resize(node.inputs.len(), 0.0);
            for (entry, &source) in scratch.direction.iter_mut().zip(&node.sources) {
                *entry = scratch.tangents[source];
            }
            for buffer in [
                &mut scratch.bar,
                &mut scratch.delta_bar,
                &mut scratch.curved,
            ] {
                buffer.resize(node.inputs.len(), f64::NAN);
                buffer.fill(f64::NAN);
            }
            let result = (|| {
                operator.vjp(input, &scratch.bars[node.outputs.clone()], &mut scratch.bar)?;
                operator.vjp(
                    input,
                    &scratch.delta_bars[node.outputs.clone()],
                    &mut scratch.delta_bar,
                )?;
                operator.curvature(
                    input,
                    &scratch.bars[node.outputs.clone()],
                    &scratch.direction,
                    &mut scratch.curved,
                )?;
                check_finite("operator cotangent", &scratch.bar)?;
                check_finite("operator cotangent tangent", &scratch.delta_bar)?;
                check_finite("operator curvature", &scratch.curved)
            })();
            result.map_err(|source| Error::Operator {
                node: node.index,
                source: Box::new(source),
            })?;
            for (entry, &source) in node.sources.iter().enumerate() {
                scratch.bars[source] += scratch.bar[entry];
                scratch.delta_bars[source] += scratch.delta_bar[entry] + scratch.curved[entry];
                if !scratch.bars[source].is_finite() || !scratch.delta_bars[source].is_finite() {
                    return Err(Error::NonFinite("accumulated curvature"));
                }
            }
        }
        output.copy_from_slice(&scratch.delta_bars[..plan.inputs]);
        Ok(())
    }
}
