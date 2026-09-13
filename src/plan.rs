//! Validated wiring, deterministic execution, and point-bound derivatives.

mod curvature;

use std::collections::{BTreeSet, VecDeque};
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{check_finite, check_len};
use crate::{Error, Operator, OperatorWorkspace, PlanExecution, Result, Shape};

static NEXT_EPOCH: AtomicU64 = AtomicU64::new(1);

/// A node belonging to one plan builder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeId {
    epoch: u64,
    index: usize,
}

impl NodeId {
    /// Select a scalar output of this node.
    ///
    /// See the [example](crate#build-a-graph).
    pub const fn output(self, index: usize) -> Source {
        Source::Node(self, index)
    }
}

/// Source of one active scalar input or published output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// An input of the whole plan.
    Input(usize),
    /// An output of an operator instance.
    Node(NodeId, usize),
}

struct PendingNode {
    operator: Box<dyn Operator>,
    sources: Vec<Source>,
}

/// Editable wiring; building consumes it and publishes an immutable plan.
pub struct PlanBuilder {
    epoch: u64,
    inputs: usize,
    nodes: Vec<PendingNode>,
}

impl PlanBuilder {
    /// Add an instance. Connections are checked when the plan is built.
    ///
    /// See the [example](crate#build-a-graph).
    pub fn add(
        &mut self,
        operator: impl Operator + 'static,
        sources: impl Into<Vec<Source>>,
    ) -> NodeId {
        let id = NodeId {
            epoch: self.epoch,
            index: self.nodes.len(),
        };
        self.nodes.push(PendingNode {
            operator: Box::new(operator),
            sources: sources.into(),
        });
        id
    }

    /// Replace a node's inputs, including forward references to other nodes.
    ///
    /// See the [example](crate#build-a-graph).
    ///
    /// # Errors
    /// Rejects a node from another builder or a missing node.
    pub fn connect(&mut self, node: NodeId, sources: impl Into<Vec<Source>>) -> Result<()> {
        if node.epoch != self.epoch {
            return Err(Error::ForeignNode);
        }
        let pending = self.nodes.get_mut(node.index).ok_or(Error::InvalidSource)?;
        pending.sources = sources.into();
        Ok(())
    }

    /// Validate wiring and cycles, then fix execution order and buffer layouts.
    ///
    /// See the [example](crate#build-a-graph).
    ///
    /// # Errors
    /// Rejects invalid dimensions, references, cycles, and overflowing layouts.
    pub fn build(self, outputs: impl Into<Vec<Source>>) -> Result<Plan> {
        let shapes: Vec<_> = self
            .nodes
            .iter()
            .map(|node| node.operator.shape())
            .collect();
        let mut slots = self.inputs;
        let mut ranges = Vec::with_capacity(shapes.len());
        for shape in &shapes {
            let middle = slots.checked_add(shape.inputs).ok_or(Error::SizeOverflow)?;
            let end = middle
                .checked_add(shape.outputs)
                .ok_or(Error::SizeOverflow)?;
            ranges.push((slots..middle, middle..end));
            slots = end;
        }
        std::alloc::Layout::array::<BTreeSet<usize>>(slots).map_err(|_| Error::SizeOverflow)?;
        let resolve = |source: Source| -> Result<usize> {
            match source {
                Source::Input(index) if index < self.inputs => Ok(index),
                Source::Node(id, index) => {
                    if id.epoch != self.epoch {
                        return Err(Error::ForeignNode);
                    }
                    let (_, range) = ranges.get(id.index).ok_or(Error::InvalidSource)?;
                    if index >= range.len() {
                        return Err(Error::InvalidSource);
                    }
                    Ok(range.start + index)
                }
                Source::Input(_) => Err(Error::InvalidSource),
            }
        };
        let outputs: Vec<_> = outputs
            .into()
            .into_iter()
            .map(resolve)
            .collect::<Result<_>>()?;
        let mut consumers = vec![Vec::new(); self.nodes.len()];
        let mut indegree = vec![0; self.nodes.len()];
        let mut sources = Vec::with_capacity(self.nodes.len());
        for (index, node) in self.nodes.iter().enumerate() {
            check_len("node inputs", node.sources.len(), shapes[index].inputs)?;
            sources.push(
                node.sources
                    .iter()
                    .copied()
                    .map(resolve)
                    .collect::<Result<Vec<_>>>()?,
            );
            for source in &node.sources {
                if let Source::Node(id, _) = source {
                    consumers[id.index].push(index);
                    indegree[index] += 1;
                }
            }
        }
        let order = execution_order(&consumers, indegree)?;
        let mut rank = vec![0; order.len()];
        for (position, index) in order.into_iter().enumerate() {
            rank[index] = position;
        }
        let mut nodes: Vec<_> = self
            .nodes
            .into_iter()
            .enumerate()
            .map(|(index, node)| Node {
                index,
                operator: node.operator,
                sources: std::mem::take(&mut sources[index]),
                inputs: ranges[index].0.clone(),
                outputs: ranges[index].1.clone(),
            })
            .collect();
        nodes.sort_by_key(|node| rank[node.index]);
        let dependencies = output_dependencies(self.inputs, slots, &nodes, &outputs);
        let sparsity = Arc::new(crate::sparse::Sparsity::new(self.inputs, &dependencies));
        Ok(Plan {
            epoch: self.epoch,
            inputs: self.inputs,
            nodes: nodes.into(),
            outputs: outputs.into(),
            slots,
            dependencies: dependencies.into(),
            sparsity,
        })
    }
}

struct Node {
    index: usize,
    operator: Box<dyn Operator>,
    sources: Vec<usize>,
    inputs: Range<usize>,
    outputs: Range<usize>,
}

fn execution_order(consumers: &[Vec<usize>], mut indegree: Vec<usize>) -> Result<Vec<usize>> {
    let mut ready: VecDeque<_> = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, &degree)| (degree == 0).then_some(index))
        .collect();
    let mut order = Vec::with_capacity(consumers.len());
    while let Some(index) = ready.pop_front() {
        order.push(index);
        for &consumer in &consumers[index] {
            indegree[consumer] -= 1;
            if indegree[consumer] == 0 {
                ready.push_back(consumer);
            }
        }
    }
    if order.len() != consumers.len() {
        return Err(Error::Cycle);
    }
    Ok(order)
}

fn output_dependencies(
    inputs: usize,
    slots: usize,
    nodes: &[Node],
    outputs: &[usize],
) -> Vec<Vec<usize>> {
    let mut dependencies = vec![BTreeSet::new(); slots];
    for (index, entries) in dependencies.iter_mut().take(inputs).enumerate() {
        entries.insert(index);
    }
    for node in nodes {
        for output in 0..node.outputs.len() {
            let mut entries = BTreeSet::new();
            for (input, &source) in node.sources.iter().enumerate() {
                if node.operator.depends_on(output, input) {
                    entries.extend(&dependencies[source]);
                }
            }
            dependencies[node.outputs.start + output] = entries;
        }
    }
    outputs
        .iter()
        .map(|&index| dependencies[index].iter().copied().collect())
        .collect()
}

/// Immutable numerical wiring and conservative output dependencies.
#[derive(Clone)]
pub struct Plan {
    epoch: u64,
    inputs: usize,
    nodes: Arc<[Node]>,
    outputs: Arc<[usize]>,
    slots: usize,
    dependencies: Arc<[Vec<usize>]>,
    sparsity: Arc<crate::sparse::Sparsity>,
}

impl Plan {
    /// Start a new plan epoch with this many active scalar inputs.
    ///
    /// See the [example](crate#build-a-graph).
    ///
    /// # Panics
    /// Panics if the process exhausts all `u64` plan identities.
    pub fn builder(inputs: usize) -> PlanBuilder {
        let epoch = NEXT_EPOCH
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("plan epoch space exhausted");
        PlanBuilder {
            epoch,
            inputs,
            nodes: Vec::new(),
        }
    }

    /// Wrap one numerical operator as a directly callable function.
    ///
    /// See the [example](crate#one-operator-as-a-function).
    ///
    /// # Errors
    /// Rejects overflowing dimensions.
    pub fn from_operator(operator: impl Operator + 'static) -> Result<Self> {
        let shape = operator.shape();
        for size in [shape.inputs, shape.outputs] {
            std::alloc::Layout::array::<Source>(size).map_err(|_| Error::SizeOverflow)?;
        }
        let mut builder = Self::builder(shape.inputs);
        let node = builder.add(
            operator,
            (0..shape.inputs).map(Source::Input).collect::<Vec<_>>(),
        );
        builder.build(
            (0..shape.outputs)
                .map(|index| node.output(index))
                .collect::<Vec<_>>(),
        )
    }

    /// Evaluate the published outputs, returning an owned vector.
    ///
    /// See the [example](crate#one-operator-as-a-function).
    ///
    /// # Errors
    /// Rejects invalid inputs and propagates numerical errors.
    pub fn eval(&self, point: &[f64]) -> Result<Vec<f64>> {
        self.check_point(point)?;
        let mut workspace = self.workspace();
        self.run(point, &mut workspace, false)?;
        Ok(workspace.result)
    }

    /// Select the gradient of a plan with exactly one published output.
    ///
    /// See the [example](crate#one-operator-as-a-function).
    /// The output count is checked when the gradient is evaluated.
    pub fn gradient(&self) -> Gradient {
        Gradient { plan: self.clone() }
    }

    /// Select the Jacobian, with output rows and input columns.
    ///
    /// See the [example](crate#one-operator-as-a-function).
    pub fn jacobian(&self) -> Jacobian {
        Jacobian { plan: self.clone() }
    }

    /// Evaluate a scalar plan's value and gradient together.
    ///
    /// See the [example](crate#one-operator-as-a-function).
    ///
    /// # Errors
    /// Requires one published output; rejects invalid inputs and numerical failures.
    pub fn value_and_gradient(&self, point: &[f64]) -> Result<(f64, Vec<f64>)> {
        check_len("gradient outputs", self.outputs.len(), 1)?;
        self.check_point(point)?;
        let mut workspace = self.workspace();
        let mut linearization = self.linearize(point, &mut workspace)?;
        let mut gradient = vec![0.0; self.inputs];
        linearization.vjp(&[1.0], &mut gradient)?;
        Ok((linearization.value()?[0], gradient))
    }
}

/// A scalar plan's gradient, evaluated with the same input order as the plan.
#[derive(Clone)]
pub struct Gradient {
    plan: Plan,
}

impl Gradient {
    /// Select the Jacobian of this gradient: the scalar plan's Hessian.
    /// See the [example](crate#second-derivatives).
    pub fn jacobian(&self) -> Hessian {
        Hessian {
            plan: self.plan.clone(),
        }
    }

    /// Evaluate the gradient and return an owned vector.
    ///
    /// See the [example](crate#one-operator-as-a-function).
    ///
    /// # Errors
    /// Requires one published output; rejects invalid inputs and numerical failures.
    pub fn eval(&self, point: &[f64]) -> Result<Vec<f64>> {
        Ok(self.plan.value_and_gradient(point)?.1)
    }
}

/// A scalar plan's Hessian, with input rows and input columns.
#[derive(Clone)]
pub struct Hessian {
    plan: Plan,
}

impl Hessian {
    /// Evaluate second derivatives using the plan's weighted curvature rules.
    /// See the [example](crate#second-derivatives).
    ///
    /// # Errors
    /// Requires a scalar plan and second-order rules on its operators.
    /// Rejects invalid inputs, overflowing dimensions, and numerical failures.
    pub fn eval(&self, point: &[f64]) -> Result<faer::Mat<f64>> {
        check_len("Hessian outputs", self.plan.outputs.len(), 1)?;
        self.plan.check_point(point)?;
        let n = self.plan.inputs;
        let length = n.checked_mul(n).ok_or(Error::SizeOverflow)?;
        std::alloc::Layout::array::<f64>(length).map_err(|_| Error::SizeOverflow)?;
        let mut matrix = faer::Mat::zeros(n, n);
        let mut workspace = self.plan.workspace();
        let mut linearization = self.plan.linearize(point, &mut workspace)?;
        let mut seed = vec![0.0; n];
        let mut product = vec![0.0; n];
        for column in 0..n {
            seed.fill(0.0);
            seed[column] = 1.0;
            linearization.curvature(&[1.0], &seed, &mut product)?;
            for row in 0..n {
                matrix[(row, column)] = product[row];
            }
        }
        Ok(matrix)
    }
}

/// A plan's Jacobian, stored in an owned faer matrix.
#[derive(Clone)]
pub struct Jacobian {
    plan: Plan,
}

impl Jacobian {
    /// The fixed CSC pattern, including entries that happen to be zero at a point.
    /// See the [example](crate#sparse-jacobians).
    pub fn sparsity(&self) -> faer::sparse::SymbolicSparseColMatRef<'_, usize> {
        self.plan.sparsity.symbolic.as_ref()
    }

    /// Evaluate into an owned faer sparse matrix using one JVP per column color.
    /// See the [example](crate#sparse-jacobians).
    ///
    /// # Errors
    /// Rejects invalid inputs and propagates numerical failures.
    pub fn eval_sparse(&self, point: &[f64]) -> Result<faer::sparse::SparseColMat<usize, f64>> {
        self.plan.check_point(point)?;
        let mut values = vec![0.0; self.plan.sparsity.symbolic.row_idx().len()];
        let mut workspace = self.plan.workspace();
        self.plan
            .linearize(point, &mut workspace)?
            .sparse_jacobian(&mut values)?;
        Ok(faer::sparse::SparseColMat::new(
            self.plan.sparsity.symbolic.clone(),
            values,
        ))
    }

    /// Evaluate all partial derivatives; index as `matrix[(output, input)]`.
    ///
    /// See the [example](crate#one-operator-as-a-function).
    ///
    /// # Errors
    /// Rejects invalid inputs, overflowing dimensions, and numerical failures.
    pub fn eval(&self, point: &[f64]) -> Result<faer::Mat<f64>> {
        self.plan.check_point(point)?;
        let rows = self.plan.outputs.len();
        let columns = self.plan.inputs;
        let length = rows.checked_mul(columns).ok_or(Error::SizeOverflow)?;
        std::alloc::Layout::array::<f64>(length).map_err(|_| Error::SizeOverflow)?;
        let mut values = vec![0.0; length];
        let mut workspace = self.plan.workspace();
        self.plan
            .linearize(point, &mut workspace)?
            .jacobian(&mut values)?;
        Ok(faer::Mat::from_fn(rows, columns, |row, column| {
            values[row * columns + column]
        }))
    }
}

impl PlanExecution for Plan {
    fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Sorted input indices that may affect each published output.
    fn dependencies(&self) -> &[Vec<usize>] {
        &self.dependencies
    }

    /// Evaluate values. Output is valid only on success.
    ///
    /// # Errors
    /// Rejects a foreign workspace or invalid buffers, and propagates operator failures.
    fn evaluate(
        &self,
        point: &[f64],
        workspace: &mut Workspace<'_>,
        output: &mut [f64],
    ) -> Result<()> {
        self.check(point, workspace)?;
        check_len("output", output.len(), self.outputs.len())?;
        output.fill(f64::NAN);
        self.run(point, workspace, false)?;
        output.copy_from_slice(&workspace.result);
        Ok(())
    }

    /// Bind derivatives to this point and an exclusive workspace borrow.
    ///
    /// # Errors
    /// Rejects a foreign workspace or invalid point, and propagates operator failures.
    fn linearize<'a, 'p>(
        &'p self,
        point: &'a [f64],
        workspace: &'a mut Workspace<'p>,
    ) -> Result<Linearization<'a, 'p>> {
        self.check(point, workspace)?;
        self.run(point, workspace, true)?;
        Ok(Linearization { point, workspace })
    }
}

impl Plan {
    /// Allocate reusable storage for this plan. Concurrent callers need separate workspaces.
    pub(crate) fn workspace(&self) -> Workspace<'_> {
        Workspace {
            plan: self,
            operators: self
                .nodes
                .iter()
                .map(|node| node.operator.workspace())
                .collect(),
            values: vec![f64::NAN; self.slots],
            result: vec![f64::NAN; self.outputs.len()],
            derivatives: Vec::new(),
            seeds: Vec::new(),
            products: Vec::new(),
            valid: false,
        }
    }

    fn check(&self, point: &[f64], workspace: &Workspace<'_>) -> Result<()> {
        if !std::ptr::eq(self, workspace.plan) {
            return Err(Error::ForeignWorkspace);
        }
        self.check_point(point)
    }

    fn check_point(&self, point: &[f64]) -> Result<()> {
        check_len("point", point.len(), self.inputs)?;
        check_finite("point", point)
    }

    fn run(&self, point: &[f64], workspace: &mut Workspace<'_>, prepare: bool) -> Result<()> {
        workspace.invalidate();
        workspace.values[..self.inputs].copy_from_slice(point);
        for (node, operator) in self.nodes.iter().zip(&mut workspace.operators) {
            for (index, &source) in node.sources.iter().enumerate() {
                workspace.values[node.inputs.start + index] = workspace.values[source];
            }
            let (before, after) = workspace.values.split_at_mut(node.outputs.start);
            let input = &before[node.inputs.clone()];
            let output = &mut after[..node.outputs.len()];
            output.fill(f64::NAN);
            let result = if prepare {
                operator.linearize(input, output)
            } else {
                operator.evaluate(input, output)
            };
            if let Err(source) = result.and_then(|()| check_finite("operator value", output)) {
                let node = node.index;
                workspace.invalidate();
                return Err(Error::Operator {
                    node,
                    source: Box::new(source),
                });
            }
        }
        for (output, &source) in workspace.result.iter_mut().zip(self.outputs.iter()) {
            *output = workspace.values[source];
        }
        workspace.valid = prepare;
        Ok(())
    }
}

impl Operator for Plan {
    fn shape(&self) -> Shape {
        Shape {
            inputs: self.inputs,
            outputs: self.outputs.len(),
        }
    }

    fn workspace(&self) -> Box<dyn OperatorWorkspace + '_> {
        Box::new(self.workspace())
    }

    fn depends_on(&self, output: usize, input: usize) -> bool {
        self.dependencies[output].binary_search(&input).is_ok()
    }
}

/// Mutable numerical storage borrowing exactly one immutable plan.
pub struct Workspace<'p> {
    plan: &'p Plan,
    operators: Vec<Box<dyn OperatorWorkspace + 'p>>,
    values: Vec<f64>,
    result: Vec<f64>,
    derivatives: Vec<f64>,
    seeds: Vec<f64>,
    products: Vec<f64>,
    valid: bool,
}

impl<'p> Workspace<'p> {
    /// Allocate reusable execution storage for one plan.
    ///
    /// See the [example](crate::advanced#plan-execution).
    pub fn new(plan: &'p Plan) -> Self {
        plan.workspace()
    }

    fn invalidate(&mut self) {
        self.valid = false;
        self.result.fill(f64::NAN);
        for operator in &mut self.operators {
            operator.invalidate();
        }
    }
}

impl Operator for Gradient {
    fn shape(&self) -> Shape {
        Shape {
            inputs: self.plan.inputs,
            outputs: self.plan.inputs,
        }
    }
    fn workspace(&self) -> Box<dyn OperatorWorkspace + '_> {
        crate::derivative::derivative_workspace(&self.plan, true)
    }
}

impl Operator for Jacobian {
    fn shape(&self) -> Shape {
        Shape {
            inputs: self.plan.inputs,
            outputs: self.plan.inputs.saturating_mul(self.plan.outputs.len()),
        }
    }
    fn workspace(&self) -> Box<dyn OperatorWorkspace + '_> {
        crate::derivative::derivative_workspace(&self.plan, false)
    }
}

impl OperatorWorkspace for Workspace<'_> {
    fn evaluate(&mut self, input: &[f64], output: &mut [f64]) -> Result<()> {
        self.plan.evaluate(input, self, output)
    }

    fn linearize(&mut self, input: &[f64], output: &mut [f64]) -> Result<()> {
        let plan = self.plan;
        plan.check(input, self)?;
        check_len("output", output.len(), plan.outputs.len())?;
        plan.run(input, self, true)?;
        output.copy_from_slice(&self.result);
        Ok(())
    }

    /// See the [example](crate::advanced#linearization-products).
    ///
    fn jvp(&mut self, input: &[f64], seed: &[f64], output: &mut [f64]) -> Result<()> {
        self.jvp_batch(input, 1, seed, output)
    }

    /// See the [example](crate::advanced#linearization-products).
    ///
    fn vjp(&mut self, input: &[f64], seed: &[f64], output: &mut [f64]) -> Result<()> {
        self.vjp_batch(input, 1, seed, output)
    }

    /// See the [example](crate::advanced#linearization-products).
    ///
    fn jvp_batch(
        &mut self,
        input: &[f64],
        count: usize,
        seeds: &[f64],
        output: &mut [f64],
    ) -> Result<()> {
        Linearization {
            point: input,
            workspace: self,
        }
        .jvp_batch(count, seeds, output)
    }

    /// See the [example](crate::advanced#linearization-products).
    ///
    fn vjp_batch(
        &mut self,
        input: &[f64],
        count: usize,
        seeds: &[f64],
        output: &mut [f64],
    ) -> Result<()> {
        Linearization {
            point: input,
            workspace: self,
        }
        .vjp_batch(count, seeds, output)
    }

    fn curvature(
        &mut self,
        input: &[f64],
        weights: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<()> {
        Linearization {
            point: input,
            workspace: self,
        }
        .curvature(weights, direction, output)
    }

    fn invalidate(&mut self) {
        Self::invalidate(self);
    }
}

/// Values and derivative products at one borrowed point and plan.
///
/// The point cannot change while its linearization is still used:
///
/// ```compile_fail
/// use mercury::{Plan, Source};
/// let plan = Plan::builder(1).build([Source::Input(0)]).unwrap();
/// let mut workspace = mercury::advanced::Workspace::new(&plan);
/// let mut point = [1.0];
/// use mercury::advanced::PlanExecution;
/// let linearization = plan.linearize(&point, &mut workspace).unwrap();
/// point[0] = 2.0;
/// assert!(linearization.value().is_ok());
/// ```
pub struct Linearization<'a, 'p> {
    point: &'a [f64],
    workspace: &'a mut Workspace<'p>,
}

impl Linearization<'_, '_> {
    /// The successfully prepared primal value.
    ///
    /// See the [example](crate::advanced#linearization-products).
    ///
    /// # Errors
    /// Returns `InvalidLinearization` after a numerical failure.
    pub fn value(&self) -> Result<&[f64]> {
        self.check_valid()?;
        Ok(&self.workspace.result)
    }

    /// Apply one input tangent.
    ///
    /// See the [example](crate::advanced#linearization-products).
    ///
    /// # Errors
    /// Same validation and numerical failures as [`Self::jvp_batch`].
    pub fn jvp(&mut self, seed: &[f64], output: &mut [f64]) -> Result<()> {
        self.jvp_batch(1, seed, output)
    }

    /// Apply one output cotangent.
    ///
    /// See the [example](crate::advanced#linearization-products).
    ///
    /// # Errors
    /// Same validation and numerical failures as [`Self::vjp_batch`].
    pub fn vjp(&mut self, seed: &[f64], output: &mut [f64]) -> Result<()> {
        self.vjp_batch(1, seed, output)
    }

    /// Apply `count` contiguous input seed rows and overwrite contiguous output rows.
    ///
    /// See the [example](crate::advanced#linearization-products).
    ///
    /// # Errors
    /// Rejects invalid linearizations, shapes, size overflow, and non-finite seeds.
    /// A numerical failure invalidates this linearization and the entire output.
    pub fn jvp_batch(&mut self, count: usize, seeds: &[f64], output: &mut [f64]) -> Result<()> {
        self.products(count, seeds, output, false)
    }

    /// Apply `count` contiguous output seed rows and overwrite contiguous input rows.
    ///
    /// See the [example](crate::advanced#linearization-products).
    ///
    /// # Errors
    /// Same validation and failure semantics as [`Self::jvp_batch`].
    pub fn vjp_batch(&mut self, count: usize, seeds: &[f64], output: &mut [f64]) -> Result<()> {
        self.products(count, seeds, output, true)
    }

    /// Assemble the dense Jacobian in row-major output-by-input order.
    ///
    /// See the [example](crate::advanced#linearization-products).
    ///
    /// # Errors
    /// Rejects invalid dimensions or linearizations, and propagates product failures.
    pub fn jacobian(&mut self, output: &mut [f64]) -> Result<()> {
        self.check_valid()?;
        let shape = self.workspace.plan.shape();
        check_len(
            "Jacobian",
            output.len(),
            shape
                .inputs
                .checked_mul(shape.outputs)
                .ok_or(Error::SizeOverflow)?,
        )?;
        let plan = self.workspace.plan;
        let mut values = vec![0.0; plan.sparsity.symbolic.row_idx().len()];
        output.fill(f64::NAN);
        self.sparse_jacobian(&mut values)?;
        output.fill(0.0);
        let pattern = &plan.sparsity.symbolic;
        for column in 0..shape.inputs {
            for entry in pattern.col_ptr()[column]..pattern.col_ptr()[column + 1] {
                output[pattern.row_idx()[entry] * shape.inputs + column] = values[entry];
            }
        }
        Ok(())
    }

    /// Write CSC values in the order returned by the Jacobian's sparsity method.
    /// See the [example](crate::advanced#sparse-storage).
    ///
    /// # Errors
    /// Rejects invalid lengths or linearizations; numerical failures poison all values.
    pub fn sparse_jacobian(&mut self, output: &mut [f64]) -> Result<()> {
        self.check_valid()?;
        let plan = self.workspace.plan;
        let pattern = &plan.sparsity.symbolic;
        check_len("sparse Jacobian", output.len(), pattern.row_idx().len())?;
        output.fill(f64::NAN);
        let mut seed = vec![0.0; plan.inputs];
        let mut product = vec![0.0; plan.outputs.len()];
        for columns in &plan.sparsity.colors {
            seed.fill(0.0);
            for &column in columns {
                seed[column] = 1.0;
            }
            if let Err(error) = self.jvp(&seed, &mut product) {
                output.fill(f64::NAN);
                return Err(error);
            }
            for &column in columns {
                for entry in pattern.col_ptr()[column]..pattern.col_ptr()[column + 1] {
                    output[entry] = product[pattern.row_idx()[entry]];
                }
            }
        }
        Ok(())
    }

    const fn check_valid(&self) -> Result<()> {
        if !self.workspace.valid {
            return Err(Error::InvalidLinearization);
        }
        Ok(())
    }

    fn products(
        &mut self,
        count: usize,
        seeds: &[f64],
        output: &mut [f64],
        reverse: bool,
    ) -> Result<()> {
        self.check_valid()?;
        let plan = self.workspace.plan;
        let (seed_size, output_size) = if reverse {
            (plan.outputs.len(), self.point.len())
        } else {
            (self.point.len(), plan.outputs.len())
        };
        check_len(
            "seeds",
            seeds.len(),
            count.checked_mul(seed_size).ok_or(Error::SizeOverflow)?,
        )?;
        check_len(
            "products",
            output.len(),
            count.checked_mul(output_size).ok_or(Error::SizeOverflow)?,
        )?;
        check_finite("seeds", seeds)?;
        let slots = count.checked_mul(plan.slots).ok_or(Error::SizeOverflow)?;
        std::alloc::Layout::array::<f64>(slots).map_err(|_| Error::SizeOverflow)?;
        if count == 0 || (plan.slots == 0 && plan.nodes.is_empty()) {
            return Ok(());
        }
        output.fill(f64::NAN);
        let result = self.sweep(count, seeds, output, slots, reverse);
        if result.is_err() {
            output.fill(f64::NAN);
            self.workspace.invalidate();
        }
        result
    }

    fn sweep(
        &mut self,
        count: usize,
        seeds: &[f64],
        output: &mut [f64],
        slots: usize,
        reverse: bool,
    ) -> Result<()> {
        let workspace = &mut *self.workspace;
        let plan = workspace.plan;
        workspace.derivatives.resize(slots, 0.0);
        workspace.derivatives.fill(0.0);
        for row in 0..count {
            let values = &mut workspace.derivatives[row * plan.slots..(row + 1) * plan.slots];
            if reverse {
                for (index, &destination) in plan.outputs.iter().enumerate() {
                    values[destination] += seeds[row * plan.outputs.len() + index];
                }
                check_finite("output cotangents", values)?;
            } else {
                values[..plan.inputs]
                    .copy_from_slice(&seeds[row * plan.inputs..(row + 1) * plan.inputs]);
            }
        }
        for step in 0..plan.nodes.len() {
            let index = if reverse {
                plan.nodes.len() - 1 - step
            } else {
                step
            };
            let node = &plan.nodes[index];
            let (seed_size, product_size) = if reverse {
                (node.outputs.len(), node.inputs.len())
            } else {
                (node.inputs.len(), node.outputs.len())
            };
            workspace.seeds.resize(count * seed_size, 0.0);
            workspace.products.resize(count * product_size, f64::NAN);
            workspace.products.fill(f64::NAN);
            for row in 0..count {
                let values = &workspace.derivatives[row * plan.slots..(row + 1) * plan.slots];
                for entry in 0..seed_size {
                    let source = if reverse {
                        node.outputs.start + entry
                    } else {
                        node.sources[entry]
                    };
                    workspace.seeds[row * seed_size + entry] = values[source];
                }
            }
            let operator = &mut workspace.operators[index];
            let input = &workspace.values[node.inputs.clone()];
            let result = if reverse {
                operator.vjp_batch(input, count, &workspace.seeds, &mut workspace.products)
            } else {
                operator.jvp_batch(input, count, &workspace.seeds, &mut workspace.products)
            };
            result
                .and_then(|()| check_finite("operator product", &workspace.products))
                .map_err(|source| Error::Operator {
                    node: node.index,
                    source: Box::new(source),
                })?;
            for row in 0..count {
                let values = &mut workspace.derivatives[row * plan.slots..(row + 1) * plan.slots];
                for entry in 0..product_size {
                    let value = workspace.products[row * product_size + entry];
                    if reverse {
                        let destination = node.sources[entry];
                        values[destination] += value;
                        if !values[destination].is_finite() {
                            return Err(Error::NonFinite("accumulated cotangent"));
                        }
                    } else {
                        values[node.outputs.start + entry] = value;
                    }
                }
            }
        }
        for row in 0..count {
            let values = &workspace.derivatives[row * plan.slots..(row + 1) * plan.slots];
            if reverse {
                output[row * plan.inputs..(row + 1) * plan.inputs]
                    .copy_from_slice(&values[..plan.inputs]);
            } else {
                for (entry, &source) in plan.outputs.iter().enumerate() {
                    output[row * plan.outputs.len() + entry] = values[source];
                }
            }
        }
        Ok(())
    }
}
