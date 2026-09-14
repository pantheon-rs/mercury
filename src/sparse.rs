//! Structural CSC storage and deterministic column coloring, shared by a plan.

use faer::sparse::SymbolicSparseColMat;

pub struct Sparsity {
    pub(crate) symbolic: SymbolicSparseColMat<usize>,
    pub(crate) colors: Vec<Vec<usize>>,
}

impl Sparsity {
    pub(crate) fn new(inputs: usize, dependencies: &[Vec<usize>]) -> Self {
        let mut rows = vec![Vec::new(); inputs];
        for (row, columns) in dependencies.iter().enumerate() {
            for &column in columns {
                rows[column].push(row);
            }
        }
        // A full row makes every column conflict. Avoid walking the same
        // dense conflicts once per row and column (especially for solve inputs).
        let colors = if dependencies.iter().any(|row| row.len() == inputs) {
            (0..inputs).map(|column| vec![column]).collect()
        } else {
            color_columns(inputs, dependencies, &rows)
        };
        let mut pointers = Vec::with_capacity(inputs + 1);
        let mut indices = Vec::new();
        pointers.push(0);
        for entries in rows {
            indices.extend(entries);
            pointers.push(indices.len());
        }
        Self {
            symbolic: SymbolicSparseColMat::new_checked(
                dependencies.len(),
                inputs,
                pointers,
                None,
                indices,
            ),
            colors,
        }
    }
}

fn color_columns(
    inputs: usize,
    dependencies: &[Vec<usize>],
    rows: &[Vec<usize>],
) -> Vec<Vec<usize>> {
    let mut column_colors = vec![None; inputs];
    let mut colors: Vec<Vec<usize>> = Vec::new();
    for (column, entries) in rows.iter().enumerate() {
        if entries.is_empty() {
            continue;
        }
        let mut forbidden = vec![false; colors.len()];
        for &row in entries {
            for &neighbor in &dependencies[row] {
                if let Some(color) = column_colors[neighbor] {
                    forbidden[color] = true;
                }
            }
        }
        let color = forbidden
            .iter()
            .position(|&used| !used)
            .unwrap_or(colors.len());
        if color == colors.len() {
            colors.push(Vec::new());
        }
        colors[color].push(column);
        column_colors[column] = Some(color);
    }
    colors
}
