import numpy as np


def get_markers(
    adata,
    cell_type,
    number,
    measurement_type="gene_expression",
):
    """Get marker features for a specific cell type in an organ."""
    # In theory, one could use various methods to find markers
    if measurement_type == "gene_expression":
        method = "fraction"
    # For ATAC-Seq, average and fraction are the same thing
    else:
        method = "average"


    # Cell types and indices
    cell_types = adata.obs_names
    ncell_types = len(cell_types)
    if cell_type not in cell_types:
        raise CellTypeNotFoundError(
            f"Cell type not found: {cell_type}",
            cell_type=cell_type,
        )

    # Matrix of measurements (rows are cell types)
    mat = adata.layers[method]

    # Index cell types
    idx = list(cell_types).index(cell_type)

    idx_other = [i for i in range(ncell_types) if i != idx]
    vector = mat[idx]
    mat_other = mat[idx_other]

    # Compute difference (vector - other)
    mat_other -= vector
    mat_other *= -1

    # Find closest cell type for each feature
    closest_value = mat_other.min(axis=0)

    # Take top features
    idx_markers = np.argsort(closest_value)[-number:][::-1]

    # Sometimes there are just not enough markers, so make sure the difference
    # is positive
    idx_markers = idx_markers[closest_value[idx_markers] > 0]

    # Get the feature names
    features = adata.var_names[idx_markers]

    return list(features)


def get_all_markers(
    adata,
    number,
    measurement_type="gene_expression",
    ):
    celltypes = adata.obs_names
    markerd = {ct: get_markers(adata, ct, number, measurement_type=measurement_type) for ct in celltypes}
    return markerd
