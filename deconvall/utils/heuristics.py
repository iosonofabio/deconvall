import re
import numpy as np
import anndata


def guess_normalisation(adata):
    """Guess normalisation of the single cell data"""
    if (adata.n_obs == 0):
        raise ValueError("Single cell data set must have at least one observation/cell.""")

    sum0 = int(adata.X[0].sum())
    if sum0 == 10000:
        return 'cptt'
    if sum0 == 1000000:
        return 'cpm'

    is_integer = (np.floor(adata.X.data[:30]) == np.ceil(adata.X.data[:30])).all()
    if is_integer:
        return 'raw'

    if adata.X.max() > 50:
        raise ValueError("Could not guess normalisation of this data set")

    Xe = np.expm1(X)
    adatae = anndata.AnnData(X=Xe)
    norm = guess_normalisation(adatae)
    if norm == 'cpm':
        return 'cpm+logp1'
    if norm == 'cptt':
        return 'cptt+logp1'
    return 'log'


def guess_measurement_type(adata):
    """Guess measurement type (gene expression, chromatin accessibility)"""
    var0 = adata.var_names[0]
    
    # ATAC-Seq peak patterns:
    # chr1-900-1000
    # Chr10-984-8222
    # 1-8894-2299
    # chr5_3999_90000
    pattern = '^(?:chr|Chr|)[0-9]+[_-][0-9]+[_-][0-9]+$'

    if re.findall(pattern, var0):
        return 'chromatin_accessibility'

    return 'gene_expression'


def guess_celltype_column(adata):
    guesses = [
        'celltype',
        'cell_type',
        'cell-type',
        'CellType',
        'cellType',
        'Cell_Type',
        'cell_annotation',
        'CellAnnotation',
        'cellAnnotation',
        'annotation',
        'cell_ontology_class',
        'free_annotation',
        'broad_type',
        'type',
    ]

    cols = adata.obs.columns

    # Exact matches
    for guess in guesses:
        if guess in cols:
            return guess

    # Exact substring
    for guess in guesses:
        for col in cols:
            if guess in col:
                return col

    raise ValueError("Cannot guess celltype column")


def guess_celltype_order(adata, celltype_column):
    """Guess the order of cell types"""
    # TODO: try to come up with something better than this...
    unique = adata.obs[celltype_column].value_counts().index
    return list(unique)
