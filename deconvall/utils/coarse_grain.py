import numpy as np
import anndata


def coarse_grain_anndata(adata, groupby):
    """Coarse grain an approximation further to broader grouping."""
    if 'approximation_groupby' not in adata.uns:
        raise KeyError(
            "missing .uns['approximation_groupby'] information, this does not look like an approximation",
        )

    if isinstance(groupby, str):
        groupby = [groupby]

    groupby = list(groupby)
    if len(groupby) == 0:
        raise ValueError(
            'groupby must be a sequence with at least one element',
        )

    groupby_original = list(adata.uns['approximation_groupby']['names'])
    for column in groupby:
        if column not in groupby_original:
            raise ValueError(
                f"Grouping not found in approximation: {column}",
            )

    # Trivial case
    if groupby_original == groupby:
        return adata.copy()

    indices_groupby = [groupby_original.index(gb) for gb in groupby]

    # Detect neighborhood
    neighborhood = 'X_ncells' in adata.obsm
    if neighborhood:
        # We only have to touch the X_ncells, the rest is grouping agnostic
        multiindex = (pd.Index(adata.uns['approximation_groupby']['order'])
                        .str
                        .split('\t', expand=True))
        multiindex.names = groupby_names
        ncells = pd.DataFrame(
            adata.obsm['X_ncells'],
            index=multiindex,
        )
        # FIXME: this probably needs more work
        ncells_cg = ncells.groupby(level=indices_groupby).sum()
        new_order = np.asarray(
            ['\t'.join(x) for x in ncells_cg.index],
        )

        adata_cg = adata.copy()
        adata_cg.obsm['X_ncells'] = ncells_cs.values
        adata_cg.uns['approximation_groupby'] = {
            'names': groupby,
            'dtypes': [adata.uns['approximation_groupby']['dtypes'][i] for i in indices_groupby],
            'order': new_order,
        }

    else:
        # If not neighborhood, we have to actually change the matrix X and, if present,
        # the layers (e.g. avg and fractions)
        ncells_with_meta = adata.obs[list(groupby_original) + ['cell_count']].copy()
        ncells_with_meta['idx'] = np.arange(len(ncells_with_meta))

        # To coarse grain, we need two things:
        # 1. the number of cells in each group
        # 2. the index of cells in each group
        gby = ncells_with_meta.groupby(groupby)

        X = np.zeros((gby.ngroups, adata.X.shape[1]), adata.X.dtype)
        obs_names = []
        if adata.layers:
            layers = {
                key: np.zeros((gby.ngroups, layer.shape[1]), layer.dtype) for key, layer in adata.layers.items()
            }

        for i, (groupid, group) in enumerate(gby):
            idx = group['idx'].values
            ncell = group['cell_count'].values
            ncell_cg = ncell.sum()
            #print(groupid)
            #print(group)
            #print(adata.X[idx].shape)
            X[i] = (adata.X[idx] * ncell).sum(axis=0) / ncell_cg
            if adata.layers:
                for key in layers:
                    layers[key][i] = (adata.layers[key][idx] * ncell).sum(axis=0) / ncell_cg
            obs_name = '\t'.join(str(gid) for gid in groupid)
            obs_names.append(obs_name)

        if adata.layers:
            adata_cg = anndata.AnnData(
                X=X,
                var=adata.var.copy(),
            )
        else:
            adata_cg = anndata.AnnData(
                X=X,
                layers=layers,
                var=adata.var.copy(),
            )

        # Obs names and metadata
        ncells_cg = gby.sum()['cell_count']
        adata_cg.obs_names = obs_names
        adata_cg.obs['cell_count'] = ncells_cg.values
        for gb in groupby:
            adata_cg.obs[gb] = ncells_cg.index.get_level_values(gb)

        adata_cg.uns['approximation_groupby'] = {
            'names': groupby,
            'dtypes': [adata.uns['approximation_groupby']['dtypes'][i] for i in indices_groupby],
        }

    return adata_cg

