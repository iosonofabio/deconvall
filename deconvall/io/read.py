"""Read from files."""
import h5py
import hdf5plugin
import pandas as pd
import anndata

from deconvall.utils.types import _infer_dtype


def read_h5_to_anndata(
    h5_data,
    neighborhood,
    measurement_type,
    tissue=None,
):
    """Get an AnnData object in which each observation is an average."""
    me = h5_data[measurement_type]
    var_names = me["features"].asstr()[:]

    if "quantisation" in me:
        quantisation = me["quantisation"][:]

    tissues = me['tissues'].asstr()[:]
    if (tissue is not None) and (tissue not in tissues):
        raise IndexError(f"Tissue not in list of tissues for this atlas: {tissues}")
    if (tissue is None) and (len(tissues) != 1):
        raise IndexError(f"Multiple tissues found for this atlas, specify one: {tissues}")

    if tissue is None:
        tissue = tissues[0]

    gby = me['by_tissue'][tissue]['celltype']


    if neighborhood:
        neigroup = gby["neighborhood"]
        Xave = neigroup["average"][:]
        if "quantisation" in me:
            Xave = quantisation[Xave]

        groupby_order = gby["index"].asstr()[:]
        obs_names = neigroup["obs_names"].asstr()[:]
        ncells = neigroup["cell_count"][:]
        coords_centroid = neigroup["coords_centroid"][:]
        convex_hulls = []
        for ih in range(len(coords_centroid)):
            convex_hulls.append(neigroup["convex_hull"][str(ih)][:])

        if measurement_type == "gene_expression":
            Xfrac = neigroup["fraction"][:]
            adata = anndata.AnnData(
                X=Xave,
                layers={
                    "average": Xave,
                    "fraction": Xfrac,
                },
            )
        else:
            adata = anndata.AnnData(X=Xave)

        adata.obsm["X_ncells"] = ncells
        adata.obsm["X_umap"] = coords_centroid
        adata.uns["convex_hulls"] = convex_hulls

        adata.obs_names = pd.Index(obs_names, name="neighborhoods")
        adata.var_names = pd.Index(var_names, name="features")

    else:
        Xave = gby["average"][:]
        if "quantisation" in me:
            Xave = quantisation[Xave]

        if measurement_type == "gene_expression":
            Xfrac = gby["fraction"][:]
        obs_names = gby["index"].asstr()[:]
        # Add obs metadata
        obs = pd.DataFrame([], index=obs_names)
        obs["cell_count"] = gby['cell_count'][:]

        if measurement_type == "gene_expression":
            adata = anndata.AnnData(
                X=Xave,
                obs=obs,
                layers={
                    "average": Xave,
                    "fraction": Xfrac,
                },
            )
        else:
            adata = anndata.AnnData(
                X=Xave,
                obs=obs,
            )

        adata.var_names = pd.Index(var_names, name="features")
        adata.obs_names = pd.Index(obs_names, name="celltype")

    adata.uns["approximation_groupby"] = {
    }
    if neighborhood:
        adata.uns["approximation_groupby"]["order"] = groupby_order
        adata.uns["approximation_groupby"]["cell_count"] = me['cell_count'][:]

    return adata

