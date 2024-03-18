import h5py
import pandas as pd
import anndata

from .utils.coarse_grain import coarse_grain_anndata
from .utils.types import _infer_dtype
from .io import read_h5_to_anndata


class Approximation:
    """Access single cell approximations."""

    def __init__(
        self,
    ):
        """
        A cell atlas approximation.

        To instantiate from an h5 file, use Approximation.read_h5(<filename>).
        """
        self._adata_dict = {}

    @classmethod
    def read_h5(
        cls,
        filename,
    ):
        """Lazy reader of approximation from file."""
        self = cls()
        self.approximation_dict = None
        self.filename = filename
        return self

    def _infer_measurement_type(self, h5_data):
        return "gene_expression"

    def _to_anndata(
        self,
        neighborhood=False,
        measurement_type=None,
    ):
        if self.approximation_dict:
            adata = self._appdict_to_anndata(
                neighborhood=neighborhood,
                measurement_type=measurement_type,
            )
        else:
            adata = self._h5file_to_anndata(
                neighborhood=neighborhood,
                measurement_type=measurement_type,
            )
        self._adata_dict[(measurement_type, neighborhood)] = adata

    def _h5file_to_anndata(
        self,
        neighborhood,
        measurement_type,
    ):
        with h5py.File(self.filename) as h5_data:
            if measurement_type is None:
                measurement_type = self._infer_measurement_type(h5_data)
            adata = read_h5_to_anndata(
                h5_data,
                neighborhood,
                measurement_type,
                tissue=None,
            )
        return adata

    def _appdict_to_anndata(
        neighborhood,
        measurement_type,
    ):
        compressed_atlas = self.approximation_dict

        if measurement_type is None:
            if len(compressed_atlas.keys()) == 1:
                measurement_type = list(compressed_atlas.keys())[0]
            else:
                raise ValueError(
                    "Multiple measurement types detected, which one would you like to look at?",
                )

        resd = compressed_atlas[measurement_type]
        var_names = resd["features"]

        if neighborhood:
            neid = resd["neighborhood"]
            Xave = neid["avg"].values
            obs_names = neid["avg"].index.values

            if measurement_type == "gene_expression":
                Xfrac = neid["frac"].values
                adata = anndata.AnnData(
                    X=Xave,
                    layers={
                        "average": Xave,
                        "fraction": Xfrac,
                    },
                )
            else:
                adata = anndata.AnnData(X=Xave)

            adata.obs["cell_count"] = neid["ncells"]
            adata.obsm["X_ncells"] = neid["ncells"]
            adata.obsm["X_umap"] = neid["coords_centroid"]
            adata.uns["convex_hulls"] = neid["convex_hull"]
            adata.obs_names = pd.Index(obs_names, name="neighborhoods")
            adata.var_names = pd.Index(var_names, name="features")

        else:
            Xave = resd["avg"].values
            if measurement_type == "gene_expression":
                Xfrac = resd["frac"].values
                adata = anndata.AnnData(
                    X=Xave,
                    layers={
                        "average": Xave,
                        "fraction": Xfrac,
                    },
                )
            else:
                adata = anndata.AnnData(
                    X=Xave,
                )

            groupby_names = resd["avg"].index.names
            groupby = "\t".join(groupby_names)
            obs_names = resd["avg"].index.map(lambda x: "\t".join(str(y) for y in x))
            obs = resd["obs"].copy()
            obs["cell_count"] = resd["ncells"]
            obs.index = obs_names
            adata.obs = obs
            adata.var_names = pd.Index(var_names, name="features")
            adata.obs_names = pd.Index(obs_names, name=groupby)

        adata.uns["approximation_groupby"] = {
            "names": groupby_names,
            "dtypes": groupby_dtypes,
        }
        if neighborhood:
            adata.uns["approximation_groupby"]["order"] = resd["avg"].index.values

        return adata

    def to_anndata(
        self,
        groupby="celltype",
        neighborhood=False,
        measurement_type=None,
    ):
        if measurement_type is None:
            with h5py.File(self.filename) as h5_data:
                measurement_type = self._infer_measurement_type(h5_data)

        if isinstance(groupby, str):
            groupby = [groupby]
        groupby = tuple(groupby)

        if (measurement_type, neighborhood) not in self._adata_dict:
            self._to_anndata(
                neighborhood=neighborhood,
                measurement_type=measurement_type,
            )

        # FIXME: specify that it's a view somehow
        adata = self._adata_dict[(measurement_type, neighborhood)]

        # Coarse grain result if the approximation includes unnecessary metadata
        #adata_cg = coarse_grain_anndata(adata, groupby)
        adata_cg = adata
        return adata_cg
