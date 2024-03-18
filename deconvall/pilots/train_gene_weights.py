import os
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path_fdn = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(path_fdn))
import deconvall

atlas_paths = path_fdn.parent / 'compressed_atlas' / 'prototype2' / 'cell_atlas_approximations_API' / 'web' / 'static' / 'atlas_data'


if __name__ == '__main__':

    species = 'c_gigas'

    app = deconvall.Approximation.read_h5(atlas_paths / (species + '.h5'))
    adata = app.to_anndata()

    #markers = deconvall.get_markers(adata, 'neuron', 10)
    markerd = deconvall.get_all_markers(adata, 10)

    from collections import Counter
    markers_all = sum(list(markerd.values()), [])
    cou = Counter(markers_all)
    if cou.most_common(1)[0][1] > 1:
        print('Markers are found in multiple cell types?')
        
    marker_ser = pd.DataFrame(markerd).stack().reset_index().set_index(0)['level_1']


    def simulate_mix(adata, fractions=None):
        celltypes = adata.obs_names
        if fractions is None:
            fractions = np.random.rand(len(celltypes))
            fractions /= fractions.sum()

        counts = adata.X.T @ fractions

        return {
            'fractions': pd.Series(fractions, index=celltypes),
            'counts': counts,
        }

    def deconvolve_multiple(adata, mixes, nmark_per_ct=5):
        # Prepare markers
        markerd = deconvall.get_all_markers(adata, nmark_per_ct)
        marker_ser = pd.DataFrame(markerd).stack().reset_index().set_index(0)['level_1']

        tmp = pd.Series(np.arange(adata.shape[1]), index=adata.var_names)
        idx = tmp.loc[marker_ser.index].values
        nmark = len(marker_ser)

        Xmark = adata[:, marker_ser.index].X
        nfrac = adata.shape[0]

        ncall = [0]
        def _deconv(weights, ncall=None, store_result=False):
            if ncall is not None:
                ncall[0] += 1
                if (ncall[0] % 10) == 0:
                    print(ncall[0])
            if store_result:
                results = []
            ssr_total = 0
            for mix in mixes:
                mixcountmark = mix['counts'][idx]
                def fun_min(fractions):
                    loss_w_weights = (Xmark.T @ fractions - mixcountmark)
                    # Apply weights
                    loss_w_weights *= weights
                    loss_w_weights /= weights.sum()
                    loss_w_weights **= 2
                    loss_w_weights = loss_w_weights.sum()
                    return loss_w_weights

                from scipy.optimize import minimize, LinearConstraint, Bounds
                bounds = [[0, 1] for i in range(nfrac)]
                c1 = LinearConstraint(np.ones(nfrac), lb=1, ub=1)
                res = minimize(
                    fun_min,
                    x0=np.ones(nfrac) / nfrac,
                    bounds=bounds,
                    constraints=[c1],
                )
                residual = mix['fractions'] - res.x
                ssr = (residual**2).sum()
                ssr_total += ssr
                if store_result:
                    results.append({
                        'fractions': mix['fractions'],
                        'fit': res.x,
                        'ssr': ssr
                    })

            if store_result:
                results = pd.DataFrame(results)
                return results
            else:
                return ssr_total
        
        from scipy.optimize import minimize, LinearConstraint, Bounds
        res = minimize(
            _deconv,
            x0=np.ones(nmark),
            args=(ncall,),
            bounds=[[0, np.inf] for i in range(nmark)],
            options={'maxiter': 1000},
            method='Nelder-Mead',
        )
        weights = res.x
        res_noweights = _deconv(np.ones(nmark), store_result=True)
        res = _deconv(weights, store_result=True)

        return {
            'genes': marker_ser.index,
            'weights': weights,
            'res_noweights': res_noweights,
            'res_final': res,
        }

    print('Test against single cells from the same dataset')
    sys.path.insert(0, str(path_fdn.parent / 'compressed_atlas' / 'prototype2' / 'cell_atlas_approximations_compression' / 'compression'))
    import anndata
    from utils import (
        load_config,
        postprocess_feature_names,
        filter_cells,
        normalise_counts,
        correct_annotations,
    )

    def simulate_mix_single_cell(adata, n=100):
        ncells = adata.shape[0]
        idx = np.random.choice(np.arange(ncells), size=n, replace=False)
        idx.sort()

        adata_sub = adata[adata.obs_names[idx]]
        fractions = 1.0 * adata_sub.obs['cellType'].value_counts()
        fractions /= fractions.sum()
        
        celltypes = adata.obs['cellType'].value_counts().index
        for ct in celltypes:
            if ct not in fractions.index:
                fractions.loc[ct] = 0

        counts = np.asarray(adata_sub.X.mean(axis=0)).ravel()
        return {
            'fractions': fractions,
            'counts': counts,
            'nsubsample': n,
        }

    nreps = 10
    measurement_type = 'gene_expression'
    config_mt = load_config(species)[measurement_type]
    load_params = {}
    if 'load_params' in config_mt:
        load_params.update(config_mt['load_params'])

    # FIXME: add proper logic
    tissues = sorted(config_mt["path"].keys())
    for itissue, tissue in enumerate(tissues):
        print(tissue)
        adata_tissue = anndata.read_h5ad(config_mt["path"][tissue], **load_params)

        print("Postprocess feature names")
        adata_tissue = postprocess_feature_names(adata_tissue, config_mt)

        print("Filter cells")
        adata_tissue = filter_cells(adata_tissue, config_mt)

        print("Normalise")
        adata_tissue = normalise_counts(
            adata_tissue,
            config_mt['normalisation'],
            measurement_type,
        )

        print("Correct cell annotations")
        adata_tissue = correct_annotations(
            adata_tissue,
            config_mt['cell_annotations']['column'],
            species,
            tissue,
            config_mt['cell_annotations']['rename_dict'],
            config_mt['cell_annotations']['require_subannotation'],
            blacklist=config_mt['cell_annotations']['blacklist'],
            subannotation_kwargs=config_mt['cell_annotations']['subannotation_kwargs'],
        )

        print('Prepare mixes')
        mixes = []
        for i in range(nreps):
            mix = simulate_mix_single_cell(adata_tissue)
            mixes.append(mix)

        print('Optimize weights')
        res = deconvolve_multiple(adata, mixes)

        fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        for ax, key in zip(axs, ['res_noweights', 'res_final']):
            results = res[key]
            ax.set_title(key)
            x = np.concatenate(results['fractions'])
            y = np.concatenate(results['fit'])
            c = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(x)))
            ax.scatter(x, y, alpha=0.8, c=c)
            sns.kdeplot(x=x, y=y, palette='Reds', ax=ax)
            ax.set_xlabel('True fraction')
            ax.set_ylabel('Fitted fraction')
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            x0 = np.linspace(0, 1, 100)
            ax.plot(x0, x0, ls='--', color='tomato')
            ax.grid(True)
        fig.tight_layout()

        plt.ion(); plt.show()

        break
