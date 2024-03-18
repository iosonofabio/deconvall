import os
import sys
import pathlib
import numpy as np
import pandas as pd

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

    def deconvolve(adata, mix):
        markerd = deconvall.get_all_markers(adata, 10)
        marker_ser = pd.DataFrame(markerd).stack().reset_index().set_index(0)['level_1']

        tmp = pd.Series(np.arange(adata.shape[1]), index=adata.var_names)
        idx = tmp.loc[marker_ser.index].values

        Xmark = adata[:, marker_ser.index].X
        mixcountmark = mix['counts'][idx]
        nfrac = adata.shape[0]

        def fun_min(fractions):
            loss = ((Xmark.T @ fractions - mixcountmark)**2).sum()
            return loss

        from scipy.optimize import minimize, LinearConstraint, Bounds
        bounds = [[0, 1] for i in range(nfrac)]
        c1 = LinearConstraint(np.ones(nfrac), lb=1, ub=1)
        res = minimize(
            fun_min,
            x0=np.ones(nfrac) / nfrac,
            bounds=bounds,
            constraints=[c1],
        )
        return res


    print('Test against artificial compositions of averages (this must work)')
    results = []
    for i in range(100):
        if (i % 10) == 0:
            print(i)
        mix = simulate_mix(adata)
        out = deconvolve(adata, mix)
        residual = mix['fractions'] - out.x
        ssr = (residual**2).sum()
        results.append({
            'fractions': mix['fractions'],
            'fit': out.x,
            'ssr': ssr
        })
        #print(f'Sum of squared residuals: {ssr}')
    print('Done')
    results = pd.DataFrame(results)
    print('It works really well - of course')


    print('Test against single cells from the same dataset')
