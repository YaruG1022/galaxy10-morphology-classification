"""Build image-only handcrafted features, training-fitted transforms and training UMAP.

Run using the pinned feature environment. All large matrices remain under ignored data/.
No classifier is trained and no held-out performance is evaluated.
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import h5py
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from eda import save_csv, save_json, CLASSES
from galaxy_features import (DEFAULT_CONFIG,HOG_DIM,N_SCALARS,SCALAR_NAMES,SCALAR_SCHEMA,
    extract_image_features,extraction_fingerprint,fit_training_preprocessor,positions_for_original_indices)
from validate_data_preparation import validate, digest


def extract_dataset(root, config, output, metadata, original, labels):
    cache=output/'raw_features.npz'; provenance=metadata/'extraction.json'
    key={"extraction_fingerprint":extraction_fingerprint(config),
         "manifest_sha256":digest(root/'metadata/data_v1/sample_manifest.csv'),
         "raw_md5":"a920094d5e470e3705f4691c0d6dff54",
         "versions":{p:version(p) for p in ['numpy','scipy','scikit-image']}}
    if cache.exists() and provenance.exists():
        record=json.loads(provenance.read_text())
        if record['key']==key and digest(cache)==record['raw_matrix_sha256']:
            with np.load(cache,allow_pickle=False) as data:
                if not np.array_equal(data['original_indices'],original) or not np.array_equal(data['y'],labels[original]):
                    raise ValueError('Cached rows do not match the frozen manifest')
                print('Verified raw-descriptor cache reused.',flush=True)
                return data['X'].copy()
    raw=np.empty((len(original),N_SCALARS+HOG_DIM),dtype=np.float32)
    lookup={int(i):k for k,i in enumerate(original)}
    with h5py.File(root/'data/raw/Galaxy10_DECals_NoDuplicated.h5','r') as f:
        images=f['images']; step=images.chunks[0] if images.chunks else 64
        done=0
        for start in range(0,len(labels),step):
            chosen=original[(original>=start)&(original<start+step)]
            if not len(chosen):continue
            block=images[start:start+step]
            for i in chosen:
                scalar,texture=extract_image_features(block[int(i)-start],config)
                raw[lookup[int(i)]]=np.r_[scalar,texture]
            done+=len(chosen)
            if (start//step)%2==0 or done==len(original):
                print(f'Features extracted: {done}/{len(original)}',flush=True)
    if np.isinf(raw).any() or not np.isfinite(raw[:,N_SCALARS:]).all():
        raise ValueError('Infinite features or invalid HOG values')
    np.savez_compressed(cache,X=raw,original_indices=original,y=labels[original])
    save_json(provenance,{"key":key,"raw_matrix_sha256":digest(cache),"shape":list(raw.shape),
                          "scalar_missing_values_are_not_imputed_here":True})
    return raw


def figures(root, config, raw, original, train_positions, processed_train, transformer, embedding, summary, out):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
    colors=plt.get_cmap('tab10')(np.arange(10))
    y=summary['training_labels']
    pca=transformer.named_transformers_['texture'].named_steps['pca']
    fig,ax=plt.subplots(figsize=(8.5,4.7),layout='constrained')
    ax.plot(np.arange(1,config['hog_pca_components']+1),np.cumsum(pca.explained_variance_ratio_),color='#315f86',linewidth=2)
    ax.set(xlabel='Number of HOG principal components',ylabel='Cumulative explained training variance',
           title=f"HOG reduction | {config['hog_pca_components']} fixed components",ylim=(0,1))
    ax.grid(alpha=.2)
    fig.savefig(out/'hog_pca_variance.png',dpi=170,bbox_inches='tight');plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,7),layout='constrained')
    for c in range(10):
        points=embedding[y==c]
        ax.scatter(points[:,0],points[:,1],s=5,alpha=.45,color=colors[c],label=f'{c}: {CLASSES[c]}',rasterized=True)
    ax.set(xlabel='UMAP 1 (arbitrary units)',ylabel='UMAP 2 (arbitrary units)',
           title='Handcrafted-feature UMAP | training only, unsupervised')
    ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),markerscale=2.5,fontsize=9,frameon=False)
    fig.supxlabel('12,359 training images; labels used for color only. Fixed seed 42, 15 neighbors, min_dist=0.1.\nA projection is not a classifier score; geometric distances do not prove class separation.',fontsize=8)
    fig.savefig(out/'training_umap.png',dpi=170,bbox_inches='tight');plt.close(fig)

    metrics=['proxy_concentration','proxy_asymmetry','proxy_smoothness','proxy_gini','proxy_m20','proxy_axis_ratio']
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for ax,name in zip(axes.flat,metrics):
        values=raw[train_positions,SCALAR_NAMES.index(name)]
        blocks=[values[(y==c)&np.isfinite(values)] for c in range(10)]
        drawing=ax.boxplot(blocks,patch_artist=True,showfliers=False,whis=(5,95),tick_labels=[str(c) for c in range(10)])
        for box,color in zip(drawing['boxes'],colors):box.set_facecolor(color);box.set_alpha(.65)
        ax.set(title=name.replace('proxy_','').replace('_',' '),xlabel='Class ID',ylabel='Display-composite proxy')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Training morphology proxies | before learned imputation or scaling',fontsize=14)
    fig.supxlabel('Boxes: 25-75%; whiskers: 5-95%. Undefined values omitted from plots and recorded in the quality table.\nThese engineering descriptors are not calibrated CAS/G-M20 astronomical measurements.',fontsize=8)
    fig.savefig(out/'training_proxy_distributions.png',dpi=170,bbox_inches='tight');plt.close(fig)

    examples=json.loads((root/'docs/eda/summary.json').read_text())['example_indices']
    chosen=[int(examples[str(c)][0]) for c in range(10)]
    train_ids=set(map(int,original[train_positions]));assert set(chosen).issubset(train_ids)
    with h5py.File(root/'data/raw/Galaxy10_DECals_NoDuplicated.h5','r') as f:
        pictures=f['images'][sorted(chosen)]
    pictures=dict(zip(sorted(chosen),pictures))
    fig,axes=plt.subplots(2,5,figsize=(14,6.5),layout='constrained')
    for c,ax in enumerate(axes.flat):
        i=chosen[c];_,_,debug=extract_image_features(pictures[i],config,return_debug=True)
        ax.imshow(debug['color'],interpolation='nearest')
        if debug['mask'].any() and not debug['mask'].all():
            ax.contour(debug['mask'],levels=[.5],colors=['#45efaa'],linewidths=.8)
        cx,cy=debug['center'];ax.plot(cx,cy,'+',color='white',markersize=8,markeredgewidth=1)
        ax.set_title(f"Class {c} | raw row {i}\n{'fallback aperture' if debug['fallback'] else 'central component'}",fontsize=9)
        ax.set_axis_off()
    fig.suptitle('Feature-mask diagnostics | fixed training examples',fontsize=14)
    fig.supxlabel('Green: extraction mask; white: light centroid. A diagnostic illustration, not ground-truth segmentation.\nGalaxy10 DECaLS / Leung & Bovy; DESI Legacy Imaging Surveys images; CC BY 4.0. Overlays and resizing added.',fontsize=8)
    fig.savefig(out/'training_segmentation_examples.png',dpi=170,bbox_inches='tight');plt.close(fig)
    return chosen


def write_report(root, summary, quality, preprocessing, config):
    report=root/'docs/features/README.md'
    fallback=next(r['training_true_count'] for r in quality if r['feature']=='flag_fallback_mask')
    missing=sum(r['training_missing_count'] for r in quality)
    explained=sum(preprocessing['hog_explained_variance_ratio'])
    lines=['# Handcrafted image features: version 1','',
           'This module converts every retained image into a documented feature row and prepares a shared classical-model input. It fits image preprocessing and an unsupervised training projection, but trains no classifier and reports no prediction score.','',
           '## Data identity and outputs','',
           f"The unchanged raw HDF5 and data-v1 split are reused. All **{summary['retained_rows']:,} retained rows** have **{N_SCALARS} scalar/quality descriptors plus {HOG_DIM} HOG values**. After scalar imputation/scaling and 64-component HOG PCA/scaling, every row has **{summary['processed_dimensions']} numeric inputs**. Metadata and labels are never input features.",'',
           '| Partition | Rows | Processed inputs |','| --- | ---: | ---: |']
    for split,shape in summary['partitions'].items():lines.append(f'| {split} | {shape[0]} | {shape[1]} |')
    lines += ['', 'Original zero-based HDF5 row IDs are retained separately from matrix positions. The large raw/processed matrices, scalar CSV and fitted transformer are local under ignored `data/processed/features_v1/`. Small schemas, configuration, provenance, quality summaries and fit-index records are versioned under `metadata/features_v1/`.','',
              '## Fixed image-only definitions','',
              'Definitions were fixed before feature extraction and before any classifier evaluation. The code accepts only 256 x 256 x 3 uint8 images. Stored channels are averaged in non-overlapping 2 x 2 blocks to form 128 x 128 color images. Their channel mean is the grayscale proxy. No redshift, pixel scale, sky coordinates, split ID, original row number or class label enters extraction.','',
              '- **Target mask:** estimate background by the median at radius >=56 pixels from the geometric center and noise by 1.4826 times its median absolute deviation. Smooth with sigma=1, detect above max(2.5 noise, 1/255) within radius 48, and choose the qualifying connected component nearest the center, with at least 16 pixels and intersection with the central radius-16 region. Fill holes. If none qualifies, use a radius-32 geometric-center aperture and set a fallback flag.','- **Color and stored-channel differences:** measure each channel mean/std and three pairwise difference mean/std in radius <=16 and 16<radius<=48 regions about the selected light centroid. Add center-minus-outskirts channel means and whole-frame channel mean/std. These are display-composite statistics, not physical g-r/r-z magnitude colors.','- **Concentration:** 5 log10(r80/r20), where radii enclose 20% and 80% of nonnegative segmented display light. Values at insufficient pixel resolution remain undefined and flagged.','- **Asymmetry:** normalized absolute difference from a 180-degree rotation about the segmented light centroid. The center is not optimized for minimum asymmetry and no blank-sky correction is applied.','- **Smoothness:** ten times the positive residual after sigma-3 Gaussian smoothing, outside a radius-2 core, divided by total segmented light. It is not the calibrated CAS clumpiness prescription.','- **Gini and M20:** finite-sample Gini over segmented nonnegative values; M20 is log10 of brightest >=20% cumulative-light second moment divided by the total moment about the light centroid. Zero-moment cases are undefined and flagged.','- **Shape and quality:** light-weighted axis ratio, elongation, eccentricity, flux radii, mask area, centroid offset, central light fraction, mean segmented signal and six explicit quality flags.','- **HOG:** 64 x 64 block-averaged grayscale, nine orientations, 8 x 8 pixels per cell, 2 x 2 cells per block, L2-Hys normalization, no square-root transform; 7 x 7 x 2 x 2 x 9 = 1,764 values. HOG is orientation-sensitive; no rotation-invariance claim is made.','',
              'These descriptors adapt concentration/asymmetry/smoothness and Gini/M20 ideas to rendered display composites. They are **morphology proxies**, not calibrated astronomical measurements: the data lack the calibrated pixel flux, uncertainty maps and standard Petrosian segmentation/background treatment needed to claim canonical CAS or G-M20 measurements. Central-target masking can still include neighbors or miss faint outskirts.','',
              '![Training mask diagnostics](figures/training_segmentation_examples.png)','',
              'The mask gallery uses one preselected training example per class from the earlier seeded gallery, not examples selected to make extraction look successful. The overlay illustrates the implemented rule and can expose target-selection limitations; it is not ground-truth segmentation.','',
              '## Undefined values and quality','',
              f'The training rows contain {missing} undefined scalar entries across all descriptors. The fixed fallback aperture is used for {fallback} training images. Undefined quantities remain NaN in the raw table; no image is dropped and the frozen partition is unchanged. Flags are exported alongside numeric descriptors. See the per-feature training counts in `metadata/features_v1/training_quality.csv`.','',
              'The scalar preprocessing pipeline learns medians from training rows only, preserves an all-missing training column with a zero fill, then standardizes. This policy makes downstream matrices finite while retaining quality flags; it does not make the underlying undefined measurement valid.','',
              '![Training proxy distributions](figures/training_proxy_distributions.png)','',
              'The distributions are descriptive training summaries. Overlapping boxes or differences in medians are not classifier results and do not validate the astronomical interpretation of a proxy. Tails are omitted only from the plot; undefined-value counts are retained in the quality record.','',
              '## Training-only preprocessing and cross-validation','',
              'The scalar block uses median imputation then StandardScaler. The HOG block uses StandardScaler, randomized PCA with 64 fixed components and seed 42, then component-wise StandardScaler. Only the 12,359 frozen training IDs enter fit. Validation and test rows are transformed with that fitted instance; no held-out distribution or label is used to choose parameters.','',
              f'The fixed 64 HOG components explain **{explained:.2%}** of standardized training HOG variance. The component count was fixed before this observation, not selected to reach a variance threshold.','',
              '![HOG PCA variance](figures/hog_pca_variance.png)','',
              '**For future five-fold tuning, use raw descriptors with `make_model_pipeline(estimator)` from `galaxy_features.py`.** Put this entire pipeline inside GridSearchCV so imputation, scaling and PCA are refitted inside each training fold. Do not cross-validate the already transformed full-training exports: those are suitable for the eventual full-training fit and held-out inference, not leakage-free inner-fold evaluation.','',
              '```python','from galaxy_features import make_model_pipeline','# Pass RAW training descriptor rows to GridSearchCV(pipeline, ...).','pipeline = make_model_pipeline(estimator)  # estimator is supplied by the next module','```','',
              '## Training representation projection','',
              '![Training handcrafted-feature UMAP](figures/training_umap.png)','',
              'UMAP is fitted without labels on the processed training features only: 15 neighbors, min_dist=0.1, Euclidean distance, two dimensions, 200 epochs, seed 42 and one worker. Labels only determine plotted colors. The projection is descriptive: visible islands, overlap or distances are not accuracy, significance, a class-separability guarantee or evidence for the CNN-versus-transformer hypothesis. Results depend on representation and projection settings. Pretrained-representation UMAP remains part of the later pretrained-model module.','',
              '## Reproduce and validate','',
              'Use the project feature environment, or install the pinned requirements into another Python 3.12 environment:','',
              '```powershell','python -m pip install -r requirements-features.txt','python build_features.py','python validate_features.py','python -m unittest discover -s tests -v','```','',
              'The local prepared environment can be invoked directly as `.venv\\Scripts\\python.exe`. `notebooks/02_handcrafted_features.ipynb` is the executed walkthrough. Raw descriptors can be cached only when extraction code, configuration, source identity and dependency versions match. Output hashes cover the local feature matrices and the tracked diagnostic artifacts. Existing data-v1 manifests and the earlier exploratory artifacts remain unchanged.','',
              '## Sources, attribution and assistance','',
              '- Conselice (2003), [The Relationship Between Stellar Light Distributions of Galaxies and their Formation Histories](https://arxiv.org/abs/astro-ph/0303065): motivation for concentration, asymmetry and clumpiness.','- Lotz, Primack & Madau (2004), [A New Non-Parametric Approach to Galaxy Morphological Classification](https://arxiv.org/abs/astro-ph/0311352): motivation for Gini and M20. The implementation above is an explicitly documented display-image adaptation.','- [scikit-image HOG](https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.hog), [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage), and [UMAP reproducibility](https://umap-learn.readthedocs.io/en/latest/reproducibility.html).','- W. Henry Leung & Jo Bovy, [Galaxy10 DECaLS](https://zenodo.org/records/10845026), CC BY 4.0; source images from DESI Legacy Imaging Surveys, labels from Galaxy Zoo. Display changes include resizing, mask overlays and derived aggregates.','',
              'Code, derived features, preprocessing, plots, notebook and this record were produced with Codex assistance. No independent team verification is recorded. No research classifier was trained or evaluated in this module.','']
    with report.open('w',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines))


def run(root):
    rawfile=root/'data/raw/Galaxy10_DECals_NoDuplicated.h5'
    frozen=root/'metadata/data_v1'
    output=root/'data/processed/features_v1';output.mkdir(parents=True,exist_ok=True)
    metadata=root/'metadata/features_v1';metadata.mkdir(parents=True,exist_ok=True)
    doc=root/'docs/features';doc.mkdir(parents=True,exist_ok=True)
    plots=doc/'figures';plots.mkdir(exist_ok=True)
    config=json.loads((root/'configs/features_v1.json').read_text())
    validate(rawfile,frozen)
    import csv
    with (frozen/'sample_manifest.csv').open() as f:manifest=list(csv.DictReader(f))
    original=np.array([int(r['original_index']) for r in manifest if r['included']=='1'],dtype=np.int64)
    split={s:np.array(json.loads((frozen/f'{s}_indices.json').read_text()),dtype=np.int64) for s in ['train','validation','test']}
    positions={s:positions_for_original_indices(original,ids) for s,ids in split.items()}
    with h5py.File(rawfile,'r') as f:labels=f['ans'][:]
    raw=extract_dataset(root,config,output,metadata,original,labels)
    schema={"version":"features_v1","scalar_features":SCALAR_SCHEMA,"hog_features":HOG_DIM,
            "raw_feature_names":SCALAR_NAMES+[f'hog_{i:04d}' for i in range(HOG_DIM)],
            "excluded_inputs":["label","redshift","pxscale","ra","dec","split","original_index"],
            "dtype":"float32","scalar_nan_policy":"Keep raw NaNs; impute from fitting training rows only"}
    save_json(metadata/'schema.json',schema)
    save_json(metadata/'config.json',config)
    rows=[]
    membership={int(i):s for s,indices in split.items() for i in indices}
    for i,values in zip(original,raw[:,:N_SCALARS]):
        row={"original_index":int(i),"label":int(labels[i]),"split":membership[int(i)]}
        row.update({name:float(value) if np.isfinite(value) else '' for name,value in zip(SCALAR_NAMES,values)})
        rows.append(row)
    save_csv(output/'scalar_features.csv',rows)
    quality=[]
    for k,name in enumerate(SCALAR_NAMES):
        values=raw[positions['train'],k]
        quality.append({"feature":name,"training_missing_count":int((~np.isfinite(values)).sum()),
                        "training_true_count":int(np.sum(values==1)) if name.startswith('flag_') else 0})
    save_csv(metadata/'training_quality.csv',quality)
    print('Fitting scalar preprocessing and HOG PCA on training rows only...',flush=True)
    with threadpool_limits(limits=2):
        transformer=fit_training_preprocessor(raw,original,split['train'],config)
        matrices={s:np.asarray(transformer.transform(raw[p]),dtype=np.float32) for s,p in positions.items()}
    for s,matrix in matrices.items():
        if not np.isfinite(matrix).all():raise ValueError(f'Nonfinite processed matrix: {s}')
        np.savez_compressed(output/f'{s}_features.npz',X=matrix,y=labels[split[s]],original_indices=split[s])
    joblib.dump(transformer,output/'preprocessor.joblib',compress=3)
    (metadata/'fit_train_indices.json').write_bytes((frozen/'train_indices.json').read_bytes())
    scalar=transformer.named_transformers_['scalar']
    texture=transformer.named_transformers_['texture']
    pca=texture.named_steps['pca']
    preprocessing={"fit_count":int(pca.n_samples_),"fit_original_indices_source_sha256":digest(frozen/'train_indices.json'),
        "scalar_imputation_medians":scalar.named_steps['impute'].statistics_.tolist(),
        "scalar_scaler_mean":scalar.named_steps['scale'].mean_.tolist(),
        "scalar_scaler_scale":scalar.named_steps['scale'].scale_.tolist(),
        "hog_scaler_mean":texture.named_steps['scale'].mean_.tolist(),
        "hog_scaler_scale":texture.named_steps['scale'].scale_.tolist(),
        "hog_explained_variance_ratio":pca.explained_variance_ratio_.tolist(),
        "final_feature_names":transformer.get_feature_names_out(schema['raw_feature_names']).tolist(),
        "fold_policy":"For CV pass raw training rows to make_model_pipeline; all learned steps refit in each fold"}
    save_json(metadata/'preprocessing.json',preprocessing)
    print('Fitting unsupervised UMAP on training rows only...',flush=True)
    import umap
    embedding=umap.UMAP(**config['umap']).fit_transform(matrices['train'])
    if embedding.shape!=(len(split['train']),2) or not np.isfinite(embedding).all():
        raise ValueError('Invalid training UMAP')
    np.savez_compressed(output/'training_umap.npz',coordinates=embedding,original_indices=split['train'],y=labels[split['train']])
    summary={"created_at_utc":datetime.now(timezone.utc).isoformat(),"version":"features_v1",
        "raw_md5":"a920094d5e470e3705f4691c0d6dff54","manifest_sha256":digest(frozen/'sample_manifest.csv'),
        "retained_rows":len(original),"scalar_dimensions":N_SCALARS,"hog_dimensions":HOG_DIM,
        "raw_dimensions":N_SCALARS+HOG_DIM,"processed_dimensions":N_SCALARS+config['hog_pca_components'],
        "partitions":{s:list(m.shape) for s,m in matrices.items()},
        "training_fit_rows":len(split['train']),"umap_fit_rows":len(split['train']),
        "classifier_training_performed":False,"test_performance_evaluated":False,
        "label_used_during_extraction_or_fit":False,"fit_preprocessing_on_held_out_rows":False,
        "source_sha256":{name:digest(root/name) for name in ['galaxy_features.py','build_features.py','configs/features_v1.json']},
        "versions":{p:version(p) for p in ['numpy','scipy','scikit-image','scikit-learn','joblib','umap-learn','numba','matplotlib']},
        "training_labels":labels[split['train']],
        "limitations":["Morphology proxies use rendered composites, not calibrated astronomical flux.",
                       "Fixed central-component detection can miss faint targets or include neighbors.",
                       "HOG is orientation-sensitive; 64 PCA components are a fixed engineering choice.",
                       "UMAP is descriptive, unsupervised, training-only and not evidence of prediction performance.",
                       "Future CV must refit the full preprocessing pipeline inside each training fold."]}
    chosen=figures(root,config,raw,original,positions['train'],matrices['train'],transformer,embedding,summary,plots)
    summary.pop('training_labels')
    summary['diagnostic_example_indices']=chosen
    summary['training_scalar_nan_entries']=sum(q['training_missing_count'] for q in quality)
    summary['training_fallback_masks']=next(q['training_true_count'] for q in quality if q['feature']=='flag_fallback_mask')
    summary['hog_pca_explained_variance']=float(pca.explained_variance_ratio_.sum())
    write_report(root,summary,quality,preprocessing,config)
    files=[*output.glob('*'),*metadata.glob('*'),*doc.rglob('*')]
    summary['artifact_sha256']={str(p.relative_to(root)).replace('\\','/'):digest(p)
                               for p in sorted(files) if p.is_file() and p.name not in ['summary.json','validation_report.json']}
    save_json(metadata/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['retained_rows','scalar_dimensions','hog_dimensions','processed_dimensions','partitions','training_scalar_nan_entries','training_fallback_masks','hog_pca_explained_variance']},indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    run(parser.parse_args().root.resolve())
