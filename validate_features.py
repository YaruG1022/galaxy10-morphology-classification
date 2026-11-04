"""Independent checks for raw feature rows, train-only fit, finite exports and provenance."""
import argparse
import csv
import json
from pathlib import Path
import warnings

import h5py
import joblib
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

from galaxy_features import N_SCALARS,HOG_DIM,SCALAR_NAMES,extract_image_features
from validate_data_preparation import validate,digest


def validate_artifacts(root, check_images=True):
    frozen=root/'metadata/data_v1'
    metadata=root/'metadata/features_v1'
    local=root/'data/processed/features_v1'
    summary=json.loads((metadata/'summary.json').read_text())
    validate(root/'data/raw/Galaxy10_DECals_NoDuplicated.h5',frozen)
    assert digest(frozen/'sample_manifest.csv')==summary['manifest_sha256']
    for name,expected in summary['source_sha256'].items():
        assert digest(root/name)==expected,f'Source changed: {name}'
    for name,expected in summary['artifact_sha256'].items():
        assert digest(root/name)==expected,f'Artifact changed: {name}'
    config=json.loads((root/'configs/features_v1.json').read_text())
    assert config==json.loads((metadata/'config.json').read_text())
    assert (metadata/'fit_train_indices.json').read_bytes()==(frozen/'train_indices.json').read_bytes()
    with (frozen/'sample_manifest.csv').open() as f:
        expected_ids=np.array([int(r['original_index']) for r in csv.DictReader(f) if r['included']=='1'])
    with np.load(local/'raw_features.npz',allow_pickle=False) as data:
        raw=data['X'].copy();original=data['original_indices'].copy();targets=data['y'].copy()
    assert np.array_equal(original,expected_ids)
    assert raw.shape==(len(original),N_SCALARS+HOG_DIM) and raw.dtype==np.float32
    assert not np.isinf(raw).any() and np.isfinite(raw[:,N_SCALARS:]).all()
    with h5py.File(root/'data/raw/Galaxy10_DECals_NoDuplicated.h5','r') as f:
        labels=f['ans'][:]
        assert np.array_equal(targets,labels[original])
        if check_images:
            chosen=sorted(summary['diagnostic_example_indices'])
            selected=f['images'][chosen]
    lookup={int(i):k for k,i in enumerate(original)}
    if check_images:
        for i,img in zip(chosen,selected):
            scalar,texture=extract_image_features(img,config)
            assert np.allclose(raw[lookup[int(i)]],np.r_[scalar,texture],atol=1e-7,equal_nan=True),f'Wrong image row: {i}'
    with (local/'scalar_features.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==len(original)
    assert [int(r['original_index']) for r in rows]==original.tolist()
    assert [int(r['label']) for r in rows]==targets.tolist()
    table=np.array([[float(r[name]) if r[name]!='' else np.nan for name in SCALAR_NAMES] for r in rows],dtype=np.float32)
    assert np.array_equal(table,raw[:,:N_SCALARS],equal_nan=True),'Scalar CSV differs from matrix'
    assert not {'ans','label','redshift','pxscale','ra','dec','split','original_index'}.intersection(SCALAR_NAMES)
    split={s:np.array(json.loads((frozen/f'{s}_indices.json').read_text()),dtype=int) for s in ['train','validation','test']}
    positions={s:np.array([lookup[int(i)] for i in ids]) for s,ids in split.items()}
    train_raw=raw[positions['train']]
    transformer=joblib.load(local/'preprocessor.joblib')
    scalars=transformer.named_transformers_['scalar'];texture=transformer.named_transformers_['texture']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        medians=np.nanmedian(train_raw[:,:N_SCALARS].astype(float),axis=0)
    medians=np.nan_to_num(medians,nan=0.)
    assert np.allclose(scalars.named_steps['impute'].statistics_,medians)
    imputed=np.where(np.isfinite(train_raw[:,:N_SCALARS]),train_raw[:,:N_SCALARS],medians)
    assert np.allclose(scalars.named_steps['scale'].mean_,imputed.mean(axis=0),rtol=1e-6,atol=1e-7)
    assert np.allclose(texture.named_steps['scale'].mean_,train_raw[:,N_SCALARS:].mean(axis=0,dtype=float),rtol=1e-6,atol=1e-9)
    pca=texture.named_steps['pca']
    assert pca.n_samples_==len(split['train']) and pca.n_components_==config['hog_pca_components']
    all_ids=[]
    with threadpool_limits(limits=2):
        for name,ids in split.items():
            with np.load(local/f'{name}_features.npz',allow_pickle=False) as data:
                X=data['X'];y=data['y'];saved_ids=data['original_indices']
                assert X.shape==(len(ids),N_SCALARS+config['hog_pca_components'])
                assert X.dtype==np.float32 and np.isfinite(X).all()
                assert np.array_equal(ids,saved_ids) and np.array_equal(y,labels[ids])
                assert np.allclose(X,transformer.transform(raw[positions[name]]),rtol=1e-5,atol=1e-5)
            members=set(map(int,ids))
            assert all(r['split']==name for r in rows if int(r['original_index']) in members)
            all_ids+=ids.tolist()
    assert len(all_ids)==len(set(all_ids))==len(original) and set(all_ids)==set(map(int,original))
    with np.load(local/'training_umap.npz',allow_pickle=False) as data:
        assert np.array_equal(data['original_indices'],split['train'])
        assert np.array_equal(data['y'],labels[split['train']])
        assert data['coordinates'].shape==(len(split['train']),2) and np.isfinite(data['coordinates']).all()
    assert not summary['classifier_training_performed'] and not summary['test_performance_evaluated']
    assert not summary['label_used_during_extraction_or_fit'] and not summary['fit_preprocessing_on_held_out_rows']
    schema=json.loads((metadata/'schema.json').read_text())
    assert schema['raw_feature_names']==SCALAR_NAMES+[f'hog_{i:04d}' for i in range(HOG_DIM)]
    assert len(summary['diagnostic_example_indices'])==10
    assert set(summary['diagnostic_example_indices']).issubset(set(split['train']))
    images=sorted((root/'docs/features/figures').glob('*.png'))
    assert len(images)==4
    for p in images:
        with Image.open(p) as img:
            assert img.width>1000 and img.height>600
            img.verify()
    result={"status":"PASS","retained_rows":len(original),"raw_dimensions":raw.shape[1],
            "processed_dimensions":N_SCALARS+config['hog_pca_components'],
            "partitions":summary['partitions'],"fit_rows":len(split['train']),
            "original_id_alignment_verified":True,"scalar_csv_matches_raw_matrix":True,
            "train_only_imputer_and_scaler_statistics_verified":True,
            "exported_transforms_reproduced":True,"finite_processed_matrices":True,
            "umap_training_rows_verified":True,"sampled_images_reextracted":10 if check_images else 0,
            "source_and_artifact_hashes_verified":True,"overlap":0,"figures":len(images)}
    with (metadata/'validation_report.json').open('w',encoding='utf-8',newline='\n') as f:
        json.dump(result,f,indent=2);f.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--skip-image-recheck',action='store_true')
    args=p.parse_args()
    print(json.dumps(validate_artifacts(args.root.resolve(),not args.skip_image_recheck),indent=2))
