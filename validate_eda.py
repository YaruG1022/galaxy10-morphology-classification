"""Verify exploratory outputs against the frozen raw-label and training-row records."""
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from validate_data_preparation import validate, digest


def validate_outputs(root):
    frozen = root/'metadata/data_v1'
    out = root/'docs/eda'
    raw = root/'data/raw/Galaxy10_DECals_NoDuplicated.h5'
    validate(raw, frozen)
    report = json.loads((out/'summary.json').read_text())
    assert report['scope']=='training_images_only'
    assert report['test_images_analyzed']==0 and not report['model_training_performed']
    for name, expected in report['output_sha256'].items():
        assert digest(out/name)==expected, f'Output changed: {name}'
    assert digest(root/'eda.py')==report['source_key']['script_sha256']
    for key, filename in [('manifest_sha256','sample_manifest.csv'),('train_indices_sha256','train_indices.json')]:
        assert digest(frozen/filename)==report['source_key'][key]
    train=np.array(json.loads((frozen/'train_indices.json').read_text()),dtype=int)
    all_splits={s:json.loads((frozen/f'{s}_indices.json').read_text()) for s in ['train','validation','test']}
    with h5py.File(raw,'r') as f:
        labels=f['ans'][:]
        z=f['redshift'][:][train]
        scales=f['pxscale'][:][train]
    assert report['training_count']==len(train)
    gallery=[]
    for c, indices in report['example_indices'].items():
        assert len(indices)==len(set(indices))==4
        assert set(indices).issubset(set(train))
        assert all(int(labels[i])==int(c) for i in indices)
        gallery+=indices
    assert len(set(gallery))==40
    with (out/'tables/class_counts.csv').open() as f: counts=list(csv.DictReader(f))
    assert len(counts)==10
    for c,row in enumerate(counts):
        for split,indices in all_splits.items():
            assert int(row[split])==int(np.sum(labels[indices]==c))
        assert int(row['retained'])==sum(int(row[s]) for s in all_splits)
    with (out/'tables/training_metadata_summary.csv').open() as f: metadata=list(csv.DictReader(f))
    for c,row in enumerate(metadata):
        values=z[labels[train]==c]
        assert int(row['redshift_missing'])==int((~np.isfinite(values)).sum())
        assert int(row['redshift_negative'])==int((np.isfinite(values)&(values<0)).sum())
        assert int(row['redshift_zero'])==int((values==0).sum())
        for scale in np.unique(scales):
            assert int(row[f'pxscale_{scale:g}_count'])==int(np.sum(scales[labels[train]==c]==scale))
    with (root/'data/processed/eda_v1/training_image_statistics.csv').open() as f: per_image=list(csv.DictReader(f))
    assert [int(r['original_index']) for r in per_image]==train.tolist()
    assert all(int(r['label'])==int(labels[i]) for i,r in zip(train,per_image))
    numeric=np.array([[float(r[k]) for k in ['channel_0_mean','channel_1_mean','channel_2_mean','intensity','spatial_std']] for r in per_image])
    assert np.isfinite(numeric).all() and np.all((numeric>=0)&(numeric<=1))
    assert np.allclose(numeric[:,3],numeric[:,:3].mean(axis=1))
    with (out/'tables/training_pixel_summary.csv').open() as f: rows=list(csv.DictReader(f))
    for c,row in enumerate(rows):
        subset=numeric[labels[train]==c]
        assert int(row['count'])==len(subset)
        variables={'intensity':subset[:,3],'spatial_std':subset[:,4],
                   'channel_0_minus_1':subset[:,0]-subset[:,1],
                   'channel_1_minus_2':subset[:,1]-subset[:,2]}
        for name,values in variables.items():
            expected=np.quantile(values,[0,.05,.25,.5,.75,.95,1])
            actual=[float(row[name+'_'+k]) for k in ['min','q05','q25','median','q75','q95','max']]
            assert np.allclose(actual,expected), f'Quantile mismatch: class {c}, {name}'
    pictures=sorted((out/'figures').glob('*.png'))
    assert len(pictures)==7
    for path in pictures:
        with Image.open(path) as img:
            assert img.width>1000 and img.height>600
            img.verify()
    checks={'status':'PASS','training_images':len(train),'held_out_gallery_images':0,
            'gallery_images':len(gallery),'png_figures':len(pictures),
            'class_count_tables_match_raw_labels':True,'metadata_counts_match_training_rows':True,
            'pixel_quantiles_match_local_training_statistics':True,'output_hashes_match':True,
            'frozen_preparation_verified':True}
    with (out/'validation_report.json').open('w',encoding='utf-8',newline='\n') as f:
        json.dump(checks,f,indent=2); f.write('\n')
    return checks


if __name__=='__main__':
    print(json.dumps(validate_outputs(Path(__file__).resolve().parent),indent=2))
