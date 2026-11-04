"""Numerical, geometry, schema and training-only regression tests."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.base import clone
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from galaxy_features import (DEFAULT_CONFIG,HOG_DIM,N_SCALARS,SCALAR_NAMES,
    extract_image_features,gini_coefficient,m20_from_pixels,make_preprocessor,
    fit_training_preprocessor,positions_for_original_indices)


def elliptical_image():
    y,x=np.indices((256,256),dtype=float)
    light=np.exp(-.5*(((x-127.5)/24)**2+((y-127.5)/12)**2))
    channel=np.round(25+180*light).astype(np.uint8)
    return np.repeat(channel[:,:,None],3,axis=2)


class FeatureTests(unittest.TestCase):
    def test_gini_known_values(self):
        self.assertAlmostEqual(gini_coefficient([1,1,1]),0)
        self.assertAlmostEqual(gini_coefficient([0,0,4]),1)
        self.assertTrue(np.isnan(gini_coefficient([0,0,0])))
        self.assertTrue(np.isnan(gini_coefficient([-1,2])))

    def test_m20_known_moments(self):
        self.assertAlmostEqual(m20_from_pixels(np.array([1.,1.]),np.array([-1.,1.]),np.zeros(2)),np.log10(.5))
        self.assertTrue(np.isnan(m20_from_pixels(np.array([1.]),np.zeros(1),np.zeros(1))))

    def test_elliptical_morphology_and_hog_shape(self):
        image=elliptical_image(); before=image.copy()
        scalar,texture=extract_image_features(image)
        f=dict(zip(SCALAR_NAMES,scalar))
        self.assertEqual(scalar.shape,(N_SCALARS,))
        self.assertEqual(texture.shape,(1764,))
        self.assertTrue(np.isfinite(texture).all())
        self.assertTrue(np.array_equal(image,before))
        self.assertAlmostEqual(float(f['proxy_axis_ratio']),.5,delta=.08)
        self.assertLess(f['proxy_asymmetry'],.03)
        self.assertGreater(f['proxy_concentration'],0)
        self.assertLess(f['proxy_m20'],0)
        self.assertEqual(f['flag_no_signal'],0)

    def test_scalar_geometry_is_invariant_to_right_angle_rotation(self):
        image=elliptical_image()
        first=dict(zip(SCALAR_NAMES,extract_image_features(image)[0]))
        second=dict(zip(SCALAR_NAMES,extract_image_features(np.rot90(image))[0]))
        for name in ['proxy_axis_ratio','proxy_concentration','proxy_asymmetry','proxy_smoothness','proxy_gini','proxy_m20']:
            self.assertAlmostEqual(float(first[name]),float(second[name]),delta=.03,msg=name)

    def test_zero_signal_is_flagged_not_silently_measured(self):
        scalar,texture=extract_image_features(np.full((256,256,3),32,dtype=np.uint8))
        f=dict(zip(SCALAR_NAMES,scalar))
        self.assertEqual(f['flag_no_signal'],1)
        self.assertEqual(f['flag_fallback_mask'],1)
        self.assertTrue(np.isnan(f['proxy_gini']))
        self.assertTrue(np.isnan(f['proxy_concentration']))
        self.assertTrue(np.all(texture==0))
        self.assertFalse(np.isinf(scalar).any())

    def test_original_rows_are_not_filtered_positions(self):
        self.assertEqual(positions_for_original_indices([10,15,30,40],[30,10]).tolist(),[2,0])
        with self.assertRaises(ValueError): positions_for_original_indices([10,15],[11])
        with self.assertRaises(ValueError): positions_for_original_indices([10,10],[10])

    def test_fit_uses_only_training_rows_with_fold_safe_clones(self):
        rng=np.random.default_rng(42)
        raw=rng.normal(size=(96,N_SCALARS+HOG_DIM)).astype(np.float32)
        raw[:80,0]=np.nan
        ids=100+np.arange(96)*2; train=ids[:80]
        config=deepcopy(DEFAULT_CONFIG); config['hog_pca_components']=4
        with threadpool_limits(limits=2):
            first=fit_training_preprocessor(raw,ids,train,config)
            changed=raw.copy(); changed[80:]=1e6
            second=fit_training_preprocessor(changed,ids,train,config)
            self.assertTrue(np.allclose(first.transform(raw[:80]),second.transform(raw[:80])))
        scaler=first.named_transformers_['scalar'].named_steps['scale']
        expected=np.nanmean(raw[:80,1:N_SCALARS].astype(float),axis=0)
        self.assertTrue(np.allclose(scaler.mean_[1:],expected))
        self.assertEqual(first.named_transformers_['texture'].named_steps['pca'].n_samples_,80)
        self.assertTrue(np.isfinite(first.transform(raw)).all())
        self.assertEqual(first.transform(raw).shape,(96,N_SCALARS+4))
        self.assertFalse(hasattr(clone(first),'transformers_'))

    def test_feature_schema_contains_no_labels_or_catalog_metadata(self):
        forbidden={'ans','label','redshift','pxscale','ra','dec','split','original_index'}
        self.assertFalse(forbidden.intersection(SCALAR_NAMES))
        self.assertEqual(len(SCALAR_NAMES),len(set(SCALAR_NAMES)))
        with self.assertRaises(ValueError): extract_image_features(np.zeros((128,128,3),dtype=np.uint8))


if __name__=='__main__':
    unittest.main()
