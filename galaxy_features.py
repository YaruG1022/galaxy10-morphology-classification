"""Image-only display-composite descriptors and leakage-safe preprocessing.

These are morphology proxies, not calibrated astronomical measurements.
Use make_model_pipeline for future cross-validation: refit preprocessing in each fold.
"""
import hashlib
import inspect
import json

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import hog
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_CONFIG = {
    "version": "features_v1", "input_shape": [256, 256, 3],
    "morphology_size": 128, "resampling": "non-overlapping 2x2 stored-channel means",
    "background_outer_radius": 56.0, "detection_radius": 48.0,
    "central_search_radius": 16.0, "fallback_radius": 32.0,
    "detection_sigma": 2.5, "detection_smoothing_sigma": 1.0,
    "minimum_component_pixels": 16, "minimum_detection_signal": 1/255,
    "color_inner_radius": 16.0, "color_outer_radius": 48.0,
    "smoothness_sigma": 3.0, "smoothness_core_radius": 2.0,
    "hog_image_size": 64, "hog_orientations": 9,
    "hog_pixels_per_cell": [8, 8], "hog_cells_per_block": [2, 2],
    "hog_block_norm": "L2-Hys", "hog_transform_sqrt": False,
    "hog_pca_components": 64, "preprocessing_seed": 42,
    "umap": {"n_neighbors": 15, "min_dist": 0.1, "n_components": 2,
             "metric": "euclidean", "n_epochs": 200, "random_state": 42, "n_jobs": 1},
}
Y, X = np.indices((128, 128), dtype=float)
GEOMETRIC_RADIUS = np.hypot(X-63.5, Y-63.5)
HOG_DIM = 1764


def scalar_schema():
    rows = []
    def add(name, group, definition, units="dimensionless"):
        rows.append({"name":name,"group":group,"definition":definition,"units":units})
    for region in ("center", "outskirts"):
        for channel in range(3):
            for stat in ("mean", "std"):
                add(f"{region}_channel_{channel}_{stat}", "color", f"{stat} of normalized stored channel {channel} in {region}; not calibrated band flux")
        for a,b in [(0,1),(1,2),(0,2)]:
            for stat in ("mean", "std"):
                add(f"{region}_difference_{a}_{b}_{stat}", "color", f"{stat} of stored pixel-channel {a} minus {b} in {region}; not a magnitude color")
    for channel in range(3):
        add(f"channel_{channel}_center_minus_outskirts", "color", "Difference of center and outskirts channel means")
    for channel in range(3):
        for stat in ("mean", "std"):
            add(f"global_channel_{channel}_{stat}", "color", f"Whole-frame normalized stored-channel {channel} {stat}")
    definitions = {
        "proxy_concentration":"5 log10(r80/r20), using nonnegative segmented display light; undefined if r20 <= 0.5 pixels",
        "proxy_asymmetry":"Sum |segmented light - 180-degree rotated light| / total segmented light; fixed moment center, no background correction",
        "proxy_smoothness":"10 times positive light-minus-Gaussian(sigma=3) residual outside a 2-pixel core, divided by total light; no background correction",
        "proxy_gini":"Finite-sample Gini coefficient over nonnegative segmented pixel values, including zeros",
        "proxy_m20":"log10 of moment in brightest >=20% cumulative-light pixels / total moment about light centroid",
        "proxy_r20":"Interpolated radius enclosing 20% of segmented display light",
        "proxy_r50":"Interpolated radius enclosing 50% of segmented display light",
        "proxy_r80":"Interpolated radius enclosing 80% of segmented display light",
        "proxy_axis_ratio":"sqrt(minor/major eigenvalue of light-weighted central second moments)",
        "proxy_elongation":"Inverse axis ratio; undefined for zero minor eigenvalue",
        "proxy_eccentricity":"sqrt(1-axis_ratio^2)",
        "object_area_fraction":"Number of target-mask pixels / 128^2",
        "centroid_offset_fraction":"Moment-centroid distance from geometric center / 128",
        "central_light_fraction":"Segmented light within 16 pixels of the target centroid / total segmented light",
        "mean_segmented_signal":"Mean background-subtracted nonnegative light over the selected mask",
    }
    for name, definition in definitions.items():
        add(name, "morphology_proxy", definition, "pixels on 128x128 image" if name in ["proxy_r20","proxy_r50","proxy_r80"] else "dimensionless")
    for name, definition in {
        "flag_fallback_mask":"No qualifying central component; use fixed central circular aperture",
        "flag_no_signal":"Zero total segmented background-subtracted display light",
        "flag_degenerate_moments":"Shape moments or minor eigenvalue are degenerate",
        "flag_undefined_concentration":"Concentration undefined at current pixel resolution",
        "flag_undefined_m20":"Brightest-pixel or total second moment is zero",
        "flag_mask_at_boundary":"Selected component reaches the fixed 48-pixel detection boundary",
    }.items():
        add(name, "quality_flag", definition, "0 or 1")
    return rows


SCALAR_SCHEMA = scalar_schema()
SCALAR_NAMES = [r["name"] for r in SCALAR_SCHEMA]
N_SCALARS = len(SCALAR_NAMES)


def gini_coefficient(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values)<2 or not np.isfinite(values).all() or np.any(values<0) or values.sum()<=0:
        return np.nan
    ordered=np.sort(values)
    n=len(ordered)
    return float(np.clip(np.dot(2*np.arange(1,n+1)-n-1,ordered)/((n-1)*ordered.sum()),0,1))


def m20_from_pixels(values, x, y):
    values=np.asarray(values,dtype=float)
    total=values.sum()
    if total<=0:
        return np.nan
    cx,cy=np.sum(values*x)/total,np.sum(values*y)/total
    moments=values*((x-cx)**2+(y-cy)**2)
    denominator=moments.sum()
    order=np.argsort(-values,kind="stable")
    count=int(np.searchsorted(np.cumsum(values[order]),.2*total,side="left"))+1
    numerator=moments[order[:count]].sum()
    return float(np.log10(numerator/denominator)) if denominator>0 and numerator>0 else np.nan


def foreground(gray, config):
    ring=gray[GEOMETRIC_RADIUS>=config["background_outer_radius"]]
    background=float(np.median(ring))
    sigma=float(1.4826*np.median(np.abs(ring-background)))
    signal=np.maximum(gray-background,0)
    smooth=ndi.gaussian_filter(gray-background,config["detection_smoothing_sigma"])
    threshold=max(config["detection_sigma"]*sigma,config["minimum_detection_signal"])
    detected=(smooth>threshold)&(GEOMETRIC_RADIUS<=config["detection_radius"])
    components,_=ndi.label(detected,structure=np.ones((3,3),dtype=int))
    central=np.unique(components[GEOMETRIC_RADIUS<=config["central_search_radius"]])
    candidates=[]
    for label in central[central>0]:
        member=components==label
        if int(member.sum())>=config["minimum_component_pixels"]:
            candidates.append((float(GEOMETRIC_RADIUS[member].min()),-float(signal[member].sum()),int(label)))
    fallback=not candidates
    if fallback:
        mask=GEOMETRIC_RADIUS<=config["fallback_radius"]
    else:
        chosen=min(candidates)[2]
        mask=ndi.binary_fill_holes(components==chosen)
    boundary=bool(np.any(mask&(GEOMETRIC_RADIUS>=config["detection_radius"]-1)))
    object_light=signal*mask
    flux=float(object_light.sum())
    if flux>0:
        cx=float((object_light*X).sum()/flux)
        cy=float((object_light*Y).sum()/flux)
    else:
        cx=cy=63.5
    return mask,object_light,cx,cy,background,sigma,fallback,boundary


def extract_image_features(image, config=None, return_debug=False):
    config=DEFAULT_CONFIG if config is None else config
    image=np.asarray(image)
    if config['morphology_size']!=128 or config['hog_image_size']!=64 or config['input_shape']!=[256,256,3]:
        raise ValueError('Version 1 supports only its documented image sizes')
    if image.shape!=(256,256,3) or image.dtype!=np.uint8:
        raise ValueError("Expected an unchanged 256x256x3 uint8 display-composite image")
    # Fixed box averaging is independent of labels, cohort, or learned statistics.
    color=image.reshape(128,2,128,2,3).mean(axis=(1,3),dtype=np.float64)/255
    gray=color.mean(axis=2)
    mask,light,cx,cy,background,sigma,fallback,boundary=foreground(gray,config)
    radius=np.hypot(X-cx,Y-cy)
    regions={"center":radius<=config["color_inner_radius"],
             "outskirts":(radius>config["color_inner_radius"])&(radius<=config["color_outer_radius"])}
    feature={}
    for region,inside in regions.items():
        pixels=color[inside]
        for channel in range(3):
            feature[f"{region}_channel_{channel}_mean"]=float(pixels[:,channel].mean())
            feature[f"{region}_channel_{channel}_std"]=float(pixels[:,channel].std(ddof=0))
        for a,b in [(0,1),(1,2),(0,2)]:
            diff=pixels[:,a]-pixels[:,b]
            feature[f"{region}_difference_{a}_{b}_mean"]=float(diff.mean())
            feature[f"{region}_difference_{a}_{b}_std"]=float(diff.std(ddof=0))
    for channel in range(3):
        feature[f"channel_{channel}_center_minus_outskirts"]=feature[f"center_channel_{channel}_mean"]-feature[f"outskirts_channel_{channel}_mean"]
        feature[f"global_channel_{channel}_mean"]=float(color[:,:,channel].mean())
        feature[f"global_channel_{channel}_std"]=float(color[:,:,channel].std(ddof=0))
    morphology=[r["name"] for r in SCALAR_SCHEMA if r["group"]=="morphology_proxy"]
    feature.update({name:np.nan for name in morphology})
    feature.update({name:0. for name in SCALAR_NAMES if name.startswith("flag_")})
    flux=float(light.sum())
    feature.update(flag_fallback_mask=float(fallback),flag_no_signal=float(flux<=0),
                   flag_mask_at_boundary=float(boundary),object_area_fraction=float(mask.mean()),
                   centroid_offset_fraction=float(np.hypot(cx-63.5,cy-63.5)/128))
    if flux>0:
        values=light[mask]
        positive=values>0
        r=radius[mask][positive]
        weights=values[positive]
        order=np.argsort(r,kind="stable")
        cumulative=np.cumsum(weights[order])
        r20,r50,r80=np.interp(np.array([.2,.5,.8])*flux,cumulative,r[order])
        feature.update(proxy_r20=float(r20),proxy_r50=float(r50),proxy_r80=float(r80))
        if r20>.5 and r80>=r20:
            feature["proxy_concentration"]=float(5*np.log10(r80/r20))
        feature["proxy_gini"]=gini_coefficient(values)
        feature["proxy_m20"]=m20_from_pixels(values,X[mask],Y[mask])
        rotated=ndi.map_coordinates(light,[2*cy-Y,2*cx-X],order=1,mode="constant",cval=0.,prefilter=False)
        aperture=radius<=config["detection_radius"]
        feature["proxy_asymmetry"]=float(np.abs(light-rotated)[aperture].sum()/flux)
        residual=np.maximum(light-ndi.gaussian_filter(light,config["smoothness_sigma"]),0)
        feature["proxy_smoothness"]=float(10*residual[mask&(radius>config["smoothness_core_radius"])].sum()/flux)
        dx,dy=X[mask]-cx,Y[mask]-cy
        covariance=np.array([[np.dot(values,dx*dx),np.dot(values,dx*dy)],
                             [np.dot(values,dx*dy),np.dot(values,dy*dy)]])/flux
        minor,major=np.linalg.eigvalsh(covariance)
        if major>0 and minor>1e-12:
            ratio=float(np.sqrt(max(0.,minor)/major))
            feature.update(proxy_axis_ratio=ratio,proxy_elongation=1/ratio,
                           proxy_eccentricity=float(np.sqrt(max(0.,1-ratio**2))))
        else:
            feature["flag_degenerate_moments"]=1.
        feature["central_light_fraction"]=float(light[radius<=config["color_inner_radius"]].sum()/flux)
        feature["mean_segmented_signal"]=float(values.mean())
    else:
        feature["flag_degenerate_moments"]=1.
    feature["flag_undefined_concentration"]=float(not np.isfinite(feature["proxy_concentration"]))
    feature["flag_undefined_m20"]=float(not np.isfinite(feature["proxy_m20"]))
    small=gray.reshape(64,2,64,2).mean(axis=(1,3))
    texture=hog(small,orientations=config["hog_orientations"],
                pixels_per_cell=tuple(config["hog_pixels_per_cell"]),
                cells_per_block=tuple(config["hog_cells_per_block"]),
                block_norm=config["hog_block_norm"],transform_sqrt=config["hog_transform_sqrt"],
                feature_vector=True,channel_axis=None)
    scalars=np.array([feature[name] for name in SCALAR_NAMES],dtype=np.float32)
    texture=np.asarray(texture,dtype=np.float32)
    if len(texture)!=HOG_DIM or not np.isfinite(texture).all() or np.isinf(scalars).any():
        raise ValueError("Invalid descriptor dimensions or infinite descriptor")
    if return_debug:
        debug={"color":color,"gray":gray,"mask":mask,"center":(cx,cy),
               "background":background,"background_mad_sigma":sigma,"fallback":fallback}
        return scalars,texture,debug
    return scalars,texture


def extraction_fingerprint(config):
    sources="\n".join(inspect.getsource(fn) for fn in [scalar_schema,gini_coefficient,m20_from_pixels,foreground,extract_image_features])
    return hashlib.sha256((sources+json.dumps(config,sort_keys=True)).encode()).hexdigest()


def positions_for_original_indices(original_indices, requested):
    indices=np.asarray(original_indices,dtype=np.int64)
    wanted=np.asarray(requested,dtype=np.int64)
    if len(np.unique(indices))!=len(indices) or len(np.unique(wanted))!=len(wanted):
        raise ValueError("Repeated original row indices")
    lookup={int(index):position for position,index in enumerate(indices)}
    try:
        return np.array([lookup[int(i)] for i in wanted],dtype=int)
    except KeyError as exc:
        raise ValueError(f"Requested row absent from feature table: {exc}") from exc


def make_preprocessor(n_hog_components=64, random_state=42):
    scalar_steps=Pipeline([("impute",SimpleImputer(strategy="median",keep_empty_features=True)),
                           ("scale",StandardScaler())])
    texture_steps=Pipeline([("scale",StandardScaler()),
                            ("pca",PCA(n_components=n_hog_components,svd_solver="randomized",random_state=random_state)),
                            ("component_scale",StandardScaler())])
    return ColumnTransformer([("scalar",scalar_steps,list(range(N_SCALARS))),
                              ("texture",texture_steps,list(range(N_SCALARS,N_SCALARS+HOG_DIM)))],
                             remainder="drop",sparse_threshold=0.,verbose_feature_names_out=True)


def fit_training_preprocessor(raw_features, original_indices, training_indices, config=None):
    config=DEFAULT_CONFIG if config is None else config
    positions=positions_for_original_indices(original_indices,training_indices)
    if raw_features.ndim!=2 or raw_features.shape[1]!=N_SCALARS+HOG_DIM:
        raise ValueError("Wrong raw feature schema")
    transformer=make_preprocessor(config["hog_pca_components"],config["preprocessing_seed"])
    # No labels are accepted, and no held-out rows enter fit.
    transformer.fit(raw_features[positions])
    return transformer


def make_model_pipeline(estimator, n_hog_components=64, random_state=42):
    """Use RAW rows inside GridSearchCV; never cross-validate the full-train-fitted exports."""
    return Pipeline([("features",make_preprocessor(n_hog_components,random_state)),("model",estimator)])
