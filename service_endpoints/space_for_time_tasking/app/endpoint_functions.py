import io
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import ee
import numpy as np
import pandas as pd
import requests
from google.cloud import storage
from google.oauth2.credentials import Credentials
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve


EMBEDDINGS = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
GEE_BAG_FRACTION = 0.5
MIN_LEAF_POPULATION = 1


def _run_name(value):
    if value:
        name = re.sub(r"[^A-Za-z0-9_-]", "-", value).strip("-_")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = f"tasking-{stamp}-{uuid4().hex[:8]}"
    if not name:
        raise ValueError("run_name contains no usable characters")
    return name[:80]


def _wait(task):
    while task.active():
        time.sleep(5)
    status = task.status()
    if status["state"] != "COMPLETED":
        raise RuntimeError(status.get("error_message", status))


def _wait_for_asset(asset_id):
    for _ in range(30):
        try:
            ee.data.getAsset(asset_id)
            return
        except ee.EEException:
            time.sleep(2)
    raise RuntimeError(f"Asset was created but is not readable: {asset_id}")


def _assert_assets_absent(asset_ids):
    existing = []
    for asset_id in asset_ids:
        try:
            ee.data.getAsset(asset_id)
            existing.append(asset_id)
        except ee.EEException:
            pass
    if existing:
        raise ValueError(f"run_name already has assets: {existing}")


def _download_csv(url):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content))


def _roc_data(fpr, tpr, thresholds):
    return {
        "false_positive_rate": fpr.tolist(),
        "true_positive_rate": tpr.tolist(),
        "threshold": [float(x) if np.isfinite(x) else None for x in thresholds],
    }


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def run_tasking(config, access_token):
    """Run the same reference-model workflow as space_for_time_tasking.ipynb."""
    run_name = _run_name(config["run_name"])
    project = config["cloud_project"]
    credentials = Credentials(token=access_token)
    ee.Initialize(credentials=credentials, project=project)

    lon = config["longitude_column"]
    lat = config["latitude_column"]
    block_x = config["block_x_column"]
    block_y = config["block_y_column"]
    target = config["target_column"]
    threshold = config["target_threshold"]
    reference_year = config["reference_year"]
    block_size = config["block_size_m"]
    test_fraction = config["test_block_fraction"]
    point_caps = config["points_per_block"]
    block_fractions = config["block_fractions"]
    auc_tolerance = config["auc_tolerance"]
    n_trees = config["number_of_trees"]
    n_bands = config["number_of_embedding_bands"]
    scale = config["sampling_scale_m"]
    seed = config["seed"]
    bands = [f"A{i:02d}" for i in range(n_bands)]
    variables_per_split = int(np.sqrt(n_bands))

    point_asset = f"projects/{project}/assets/{run_name}_points"
    sample_asset = f"projects/{project}/assets/{run_name}_reference_samples"
    model_asset = f"projects/{project}/assets/{run_name}_random_forest"
    _assert_assets_absent([point_asset, sample_asset, model_asset])

    observations = _download_csv(config["csv_url"])
    required = [lon, lat, block_x, block_y, target]
    missing = [column for column in required if column not in observations]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    observations = observations.dropna(subset=required).reset_index(drop=True)
    observations["_row"] = np.arange(len(observations))
    observations["y"] = (observations[target] >= threshold).astype(int)
    observations["block_id"] = (
        (observations[block_x] // block_size).astype(int).astype(str)
        + "_"
        + (observations[block_y] // block_size).astype(int).astype(str)
    )

    features = [
        ee.Feature(
            ee.Geometry.Point([float(x), float(y)]),
            {
                "_row": int(row),
                "lon": float(x),
                "lat": float(y),
                "y": int(label),
                "block_id": str(block),
            },
        )
        for row, x, y, label, block in observations[
            ["_row", lon, lat, "y", "block_id"]
        ].itertuples(index=False, name=None)
    ]
    task = ee.batch.Export.table.toAsset(
        collection=ee.FeatureCollection(features),
        description=f"{run_name}_points",
        assetId=point_asset,
    )
    task.start()
    _wait(task)
    _wait_for_asset(point_asset)
    points = ee.FeatureCollection(point_asset)

    def embedding(year):
        return (
            ee.ImageCollection(EMBEDDINGS)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .filterBounds(points.geometry())
            .mosaic()
            .select(bands)
        )

    reference_samples = embedding(reference_year).sampleRegions(
        collection=points,
        properties=["_row", "lon", "lat", "y", "block_id"],
        scale=scale,
        geometries=True,
        tileScale=4,
    )
    task = ee.batch.Export.table.toAsset(
        collection=reference_samples,
        description=f"{run_name}_reference_samples",
        assetId=sample_asset,
    )
    task.start()
    _wait(task)
    _wait_for_asset(sample_asset)
    reference_samples = ee.FeatureCollection(sample_asset)
    reference = ee.data.computeFeatures(
        {
            "expression": reference_samples,
            "fileFormat": "PANDAS_DATAFRAME",
            "pageSize": 5000,
        }
    )
    reference = reference.dropna(subset=bands).reset_index(drop=True)
    reference["_row"] = reference["_row"].astype(int)
    reference["y"] = reference["y"].astype(int)

    rng = np.random.default_rng(seed)
    blocks = np.sort(reference.block_id.unique())
    n_test_blocks = max(1, round(test_fraction * len(blocks)))
    test_blocks = set(rng.choice(blocks, size=n_test_blocks, replace=False))
    train = reference[~reference.block_id.isin(test_blocks)].copy()
    test = reference[reference.block_id.isin(test_blocks)].copy()
    if train.y.nunique() != 2 or test.y.nunique() != 2:
        raise ValueError("Training and held-out samples must each contain both classes")

    train = train.assign(_order=np.random.default_rng(seed).random(len(train)))
    block_order = np.random.default_rng(seed).permutation(train.block_id.unique())

    def cap_per_block(frame, cap):
        ordered = frame.sort_values(["block_id", "_order"])
        return ordered if cap == "all" else ordered.groupby("block_id").head(int(cap))

    def fit_score(frame):
        model = RandomForestClassifier(
            n_estimators=n_trees,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(frame[bands], frame.y)
        probability = model.predict_proba(test[bands])[:, 1]
        return model, probability, {
            "n_points": int(len(frame)),
            "n_blocks": int(frame.block_id.nunique()),
            "auc": roc_auc_score(test.y, probability),
            "accuracy": accuracy_score(test.y, probability >= 0.5),
        }

    point_results = []
    for cap in point_caps:
        _, _, result = fit_score(cap_per_block(train, cap))
        point_results.append({"points_per_block": cap, **result})

    full_auc = point_results[-1]["auc"]
    selected_cap = next(
        row["points_per_block"]
        for row in point_results
        if row["auc"] >= full_auc - auc_tolerance
    )
    capped_train = cap_per_block(train, selected_cap)

    block_results = []
    for fraction in block_fractions:
        n_blocks = max(1, round(fraction * len(block_order)))
        subset = capped_train[capped_train.block_id.isin(block_order[:n_blocks])]
        _, _, result = fit_score(subset)
        block_results.append({"block_fraction": fraction, **result})

    full_block_auc = block_results[-1]["auc"]
    recommended_block_fraction = next(
        row["block_fraction"]
        for row in block_results
        if row["auc"] >= full_block_auc - auc_tolerance
    )

    sklearn_model, sklearn_probability, sklearn_metrics = fit_score(capped_train)
    sklearn_all_probability = sklearn_model.predict_proba(reference[bands])[:, 1]
    sklearn_fpr, sklearn_tpr, sklearn_thresholds = roc_curve(
        test.y, sklearn_probability
    )
    sklearn_threshold = float(
        sklearn_thresholds[np.argmax(sklearn_tpr - sklearn_fpr)]
    )

    selected_train_ids = capped_train["_row"].astype(int).tolist()
    gee_train = reference_samples.filter(
        ee.Filter.inList("_row", selected_train_ids)
    )
    def gee_forest():
        return ee.Classifier.smileRandomForest(
            numberOfTrees=n_trees,
            variablesPerSplit=variables_per_split,
            minLeafPopulation=MIN_LEAF_POPULATION,
            bagFraction=GEE_BAG_FRACTION,
            seed=seed,
        )

    gee_holdout_model = (
        gee_forest()
        .setOutputMode("PROBABILITY")
        .train(features=gee_train, classProperty="y", inputProperties=bands)
    )
    gee_all_scored = ee.data.computeFeatures(
        {
            "expression": reference_samples.classify(gee_holdout_model).select(
                ["_row", "y", "classification"]
            ),
            "fileFormat": "PANDAS_DATAFRAME",
            "pageSize": 5000,
        }
    )
    gee_all_scored["_row"] = gee_all_scored["_row"].astype(int)
    gee_all_scored["y"] = gee_all_scored["y"].astype(int)
    gee_all_scored["classification"] = gee_all_scored["classification"].astype(float)
    gee_scored = gee_all_scored[
        gee_all_scored["_row"].isin(test["_row"])
    ].copy()
    gee_y = gee_scored["y"].astype(int).to_numpy()
    gee_probability = gee_scored["classification"].astype(float).to_numpy()
    gee_fpr, gee_tpr, gee_thresholds = roc_curve(gee_y, gee_probability)
    gee_auc = roc_auc_score(gee_y, gee_probability)
    gee_accuracy = accuracy_score(gee_y, gee_probability >= 0.5)
    gee_threshold = float(gee_thresholds[np.argmax(gee_tpr - gee_fpr)])

    production = reference.copy().assign(
        _order=np.random.default_rng(seed).random(len(reference))
    )
    production = cap_per_block(production, selected_cap)
    production_ids = production["_row"].astype(int).tolist()
    gee_production = reference_samples.filter(
        ee.Filter.inList("_row", production_ids)
    )
    final_model = gee_forest().train(
        features=gee_production,
        classProperty="y",
        inputProperties=bands,
    )
    task = ee.batch.Export.classifier.toAsset(
        classifier=final_model,
        description=f"{run_name}_random_forest",
        assetId=model_asset,
    )
    task.start()
    _wait(task)
    _wait_for_asset(model_asset)

    point_data = reference[["_row", "lon", "lat", "block_id", "y"]].copy()
    point_data["split"] = np.where(
        point_data.block_id.isin(test_blocks), "holdout", "train"
    )
    point_data["truth"] = point_data.pop("y").astype(int)
    point_data["sklearn_probability"] = sklearn_all_probability
    point_data["sklearn_prediction"] = (
        point_data["sklearn_probability"] >= 0.5
    ).astype(int)
    point_data = point_data.merge(
        gee_all_scored[["_row", "classification"]].rename(
            columns={"classification": "earth_engine_probability"}
        ),
        on="_row",
        how="inner",
        validate="one_to_one",
    )
    point_data["earth_engine_prediction"] = (
        point_data["earth_engine_probability"] >= 0.5
    ).astype(int)

    def outcomes(prediction):
        return np.select(
            [
                (point_data.truth == 1) & (prediction == 1),
                (point_data.truth == 0) & (prediction == 0),
                (point_data.truth == 0) & (prediction == 1),
                (point_data.truth == 1) & (prediction == 0),
            ],
            ["true_positive", "true_negative", "false_positive", "false_negative"],
            default="unknown",
        )

    point_data["sklearn_outcome"] = outcomes(point_data.sklearn_prediction)
    point_data["earth_engine_outcome"] = outcomes(
        point_data.earth_engine_prediction
    )
    point_data["models_disagree"] = (
        point_data.sklearn_prediction != point_data.earth_engine_prediction
    )
    point_data["disagreement"] = np.where(
        point_data.sklearn_prediction > point_data.earth_engine_prediction,
        "sklearn_positive",
        np.where(point_data.models_disagree, "earth_engine_positive", "agree"),
    )
    sk_tn, sk_fp, sk_fn, sk_tp = confusion_matrix(
        test.y, sklearn_probability >= 0.5
    ).ravel()
    gee_tn, gee_fp, gee_fn, gee_tp = confusion_matrix(
        gee_y, gee_probability >= 0.5
    ).ravel()

    report = {
        "schema_version": "1.1",
        "run": {
            "name": run_name,
            "reference_year": reference_year,
            "seed": seed,
            "embedding_collection": EMBEDDINGS,
            "number_of_embedding_bands": n_bands,
            "sampling_scale_m": scale,
            "target_column": target,
            "target_threshold": threshold,
            "block_size_m": block_size,
            "test_block_fraction": test_fraction,
            "points_per_block": point_caps,
            "block_fractions": block_fractions,
            "auc_tolerance": auc_tolerance,
            "csv_url": config["csv_url"],
            "columns": {
                "longitude": lon,
                "latitude": lat,
                "block_x": block_x,
                "block_y": block_y,
                "target": target,
            },
        },
        "assets": {
            "points": point_asset,
            "reference_samples": sample_asset,
            "classifier": model_asset,
        },
        "input": {
            "n_observations": len(reference),
            "n_positive": int(reference.y.sum()),
            "positive_fraction": float(reference.y.mean()),
            "n_blocks": int(reference.block_id.nunique()),
        },
        "split": {
            "n_training_points": len(train),
            "n_test_points": len(test),
            "n_training_blocks": int(train.block_id.nunique()),
            "n_test_blocks": int(test.block_id.nunique()),
            "points": point_data.to_dict("records"),
        },
        "experiments": {
            "points_per_block": point_results,
            "block_fraction": block_results,
        },
        "selection": {
            "auc_tolerance": auc_tolerance,
            "points_per_block": selected_cap,
            "recommended_minimum_block_fraction": recommended_block_fraction,
            "production_uses_all_blocks": True,
            "production_points": int(len(production)),
            "production_blocks": int(production.block_id.nunique()),
        },
        "sklearn": {
            "parameters": {
                "number_of_trees": n_trees,
                "max_features": "sqrt",
                "bootstrap": True,
                "seed": seed,
            },
            "auc": sklearn_metrics["auc"],
            "accuracy": sklearn_metrics["accuracy"],
            "threshold": sklearn_threshold,
            "confusion_matrix": {
                "tn": int(sk_tn),
                "fp": int(sk_fp),
                "fn": int(sk_fn),
                "tp": int(sk_tp),
            },
            "roc": _roc_data(sklearn_fpr, sklearn_tpr, sklearn_thresholds),
        },
        "earth_engine": {
            "parameters": {
                "number_of_trees": n_trees,
                "variables_per_split": variables_per_split,
                "minimum_leaf_population": MIN_LEAF_POPULATION,
                "bag_fraction": GEE_BAG_FRACTION,
                "seed": seed,
            },
            "auc": float(gee_auc),
            "accuracy": float(gee_accuracy),
            "threshold": gee_threshold,
            "confusion_matrix": {
                "tn": int(gee_tn),
                "fp": int(gee_fp),
                "fn": int(gee_fn),
                "tp": int(gee_tp),
            },
            "roc": _roc_data(gee_fpr, gee_tpr, gee_thresholds),
        },
    }
    report_json = json.dumps(report, default=_json_default, separators=(",", ":"))

    bucket_name = os.environ["RESULTS_BUCKET"]
    object_name = f"{run_name}/results.json"
    storage.Client().bucket(bucket_name).blob(object_name).upload_from_string(
        report_json, content_type="application/json"
    )
    viewer_name = f"{run_name}/viewer.html"
    storage.Client().bucket(bucket_name).blob(viewer_name).upload_from_string(
        Path("/app/viewer/index.html").read_text(), content_type="text/html"
    )
    results_uri = f"gs://{bucket_name}/{object_name}"
    results_url = f"https://storage.googleapis.com/{bucket_name}/{object_name}"
    viewer_url = f"https://storage.googleapis.com/{bucket_name}/{viewer_name}"

    return {
        "status": "success",
        "run_name": run_name,
        "results_uri": results_uri,
        "results_url": results_url,
        "viewer_url": viewer_url,
        "assets": report["assets"],
        "results": report,
    }
