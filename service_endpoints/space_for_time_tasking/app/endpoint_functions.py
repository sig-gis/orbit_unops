import io
import base64
import json
import os
import re
import time
from html import escape
from datetime import datetime, timezone
from uuid import uuid4

import ee
import numpy as np
import pandas as pd
import requests
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from google.cloud import storage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

from app.gcs import public_https_url


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


def _fig_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _plot_split_map(report):
    points = pd.DataFrame(report.get("split", {}).get("points", []))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not points.empty and {"lon", "lat", "split"}.issubset(points.columns):
        train = points[points["split"] == "train"]
        holdout = points[points["split"] == "holdout"]
        ax.scatter(
            train["lon"],
            train["lat"],
            s=8,
            alpha=0.35,
            label=f"Train: {train.get('block_id', pd.Series(dtype=object)).nunique()} blocks",
        )
        ax.scatter(
            holdout["lon"],
            holdout["lat"],
            s=8,
            alpha=0.55,
            label=f"Holdout: {holdout.get('block_id', pd.Series(dtype=object)).nunique()} blocks",
        )
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No split point data available", ha="center", va="center", transform=ax.transAxes)
    ax.set(xlabel="Longitude", ylabel="Latitude", title="Training and held-out observations")
    ax.grid(alpha=0.25)
    return _fig_to_base64(fig)


def _plot_experiments(report):
    point_frame = pd.DataFrame(report.get("experiments", {}).get("points_per_block", []))
    block_frame = pd.DataFrame(report.get("experiments", {}).get("block_fraction", []))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if not point_frame.empty:
        point_labels = point_frame["points_per_block"].astype(str)
        axes[0].plot(point_labels, point_frame["auc"], "o-", label="AUC")
        axes[0].plot(point_labels, point_frame["accuracy"], "o-", label="Accuracy")
    else:
        axes[0].text(0.5, 0.5, "No point-cap data", ha="center", va="center", transform=axes[0].transAxes)

    if not block_frame.empty:
        axes[1].plot(block_frame["block_fraction"] * 100, block_frame["auc"], "o-", label="AUC")
        axes[1].plot(block_frame["block_fraction"] * 100, block_frame["accuracy"], "o-", label="Accuracy")
    else:
        axes[1].text(0.5, 0.5, "No block-fraction data", ha="center", va="center", transform=axes[1].transAxes)

    axes[0].set(title="Points within each training block", xlabel="Points per block")
    axes[1].set(title="Training-block coverage", xlabel="Training blocks used (%)")
    for ax in axes:
        ax.set_ylabel("Held-out score")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    plt.tight_layout()
    return _fig_to_base64(fig)


def _plot_roc(report):
    sklearn = report.get("sklearn", {})
    earth_engine = report.get("earth_engine", {})
    sk_roc = sklearn.get("roc", {})
    gee_roc = earth_engine.get("roc", {})
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="0.7")
    if sk_roc.get("false_positive_rate") and sk_roc.get("true_positive_rate"):
        ax.plot(
            sk_roc["false_positive_rate"],
            sk_roc["true_positive_rate"],
            label=f"sklearn AUC {float(sklearn.get('auc', 0)):.3f}",
        )
    if gee_roc.get("false_positive_rate") and gee_roc.get("true_positive_rate"):
        ax.plot(
            gee_roc["false_positive_rate"],
            gee_roc["true_positive_rate"],
            label=f"Earth Engine AUC {float(earth_engine.get('auc', 0)):.3f}",
        )
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="ROC on held-out blocks",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    return _fig_to_base64(fig)


def _viewer_html(report, results_url):
    """Build a static HTML report with matplotlib plots for browser viewing from GCS."""
    selection = report.get("selection", {})
    input_summary = report.get("input", {})
    sklearn = report.get("sklearn", {})
    earth_engine = report.get("earth_engine", {})
    assets = report.get("assets", {})
    run = report.get("run", {})
    plots = {
        "split_map": _plot_split_map(report),
        "experiments": _plot_experiments(report),
        "roc": _plot_roc(report),
    }

    def metric_row(label, data):
        return (
            "<tr>"
            f"<th>{escape(label)}</th>"
            f"<td>{escape(str(data.get('auc', 'n/a')))}</td>"
            f"<td>{escape(str(data.get('accuracy', 'n/a')))}</td>"
            f"<td>{escape(str(data.get('threshold', 'n/a')))}</td>"
            "</tr>"
        )

    def confusion_matrix_table(label, matrix):
        matrix = matrix or {}
        return f"""
        <div class=\"matrix\">
          <h3>{escape(label)}</h3>
          <table>
            <tr><th></th><th>Predicted 0</th><th>Predicted 1</th></tr>
            <tr><th>Actual 0</th><td>{escape(str(matrix.get('tn', 'n/a')))}</td><td>{escape(str(matrix.get('fp', 'n/a')))}</td></tr>
            <tr><th>Actual 1</th><td>{escape(str(matrix.get('fn', 'n/a')))}</td><td>{escape(str(matrix.get('tp', 'n/a')))}</td></tr>
          </table>
        </div>"""

    asset_items = "".join(
        f"<li><strong>{escape(str(key))}:</strong> <code>{escape(str(value))}</code></li>"
        for key, value in assets.items()
    )
    report_json = escape(json.dumps(report, default=_json_default, indent=2))
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Space-for-time tasking report - {escape(report['run']['name'])}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; line-height: 1.45; color: #1f2937; background: #f8fafc; }}
    h1, h2, h3 {{ color: #111827; }}
    code, pre {{ background: #f3f4f6; padding: 0.15rem 0.3rem; border-radius: 4px; }}
    pre {{ padding: 1rem; overflow-x: auto; }}
    table {{ border-collapse: collapse; margin: 1rem 0; }}
    th, td {{ border: 1px solid #d1d5db; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #f9fafb; }}
    .card {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; margin: 1rem 0; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 1rem; }}
    .plot {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
    .matrix-grid {{ display: flex; flex-wrap: wrap; gap: 2rem; }}
    .muted {{ color: #6b7280; }}
  </style>
</head>
<body>
  <h1>Space-for-time tasking report</h1>
  <p><strong>Run:</strong> {escape(report['run']['name'])}</p>
  <p><strong>Raw JSON:</strong> <a href=\"{escape(results_url)}\">results.json</a></p>

  <div class=\"card\">
    <h2>Recommendation</h2>
    <p><strong>Points per block:</strong> {escape(str(selection.get('points_per_block', 'n/a')))}</p>
    <p><strong>Recommended minimum block fraction:</strong> {escape(str(selection.get('recommended_minimum_block_fraction', 'n/a')))}</p>
    <p><strong>Production points:</strong> {escape(str(selection.get('production_points', 'n/a')))}</p>
    <p><strong>Production blocks:</strong> {escape(str(selection.get('production_blocks', 'n/a')))}</p>
  </div>

  <div class=\"card\">
    <h2>Spatial blocking</h2>
    <p><strong>Block coordinate source:</strong> {escape(str(run.get('block_coordinate_source', 'n/a')))}</p>
    <p><strong>Block CRS:</strong> {escape(str(run.get('block_crs', 'n/a')))}</p>
    <p><strong>Block size:</strong> {escape(str(run.get('block_size_m', 'n/a')))} meters</p>
  </div>

  <div class=\"card\">
    <h2>Input summary</h2>
    <p><strong>Observations:</strong> {escape(str(input_summary.get('n_observations', 'n/a')))}</p>
    <p><strong>Positive observations:</strong> {escape(str(input_summary.get('n_positive', 'n/a')))}</p>
    <p><strong>Positive fraction:</strong> {escape(str(input_summary.get('positive_fraction', 'n/a')))}</p>
    <p><strong>Spatial blocks:</strong> {escape(str(input_summary.get('n_blocks', 'n/a')))}</p>
  </div>

  <div class=\"card\">
    <h2>Training and held-out observations</h2>
    <p class=\"muted\">Matplotlib scatter plot equivalent to the original notebook spatial holdout map.</p>
    <img class=\"plot\" alt=\"Training and held-out observations\" src=\"data:image/png;base64,{plots['split_map']}\">
  </div>

  <div class=\"card\">
    <h2>Tasking experiments</h2>
    <img class=\"plot\" alt=\"Tasking experiment plots\" src=\"data:image/png;base64,{plots['experiments']}\">
  </div>

  <div class=\"card\">
    <h2>ROC on held-out blocks</h2>
    <img class=\"plot\" alt=\"ROC curve\" src=\"data:image/png;base64,{plots['roc']}\">
  </div>

  <div class=\"card\">
    <h2>Model metrics</h2>
    <table>
      <tr><th>Model</th><th>AUC</th><th>Accuracy</th><th>Threshold</th></tr>
      {metric_row('sklearn', sklearn)}
      {metric_row('earth_engine', earth_engine)}
    </table>
  </div>

  <div class=\"card\">
    <h2>Confusion matrices</h2>
    <div class=\"matrix-grid\">
      {confusion_matrix_table('sklearn', sklearn.get('confusion_matrix'))}
      {confusion_matrix_table('earth_engine', earth_engine.get('confusion_matrix'))}
    </div>
  </div>

  <div class=\"card\">
    <h2>Earth Engine assets</h2>
    <ul>{asset_items}</ul>
  </div>

  <details>
    <summary>Full report JSON</summary>
    <pre>{report_json}</pre>
  </details>
</body>
</html>"""

def run_tasking(config):
    """Run the same reference-model workflow as space_for_time_tasking.ipynb."""
    run_name = _run_name(config["run_name"])
    project = config["cloud_project"]
    ee.Initialize(project=project)

    lon = config["longitude_column"]
    lat = config["latitude_column"]
    block_x = config.get("block_x_column")
    block_y = config.get("block_y_column")
    block_crs = config.get("block_crs") or "EPSG:6933"
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

    if bool(block_x) != bool(block_y):
        raise ValueError(
            "Provide both block_x_column and block_y_column, or omit both to derive spatial blocks from geometry."
        )
    use_existing_block_columns = bool(block_x and block_y)

    asset_root = config.get("asset_root") or f"projects/{project}/assets"
    asset_root = asset_root.rstrip("/")
    point_asset = config.get("point_asset_id") or f"{asset_root}/{run_name}_points"
    sample_asset = config.get("sample_asset_id") or f"{asset_root}/{run_name}_reference_samples"
    model_asset = config.get("model_asset_id") or f"{asset_root}/{run_name}_random_forest"

    input_asset_id = config.get("input_asset_id")
    csv_url = config.get("csv_url")
    if bool(input_asset_id) == bool(csv_url):
        raise ValueError("Provide exactly one of input_asset_id or csv_url")

    assets_to_check = [sample_asset, model_asset]
    if not input_asset_id:
        assets_to_check.append(point_asset)
    _assert_assets_absent(assets_to_check)

    def feature_block_xy(feature):
        if use_existing_block_columns:
            return ee.Dictionary(
                {
                    "x": ee.Number(feature.get(block_x)),
                    "y": ee.Number(feature.get(block_y)),
                }
            )

        projected = feature.geometry().centroid(1).transform(block_crs, 1)
        coords = projected.coordinates()
        return ee.Dictionary(
            {
                "x": ee.Number(coords.get(0)),
                "y": ee.Number(coords.get(1)),
            }
        )

    def normalize_feature(feature):
        row_id = ee.Algorithms.If(feature.get("_row"), feature.get("_row"), feature.id())
        xy = feature_block_xy(feature)
        x = ee.Number(xy.get("x"))
        y_coord = ee.Number(xy.get("y"))
        block_id = x.divide(block_size).floor().format().cat("_").cat(
            y_coord.divide(block_size).floor().format()
        )
        return feature.set(
            {
                "_row": row_id,
                "lon": ee.Number(feature.get(lon)),
                "lat": ee.Number(feature.get(lat)),
                "y": ee.Number(feature.get(target)).gte(threshold).int(),
                "block_x_m": x,
                "block_y_m": y_coord,
                "block_id": block_id,
            }
        )

    if input_asset_id:
        required = [lon, lat, target]
        if use_existing_block_columns:
            required.extend([block_x, block_y])
        source_points = ee.FeatureCollection(input_asset_id).filter(ee.Filter.notNull(required))

        points = source_points.map(normalize_feature)
        point_asset = input_asset_id
    else:
        observations = _download_csv(csv_url)
        required = [lon, lat, target]
        if use_existing_block_columns:
            required.extend([block_x, block_y])
        missing = [column for column in required if column not in observations]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        observations = observations.dropna(subset=required).reset_index(drop=True)
        observations["_row"] = np.arange(len(observations))

        features = []
        for record in observations.to_dict("records"):
            properties = {
                "_row": int(record["_row"]),
                lon: float(record[lon]),
                lat: float(record[lat]),
                target: float(record[target]),
            }
            if use_existing_block_columns:
                properties[block_x] = float(record[block_x])
                properties[block_y] = float(record[block_y])
            features.append(
                ee.Feature(
                    ee.Geometry.Point([float(record[lon]), float(record[lat])]),
                    properties,
                )
            )
        task = ee.batch.Export.table.toAsset(
            collection=ee.FeatureCollection(features).map(normalize_feature),
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

    _, sklearn_probability, sklearn_metrics = fit_score(capped_train)
    sklearn_fpr, sklearn_tpr, sklearn_thresholds = roc_curve(
        test.y, sklearn_probability
    )
    sklearn_threshold = float(
        sklearn_thresholds[np.argmax(sklearn_tpr - sklearn_fpr)]
    )

    selected_train_ids = capped_train["_row"].tolist()
    gee_train = reference_samples.filter(
        ee.Filter.inList("_row", selected_train_ids)
    )
    gee_test = reference_samples.filter(
        ee.Filter.inList("block_id", sorted(test_blocks))
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
    gee_scored = ee.data.computeFeatures(
        {
            "expression": gee_test.classify(gee_holdout_model).select(
                ["y", "classification"]
            ),
            "fileFormat": "PANDAS_DATAFRAME",
        }
    )
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
    production_ids = production["_row"].tolist()
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

    split_data = reference[["lon", "lat", "block_id"]].copy()
    split_data["split"] = np.where(
        split_data.block_id.isin(test_blocks), "holdout", "train"
    )
    sk_tn, sk_fp, sk_fn, sk_tp = confusion_matrix(
        test.y, sklearn_probability >= 0.5
    ).ravel()
    gee_tn, gee_fp, gee_fn, gee_tp = confusion_matrix(
        gee_y, gee_probability >= 0.5
    ).ravel()

    report = {
        "schema_version": "1.0",
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
            "block_crs": block_crs,
            "block_coordinate_source": "columns" if use_existing_block_columns else "geometry",
            "test_block_fraction": test_fraction,
            "points_per_block": point_caps,
            "block_fractions": block_fractions,
            "auc_tolerance": auc_tolerance,
            "csv_url": csv_url,
            "input_asset_id": input_asset_id,
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
            "points": split_data.to_dict("records"),
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

    bucket_name = config.get("results_bucket") or os.environ["RESULTS_BUCKET"]
    results_prefix = (config.get("results_prefix") or "").strip("/")
    object_prefix = f"{results_prefix}/{run_name}" if results_prefix else run_name
    object_name = f"{object_prefix}/results.json"
    viewer_object_name = f"{object_prefix}/viewer.html"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(object_name).upload_from_string(
        report_json, content_type="application/json"
    )
    results_uri = f"gs://{bucket_name}/{object_name}"
    results_url = public_https_url(bucket_name, object_name)

    viewer_html = _viewer_html(report, results_url)
    bucket.blob(viewer_object_name).upload_from_string(
        viewer_html, content_type="text/html; charset=utf-8"
    )
    viewer_uri = f"gs://{bucket_name}/{viewer_object_name}"
    viewer_url = public_https_url(bucket_name, viewer_object_name)

    return {
        "status": "success",
        "run_name": run_name,
        "results_uri": results_uri,
        "results_url": results_url,
        "viewer_uri": viewer_uri,
        "viewer_url": viewer_url,
        "assets": report["assets"],
        "outputs": {
            "results_json": {"gcs_uri": results_uri, "url": results_url},
            "viewer_html": {"gcs_uri": viewer_uri, "url": viewer_url},
        },
        "summary": {
            "recommended_points_per_block": selected_cap,
            "recommended_minimum_block_fraction": recommended_block_fraction,
            "sklearn_auc": sklearn_metrics["auc"],
            "earth_engine_auc": float(gee_auc),
        },
        "results": report,
    }
