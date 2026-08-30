"""SDG 11.3.1 — Urban Land Consumption retrieval method.

Trains a Random Forest classifier on Google Satellite Embeddings to
estimate annual urban extent, compares with Dynamic World, computes
population via multiple datasets, and exports SDG 11.3.1 metrics
to Google Cloud Storage via Earth Engine batch tasks.
"""

import json
import os
from typing import Any, Dict, Optional

import ee
import yaml

if __package__:
    from ....utils.gee_common import (
        _validate_inputs,
        aggregate_regional_stats,
        export_image_to_gcs,
        export_table_to_gcs,
        export_vector_to_gcs,
        initialize_ee,
    )
    from ....utils.ae_specific import embeddings_by_year, generate_stratified_sample
else:
    from utils.gee_common import (
        _validate_inputs,
        aggregate_regional_stats,
        export_image_to_gcs,
        export_table_to_gcs,
        export_vector_to_gcs,
        initialize_ee,
    )
    from utils.ae_specific import embeddings_by_year, generate_stratified_sample

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_CONFIG_PATH, "r") as _f:
    _CONFIG = yaml.safe_load(_f)
_DATASETS = _CONFIG["datasets"]
_PARAMS = _CONFIG["parameters"]


def run_11_03_01(
    map_year: int,
    year_start: int,
    year_end: int,
    gcs_bucket: str,
    country: Optional[str] = None,
    aoi_geojson: Optional[Dict[str, Any]] = None,
    sample_points: Optional[int] = None,
    sample_scale: Optional[int] = None,
    embedding_scale: Optional[int] = None,
    threshold: Optional[float] = None,
    trees: Optional[int] = None,
    seed: Optional[int] = None,
    project: Optional[str] = None,
    export_name: Optional[str] = None,
    gcs_prefix: Optional[str] = None,
    export_formats: Optional[list[str]] = None,
    model_asset_id: Optional[str] = None,
    span_target: Optional[int] = None,
    area_scale: Optional[int] = None,
    wb_population_dict: Optional[Dict[int, int]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the classification pipeline and start GCS table export tasks."""
    sample_points = sample_points if sample_points is not None else _PARAMS["sample_points"]
    sample_scale = sample_scale if sample_scale is not None else _PARAMS["sample_scale"]
    embedding_scale = embedding_scale if embedding_scale is not None else _PARAMS["embedding_scale"]
    threshold = threshold if threshold is not None else _PARAMS["threshold"]
    trees = trees if trees is not None else _PARAMS["trees"]
    seed = seed if seed is not None else _PARAMS["seed"]
    span_target = span_target if span_target is not None else _PARAMS.get("span_target", 5)
    
    # The scale (in meters) to compute urban areas and SDG metrics.
    # Recommended: 100 for sample runs to prevent unexpected compute costs,
    # and 10 for final export runs (heavier compute).
    area_scale = area_scale if area_scale is not None else _PARAMS.get("area_scale", 100)

    # =====================================================================
    # Cell 1 -- Config + datasets + AOI
    # =====================================================================
    _validate_inputs(country, year_start, year_end, threshold, gcs_bucket)
    initialize_ee(project=project)

    countries = ee.FeatureCollection(_DATASETS["countries"])
    ghs_built_collection = ee.ImageCollection(_DATASETS.get("ghs_built", "JRC/GHSL/P2023A/GHS_BUILT_S_10m"))
    dynamic_world = ee.ImageCollection(_DATASETS.get("dynamic_world", "GOOGLE/DYNAMICWORLD/V1"))
    embeddings = ee.ImageCollection(_DATASETS["embeddings"])

    if aoi_geojson:
        boundary_geometry = ee.Geometry(aoi_geojson)
        boundary = ee.FeatureCollection([ee.Feature(boundary_geometry)])
        bbox = boundary
        region_label = "custom_aoi"
    else:
        boundary = countries.filter(ee.Filter.eq("ADM0_NAME", country))
        boundary_size = boundary.size().getInfo()
        if boundary_size == 0:
            raise ValueError(f'No country found with ADM0_NAME="{country}"')
        bbox = boundary.bounds()
        boundary_geometry = boundary.geometry()
        region_label = country.lower().replace(" ", "_").replace("-", "_")

    # =====================================================================
    # Cell 3 -- Train RF on MAP_YEAR GHS labels, then validate
    # =====================================================================
    ghs_img = ee.Image(f"{_DATASETS.get('ghs_built', 'JRC/GHSL/P2023A/GHS_BUILT_S_10m')}/{map_year}").select('built_surface')
    filtered_label = ghs_img.clip(boundary_geometry).gt(0).unmask(0).rename("b1").toByte()

    map_year_embeddings = embeddings_by_year(embeddings, boundary, map_year)
    input_properties = map_year_embeddings.bandNames()

    combined_img = map_year_embeddings.addBands(filtered_label)
    
    sample = combined_img.stratifiedSample(
        numPoints=sample_points,
        classBand="b1",
        classValues=[0, 1],
        classPoints=[sample_points, sample_points],
        region=bbox,
        scale=sample_scale,
        geometries=True,
        seed=seed,
        tileScale=16
    ).filterBounds(boundary).filter(ee.Filter.notNull(input_properties))

    collection_with_random = sample.randomColumn(columnName="random", seed=seed)
    training_set = collection_with_random.filter(ee.Filter.lt("random", 0.7))
    validation_set = collection_with_random.filter(ee.Filter.gte("random", 0.7))

    binary_filter = ee.Filter.inList("b1", [0, 1])
    filtered_collection = training_set

    default_export_name = f"urban_extent_{region_label}_{year_start}-{year_end}"
    final_export_name = export_name or default_export_name

    classifier_export_task = None
    if model_asset_id:
        classifier = ee.Classifier.load(model_asset_id)
    else:
        classifier = (
            ee.Classifier.smileRandomForest(trees, None, 1, 0.5, None, seed)
            .train(
                features=filtered_collection,
                classProperty="b1",
                inputProperties=input_properties,
            )
        )
        assetId = f"projects/{project}/assets/{final_export_name}_classifier" if project else f"{final_export_name}_classifier"
        classifier_export_task = ee.batch.Export.classifier.toAsset(
            classifier=classifier, 
            description=f"{final_export_name}_classifier_export", 
            assetId=assetId
        )
        classifier_export_task.start()

    # =====================================================================
    # Cell 4 -- Urban area per year (RF, DW) at area_scale, plus 3-yr smoothing
    # =====================================================================
    year_list = list(range(year_start, year_end + 1))
    year_names = [f"Y{year}" for year in year_list]
    
    # Keep output_image for GeoTIFF exports
    yearly_images = []
    for year in year_list:
        classified = embeddings_by_year(embeddings, boundary, year).classify(classifier)
        yearly_images.append(classified)
    
    all_year_results = ee.ImageCollection(yearly_images).toBands().rename(year_names)
    output_image = all_year_results.toByte()

    def urbanKm2(mask, scale=area_scale):
        m2 = aggregate_regional_stats(
            image=mask.multiply(ee.Image.pixelArea()),
            geometry=boundary_geometry,
            scale=scale
        )
        return ee.Number(m2).divide(1e6)
        
    urbRF = {}
    urbDW = {}
    
    for y, y_name in zip(year_list, year_names):
        rfMask = output_image.select(y_name).rename("built")
        dw_s = ee.Date.fromYMD(y, 1, 1)
        dwMask = dynamic_world.filterDate(dw_s, dw_s.advance(1, "year")).filterBounds(boundary_geometry).select("built").median().gte(0.5).rename("dw").toByte()
        
        urbRF[y] = urbanKm2(rfMask)
        urbDW[y] = urbanKm2(dwMask)
        
    def smooth3(rawDict, yearsArr):
        out = {}
        sm = []
        for i in range(1, len(yearsArr) - 1):
            y = yearsArr[i]
            y0 = yearsArr[i-1]
            y2 = yearsArr[i+1]
            out[y] = ee.Number(rawDict[y0]).add(rawDict[y]).add(rawDict[y2]).divide(3)
            sm.append(y)
        return out, sm
        
    smRF, sm_years = smooth3(urbRF, year_list)
    smDW, _ = smooth3(urbDW, year_list)
    
    # =====================================================================
    # Cell 5 -- Population per year: World Bank + GHS-POP + WorldPop + GPW v4.11
    # =====================================================================
    def popSum(img, band, scale):
        return aggregate_regional_stats(img.select(band), boundary_geometry, scale)
        
    ghs_pop_coll = ee.ImageCollection(_DATASETS.get("ghs_pop", "JRC/GHSL/P2023A/GHS_POP"))
    gpw_coll = ee.ImageCollection(_DATASETS.get("gpw411", "CIESIN/GPWv411/GPW_Population_Count"))
    wp_coll = ee.ImageCollection(_DATASETS.get("worldpop", "WorldPop/GP/100m/pop"))
    
    pop_anchor_years = _PARAMS.get("pop_anchor_years", [2015, 2020])
    
    ghsAnchors = []
    gpwAnchors = []
    wpAnchors = []
    
    for y in pop_anchor_years:
        ghsAnchors.append((y, popSum(ghs_pop_coll.filterDate(f"{y}-01-01", f"{y+1}-01-01").first(), "population_count", 100)))
        gpwAnchors.append((y, popSum(gpw_coll.filterDate(f"{y}-01-01", f"{y+1}-01-01").first(), "population_count", 1000)))
        wpAnchors.append((y, popSum(wp_coll.filterBounds(boundary_geometry).filter(ee.Filter.eq("year", y)).select("population").mosaic(), "population", 100)))
    
    def interpVal(points, year):
        y0, p0 = points[0]
        y1, p1 = points[1]
        f = (year - y0) / (y1 - y0)
        return p0.add(p1.subtract(p0).multiply(f))
        
    # Auto-fetch World Bank population if not provided, matching the Python notebook logic
    if not wb_population_dict and country:
        try:
            import requests
            _norm = lambda s: "".join(c for c in s.lower() if c.isalnum())
            _clist = requests.get("https://api.worldbank.org/v2/country",
                                  params={"format": "json", "per_page": 400}, timeout=30).json()[1]
            _iso3 = next((c["id"] for c in _clist if _norm(c["name"]) == _norm(country)),
                         next((c["id"] for c in _clist if _norm(country) in _norm(c["name"])), None))
            if _iso3:
                _pop_res = requests.get(f"https://api.worldbank.org/v2/country/{_iso3}/indicator/SP.POP.TOTL",
                                        params={"format": "json", "date": f"2000:{year_end}", "per_page": 100}, timeout=30).json()
                if len(_pop_res) > 1 and _pop_res[1]:
                    wb_population_dict = {
                        int(d["date"]): d["value"]
                        for d in _pop_res[1]
                        if d["value"] is not None
                    }
        except Exception as e:
            print(f"Warning: Failed to fetch World Bank population for {country}: {e}")

    def wbPop(y):
        if not wb_population_dict:
            return None
        if y in wb_population_dict:
            return ee.Number(wb_population_dict[y])
        nearest = min(wb_population_dict.keys(), key=lambda k: abs(k - y))
        return ee.Number(wb_population_dict[nearest])
        
    pop_sources = {
        "GHS_POP": lambda y: interpVal(ghsAnchors, y),
        "WorldPop": lambda y: interpVal(wpAnchors, y),
        "GPW_v411": lambda y: interpVal(gpwAnchors, y)
    }
    if wb_population_dict:
        pop_sources["WorldBank"] = wbPop
    
    pop_keys = list(pop_sources.keys())
    
    # =====================================================================
    # Cell 6 -- SDG 11.3.1 LCRPGR on the 3-yr smoothed series
    # =====================================================================
    def annualLcrpgrRows(method, Usm, smYears, popKeys, popSources):
        rows = []
        for i in range(len(smYears) - 1):
            y0 = smYears[i]
            y1 = smYears[i+1]
            span = y1 - y0
            U0 = Usm[y0]
            U1 = Usm[y1]
            lcr = U1.subtract(U0).divide(U0.multiply(span))
            for ps in popKeys:
                P0 = popSources[ps](y0)
                P1 = popSources[ps](y1)
                pgr = P1.divide(P0).log().divide(span)
                rows.append(ee.Feature(None, {
                    "window": f"{y0}-{y1}",
                    "mid_year": y0,
                    "urban_method": method,
                    "pop_source": ps,
                    "LCR": lcr,
                    "PGR": pgr,
                    "LCRPGR": lcr.divide(pgr)
                }))
        return rows
        
    def spanEnd(smYears, startYr, target):
        s1 = smYears[-1]
        for y in smYears:
            if y <= startYr + target:
                s1 = y
        return s1
        
    def spanRows(method, Usm, smYears, popKeys, popSources, target):
        rows = []
        if not smYears:
            return rows
        t0 = smYears[0]
        t1 = spanEnd(smYears, t0, target)
        span = t1 - t0
        if span == 0:
            return rows
            
        U0 = Usm[t0]
        U1 = Usm[t1]
        lcr = U1.subtract(U0).divide(U0.multiply(span))
        totChg = U1.subtract(U0).divide(U0)
        
        for ps in popKeys:
            P0 = popSources[ps](t0)
            P1 = popSources[ps](t1)
            pgr = P1.divide(P0).log().divide(span)
            rows.append(ee.Feature(None, {
                "window": f"{t0}-{t1}",
                "span": span,
                "urban_method": method,
                "pop_source": ps,
                "LCR": lcr,
                "PGR": pgr,
                "LCRPGR": lcr.divide(pgr),
                "BUpc_t0_m2": U0.multiply(1e6).divide(P0),
                "BUpc_t1_m2": U1.multiply(1e6).divide(P1),
                "BU_total_change": totChg
            }))
        return rows
        
    if not sm_years:
        area_smoothed_fc = ee.FeatureCollection([ee.Feature(None, {"error": "Not enough years for smoothing (need >=3)"})])
        annual_fc = ee.FeatureCollection([ee.Feature(None, {"error": "Not enough years for smoothing (need >=3)"})])
        span_fc = ee.FeatureCollection([ee.Feature(None, {"error": "Not enough years for smoothing (need >=3)"})])
    else:
        area_smoothed_fc = ee.FeatureCollection([
            ee.Feature(None, {"year": y, "RF_sm_km2": smRF[y], "DW_sm_km2": smDW[y]})
            for y in sm_years
        ])
        
        annual_rows = annualLcrpgrRows("RF", smRF, sm_years, pop_keys, pop_sources) + \
                      annualLcrpgrRows("DW", smDW, sm_years, pop_keys, pop_sources)
        annual_fc = ee.FeatureCollection(annual_rows if annual_rows else [ee.Feature(None, {"error": "empty"})])
        
        span_rows = spanRows("RF", smRF, sm_years, pop_keys, pop_sources, span_target) + \
                    spanRows("DW", smDW, sm_years, pop_keys, pop_sources, span_target)
        span_fc = ee.FeatureCollection(span_rows if span_rows else [ee.Feature(None, {"error": "empty"})])
    
    normalized_prefix = (gcs_prefix or "").strip().strip("/")
    base_prefix = f"{normalized_prefix}/{final_export_name}" if normalized_prefix else final_export_name
    
    result: Dict[str, Any] = {
        "country": country,
        "aoi_geojson": aoi_geojson,
        "project": project,
        "training_year": map_year,
        "year_start": year_start,
        "year_end": year_end,
        "threshold": threshold,
        "export_name": final_export_name,
        "export_started": False,
        "export_target": "gcs_tables",
        "gcs_bucket": gcs_bucket,
        "gcs_prefix": normalized_prefix or None,
    }

    metrics_for_export = {
        "training_accuracy": None,
        "validation_accuracy": None,
        "validation_kappa": None,
        "training_confusion_matrix": None,
        "validation_confusion_matrix": None,
    }

    try:
        result["training_samples"] = training_set.size().getInfo()
        result["validation_samples"] = validation_set.size().getInfo()
        result["filtered_training_samples"] = filtered_collection.size().getInfo()

        filtered_validation_set = validation_set.filter(binary_filter)
        validation_classified = filtered_validation_set.classify(classifier)
        training_confusion = classifier.confusionMatrix()
        validation_confusion = validation_classified.errorMatrix("b1", "classification")

        training_accuracy = training_confusion.accuracy().getInfo()
        validation_accuracy = validation_confusion.accuracy().getInfo()
        validation_kappa = validation_confusion.kappa().getInfo()
        training_confusion_matrix = training_confusion.getInfo()
        validation_confusion_matrix = validation_confusion.getInfo()

        metrics_for_export = {
            "training_accuracy": training_accuracy,
            "validation_accuracy": validation_accuracy,
            "validation_kappa": validation_kappa,
            "training_confusion_matrix": json.dumps(training_confusion_matrix),
            "validation_confusion_matrix": json.dumps(validation_confusion_matrix),
        }
        
        result["metrics"] = {
            "training_accuracy": training_accuracy,
            "validation_accuracy": validation_accuracy,
            "validation_kappa": validation_kappa,
            "training_confusion_matrix": training_confusion_matrix,
            "validation_confusion_matrix": validation_confusion_matrix,
        }
    except Exception:
        pass

    stats_properties = {
        "country": country or "",
        "project": project or "",
        "training_year": map_year,
        "year_start": year_start,
        "year_end": year_end,
        "threshold": threshold,
        "sample_points": sample_points,
        "sample_scale": sample_scale,
        "embedding_scale": embedding_scale,
        "trees": trees,
        "seed": seed,
        "training_samples": training_set.size(),
        "validation_samples": validation_set.size(),
        "filtered_training_samples": filtered_collection.size(),
        "training_accuracy": metrics_for_export["training_accuracy"],
        "validation_accuracy": metrics_for_export["validation_accuracy"],
        "validation_kappa": metrics_for_export["validation_kappa"],
        "training_confusion_matrix": metrics_for_export["training_confusion_matrix"],
        "validation_confusion_matrix": metrics_for_export["validation_confusion_matrix"],
    }
    stats_fc = ee.FeatureCollection([ee.Feature(None, stats_properties)])

    task_ids = {}
    task_states = {}
    task_descriptions = {}

    if classifier_export_task:
        try:
            classifier_status = classifier_export_task.status()
            task_ids["classifier_export"] = classifier_status.get("id")
            task_states["classifier_export"] = classifier_status.get("state")
            task_descriptions["classifier_export"] = classifier_status.get("description")
        except Exception:
            pass

    # =====================================================================
    # Cell 7 -- BATCH: considered area_scale SDG numbers -> GCS CSV/GeoJSON
    # =====================================================================
    # --- CSV Exports ---
    if not export_formats or any(fmt.lower() == "csv" for fmt in export_formats):
        stats_task = export_table_to_gcs(
            collection=stats_fc,
            description=f"{final_export_name}_prediction_stats",
            bucket=gcs_bucket,
            filename_prefix=f"{base_prefix}_prediction_stats",
        )
        task_ids["prediction_stats"] = stats_task.status().get("id")

        area_task = export_table_to_gcs(
            collection=area_smoothed_fc,
            description=f"{final_export_name}_area{area_scale}m_smoothed",
            bucket=gcs_bucket,
            filename_prefix=f"{base_prefix}_area{area_scale}m_smoothed",
        )
        task_ids[f"area{area_scale}m_smoothed"] = area_task.status().get("id")
        
        annual_task = export_table_to_gcs(
            collection=annual_fc,
            description=f"{final_export_name}_LCRPGR_annual_{area_scale}m",
            bucket=gcs_bucket,
            filename_prefix=f"{base_prefix}_LCRPGR_annual_{area_scale}m",
        )
        task_ids[f"LCRPGR_annual_{area_scale}m"] = annual_task.status().get("id")
        
        span_task = export_table_to_gcs(
            collection=span_fc,
            description=f"{final_export_name}_LCRPGR_5yr_{area_scale}m",
            bucket=gcs_bucket,
            filename_prefix=f"{base_prefix}_LCRPGR_5yr_{area_scale}m",
        )
        task_ids[f"LCRPGR_5yr_{area_scale}m"] = span_task.status().get("id")

    # --- GeoJSON Exports ---
    if export_formats and any(fmt.lower() == "geojson" for fmt in export_formats):
        stats_geojson_task = export_vector_to_gcs(
            collection=stats_fc,
            description=f"{final_export_name}_prediction_stats_geojson",
            bucket=gcs_bucket,
            filename_prefix=f"{base_prefix}_prediction_stats_geojson",
        )
        task_ids["prediction_stats_geojson"] = stats_geojson_task.status().get("id")

        area_geojson_task = export_vector_to_gcs(
            collection=area_smoothed_fc,
            description=f"{final_export_name}_area{area_scale}m_smoothed_geojson",
            bucket=gcs_bucket,
            filename_prefix=f"{base_prefix}_area{area_scale}m_smoothed_geojson",
        )
        task_ids[f"area{area_scale}m_smoothed_geojson"] = area_geojson_task.status().get("id")
        
        annual_geojson_task = export_vector_to_gcs(
            collection=annual_fc,
            description=f"{final_export_name}_LCRPGR_annual_{area_scale}m_geojson",
            bucket=gcs_bucket,
            filename_prefix=f"{base_prefix}_LCRPGR_annual_{area_scale}m_geojson",
        )
        task_ids[f"LCRPGR_annual_{area_scale}m_geojson"] = annual_geojson_task.status().get("id")
        
        span_geojson_task = export_vector_to_gcs(
            collection=span_fc,
            description=f"{final_export_name}_LCRPGR_5yr_{area_scale}m_geojson",
            bucket=gcs_bucket,
            filename_prefix=f"{base_prefix}_LCRPGR_5yr_{area_scale}m_geojson",
        )
        task_ids[f"LCRPGR_5yr_{area_scale}m_geojson"] = span_geojson_task.status().get("id")

    # =====================================================================
    # Cell 8 -- OPTIONAL : export the per-year RF built layers
    # =====================================================================
    # --- GeoTIFF Exports ---
    if export_formats and any(fmt.lower() == "geotiff" for fmt in export_formats):
        geotiff_file_name_prefix = f"{base_prefix}_urban_extent"
        geotiff_task = export_image_to_gcs(
            image=output_image,
            description=f"{final_export_name}_urban_extent_geotiff",
            bucket=gcs_bucket,
            filename_prefix=geotiff_file_name_prefix,
            scale=embedding_scale,
            region=boundary_geometry,
        )
        task_ids["urban_extent_geotiff"] = geotiff_task.status().get("id")
        result["geotiff_file_name_prefix"] = geotiff_file_name_prefix

    result.update({
        "export_started": True,
        "task_ids": task_ids,
        "task_states": task_states,
        "task_descriptions": task_descriptions,
    })

    try:
        url = f"/cog/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}?url=gs://{gcs_bucket}/{geotiff_file_name_prefix}.tif&bidx=1&nodata=0&colormap=%7B%221%22:%22%23FF5722FF%22%7D"
        result["layers"] = [
            {
                "id": "urban_extent",
                "name": f"Urban Extent ({year_end})",
                "tile_url": url,
                "is_cog": True
            }
        ]
    except Exception as e:
        print(f"Failed to generate map tile: {e}")

    return result
