export const State = {
    isDemoMode: false,
    file: null,
    headers: [],
    
    // Default Ireland demo parameters
    demoParams: {
        input_asset_id: "projects/pc655-gcpa-unops-geo-is/assets/test_ireland_noAE",
        cloud_project: "pc655-gcpa-unops-geo-is",
        run_name: "test_4",
        longitude_column: "lon",
        latitude_column: "lat",
        block_crs: "EPSG:3857",
        target_column: "LOI_PCT",
        target_threshold: 30,
        reference_year: 2022,
        block_size_m: 10000,
        test_block_fraction: 0.2,
        points_per_block: [ 1, 2, 5, 10, 20, "all" ],
        block_fractions: [ 0.1, 0.25, 0.5, 0.75, 1.0 ],
        auc_tolerance: 0.01,
        number_of_trees: 100,
        number_of_embedding_bands: 64,
        sampling_scale_m: 10,
        seed: 42,
        asset_root: "projects/pc655-gcpa-unops-geo-is/assets/space_for_time_tasking",
        sample_asset_id: "projects/pc655-gcpa-unops-geo-is/assets/space_for_time_tasking/samples_asset_run_004",
        model_asset_id: "projects/pc655-gcpa-unops-geo-is/assets/space_for_time_tasking/rf_asset_run_004",
        results_bucket: "orbit-lc",
        results_prefix: "space_for_time_tasking/results"
    },
    
    setFile(file) {
        this.file = file;
    },
    
    setHeaders(headers) {
        this.headers = headers;
    },
    
    setDemoMode(enabled) {
        this.isDemoMode = enabled;
    }
};
