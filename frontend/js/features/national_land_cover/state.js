export const State = {
    customSourceType: 'csv', // Default source type
    file: null,
    headers: [],
    
    // Hardcoded advanced parameters for all Tasking jobs
    taskingDefaults: {
        cloud_project: "pc655-gcpa-unops-geo-is",
        block_crs: "EPSG:3857",
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
        results_bucket: "orbit-lc",
        results_prefix: "space_for_time_tasking/results"
    },
    
    setFile(file) {
        this.file = file;
    },
    
    setHeaders(headers) {
        this.headers = headers;
    }
};
