export const parseCSVHeaders = (file) => {
    return new Promise((resolve, reject) => {
        Papa.parse(file, {
            header: true,
            preview: 1, // We only need the first row to get headers
            complete: function(results) {
                if (results.meta && results.meta.fields) {
                    resolve(results.meta.fields);
                } else {
                    reject(new Error("Could not parse headers from CSV."));
                }
            },
            error: function(error) {
                reject(error);
            }
        });
    });
};
