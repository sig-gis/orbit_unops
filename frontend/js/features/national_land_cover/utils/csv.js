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

export const parseCSVData = (file) => {
    return new Promise((resolve, reject) => {
        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: function(results) {
                if (results.data) {
                    resolve(results.data);
                } else {
                    reject(new Error("Could not parse data from CSV."));
                }
            },
            error: function(error) {
                reject(error);
            }
        });
    });
};
