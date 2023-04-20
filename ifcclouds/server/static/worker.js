const colors = [
    { r: 1.0, g: 0.0, b: 0.0 },
    { r: 0.0, g: 1.0, b: 0.0 },
    { r: 0.0, g: 0.0, b: 1.0 },
    { r: 1.0, g: 1.0, b: 0.0 },
    { r: 1.0, g: 0.0, b: 1.0 },
    { r: 0.0, g: 1.0, b: 1.0 },
    { r: 1.0, g: 1.0, b: 1.0 },
    { r: 0.5, g: 0.5, b: 0.5 },
    { r: 0.5, g: 0.0, b: 0.0 },
    { r: 0.0, g: 0.5, b: 0.0 },
    { r: 0.0, g: 0.0, b: 0.5 },
    { r: 0.5, g: 0.5, b: 0.0 },
];

/*
 * segment point array in chunks of size n
 */
function chunkArray(array, n=4096) {
    const chunks = [];
    for (let i = 0; i < array.length; i += n) {
        chunks.push(array.slice(i, i + n));
    }
    return chunks;
}

/*
 * send pointcloud to server in chunks
 */
async function sendPointCloud(pointcloud) {
    // shuffle pointcloud
    for (let i = pointcloud.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [pointcloud[i], pointcloud[j]] = [pointcloud[j], pointcloud[i]];
    }
    const chunks = chunkArray(pointcloud);
    const data = [];
    for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        let res = await sendChunkToServer(chunk);
        res = res.output[0];
        data.push(chunk.map((point, j) => {
            return {
                x: point[0],
                y: point[1],
                z: point[2],
                ...colors[res[j]],
            };
        }));
    }

    const unchunked = [].concat.apply([], data);
    return unchunked;
}

async function sendChunkToServer(chunk) {
    chunk = { input: chunk };
    const data = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(chunk),
    });
    return data.json();
}


onmessage = function(e) {
    const pointcloud = e.data;
    sendPointCloud(pointcloud.map((point) => {
        return [point.x, point.y, point.z];
    })).then((data) => {
        postMessage(data);
    });
}

