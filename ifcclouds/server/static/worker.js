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
    const chunks = chunkArray(pointcloud);
    const data = [];
    for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        const res = await sendChunkToServer(chunk);
        data.push(res);
    }

    // unchunk data
    const unchunked = [];
    for (let i = 0; i < data.length; i++) {
        const chunk = data[i];
        for (let j = 0; j < chunk.length; j++) {
            unchunked.push(chunk[j]);
        }
    }
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

