import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/*
 * Create pointcloud from array of points
 * @param {Array} points - Array of points
*/
function createPointCloud(points) {
    console.log('Creating pointcloud');
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(points.length * 3);
    const colors = new Float32Array(points.length * 3);
    const color = new THREE.Color();
    let k = 0;
    for (let i = 0; i < points.length; i++) {
        const point = points[i];
        positions[k] = point.x;
        positions[k + 1] = point.y;
        positions[k + 2] = point.z;
        color.setRGB(point.r, point.g, point.b);
        colors[k] = color.r;
        colors[k + 1] = color.g;
        colors[k + 2] = color.b;
        k += 3;
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.computeBoundingSphere();
    const material = new THREE.PointsMaterial({
        size: 0.01,
        vertexColors: true,
    });
    const pointcloud = new THREE.Points(geometry, material);
    return pointcloud;
}

function classToColor(class_idx, num_classes=20) {
    // create num_classes colors
    const colors = [];
    for (let i = 0; i < num_classes; i++) {
        const r = Math.random();
        const g = Math.random();
        const b = Math.random();
        colors.push({ r: r, g: g, b: b });
    }
    return colors[class_idx];
}

/*
 * Read ply file and return array of points
 * file has the format x, y, z, class_idx
 */
function readPlyFile(file, pointArray) {
    console.log('Reading file');
    return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = function(e) {
        const text = reader.result;
        const lines = text.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const values = line.split(' ');
            if (values.length === 4) {
                const x = parseFloat(values[0]);
                const y = parseFloat(values[1]);
                const z = parseFloat(values[2]);
                const class_idx = parseInt(values[3]);
                const color = classToColor(class_idx);
                pointArray.push({
                    x: x,
                    y: y,
                    z: z,
                    r: color.r,
                    g: color.g,
                    b: color.b,
                });
            }
        }
        resolve(pointArray);
    };
    reader.onerror = reject;
    reader.readAsText(file);
});
}

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );

const renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
const controls = new OrbitControls( camera, renderer.domElement );

document.body.appendChild( renderer.domElement );

camera.position.z = 5;
controls.update();

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

window.addEventListener('resize', onWindowResize, false);

const fileInput = document.getElementById('file-input');
fileInput.addEventListener('change', async function(e) {
    const file = e.target.files[0];
    const pointArray = [];
    await readPlyFile(file, pointArray);
    const pointcloud = createPointCloud(pointArray);
    console.log('Adding pointcloud to scene');
    scene.add(pointcloud);
});

function animate() {
	requestAnimationFrame( animate );

    controls.update();
	renderer.render( scene, camera );
}

animate();