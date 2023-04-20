import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { DragControls } from 'three/addons/controls/DragControls.js';
import Stats from 'three/addons/libs/stats.module'

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
                    pointArray.push({
                        x: x,
                        y: y,
                        z: z,
                        ...colors[class_idx],
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

const dragTargets = [];

const controls = new OrbitControls( camera, renderer.domElement );
const dragControls = new DragControls( dragTargets, camera, renderer.domElement );

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
    // remove previous pointcloud
    const prev = scene.children[0];
    if(prev) {
        console.log('Removing previous pointcloud');
        prev.traverse((obj) => {
            if(obj.geometry) {
              obj.geometry.dispose();
            }
            if(obj.material) {
              if(Array.isArray(obj.material)) {
                obj.material.forEach((m) => {
                  m.dispose();
                });
              } else {
                obj.material.dispose();
              }
            }
        });
        scene.remove(prev);
    }

    const file = e.target.files[0];
    const pointArray = [];
    await readPlyFile(file, pointArray);
    console.log('Sending pointcloud to server');
    const worker = new Worker('worker.js');
    worker.postMessage(pointArray);
    worker.onmessage = function(e) {
        console.log('Received pointcloud from server', e.data);
    };
    const pointcloud = createPointCloud(pointArray);
    console.log('Adding pointcloud to scene');
    scene.add(pointcloud);
});

const stats = new Stats();
stats.dom.style.position = 'fixed';
stats.dom.style.left = 'auto';
stats.dom.style.right = '0';
stats.dom.style.top = '0';
document.body.appendChild(stats.dom);

function animate() {
	requestAnimationFrame( animate );

    controls.update();
	renderer.render( scene, camera );
    stats.update();
}

animate();