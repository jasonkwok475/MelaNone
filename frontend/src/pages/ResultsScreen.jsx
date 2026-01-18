import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import '../styles/ResultsScreen.css';

export default function ResultsScreen({ data }) {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const meshRef = useRef(null);
  const autoRotateRef = useRef(true);

  // Sample data - will be replaced with actual backend data
  const analysisData = data || {
    totalObjectsAnalyzed: 24,
    concerningSpots: 3
  };

  useEffect(() => {
    if (!containerRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0e27);
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.z = 3;
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Create a sample 3D object (sphere with bumpy texture)
    const geometry = new THREE.IcosahedronGeometry(1.5, 4);
    const material = new THREE.MeshPhongMaterial({
      color: 0xff6b9d,
      emissive: 0x2a0845,
      shininess: 100
    });
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);
    meshRef.current = mesh;

    // Lighting
    const light1 = new THREE.DirectionalLight(0xffffff, 1);
    light1.position.set(5, 5, 5);
    scene.add(light1);

    const light2 = new THREE.DirectionalLight(0x00d4ff, 0.5);
    light2.position.set(-5, -5, 5);
    scene.add(light2);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);

    // Mouse controls for rotation
    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };

    renderer.domElement.addEventListener('mousedown', (e) => {
      isDragging = true;
      previousMousePosition = { x: e.clientX, y: e.clientY };
      autoRotateRef.current = false;
    });

    renderer.domElement.addEventListener('mousemove', (e) => {
      if (isDragging && meshRef.current) {
        const deltaX = e.clientX - previousMousePosition.x;
        const deltaY = e.clientY - previousMousePosition.y;

        meshRef.current.rotation.y += deltaX * 0.01;
        meshRef.current.rotation.x += deltaY * 0.01;

        previousMousePosition = { x: e.clientX, y: e.clientY };
      }
    });

    renderer.domElement.addEventListener('mouseup', () => {
      isDragging = false;
      autoRotateRef.current = true;
    });

    renderer.domElement.addEventListener('mouseleave', () => {
      isDragging = false;
      autoRotateRef.current = true;
    });

    // Zoom with mouse wheel
    renderer.domElement.addEventListener('wheel', (e) => {
      e.preventDefault();
      camera.position.z += e.deltaY * 0.005;
      camera.position.z = Math.max(1, Math.min(10, camera.position.z));
    }, { passive: false });

    // Handle window resize
    const handleResize = () => {
      if (!containerRef.current) return;
      const width = containerRef.current.clientWidth;
      const height = containerRef.current.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };

    window.addEventListener('resize', handleResize);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);

      if (autoRotateRef.current && meshRef.current) {
        meshRef.current.rotation.x += 0.002;
        meshRef.current.rotation.y += 0.003;
      }

      renderer.render(scene, camera);
    };

    animate();

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      renderer.domElement.removeEventListener('wheel', handleResize);
      if (containerRef.current && renderer.domElement.parentNode === containerRef.current) {
        containerRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  return (
    <div className="results-container">
      <div className="results-content">
        <h1 className="results-title">Analysis Complete</h1>
        
        <div className="results-layout">
          <div className="summary-section">
            <div className="summary-card">
              <div className="summary-stat">
                <span className="stat-label">Total Objects Analyzed</span>
                <span className="stat-value">{analysisData.totalObjectsAnalyzed}</span>
              </div>
            </div>
            <div className="summary-card warning">
              <div className="summary-stat">
                <span className="stat-label">Concerning Spots</span>
                <span className="stat-value">{analysisData.concerningSpots}</span>
              </div>
            </div>
          </div>

          <div className="scan-section">
            <h3>3D Scan Viewer</h3>
            <p className="scan-hint">Drag to rotate • Scroll to zoom</p>
            <div className="scan-container" ref={containerRef}></div>
          </div>
        </div>
      </div>
    </div>
  );
}
