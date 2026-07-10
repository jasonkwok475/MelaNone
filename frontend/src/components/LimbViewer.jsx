/**
 * LimbViewer — interactive 3D view of a completed scan's reconstructed limb.
 *
 * Loads the scan's OBJ mesh + texture (served by the backend) and places a marker at each
 * lesion's {x,y,z} surface coordinate, colored by classification (severity, not diagnosis).
 * Rotate/pan/zoom via OrbitControls. Clicking a marker selects the lesion (bidirectional with
 * the lesion table in ScanPage). Toggles: texture, wireframe, markers, auto-rotate.
 *
 * The mesh's MTL references texture.png by filename, which would not resolve over HTTP — so we
 * load geometry with OBJLoader only and apply our own material with the texture explicitly.
 */
import { Component, Suspense, useMemo, useState } from 'react'
import { Canvas, useLoader } from '@react-three/fiber'
import { OrbitControls, Html } from '@react-three/drei'
import * as THREE from 'three'
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js'
import { Loader2, Rotate3d } from 'lucide-react'
import { classificationColor, classificationMeta } from '@/lib/lesions'
import { cn } from '@/lib/utils'

function Limb({ meshUrl, textureUrl, wireframe, showTexture }) {
  const obj = useLoader(OBJLoader, meshUrl)
  const loadedTexture = useLoader(THREE.TextureLoader, textureUrl)

  const geometry = useMemo(() => {
    let found = null
    obj.traverse((child) => {
      if (child.isMesh && !found) found = child.geometry
    })
    return found
  }, [obj])

  // Clone the loaded texture so we own the copy we configure (can't mutate a hook's return).
  const texture = useMemo(() => {
    const tex = loadedTexture.clone()
    tex.colorSpace = THREE.SRGBColorSpace
    tex.needsUpdate = true
    return tex
  }, [loadedTexture])

  if (!geometry) return null
  return (
    <mesh geometry={geometry} castShadow receiveShadow>
      <meshStandardMaterial
        map={showTexture ? texture : null}
        color={showTexture ? '#ffffff' : '#d8b49b'}
        wireframe={wireframe}
        roughness={0.85}
        metalness={0}
        side={THREE.DoubleSide}
      />
    </mesh>
  )
}

function Marker({ lesion, selected, onSelect }) {
  const [hovered, setHovered] = useState(false)

  // Nudge the marker slightly outward along the surface normal so it sits proud of the mesh.
  const position = useMemo(() => {
    const normal = new THREE.Vector3(lesion.x, 0, lesion.z)
    if (normal.lengthSq() > 0) normal.normalize()
    return [lesion.x + normal.x * 0.04, lesion.y, lesion.z + normal.z * 0.04]
  }, [lesion.x, lesion.y, lesion.z])

  const color = classificationColor(lesion.classification)
  const radius = selected ? 0.12 : hovered ? 0.09 : 0.07

  return (
    <mesh
      position={position}
      onClick={(e) => {
        e.stopPropagation()
        onSelect(lesion.id)
      }}
      onPointerOver={(e) => {
        e.stopPropagation()
        setHovered(true)
        document.body.style.cursor = 'pointer'
      }}
      onPointerOut={() => {
        setHovered(false)
        document.body.style.cursor = ''
      }}
    >
      <sphereGeometry args={[radius, 16, 16]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={selected ? 0.9 : 0.35}
      />
    </mesh>
  )
}

function LoadingOverlay() {
  return (
    <Html center>
      <div className="flex items-center gap-2 whitespace-nowrap text-sm text-[var(--text-muted)]">
        <Loader2 className="h-4 w-4 animate-spin" /> Loading 3D model…
      </div>
    </Html>
  )
}

class ViewerErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { failed: false }
  }
  static getDerivedStateFromError() {
    return { failed: true }
  }
  render() {
    if (this.state.failed) {
      return (
        <div className="flex h-full flex-col items-center justify-center gap-1 p-6 text-center">
          <p className="text-sm font-medium text-concern">Could not load the 3D model</p>
          <p className="text-xs text-[var(--text-muted)]">
            The mesh or texture artifact is missing or failed to load. Try re-running the scan.
          </p>
        </div>
      )
    }
    return this.props.children
  }
}

function Toggle({ active, onClick, children }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        'rounded-md border px-2.5 py-1 text-xs font-medium transition-colors',
        active
          ? 'border-brand-500 bg-brand-50 text-brand-700 dark:bg-brand-500/15 dark:text-brand-300'
          : 'border-[var(--surface-border)] text-[var(--text-muted)] hover:bg-black/5 dark:hover:bg-white/5',
      )}
    >
      {children}
    </button>
  )
}

export default function LimbViewer({ meshUrl, textureUrl, lesions = [], selectedId, onSelectLesion }) {
  const [showTexture, setShowTexture] = useState(true)
  const [wireframe, setWireframe] = useState(false)
  const [showMarkers, setShowMarkers] = useState(true)
  const [autoRotate, setAutoRotate] = useState(true)

  return (
    <div className="relative h-[440px] w-full overflow-hidden rounded-[var(--radius-card)] border border-[var(--surface-border)] bg-slate-100 dark:bg-slate-900">
      <div className="pointer-events-none absolute inset-x-0 top-0 z-10 flex flex-wrap gap-1.5 p-3">
        <div className="pointer-events-auto flex flex-wrap gap-1.5">
          <Toggle active={showTexture} onClick={() => setShowTexture((v) => !v)}>
            Texture
          </Toggle>
          <Toggle active={wireframe} onClick={() => setWireframe((v) => !v)}>
            Wireframe
          </Toggle>
          <Toggle active={showMarkers} onClick={() => setShowMarkers((v) => !v)}>
            Markers
          </Toggle>
          <Toggle active={autoRotate} onClick={() => setAutoRotate((v) => !v)}>
            <span className="inline-flex items-center gap-1">
              <Rotate3d className="h-3.5 w-3.5" /> Auto-rotate
            </span>
          </Toggle>
        </div>
      </div>

      <ViewerErrorBoundary>
        <Canvas camera={{ position: [0, 0, 5], fov: 45 }} dpr={[1, 2]}>
          <ambientLight intensity={0.7} />
          <directionalLight position={[5, 5, 5]} intensity={1.1} />
          <directionalLight position={[-5, -2, -5]} intensity={0.4} />
          <Suspense fallback={<LoadingOverlay />}>
            <Limb
              meshUrl={meshUrl}
              textureUrl={textureUrl}
              wireframe={wireframe}
              showTexture={showTexture}
            />
            {showMarkers &&
              lesions.map((le) => (
                <Marker
                  key={le.id}
                  lesion={le}
                  selected={le.id === selectedId}
                  onSelect={onSelectLesion}
                />
              ))}
          </Suspense>
          <OrbitControls
            makeDefault
            enablePan
            autoRotate={autoRotate}
            autoRotateSpeed={1.2}
            minDistance={2}
            maxDistance={12}
          />
        </Canvas>
      </ViewerErrorBoundary>

      {showMarkers && lesions.length > 0 && (
        <div className="pointer-events-none absolute bottom-0 left-0 z-10 flex flex-wrap gap-x-3 gap-y-1 p-3 text-xs text-[var(--text-muted)]">
          {['melanoma', 'keratosis', 'benign'].map((c) => (
            <span key={c} className="inline-flex items-center gap-1">
              <span
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: classificationColor(c) }}
              />
              {classificationMeta(c).label}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
