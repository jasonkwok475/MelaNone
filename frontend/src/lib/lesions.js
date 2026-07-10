/** Presentation helpers for lesion classifications (shared across scan/history views). */

// Maps a classification to a Badge variant + human label. Severity, not diagnosis.
const CLASS_META = {
  melanoma: { variant: 'concern', label: 'Melanoma' },
  keratosis: { variant: 'watch', label: 'Keratosis' },
  nevus: { variant: 'watch', label: 'Nevus' },
  benign: { variant: 'benign', label: 'Benign' },
  unknown: { variant: 'neutral', label: 'Unknown' },
}

export function classificationMeta(classification) {
  return CLASS_META[classification] ?? { variant: 'neutral', label: classification }
}

// Literal hex colors for WebGL markers (design-token CSS vars are DOM-only, not usable in
// three.js materials). Kept in sync with the severity palette. Severity, not diagnosis.
const CLASS_COLOR = {
  melanoma: '#dc2626', // concern / red
  keratosis: '#d97706', // watch / amber
  nevus: '#d97706',
  benign: '#16a34a', // green
  unknown: '#64748b',
}

export function classificationColor(classification) {
  return CLASS_COLOR[classification] ?? CLASS_COLOR.unknown
}

export function confidencePct(confidence) {
  return `${Math.round((confidence ?? 0) * 100)}%`
}

// Common limb body sites for the scan dialog. Same site across scans enables comparison.
export const BODY_SITES = [
  'left_forearm',
  'right_forearm',
  'left_upper_arm',
  'right_upper_arm',
  'left_hand',
  'right_hand',
  'left_thigh',
  'right_thigh',
  'left_shin',
  'right_shin',
  'left_foot',
  'right_foot',
]

export function bodySiteLabel(site) {
  return site.replaceAll('_', ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}
