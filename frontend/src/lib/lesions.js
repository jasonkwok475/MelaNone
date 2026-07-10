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
