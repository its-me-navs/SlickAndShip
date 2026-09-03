# Demo scenes
Synthetic (not real satellite imagery) — for live upload during the demo.
### `scene_A_compact_recent.tif`
Compact, high-contrast, well-defined edges — reads as a recent spill.

- Detected confidence: 1.00
- Area: 20.69 km²
- Age estimate: likely recent (compact, well-defined edges)
- Detected location: 15.3210, 73.2490

### `scene_B_large_offshore.tif`
Larger slick, offshore position — good for showing area/geometry stats.

- Detected confidence: 1.00
- Area: 102.04 km²
- Age estimate: likely recent (compact, well-defined edges)
- Detected location: 15.5510, 73.5990

### `scene_C_elongated_weathered.tif`
Elongated + fragmented — reads as weathered/older in the age heuristic.

- Detected confidence: 0.88
- Area: 84.57 km²
- Age estimate: likely weathered (fragmented, diffuse — treat with caution, verify manually)
- Detected location: 15.1510, 73.1500

### `scene_D_faint_lowconf.tif`
Subtle/low-contrast — deliberately shows a lower confidence score, not every detection is a slam dunk.

- Detected confidence: 0.63
- Area: 21.54 km²
- Age estimate: moderate age (some spreading/fragmentation)
- Detected location: 15.4010, 73.7491

