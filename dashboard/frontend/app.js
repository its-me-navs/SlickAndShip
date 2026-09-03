const API_BASE = "";

let map, layerGroups = {};
let selectedMmsi = null;
let pipelineData = null;

function initMap() {
  map = L.map("map", { zoomControl: true, attributionControl: true }).setView([15.3, 73.4], 10);

  // dark chart-style basemap to match the console aesthetic.
  // Esri's Canvas basemaps are free and keyless (CARTO's raster tiles now
  // require an API key as of Aug 2026, so we're not using those).
  L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}", {
    attribution: "&copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
    maxZoom: 16,
  }).addTo(map);

  // reference layer adds place labels/borders on top of the base layer
  L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}", {
    maxZoom: 16, opacity: 0.9,
  }).addTo(map);

  layerGroups.origin = L.layerGroup().addTo(map);
  layerGroups.forecast = L.layerGroup().addTo(map);
  layerGroups.tracks = L.layerGroup().addTo(map);
  layerGroups.gaps = L.layerGroup().addTo(map);
  layerGroups.spill = L.layerGroup().addTo(map);
  layerGroups.suspectHighlight = L.layerGroup().addTo(map);
}

function fmtTime(h) {
  const days = Math.floor(h / 24);
  const hrs = Math.round(h % 24);
  return `t+${h.toFixed(1)}h`;
}

function confidenceColor(pct) {
  if (pct >= 25) return "var(--accent-high-conf)".trim();
  if (pct >= 10) return getCss("--accent-mid-conf");
  return getCss("--accent-low-conf");
}
function getCss(varName) {
  return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
}

function renderTopbarStats(data) {
  const top = data.suspects[0];
  const el = document.getElementById("topbar-stats");
  el.innerHTML = `
    <span><span class="stat-label">SLICK</span><span class="stat-value">${data.spill.lat.toFixed(3)}, ${data.spill.lon.toFixed(3)}</span></span>
    <span><span class="stat-label">EST. ORIGIN</span><span class="stat-value">${data.origin.lat.toFixed(3)}, ${data.origin.lon.toFixed(3)} &plusmn;${data.origin.spread_km}km</span></span>
    <span><span class="stat-label">TOP SUSPECT</span><span class="stat-value warn">MMSI ${top.mmsi} — ${top.confidence_pct}% confidence</span></span>
  `;
}

function renderSpill(data) {
  layerGroups.spill.clearLayers();
  const m = L.circleMarker([data.spill.lat, data.spill.lon], {
    radius: 8, color: getCss("--accent-spill"), weight: 2, fillOpacity: 0.35,
  }).addTo(layerGroups.spill);
  m.bindPopup(`<strong>Detected slick</strong><br>${data.spill.lat.toFixed(4)}, ${data.spill.lon.toFixed(4)}<br>${fmtTime(data.spill.detection_time_h)}`);
}

function renderOrigin(data) {
  layerGroups.origin.clearLayers();
  data.origin.ensemble.forEach(p => {
    L.circleMarker([p.lat, p.lon], {
      radius: 2.5, color: getCss("--accent-origin"), weight: 0, fillOpacity: 0.5,
    }).addTo(layerGroups.origin);
  });
  const centroid = L.circleMarker([data.origin.lat, data.origin.lon], {
    radius: 7, color: getCss("--accent-origin"), weight: 2, fillOpacity: 0.15,
  }).addTo(layerGroups.origin);
  centroid.bindPopup(`<strong>Estimated origin</strong><br>${data.origin.lat.toFixed(4)}, ${data.origin.lon.toFixed(4)}<br>${fmtTime(data.origin.time_h)}<br>spread: &plusmn;${data.origin.spread_km} km`);
}

function renderForecast(data) {
  layerGroups.forecast.clearLayers();
  const latlngs = data.forward_forecast.map(p => [p.lat, p.lon]);
  L.polyline(latlngs, { color: "#ffffff", weight: 1.5, opacity: 0.45, dashArray: "3,6" })
    .addTo(layerGroups.forecast)
    .bindPopup("Forward drift forecast");
}

function renderTracks(data) {
  layerGroups.tracks.clearLayers();
  Object.entries(data.tracks).forEach(([mmsi, track]) => {
    if (mmsi === data.guilty_mmsi) return; // guilty vessel drawn via suspect highlight only
    if (track.points.length < 2) return;
    const latlngs = track.points.map(p => [p.lat, p.lon]);
    const line = L.polyline(latlngs, {
      color: getCss("--accent-track"), weight: 1.5, opacity: 0.55,
    }).addTo(layerGroups.tracks);
    line.bindPopup(`<strong>MMSI ${mmsi}</strong><br>${track.vessel_type}`);
  });
}

function renderGapsForVessel(mmsi, data) {
  layerGroups.gaps.clearLayers();
  const relevantGaps = data.gaps.filter(g => g.mmsi.toString() === mmsi && g.gap_relevance >= 0.05);
  relevantGaps.forEach(g => {
    const gapColor = getCss("--accent-gap");

    // No connecting line: during the gap we genuinely don't know the
    // vessel's path, so drawing one would misleadingly imply we do.
    // Instead: a distinct "lost" marker, a "reacquired" marker, and a
    // permanent label — reads as "signal dropped here" not "traveled here".
    const lostIcon = L.divIcon({
      className: "gap-marker gap-marker-lost",
      html: '<span></span>', iconSize: [14, 14], iconAnchor: [7, 7],
    });
    const reacquiredIcon = L.divIcon({
      className: "gap-marker gap-marker-reacquired",
      html: '<span></span>', iconSize: [14, 14], iconAnchor: [7, 7],
    });

    const lostMarker = L.marker([g.lat_before, g.lon_before], { icon: lostIcon }).addTo(layerGroups.gaps);
    lostMarker.bindPopup(`<strong>MMSI ${g.mmsi} — signal lost</strong><br>Last AIS ping before gap`);

    const reacqMarker = L.marker([g.lat_after, g.lon_after], { icon: reacquiredIcon }).addTo(layerGroups.gaps);
    reacqMarker.bindPopup(`<strong>MMSI ${g.mmsi} — signal reacquired</strong><br>First AIS ping after gap`);

    // faint uncertainty indicator between the two points, deliberately
    // NOT styled as a track (very thin, low opacity, no travel implication)
    L.polyline([[g.lat_before, g.lon_before], [g.lat_after, g.lon_after]], {
      color: gapColor, weight: 1, opacity: 0.25, dashArray: "1,8",
    }).addTo(layerGroups.gaps);

    const mid = [(g.lat_before + g.lat_after) / 2, (g.lon_before + g.lon_after) / 2];
    L.marker(mid, {
      icon: L.divIcon({
        className: "gap-label",
        html: `<div>AIS GAP &middot; ${g.gap_duration_h.toFixed(1)}h dark</div>`,
        iconSize: null,
      }),
      interactive: false,
    }).addTo(layerGroups.gaps);
  });
}

function highlightSuspect(mmsi, data) {
  layerGroups.suspectHighlight.clearLayers();
  const track = data.tracks[mmsi];
  if (track && track.points.length >= 2) {
    const latlngs = track.points.map(p => [p.lat, p.lon]);
    L.polyline(latlngs, { color: getCss("--accent-suspect"), weight: 3.5, opacity: 0.9 })
      .addTo(layerGroups.suspectHighlight);
    map.fitBounds(latlngs, { padding: [60, 60], maxZoom: 12 });
  }
  renderGapsForVessel(mmsi, data);
}

function renderSuspectList(data) {
  const list = document.getElementById("suspect-list");
  document.getElementById("suspect-count").textContent = `${data.suspects.length} vessels evaluated`;
  list.innerHTML = "";

  data.suspects.forEach(s => {
    const card = document.createElement("div");
    card.className = "suspect-card" + (s.mmsi.toString() === selectedMmsi ? " selected" : "");
    card.tabIndex = 0;
    const color = confidenceColor(s.confidence_pct);
    const gapFlag = s.gap_score > 0.3;

    card.innerHTML = `
      <div class="suspect-card-top">
        <span class="suspect-rank">#${s.rank}</span>
        <span class="suspect-mmsi">MMSI ${s.mmsi}</span>
        <span class="suspect-type">${s.vessel_type}</span>
      </div>
      <div class="confidence-row">
        <div class="confidence-bar-track"><div class="confidence-bar-fill" style="width:${Math.min(100, s.confidence_pct * 2.2)}%; background:${color}"></div></div>
        <span class="confidence-pct" style="color:${color}">${s.confidence_pct}%</span>
      </div>
      <div class="score-breakdown">
        <div class="score-item"><div class="score-item-label">Proximity</div><div class="score-item-value">${s.proximity_score}</div></div>
        <div class="score-item"><div class="score-item-label">Trajectory</div><div class="score-item-value">${s.trajectory_score}</div></div>
        <div class="score-item"><div class="score-item-label">Behavior</div><div class="score-item-value">${s.behavior_score}</div></div>
        <div class="score-item"><div class="score-item-label">AIS gap</div><div class="score-item-value${gapFlag ? " flag" : ""}">${s.gap_score}</div></div>
      </div>
    `;
    card.addEventListener("click", () => selectSuspect(s.mmsi.toString(), data));
    list.appendChild(card);
  });
}

function selectSuspect(mmsi, data) {
  selectedMmsi = mmsi;
  renderSuspectList(data);
  highlightSuspect(mmsi, data);
  openDrawer(mmsi, data);
}

function openDrawer(mmsi, data) {
  const s = data.suspects.find(x => x.mmsi.toString() === mmsi);
  const track = data.tracks[mmsi];
  const drawer = document.getElementById("detail-drawer");
  const content = document.getElementById("drawer-content");
  const isGuilty = mmsi === data.guilty_mmsi;

  content.innerHTML = `
    <h2 style="font-size:15px; margin:4px 0 2px;">MMSI ${s.mmsi}</h2>
    <p style="color:var(--text-muted); font-size:12px; margin:0 0 18px;">${s.vessel_type} · rank #${s.rank} of ${data.suspects.length}</p>

    <div style="font-family:var(--font-data); font-size:28px; color:${confidenceColor(s.confidence_pct)}; margin-bottom:4px;">${s.confidence_pct}%</div>
    <p style="color:var(--text-faint); font-size:11px; margin:0 0 20px;">composite attribution confidence</p>

    <div style="border-top:1px solid var(--border); padding-top:14px;">
      <p style="font-size:12px; color:var(--text-muted); line-height:1.6; margin:0 0 14px;">
        <strong style="color:var(--text-primary)">Proximity</strong> ${s.proximity_score} — closeness of the vessel's track to the estimated origin cone.<br>
        <strong style="color:var(--text-primary)">Trajectory</strong> ${s.trajectory_score} — how well its heading near the origin time points toward the spill site.<br>
        <strong style="color:var(--text-primary)">Behavior</strong> ${s.behavior_score} — anomalous speed/course changes relative to its own pattern.<br>
        <strong style="color:var(--text-primary)">AIS gap</strong> ${s.gap_score} — whether its transponder went dark during the origin window.
      </p>
    </div>
    ${track && track.points.length ? `<p style="font-size:11px; color:var(--text-faint);">${track.points.length} AIS pings reconstructed in window</p>` : ""}
  `;
  drawer.classList.add("open");
}

function setupLayerToggles() {
  const map_ = {
    "toggle-origin": "origin", "toggle-forecast": "forecast",
    "toggle-tracks": "tracks", "toggle-gaps": "gaps",
  };
  Object.entries(map_).forEach(([id, key]) => {
    document.getElementById(id).addEventListener("change", e => {
      if (e.target.checked) map.addLayer(layerGroups[key]);
      else map.removeLayer(layerGroups[key]);
    });
  });
}

function renderAll(data) {
  pipelineData = data;
  renderTopbarStats(data);
  renderSpill(data);
  renderOrigin(data);
  renderForecast(data);
  renderTracks(data);
  renderSuspectList(data);
  if (data.suspects.length) selectSuspect(data.suspects[0].mmsi.toString(), data);
}

async function loadPipeline(refresh = false) {
  const url = refresh ? "/api/pipeline/refresh" : "/api/pipeline";
  const res = await fetch(url);
  const data = await res.json();
  renderAll(data);
}

let lastDetection = null;

async function handleFileUpload(file) {
  document.getElementById("upload-label").textContent = "Detecting…";
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/detect", { method: "POST", body: formData });
    const data = await res.json();
    document.getElementById("upload-label").textContent = "Upload SAR scene (.tif)";

    if (!data.has_detection) {
      alert("No slick detected in this scene.");
      return;
    }
    lastDetection = data;

    document.getElementById("detection-overlay").src = "data:image/png;base64," + data.overlay_png_base64;
    document.getElementById("det-method").textContent = data.method;
    document.getElementById("det-confidence").textContent = (data.confidence * 100).toFixed(0) + "%";
    document.getElementById("det-area").textContent = data.area_km2 ? `${data.area_km2} km²` : "—";
    document.getElementById("det-age").textContent = data.age_estimate;
    document.getElementById("detection-result").hidden = false;

    const traceBtn = document.getElementById("trace-btn");
    traceBtn.disabled = !data.geo_available;
    traceBtn.textContent = data.geo_available
      ? "Trace to origin & rank suspects"
      : "No georeferencing in scene";
  } catch (err) {
    document.getElementById("upload-label").textContent = "Upload SAR scene (.tif)";
    alert("Detection failed: " + err.message);
  }
}

async function traceFromDetection() {
  if (!lastDetection || !lastDetection.geo_available) return;
  const btn = document.getElementById("trace-btn");
  btn.disabled = true;
  btn.textContent = "Running pipeline…";
  const url = `/api/pipeline/from-detection?lat=${lastDetection.lat}&lon=${lastDetection.lon}`;
  const res = await fetch(url);
  const data = await res.json();
  renderAll(data);
  btn.disabled = false;
  btn.textContent = "Trace to origin & rank suspects";
}

function setupDetectionPanel() {
  const input = document.getElementById("sar-file-input");
  const drop = document.getElementById("upload-drop");
  drop.addEventListener("click", () => input.click());
  input.addEventListener("change", (e) => {
    if (e.target.files[0]) handleFileUpload(e.target.files[0]);
  });
  document.getElementById("trace-btn").addEventListener("click", traceFromDetection);
}

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  setupLayerToggles();
  setupDetectionPanel();
  loadPipeline();

  document.getElementById("refresh-btn").addEventListener("click", async (e) => {
    e.target.classList.add("loading");
    e.target.textContent = "Running…";
    await loadPipeline(true);
    e.target.classList.remove("loading");
    e.target.textContent = "New scenario";
  });

  document.getElementById("drawer-close").addEventListener("click", () => {
    document.getElementById("detail-drawer").classList.remove("open");
  });
});
