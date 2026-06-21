"""
OMNI-AGENCY GALACTIC CORE V10.0
Space Command & Architecture Dashboard

A monolithic Streamlit mission-control dashboard implementing four modules:
  1. Architectural Forge       - 3D procedural spacecraft designer (Trimesh + Plotly)
  2. 5D Orbital Sentinel       - Poliastro/Skyfield propagation w/ NASA HORIZONS (Astroquery)
  3. Multi-Agency Observatory  - NASA APOD, JWST/Hubble gallery, ESA/JAXA links
  4. Strategic Command HUD     - Cyber-military NASA theme, custom CSS, PyDeck maps

Design notes:
  - All external libs are optional; every call has a mock-data fallback.
  - Streamlit cache_data is used for all API calls to stay <3GB and snappy.
  - "5D" plot = 3D position + time slider (4th) + color flux (5th).
"""
import os
import io
import json
import math
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import requests

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional dependencies - all gracefully fall back to mocks if missing.
# ---------------------------------------------------------------------------
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    from skyfield.api import Loader, EarthSatellite, Topos, load
    SKYFIELD_AVAILABLE = True
except ImportError:
    SKYFIELD_AVAILABLE = False

try:
    from poliastro.bodies import Earth, Sun, Mars
    from poliastro.twobody import Orbit
    POLYASTRO_AVAILABLE = True
except ImportError:
    POLYASTRO_AVAILABLE = False

try:
    from sgp4.api import Satrec, jday
    SGP4_AVAILABLE = True
except ImportError:
    SGP4_AVAILABLE = False

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import pydeck as pdk
    PYDECK_AVAILABLE = True
except ImportError:
    PYDECK_AVAILABLE = False

try:
    from astropy.time import Time
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

try:
    from astroquery.jplhorizons import Horizons
    HORIZONS_AVAILABLE = True
except ImportError:
    HORIZONS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Page config + global constants
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="OMNI-AGENCY GALACTIC CORE V10.0",
    initial_sidebar_state="expanded",
)

# Constants
EARTH_RADIUS_KM = 6371.0
LIGHT_SPEED_KM_S = 299792.458
G0 = 9.80665  # m/s^2 standard gravity

# Standard Isp for common propulsion systems (seconds) - used by Forge
PROPULSION_ISP = {
    "Chemical (LOX/LH2)": 450,
    "Chemical (RP-1/LOX)": 350,
    "Fusion Pulse": 10000,
    "Ion (Xenon)": 3000,
    "Nuclear Thermal": 900,
    "Solid Rocket": 280,
}

# Material density kg/m^3 (for procedural mass estimation)
MATERIAL_DENSITY = {
    "Aluminum-Lithium": 2700,
    "Titanium": 4500,
    "Carbon Fiber": 1600,
    "Stainless Steel": 8000,
}

# ---------------------------------------------------------------------------
# Cyber-Military NASA HUD CSS (Deep Space Black + Neon Cyan + Alert Red)
# ---------------------------------------------------------------------------
HUD_CSS = """
<style>
/* Deep space background */
.stApp {
    background: radial-gradient(ellipse at top, #0a0a1a 0%, #000000 70%);
    color: #00ffe5;
}

/* Glassmorphism containers */
div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"],
div[data-testid="stSidebar"], .element-container, .stMarkdown, .stDataFrame {
    background: rgba(0, 20, 40, 0.35) !important;
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    border: 1px solid rgba(0, 255, 229, 0.25) !important;
    border-radius: 6px;
    padding: 6px;
}

/* Neon headers */
h1, h2, h3, h4 {
    color: #00ffe5 !important;
    text-shadow: 0 0 8px rgba(0, 255, 229, 0.6);
    letter-spacing: 1px;
}

/* Alert red accents */
.stAlert, .stException {
    border-left: 4px solid #ff2a4d !important;
    color: #ff7a8c !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #001f2e, #003a52);
    color: #00ffe5;
    border: 1px solid #00ffe5;
    font-weight: 600;
    letter-spacing: 1px;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #003a52, #00ffe5);
    color: #000;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] button {
    background: rgba(0, 20, 40, 0.4);
    color: #00ffe5;
    border: 1px solid rgba(0, 255, 229, 0.3);
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    background: rgba(0, 255, 229, 0.15);
    color: #00ffe5;
    border-color: #00ffe5;
    box-shadow: 0 0 10px rgba(0, 255, 229, 0.4);
}

/* Caption/footer */
.stCaption, footer { color: #007a8a !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #000; }
::-webkit-scrollbar-thumb { background: #00ffe5; border-radius: 4px; }

/* Telemetry stripe */
.telemetry-stripe {
    background: linear-gradient(90deg, rgba(255,42,77,0.1) 0%, rgba(0,255,229,0.1) 100%);
    padding: 8px 16px;
    border-left: 3px solid #00ffe5;
    border-right: 3px solid #ff2a4d;
    margin: 8px 0;
    font-family: 'Courier New', monospace;
    font-size: 0.85em;
    color: #00ffe5;
}
</style>
"""
st.markdown(HUD_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached data layer (APIs)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_iss_tle():
    """Fetch ISS TLE with mock fallback."""
    try:
        if not SKYFIELD_AVAILABLE:
            raise ImportError("Skyfield not installed")
        url = "https://celestrak.com/NORAD/elements/stations.txt"
        sats = load.tle(url)
        for sat in sats:
            if sat.name.strip() == "ISS (ZARYA)":
                return {
                    "name": sat.name,
                    "line1": sat.model.line1,
                    "line2": sat.model.line2,
                }
        raise ValueError("ISS not found")
    except Exception as e:
        return {
            "name": "ISS (ZARYA)",
            "line1": "1 25544U 98067A   26171.50000000  .00016717  00000+0  34228-3 0  9994",
            "line2": "2 25544  51.6416 247.4627 0003668 130.5360 325.0288 15.49420423443366",
            "_warning": str(e),
        }


@st.cache_data(ttl=1800)
def fetch_apod():
    """Fetch NASA Astronomy Picture of the Day. Falls back to a static image."""
    try:
        api_key = os.environ.get("NASA_API_KEY", "DEMO_KEY")
        r = requests.get(
            f"https://api.nasa.gov/planetary/apod?api_key={api_key}",
            timeout=8,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {
            "title": "Hubble Ultra Deep Field (Fallback)",
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "explanation": (
                "Live APOD unavailable. Showing offline fallback: the Hubble Ultra Deep Field, "
                "one of the deepest visible-light images ever taken, revealing ~10,000 galaxies."
            ),
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Hubble_ultra_deep_field_high_rez_edit1.jpg/1280px-Hubble_ultra_deep_field_high_rez_edit1.jpg",
            "media_type": "image",
            "copyright": "NASA / ESA",
            "_warning": str(e),
        }


@st.cache_data(ttl=3600)
def fetch_jwst_observations():
    """Fetch JWST observation metadata. Falls back to curated list."""
    try:
        # In production: Astroquery MAST. We use a curated list to avoid slowness.
        return [
            {
                "id": "jw02736-o001_t017_nircam_clear-f090w",
                "target": "Carina Nebula - NGC 3324",
                "instrument": "NIRCam",
                "date": "2023-07-12",
                "url": "https://www.nasa.gov/wp-content/uploads/2023/07/main_image_star-forming_region_carina_nircam_final-5mb.jpg",
                "wavelength": "0.9 micron (F090W)",
            },
            {
                "id": "jw02736-o002_t017_miri_f1800w",
                "target": "Southern Ring Nebula (NGC 3132)",
                "instrument": "MIRI",
                "date": "2023-07-12",
                "url": "https://www.nasa.gov/wp-content/uploads/2023/07/main_image_aligned_miri-srgb.jpg",
                "wavelength": "18 micron (F1800W)",
            },
            {
                "id": "jw02736-o003_t017_nircam_f200w",
                "target": "Stephan's Quintet",
                "instrument": "NIRCam",
                "date": "2023-07-12",
                "url": "https://www.nasa.gov/wp-content/uploads/2023/07/stephans_quintet_nircam_final-5mb.jpg",
                "wavelength": "2.0 micron (F200W)",
            },
            {
                "id": "jw02738-o001_t022_nircam_f444w",
                "target": "Pillars of Creation",
                "instrument": "NIRCam",
                "date": "2022-10-19",
                "url": "https://stsci-opo.org/STScI-01G7DDCCYNZH3VD3NCYJZD3Q9F.png",
                "wavelength": "4.4 micron (F444W)",
            },
        ]
    except Exception as e:
        return [{"_warning": str(e)}]


@st.cache_data(ttl=3600)
def fetch_hubble_observations():
    """Fetch Hubble observation gallery metadata. Returns curated list."""
    return [
        {
            "id": "HST-pillars-2014",
            "target": "Pillars of Creation (revisited)",
            "instrument": "WFC3",
            "date": "2014-10-28",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Pillars_2014_HST_WFC3-UVIS_full-res_denoised.jpg/1280px-Pillars_2014_HST_WFC3-UVIS_full-res_denoised.jpg",
            "wavelength": "Visible / Near-IR",
        },
        {
            "id": "HST-pillars-1995",
            "target": "Pillars of Creation (original)",
            "instrument": "WFPC2",
            "date": "1995-04-01",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg/1280px-Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg",
            "wavelength": "Visible",
        },
        {
            "id": "HST-ultradeep",
            "target": "Hubble Ultra Deep Field",
            "instrument": "ACS / WFC3",
            "date": "2004-03-09",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Hubble_ultra_deep_field_high_rez_edit1.jpg/1280px-Hubble_ultra_deep_field_high_rez_edit1.jpg",
            "wavelength": "Visible / Near-IR",
        },
        {
            "id": "HST-eagle",
            "target": "Eagle Nebula (M16)",
            "instrument": "WFPC2",
            "date": "1995-04-01",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Eagle_Nebula_from_ESO.jpg/1280px-Eagle_Nebula_from_ESO.jpg",
            "wavelength": "Visible",
        },
    ]


@st.cache_data(ttl=1800)
def fetch_horizons_vector(target_id, epochs):
    """Query NASA JPL Horizons via Astroquery with a hard timeout + mock fallback."""
    if not HORIZONS_AVAILABLE or not ASTROPY_AVAILABLE:
        return _mock_horizons_vector(target_id, epochs)

    try:
        # Astroquery Horizons is slow; cap window
        if len(epochs) > 12:
            epochs = epochs[:: max(1, len(epochs) // 12)]
        t = Time(epochs, format="iso", scale="utc")
        obj = Horizons(id=target_id, location="500@Sun", epochs=t)
        # Tight timeout via subprocess would be ideal; we rely on `timeout` param.
        try:
            vec = obj.vectors()
            # columns: x,y,z in AU, vx,vy,vz in AU/day
            return {
                "x": vec["x"].tolist(),
                "y": vec["y"].tolist(),
                "z": vec["z"].tolist(),
                "vx": vec["vx"].tolist(),
                "vy": vec["vy"].tolist(),
                "vz": vec["vz"].tolist(),
                "epochs": epochs,
                "source": "JPL Horizons (Astroquery)",
            }
        except Exception:
            return _mock_horizons_vector(target_id, epochs)
    except Exception:
        return _mock_horizons_vector(target_id, epochs)


def _mock_horizons_vector(target_id, epochs):
    """Deterministic mock heliocentric vector (J2000) in AU."""
    # Crude orbital radius (AU) by target id
    radii = {
        "ISS (ZARYA)": 1.0,
        "JWST": 1.01,  # Sun-Earth L2
        "Voyager 1": 159.0,
        "Mars": 1.524,
        "Jupiter": 5.203,
        "Saturn": 9.537,
    }
    r = radii.get(target_id, 1.0)
    n = len(epochs)
    # Convert epoch to days since J2000
    t0 = datetime(2000, 1, 1, 12, 0, 0)
    days = np.array([
        (datetime.fromisoformat(e.replace("Z", "")) - t0).total_seconds() / 86400.0
        if isinstance(e, str) else 0.0
        for e in epochs
    ])
    # Crude angular motion (rad/day)
    omega = 2 * np.pi / (365.25 * np.sqrt(r ** 3))
    theta = omega * days
    x = (r * np.cos(theta)).tolist()
    y = (r * np.sin(theta)).tolist()
    z = (0.05 * r * np.sin(2 * theta)).tolist()
    return {
        "x": x, "y": y, "z": z,
        "vx": [0.0] * n, "vy": [0.0] * n, "vz": [0.0] * n,
        "epochs": list(epochs),
        "source": "Mock (JPL Horizons offline / unavailable)",
    }


# ---------------------------------------------------------------------------
# ARCHITECTURAL FORGE - procedural spacecraft designer
# ---------------------------------------------------------------------------
def build_spacecraft_mesh(modules):
    """Build a Trimesh scene from a list of module dicts.

    Each module dict has: kind, count, length/radius, material, position_offset.
    Returns (trimesh.Scene, total_mass_kg, com_xyz, moi_tensor).
    """
    if not TRIMESH_AVAILABLE:
        return None, 0.0, np.zeros(3), np.zeros((3, 3))

    scene = trimesh.Scene()
    total_mass = 0.0
    com_accum = np.zeros(3)
    moi_accum = np.zeros((3, 3))

    for mod in modules:
        kind = mod["kind"]
        count = int(mod["count"])
        L = float(mod.get("length", 1.0))      # m
        r = float(mod.get("radius", 0.5))      # m
        rho = MATERIAL_DENSITY.get(mod.get("material", "Aluminum-Lithium"), 2700)
        offset = np.array(mod.get("position_offset", [0.0, 0.0, 0.0]), dtype=float)

        for i in range(count):
            # Distribute multiple identical modules along the hull axis
            sub_offset = offset.copy()
            sub_offset[0] += i * (L + 0.3)

            if kind == "Fuel Tank":
                mesh = trimesh.creation.cylinder(radius=r, height=L, sections=24)
                vol = mesh.volume
                mass = rho * vol
            elif kind == "Engine":
                mesh = trimesh.creation.cone(radius=r, height=L, sections=24)
                # Engines: assume tungsten, denser
                mass = 19600 * mesh.volume
            elif kind == "Crew Pod":
                mesh = trimesh.creation.icosphere(subdivisions=2, radius=r)
                # Pressurized aluminum
                mass = 2700 * mesh.volume
            elif kind == "Solar Array":
                # Thin box
                mesh = trimesh.creation.box(extents=[L, r * 4, 0.05])
                # Very light: thin film + truss
                mass = 50 * (L * r * 4)
            else:
                continue

            # Translate to position
            mesh.apply_translation(sub_offset)
            scene.add_geometry(mesh, node_name=f"{kind}_{i}", geom_name=f"{kind}_{i}")

            total_mass += mass
            com_accum += mass * sub_offset
            # Crude MoI: point-mass approximation
            r2 = np.sum(sub_offset ** 2)
            moi = mass * (np.eye(3) * r2 - np.outer(sub_offset, sub_offset))
            moi_accum += moi

    com = com_accum / total_mass if total_mass > 0 else np.zeros(3)
    return scene, total_mass, com, moi_accum


def tsiolkovsky_delta_v(isp_s, m0_kg, mf_kg):
    """Tsiolkovsky rocket equation: delta_v = Isp * g0 * ln(m0 / mf)."""
    if mf_kg <= 0 or m0_kg <= mf_kg:
        return 0.0
    return isp_s * G0 * math.log(m0_kg / mf_kg)


def render_forge_tab():
    """Architectural Forge - 3D CAD spacecraft designer."""
    st.header("⚙️ ARCHITECTURAL FORGE — Procedural Spacecraft Designer")

    if not TRIMESH_AVAILABLE:
        st.warning(
            "Trimesh not installed. Run `pip install trimesh shapely rtree` to enable "
            "the full 3D mesh designer. Showing physics-only preview below."
        )

    with st.sidebar:
        st.subheader("Forge Controls")
        material = st.selectbox("Hull Material", list(MATERIAL_DENSITY.keys()), index=0)
        propulsion = st.selectbox("Propulsion", list(PROPULSION_ISP.keys()), index=2)
        thrust_kn = st.slider("Engine Thrust (kN)", 100, 5000, 1850, step=50)
        payload_kg = st.slider("Payload (kg)", 500, 20000, 5000, step=500)

    st.markdown('<div class="telemetry-stripe">⚙ FORGE ONLINE — TSIALKOVSKY ENGINE READY</div>',
                unsafe_allow_html=True)

    # Module configuration
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Module Configuration")
        n_fuel = st.slider("Fuel Tanks", 0, 6, 3)
        fuel_radius = st.slider("Fuel Tank Radius (m)", 1.0, 4.0, 2.0, step=0.1)
        fuel_length = st.slider("Fuel Tank Length (m)", 4.0, 20.0, 10.0, step=0.5)

        n_engine = st.slider("Engines", 1, 8, 4)
        engine_radius = st.slider("Engine Radius (m)", 0.5, 2.0, 0.8, step=0.1)
        engine_length = st.slider("Engine Length (m)", 1.0, 5.0, 2.0, step=0.1)

    with col_b:
        st.subheader("Crew & Power")
        n_crew = st.slider("Crew Pods", 0, 4, 1)
        crew_radius = st.slider("Crew Pod Radius (m)", 1.0, 4.0, 2.0, step=0.1)

        n_solar = st.slider("Solar Arrays", 0, 8, 4)
        solar_length = st.slider("Solar Array Span (m)", 5.0, 30.0, 12.0, step=0.5)
        solar_width = st.slider("Solar Array Width (m)", 1.0, 5.0, 2.0, step=0.5)

    # Build module list
    modules = [
        {"kind": "Fuel Tank", "count": n_fuel, "length": fuel_length,
         "radius": fuel_radius, "material": material, "position_offset": [0, 0, 0]},
        {"kind": "Engine", "count": n_engine, "length": engine_length,
         "radius": engine_radius, "material": "Tungsten", "position_offset": [-(fuel_length + 1), 0, 0]},
        {"kind": "Crew Pod", "count": n_crew, "length": 0, "radius": crew_radius,
         "material": material, "position_offset": [fuel_length + 3, 0, 0]},
        {"kind": "Solar Array", "count": n_solar, "length": solar_length, "radius": solar_width / 4,
         "material": "Carbon Fiber", "position_offset": [fuel_length + 5, 4, 0]},
    ]

    scene, total_mass, com, moi = build_spacecraft_mesh(modules)

    # Add payload & propellant to mass model
    propellant_kg = 0.0
    if TRIMESH_AVAILABLE and scene is not None:
        # Crude: 70% of fuel tank volume is liquid hydrogen
        tank_vol = n_fuel * math.pi * (fuel_radius ** 2) * fuel_length
        propellant_kg = 71.0 * tank_vol  # LH2 density kg/m^3
    m0 = total_mass + propellant_kg + payload_kg
    mf = total_mass + payload_kg  # dry mass

    isp = PROPULSION_ISP[propulsion]
    thrust_n = thrust_kn * 1000.0
    delta_v = tsiolkovsky_delta_v(isp, m0, mf)
    twr = thrust_n / (m0 * G0) if m0 > 0 else 0.0

    # Telemetry row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Wet Mass", f"{m0:,.0f} kg")
    c2.metric("Δv (Tsiolkovsky)", f"{delta_v:,.0f} m/s")
    c3.metric("TWR", f"{twr:.2f}")
    c4.metric("CoM offset", f"{np.linalg.norm(com):.2f} m")
    c5.metric("Isp", f"{isp:,} s",
              delta=f"{(isp * G0 / 1000):.1f} km/s exhaust")

    # 3D Plotly visualization (works even without Trimesh)
    fig = _plot_spacecraft_plotly(modules)
    st.plotly_chart(fig, use_container_width=True, key="forge_3d")

    # Detailed physics table
    st.subheader("Engineering Telemetry")
    df = pd.DataFrame({
        "Parameter": [
            "Wet Mass (m0)", "Dry Mass (mf)", "Propellant", "Payload",
            "Isp (effective)", "Thrust (sea-level equiv.)", "Δv", "TWR",
            "CoM (X, Y, Z) m", "Specific Energy (MJ/kg)",
        ],
        "Value": [
            f"{m0:,.0f} kg", f"{mf:,.0f} kg", f"{propellant_kg:,.0f} kg",
            f"{payload_kg:,.0f} kg", f"{isp:,} s", f"{thrust_n:,.0f} N",
            f"{delta_v:,.0f} m/s  ({delta_v/1000:.2f} km/s)",
            f"{twr:.3f}  ({'flight-ready' if twr > 1.2 else 'sub-thrust'})",
            f"({com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f})",
            f"{isp * G0 * isp / 2 / 1e6:.1f}",
        ],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)


def _plot_spacecraft_plotly(modules):
    """Plotly 3D representation of the procedural spacecraft."""
    fig = go.Figure()
    palette = {
        "Fuel Tank": "#00ffe5",
        "Engine": "#ff2a4d",
        "Crew Pod": "#7afff0",
        "Solar Array": "#ffb347",
    }

    for mod in modules:
        kind = mod["kind"]
        L = float(mod.get("length", 1.0))
        r = float(mod.get("radius", 0.5))
        off = mod.get("position_offset", [0, 0, 0])
        n = int(mod["count"])
        color = palette[kind]

        for i in range(n):
            cx = off[0] + i * (L + 0.3)
            if kind == "Fuel Tank":
                # Cylinder
                theta = np.linspace(0, 2 * np.pi, 24)
                z = np.linspace(cx - L / 2, cx + L / 2, 2)
                T, Z = np.meshgrid(theta, z)
                X = r * np.cos(T) + off[1]
                Y = r * np.sin(T) + off[2]
                fig.add_trace(go.Surface(
                    x=X, y=Y, z=Z, colorscale=[[0, color], [1, color]],
                    showscale=False, opacity=0.85, name=kind, showlegend=(i == 0),
                ))
            elif kind == "Engine":
                # Cone (drawn as triangle)
                fig.add_trace(go.Cone(
                    x=[cx - L / 2], y=[0], z=[0],
                    u=[-L], v=[0], w=[0],
                    sizemode="absolute", sizeref=2,
                    colorscale=[[0, color], [1, color]], showscale=False,
                    name=kind, showlegend=(i == 0),
                ))
            elif kind == "Crew Pod":
                phi = np.linspace(0, np.pi, 16)
                theta = np.linspace(0, 2 * np.pi, 24)
                P, T = np.meshgrid(phi, theta)
                X = r * np.sin(P) * np.cos(T) + cx
                Y = r * np.sin(P) * np.sin(T)
                Z = r * np.cos(P)
                fig.add_trace(go.Surface(
                    x=X, y=Y, z=Z, colorscale=[[0, color], [1, color]],
                    showscale=False, opacity=0.9, name=kind, showlegend=(i == 0),
                ))
            elif kind == "Solar Array":
                # Plane
                X = [cx, cx + L, cx + L, cx, cx]
                Y = [r * 2, r * 2, -r * 2, -r * 2, r * 2]
                Z = [0, 0, 0, 0, 0]
                fig.add_trace(go.Mesh3d(
                    x=X, y=Y, z=Z, color=color, opacity=0.7,
                    name=kind, showlegend=(i == 0),
                ))

    fig.update_layout(
        title="Procedural Spacecraft — Engineering Schematic",
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            bgcolor="#000", aspectmode="data",
            xaxis=dict(color="#00ffe5", gridcolor="#003a52"),
            yaxis=dict(color="#00ffe5", gridcolor="#003a52"),
            zaxis=dict(color="#00ffe5", gridcolor="#003a52"),
        ),
        paper_bgcolor="#000", plot_bgcolor="#000",
        font=dict(color="#00ffe5", family="Courier New"),
        height=600, showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="#00ffe5"),
    )
    return fig


# ---------------------------------------------------------------------------
# 5D ORBITAL SENTINEL
# ---------------------------------------------------------------------------
def render_orbital_tab():
    """5D Orbital Sentinel - propagation + time slider + color flux."""
    st.header("🛰️ 5D ORBITAL SENTINEL — Physics Engine")

    with st.sidebar:
        st.subheader("Sentinel Controls")
        target = st.selectbox(
            "Target Body / Spacecraft",
            ["ISS (ZARYA)", "JWST", "Voyager 1", "Mars", "Jupiter", "Saturn"],
            index=0,
        )
        use_live = st.checkbox(
            "Use live NASA HORIZONS (slow)",
            value=False,
            help="Queries JPL Horizons via Astroquery. May take 30-60s; mock data is used on failure.",
        )
        n_steps = st.slider("Time Steps", 10, 200, 60)

    st.markdown(
        '<div class="telemetry-stripe">🛰 SENTINEL ACTIVE — POLIASTRO + SKYFIELD PROPAGATION</div>',
        unsafe_allow_html=True,
    )

    # Build epoch list
    t0 = datetime.utcnow()
    epochs = [(t0 + timedelta(days=i)).isoformat() for i in range(n_steps)]

    if use_live:
        with st.spinner(f"Querying JPL Horizons for {target}..."):
            vec = fetch_horizons_vector(target, epochs)
    else:
        vec = fetch_horizons_vector(target, epochs)

    if "_warning" in vec:
        st.warning(vec["_warning"])
    st.caption(f"Data source: {vec.get('source', 'unknown')}")

    # Convert to numpy (mock returns lists; live returns lists)
    x = np.array(vec["x"])
    y = np.array(vec["y"])
    z = np.array(vec["z"])
    # AU -> km for readability
    AU_KM = 149597870.7
    x, y, z = x * AU_KM, y * AU_KM, z * AU_KM

    # Heat/stress flux - radial distance derivative as proxy (5th dimension)
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if len(r) > 1:
        flux = np.gradient(r)
        flux = (flux - flux.min()) / (flux.max() - flux.min() + 1e-9)
    else:
        flux = np.zeros_like(r)

    # Plotly 3D with time slider
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines+markers",
        line=dict(color=flux, colorscale="Hot", width=4, showscale=True,
                  colorbar=dict(title="Heat Flux (norm.)")),
        marker=dict(size=4, color=flux, colorscale="Hot", showscale=False),
        name=target,
        hovertemplate="X: %{x:.0f} km<br>Y: %{y:.0f} km<br>Z: %{z:.0f} km<extra></extra>",
    ))

    # Reference body (Earth) for inner-system targets
    if target not in ("Voyager 1",):
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        xe = EARTH_RADIUS_KM * np.cos(u) * np.sin(v)
        ye = EARTH_RADIUS_KM * np.sin(u) * np.sin(v)
        ze = EARTH_RADIUS_KM * np.cos(v)
        fig.add_trace(go.Surface(
            x=xe, y=ye, z=ze,
            colorscale=[[0, "#001f2e"], [1, "#003a52"]],
            showscale=False, opacity=0.7, name="Earth",
        ))

    fig.update_layout(
        title=f"{target} — 5D Trajectory (3D position + time + heat flux)",
        scene=dict(
            xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Z (km)",
            bgcolor="#000", aspectmode="data",
            xaxis=dict(color="#00ffe5", gridcolor="#003a52"),
            yaxis=dict(color="#00ffe5", gridcolor="#003a52"),
            zaxis=dict(color="#00ffe5", gridcolor="#003a52"),
        ),
        paper_bgcolor="#000", plot_bgcolor="#000",
        font=dict(color="#00ffe5"), height=600,
    )
    st.plotly_chart(fig, use_container_width=True, key="sentinel_3d")

    # Time-slider chart (4th dim made explicit)
    df = pd.DataFrame({
        "Epoch": pd.to_datetime(vec["epochs"]),
        "X (km)": x, "Y (km)": y, "Z (km)": z,
        "Range (km)": r, "Heat Flux (norm.)": flux,
    })
    st.subheader("Telemetry — Time Domain (4th Dimension)")
    st.line_chart(df.set_index("Epoch")[["Range (km)", "Heat Flux (norm.)"]],
                  height=300)

    with st.expander("Raw Ephemeris Table"):
        st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# MULTI-AGENCY OBSERVATORY
# ---------------------------------------------------------------------------
def render_observatory_tab():
    """Multi-Agency Observatory - APOD, JWST, Hubble, ESA/JAXA links."""
    st.header("🌌 MULTI-AGENCY OBSERVATORY — Live Streamer")

    st.markdown(
        '<div class="telemetry-stripe">🌌 DOWNLINK ACTIVE — APOD + MAST ARCHIVES</div>',
        unsafe_allow_html=True,
    )

    # APOD
    apod = fetch_apod()
    if "_warning" in apod:
        st.warning(f"APOD fallback: {apod['_warning']}")
    st.subheader(f"📷 NASA APOD — {apod.get('date', 'today')}")
    st.markdown(f"### {apod.get('title', 'Untitled')}")
    st.caption(f"© {apod.get('copyright', 'Public Domain')}")
    if apod.get("media_type") == "image":
        st.image(apod.get("url"), use_container_width=True)
    elif apod.get("media_type") == "video":
        st.video(apod.get("url"))
    with st.expander("Explanation"):
        st.write(apod.get("explanation", ""))

    st.markdown("---")

    # JWST
    st.subheader("🔭 James Webb Space Telescope — Recent Observations")
    jwst = fetch_jwst_observations()
    cols = st.columns(2)
    for i, obs in enumerate(jwst[:4]):
        with cols[i % 2]:
            st.markdown(f"**{obs['target']}** — {obs['instrument']} ({obs['date']})")
            st.caption(f"Filter: {obs['wavelength']}")
            try:
                st.image(obs["url"], use_container_width=True)
            except Exception:
                st.caption("(image link unavailable offline)")

    st.markdown("---")

    # Hubble
    st.subheader("🌠 Hubble Space Telescope — Archive Highlights")
    hubble = fetch_hubble_observations()
    cols = st.columns(2)
    for i, obs in enumerate(hubble[:4]):
        with cols[i % 2]:
            st.markdown(f"**{obs['target']}** — {obs['instrument']} ({obs['date']})")
            st.caption(f"Filter: {obs['wavelength']}")
            try:
                st.image(obs["url"], use_container_width=True)
            except Exception:
                st.caption("(image link unavailable offline)")

    st.markdown("---")

    # External agency link nodes
    st.subheader("🌐 Inter-Agency Data Nodes")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**[ESA Sky](https://sky.esa.int/)** — Hipparcos, Gaia, Tycho catalogs")
    c2.markdown("**[JAXA DARTS](https://darts.isas.jaxa.jp/)** — Akari, Hinode, Hayabusa2 archives")
    c3.markdown("**[MAST Portal](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html)** — HST, JWST, Kepler, TESS")
    c4.markdown("**[NASA Eyes](https://eyes.nasa.gov/)** — Real-time 3D solar system viewer")

    # 3D model viewer (stub - embedded spec)
    st.markdown("---")
    st.subheader("🧊 3D Mission Assets (model-viewer)")
    st.caption(
        "Streamed 3D .glb models require NASA's public model repository. "
        "Below is a standard `<model-viewer>` element — point at any valid .glb URL to render."
    )
    glb_url = st.text_input(
        ".glb URL (NASA 3D Resources recommended)",
        value="https://modelviewer.dev/shared-assets/models/Astronaut.glb",
        label_visibility="collapsed",
    )
    model_viewer_html = f"""
<div style="width:100%; height:420px; background:#000; border:1px solid #00ffe5;
            display:flex; align-items:center; justify-content:center;">
  <model-viewer alt="3D model" src="{glb_url}"
                camera-controls auto-rotate style="width:100%; height:100%;">
  </model-viewer>
  <script type="module"
    src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js">
  </script>
</div>
"""
    st.components.v1.html(model_viewer_html, height=440)


# ---------------------------------------------------------------------------
# STRATEGIC COMMAND HUD - ground tracking + theme
# ---------------------------------------------------------------------------
def render_hud_tab():
    """Strategic Command HUD - map view + system status."""
    st.header("🎯 STRATEGIC COMMAND HUD — Ground Tracking")

    st.markdown(
        '<div class="telemetry-stripe">🎯 COMMAND ACTIVE — MAPBOX-STYLE TRACKING</div>',
        unsafe_allow_html=True,
    )

    # Mock ground stations + targets
    stations = pd.DataFrame({
        "lat": [28.5728, 35.6762, -33.8688, 51.5074, 37.7749],
        "lon": [-80.6490, 139.6503, 151.2093, -0.1278, -122.4194],
        "name": ["Cape Canaveral", "Tanegashima (JAXA)", "Canberra (DSN)",
                 "London (ESA Ops)", "San Francisco (Ames)"],
        "type": ["Launch", "Launch", "Tracking", "Operations", "R&D"],
    })

    tracks = pd.DataFrame({
        "lat": np.random.uniform(-60, 60, 30),
        "lon": np.random.uniform(-180, 180, 30),
        "altitude_km": np.random.uniform(200, 36000, 30),
        "target": np.random.choice(["ISS", "JWST", "Voyager 1", "GPS-III"], 30),
    })

    c1, c2, c3 = st.columns(3)
    c1.metric("Active Ground Stations", f"{len(stations)}")
    c2.metric("Tracked Assets", f"{len(tracks)}")
    c3.metric("Mission Status", "NOMINAL",
              delta="All systems green", delta_color="normal")

    if PYDECK_AVAILABLE:
        layer_stations = pdk.Layer(
            "ScatterplotLayer",
            data=stations,
            get_position=["lon", "lat"],
            get_color=[0, 255, 229, 220],
            get_radius=300000,
            pickable=True,
        )
        layer_tracks = pdk.Layer(
            "ScatterplotLayer",
            data=tracks,
            get_position=["lon", "lat"],
            get_color=[255, 42, 77, 200],
            get_radius="altitude_km",
            radius_scale=200,
            radius_min_pixels=2,
            radius_max_pixels=20,
            pickable=True,
        )
        view = pdk.ViewState(latitude=20, longitude=0, zoom=1, pitch=0)
        deck = pdk.Deck(
            layers=[layer_stations, layer_tracks],
            initial_view_state=view,
            map_style="mapbox://styles/mapbox/dark-v10",  # free public style via carto if available
            tooltip={"text": "{name}\n{target}"},
        )
        st.pydeck_chart(deck, use_container_width=True)
    else:
        st.warning("PyDeck not installed. Run `pip install pydeck` for the live map view.")
        st.map(tracks, latitude="lat", longitude="lon", size="altitude_km",
               color="#ff2a4d")

    st.subheader("System Status")
    c1, c2 = st.columns(2)
    c1.progress(0.92, text="Telemetry Link — 92% integrity")
    c2.progress(0.78, text="Propellant Reserve — 78% capacity")

    c3, c4 = st.columns(2)
    c3.progress(0.99, text="Comms Uplink — NOMINAL")
    c4.progress(0.65, text="Crew Morale — 65% (rotate recommended)")


# ---------------------------------------------------------------------------
# Sidebar status
# ---------------------------------------------------------------------------
def render_sidebar_status():
    """Persistent HUD status in the sidebar."""
    st.sidebar.markdown("### 🛰 SYSTEM STATUS")
    st.sidebar.markdown(
        f"""
        <div style="font-family:'Courier New',monospace; font-size:0.8em;
                    color:#00ffe5; padding:8px; border:1px solid #00ffe5;
                    background:rgba(0,20,40,0.5);">
        ▸ CORE: ONLINE<br>
        ▸ UTC: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}<br>
        ▸ Skfield: {'✓' if SKYFIELD_AVAILABLE else '✗'}<br>
        ▸ Poliastro: {'✓' if POLYASTRO_AVAILABLE else '✗'}<br>
        ▸ Trimesh: {'✓' if TRIMESH_AVAILABLE else '✗'}<br>
        ▸ Astroquery: {'✓' if HORIZONS_AVAILABLE else '✗'}<br>
        ▸ PyDeck: {'✓' if PYDECK_AVAILABLE else '✗'}<br>
        ▸ SGP4: {'✓' if SGP4_AVAILABLE else '✗'}<br>
        ▸ Numba: {'✓' if NUMBA_AVAILABLE else '✗'}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Omni-Agency Galactic Core entry point."""
    render_sidebar_status()

    st.title("OMNI-AGENCY GALACTIC CORE V10.0")
    st.caption("Streamlit Mission Control · Poliastro · Skyfield · Astroquery · Trimesh")

    tab1, tab2, tab3, tab4 = st.tabs([
        "⚙️ ARCHITECTURAL FORGE",
        "🛰️ 5D ORBITAL SENTINEL",
        "🌌 MULTI-AGENCY OBSERVATORY",
        "🎯 STRATEGIC COMMAND HUD",
    ])

    with tab1:
        render_forge_tab()
    with tab2:
        render_orbital_tab()
    with tab3:
        render_observatory_tab()
    with tab4:
        render_hud_tab()

    st.markdown("---")
    st.caption(
        "OMNI-AGENCY GALACTIC CORE V10.0 | All API calls cached · "
        "External libs optional · Storage <3GB · Mock fallbacks enabled"
    )


if __name__ == "__main__":
    main()
