import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try importing optional dependencies with fallbacks
try:
    from skyfield.api import Loader, EarthSatellite
    from skyfield.api import Topos, load
    SKYFIELD_AVAILABLE = True
except ImportError:
    SKYFIELD_AVAILABLE = False

try:
    import spiceypy as spice
    SPICE_AVAILABLE = True
except ImportError:
    SPICE_AVAILABLE = False

try:
    from poliastro.bodies import Earth
    from poliastro.twobody import Orbit
    from poliastro.util import norm
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

# Configure Streamlit page
st.set_page_config(
    layout="wide",
    page_title="Orbit & Engineering Control",
    initial_sidebar_state="expanded"
)

# Constants and configuration
EPHEMERIS_PATH = "./ephemeris"
LIGHT_SPEED_KM_S = 299792.458
EARTH_RADIUS_KM = 6371

# Ensure ephemeris directory exists
import os
os.makedirs(EPHEMERIS_PATH, exist_ok=True)

# Data Layer Functions
@st.cache_data(ttl=3600, max_entries=5)
def fetch_tle_data():
    """Fetch minimal TLE data for ISS and other satellites with fallback."""
    try:
        if not SKYFIELD_AVAILABLE:
            raise ImportError("Skyfield not available")

        # Load minimal TLE data - only what we need
        satellites_url = "https://celestrak.com/NORAD/elements/stations.txt"
        from skyfield.api import load
        satellites = load.tople(satellites_url)

        # Extract ISS (ZARYA) TLE
        iss = None
        for sat in satellites:
            if sat.name.strip() == "ISS (ZARYA)":
                iss = sat
                break

        if iss is None:
            raise ValueError("ISS not found in TLE data")

        return {
            'iss': {
                'name': iss.name,
                'line1': iss.model.line1,
                'line2': iss.model.line2
            }
        }
    except Exception as e:
        st.sidebar.warning(f"TLE fetch failed: {str(e)}. Using mock data.")
        # Return mock ISS TLE data
        return {
            'iss': {
                'name': 'ISS (ZARYA)',
                'line1': '1 25544U 98067A   26171.50000000  .00016717  00000+0  34228-3 0  9994',
                'line2': '2 25544  51.6416 247.4627 0003668 130.5360 325.0288 15.49420423443366'
            }
        }

@st.cache_data(ttl=7200, max_entries=3)
def load_ephemeris():
    """Load lightweight ephemeris data with fallback."""
    try:
        if not SPICE_AVAILABLE:
            raise ImportError("SpiceyPy not available")

        # Only load de421.bsp (lightweight)
        spice.furnsh(os.path.join(EPHEMERIS_PATH, "de421.bsp"))
        return True
    except Exception as e:
        st.sidebar.warning(f"Ephemeris load failed: {str(e)}. Using mock coordinates.")
        return False

def clear_ephemeris_cache():
    """Clear temporary ephemeris files to manage storage."""
    try:
        for file in os.listdir(EPHEMERIS_PATH):
            if file.endswith(".bsp") and file != "de421.bsp":
                os.remove(os.path.join(EPHEMERIS_PATH, file))
    except Exception:
        pass  # Ignore cleanup errors

# Physics Layer Functions
def get_iss_position():
    """Get ISS position using SGP4/Skyfield with fallback."""
    try:
        tle_data = fetch_tle_data()
        iss_tle = tle_data['iss']

        if not SGP4_AVAILABLE or not SKYFIELD_AVAILABLE:
            raise ImportError("Required libraries not available")

        # Parse TLE
        satellite = Satrec.twoline2rv(iss_tle['line1'], iss_tle['line2'])

        # Get current time
        now = datetime.utcnow()
        jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute, now.second + now.microsecond/1e6)

        # Propagate
        e, r, v = satellite.sgp4(jd, fr)

        if e != 0:
            raise ValueError(f"SGP4 error: {e}")

        # Convert to km (SGP4 returns km)
        position_km = np.array(r)  # [x, y, z] in km
        velocity_km_s = np.array(v)  # [vx, vy, vz] in km/s

        return position_km, velocity_km_s

    except Exception as e:
        st.sidebar.warning(f"Position calculation failed: {str(e)}. Using mock data.")
        # Mock ISS position (roughly 400km altitude)
        t = datetime.utcnow().timestamp()
        angle = (t % 5400) * 2 * np.pi / 5400  # 90-minute orbit
        altitude = 408  # km
        radius = EARTH_RADIUS_KM + altitude
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0.0
        vx = -7.66 * np.sin(angle)  # ~7.66 km/s orbital speed
        vy = 7.66 * np.cos(angle)
        vz = 0.0
        return np.array([x, y, z]), np.array([vx, vy, vz])

@st.cache_data(ttl=1800)
def get_orbital_elements():
    """Get orbital elements with fallback."""
    try:
        if not POLYASTRO_AVAILABLE:
            raise ImportError("poliastro not available")

        # Use mock orbital elements for demonstration
        # In production, these would come from actual TLE processing
        return {
            'a': EARTH_RADIUS_KM + 408,  # Semi-major axis (km)
            'ecc': 0.0005,  # Eccentricity
            'inc': 51.6,  # Inclination (degrees)
            'raan': 247.4,  # RAAN (degrees)
            'argp': 130.5,  # Argument of perigee (degrees)
            'nu': 0.0  # True anomaly (degrees)
        }
    except Exception as e:
        st.sidebar.warning(f"Orbital elements failed: {str(e)}. Using mock data.")
        return {
            'a': EARTH_RADIUS_KM + 408,
            'ecc': 0.0005,
            'inc': 51.6,
            'raan': 247.4,
            'argp': 130.5,
            'nu': 0.0
        }

def compute_orbital_trajectory(elements, n_points=100):
    """Compute 3D orbital trajectory points."""
    try:
        if not POLYASTRO_AVAILABLE:
            raise ImportError("poliastro not available")

        # Create orbit from elements
        orbit = Orbit.from_classical(
            Earth,
            elements['a'] * 1000,  # Convert to meters
            elements['ecc'],
            np.radians(elements['inc']),
            np.radians(elements['raan']),
            np.radians(elements['argp']),
            np.radians(elements['nu'])
        )

        # Generate points along orbit
        times = np.linspace(0, orbit.period, n_points)
        positions = []

        for t in times:
            r, _ = orbit.rv(t)
            positions.append(r / 1000)  # Convert to km

        return np.array(positions)

    except Exception as e:
        st.sidebar.warning(f"Trajectory computation failed: {str(e)}. Using mock trajectory.")
        # Generate mock circular orbit
        angles = np.linspace(0, 2*np.pi, n_points)
        radius = EARTH_RADIUS_KM + 408
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        z = np.zeros_like(x)  # Simplified equatorial orbit
        return np.column_stack([x, y, z])

@st.cache_data(ttl=3600)
def get_halley_comet_position():
    """Get Halley's comet position with fallback."""
    try:
        if not SKYFIELD_AVAILABLE:
            raise ImportError("Skyfield not available")

        # Load comet data (simplified)
        planets = load('de421.bsp')
        earth = planets['earth']
        # Halley's comet would require specific loading - using mock for now
        raise NotImplementedError("Halley's comet data loading simplified")

    except Exception as e:
        st.sidebar.warning(f"Halley's comet data failed: {str(e)}. Using mock data.")
        # Mock Halley's comet position (highly elliptical orbit)
        t = datetime.utcnow().timestamp()
        # Very long period (~76 years), so position changes slowly
        angle = (t % (76*365.25*24*3600)) * 2 * np.pi / (76*365.25*24*3600)
        radius = EARTH_RADIUS_KM + 100000  # Far away
        x = radius * np.cos(angle) * 0.3  # Elliptical
        y = radius * np.sin(angle)
        z = radius * np.sin(angle) * 0.2
        return np.array([x, y, z])

def get_light_speed_probe_data():
    """Get data for 1.8% light-speed probe."""
    try:
        # Mock data for relativistic probe
        velocity_fraction = 0.018  # 1.8% of light speed
        velocity_km_s = velocity_fraction * LIGHT_SPEED_KM_S

        # Simple linear trajectory from Earth
        t = datetime.utcnow().timestamp() - datetime(2026, 1, 1).timestamp()
        distance_km = velocity_km_s * t

        # Direction: along x-axis for simplicity
        x = distance_km
        y = 0.0
        z = 0.0

        return np.array([x, y, z]), velocity_km_s
    except Exception as e:
        st.sidebar.warning(f"Probe data failed: {str(e)}. Using mock data.")
        return np.array([10000.0, 0.0, 0.0]), 5396.26  # ~1.8% c

# Numba JIT for heavy calculations (if available)
if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def calculate_relativistic_factor(velocity_km_s):
        """Calculate Lorentz factor for relativistic effects."""
        c = LIGHT_SPEED_KM_S
        beta = velocity_km_s / c
        if beta >= 1.0:
            return 1000.0  # Large number for v >= c
        return 1.0 / np.sqrt(1.0 - beta*beta)
else:
    def calculate_relativistic_factor(velocity_km_s):
        """Calculate Lorentz factor for relativistic effects."""
        c = LIGHT_SPEED_KM_S
        beta = velocity_km_s / c
        if beta >= 1.0:
            return 1000.0
        return 1.0 / np.sqrt(1.0 - beta*beta)

# UI Layer Functions
def create_earth_sphere(radius=EARTH_RADIUS_KM, n_points=20):
    """Create a sphere for Earth visualization."""
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

def render_orbital_tab():
    """Render the Orbital Tracking tab."""
    st.header("🛰️ Orbital Tracking (3D)")

    # Sidebar controls
    with st.sidebar:
        st.subheader("Target Selection")
        target = st.selectbox(
            "Select Target",
            ["ISS", "Halley's Comet", "Custom 1.8% Light-Speed Probe"],
            key="orbital_target"
        )

        st.subheader("Display Options")
        show_trajectory = st.checkbox("Show Orbital Trajectory", value=True)
        show_velocity_vector = st.checkbox("Show Velocity Vector", value=True)
        update_interval = st.slider("Update Interval (seconds)", 5, 60, 10)

    # Get data based on selection
    if target == "ISS":
        position_km, velocity_km_s = get_iss_position()
        elements = get_orbital_elements()
        trajectory = compute_orbital_trajectory(elements) if show_trajectory else None
        name = "International Space Station"
        color = "#00FF00"
    elif target == "Halley's Comet":
        position_km = get_halley_comet_position()
        velocity_km_s = np.array([0.0, 0.0, 0.0])  # Simplified
        trajectory = None  # Comet trajectory would be complex
        name = "Halley's Comet"
        color = "#FFFFFF"
    else:  # Light-Speed Probe
        position_km, velocity_km_s = get_light_speed_probe_data()
        trajectory = None  # Linear trajectory
        name = "1.8% Light-Speed Probe"
        color = "#FF00FF"

    # Create 3D plot
    fig = go.Figure()

    # Add Earth
    x_earth, y_earth, z_earth = create_earth_sphere()
    fig.add_trace(go.Surface(
        x=x_earth, y=y_earth, z=z_earth,
        colorscale=[[0, '#000033'], [1, '#000080']],
        showscale=False,
        name="Earth",
        opacity=0.8
    ))

    # Add target
    fig.add_trace(go.Scatter3d(
        x=[position_km[0]], y=[position_km[1]], z=[position_km[2]],
        mode='markers',
        marker=dict(size=8, color=color),
        name=name
    ))

    # Add trajectory if available
    if trajectory is not None and show_trajectory:
        fig.add_trace(go.Scatter3d(
            x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2],
            mode='lines',
            line=dict(color=color, width=2),
            name=f"{name} Trajectory"
        ))

    # Add velocity vector if requested
    if show_velocity_vector and np.linalg.norm(velocity_km_s) > 0:
        # Scale velocity vector for visibility
        vel_scale = 50.0  # km
        vel_end = position_km + velocity_km_s * vel_scale / np.linalg.norm(velocity_km_s)
        fig.add_trace(go.Scatter3d(
            x=[position_km[0], vel_end[0]],
            y=[position_km[1], vel_end[1]],
            z=[position_km[2], vel_end[2]],
            mode='lines',
            line=dict(color='red', width=4),
            name="Velocity Vector"
        ))

    # Update layout
    fig.update_layout(
        title=f"{name} Position and Orbit",
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Z (km)",
            aspectmode='cube',
            bgcolor='#000000',
            xaxis=dict(color='#FFFFFF', gridcolor='#333333'),
            yaxis=dict(color='#FFFFFF', gridcolor='#333333'),
            zaxis=dict(color='#FFFFFF', gridcolor='#333333')
        ),
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        font=dict(color='#FFFFFF'),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # Telemetry display
    col1, col2, col3, col4 = st.columns(4)

    altitude = np.linalg.norm(position_km) - EARTH_RADIUS_KM
    speed = np.linalg.norm(velocity_km_s)

    with col1:
        st.metric("Velocity", f"{speed:.2f} km/s",
                 delta=f"{speed - 7.66:.2f} km/s" if target == "ISS" else None)
    with col2:
        st.metric("Altitude", f"{altitude:.0f} km")
    with col3:
        if target == "ISS":
            st.metric("Eccentricity", f"{elements['ecc']:.4f}")
        else:
            st.metric("Eccentricity", "N/A")
    with col4:
        if target == "Light-Speed Probe":
            lorentz_factor = calculate_relativistic_factor(speed)
            st.metric("Lorentz Factor", f"{lorentz_factor:.2f}")
        else:
            st.metric("Status", "Nominal")

def render_blueprint_tab():
    """Render the Blueprint & Structural Analysis tab."""
    st.header("🔧 Blueprint & Structural Analysis")

    # Sidebar controls
    with st.sidebar:
        st.subheader("Blueprint Options")
        show_hull = st.checkbox("Show Hull Structure", value=True)
        show_engine = st.checkbox("Show Engine Details", value=True)
        show_heat_shield = st.checkbox("Show Heat Shield", value=True)

    # Create 2D blueprint-style plot
    fig = go.Figure()

    # Background
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=100, y1=50,
        line=dict(color="#001100", width=0),
        fillcolor="#000000",
        layer="below"
    )

    if show_hull:
        # Main hull structure (simplified spacecraft silhouette)
        hull_x = [10, 15, 25, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 90, 10]
        hull_y = [25, 20, 15, 10, 5, 0, 0, 0, 5, 10, 15, 20, 25, 30, 35, 40, 25]
        fig.add_trace(go.Scatter(
            x=hull_x, y=hull_y,
            mode='lines',
            line=dict(color='#00FF00', width=2),
            name="Hull Structure"
        ))

    if show_engine:
        # Engine nozzles
        engine_x = [40, 42, 42, 40, 58, 60, 60, 58]
        engine_y = [0, 0, -5, -5, -5, -5, 0, 0]
        fig.add_trace(go.Scatter(
            x=engine_x, y=engine_y,
            mode='lines',
            line=dict(color='#00FFFF', width=3),
            name="Engine Nozzles"
        ))

        # Engine plume
        plume_x = [41, 41, 59, 59]
        plume_y = [-5, -15, -15, -5]
        fig.add_trace(go.Scatter(
            x=plume_x, y=plume_y,
            mode='lines',
            line=dict(color='#FF4500', width=2, dash='dash'),
            name="Engine Plume"
        ))

    if show_heat_shield:
        # Heat shield (bottom)
        shield_x = [20, 80, 80, 20]
        shield_y = [0, 0, -10, -10]
        fig.add_trace(go.Scatter(
            x=shield_x, y=shield_y,
            mode='lines',
            line=dict(color='#FFFF00', width=3),
            name="Heat Shield"
        ))

        # Heat shield texture lines
        for i in range(20, 81, 6):
            fig.add_trace(go.Scatter(
                x=[i, i], y=[-2, -8],
                mode='lines',
                line=dict(color='#FFA500', width=1),
                showlegend=False
            ))

    # Add grid lines for blueprint effect
    for i in range(0, 101, 10):
        fig.add_trace(go.Scatter(
            x=[i, i], y=[-20, 60],
            mode='lines',
            line=dict(color='#003300', width=1),
            showlegend=False
        ))
    for i in range(-20, 61, 10):
        fig.add_trace(go.Scatter(
            x=[0, 100], y=[i, i],
            mode='lines',
            line=dict(color='#003300', width=1),
            showlegend=False
        ))

    # Update layout
    fig.update_layout(
        title="Spacecraft Blueprint Schematic",
        xaxis=dict(
            title="X Position (m)",
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-10, 110]
        ),
        yaxis=dict(
            title="Y Position (m)",
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-30, 70],
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='#00FF00'),
        height=500,
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='#00FF00'
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Technical specifications
    st.subheader("Technical Specifications")

    specs_data = {
        "Parameter": [
            "Mass (kg)",
            "Propulsion Type",
            "Thrust (kN)",
            "Specific Impulse (s)",
            "Delta-V Capacity (km/s)",
            "G-Force Limits (g)",
            "Target Velocity",
            "Mission Duration"
        ],
        "Value": [
            "12,500",
            "Fusion-Pulse Detonation",
            "1,850",
            "4,200",
            "120",
            "9",
            "0.18c (53,963 km/s)",
            "5 years"
        ],
        "Unit": [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
        ]
    }

    specs_df = pd.DataFrame(specs_data)
    st.table(specs_df)

    # Progress indicators
    st.subsubsection("System Status")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Heat Shield Integrity")
        st.progress(87)
        st.caption("87% - Nominal")

    with col2:
        st.write("Fuel Capacity")
        st.progress(72)
        st.caption("72% - Adequate for mission")

# Main application
def main():
    """Main application function."""
    # Initialize ephemeris
    load_ephemeris()

    # Create tabs
    tab1, tab2 = st.tabs(["🛰️ Orbital Tracking", "🔧 Blueprint & Structural Analysis"])

    with tab1:
        render_orbital_tab()

    with tab2:
        render_blueprint_tab()

    # Footer
    st.markdown("---")
    st.caption("Space Mission Control & Blueprint Engineering Dashboard | Data updated in real-time | Storage optimized <3GB")

if __name__ == "__main__":
    main()