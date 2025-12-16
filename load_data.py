import glob

import numpy as np
import pandas as pd

import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic

import astropy.coordinates as coord
import astropy.units as u
from astropy.constants import G

from scipy.special import gammainc, gamma

_ = coord.galactocentric_frame_defaults.set("v4.0")

import h5py
import csv

gala_pot = gp.BovyMWPotential2014(units=galactic)

acent_R0 = 6.32793804994 * u.pc / (u.Myr) ** 2.0
R0_nbody = 8.0 * u.kpc
scale_mass = (acent_R0 * R0_nbody**2.0) / G

# bulge
fb = 0.05
bulge_rc = 1.9 * u.kpc
bulge_alpha = 1.8
bulge_s = 0.5 * (3.0 - bulge_alpha)  # 0.5 * (3.0 - bulge_alpha)

Gamma_s = gamma(bulge_s)
Gamma_numerator = gamma(1 - bulge_alpha / 2.0)

x_R0 = ((R0_nbody / bulge_rc) ** 2.0).value
gamma_s_x_R0 = gammainc(bulge_s, x_R0) * Gamma_s
Mb = fb * scale_mass * (Gamma_s / gamma_s_x_R0)

psi_bulge_at_infinity = (G * Mb / bulge_rc) * (Gamma_numerator / Gamma_s)


def read_from_filename(filename):
    """
    get cleaned pandas df from csv file
    """

    delimiter = ","  # default
    with open(filename, "r") as f:
        sample = f.read(1024)
        try:
            dialect = csv.Sniffer().sniff(sample)
            delimiter = dialect.delimiter
        except csv.Error:
            pass

    df = pd.read_csv(
        filename,
        header=0,
        sep=delimiter,
        engine="python",
    )

    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.replace(["-nan", "nan", "NaN", "-NaN"], np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors="ignore")
    df = df.dropna()

    return df


def get_orbital_pole(pos, vel):
    """
    compute the unit vector that points in the direction of r x v
    return normalized orbital pole of all particles
    """
    r_cross_v = np.cross(pos, vel)
    r_cross_v_norm = np.linalg.norm(r_cross_v)

    return r_cross_v.T / r_cross_v_norm


def get_phi_info(df, pole_vec, ref_point_vec):
    """
    Compute phi_1 and phi_2 for a given dataframe containing MW phase space (x, y, z, vx, vy, vz),
    dynamically determining the best great circle fit from the dataset
    """

    # x direction: vector from origin to progenitor (ref point in position space)
    x_vec = ref_point_vec
    x_vec /= np.linalg.norm(x_vec)

    # z direction: use the orbital pole
    z_vec = pole_vec
    z_vec /= np.linalg.norm(z_vec)

    # y direction: complete right-handed system
    y_vec = np.cross(z_vec, x_vec)
    y_vec /= np.linalg.norm(y_vec)

    xnew = coord.representation.CartesianRepresentation(*(x_vec * u.kpc))
    znew = coord.representation.CartesianRepresentation(*(z_vec * u.kpc))

    frame = gc.GreatCircleICRSFrame.from_xyz(xnew=xnew, znew=znew)

    # positions in kpc
    positions = coord.representation.CartesianRepresentation(
        df["x"].values * u.kpc, df["y"].values * u.kpc, df["z"].values * u.kpc
    )

    # velocities in km/s
    velocities = coord.representation.CartesianDifferential(
        df["vx"].values * u.km / u.s,
        df["vy"].values * u.km / u.s,
        df["vz"].values * u.km / u.s,
    )

    # construct SkyCoord from these phase-space coordinates
    positions_with_velocities = positions.with_differentials(velocities)
    skycoord = coord.SkyCoord(positions_with_velocities)
    coord_new = skycoord.transform_to(frame)

    mu_phi1 = coord_new.pm_phi1_cosphi2 / np.cos(coord_new.phi2)
    mu_phi1 = mu_phi1.to(u.mas / u.yr).value
    mu_phi2 = coord_new.pm_phi2.to(u.mas / u.yr).value

    phi_info_dict = {
        "phi1": coord_new.phi1.to(u.deg).value,
        "phi2": coord_new.phi2.to(u.deg).value,
        "mu_phi1": mu_phi1,
        "mu_phi2": mu_phi2,
    }

    return phi_info_dict


def load_krios_snapshot_data(data_dir, orbit_id, time):
    """
    returns two pd.DataFrame objects: for the particles, another for the basis-function weights
    """

    df_krios_particles = read_from_filename(
        glob.glob(f"{data_dir}/snapshot_job_id_{orbit_id}_time_{time}*.csv")[0]
    )

    df_basis_fn_weights = read_from_filename(
        glob.glob(f"{data_dir}/basis_fn_weights_job_id_{orbit_id}_time_{time}*.csv")[0]
    )

    df_scf_frame_info = read_from_filename(
        glob.glob(f"{data_dir}/scf_frame_*.csv")[0],
    )

    df_krios_particles["E_wrt_cluster"] = (
        df_krios_particles["K_wrt_SCF_center"] + df_krios_particles["U_nbody"]
    )

    pos = df_krios_particles[["x", "y", "z"]].values * u.kpc
    vel = df_krios_particles[["vx", "vy", "vz"]].values * u.km / u.s
    phase_space = gd.PhaseSpacePosition(pos=pos.T, vel=vel.T)

    E_wrt_host = (
        gala_pot.total_energy(phase_space.pos.xyz, phase_space.vel.d_xyz)
        - psi_bulge_at_infinity
    ).to_value((u.km / u.s) ** 2.0)
    Lz = phase_space.angular_momentum()[2].to_value(u.kpc * u.km / u.s)

    df_krios_particles["L_z"] = Lz / 1e3
    df_krios_particles["E_wrt_host"] = E_wrt_host / 1e5

    idx_closest = np.argmin(np.abs(df_scf_frame_info["t"] - time))
    scf_frame = df_scf_frame_info.iloc[idx_closest].to_dict()

    scf_pos = np.array([scf_frame["x"], scf_frame["y"], scf_frame["z"]])
    scf_vel = np.array([scf_frame["vx"], scf_frame["vy"], scf_frame["vz"]])

    pole_vec = get_orbital_pole(scf_pos, scf_vel)

    phi_info_dict = get_phi_info(
        df_krios_particles, ref_point_vec=scf_pos, pole_vec=pole_vec
    )
    df_krios_particles["phi1"] = phi_info_dict["phi1"]
    df_krios_particles["phi2"] = phi_info_dict["phi2"]
    df_krios_particles["mu_phi1"] = phi_info_dict["mu_phi1"]
    df_krios_particles["mu_phi2"] = phi_info_dict["mu_phi2"]

    df_krios_particles["is_stream"] = df_krios_particles["E_wrt_cluster"] > 0.0

    return {
        "particle_data": df_krios_particles,
        "basis_fn_weights": df_basis_fn_weights,
    }


def load_krios_run_data(data_dir):
    """
    collects metadata from the KRIOS run e.g. the El power spectrum as a function of time
    """

    return {
        "conserved_quantities": read_from_filename(
            glob.glob(f"{data_dir}/conserved_quantities_*_time_series.csv")[0]
        ),
        "scf_frame": read_from_filename(glob.glob(f"{data_dir}/scf_frame_*.csv")[0]),
        "barycenter_frame": read_from_filename(
            glob.glob(f"{data_dir}/barycenter_*.csv")[0]
        ),
        "density_center_frame": read_from_filename(
            glob.glob(f"{data_dir}/scf_frame_*.csv")[0]
        ),
        "radii_info": read_from_filename(
            glob.glob(f"{data_dir}/radii_info_*_time_series.csv")[0]
        ),
        "tidal_tensor_info": read_from_filename(
            glob.glob(f"{data_dir}/tidal_tensor_*.csv")[0]
        ),
        "scf_power_info": read_from_filename(
            glob.glob(f"{data_dir}/scf_power_spectrum_*.csv")[0]
        ),
    }


def load_ps_data(data_dir):
    """
    get article-spray data for filename
    """

    filename = data_dir + "eccentric_orbit_ps_plummer_progenitor_stream_seed0.csv"
    df_ps = read_from_filename(filename)
    df_ps["is_stream"] = df_ps["source"] == "stream"

    # compute phase-space quantities
    pos = df_ps[["x", "y", "z"]].values * u.kpc
    vel = df_ps[["vx", "vy", "vz"]].values * u.km / u.s
    phase_space = gd.PhaseSpacePosition(pos=pos.T, vel=vel.T)

    E_wrt_host = (
        (
            gala_pot.total_energy(phase_space.pos.xyz, phase_space.vel.d_xyz)
            - psi_bulge_at_infinity
        )
        .to((u.km / u.s) ** 2.0)
        .value
    )
    Lz = phase_space.angular_momentum()[2].value

    df_ps["L_z"] = Lz / 1e3
    df_ps["E_wrt_host"] = E_wrt_host / 1e5

    # progenitor-based reference point
    df_ps_bound = df_ps[df_ps["source"] == "prog"]

    prog_pos = np.array(
        [
            df_ps_bound["x"].mean(),
            df_ps_bound["y"].mean(),
            df_ps_bound["z"].mean(),
        ]
    )
    prog_vel = np.array(
        [
            df_ps_bound["vx"].mean(),
            df_ps_bound["vy"].mean(),
            df_ps_bound["vz"].mean(),
        ]
    )
    pole_vec = get_orbital_pole(prog_pos, prog_vel)
    phi_info_dict = get_phi_info(df_ps, ref_point_vec=prog_pos, pole_vec=pole_vec)

    df_ps["phi1"] = phi_info_dict["phi1"]
    df_ps["phi2"] = phi_info_dict["phi2"]
    df_ps["mu_phi1"] = phi_info_dict["mu_phi1"]
    df_ps["mu_phi2"] = phi_info_dict["mu_phi2"]

    return df_ps


def load_nbody_run_data(data_dir):
    """
    load N-body data from .hf5 files into a dataframe
    """

    # find all hf5 files
    filename = data_dir + "iom_cluster_5e4_eccentric_seed_0.hf5"

    df = pd.DataFrame()
    with h5py.File(filename, "r") as f:
        df["t"] = f["data_time_Myr"][()]
        df["r_0.01"] = f["data_lagrange_rad_01_pc"][()]
        df["r_0.1"] = f["data_lagrange_rad_10_pc"][()]
        df["r_0.3"] = f["data_lagrange_rad_30_pc"][()]
        df["r_0.5"] = f["data_lagrange_rad_50_pc"][()]
        df["r_0.7"] = f["data_lagrange_rad_70_pc"][()]
        df["r_0.8"] = f["data_lagrange_rad_80_pc"][()]
        df["r_0.9"] = f["data_lagrange_rad_90_pc"][()]
        df["r_0.99"] = f["data_lagrange_rad_99_pc"][()]
        df["unbound_frac"] = f["data_unbound_frac"][()]

    return df


def load_nbody_snapshot_data(data_dir):
    """
    Loads N-body simulation data
    """

    # find all hf5 files
    filename = data_dir + "snapshot_cluster_5e4_eccentric_seed_0_last_snapshot.hf5"

    with h5py.File(filename, "r") as f:
        # each are (N, 3)
        data_pos_in_kpc = f["data_pos_in_kpc"][:].T
        data_vel_in_kms = f["data_vel_in_kms"][:].T  # in km/s

        # (N,2)
        data_phi1_phi2_deg = f["data_phi1_phi2_in_deg"][:].T

        # (N, 1) but sometimes read as (N,)
        data_E_wrt_cluster = f["data_E_wrt_cluster_in_kms"][:]
        if data_E_wrt_cluster.ndim == 1:
            data_E_wrt_cluster = data_E_wrt_cluster[:, np.newaxis]  # make (N,1)

    # Create DataFrame
    df = pd.DataFrame(
        np.hstack(
            [
                data_pos_in_kpc,
                data_vel_in_kms,
                data_phi1_phi2_deg,
                data_E_wrt_cluster,
            ]
        ),
        columns=[
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "phi1",
            "phi2",
            "E_wrt_cluster",
        ],
    )

    # Use bound to define angles, keep unbound only for subsequent analysis
    df_bound = df[(df["E_wrt_cluster"] <= 0.0)]

    # Get index of most bound particle
    idx_mbp = df_bound["E_wrt_cluster"].idxmin()

    # Extract position and velocity as numpy arrays
    frame_pos = df_bound.loc[idx_mbp, ["x", "y", "z"]].to_numpy()

    df["dist_from_center"] = np.sqrt(
        (df["x"] - frame_pos[0]) ** 2.0
        + (df["y"] - frame_pos[1]) ** 2.0
        + (df["z"] - frame_pos[2]) ** 2.0
    )

    bound_particles = df[(df["E_wrt_cluster"] <= 0.0)]
    bound_radius = bound_particles["dist_from_center"].max()

    outside_radius = df["dist_from_center"] >= bound_radius
    df["is_stream"] = outside_radius

    pos = df[["x", "y", "z"]].values * u.kpc
    vel = df[["vx", "vy", "vz"]].values * u.km / u.s
    phase_space = gd.PhaseSpacePosition(pos=pos.T, vel=vel.T)

    df["E_wrt_host"] = (
        (
            gala_pot.total_energy(phase_space.pos.xyz, phase_space.vel.d_xyz)
            - psi_bulge_at_infinity
        )
        .to((u.km / u.s) ** 2.0)
        .value
    ) / 1e5
    df["L_z"] = phase_space.angular_momentum()[2].value / 1e3

    return df
