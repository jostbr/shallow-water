
"""File with several visualization functions intended to use
with results from 2D shallow water model swe2D.py"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import fourier_transform as ft

#plt.style.use("seaborn")

def eta_animation(X, Y, eta_list, frame_interval, filename):
    """Function that takes in the domain x, y (2D meshgrids) and a list of 2D arrays
    eta_list and creates an animation of all eta images. To get updating title one
    also need specify time step dt between each frame in the simulation, the number
    of time steps between each eta in eta_list and finally, a filename for video."""
    fig, ax = plt.subplots(1, 1)
    #plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days", fontname = "serif", fontsize = 17)
    plt.xlabel("x [m]", fontname = "serif", fontsize = 12)
    plt.ylabel("y [m]", fontname = "serif", fontsize = 12)
    pmesh = plt.pcolormesh(X, Y, eta_list[0], vmin = -0.7*np.abs(eta_list[int(len(eta_list)/2)]).max(),
        vmax = np.abs(eta_list[int(len(eta_list)/2)]).max(), cmap = plt.cm.RdBu_r)
    plt.colorbar(pmesh, orientation = "vertical")

    # Update function for quiver animation.
    def update_eta(num):
        ax.set_title("Surface elevation $\eta$ after t = {:.2f} hours".format(
            num*frame_interval/3600), fontname = "serif", fontsize = 16)
        pmesh.set_array(eta_list[num][:-1, :-1].flatten())
        return pmesh,

    anim = animation.FuncAnimation(fig, update_eta,
        frames = len(eta_list), interval = 10, blit = False)
    mpeg_writer = animation.FFMpegWriter(fps = 24, bitrate = 10000,
        codec = "libx264", extra_args = ["-pix_fmt", "yuv420p"])
    anim.save("{}.mp4".format(filename), writer = mpeg_writer)
    return anim    # Need to return anim object to see the animation

def velocity_animation(X, Y, u_list, v_list, frame_interval, filename):
    """Function that takes in the domain x, y (2D meshgrids) and a lists of 2D arrays
    u_list, v_list and creates an quiver animation of the velocity field (u, v). To get
    updating title one also need specify time step dt between each frame in the simulation,
    the number of time steps between each eta in eta_list and finally, a filename for video."""
    fig, ax = plt.subplots(figsize = (8, 8), facecolor = "white")
    plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days", fontname = "serif", fontsize = 19)
    plt.xlabel("x [km]", fontname = "serif", fontsize = 16)
    plt.ylabel("y [km]", fontname = "serif", fontsize = 16)
    q_int = 3
    Q = ax.quiver(X[::q_int, ::q_int]/1000.0, Y[::q_int, ::q_int]/1000.0, u_list[0][::q_int,::q_int], v_list[0][::q_int,::q_int],
        scale=0.2, scale_units='inches')
    #qk = plt.quiverkey(Q, 0.9, 0.9, 0.001, "0.1 m/s", labelpos = "E", coordinates = "figure")

    # Update function for quiver animation.
    def update_quiver(num):
        u = u_list[num]
        v = v_list[num]
        ax.set_title("Velocity field $\mathbf{{u}}(x,y,t)$ after t = {:.2f} hours".format(
            num*frame_interval/3600), fontname = "serif", fontsize = 19)
        Q.set_UVC(u[::q_int, ::q_int], v[::q_int, ::q_int])
        return Q,

    anim = animation.FuncAnimation(fig, update_quiver,
        frames = len(u_list), interval = 10, blit = False)
    mpeg_writer = animation.FFMpegWriter(fps = 24, bitrate = 10000,
        codec = "libx264", extra_args = ["-pix_fmt", "yuv420p"])
    fig.tight_layout()
    anim.save("{}.mp4".format(filename), writer = mpeg_writer)
    return anim    # Need to return anim object to see the animation

def eta_animation3D(X, Y, eta_list, frame_interval, filename):
    fig = plt.figure(figsize = (8, 8), facecolor = "white")
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, eta_list[0], cmap = plt.cm.RdBu_r)

    def update_surf(num):
        ax.clear()
        surf = ax.plot_surface(X/1000, Y/1000, eta_list[num], cmap = plt.cm.RdBu_r)
        ax.set_title("Surface elevation $\eta(x,y,t)$ after $t={:.2f}$ hours".format(
            num*frame_interval/3600), fontname = "serif", fontsize = 19, y=1.04)
        ax.set_xlabel("x [km]", fontname = "serif", fontsize = 14)
        ax.set_ylabel("y [km]", fontname = "serif", fontsize = 14)
        ax.set_zlabel("$\eta$ [m]", fontname = "serif", fontsize = 16)
        ax.set_xlim(X.min()/1000, X.max()/1000)
        ax.set_ylim(Y.min()/1000, Y.max()/1000)
        ax.set_zlim(-0.3, 0.7)
        plt.tight_layout()
        return surf,

    anim = animation.FuncAnimation(fig, update_surf,
        frames = len(eta_list), interval = 10, blit = False)
    mpeg_writer = animation.FFMpegWriter(fps = 24, bitrate = 10000,
        codec = "libx264", extra_args = ["-pix_fmt", "yuv420p"])
    anim.save("{}.mp4".format(filename), writer = mpeg_writer)
    return anim    # Need to return anim object to see the animation

def surface_plot3D(X, Y, eta, x_lim, y_lim, z_lim):
    """Function that takes input 1D coordinate arrays x, y and 2D array
    array psi. Then plots psi as a surface in 3D space on a meshgrid."""
    fig = plt.figure(figsize = (11, 7))
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, eta, rstride = 1, cstride = 1,
        cmap = plt.cm.jet, linewidth = 0, antialiased = True)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)
    ax.set_title("Surface elevation $\eta$", fontname = "serif", fontsize = 17)
    ax.set_xlabel("x [m]", fontname = "serif", fontsize = 16)
    ax.set_ylabel("y [m]", fontname = "serif", fontsize = 16)
    ax.set_zlabel("Surface elevation [m]", fontname = "serif", fontsize = 16)
    plt.show()

def pmesh_plot(X, Y, eta, plot_title):
    """Function that generates a colored contour plot of eta in the domain X, Y"""
    plt.figure(figsize = (9, 8))
    plt.pcolormesh(X, Y, eta, cmap = plt.cm.RdBu_r)
    plt.colorbar(orientation = "vertical")
    plt.title(plot_title, fontname = "serif", fontsize = 17)
    plt.xlabel("x [m]", fontname = "serif", fontsize = 12)
    plt.ylabel("y [s]", fontname = "serif", fontsize = 12)

def quiver_plot(X, Y, U, V, plot_title):
    """Function that makes a quiver plot of (U, V) at points (X, Y)."""
    plt.figure()
    plt.title(plot_title, fontname = "serif", fontsize = 17)
    plt.xlabel("x [m]", fontname = "serif", fontsize = 12)
    plt.ylabel("y [m]", fontname = "serif", fontsize = 12)
    Q = plt.quiver(X[::4, ::4], Y[::4, ::4], U[::4, ::4], V[::4, ::4],
        units = "xy", scale = 0.002, scale_units = "inches")
    qk = plt.quiverkey(Q, 0.9, 0.9, 0.001, "0.1 m/s",
        labelpos = "E", coordinates = "figure")

def hovmuller_plot(x, t, eta):
    """Function that generates a Hovmuller diagram of
    eta as a function of x and t at a choosen y-coordinate"""
    X, T = np.meshgrid(x, np.array(t))
    X = np.transpose(X)         # Transpose for plotting
    T = np.transpose(T)         # Transpose for plotting
    eta_hm = np.transpose(np.array(eta))  # Transpose for plotting

    plt.figure(figsize = (5, 8))
    plt.pcolormesh(X, T, eta_hm, vmin = eta_hm.min(), vmax = eta_hm.max(), cmap = plt.cm.PiYG)
    plt.colorbar(orientation = "vertical")
    plt.title("x-t plot for middle of domain", fontname = "serif", fontsize = 17)
    plt.xlabel("x [m]", fontname = "serif", fontsize = 12)
    plt.ylabel("t [s]", fontname = "serif", fontsize = 12)

def plot_time_series_and_ft(t, signal):
    """Function that takes a signal and its corresponding time array.
    Then plots the time signal as well as its Fourier transform."""
    t = np.array(t)
    signal = np.array(signal)

    # Plotting the time series.
    plt.figure(figsize = (8, 7))
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, linewidth = 2)
    plt.title("Time series of $\eta$ at center of domain", fontname = "serif", fontsize = 17)
    plt.xlabel("t [s]", fontname = "serif", fontsize = 12)
    plt.ylabel("$\eta$ [m]", fontname = "serif", fontsize = 12)

    # Plotting the Fourier transform of the time series (calling homemade ft).
    freq, spectrum = ft.fourier_transform(signal, len(signal), len(signal)*np.diff(t)[1])
    plt.subplot(2, 1, 2)
    plt.plot(freq, spectrum, linewidth = 2)
    plt.title("Fourier transformed signal", fontname = "serif", fontsize = 17)
    plt.xlabel("Frequency [Hz]", fontname = "serif", fontsize = 12)
    plt.ylabel("Amplitude", fontname = "serif", fontsize = 12)
    plt.tight_layout()