
import numpy as np

def fourier_transform(signal, N, T):
    """Function for computing the Fourier ransform of <signal>
    consisting of <N> data points and and upper sample time <T>"""
    N_half = int(round(N/2))

    A = np.zeros(N_half)
    B = np.zeros(N_half)
    A[0] = signal.mean()

    q = np.arange(1, N+1)

    for n in range(1, N_half):
        A[n-1] = np.sum(2/N*signal*np.cos((2*np.pi*q*n)/N))
        B[n-1] = np.sum(2/N*signal*np.sin((2*np.pi*q*n)/N))

    A[N_half-1] = np.sum(1/N*signal*np.cos(np.pi*q))
    B[N_half-1] = 0

    freq = np.zeros(N_half)

    for n in range(1, N_half+1):
        freq[n-1] = n/T

    E = A**2 + B**2

    return freq, E