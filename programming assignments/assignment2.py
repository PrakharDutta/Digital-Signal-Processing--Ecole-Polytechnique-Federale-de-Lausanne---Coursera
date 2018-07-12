
from scipy import signal

def prototype_filter():
    """ ASSIGNMENT 2

        Compute the prototype filter used in subband coding. The filter
        is a 512-point lowpass FIR h[n] with bandwidth pi/64 and stopband
        starting at pi/32

        You should use the remez routine (signal.remez()). See
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.remez.html
    """
    import numpy as np

    M = 512     # number of taps
    Fs = np.pi  # sampling rate
    FPass = np.pi/128    # passband edge
    Fstop = np.pi/32     # stopband edge

    h = signal.remez(M, [0, FPass/2, Fstop/2, Fs/2], [2, 0], fs=Fs)
    return h


