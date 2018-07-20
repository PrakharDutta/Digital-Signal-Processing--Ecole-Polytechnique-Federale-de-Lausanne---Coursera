import numpy as np

def subband_filtering(x, h):
    """ ASSIGNMENT 3

        Write a routine to implement the efficient version of the subband filter
        as specified by the MP3 standard

        Arguments:
        x:  a new 512-point data buffer, in time-reversed order [x[n],x[n-1],...,x[n-511]].
        h:  The prototype filter of the filter bank you found in the previous assignment

        Returns:
        s: 32 new output samples
    """

    r = h * x
    q = np.arange(64)    # range from 0 to 63
    c = np.sum((-1)**np.arange(8)[:, np.newaxis] * r[q + 64*np.arange(8)[:, np.newaxis]], axis=0)
    # Compute subband outputs
    s = np.sum(np.cos(np.pi / 64. * (2 * np.arange(32)[:, np.newaxis] + 1) * (np.arange(q.shape[0]) - 16))*c, axis=1)
    return s
