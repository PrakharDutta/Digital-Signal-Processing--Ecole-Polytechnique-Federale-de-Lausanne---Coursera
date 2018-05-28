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

    # Your code goes here
