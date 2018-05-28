import numpy as np

def scaled_fft_db(x):
    """ ASSIGNMENT 1:
        a) Compute a 512-point Hann window and use it to weigh the input data.
        b) Compute the DFT of the weighed input, take the magnitude in dBs and
        normalize so that the maximum value is 96dB.
        c) Return the first 257 values of the normalized spectrum

        Arguments:
        x: 512-point input buffer.

        Returns:
        first 257 points of the normalized spectrum, in dBs
    """

    # Your code goes here

