
import numpy as np

def quantization(sample, sf, ba, QCa, QCb):
    """ ASSIGNMENT 4

        Arguments:
        sample: the sample to quantize
        sf:     the scale factor
        ba:     the bit allocation
        QCa:    the multiplicative uniform quantization parameter
        QCb:    the additive uniform quantization parameter

        Returns:
        The uniformly quantized sample.
    """

    return np.floor((QCa*sample/sf + QCb)*2**(ba-1))
