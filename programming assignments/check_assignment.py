
import numpy as np
from scipy.io import wavfile

try:
    from assignment1 import *
except:
    print 'Assignment 1 files not found.'
try:
    from assignment2 import *
except:
    print 'Assignment 2 files not found.'
try:
    from assignment3 import *
except:
    print 'Assignment 3 files not found.'
try:
    from assignment4 import *
except:
    print 'Assignment 4 files not found.'


def check_assignment1():

    pass_test = True

    for i in range(1,5):
        print '  Test %d:' % (i)
        fin = 'data/testInput' + str(i) + '.wav'
        fout = 'data/testOutput' + str(i) + '.wav'

        r,x = wavfile.read(fin)

        # compute output of assignment function
        X = scaled_fft_db(x)

        print '    Signal size is 257:',
        if X.shape[0] == 257:
            print 'okay'
        else:
            print 'fail'
            pass_test = pass_test and False

        print '    Maximum is 96 dB:',
        if np.abs(X.max() - 96) < 1e-5:
            print 'okay'
        else:
            print 'fail'
            pass_test = pass_test and False

        # compare to test output file content
        X_check = np.loadtxt(fout)

        print '    Test signals output match:',
        if np.allclose(X, X_check,atol=1e-1):
            print 'okay'
        else:
            print 'fail'
            pass_test = pass_test and False

    print '  Test hanning window:',
    x = np.ones(512)
    x[1:-1] = 1./np.hanning(512)[1:-1]
    X = scaled_fft_db(x)
    win_test = np.zeros(257)
    win_test[0] = 96

    if X[0] == 96 and np.all(X[1:] < 50):
        print 'okay'
    else:
        print 'fail'
        pass_test = pass_test and False

    if pass_test:
        print 'Congratulations, your algorithm passed all the tests.'
    else:
        print 'Sorry, your algorithm is not ready for submission yet.'


def check_assignment2(plot=False):


    from parameters import filter_coeffs

    h = prototype_filter()

    # Create the cosine filter bank
    cosine_bank = np.cos(np.pi/64. * (2*np.arange(32)[:,np.newaxis]+1)*(np.arange(h.shape[0])-16))
    fb = cosine_bank*h

    # Frequency response
    from numpy import fft
    f = np.arange(257)/256./2.
    H = fft.fft(h)[:257]
    FB = fft.fft(fb, axis=1)[:,:257].T

    # ideal filter template
    ideal = np.zeros(257)
    f_pass = 1./256. # pass band is set according to standard
    f_stop = 1./64.  # stop band
    Ilo = f <= f_pass
    Ihi = f >= f_stop
    I_both = np.logical_or(Ilo,Ihi)
    ideal[f < f_stop] = 0.5
    ideal[f <= f_pass] = 2.

    if plot:
        # plot prototype filter and constraints
        import matplotlib.pyplot as plt
        plt.subplot(2,1,1)
        plt.plot(f, np.abs(H))
        plt.plot(f[I_both], ideal[I_both], 'o')

        plt.legend(('prototype filter','constraints'))
        plt.title('Prototype Filter')
        plt.xlabel('Normalized frequency')
        plt.ylabel('Magnitude')

        err_hi = np.sqrt(np.sum(np.abs(np.abs(H[Ihi])-ideal[Ihi])**2))
        if (err_hi < 1e-2):
            plt.xlim((0.,2*f_stop))

        # plot the filter bank and the sum response
        plt.subplot(2,1,2)
        plt.plot(f, np.abs(FB))
        plt.plot(f, np.abs(np.sum(FB, axis=1)))
        plt.title('Filter Bank')
        plt.xlabel('Normalized frequency')
        plt.ylabel('Magnitude')

        plt.show()

    # compute the error only on pass band and stop
    # band, not the transition band.
    error = np.abs(ideal[I_both]-np.abs(H[I_both]))

    if error.max() < 0.05:
        print 'Congratulations, the filter seems to satisfy the design constraints.'
    else:
        print 'The filter fails to satisfy the constraints.'


def check_assignment3():

    h = prototype_filter()

    # Create the cosine filter bank
    cosine_bank = np.cos(np.pi/64. * (2*np.arange(32)[:,np.newaxis]+1)*(np.arange(h.shape[0])-16))
    fb = cosine_bank*h

    pass_test = True
    n_tests = 4

    for i in range(1,n_tests+1):
        fin = 'data/testInput' + str(i) + '.wav'
        fout = 'data/a3_testOutput' + str(i) + '.txt'

        r,x_in = wavfile.read(fin)

        h = np.hanning(512)
        X = subband_filtering(x_in, h)
        X_check = np.loadtxt(fout)

        # compare to test output file content
        if np.allclose(X, X_check):
            print 'Test ' + str(i) + ' : passed.'
            pass_test = pass_test and True
        else:
            print 'Test ' + str(i) + ' : failed.'
            pass_test = pass_test and False


def check_assignment4():

    n_tests = 4

    from parameters import EncoderParameters
    params = EncoderParameters(44100, 2, 64)

    for i in range(1,n_tests+1):
        fin_name = 'data/a4_testInput' + str(i) + '.txt'
        fout_name = 'data/a4_testOutput' + str(i) + '.txt'

        val_in = np.loadtxt(fin_name)
        val_out = np.loadtxt(fout_name)

        val_chk = np.zeros(val_out.shape)

        for r,row in enumerate(val_in):
            val = row[0]
            scf = row[1]
            ba = int(row[2])
            QCa = params.table.qca[ba-2]
            QCb = params.table.qcb[ba-2]
            val_chk[r] = quantization(val, scf, ba, QCa, QCb)

        if np.allclose(val_chk, val_out):
            print 'Quantization test ' + str(i) + ' passed.'
        else:
            print 'Quantization test ' + str(i) + ' failed.'


""" When the script is called we check the outputs from
    the test inputs are correct.
"""
if __name__ == "__main__":

    # check all assignments
    print '*** Check assignment 1 : Scaled FFT in dB ***'
    try:
        check_assignment1()
    except:
        print 'Exception occured while checking assignment 1.'
    print '*** Done. ***\n'

    print '*** Check assignment 2 : Prototype filter design ***'
    try:
        check_assignment2()
    except:
        print 'Exception occured while checking assignment 2.'
    print '*** Done. ***\n'

    print '*** Check assignment 3 : Subband filtering ***'
    try:
        check_assignment3()
    except:
        print 'Exception occured while checking assignment 3.'
    print '*** Done. ***\n'

    print '*** Check assignment 4 : Quantization ***'
    try:
        check_assignment4()
    except:
        print 'Exception occured while checking assignment 4.'
    print '*** Done. ***\n'

