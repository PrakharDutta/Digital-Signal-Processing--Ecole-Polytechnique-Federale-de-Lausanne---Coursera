import sys
import numpy as np
from scipy.io import wavfile


def output(partIdx):
  outputString = ''

  if partIdx == '1': # This is ScaledFFTdB

    from assignment1 import scaled_fft_db

    r,x = wavfile.read('data/a1_submissionInput.wav')
    X = scaled_fft_db(x)

    for val in X:
      outputString += '%.5f ' % (val)


  elif partIdx == '2': # This is PrototypeFilter

    from assignment2 import prototype_filter

    h = prototype_filter()
      
    # test signal
    s = np.loadtxt('data/a2_submissionInput.txt')
    r = np.convolve(h, s)[4*512:5*512]/2

    for val in r:
      outputString += '%.5f ' % val

  elif partIdx == '3': # This is SubbandFiltering

    from assignment3 import subband_filtering

    r,x = wavfile.read('data/a3_submissionInput.wav')

    h = np.hanning(512)
    X = subband_filtering(x, h)

    for val in X:
      outputString += '%.5f ' % (val)

  elif partIdx == '4': # This is Quantization

    from assignment4 import quantization

    from parameters import EncoderParameters
    params = EncoderParameters(44100, 2, 64)

    val_in = np.loadtxt('data/a4_submissionInput.txt')

    for r,row in enumerate(val_in):
      val = row[0]
      scf = row[1]
      ba = int(row[2])
      QCa = params.table.qca[ba-2]
      QCb = params.table.qcb[ba-2]
      val = quantization(val, scf, ba, QCa, QCb)
      outputString += '%d ' % (val)

  else:
    print "Unknown assigment part number"

  if len(outputString) > 0:
    fileName = "res%s.txt" % partIdx;
    with open(fileName, "w") as f:
      f.write(outputString.strip())
      print "You can now submit the file " + fileName
  else:
    print "there was an error with the computation. Please check your code"


if __name__ == "__main__":
  output(sys.argv[1])
