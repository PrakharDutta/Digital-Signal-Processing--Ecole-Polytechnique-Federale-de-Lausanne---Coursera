import sys
import os.path
import numpy as np
import psychoacoustic as psycho
from common import *
from parameters import *
	
import assignment2
import assignment3
import assignment4

def main(inwavfile, outmp3file, bitrate):
  """Encoder main function."""

  #inwavfile  = "../samples/sinestereo.wav"
  #outmp3file = "../samples/sinestereo.mp3"
  #bitrate = 320
  
  
  # Read WAVE file and set MPEG encoder parameters.
  input_buffer = WavRead(inwavfile)
  params = EncoderParameters(input_buffer.fs, input_buffer.nch, bitrate)
  

  
  # Subband filter calculation from baseband prototype.
  # Very detailed analysis of MP3 subband filtering available at
  # http://cnx.org/content/m32148/latest/?collection=col11121/latest

  # Read baseband filter samples
  """
  ASSIGNMENT 2
  """
  baseband_filter = assignment2.prototype_filter().astype('float32')

  subband_samples = np.zeros((params.nch, N_SUBBANDS, FRAMES_PER_BLOCK), dtype='float32') 

  # Main loop, executing until all samples have been processed.
  while input_buffer.nprocessed_samples < input_buffer.nsamples:

    # In each block 12 frames are processed, which equals 12x32=384 new samples per block.
    for frm in range(FRAMES_PER_BLOCK):
      samples_read = input_buffer.read_samples(SHIFT_SIZE)

      # If all samples have been read, perform zero padding.
      if samples_read < SHIFT_SIZE:
        for ch in range(params.nch):
          input_buffer.audio[ch].insert(np.zeros(SHIFT_SIZE - samples_read))

      # Filtering = dot product with reversed buffer.
      """
      ASSIGNMENT 3 : Subband filtering
      """
      for ch in range(params.nch):
        subband_samples[ch,:,frm] = assignment3.subband_filtering(input_buffer.audio[ch].reversed(), baseband_filter)
      
    # Declaring arrays for keeping table indices of calculated scalefactors and bits allocated in subbands.
    # Number of bits allocated in subband is either 0 or in range [2,15].
    scfindices = np.zeros((params.nch, N_SUBBANDS), dtype='uint8')
    subband_bit_allocation = np.zeros((params.nch, N_SUBBANDS), dtype='uint8') 
    smr = np.zeros((params.nch, N_SUBBANDS), dtype='float32')

    
    # Finding scale factors, psychoacoustic model and bit allocation calculation for subbands. Although 
    # scaling is done later, its result is necessary for the psychoacoustic model and calculation of 
    # sound pressure levels.
    for ch in range(params.nch):
      scfindices[ch,:] = get_scalefactors(subband_samples[ch,:,:], params.table.scalefactor)
      subband_bit_allocation[ch,:] = psycho.model1(input_buffer.audio[ch].ordered(), params,scfindices)

    """
    ASSIGNMENT 4 : Quantization
    """
    subband_samples_quantized = np.zeros(subband_samples.shape, dtype='uint32')
    for ch in range(params.nch):
      for sb in range(N_SUBBANDS):
        QCa = params.table.qca[subband_bit_allocation[ch,sb]-2]
        QCb = params.table.qcb[subband_bit_allocation[ch,sb]-2]
        scf = params.table.scalefactor[scfindices[ch,sb]]
        ba = subband_bit_allocation[ch,sb]
        for ind in range(FRAMES_PER_BLOCK):
          subband_samples_quantized[ch,sb,ind] = assignment4.quantization(subband_samples[ch,sb,ind], scf, ba, QCa, QCb)


    # Forming output bitsream and appending it to the output file.
    bitstream_formatting(outmp3file,
                         params,
                         subband_bit_allocation,
                         scfindices,
                         subband_samples_quantized)


if __name__ == "__main__":
  if len(sys.argv) < 3:
    sys.exit('Please provide input WAVE file and desired bitrate.')
  inwavfile = sys.argv[1]
  if len(sys.argv) == 4:
    outmp3file = sys.argv[2]
    bitrate    = int(sys.argv[3])
  else:
    outmp3file = inwavfile[:-3] + 'mp3'
    bitrate    = int(sys.argv[2])

  if os.path.exists(outmp3file):
    sys.exit('Output file already exists.')

  main(inwavfile,outmp3file,bitrate)
