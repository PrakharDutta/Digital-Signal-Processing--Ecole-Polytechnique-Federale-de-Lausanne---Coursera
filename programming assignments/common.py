import numpy as np
import sys
import struct
import os.path



FRAME_SIZE = 512
FFT_SIZE   = 512
N_SUBBANDS =  32
SHIFT_SIZE =  32
SLOT_SIZE  =  32
FRAMES_PER_BLOCK = 12

EPS = 1e-6
INF = 123456




class WavRead:
  """Read WAVE or PCM file into a circular buffer. Only standard PCM WAVE supported for now."""
  

  def __init__(self, filename, fs=0, nch=0, nbits=0):
    """Open file and read header information."""
    
    if filename[-3:] == 'pcm':
      if fs == 0 or nch == 0 or nbits == 0:
        sys.exit('Please provide sampling frequency, number of channels \
                  and number of bits per sample for PCM audio file.')
      self.fs  = fs
      self.nch = nch
      self.nbits = nbits
      self.nsamples = os.path.getsize(filename) * 8 / self.nbits / self.nch
    
    self.filename = filename
    self.fp = open(self.filename, 'r')
    
    if filename[-3:] == 'wav':
      self.read_header()
      
    if self.nbits == 8:
      self.datatype = 'int8'
    elif self.nbits == 16:
      self.datatype = 'int16'
    else:
      self.datatype = 'int32'
    
    self.nprocessed_samples = 0
    self.audio = []
    for ch in range(self.nch):
      self.audio.append(CircBuffer(FRAME_SIZE))


  def read_header(self):
    """Read header information and determine if it is a valid MP3 file with PCM audio samples."""
    
    buffer = self.fp.read(128)
    ind = buffer.find('RIFF')
    if ind == -1:
      sys.exit('Bad WAVE file.')
    ind += 4
    self.chunksize = struct.unpack('<I', buffer[ind:ind+4])[0]
    ind = buffer.find('WAVE')
    if ind == -1:
      sys.exit('Bad WAVE file.')
    ind = buffer.find('fmt ')
    if ind == -1:
      sys.exit('Bad WAVE file.')

    ind += 4
    sbchk1sz = struct.unpack('<I', buffer[ind:ind+4])[0]
    if sbchk1sz != 16:
      sys.exit('Unsupported WAVE file, compression used instead of PCM.')
    ind += 4
    audioformat = struct.unpack('<H', buffer[ind:ind+2])[0]
    if audioformat != 1:
      sys.exit('Unsupported WAVE file, compression used instead of PCM.')
    ind += 2
    self.nch = struct.unpack('<H', buffer[ind:ind+2])[0]
    ind += 2
    self.fs  = struct.unpack('<I', buffer[ind:ind+4])[0]
    ind += 4
    self.byterate = struct.unpack('<I', buffer[ind:ind+4])[0]
    ind += 4
    self.blockalign = struct.unpack('<H', buffer[ind:ind+2])[0]
    ind += 2
    self.nbits = struct.unpack('<H', buffer[ind:ind+2])[0]
    if not (self.nbits in (8,16,32)):
      sys.exit('Unsupported WAVE file, samples not int8, int16 or int32 type.')
    ind = buffer.find('data')
    if ind == -1:
      sys.exit('Unsupported WAVE file, "data" keyword not found in file.')
    ind += 4
    sbchk2sz = struct.unpack('<I', buffer[ind:ind+4])[0]
    self.nsamples = sbchk2sz * 8 / self.nbits / self.nch
    self.fp.seek(ind+4)
    

  def read_samples(self, nsamples):
    """Read desired number of samples from WAVE file and insert it in circular buffer."""
    
    readsize = self.nch * nsamples
    frame = np.fromfile(self.fp, self.datatype, readsize)
    frame.shape = (-1, self.nch)
    for ch in range(self.nch):
      self.audio[ch].insert(frame[:,ch].astype('float32') / (1<<self.nbits-1))
    self.nprocessed_samples += frame.shape[0]
    return frame.shape[0]





class CircBuffer:
  """Circular buffer used for audio input."""
  
  def __init__(self, size, type='float32'):
    self.size = size
    self.pos  = 0
    self.samples = np.zeros(size, dtype=type)
  
  
  def insert(self, frame):
    length = len(frame)
    if self.pos + length <= self.size:
      self.samples[self.pos:self.pos+length] = frame
    else:
      overhead = length - (self.size - self.pos)
      self.samples[self.pos:self.size] = frame[:-overhead]
      self.samples[0:overhead] = frame[-overhead:]
    self.pos += length
    self.pos %= self.size
  def ordered(self):
    return np.concatenate((self.samples[self.pos:], self.samples[:self.pos]))
  def reversed(self):
    return np.concatenate((self.samples[self.pos-1::-1], self.samples[:self.pos-1:-1]))



  
  
  
class BitStream:
  """Form an array of bytes and fill it as a bitstream."""
  
  def __init__(self, size):
    """Initialize OutputBuffer with size in bytes."""
    
    self.size = size
    self.pos  = 0
    self.data = np.zeros(size, dtype='uint8')
  
  
  def insert(self, data, nbits, invmsb=False):
    """Insert lowest nbits of data in OutputBuffer."""
    
    if invmsb:
      data = self.invertmsb(data,nbits)
    datainbytes = self.splitinbytes(data, nbits, self.pos&0x7)
    ind = self.pos // 8
    for byte in datainbytes:
      if ind >= self.size:
        break
      self.data[ind] |= byte
      ind += 1
    self.pos += nbits 
    
    
  def maskupperbits(self,data,nbits):
    """Set all bits higher than nbits to zero."""
    mask = ~( (0xFFFFFFFF<<nbits) & 0xFFFFFFFF )
    return data&mask
  
  
  def invertmsb(self, data, nbits):
    """Invert MSB of data, data being only lowest nbits."""
    mask = 1<<(nbits-1)
    return data^mask


  def splitinbytes(self,data,nbits,pos):
    """Split input data in bytes to allow insertion in buffer by OR operation."""
    
    data = self.maskupperbits(data, nbits)
    shift = (8 - (nbits & 0x7) + 8 - pos) & 0x7
    data <<= shift
    nbits += shift
    datainbytes = ()
    loopcount = 1 + (nbits - 1) // 8
    for i in range(loopcount):
      datainbytes = (data & 0xFF,) + datainbytes 
      data >>= 8
    return datainbytes






def bitstream_formatting(filename, params, allocation, scalefactor, sample):
  """Form a MPEG-1 Layer 1 bitstream and append it to output file."""
  
  buffer = BitStream((params.nslots + params.padbit) * 4)

  buffer.insert(params.header, 32)
  params.updateheader()
  
  for sb in range(N_SUBBANDS):
    for ch in range(params.nch):
      buffer.insert(np.max((allocation[ch][sb]-1, 0)), 4)
  
  for sb in range(N_SUBBANDS):
    for ch in range(params.nch):
      if allocation[ch][sb] != 0:        
        buffer.insert(scalefactor[ch][sb], 6)

  for s in range(FRAMES_PER_BLOCK):
    for sb in range(N_SUBBANDS):
      for ch in range(params.nch):
        if allocation[ch][sb] != 0:
          buffer.insert(sample[ch][sb][s], allocation[ch][sb], True)
  
  fp = file(filename, 'a+')
  buffer.data.tofile(fp)
  fp.close()






def get_scalefactors(sbsamples,sftable):
  """Calculate scale factors for subbands. Scale factor is equal to the smallest number in the table 
  greater than all the subband samples in a particular subband. Scalefactor table indices are returned."""
  
  sfactorindices = np.zeros(sbsamples.shape[0:-1], dtype='uint8')
  sbmaxvalues = np.max(np.absolute(sbsamples), axis = 1)
  for sb in range(N_SUBBANDS):
    i = 0
    while sftable[i+1] > sbmaxvalues[sb]:
      i+=1
    sfactorindices[sb] = i
  return sfactorindices



def add_db(values):
  """Add power magnitude values."""
  
  powers = []
  for val in values:
    powers.append(np.power(10.0, val / 10.0))
  return 10 * np.log10( np.sum(powers) + EPS)
