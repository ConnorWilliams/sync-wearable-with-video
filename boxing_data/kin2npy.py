import numpy as np
import gzip 
import cPickle as pickle


name = "kinect2"

lines = open( "%s.txt" % name ).readlines()

d = []
for i in np.arange( len( lines ) ):
  line = lines[i].strip()
  if len( line ) < 10: 
    continue 
  
  line = map( np.float, line.split() )[1:]
  del line[0::4]
  
  line = np.asarray( line ) - line[:3] * ( len( line ) / 3 )
  d.append( line )

d = np.asarray( d )

print d.shape
pickle.dump( d, gzip.open( "%s.pkl.gz" % name, "wb" ), protocol = -1 )

