import matplotlib.pyplot as pl
from scipy.io import loadmat 
import numpy as np
import time

from mpl_toolkits.mplot3d import Axes3D


def getCoords( data, i ):
  s = data[:, np.arange( 3 ) + 3 * i[0]]
  e = data[:, np.arange( 3 ) + 3 * i[1]]
  w = data[:, np.arange( 3 ) + 3 * i[2]]
  
  sd = np.copy( s - e )
  wd = np.copy( w - e )
  
  return s, e, w, sd, wd

def getAngles( data, i ):
  _, _, _, sd, wd = getCoords( data, i )
  
  dot = ( wd * sd ).sum( 1 )
  wn = np.sum( np.sqrt( wd ** 2.0 ), 1 )
  sn = np.sum( np.sqrt( sd ** 2.0 ), 1 )
  
  return np.arccos( dot / ( wn * sn ) )



if __name__ == "__main__":
  filename = r"C:\workspace\matlab\kinect\datas.mat"
  data = loadmat( filename, squeeze_me = True )["datas"]
  
  data = data[:250]
  
  which = [4, 5, 6]
  s, e, w, sd, wd = getCoords( data, which )
  angles = getAngles( data, which )
  
  
  fig = pl.figure()
  ax1 = fig.add_subplot( 111, projection = '3d' )
  fig = pl.figure()
  ax2 = fig.add_subplot( 111, projection = '3d' )
  fig = pl.figure()
  ax3 = fig.add_subplot( 111 )
  
  
  
  minr = -0.5
  maxr = 0.5
  r = np.arange( len( data ) )
  
  
  alphas = ( angles - angles.min() ) / ( angles.max() - angles.min() )
  pl.sca( ax1 )
  pl.cla()
  pl.plot( s[r, 0], s[r, 2], s[r, 1], c = 'b', marker = "o" )
  pl.plot( e[r, 0], e[r, 2], e[r, 1], c = 'g', marker = "o" )
  pl.plot( w[r, 0], w[r, 2], w[r, 1], c = 'r', marker = "o" )
  ax1.set_xlim( minr, maxr )
  ax1.set_ylim( minr, maxr )
  ax1.set_zlim( minr, maxr )
  
  
  
  pl.sca( ax2 )
  pl.cla()
  pl.plot( sd[r, 0], sd[r, 2], sd[r, 1], c = 'b', marker = "o" )
  pl.plot( wd[r, 0], wd[r, 2], wd[r, 1], c = 'r', marker = "o" )
  ax1.set_xlim( minr, maxr )
  ax1.set_ylim( minr, maxr )
  ax1.set_zlim( minr, maxr )
  
  
  
  pl.sca( ax3 )
  pl.plot( angles )
  
  
  
  pl.show()
