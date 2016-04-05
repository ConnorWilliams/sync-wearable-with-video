import sys

import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
import numpy as np
import gzip
import cPickle as pickle
from sklearn.decomposition import PCA

import joints as joints
from filter import smooth
import pre_processing as pre

def getCoords( data, measurement ):
  a = measurement[0]
  b = measurement[1]

  this1 = data[:, 7 + 4*a[0] + np.arange(1,4,1)]
  that1 = data[:, 7 + 4*a[1] + np.arange(1,4,1)]
  this2 = data[:, 7 + 4*b[0] + np.arange(1,4,1)]
  that2 = data[:, 7 + 4*b[1] + np.arange(1,4,1)]

  sd = np.copy( this1 - that1 )
  wd = np.copy( this2 - that2 )

  return this1, that1, this2, that2, sd, wd


def getAngles( data, measurement ):
  _, _, _, _, sd, wd = getCoords( data, measurement )

  dot = ( wd * sd ).sum( 1 )
  wn = np.sum( np.sqrt( wd ** 2.0 ), 1 )
  sn = np.sum( np.sqrt( sd ** 2.0 ), 1 )

  val = ( dot / ( wn * sn ) )
  return val


def drahSkeleton( example ):
  for n in joints.jointNeighbours:
    a = example[7 + 4*n[0] + np.arange(1,4,1)]
    b = example[7 + 4*n[1] + np.arange(1,4,1)]
    pl.plot( [a[0], b[0]], [a[2], b[2]], [a[1], b[1]], marker = "o", c = "k" )


def plotJointMovements( examples, jointMeasurements ):
  for measurement in jointMeasurements:
    _, _, _, _, sd, wd = getCoords( examples, measurement )

    for i in np.arange( len( examples ) ):
      pl.plot( [sd[i, 0], wd[i, 0]],
               [sd[i, 2], wd[i, 2]],
               [sd[i, 1], wd[i, 1]] )

def makeImage( data ):
    # Measure this joint's position relative to that joint's position:
    # [ (this1,that1), (this2,that2) ]
    jointMeasurements = [
      [( joints.ras, joints.rae ), ( joints.rah, joints.rae )], # Elbow joint
      [( joints.ras, joints.rae ), ( joints.torso, joints.neck )], # Elbow WRT torso
      [( joints.ras, joints.rah ), ( joints.torso, joints.neck )], # Wrist WRT torso
      [( joints.ras, joints.rae ), ( joints.ras, joints.neck )], # Elbow WRT neck
      [( joints.ras, joints.rah ), ( joints.ras, joints.neck )], # Wrist WRT neck
      [( joints.rah, joints.rae ), ( joints.neck, joints.head )],

      [( joints.las, joints.lae ), ( joints.rah, joints.rae )], # Elbow joint
      [( joints.las, joints.lae ), ( joints.torso, joints.neck )], # Elbow WRT torso
      [( joints.las, joints.lah ), ( joints.torso, joints.neck )], # Wrist WRT torso
      [( joints.las, joints.lae ), ( joints.ras, joints.neck )], # Elbow WRT neck
      [( joints.las, joints.lah ), ( joints.ras, joints.neck )], # Wrist WRT neck
      [( joints.lah, joints.lae ), ( joints.neck, joints.head )],

      [( joints.head, joints.neck ), ( joints.torso, joints.neck )], # Head WRT torso
      [( joints.head, joints.neck ), ( joints.torso, joints.ras )], # Head WRT neck
      [( joints.head, joints.neck ), ( joints.torso, joints.las )], # Head WRT neck
      [( joints.llh, joints.rlh ), ( joints.las, joints.ras )],
      [( joints.llh, joints.rlh ), ( joints.lae, joints.rae )],
      [( joints.llh, joints.rlh ), ( joints.lah, joints.rah )],
      [( joints.llh, joints.rlh ), ( joints.llk, joints.rlk )],
      [( joints.llh, joints.rlh ), ( joints.llf, joints.rlf )],

      [( joints.rlh, joints.rlk ), ( joints.rlf, joints.rlk )],
      [( joints.rlk, joints.rlf ), ( joints.rlf, joints.rlf )],

      [( joints.llh, joints.llk ), ( joints.llf, joints.llk )],
      [( joints.llk, joints.llf ), ( joints.llf, joints.llf )],
    ]

    angles = np.asarray( [getAngles( data, meas ) for meas in jointMeasurements] ).T

    plotwin = 50

    l = 1

    minx = data[:, 8::4].min()
    miny = data[:, 9::4].min()
    minz = data[:, 10::4].min()
    maxx = data[:, 8::4].max()
    maxy = data[:, 9::4].max()
    maxz = data[:, 10::4].max()

    fig = pl.figure( figsize = ( 12.5, 5 ) )
    ax_skeleton = fig.add_subplot( 1, 4, 1, projection = '3d' )

    axes = [
      ( fig.add_subplot( 2, 4, 2 ), np.arange( 6 ) + 0 ),
      ( fig.add_subplot( 2, 4, 4 ), np.arange( 6 ) + 6 ),
      ( fig.add_subplot( 1, 4, 3 ), np.arange( 8 ) + 12 ),
      ( fig.add_subplot( 2, 4, 6 ), np.arange( 2 ) + 20 ),
      ( fig.add_subplot( 2, 4, 8 ), np.arange( 2 ) + 22 ),
    ]

    for i in np.arange( plotwin / 2, len( data ) - plotwin / 2 ):
      ii = i - plotwin / 2

      r = np.arange( i - l, i )
      sys.stdout.write("\r%i of %i" % (ii, len(data)) )
      sys.stdout.flush()

      ax_skeleton.cla()
      pl.sca( ax_skeleton )
      for j in r:
        drahSkeleton( data[j] )
      ax_skeleton.auto_scale_xyz( [minx, maxx], [miny, maxy], [minz, maxz] )
      ax_skeleton.xaxis.set_ticklabels( [] )
      ax_skeleton.yaxis.set_ticklabels( [] )
      ax_skeleton.zaxis.set_ticklabels( [] )

      plotRange = np.arange( i - plotwin / 2, i + plotwin / 2 )

      for ax, which in axes:
        ax.cla()

        x = plotRange
        y = angles[plotRange]
        y = y[:, which]

        ax.plot( x, y )

        ax.plot( [i, i], [-1, 1], 'k' )

        ax.set_xlim( i - plotwin / 2, i + plotwin / 2 )
        #ax.set_ylim( y.min(), y.max() )
        ax.set_ylim( -1, 1 )

        ax.xaxis.set_ticklabels( [] )
        ax.yaxis.set_ticklabels( [] )

      fig.tight_layout()


      pl.savefig( "out/%05d.png" % ( ii ) )
