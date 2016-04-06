head = 0
neck = 1
las = 2
ras = 3
lae = 4
rae = 5
lah = 6
rah = 7
torso = 8
llh = 9
rlh = 10
llk = 11
rlk = 12
llf = 13
rlf = 14

joints = dict(
  head = head,
  neck = neck,
  torso = torso,
  las = las, ras = ras,
  lae = lae, rae = rae,
  lah = lah, rah = rah,
  llh = llh, rlh = rlh,
  llk = llk, rlk = rlk,
  llf = llf, rlf = rlf )

jointNeighbours = [
  (0, 1),
  (1, 2), (2, 4), (4, 6),
  (1, 3), (3, 5), (5, 7),
  (1, 8),
  (8, 9), (9, 11), (11, 13),
  (8, 10), (10, 12), (12, 14)
]

# Measure this joint's position relative to that joint's position:
# [ (this1,that1), (this2,that2) ]
measurements = [
  [( ras, rae ), ( rah, rae )],
  [( ras, rae ), ( torso, neck )],
  [( ras, rah ), ( torso, neck )],
  [( ras, rae ), ( ras, neck )],
  [( ras, rah ), ( ras, neck )],
  [( rah, rae ), ( neck, head )],

  [( las, lae ), ( rah, rae )],
  [( las, lae ), ( torso, neck )],
  [( las, lah ), ( torso, neck )],
  [( las, lae ), ( ras, neck )],
  [( las, lah ), ( ras, neck )],
  [( lah, lae ), ( neck, head )],

  [( head, neck ), ( torso, neck )],
  [( head, neck ), ( torso, ras )],
  [( head, neck ), ( torso, las )],
  [( llh, rlh ), ( las, ras )],
  [( llh, rlh ), ( lae, rae )],
  [( llh, rlh ), ( lah, rah )],
  [( llh, rlh ), ( llk, rlk )],
  [( llh, rlh ), ( llf, rlf )],

  [( rlh, rlk ), ( rlf, rlk )],
  [( rlk, rlf ), ( rlf, rlf )],

  [( llh, llk ), ( llf, llk )],
  [( llk, llf ), ( llf, llf )],
]
