STATES_TO_COORDINATES = [ # y 0 is at top
      (4,3), (4,2), (4,5), (4,6), (3,4), (2,4), (5,4), (6,4), #center plus
      (3,1), (2,1), (5,1), (6,1), # top perimeter
      (3,7), (2,7), (5,7), (6,7), # bottom perimeter
      (1, 3), (1,2), (1,5), (1,6), #left perimeter
      (7,3), (7,2), (7,5), (7,6), # right perimeter
      (7,1), (8,0), # top right
      (7,7), (8,8), # bottom right corner
      (1,7), (0,8), # bottom left corner 
      (1,1), (0,0) # top left corner
]

STATES_TO_ORIENTATIONS = [2, 0, 0, 2, 1, 3, 3, 1, # center plus
      1, 3, 3, 1, # top perimeter
      1, 3, 3, 1, # bottom perimeter
      2, 0, 0, 2, #left perimeter
      2, 0, 0, 2, # right perimeter
      2, 0, # top right
      0, 2, # bottom right corner
      0, 2, # bottom left corner
      2, 0 # top left corner
    ]
LEGAL_MOVEMENT = [ # forward, left, right, about face | 0 is not allowed
      [4, 8, 6, 2],
      [0, 10, 12, 1],
      [2, 6, 8, 4],
      [0, 16, 14, 3], #4
      [8, 2, 4, 6],
      [0, 20, 18, 5],
      [6, 4, 2, 8],
      [0, 22, 24, 7], #8
      [12, 0, 1, 10], #9
      [0, 17, 32, 9],
      [10, 1, 0, 12], #11
      [0, 26, 21, 11], # 12
      [16, 3, 0, 14],
      [0, 30, 19, 13],
      [14, 0, 3, 16],
      [0, 23, 28, 15], #16
      [20, 5, 0, 18],
      [0, 32, 9, 17],
      [18, 0, 5, 20],
      [0, 13, 30, 19], #20
      [24, 0, 7, 22],
      [0, 11, 26, 21],
      [22, 7, 0, 24],
      [0, 28, 15, 23], #24
      [0, 21, 11, 26],
      [0, 0, 0, 25],
      [0, 15, 23, 28],
      [0, 0, 0, 27], #28
      [0, 19, 13, 30],
      [0, 0, 0, 29],
      [0, 9, 17, 32],
      [0, 0, 0, 31], #32
    ]