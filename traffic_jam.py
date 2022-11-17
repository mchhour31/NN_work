import numpy as np

num_rows = 8
num_cols = 8

s = 0
w = -1
e = -2

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
alphabet = [i.upper() for i in alphabet]

def GenerateGrid(num_rows, num_cols):
    arr = [[0] * num_rows for _ in range(num_cols)]
    
    for i in range(num_rows):
        for j in range(num_cols):   
            if (i == 0 or j == 0 or i == num_rows-1 or j == num_cols-1):
                arr[i][j] = -1
    
    return arr

def InitialiseRoad(num_rows, num_cols, side, pos):
    g = GenerateGrid(num_rows, num_cols)
    
    if (side == 'N'):
        r, c = 0, pos
    elif (side == 'E'):
        r, c = pos, num_cols-1
    elif (side == 'W'):
        r, c = pos, 0
    elif (side == 'S'):
        r, c = num_rows-1, pos
        
    g[r][c] = -2
     
    return g

def PrintRoad(road):
    replacements = {-1: '#', -2: 'O', 0: ' '}
    replacer = replacements.get

    new_road = [[replacer(n,n) for n in i] for i in road]
    
    for i in range(len(new_road)):
        print(new_road[i], end='\n')
    
    return new_road

def PercentUsed(road, num_rows, num_cols):
    c = 0
    total = (num_rows-2) * (num_cols-2)
    
    for i in range(1, num_rows-1):
        for j in range(1, num_cols-1):
            if (road[i][j] != 0):
                c += 1

    return (c / total) * 100

def AddCar(road, row, col, size):
    if size < 0:
        size = size * -1
        
        for i in range(row, row+size):
            road[i][col] = alphabet[0]

        alphabet.pop(0)
        
    else:
        # horizontal
        x = 1
        i = col
        
        while i < col+size and x:
            if (road[row][i]):
                x = 0
            
            i += 1

        if x:
            for i in range(col, col+size):
                road[row][i] = alphabet[0]
                
            alphabet.pop(0)
                
    return road

def FindCar(road, move):
    n1, m1, n2, m2 = -1, -1, -1, -1
    for i in range(len(road)):
        for j in range(len(road[0])):
            if (road[i][j] == move):
                if (i > n1 and j > m1):
                    n1, m1 = i, j
                    
                n2, m2 = i, j # stores max
    
    return n1, m1, n2, m2
    
def MoveCar(road, r0, c0, r1, c1):
    

road = InitialiseRoad(num_rows, num_cols, 'E', 3)
AddCar(road, 3, 1, 2)
AddCar(road, 2, 4, -4)
AddCar(road, 5, 3, 3)
AddCar(road, 6, 3, 3)
# AddCar(road, 1, 1, 6); 
# AddCar(road, 2, 1, 6); 
# AddCar(road, 3, 1, 6); 
# AddCar(road, 4, 1, 6);

PrintRoad(road)
print(f"Percent used: {np.round(PercentUsed(road, num_rows, num_cols), decimals=3)}")

# rowA, colA, rowB, colB = 0,0,0,0
move = ['A', 'B', 'C']

for i in move:
    rowA, colA, rowB, colB = FindCar(road, i)
    FindCar(road, i)
    print(f"Car {i} is at: ({rowA}, {colA}) - ({rowB}, {colB})")   

result = MoveCar(road, rowA, colA, rowB, colB)
print(f"Result = {result}")