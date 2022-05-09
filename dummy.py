import math
from math import atan2
import time

test1 = False
if test1:
    class MyObject:
        def __init__(self, a, b):
            self.x  = a
            self.y  = b
        
        def __index__(self):
            return self.x +  self.y

    obj = MyObject(10, 30)
    print(bin(obj))
    a = bin(486269830)
    b = bin(421234)
    print(a)
    print(b)
    a = a + b[2:]
    print(a)
    a = int(a, 2)
    print(a)
    
myradians = math.atan2(5-11, 18-11)
deg = math.degrees(myradians)

print(myradians)
print(deg)

t0= time.clock()
print("HI")
t= time.clock() - t0
print(t)