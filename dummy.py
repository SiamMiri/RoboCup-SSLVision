<<<<<<< HEAD
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
=======
from math import acos
from math import sqrt
from math import pi
import can

def length(v):
    return sqrt(v[0]**2+v[1]**2)
def dot_product(v,w):
   return v[0]*w[0]+v[1]*w[1]
def determinant(v,w):
   return v[0]*w[1]-v[1]*w[0]
def inner_angle(v,w):
   cosx=dot_product(v,w)/(length(v)*length(w))
   rad=acos(cosx) # in radians
   return rad*180/pi # returns degrees
def angle_clockwise(A, B):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else: # if the det > 0 then A is immediately clockwise of B
        return 360-inner
        
def send_one():
    """Sends a single message."""
    # this uses the default configuration (for example from the config file)
    with can.interface.Bus() as bus:
        bus = can.interface.Bus(bustype='pcan', channel='PCAN_USBBUS1', bitrate=250000)
        msg = can.Message(arbitration_id=0xC0FFEE, data=[0, 25, 0, 1, 3, 1, 4, 1], is_extended_id=True)

        try:
            bus.send(msg)
            print(f"Message sent on {bus.channel_info}")
        except can.CanError:
            print("Message NOT sent")

send_one()
>>>>>>> bd61e1a7f4ee447e26f0532ed44c58a99db19f81
