from pylwpr import *
import numpy as np
from random import *
import math

def cross_2D(x1,x2): 
   return float(max(math.exp(-10.0*x1*x1), math.exp(-50.0*x2*x2), 
               1.25*math.exp(-5.0*(x1*x1 + x2*x2))))

model = LWPR(2,1)

R = Random()
x = np.zeros(2)
y = np.zeros(1)

model.init_D = 50*np.eye(2)
model.init_alpha = 250*np.ones([2,2])

for i in range(10000):
   x[0] = R.uniform(-1,1)
   x[1] = R.uniform(-1,1)
   y[0] = cross_2D(x[0],x[1]) + R.gauss(0,0.1)
   model.update(x,y)   

for x[0] in np.arange(-1,1,0.04):
   for x[1] in np.arange(-1,1,0.04):
      y = model.predict(x)
      print("%6.2f %6.2f %8.4f"%(x[0],x[1],y[0]))


print(model)
print(model.kernel)

model.write_XML("cross2d.xml")
