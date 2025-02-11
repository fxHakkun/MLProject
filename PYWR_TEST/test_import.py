import sys
import pywr

if "sys" not in dir():
    print("sys not imported")    
if ('pywr' in sys.modules) and ('pywr' in dir()):
    print("Pywr exist")
else:
    print("No pywr founded")

x = "name"
while x == "name":
    x = [0,1,2,3,4]

print(x)