import numpy as np 
import pandas as pd
from scipy.optimize import fsolve, minimize  
import matplotlib.pyplot as pl
from scipy.interpolate import griddata,CubicSpline
from matplotlib.ticker import FormatStrFormatter
import open3d as o3d

gamma = 1.4

# Utility Functions
def machAngle (M):
    if M>=1:
        return (np.arcsin(1/M)*180/np.pi).item()
    else:
        return None
    
def prandtlMeyerAngle (M,gamma): 
    if M>=1:
        return ((np.sqrt((gamma+1)/(gamma-1))*np.arctan(np.sqrt((gamma-1)/(gamma+1)*(M**2-1))) - np.arctan(np.sqrt(M**2-1)))*180/np.pi).item()
    else:
        return None
    
def machFromPrandtlMeyerAngle (nu,gamma):
    M = fsolve(lambda M: nu - prandtlMeyerAngle(M,gamma), 1)
    return M.item()

def maxWallAngleMinLength (Mexit,gamma):
    theta = prandtlMeyerAngle(Mexit,gamma)/2
    return theta

def AbyAcrit (M,gamma):
    return 1/M*np.power(2/(gamma+1)*(1+(gamma-1)/2*M**2),(gamma+1)/2/(gamma-1))

def TbyT0 (M,gamma):
    return 1/(1 + (gamma-1)/2*M**2)

def PbyP0 (M,gamma):
    return np.power(1 + (gamma-1)/2*M**2,-gamma/(gamma-1))

# Object to store the numerical details of each grid point
class Point:
    
    def __init__(self, id_):
        self.id = id_
    
    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"
    
    def setCoordinates(self,x_,y_):
        self.x = x_
        self.y = y_
        
    def setTheta(self,theta_):
        self.theta = theta_
        
    def setNu(self,nu_):
        self.nu = nu_
        self.M = machFromPrandtlMeyerAngle(self.nu,gamma)
        
    def setM(self,M_):
        self.M = M_
        self.nu = prandtlMeyerAngle(self.M,gamma)
        
    def updateCharacteristics(self):
        self.Kplus = self.theta - self.nu
        self.Kminus = self.theta + self.nu
        self.mu = machAngle(self.M)

    def print(self):
        print("\nPoint ID : ",self.id)
        print("M = ",self.M)
        print("theta = ",self.theta)
        print("nu = ",self.nu)
        print("mu = ",self.mu)
        print("(x,y) = (",self.x,",",self.y,")")
        print("K+ = ",self.Kplus)
        print("K- = ",self.Kminus)

# Object to store the numerical details for each characteristic
class characteristic:
    
    def __init__(self,id_):
        self.id = id_
        self.Points = []
        self.nPoints = 0
        
    def __repr__(self):
        return f"Char(id={self.id}, nPoints={self.nPoints})"
    
    def addPoint(self,Point_):
        self.Points.append(Point_)
        self.nPoints = len(self.Points)
    
    def setK(self,K_):
        self.K = K_
    
    def print(self):
        print("\nChar ID = ",self.id)
        print("nPoints = ",self.nPoints)
        print("K = ",self.K)
        
    def printDetailed(self):
        print("\nChar ID = ",self.id)
        print("nPoints = ",self.nPoints)
        print("K = ",self.K)
        
        for j in range(len(self.Points)):
            self.Points[j].print()
    
    def returnCoordinates(self):
        X = []
        Y = []
        for j in range(0,len(self.Points)):
            X.append(self.Points[j].x)
            Y.append(self.Points[j].y)
        
        return X,Y
    def plot(self): 
        X = []
        Y = []
        for j in range(0,len(self.Points)):
            X.append(self.Points[j].x)
            Y.append(self.Points[j].y)

        pl.plot(X,Y)
        pl.show()   

# Unit Processes
def getInternalIntersectionPoint (Point_minus, Point_plus,id):
    intxPoint = Point(id)
    intxPoint.setTheta(0.5*(Point_minus.Kminus + Point_plus.Kplus))
    intxPoint.setNu(0.5*(Point_minus.Kminus - Point_plus.Kplus))
    intxPoint.updateCharacteristics()
    
    thetaMinusMean = 0.5*(Point_minus.theta + intxPoint.theta)
    muMinusMean = 0.5*(Point_minus.mu + intxPoint.mu)
    
    thetaPlusMean = 0.5*(Point_plus.theta + intxPoint.theta)
    muPlusMean = 0.5*(Point_plus.mu + intxPoint.mu)
    
    slopeCminus = np.tan(np.pi/180*(thetaMinusMean-muMinusMean))
    slopeCplus = np.tan(np.pi/180*(thetaPlusMean+muPlusMean))
    
    A = np.array([[slopeCminus, -1],[slopeCplus, -1]])
    b = np.array([Point_minus.x*slopeCminus-Point_minus.y, Point_plus.x*slopeCplus-Point_plus.y])
    
    X = np.linalg.solve(A,b)
    intxPoint.setCoordinates(X[0],X[1])
    return intxPoint
    
def getCenterlinePoint (Point_minus,id):
    centerlinePoint = Point(id)
    centerlinePoint.setTheta(0)
    centerlinePoint.setNu(Point_minus.Kminus)
    centerlinePoint.updateCharacteristics()
    
    thetaMinusMean = 0.5*(Point_minus.theta + centerlinePoint.theta)
    muMinusMean = 0.5*(Point_minus.mu + centerlinePoint.mu)
    
    slopeCminus = np.tan(np.pi/180*(thetaMinusMean-muMinusMean))
    
    xw = Point_minus.x - Point_minus.y/slopeCminus
    centerlinePoint.setCoordinates(xw,0)
    return centerlinePoint
    
def getWallPoint (Point_wall, Point_plus, id):
    wallPoint = Point(id)
    wallPoint.setTheta(Point_plus.theta)
    wallPoint.setNu(Point_plus.nu)
    wallPoint.updateCharacteristics()
    
    thetaWallMean = 0.5*(Point_wall.theta + wallPoint.theta)
    
    thetaPlusMean = 0.5*(Point_plus.theta + wallPoint.theta)
    muPlusMean = 0.5*(Point_plus.mu + wallPoint.mu)
    
    slopeCminus = np.tan(np.pi/180*(thetaWallMean))
    slopeCplus = np.tan(np.pi/180*(thetaPlusMean+muPlusMean))
    
    A = np.array([[slopeCminus, -1],[slopeCplus, -1]])
    b = np.array([Point_wall.x*slopeCminus-Point_wall.y, Point_plus.x*slopeCplus-Point_plus.y])
    
    X = np.linalg.solve(A,b)
    wallPoint.setCoordinates(X[0],X[1])
    
    return wallPoint

gamma = 1.4  
t = 1       # Throat half-width      
Mexit = 2.1 # Exit mach number
thetaWallMax = maxWallAngleMinLength(Mexit,gamma)
theta0 = 0.1 # Angle of first characteristic
nChar = 30   # Number of characteristics
thetaExpFan = np.linspace(start=theta0,stop=thetaWallMax,num=nChar,endpoint=True)	# Angles of all characteristics

# Initialize Characteristics

Chars = []
CenterlinePoints = []
for i in range(0,nChar):
    
    Chars.append(characteristic(i))
    Chars[i].addPoint(Point(0))
    Chars[i].Points[0].setCoordinates(0,t)
    Chars[i].Points[0].setTheta(thetaExpFan[i])
    Chars[i].Points[0].setNu(thetaExpFan[i])
    Chars[i].Points[0].updateCharacteristics()
    Chars[i].setK(Chars[i].Points[0].Kminus)


# Solve for first intersection point on each characteristic
i = 0

Chars[i].addPoint(getCenterlinePoint(Chars[i].Points[i],i+1))
CenterlinePoints.append(Chars[i].Points[-1])
for j in range(i+2,nChar+1):
    Chars[i].addPoint(getInternalIntersectionPoint(Chars[j-1].Points[i],Chars[i].Points[j-1],j))    

Chars[i].addPoint(getWallPoint(Chars[-1].Points[0],Chars[i].Points[nChar],nChar+1))
Chars[i].Points[1].setTheta(Chars[i].Points[0].theta)
Chars[i].Points[1].setNu(Chars[i].Points[0].nu)
Chars[i].Points[1].updateCharacteristics()

# Solve for subsequent intersection points
for i in range(1,nChar):
    
    for j in range(1,i+1):
        Chars[i].addPoint(Chars[j-1].Points[i+1])
    
    Chars[i].addPoint(getCenterlinePoint(Chars[i].Points[i],i+1))
    CenterlinePoints.append(Chars[i].Points[-1])
    
    for j in range(i+2,nChar+1):
        Chars[i].addPoint(getInternalIntersectionPoint(Chars[i-1].Points[j],Chars[i].Points[j-1],j))

    Chars[i].addPoint(getWallPoint(Chars[i-1].Points[nChar+1],Chars[i].Points[nChar],nChar+1)) 


# Get Nozzle Contour
nozzleX = np.array([])
nozzleY = np.array([])
nozzleX = np.append(nozzleX,0)
nozzleY = np.append(nozzleY,t)
for j in range(0,nChar):
    nozzleX = np.append(nozzleX,Chars[j].Points[-1].x)
    nozzleY = np.append(nozzleY,Chars[j].Points[-1].y)
    
contour = CubicSpline(nozzleX,nozzleY)
x_ = np.linspace(0,nozzleX.max(),100)
y_ = contour(x_)
nozzleLength = nozzleX.max()

'''
# Save for 3D modeling
df = pd.DataFrame({'x': x_, 'y': y_})
df['z'] = 0

# Convert DataFrame to numpy array for Open3D
points = np.array(df)  # Adjust column names as needed

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Save as .pcd file
o3d.io.write_point_cloud('nozzleContour.pcd', pcd)  # Replace with desired filename
# Write to CSV
df.to_csv('nozzleContour.csv', index=False)
'''
'''
pl.plot(x_,y_,color='black')
pl.ylim((0,3))
pl.xlim((0,x_[-1]))
pl.show()
'''


'''
# Plot Characteristics
for j in range(0,nChar):
    X,Y = Chars[j].returnCoordinates()
    pl.plot(X,Y,color='black')


# Plot nozzle
X = []
Y = []
X.append(0)
Y.append(t)
for j in range(0,nChar):
    X.append(Chars[j].Points[-1].x)
    Y.append(Chars[j].Points[-1].y)

pl.plot(X,Y,color='black')
pl.ylim((0,3))
pl.xlim((0,X[-1]))
pl.xlabel(r'x',fontsize=14)
pl.ylabel(r'y',fontsize=14)
pl.savefig("Characteristics.png",dpi=720)
pl.show()
'''
'''
# Plot Centerline Mach Number
X = []
M = []
for j in range(0,nChar):
    X.append(Chars[j].Points[j+1].x)
    M.append(Chars[j].Points[j+1].M)

pl.plot(X,M,color='black')
pl.ylim((0,3))
pl.xlim((0,X[-1]))
pl.show()
'''



# Mach number contours
X = np.array([])
Y = np.array([])
M = np.array([])


for j in range(0,nChar):
    
    for i in range(Chars[j].nPoints):
        X = np.append(X,[Chars[j].Points[i].x])
        Y = np.append(Y,[Chars[j].Points[i].y])
        M = np.append(M,[Chars[j].Points[i].M])



# Create a grid for x and y
xi = np.linspace(X.min(), X.max(), 1000)
yi = np.linspace(Y.min(), Y.max(), 1000)
x, y = np.meshgrid(xi, yi)
m = griddata((X, Y), M, (x, y), method='linear')


y_nozzle = contour(x)

Ae_th = AbyAcrit(Mexit,gamma)
Ae_moc = Y.max()
err = np.abs(Ae_moc-Ae_th)/Ae_th*100
print("Nozzle Length = ",nozzleLength)
print("Area Ratio = ",Ae_moc)
print("Error in exit area = ",err,"%")

m[(x>CenterlinePoints[-1].x) & (y<y_nozzle) & (np.isnan(m))] = Mexit + 1e-6
m[(x<CenterlinePoints[0].x) & (y<y_nozzle) & (np.isnan(m))] = 1 - 1e-6

t = TbyT0(m,gamma)
p = PbyP0(m,gamma)

'''
# Plot Mach contours
contourplot = pl.contourf(x,y,m,levels=1000, cmap="viridis")
pl.ylim([0,3])
pl.xlabel(r'x',fontsize=14)
pl.ylabel(r'y',fontsize=14)
cbar = pl.colorbar(contourplot)
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
cbar.set_label(r'Mach Number',fontsize=14)
pl.savefig("MachNumber.png",dpi=720)
pl.show()
'''

'''
# Plot Temp contours
contourplot = pl.contourf(x,y,t,levels=1000, cmap="viridis")
pl.ylim([0,3])
pl.xlabel(r'x',fontsize=14)
pl.ylabel(r'y',fontsize=14)
cbar = pl.colorbar(contourplot)
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
cbar.set_label(r'$\frac{T}{T_0}$',fontsize=20)
pl.savefig("Temperature.png",dpi=720)
pl.show()
'''

'''
# Plot Pressure contours
contourplot = pl.contourf(x,y,p,levels=1000, cmap="viridis")
pl.ylim([0,3])
pl.xlabel(r'x',fontsize=14)
pl.ylabel(r'y',fontsize=14)
cbar = pl.colorbar(contourplot)
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
cbar.set_label(r'$\frac{P}{P_0}$',fontsize=20)
pl.savefig("Pressure.png",dpi=720)
pl.show()
'''
