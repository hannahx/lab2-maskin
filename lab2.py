from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy.matlib
import numpy, pylab, random, math

print('HEJSAN')

#Generate data
'''
classA = [
    (random.normalvariate(-1.5,1),
    random.normalvariate(0.5,1),1.0)
    for i in range(5)] + \
    [(random.normalvariate(-1.5,1),
    random.normalvariate(0.5,1),1.0)
    for i in range(5)]

classB = [
    (random.normalvariate(0,0.5),
    random.normalvariate(-0.5,0.5),-1.0)
    for i in range(10)]
'''
classA = [(-2.2918553716277232, -0.220373700069746, 1.0), (-1.6521429959208507, 1.1898562336375478, 1.0), (-0.9930212086690492, 1.8551197704218274, 1.0), (-2.630851407030102, 0.3947058246630847, 1.0), (-1.9015617230776474, -0.556197491010018, 1.0), (-0.7436191971237376, 1.336953380545586, 1.0), (0.6429206814079937, 0.4867835697497979, 1.0), (-2.0981920079452494, 0.5404648113730255, 1.0), (-3.1928947974732775, 1.336470368555728, 1.0), (-2.811779647728428, -0.09789640010189316, 1.0)]
classB = [(0.3466793457533669, -0.7011024366228709, -1.0), (0.10808278098344137, -0.8939389525527817, -1.0), (-0.7502236939383479, -1.944038094672238, -1.0), (-1.310776109385966, -0.8091655706399122, -1.0), (0.2873341206145037, -0.592819846134157, -1.0), (0.04036977645739818, -0.44464951421821886, -1.0), (-0.20975952221026842, -0.2585136551250332, -1.0), (0.03582223600757029, 0.31168261180831029, -1.0), (0.441070903791048, -0.7125794299083623, -1.0), (-0.4606224652860254, -1.0776103556215242, -1.0)]

#classA = [(-1.1059544059824913, 1.249463874815591, 1.0), (-0.9341647951230858, 1.8084939265599926, 1.0), (-0.4801219266009751, 1.9657415569751924, 1.0), (-1.5024801050563532, 1.3028263024015283, 1.0), (-0.8269696841063974, -0.10940734166726718, 1.0), (-2.402428096626385, 0.8982921996349074, 1.0), (-1.0817893527733076, 0.7342075890074069, 1.0), (-1.7635891298829531, 0.13920515135610334, 1.0), (-2.380834942367402, 0.539172058054223, 1.0), (-1.6895415203441642, -0.06512302524271918, 1.0)]
#classB = [(-0.557374617830797, -0.1477799803713078, -1.0), (0.70614835350012, -1.0106613999139664, -1.0), (-0.5489631421669352, -0.686640166555052, -1.0), (0.05888771973585447, -0.2772254819731478, -1.0), (0.8859889790387119, 0.3409965855539733, -1.0), (0.04912511663260322, -0.3214891631010094, -1.0), (0.6826480625729144, -0.43238872966362624, -1.0), (-0.3283692878006666, -0.2600160601409711, -1.0), (0.9646083551019333, 0.22455015821333235, -1.0), (-0.5843486892963347, -0.44409276885235915, -1.0)]


data = classA+classB
random.shuffle(data)

print(classA)
print(classB)

def kernel_1(x,y):
    return x[0]*y[0]+x[1]*y[1] +1

def kernel_2(x,y):
    P = 15
    return (x[0]*y[0]+x[1]*y[1] +1)**P

def kernel_3(x,y):
    S = 5
    return numpy.exp(-((x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1])))/(2*S**2)

def kernel_4(x,y):
    D = -1
    K = .2
    return numpy.tanh( K*(x[0]*y[0]+x[1]*y[1]) - D)

def kernel(x,y):
    #print(kernel_4(x,y))
    return kernel_1(x,y)

def indicator(x,y,supports):
    s = 0
    v = numpy.array([x,y])
    for i in range(len(supports)):
        s = s + supports[i][0] * supports[i][1][2]*kernel(v,supports[i][1])
    return s


#plot data
pylab.hold(True)
pylab.plot([p[0] for p in classA],[p[1] for p in classA],'bo')
pylab.plot([p[0] for p in classB],[p[1] for p in classB],'ro')
#pylab.show()

#create matrixes
N = len(data)
q = -numpy.ones(N)
h = numpy.zeros(N)
G = numpy.diag(-numpy.ones(N))
P = numpy.matlib.zeros((N,N))

#add row to G
rows = numpy.diag(numpy.ones(N))
G = numpy.vstack([G, rows])

C = 10**-0.1;
h = numpy.append(h, C*numpy.ones(N))

for i in range(N):
    for j in range(N):
            P[i,j] = (data[i][2]) * (data[j][2]) * kernel(data[i],data[j])

#call qp
r = qp (matrix(P),matrix(q),matrix(G),matrix(h))
alpha=list(r['x'])

supportAlphas = []
for i in range(N):
    if alpha[i] > 10**-0.5:
        supportAlphas.append([alpha[i],data[i]])

xrange=numpy.arange(-4,4,0.05)
yrange=numpy.arange(-4,4,0.05)

grid = matrix([[indicator(x,y,supportAlphas) for y in yrange] for x in xrange])

pylab.contour(xrange, yrange, grid, (-1.0,0.0,1.0),
    colors=('red', 'black', 'blue'),linewidths=(1,3,1))

for i in range(len(supportAlphas)):
    if(supportAlphas[i][1][2] == 1):
        pylab.plot(supportAlphas[i][1][0],supportAlphas[i][1][1],'b*',markersize = 15)
    else:
        pylab.plot(supportAlphas[i][1][0],supportAlphas[i][1][1],'r*',markersize = 15)


pylab.show()
