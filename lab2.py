from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy.matlib
import numpy, pylab, random, math

print('HEJSAN')

#Generate data
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

data = classA+classB
random.shuffle(data)

def kernel_1(x,y):
    return x[0]*y[0]+x[1]*y[1] +1
    #return numpy.dot(x,y) +1

def kernel_2(x,y):
    P = 2
    return (x[0]*y[0]+x[1]*y[1] +1)**P

def indicator(x,y,supports):
    s = 0
    v = numpy.array([x,y,0])
    for i in range(len(supports)):
        s = s + supports[i][0] * supports[i][1][2]*kernel_2(v,supports[i][1])
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

for i in range(N):
    for j in range(N):
            P[i,j] = (data[i][2]) * (data[j][2]) * kernel_2(data[i],data[j])

#call qp
r = qp (matrix(P),matrix(q),matrix(G),matrix(h))
alpha=list(r['x'])

supportAlphas = []
for i in range(N):
    if alpha[i] > 10**-5:
        supportAlphas.append([alpha[i],data[i]])

xrange=numpy.arange(-4,4,0.05)
yrange=numpy.arange(-4,4,0.05)

grid = matrix([[indicator(x,y,supportAlphas) for y in yrange] for x in xrange])

pylab.contour(xrange, yrange, grid, (-1.0,0.0,1.0),
    colors=('red', 'black', 'blue'),linewidths=(1,3,1))

pylab.show()
