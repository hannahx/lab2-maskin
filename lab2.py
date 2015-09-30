from cvxopt.solvers import qp
from cvxopt.base import matrix
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

#plot
pylab.hold(True)
pylab.plot([p[0] for p in classA],[p[1] for p in classA],'bo')
pylab.plot([p[0] for p in classB],[p[1] for p in classB],'ro')
pylab.show()
