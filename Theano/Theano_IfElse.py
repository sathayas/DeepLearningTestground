import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse

# first option, using the max between two inputs
def hinge_a(x,y):
    return T.max([0 * x, 1-x*y])

# second option, ifelse, in which only the one outcome is calculated
def hinge_b(x,y):
    return ifelse(T.lt(1-x*y,0), 0 * x, 1-x*y)

# third option, switch, in which both outcomes are calculated regardless
def hinge_c(x,y):
    return T.switch(T.lt(1-x*y,0), 0 * x, 1-x*y)

# defining tensor variables
x = T.dscalar('x')
y = T.dscalar('y')

# defining outcomes
z1 = hinge_a(x, y)
z2 = hinge_b(x, y)
z3 = hinge_b(x, y)

# defining functions
f1 = theano.function([x,y], z1)
f2 = theano.function([x,y], z2)
f3 = theano.function([x,y], z3)

# checking outcomes. They should be the same
print("f(-2, 1) =",f1(-2, 1), f2(-2, 1), f3(-2, 1))
print("f(-1,1 ) =",f1(-1, 1), f2(-1, 1), f3(-1, 1))
print("f(0,1) =",f1(0, 1), f2(0, 1), f3(0, 1))
print("f(1, 1) =",f1(1, 1), f2(1, 1), f3(1, 1))
print("f(2, 1) =",f1(2, 1), f2(2, 1), f3(2, 1))
