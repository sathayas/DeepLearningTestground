import numpy
import theano.tensor as T
from theano import function

# vectors
a = T.dmatrix('a')
b = T.dmatrix('b')
c = T.dmatrix('c')
d = T.dmatrix('d')

# scalars
p = T.dscalar('p')
q = T.dscalar('q')
r = T.dscalar('r')
s = T.dscalar('s')
u = T.dscalar('u')

# function
e = (((a * p) + (b - q) - (c + r )) * d/s) * u

# function object
f = function([a,b,c,d,p,q,r,s,u], e)

# matrix data
a_data = numpy.array([[1,1],[1,1]])
b_data = numpy.array([[2,2],[2,2]])
c_data = numpy.array([[5,5],[5,5]])
d_data = numpy.array([[3,3],[3,3]])

# scalar data
p_data = 1.0
q_data = 2.0
r_data = 3.0
s_data = 4.0
u_data = 5.0

# regular calculation
f_reg = (((a_data * p_data) + (b_data - q_data) - (c_data + r_data))
         * d_data/s_data) * u_data
print("Expected:", f_reg)

# theano calculation
f_theano = f(a_data,b_data,c_data,d_data,
             p_data,q_data,r_data,s_data,u_data)
print("Via Theano:", f_theano)

      
# Expected: [[-26.25 -26.25]
#  [-26.25 -26.25]]
# Via Theano: [[-26.25 -26.25]
#  [-26.25 -26.25]]
