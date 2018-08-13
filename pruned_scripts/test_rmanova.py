
from numpy.random import normal
import pyvttbl as pt
from collections import namedtuple
 
N = 40
P = ["noise","quiet"]
rts = [998,511]
mus = rts*N
 
Sub = namedtuple('Sub', ['Sub_id', 'rt','condition'])               
df = pt.DataFrame()
for subid in xrange(0,N):
    for i,condition in enumerate(P):
        df.insert(Sub(subid+1,
                     normal(mus[i], scale=112., size=1)[0],
                           condition)._asdict())     