import scipy.integrate as integrate
from scipy.stats import rv_continuous
import uncertainties
from uncertainties import ufloat
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
from math import log
from math import log10
from scipy import stats
from math import exp
from math import pi
from math import sqrt
from math import atan2
from math import asin
#import topologylimit
from scipy.optimize import fmin_tnc
from scipy import special
from operator import itemgetter, attrgetter
from scipy.integrate import quad
from math import sin
import matplotlib
import matplotlib.pyplot as plt
import pickle as pickle
import sys
import numdifftools as nd
from numpy.linalg import inv

#from pyLIMA import event
#from pyLIMA import telescopes
#from pyLIMA import microlmodels
from pyLIMA import microlsimulator

#from multiprocessing import Process, Queue, Pool


# In[2]:


#image track uncertainties

def images_and_uncertainties(t,u0_val,te_val,t0_val,u0_err,te_err,t0_err):
    u0 = ufloat(u0_val, u0_err)
    t0 = ufloat(t0_val, t0_err)
    te = ufloat(te_val, te_err)    
    usqr = u0**2 + ((t-t0)/te)**2
    usqrt4 = (usqr + 4.)**0.5
    u = (usqr)**0.5
   #the two image positions, as radial distance
    uplus  = 0.5*(u+usqrt4)
    uminus = 0.5*(u-usqrt4)
   #the corresponding magnification
    muplus=uplus**2/(uplus*2-uminus**2)
    muminus=uminus**2/(uplus*2-uminus**2)    
    spplus=2.*muplus-1.
    spminus=2.*muminus
    prox=(t-t0)/(te*u)
    proy=u0/u
    pspl=(usqr+2.0)/(u*usqrt4)    
    return pspl,uplus,uminus,prox,proy,muplus,muminus,spplus,spminus
def images_and_err_array(t,u0_val,te_val,t0_val,u0_err,te_err,t0_err):
    images_tuple = images_and_uncertainties(t,u0_val,te_val,t0_val,u0_err,te_err,t0_err)
    uplus = images_tuple[1]
    uminus = images_tuple[2]
    prox = images_tuple[3]
    proy = images_tuple[4]
    u0 = ufloat(u0_val, u0_err)
    t0 = ufloat(t0_val, t0_err)
    te = ufloat(te_val, te_err)
    image_position_array = [(prox * uplus).nominal_value, (proy * uplus).nominal_value,
                           (prox * uminus).nominal_value, (proy * uminus).nominal_value,
                           (prox * uplus).nominal_value + (prox * uplus).std_dev,
                           (proy * uplus).nominal_value + (proy * uplus).std_dev,
                           (prox * uminus).nominal_value + (prox * uminus).std_dev,
                           (proy * uminus).nominal_value + (proy * uminus).std_dev,
                           (prox * uplus).nominal_value - (prox * uplus).std_dev,
                           (proy * uplus).nominal_value - (proy * uplus).std_dev,
                           (prox * uminus).nominal_value - (prox * uminus).std_dev,
                           (proy * uminus).nominal_value - (proy * uminus).std_dev,
                           ((ufloat(t,0)-t0)/te).nominal_value, u0.nominal_value,
                           ((ufloat(t,0)-t0)/te).nominal_value + ((ufloat(t,0)-t0)/te).std_dev,
                           u0.nominal_value + u0.std_dev,
                           ((ufloat(t,0)-t0)/te).nominal_value - ((ufloat(t,0)-t0)/te).std_dev,
                           u0.nominal_value - u0.std_dev]
    return image_position_array
def generate_image_arrays(t_arr,u0,te,t0,u0_err,te_err,t0_err):
    image_position_matrix = []
    for idx in range(len(t_arr)):image_position_matrix.append(images_and_err_array(t_arr[idx],u0,te,t0,u0_err,te_err,t0_err))
    image_position_matrix = np.array(image_position_matrix)
    return image_position_matrix
#corresponding with observed data
def amp_and_imagesf(t,u0,te,t0):
    usqr = u0**2 + ((t-t0)/te)**2
    usqrt4 = (usqr + 4.)**0.5
    u = (usqr)**0.5
    #the two image positions, as radial distance
    uplus  = 0.5*(u+usqrt4)
    uminus = 0.5*(u-usqrt4)
    #the corresponding magnification
    muplus=uplus**2/(uplus*2-uminus**2)
    muminus=uminus**2/(uplus*2-uminus**2)
    spplus=2.*muplus-1.
    spminus=2.*muminus
    prox=(t-t0)/(te*u)
    proy=u0/u
    pspl=(usqr+2.0)/(u*usqrt4)
    return pspl,uplus,uminus,prox,proy,muplus,muminus,spplus,spminus

matplotlib.rcParams.update({'font.size': 14})

m10j=3178.28133
m5e=5.
lm10j=log10(3178.28133)
lm5e=log10(5.)
nacloc=0.
nbinary=0.
nbd_ffp=0.
nvbblimit=0.
ntotsample=0.

def modelvector(x,p):
    return modelvec(x,p[0],p[1],p[2])

def model(x,par1,par2,par3):
    usqr=par1*par1+(x-par3)*(x-par3)/(par2*par2)

    return (usqr+2.0)/(usqr*(usqr+4.0))**0.5
    
modelvec=np.vectorize(model)

def pspl_n_imagesf(t,u0,te,t0):
    usqr=u0**2+((t-t0)/te)**2
    usqrt4=(usqr+4.)**0.5
    u=(usqr)**0.5
    uplus  = 0.5*(u+usqrt4)
    uminus = 0.5*(u-usqrt4)
    muplus=uplus**2/(uplus*2-uminus**2)
    muminus=uminus**2/(uplus*2-uminus**2)
    spplus=2.*muplus-1.
    spminus=2.*muminus
    prox=(t-t0)/(te*u)
    proy=u0/u
    pspl=(usqr+2.0)/(u*usqrt4)
    return pspl,uplus,uminus,prox,proy,muplus,muminus,spplus,spminus

def modelvector(x,p):
    return modelvec(x,p[0],p[1],p[2])

def chancemultistar(mass):
    return 0.3*log10(mass)+0.5

def pmfjustlog():
    logm=np.random.rand()*(lm10j-lm5e)+lm5e
    return 10.0**logm

def ajustlog2():
    loga=np.random.rand()*2.-1.
    return 10.0**loga

lmp=log10(0.03)
lmlimit=log10(1e-5)
def pmfjustlog2():
    logm=np.random.rand()*(lmp-lmlimit)+lmlimit
    return 10.0**logm

def pmfcassan():
    #slope uncertainty 0.17
#   disp=np.random.randn()*0.17
    #Correction -1 in coeff for log m -> m 
    disp=0.
    fmax=m5e**(-1.73+disp)
    #envelope for rejectionsampling
    envsample=1.1*fmax
    #accept and reject
    f=0.
    while envsample>f:
        #5 MEarth to 10 Jupiter range
        m=np.random.rand()*(m10j-m5e)+m5e
        envsample=np.random.rand()*fmax
        f=m**(-1.73+disp)
    return m

import VBBinaryLensing
VBB=VBBinaryLensing.VBBinaryLensing()
VBB.Tol=float(0.005)
VBB.satellite=int(0)

NSAMPLE=5000
NDEVREQ=15

###Galactic model Tsapras (cf. Tsapras et al. 2016)
mu=np.array([-0.184022221038,1.38984735765,0.358441776558])
cov=np.matrix([[ 0.47271887 ,0.25087375 , 0.24034958], [ 0.25087375,  0.21363205,  0.15376765], [ 0.24034958  ,0.15376765 , 0.14567189]])
#sample=np.random.multivariate_normal(mu,cov)
#mass in msun: 10.0**(sample[0]), 
#tE in days: 10.0**(sample[1])
#Einstein radius in AU: 10.0**(sample[2])
#norm_fact=1.0/((2.0*pi)**(0.5*3.0)*sqrt(det(cov)))

#Prepare global kernel for planet simulation
#planetpars=np.loadtxt('planets_ml_nov2016.tab',comments=['row','#','pl'],usecols=(2,3))
#planetpars[:,1]=planetpars[:,1]/0.000954265748
#qpars=np.divide(planetpars[:,0],planetpars[:,1])
#qpars=np.log(qpars)
#values=np.vstack([qpars])
#kernel=stats.gaussian_kde(values)

def find_idx_nearest(array,value):
    idx_sorted=np.argsort(array)
    sorted_array=np.array(array[idx_sorted])
    idx= np.searchsorted(sorted_array, value, side="left")
    if idx>= len(array):
        idx_nearest=idx_sorted[len(array)-1]
    elif idx==0:
        idx_nearest=idx_sorted[0]
    else:
        if abs(value-sorted_array[idx-1])<abs(value-sorted_array[idx]):
            idx_nearest=idx_sorted[idx-1]
        else:
            idx_nearest=idx_sorted[idx]
    return idx_nearest

#from multiprocessing import Process, Queue, Pool

def binary_function(s,q,xs,ys,rho,tol):
    return VBB.BinaryMag0(s,q,xs,ys)

def binary_function2(s,q,xs,ys,rho,tol):
    return VBB.BinaryMag(s,q,xs,ys)

def reldev(q,t,u0,te,t0,fs,fb,ferr,rs,xp,yp,d):
    pspl,uplus,uminus,prox,proy,muplus,muminus,spplus,spminus=pspl_n_imagesf(t,u0,te,t0)
    bsx=(t-t0)/te
    bsy=u0
    invq1=1.0/(q+1.0)
    #close=0.
    #interm=0.
    #wide=0.
    total=0.
    newout = 0
    #Erdl and Schneider 1993, Dominik 1999
    #limc,limw=topologylimit.limits(q)
    #avoid innermost planets
    if d>0.005:
        #Determine offset from magnification origin (VBB)	
        com=np.array([xp/d,yp/d])
        com=invq1*com
        com=d*q*com
        nsvec=np.array([bsx,bsy])-com
        #Determine rotation between shifted source position to put binary y1 axis (symmetry axis) on y2=0
        phi1=np.angle(complex(xp,yp))
        phi2=np.angle(complex(nsvec[0],nsvec[1]))
        phi3=np.angle(complex(bsx,bsy))
        angle=phi2-phi1-phi3+pi*0.5
        angle=-phi1+pi*0.5
        rmatrix=np.matrix([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        bsinmap=np.dot(rmatrix,np.array([bsx,bsy])).T            
        #Avoid central region with numerical artefacts, corresponds to d and 1/d degeneracy for a very distant or close target
        newout=0.
        #fspl = fsplmodel(t,u0,te,t0,rs)     

#        result = VBB.BinaryMag(d,q,float(bsinmap[1][0]),float(bsinmap[0][0]),rs,0.005)
        result = VBB.BinaryMag0(d,q,float(bsinmap[1][0]),float(bsinmap[0][0]))

        #result = binary_function(d,q,float(bsinmap[1][0]),float(bsinmap[0][0]),rs,0.01))
        delta_chisqr = (fs*result-fs*pspl)**2/ferr**2   
        return delta_chisqr,q,xp,yp,d,rs,float(bsinmap[1][0]),float(bsinmap[0][0]),newout    
        #print(fs*VBB.BinaryMag(d,q,float(bsinmap[1][0]),float(bsinmap[0][0]),rs,0.005)-fs*pspl)**2,ferr**2)
    else:
        delta_chisqr=0
        newout=-1
        return 0,q,xp,yp,d,rs,1,1,newout
#    newout=-1

def simulate_simple(te):
    #rs=1e-3#not yet implemented -> likely: log-normal      
    nsamples=1
    xp=0
    yp=0
    sample=[0,0.,0.]
    while(abs(10.0**(sample[1])-te)>(0.1*te)):
        sample=np.random.multivariate_normal(mu,cov)	
    masstsol=10.0**(sample[0])
  
    probmultistar=chancemultistar(masstsol)
    multichance=np.random.rand()
    multiflag=0.
    if multichance<probmultistar:
        multiflag=1.

    masst=masstsol/3.0024584e-6
    bdflag=0.
    ffpflag=0.
    
    if masstsol<0.0667986024:
        bdflag=1.
      
    rs=1e-3
    sample=[0,0.,0.]
    #simulate mass and sanity check mass bin
    q=pmfjustlog2()
    rout=ajustlog2()
    x=np.random.normal()
    y=np.random.normal()
    z=np.random.normal()
    r=(x*x+y*y+z*z)**0.5
    xp=x/r*rout
    yp=y/r*rout
    dout=(xp**2+yp**2)**0.5

    if dout>6. or q<1e-5 or rs>0.1:
        return -1,-1,np.nan,-1,-1,-1,1,1,-1,-1,-1

    massfac=q/(q+1.)
    mass=massfac*masst
    if masst<13.:
        ffpflag=1.

    return xp,yp,q,rs,1,1,dout,1,multiflag,bdflag,ffpflag

norm = integrate.quad(lambda q : 0.61*((q/(1.7e-4))**(-0.92)*np.heaviside(q-1.7e-4,0.5)+(q/(1.7e-4))**0.44*np.heaviside(1.74e-4-q,0.5)),3.16e-5,3e-2)[0]
norm_left = integrate.quad(lambda q : 0.61*((q/(1.7e-4))**(-0.92)*np.heaviside(q-1.7e-4,0.5)+(q/(1.7e-4))**0.44*np.heaviside(1.74e-4-q,0.5)),3.16e-5,1.7e-4)[0]
norm_right = integrate.quad(lambda q : 0.61*((q/(1.7e-4))**(-0.92)*np.heaviside(q-1.7e-4,0.5)+(q/(1.7e-4))**0.44*np.heaviside(1.74e-4-q,0.5)),1.7e-4,3e-2)[0]
right = norm_right/norm
left = norm_left/norm

class suzuki_mass_ratio_right(rv_continuous):
    "Suzuki mass ratio distribution"
    def _pdf(self, q):
        return 1/norm_right * 0.61*((q/(1.7e-4))**(-0.92))
suzukian_q_right = suzuki_mass_ratio_right(name='suzukian_q_right', a=1.7e-4,b=3e-2)
                              
class suzuki_mass_ratio_left(rv_continuous):
    "Suzuki mass ratio distribution"
    def _pdf(self, q):
        return 1/norm_left * 0.61*((q/(1.7e-4))**0.44)
suzukian_q_left = suzuki_mass_ratio_left(name='suzukian_q_left', a=3.16e-5,b=1.7e-4)

norm_s = integrate.quad(lambda s : s**0.5,0.1,10)[0]
class suzuki_separation(rv_continuous):
    "Suzuki separation distribution"
    def _pdf(self, s):
        return 1/norm_s * s**0.50
suzukian_s = suzuki_separation(name='suzukian_s', a=0.1,b=10)

def simulate_planet_pos(d):
    x = np.random.normal()
    y = np.random.normal()
    r = (x**2+y**2)**0.5
    xp = d*x/r
    yp = d*y/r
    return xp,yp

def binary_function_4_fisher(q,t,u0,te,t0,xp,yp,d):
    pspl,uplus,uminus,prox,proy,muplus,muminus,spplus,spminus=pspl_n_imagesf(t,u0,te,t0)
    bsx=(t-t0)/te
    bsy=u0
    invq1=1.0/(q+1.0)
    #close=0.
    #interm=0.
    #wide=0.
    total=0.
    newout = 0
    #Erdl and Schneider 1993, Dominik 1999
    #limc,limw=topologylimit.limits(q)
    #Determine offset from magnification origin (VBB)	
    com=np.array([xp/d,yp/d])
    com=invq1*com
    com=d*q*com
    nsvec=np.array([bsx,bsy])-com
    #Determine rotation between shifted source position to put binary y1 axis (symmetry axis) on y2=0
    phi1=np.angle(complex(xp,yp))
    phi2=np.angle(complex(nsvec[0],nsvec[1]))
    phi3=np.angle(complex(bsx,bsy))
    angle=phi2-phi1-phi3+pi*0.5
    angle=-phi1+pi*0.5
    rmatrix=np.matrix([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    bsinmap=np.dot(rmatrix,np.array([bsx,bsy])).T            
    #Avoid central region with numerical artefacts, corresponds to d and 1/d degeneracy for a very distant or close target
    newout=0. 
    result = VBB.BinaryMag0(d,q,float(bsinmap[1][0]),float(bsinmap[0][0]))
    return result 

def fisher(data_time,q,u0,te,t0,xp,yp,fs,fb,flux,fs_err,fb_err,flux_err,d):
    func = lambda c: binary_function_4_fisher(c[0],t,u0,c[1],t0,xp,yp,d)
    Fisher = [[0, 0], [ 0,  0]]
    for i in range(len(data_time)):
        f = flux[i]
        f_err = flux_err[i]
        t = data_time[i]
        if t > t0-te and t < t0+te:
            sigma = np.sqrt((fb_err/fs)**2+((f-fb)*fs_err/fs**2)**2+(f_err/fs)**2)
            jac1 = nd.Jacobian(func)
            jac = jac1([q,te])
            f = 1/sigma**2 * (np.transpose(jac) @ jac)
            Fisher = np.add(Fisher,f)
    Fisher_inv = inv(Fisher)
    return Fisher_inv