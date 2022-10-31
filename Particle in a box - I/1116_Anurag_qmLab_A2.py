import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import pandas as pd

P=[]
X=[]

def  RK4 (f , X0 , tmin , tmax , N , e) :
    h=  ( tmax - tmin ) / N
    t = np . linspace (tmin , tmax , N +1)
    X = np . zeros ([N +1 , len( X0 ) ])
    X[0] = X0
    for i in range(N):
        k1 = f ( t [i] , X [i], e)
        k2 = f(t[i] + h / 2, X[i] + (k1 * h) / 2, e)
        k3 = f(t[i] + h / 2, X[i] + (k2 * h) / 2, e)
        k4 = f(t[i] + h, X[i] + k3 * h, e)
        X[i + 1] = X[i] + (h / 6) * (k1 + 2 * (k2 + k3) + k4)

    return X , t


def func (x , Y ,e) : # functions along with the initial conditions
    psi = 0
    y1 , y2 = Y

    psider1 = y2
    psider2 = -e * y1
    return np.array([psider1 , psider2])

def main_f(f, ic, tmin, tmax, N, e):
    rk4 = RK4(f, ic, tmin, tmax, N, e)
    rk4_X = rk4[0].T
    t = rk4[-1]
    return rk4_X, t

initial_conditions = [0, 1]
Z ,t = main_f(func , initial_conditions , -1/2 ,  1/2, 100, 8)

plt.plot(t,Z[0],label='NOT NORMALIZED')
plt.xlabel("x")
plt.ylabel("psi")
plt.title("psi vs x for e=8")
plt.grid()

initial_conditions = [0, 5]
Z2 ,t = main_f(func , initial_conditions , -1/2 ,  1/2, 100, 8)
print(Z2[1])
#NORMALIZATION
c=integrate.simps(Z[0]**2,t)
N=Z[0]/np.sqrt(c)

plt.plot(t,N,label=" NORMALIZED with u'(0)=0")
c=integrate.simps(Z2[0]**2,t)
N=Z2[0]/np.sqrt(c)
plt.plot(t,N,label=" NORMALIZED with u'(0)=5")
plt.legend()
plt.show()

#e=11

initial_conditions = [0, 1]
Z ,t = main_f(func , initial_conditions , -1/2 ,  1/2, 100, 11)

plt.plot(t,Z[0],label='NOT NORMALIZED')
plt.grid()

initial_conditions = [0, 5]
Z2 ,t = main_f(func , initial_conditions , -1/2 ,  1/2, 100, 11)

#NORMALIZATION
c=integrate.simps(Z[0]**2,t)
N=Z[0]/np.sqrt(c)

plt.plot(t,N,label=" NORMALIZED with u'(0)=0")
c=integrate.simps(Z2[0]**2,t)
N=Z2[0]/np.sqrt(c)
plt.plot(t,N,label=" NORMALIZED with u'(0)=5")
plt.legend()
plt.title("psi vs x for e=11")

plt.xlabel("x")
plt.ylabel("psi")
plt.show()



def energy(points):
    e=np.linspace(0.9*(np.pi**2), 1.1*(np.pi**2), points)
    print("step size: ", e[2]-e[1])
    psi_l = []
    for i in e:
        #print("eigenvalue: ", i)
        psi, t2 = main_f(func, initial_conditions, -1 / 2, 1 / 2, 100, i)
        if abs(psi[0][-1])< 0.5 * 10**(-5):
            return psi[0][-1], i, psi,t2
            #psi_l.append(psi[0][-1])
    #return psi_l

points=5
psi_l,i,psi,x = energy(points)
print("EIGENFUNCTION: ",psi_l)
print("EIGENVALUE: ",i)
print(pd.DataFrame({"PSI": psi[0], "x": x}))

#NORMALIZATION

c=integrate.simps(psi[0]**2,t)
N=psi[0]/np.sqrt(c)

plt.plot(x,N)
plt.grid()
plt.xlabel("x")
plt.ylabel("psi")
plt.title("Ground State Normalized eigen function")
plt.show()


#FIRST EXCITED STATE

initial_conditions = [0, 1]
Z ,t = main_f(func , initial_conditions , -1/2 ,  1/2, 100, 4*(np.pi)**2)

plt.plot(t,Z[0],label='NOT NORMALIZED')
plt.grid()

initial_conditions = [0, 5]
Z2 ,t = main_f(func , initial_conditions , -1/2 ,  1/2, 100, 4*(np.pi)**2)

c=integrate.simps(Z[0]**2,t)
N=Z[0]/np.sqrt(c)

plt.plot(t,N,label=" NORMALIZED with u'(0)=0")
c=integrate.simps(Z2[0]**2,t)
N=Z2[0]/np.sqrt(c)
plt.plot(t,N,label=" NORMALIZED with u'(0)=5")
plt.legend()
plt.title("psi vs x for e=4*pi**2")
plt.xlabel("x")
plt.ylabel("psi")
plt.show()
