import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import pandas as pd
from sklearn.linear_model import LinearRegression

P = []
X = []


def RK4(f, X0, tmin, tmax, N, e):
    h = (tmax - tmin) / N
    t = np.linspace(tmin, tmax, N + 1)
    X = np.zeros([N + 1, len(X0)])
    X[0] = X0
    for i in range(N):
        k1 = f(t[i], X[i], e)
        k2 = f(t[i] + h / 2, X[i] + (k1 * h) / 2, e)
        k3 = f(t[i] + h / 2, X[i] + (k2 * h) / 2, e)
        k4 = f(t[i] + h, X[i] + k3 * h, e)
        X[i + 1] = X[i] + (h / 6) * (k1 + 2 * (k2 + k3) + k4)

    return X, t


def func(x, Y, e):  # functions along with the initial conditions
    psi = 0
    y1, y2 = Y

    psider1 = y2
    psider2 = -e * y1
    return np.array([psider1, psider2])


def main_f(f, ic, tmin, tmax, N, e):
    rk4 = RK4(f, ic, tmin, tmax, N, e)
    rk4_X = rk4[0].T
    t = rk4[-1]
    return rk4_X, t

#Part (a)

initial_conditions = [0, 1]

last_u = []
x = []
e = np.arange(0, 251, 1)
for i in range(0, 251):
    Z, t = main_f(func, initial_conditions, -1 / 2, 1 / 2, 100, i)
    # NORMALIZATION
    c = integrate.simps(Z[0] ** 2, t)
    N = Z[0] / np.sqrt(c)
    last_u.append(Z[0][-1])
    x.append(t[-1])

plt.scatter(e, last_u)
plt.grid()
plt.xlabel('e')
plt.ylabel('psi')
plt.show()

# CHECKING
last_u = np.array(last_u)
last_u2 = last_u[:-1]
last_u3 = last_u[1:]
new_list = last_u2 * last_u3
new = []
new1 = []
new2 = []

index1 = []
index2 = []
for i, j in zip(last_u2, last_u3):
    if i * j < 0:
        new1.append(i)
        new2.append(j)
        index1.append(np.where(last_u == i))
        index2.append(np.where(last_u == j))

I = index1 + index2
I.sort()

u_req = np.array(new1) * np.array(new2)

index2 = []
for i in I:
    t = i[0][0]
    index2.append(t)


energy_req = []
psi_req = []
for i in index2:
    energy_req.append(e[i])
    psi_req.append(last_u[i])


# FILTERING

energy = np.array(energy_req)
energy_2 = energy[:-1]
energy_3 = energy[1:]

psi_req_2 = np.array(psi_req[:-1])
psi_req_3 = np.array(psi_req[1:])
approx = energy_3 - ((energy_3 - energy_2) / (psi_req_3 - psi_req_2)) * psi_req_3
approx2 = approx[::2]


fig, axs = plt.subplots(len(approx2))
p=0

psi_approx2=[]
psi_der_approx2=[]
t_approx2=[]

for i in approx2:
    Z, t = main_f(func, initial_conditions, -1 / 2, 1 / 2, 100, i)
    # NORMALIZATION
    c = integrate.simps(Z[0] ** 2, t)
    N = Z[0] / np.sqrt(c)
    psi_approx2.append(N)

    d = integrate.simps(Z[1] ** 2, t)
    N2 = Z[1] / np.sqrt(c)
    psi_der_approx2.append(N2)
    t_approx2.append(t)

    #fig.suptitle("FOR e= " + str(i))
    axs[p].plot(t, N)
    axs[p].set_title("FOR e= " + str(i))
    axs[p].grid()
    p+=1

plt.show()

#Part (b) CURVE FITTING

nsqr = np.array(approx2 / np.pi ** 2)
plt.scatter(nsqr,approx2)
model= LinearRegression()
model.fit(nsqr.reshape((-1,1)),approx2)
ypred=model.predict(nsqr.reshape((-1,1)))
print("slope: ",model.coef_)
print("Actual Value Of Slope: ", np.pi**2)
print("R_sqr: ",model.score(nsqr.reshape((-1,1)),approx2))

plt.plot(nsqr,ypred,linestyle='dashdot',color='red')
plt.grid()
plt.xlabel('n square')
plt.ylabel('e')
plt.show()

#Part (c)

L=1
x=np.linspace(-1/2,1/2,100)

for i in range(1,6):
    e=(i**2)*(np.pi**2)
    psi=[]
    Z,t=main_f(func,initial_conditions,-1/2,1/2,100,e)
    # NORMALIZATION
    c = integrate.simps(Z[0] ** 2, t)
    N = Z[0] / np.sqrt(c)
    plt.scatter(t,N**2,label='calculated psi vs x for e='+str(i))
    if i % 2 !=0:
        k= i*(np.pi/L)
        psiodd=np.sqrt(2)*np.cos(k*x)
        plt.plot(x,psiodd**2,label='analytic solution of psi for n='+str(i))
    else:
        k = i * (np.pi / L)
        psieven = np.sqrt(2) * np.sin(k * x)
        plt.plot(x, psieven**2, label='analytic solution of psi for n=' + str(i))
plt.grid()
plt.xlabel('x')
plt.ylabel('probability density')
plt.legend()
plt.show()

#Part (d) and (e)

def eV(m,L,e_approx):
    h=6.63* 10**(-34)

    Eval=[]
    for i in range(1,6):
        Eval.append((i**2 * np.pi**2 * h**2)/(8* m * (L**2)))
    Eval=np.array(Eval)* 6.242 * (10**(17))
    Eigenval=e_approx*((h**2)/(8* m * (L**2))) * 6.242 * (10**(17))
    dtf1=pd.DataFrame({"Analytical Energy Value": Eval, "Eigen ENergy Val": Eigenval})
    print(dtf1)

print("--------FOR ELECTRON WHEN WIDTH OF WELL= 5 Angstrom------")
eV(9.11*(10**(-31)),5 * (10**(-10)),approx2)
print()

print("--------FOR ELECTRON WHEN WIDTH OF WELL= 10 Angstrom------")
eV(9.11*(10**(-31)),10 * (10**(-10)),approx2)
print()

print("--------FOR ELECTRON WHEN WIDTH OF WELL= 5 Fermi------")
eV(1.67*(10**(-27)),5 * (10**(-15)),approx2)
print()


#EXPECTAION VALUES

#Part (f)

exp_x=[]
exp_x2=[]

for i,j in zip(psi_approx2,t_approx2):
    exp_x.append(integrate.simps(i**2 * j,j))
    exp_x2.append(integrate.simps((i ** 2) * (j ** 2), j))

psi_2der_approx2=[]
for i,j in zip(approx2,psi_approx2):
    
    psi_2der_approx2.append(j*i)


exp_p=[]
exp_p2=[]


for i,j,k,l in zip(psi_approx2,psi_der_approx2,psi_2der_approx2,t_approx2):
    exp_p.append(integrate.simps(j*i,l))
    exp_p2.append(integrate.simps(k *i, l))

print(pd.DataFrame({'Expectation Value Of momentum':np.array(exp_p) * -1j * 1.05}))
print(pd.DataFrame({'Expectation Value Of momentum square':np.array(exp_p2) * 1.11}))

sigma_p=np.sqrt((np.array(exp_p2) * 1.11*10**(-34)) - (np.array(exp_p) * -1j * 1.05*10**(-34))**2)
sigma_x=np.sqrt((np.array(exp_x2)) - (np.array(exp_x)**2))
print(pd.DataFrame({'Expectation Value Of Uncertainity':sigma_x*sigma_p}))

#Part (g)

subr=[]
index=[]
subpsi=[]
for i in t_approx2[0]:
    if i>=-1/4 and i<=1/4:
        subr.append(i)
        index.append(np.where(t_approx2[0]==i))
psi_n1=psi_approx2[0]

for i in index:
    subpsi.append(psi_n1[i])

x=np.arange(-1/4,1/4,1)

subpsi2=[]
for i in subpsi:
    subpsi2.append(i[0])

print("Probaility Of Finding The Particle Between L/4 to L/4: ",integrate.simps(np.array(subpsi2)**2,np.array(subr)))