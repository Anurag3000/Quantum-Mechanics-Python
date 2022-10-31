import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import scipy.integrate as integrate
import pandas as pd

def matrix(a, b, n, l,alpha):
    x = np.arange(a, b, n)
    # print(x)
    h = x[1] - x[0]
    u = np.zeros(shape=(len(x), len(x)))
    V = np.zeros(shape=(len(x), len(x)))
    for i in range(1, len(x) - 1):
        for j in range(1, len(x)):
            if i == j:
                u[i][j] = (2 / h ** 2)
                V[i][j] = x[i]**2 + (2/3)*(alpha*(x[i])**3)
            elif i == j + 1:
                u[i][j] = -1 / h ** 2
            elif i == j - 1:
                u[i][j] = -1 / h ** 2
    return u + V, x

def plot(i, l, power,ratio):
    H, x = matrix(-5, 5, 0.01, l,ratio)
    u = eigh(H)[1][:, i+2]
    # print(u)
    # NORMALIZATION
    c = integrate.simps(u ** 2, x)
    N = u / np.sqrt(c)
    if power==2:
        plt.ylabel("Ψ square")
    else:
        plt.ylabel("Ψ")
    plt.xlabel('x')
    plt.plot(x, N ** power, label='for alpha='+str(ratio))
    plt.legend()

def eigen(a, b, h, l,ratio, i):
    H, x = matrix(a, b, h, l,ratio)
    u = eigh(H)[0][i+2]
    v = eigh(H)[1][:, i]
    return u

x=np.linspace(-10,10,200)
alpha=[0,1,10**(-1),10**(-2)]
for i in alpha:
    V=x**2 + (2/3)*(i*(x)**3)
    plt.plot(x,V,label='alpha= '+str(i))
plt.grid()
plt.legend()
plt.xlabel('x')
plt.ylabel('V')
plt.title('Ans.1 (B) Potential VS x')
plt.savefig('Potential.jpg')
plt.show()
# A,B

for j in range(0,11):
    En_calc=[]
    En_analy=[]
    alpha=[]
    for i in [0,1,10**(-1),10**(-2),10**(-3),10**(-4)]:
        alpha.append(i)
        energy=eigen(-20, 20, 0.1, 0, i, j)
        analytic= (2*j + 1) - (1/8)*(i**2)*(15*(2*j + 1)**2 + 7)
        En_calc.append(energy)
        En_analy.append(analytic)
    print('Energy eigen values For n=',j)
    print(pd.DataFrame({'alpha':alpha,'Calculated':En_calc,'Analytic': En_analy}))
    print("========================================================")

#C,D

En=[]
ratio=[0,1,10**(-1),10**(-2),10**(-3),10**(-4)]
n=[0,1,2,3,4,5,6,7,8,9,10]
for j in ratio:
    E=[]
    for i in n:
        energy = eigen(-20, 20, 0.1, 0, j, i)
        E.append(energy)
    En.append(E)
p=0
for i in En:
    plt.scatter(n,i)
    plt.grid()
    plt.xlabel('n (states)')
    plt.ylabel('Energy eigen value')
    plt.title('For alpha= '+str(alpha[p]))
    plt.savefig('alpha=' + str(alpha[p])+'.jpg')
    plt.show()
    p+=1
n=[0,1,2,3,4,5]
ratio=[0,1,10**(-1),10**(-2)]
for j in n:
    E=[]
    for i in ratio:
        plot(j,0,1,i)
    plt.title('For n='+str(j)+' Ψ VS x')
    plt.grid()
    plt.savefig('n='+str(j)+'_power1')
    plt.show()
    for i in ratio:
        plot(j,0,2,i)
    plt.title('For n='+str(j)+' Ψ square VS x')
    plt.grid()
    plt.savefig('n=' + str(j) + '_power2')
    plt.show()
