import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import scipy.integrate as integrate


def matrix(a, b, n, l,ratio):
    x = np.arange(a, b, n)
    # print(x)
    h = x[1] - x[0]
    u = np.zeros(shape=(len(x), len(x)))
    V = np.zeros(shape=(len(x), len(x)))
    for i in range(1, len(x) - 1):
        for j in range(1, len(x)):
            if i == j:
                u[i][j] = (2 / h ** 2)
                V[i][j] = (l * (l + 1) / 2*((x[i] ** 2))) - (2 / x[i])*np.exp(-x[i]/ratio)
            elif i == j + 1:
                u[i][j] = -1 / h ** 2
            elif i == j - 1:
                u[i][j] = -1 / h ** 2
    return u + V, x

def plot(i, l, power,ratio):
    H, x = matrix(0.01, 10, 0.01, l,ratio)
    u = eigh(H)[1][:, i]
    # print(u)
    # NORMALIZATION
    c = integrate.simps(u ** 2, x)
    N = u / np.sqrt(c)
    if power==2:
        plt.title("Ψ square VS x")
        plt.ylabel("Ψ square")
    else:
        plt.title("Ψ VS x")
        plt.ylabel("Ψ")
    plt.xlabel('x')
    plt.plot(x, N ** power, label='for ratio='+str(ratio)+' for l='+str(l))
    plt.legend()

def Veff(x, l, ratio):
    Vef = (l * (l + 1) / (x ** 2)) - (2 / x)*np.exp(-x/ratio)
    V = -2 / (x)
    return Vef, V

def eigen(a, b, h, l,ratio, i):
    H, x = matrix(a, b, h, l,ratio)
    u = eigh(H)[0][i]
    v = eigh(H)[1][:, i]
    return u

# A,B

for j in range(0,2):
    print("For n=", j + 1)
    for i in [2,5,10,20,100]:
        energy=eigen(0.01, 5*(j+1), 0.01, 0, i, j)
        if energy<0:
            print("bound state energy eigen value exists for alpha=",i,": ",energy)

#C,D

E=[]
ratio=[2,5,10,20,100]
for i in ratio:
    plot(0,0,1,i)
    energy = eigen(0.01, 10, 0.01, 0, i, 0)
    E.append(energy)
plt.grid()
#plot(0,0,2,0.00001)
plt.savefig('plot1.jpg')
plt.show()

for i in [2,5,10,20,100]:
    plot(0,0,2,i)
plt.grid()
#plot(0,0,2,0.00001)
plt.savefig('plot2.jpg')
plt.show()

#E

plt.scatter(ratio,E)
plt.xlabel('ratio')
plt.ylabel('Energy')
plt.title('Ground state Energy as a Function Of alpha')
plt.grid()
plt.savefig('plot3.jpg')
plt.show()

