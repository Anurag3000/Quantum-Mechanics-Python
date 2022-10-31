import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import scipy.integrate as integrate


def matrix(a, b, n, l):
    x = np.arange(a, b, n)
    # print(x)
    h = x[1] - x[0]
    u = np.zeros(shape=(len(x), len(x)))
    V = np.zeros(shape=(len(x), len(x)))
    for i in range(1, len(x) - 1):
        for j in range(1, len(x)):
            if i == j:
                u[i][j] = (2 / h ** 2)
                V[i][j] = ((-2 / x[i]) + (l * (l + 1)) / x[i] ** 2)
            elif i == j + 1:
                u[i][j] = -1 / h ** 2
            elif i == j - 1:
                u[i][j] = -1 / h ** 2
    return u + V, x

def plot(i, l, power):
    H, x = matrix(0.01, 50, 0.01, l)
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
    plt.plot(x, N ** power, label='for n='+str(i+1)+' for l='+str(l))
    plt.legend()
    # plt.show()

def Veff(x, l):
    Vef = (l * (l + 1) / (x ** 2)) - (2 / x)
    V = -2 / (x)
    return Vef, V

def eigen(a, b, h, l, i):
    H, x = matrix(a, b, h, l)
    u = eigh(H)[0][i]
    v = eigh(H)[1][:, i]
    print("energy eigen value for n=", i + 1, u)
    return v,x

# A-i
for i in range(0, 5):
    H, x = matrix(0.01, 5, 0.01, i)
    Vef, V = Veff(x, i)
    plt.plot(x, Vef, label='for l= '+str(i))
    plt.xlabel('x')
    plt.ylabel('Veff')
    plt.title('x VS Veff')
plt.grid()
plt.legend()
plt.savefig('Veff VS x')
plt.show()

for i in range(0, 4):
    H, x = matrix(0.01, 5, 0.01, i)
    Vef, V = Veff(x, i)
    plt.plot(x, V, label='for l= '+str(i))
    plt.xlabel('x')
    plt.ylabel('V')
    plt.title('x VS V')
plt.grid()
plt.legend()
plt.savefig('V VS x')
plt.show()

# A-ii
H, x = matrix(10 ** (-14), 50, 0.5, 0)
u = eigh(H)[0]
for i in range(10):
    eigen(10 ** (-14), 50, 0.5, 0, i)

# A-iii
for i in range(0, 4):
    plot(i, 0, 1)
plt.grid()
plt.savefig('psi vs x')
plt.show()

# B
print("=======================FOR l=1========================")
for i in range(10):
    v,x = eigen(10 ** (-14), 150, 0.1, 1, i)
    c = integrate.simps(v ** 2, x)
    N = u / np.sqrt(c)
    print("energy eigen vector for n=", i + 1, N[:10])

print("=======================FOR l=2========================")
for i in range(10):
    v,x = eigen(10 ** (-14), 150, 0.1, 2, i)
    c = integrate.simps(v ** 2, x)
    N = u / np.sqrt(c)
    print("energy eigen vector for n=", i + 1, N[:10])

# C

'''for n=1:'''
plot(0, 0, 2)
plt.grid()
plt.savefig('psi2 vs x n=0')
#plt.show()
'''for n=2:'''
plot(1, 0, 2)
plot(1, 1, 2)
plt.grid()
plt.savefig('psi2 vs x n=1')
#plt.show()
