import numpy as np
from numpy import linalg
import matplotlib as mpl
from matplotlib import pyplot as plt

#Define data
x = np.linspace(-2,2,401)
Nx = np.size(x)

amp1 = 1
x01 = 0.5
sigmay1 = 0.6

amp2 = 1.2
x02 = -0.5
sigmay2 = 0.3

dt = 0.01
Nt = 1001
tend = dt*(Nt-1)
t = np.linspace(0,tend,Nt) #time

omega1 = 1.3
omega2 = 4.1

y1 = amp1*np.exp(-((x-x01)**2)/((2*sigmay1**2)))
y2 = amp2*np.exp(-((x-x02)**2)/((2*sigmay2**2)))

Y = np.zeros([Nx,Nt],dtype='d')
for tt in range(Nt):
    Y[:,tt] = y1*np.sin(2*np.pi*omega1*t[tt]) + y2*np.sin(2*np.pi*omega2*t[tt])

# show y1 and y2
plt.plot(x,y1,label='y1')
plt.plot(x,y2,label='y2')
plt.xlabel('x', fontsize=18)
plt.ylabel('y', fontsize=18)
plt.legend()
plt.show()

#plt all data
Tgrid, Ygrid = np.meshgrid(t,x)

#contour
plt.contour(Ygrid, Tgrid, np.abs(Y))
plt.xlabel('x', fontsize=18)
plt.ylabel('time', fontsize=18)
plt.ylim(0,4)
plt.show()

U, S, VT = linalg.svd(Y, full_matrices= False)

plt.semilogy(S,'-o')
plt.xlim(0,10)
plt.ylabel('Singular Value', fontsize=18)
plt.xlabel('Index', fontsize=18)
plt.show()

# x, U
plt.plot(x,U[:,0],label='U1')
plt.plot(x,U[:,1],label='U2')
plt.xlabel('x', fontsize=18)
plt.ylabel('U', fontsize=18)
plt.title('POD modes', fontsize=18)
plt.legend()
plt.show()

# t,VT
plt.plot(t,VT[0,:],label='VT1')
plt.plot(t,VT[1,:],label='VT2')
plt.xlim(0,4)
plt.xlabel('time', fontsize=18)
plt.ylabel('VT', fontsize=18)
plt.title('mode coefficients', fontsize=18)
plt.legend()
plt.show()


