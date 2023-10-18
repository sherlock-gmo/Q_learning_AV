import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib_Qlearning_OE01 import autominy
#************************************************************************************
#************************************************************************************
#************************************************************************************
def Q_graph(Q):
	Nst1, Nst2, Nact = Q.shape
	X = []
	Y = []
	Z = []
	for i in range(Nst1):
		for j in range(Nst2):
			k = np.argmax(Q[i,j,:])
			rho = float(i*(3.0)/Nst1)
			gamma = float(((j*2.0*np.pi)/Nst2)-np.pi)
			h = k*(-2.0)+1.0
			x = rho*np.cos(gamma)
			y = rho*np.sin(gamma)
			z = h
			X.append(x)
			Y.append(y)
			Z.append(z)
	X = np.array(X)	
	Y = np.array(Y)
	Z = np.array(Z)
	return X, Y, Z
		
#************************************************************************************
#************************************************************************************
#************************************************************************************
Autominy = autominy()
Xr,Yr,Xc,Yc,Xl,Yl = Autominy.road_02()
Road_r = np.transpose(np.array([Xr,Yr]))
Road_l = np.transpose(np.array([Xl,Yl]))
st_car1 = Autominy.static_car(12.5,2.5,0.2,0.0)
st_car2 = Autominy.static_car(7.5,2.5,3.4,0.0)

q_table = np.load('Qlearning_OE01_01.npy')
Nrho,Ngamma, Nlane = q_table.shape
print(q_table.shape)

# Inicializacion
i = 0
X = []
Y = []
DELTA = []
Ey = []
SIDE = []
RHO = []
GAMMA = []
TIME = []
T = 90.0 			# Duracion de la simulación en [s]
h = 1.0/30.0	# Periodo de muestreo en [s]
It = int(T/h)	# Numero de iteraciones
x_0, y_0, theta_0, Vmax, delta, side = Autominy.reset()	
Vmax = 1.0
rho0, gamma0 = Autominy.v_lidar(st_car1,x_0,y_0,theta_0)
rho0_disc, gamma0_disc = Autominy.disc_states(rho0,gamma0,Nrho,Ngamma)
while (i<=It):
	lane_disc = np.argmax(q_table[rho0_disc,gamma0_disc,:])	
	# Lleva a cabo la accion	
	if (lane_disc==0): 
		R=Road_r
		side = 1
	else: 
		R=Road_l
		side = -1	
	e_y,e_th,in_cam = Autominy.calc_err(R,x_0,y_0,theta_0,side)
	delta = Autominy.optimal_control(e_y,e_th)	
	action = [Vmax, delta, side]
	pose = [x_0,y_0,theta_0,h]
	# Detecta que obstaculo esta mas cerca
	rho_c1, _ = Autominy.v_lidar(st_car1,x_0,y_0,theta_0)
	rho_c2, _ = Autominy.v_lidar(st_car2,x_0,y_0,theta_0)
	if (rho_c1<=rho_c2): Obj = st_car1
	else: Obj = st_car2
	states_disc, reward, done, pose_k1 = Autominy.step(action,pose,R,Obj,Nrho,Ngamma) 
	# Actualiza la pose del coche
	x_0 = pose_k1[0] 
	y_0 = pose_k1[1] 
	theta_0 = pose_k1[2]
	# Actualiza el estado obtenido para la siguiente iteracion	
	rho0_disc, gamma0_disc = states_disc
	# Des-discritetiza los estados
	rho = float(rho0_disc*(3.0)/Nrho)
	gamma = float(((gamma0_disc*2.0*np.pi)/Ngamma)-np.pi)
	# Graficas
	TIME.append(i*h)
	X.append(x_0)
	Y.append(y_0)
	Ey.append(e_y)
	SIDE.append(side)
	DELTA.append(delta)
	RHO.append(rho)
	GAMMA.append(gamma)
	i = i+1
	if (done==False): break
# Muestra la animacion
Autominy.render2(X,Y,Xr,Yr,Xl,Yl,Xc,Yc,st_car1,st_car2)		

plot1 = plt.figure(1)
plt.plot(TIME, DELTA)
plt.title('steering')
plt.grid()

plot2 = plt.figure(2)
plt.plot(TIME, Ey)
plt.title('e_y')
plt.grid()

plot3 = plt.figure(3)
plt.scatter(TIME, SIDE, marker='o')
plt.title('side')
plt.grid()


Xq, Yq, Zq = Q_graph(q_table)
fig = plt.figure(4)
ax = fig.add_subplot(projection='3d')
ax.scatter(Xq, Yq, Zq, marker='o')
ax.set_xlabel('[m]')
ax.set_ylabel('[m]')
ax.set_zlabel('side')

plot5 = plt.figure(5)
plt.scatter(TIME, RHO, marker='o')
plt.title('rho')
plt.grid()

plot6 = plt.figure(6)
plt.scatter(TIME, GAMMA, marker='o')
plt.title('gamma')
plt.grid()

plt.show()
#************************************************************************************
#************************************************************************************
#************************************************************************************
