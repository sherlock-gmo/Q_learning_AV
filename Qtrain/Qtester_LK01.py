import numpy as np
import matplotlib.pyplot as plt

from lib_Qlearning_LK01 import autominy

#************************************************************************************
#************************************************************************************
#************************************************************************************
def Q_graph(Q):
	Ney, Ndelta = Q.shape
	#step_y = 2.0/Ney
	#step_delta = (2.0*0.38)/Ndelta
	#x = np.arange(-1.0,1.0,step_y)
	#y = np.arange(-0.38,0.38,step_delta)
	#X = np.arange(0,Ney,1)
	X = []
	Y = []
	for i in range(Ney):
		j = np.argmax(Q[i,:])
		x = i*(2.0/float(Ney))-1.0						# Desnormaliza los estados
		y = ((2.0*j/float(Ndelta))-1.0)*0.38	# Desnormaliza las acciones
		X.append(x)
		Y.append(y)
	X = np.array(X)	
	Y = np.array(Y)
	return X, Y
#************************************************************************************
#************************************************************************************
#************************************************************************************
Autominy = autominy()
Xr,Yr,Xc,Yc,Xl,Yl = Autominy.road_02()
Road_r = np.transpose(np.array([Xr,Yr]))

# Se carga la Q-table que se desea entrenar
q_table = np.load('Qlearning_LK01_02.npy')

# Inicializacion
i = 0
X = []
Y = []
TIME = []
DELTA = []
DDELTA = []
Ey = []
T = 60.0 			# Duracion de la simulación en [s]
h = 1.0/30.0	# Periodo de muestreo en [s]
It = int(T/h)	# Numero de iteraciones

# Calculo de los estados, la recompensa, etc
Ney, Ndelta = q_table.shape
x_0, y_0, theta_0, Vmax, delta, side = Autominy.reset()	
e_y_0, _, _ = Autominy.calc_err(Road_r,x_0,y_0,theta_0,side)
e_y_disc0 = Autominy.disc_err(e_y_0,Ney)
delta_0 = 0.0

while (i<=It):
	delta_disc = np.argmax(q_table[e_y_disc0,:])	
	# Lleva a cabo la accion	
	delta_undisc = Autominy.undisc_act(delta_disc, Ndelta)	
	Vmax = 1.0			# Se cambia la velocidad
	action = [Vmax, delta_undisc, side]
	pose = [x_0,y_0,theta_0,h]
	state_e, reward, done, pose_k1 = Autominy.step(action,pose,Road_r, delta_0)
	# Actualiza la pose del coche
	x_0 = pose_k1[0] 
	y_0 = pose_k1[1] 
	theta_0 = pose_k1[2]
	e_y,_ = state_e
	e_y_disc0 = Autominy.disc_err(e_y, Ney)
	Ddelta = delta_undisc-delta_0
	# Graficas
	TIME.append(i*h)
	X.append(x_0)
	Y.append(y_0)
	Ey.append(e_y)
	DELTA.append(delta_undisc)
	DDELTA.append(Ddelta)
	i = i+1
	if (abs(e_y)>1.0): break

# Muestra la animacion
Autominy.render(X,Y,Xr,Yr,Xl,Yl,Xc,Yc)		

plot1 = plt.figure(1)
plt.plot(TIME, DELTA)
plt.title('steering')
plt.xlabel('t [s]')
plt.ylabel('delta [rad]')
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()

plot2 = plt.figure(2)
plt.plot(TIME, Ey)
plt.title('e_y')
plt.xlabel('e_y [m]')
plt.ylabel('t [s]')
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()

plot3 = plt.figure(3)
plt.plot(TIME, DDELTA)
plt.title('Ddelta')
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()

Xq, Yq = Q_graph(q_table)
plot4 = plt.figure(4)
plt.plot(Xq, Yq, marker = 'o')
plt.title('Q table')
plt.xlabel('e_y [m]')
plt.ylabel('delta [rad]')
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()

plot5 = plt.figure(5)
plt.plot(Xr,Yr,'b',Xc,Yc,'--b',Xl,Yl,'b')
plt.plot(X,Y,'r')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()

plt.show()



