import numpy as np
import matplotlib.pyplot as plt
from lib_Qlearning_OE01 import autominy

		
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
# Q-learning
def Q_game(q_table, render, update):
	# Inicializacion
	i = 0
	explore = 0
	exploit = 0
	success = False
	X = []
	Y = []
	TIME = []
	T = 90.0 			# Duracion de la simulacion en [s]
	h = 1.0/30.0	# Periodo de muestreo en [s]
	It = int(T/h)	# Numero de iteraciones
	x_0, y_0, theta_0, Vmax, delta, side = Autominy.reset()	
	rho0, gamma0 = Autominy.v_lidar(st_car1,x_0,y_0,theta_0)
	rho0_disc, gamma0_disc = Autominy.disc_states(rho0,gamma0,Nrho,Ngamma)
	while (i<=It):
		# Si una variable aleatoria es mayor que eps, explota; de lo contrario explora
		# Exploit - Ejecuta la acción con "value" maximo
		if (np.random.random() > epsilon):						
			lane_disc = np.argmax(q_table[rho0_disc,gamma0_disc,:])	
			exploit = exploit+1										# Cuenta cuantas veces ha explotado
		# Explore - Ejecuta una accion aleatoria para actualizar el tensor Q
		else:	
			lane_disc = np.random.randint(0,2)		# Las acciones toman valores enteros de 0 o 1
			explore = explore+1										# Cuenta cuantas veces ha explorado		
		# Lleva a cabo la accion	side=1 --> C. derecho // side=-1 --> C. izquierdo
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
		# Lleva a cabo la accion
		states_disc, reward, done, pose_k1 = Autominy.step(action,pose,R,st_car1,Nrho,Ngamma) 
		# Actualiza la pose del coche
		x_0 = pose_k1[0] 
		y_0 = pose_k1[1] 
		theta_0 = pose_k1[2]
		# Q Update
		states_disc0 = rho0_disc, gamma0_disc
		belman_P = [reward, DISCOUNT, LEARNING_RATE]
		q_table = Autominy.Q_update(states_disc0,states_disc, lane_disc, q_table, belman_P, update)
		# Actualiza el estado obtenido para la siguiente iteracion	
		rho0_disc, gamma0_disc = states_disc
		# Graficas
		TIME.append(i*h)
		X.append(x_0)
		Y.append(y_0)
		i = i+1
		if (done==False): break
	if (i==It+1): success = True
	if (render==True): Autominy.render(X,Y,Xr,Yr,Xl,Yl,Xc,Yc,st_car1)		# Muestra la animacion?
	return success, explore, exploit, q_table

#************************************************************************************
#************************************************************************************
#************************************************************************************
Autominy = autominy()
Xr,Yr,Xc,Yc,Xl,Yl = Autominy.road_02()
Road_r = np.transpose(np.array([Xr,Yr]))
Road_l = np.transpose(np.array([Xl,Yl]))
st_car1 = Autominy.static_car(7.5,2.5,0.2,0.0) # Crea el obstaculo con coordenadas (x,y); x in [5,10], y=0.2

# Parametros iniciales
LEARNING_RATE = 0.75  		# Tasa de aprendizaje
DISCOUNT = 0.95 					# Parametro relacionado a la memoria
EPISODES = 15000					# Total de episodios
SHOW_EVERY = 100000				# Cada N episodios evalua el desempeno
epsilon = 0.85						# Parametro que sirve para escoger entre explorar o explotar
success_count = 0					# Contador de exitos cada N iteraciones
episode = 0								# Contador de episodios

# Matrices necesarias para graficar
e = []
#EPSILON = []
EXPLORE = []
EXPLOIT = []
SUCC = []
# Numero de estados
Nrho = 26
Ngamma = 360
Nlane = 2
# La Q-table es un tensor de 26x360x2
q_table = np.zeros((Nrho+1,Ngamma+1,Nlane)) 	# Genera una Q-table llena de ceros
#q_table = np.load('Qlearning_DM01_01.npy')		# Carga una Q-table pre-entrenada

# Entrenamiento
while episode<EPISODES:
	episode = episode+1
	mse = 0.0
	success_count = 0
	if (episode % SHOW_EVERY == 0): 												# Cada cierta cantidad de episodios muestra la animacion
		print('++++++++++++++++++++++++++++++++++++++++++++')
		print('episode ', episode)
		print('success ', float(success_count)/SHOW_EVERY)  	# Muestra cuantos exitos tuvo en este intervalo
		success, _,_,q_table = Q_game(q_table, True, False)		# Funcion que ejecuta el juego en un episodio y con animacion
		success_count = 0 																		# Reinicia el contador de exitos
	else:
		print('episode ', episode)
		success, explore, exploit, _  = Q_game(q_table, False, True)	# Ejecuta el juego sin animacion
	if (success==True): success_count = success_count+1							# Cuenta cuantos exitos tuvo en este intervalo
	epsilon = np.exp(-0.0005*episode)	
	# Guarda valores para ser graficados
	e.append(episode)
	EXPLORE.append(explore)
	EXPLOIT.append(exploit)
	SUCC.append(success_count)
	#EPSILON.append(epsilon)

# Graficas
plot1 = plt.figure(1)
plt.plot(e, SUCC)
plt.title('Success Count')

plot2 = plt.figure(2)
plt.plot(e, EXPLORE)
plt.title('explore')

plot3 = plt.figure(3)
plt.plot(e, EXPLOIT)
plt.title('exploit')

"""
plot5 = plt.figure(5)
plt.plot(e, EPSILON)
plt.title('epsilon')
"""
"""
plot1 = plt.figure(6)
plt.plot(Xr,Yr,'b',Xc,Yc,'--b',Xl,Yl,'b')
plt.plot(Scar1_p[:,0],Scar1_p[:,1],'k.', markersize=5)
plt.title('MSE')
"""
plt.show()

# Muestra la animacion final
Q_game(q_table, True, False)
np.save('Qlearning_OE01_02.npy', q_table)
