import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from lib_Qlearning_LK01 import autominy
		
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
# Q-learning
def Q_game(q_table, render, update):
	# Inicializacion
	i = 0
	mse = 0.0
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
	e_y_0, _, _ = Autominy.calc_err(Road_r,x_0,y_0,theta_0,side)
	e_y_disc0 = Autominy.disc_err(e_y_0,Ney)
	delta_0 = 0.0
	
	while (i<=It):
		# Si una variable aleatoria es mayor que eps, explota; de lo contrario explora
		# Exploit - Ejecuta la acción con "value" maximo
		if (np.random.random() > epsilon):						
			delta_disc = np.argmax(q_table[e_y_disc0,:])	
			exploit = exploit+1													# Cuenta cuantas veces ha explotado
		# Explore - Ejecuta la accion aleatoria para actualizar el tensor Q
		# Las acciones toman valores enteros de [0,1,...,10]	
		else:	
			delta_disc = np.random.randint(0, Ndelta+1)		
			explore = explore+1													# Cuenta cuantas veces ha explorado		
		# Lleva a cabo la accion	
		delta_undisc = Autominy.undisc_act(delta_disc,Ndelta)	
		action = [Vmax, delta_undisc, side]
		pose = [x_0,y_0,theta_0,h]
		state_e, reward, done, pose_k1 = Autominy.step(action,pose,Road_r,delta_0) 
		e_y, _ = state_e
		e_y_disc = Autominy.disc_err(e_y,Ney)
		# Actualiza la pose del coche
		x_0 = pose_k1[0] 
		y_0 = pose_k1[1] 
		theta_0 = pose_k1[2]
		# Q Update
		states_disc = [e_y_disc, e_y_disc0]
		belman_P = [reward, DISCOUNT, LEARNING_RATE]
		q_table = Autominy.Q_update(states_disc, delta_disc, q_table, belman_P, update)
		# Actualiza el estado obtenido para la siguiente iteracion	
		e_y_disc0 = e_y_disc
		delta_0 = delta_undisc 
				
		# Guarda valores para ser graficados
		TIME.append(i*h)
		X.append(x_0)
		Y.append(y_0)
		mse = mse+(state_e[0])**2+(state_e[1])**2
		i = i+1
		if (done==False): break
	
	MSE.append(mse)	
	if (i==It+1): success = True
	# Muestra la animacion?
	if (render==True): Autominy.render(X,Y,Xr,Yr,Xl,Yl,Xc,Yc)		
	return success, explore, exploit, q_table

#************************************************************************************
#************************************************************************************
#************************************************************************************
Autominy = autominy()
Xr,Yr,Xc,Yc,Xl,Yl = Autominy.road_02()
Road_r = np.transpose(np.array([Xr,Yr]))	# Linea derecha del camino
#Road_l = np.transpose(np.array([Xl,Yl])	# Linea izquierda del camino)

# Parametros iniciales
LEARNING_RATE = 0.75			# Tasa de aprendizaje
DISCOUNT = 0.95 					# Parametro relacionado a la memoria
EPISODES = 15000					# Total de episodios
SHOW_EVERY = 100000					# Cada N episodios evalua el desempeno
epsilon = 0.85						# Parametro que sirve para escoger entre explorar o explotar
#success = False						# Variable binaria que revisa si se alcanzo el estado final
success_count = 0					# Contador de exitos cada N iteraciones
episode = 0								# Contador de episodios

# Matrices necesarias para graficar
e = []
#eps = []
EXPLORE = []
EXPLOIT = []
MSE = []
SUCC = []
# Q-Table Tensor con los "values" que relaciona a los estados con las acciones
# Es de tamano e_y x delta; en este caso 200x10
# e_y = [0,1,...,200]
# delta = [0,1,...,9]
Ney = 200 	# Cuantizacion cada 1[cm]
Ndelta = 10	#Cuantizacion cada 0.076 [rad]= 4.35 [deg]
q_table = np.zeros((Ney+1,Ndelta+1))						# Genera una Q-table llena de ceros
#q_table = np.load('Qlearning_LK01_01.npy')	# Carga una Q-table preentrenada

# Entrenamiento
while episode<EPISODES:
	episode = episode+1
	mse = 0.0
	success_count = 0
	if (episode % SHOW_EVERY == 0): 												# Cada cierta cantidad de episodios muestra la animacion
		print('++++++++++++++++++++++++++++++++++++++++++++')
		print('episode ', episode)
		print('success ', float(success_count)/SHOW_EVERY)  	# Muestra cuantos exitos tuvo en este intervalo
		success, _,_,q_table = Q_game(q_table, True, False)	# Funcion que ejecuta el juego en un episodio y con animacion
		success_count = 0 																		# Reinicia el contador de exitos
	else:
		print('episode ', episode)
		success, explore, exploit, _  = Q_game(q_table, False, True)# Ejecuta el juego sin animacion
	if (success==True): success_count = success_count+1							# Cuenta cuantos exitos tuvo en este intervalo
	epsilon = np.exp(-0.0005*episode)	

	# Guarda valores para ser graficados
	e.append(episode)
	#eps.append(epsilon)
	SUCC.append(success_count)
	EXPLORE.append(explore)
	EXPLOIT.append(exploit)

# Graficas
plot1 = plt.figure(1)
plt.plot(e, MSE)
plt.title('MSE')
plt.grid()

plot2 = plt.figure(2)
plt.plot(e, EXPLORE)
plt.title('explore')

plot3 = plt.figure(3)
plt.plot(e, EXPLOIT)
plt.title('exploit')

plot4 = plt.figure(4)
plt.plot(e, SUCC)
plt.title('success count')

plt.show()

# Muestra la animacion final
Q_game(q_table, True, False)
np.save('Qlearning_LK01_02.npy', q_table)



