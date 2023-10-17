import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 

#************************************************************************************
#************************************************************************************
#************************************************************************************
class autominy():
#************************************************************************************
#*********************************** INICIO *****************************************
#************************************************************************************
	def __init__(self):
		# Parametros del modelo
		self.L = 0.26 			# Separacion de los ejes en m
		self.Lh = 0.36			# Distancia entre el eje trasero y la homografia en m
		self.l = 0.3				# Tamano de la recta que modela el camino en m	
#************************************************************************************
#****************************** MODELO CINEMATICO ***********************************
#************************************************************************************
	def kin_model(self,v,delta, x_k, y_k, theta_k, h):
		if (abs(delta)>=0.38): delta = 0.38*np.sign(delta)
		x_k1 = x_k+h*(v*np.cos(theta_k))
		y_k1 = y_k+h*(v*np.sin(theta_k))	
		theta_k1 = theta_k+h*((v/self.L)*np.tan(delta))
		return x_k1, y_k1, theta_k1
#************************************************************************************
#************************* CONSTRUCTOR DE OBSTACULOS ********************************
#************************************************************************************
	def static_car(self,xc_r,Rx,yc_r,Ry):
		xc = random.uniform(xc_r-Rx,xc_r+Rx)	# Posicion x inicial aleatoria [m]
		yc = random.uniform(yc_r-Ry,yc_r+Ry)	# Posicion y inicial aleatoria [m]
		N = 30.0															# Numero de puntos en el perimetro
		w = random.uniform(0.4,1.2)						# Largo aleatorio del obstaculo [m]
		h = 0.2																# Ancho del obstaculo [m]
		# Calcula las coordenadas de los vertices del rectangulo
		x1 = xc - w/2.0
		y1 = yc - h/2.0
		x2 = xc + w/2.0
		y2 = yc + h/2.0
		# Genera las coordenadas del perimetro del rectangulo
		p1 = w/(N/2.0)
		p2 = h/(N/2.0)
		coordenadas = []
		X = np.arange(x1, x2,p1)
		Y = np.arange(y1, y2,p2)
		for i in X:
			coordenadas.append((i, y1))
			coordenadas.append((i, y2))
		for i in Y:
			coordenadas.append((x1, i))
			coordenadas.append((x2, i))
		return np.array(coordenadas)
#************************************************************************************
#************************* CONSTRUCTOR DE CAMINOS ***********************************
#************************************************************************************
	# Tramo Recto
	def rect_road(self,xi,yi,xf,yf,N):
		if (xf-xi!=0.0):
			m = (yf-yi)/(xf-xi)
			b = yf-m*xf
			d = (xf-xi)/N
			X = np.zeros(N)
			Y = np.zeros(N)
			for k in range (N):	
				x = xi+k*d
				y = m*x+b
				X[k]=x
				Y[k]=y
		else:
			i = (yf-yi)/N
			Y = np.arange(yi,yf,i)
			X = xi*np.ones((N))
		return X,Y
		
	# Tramo Circular
	def circ_road(self,xc,yc,r,arci,arcf,N):
		# El circulo siempre se define en sentido anti-horario
		X = np.zeros(N)
		Y = np.zeros(N)
		d = (arcf-arci)/N
		for k in range (N):
			Dtheta = arci+k*d
			X[k]= xc+r*np.cos(Dtheta)
			Y[k]= yc+r*np.sin(Dtheta) 
		return X,Y
		
	# PISTA 01 // Construido con segmentos de recta de 5 [cm]
	def road_01(self):
		# CARRIL DERECHO
		Xr1, Yr1 = self.rect_road(0.0,0.0,20.0,0.0,400)										#xi,yi,xf,yf,sub_div [m]
		Xr2, Yr2 = self.circ_road(20.0,5.0,5.0,-np.pi/2.0,np.pi/2.0,314)	#xc,yc,R,arci,arcf,sub_div [m],[rad]
		Xr3, Yr3 = self.rect_road(20.0,10.0,0.0,10.0,400)
		Xr4, Yr4 = self.circ_road(0.0,5.0,5.0,np.pi/2.0,3.0*np.pi/2.0,314)	
		Xr = np.concatenate([Xr1,Xr2,Xr3,Xr4])
		Yr = np.concatenate([Yr1,Yr2,Yr3,Yr4])
		#Road_r = np.transpose(np.array([Xr,Yr]))
		# CARRIL CENTRAL
		Xr1, Yr1 = self.rect_road(0.0,0.4,20.0,0.4,400)										#xi,yi,xf,yf,sub_div [m]
		Xr2, Yr2 = self.circ_road(20.0,5.0,4.6,-np.pi/2.0,np.pi/2.0,314)	#xc,yc,R,arci,arcf,sub_div [m],[rad]
		Xr3, Yr3 = self.rect_road(20.0,9.6,0.0,9.6,400)
		Xr4, Yr4 = self.circ_road(0.0,5.0,4.6,np.pi/2.0,3.0*np.pi/2.0,314)	
		Xc = np.concatenate([Xr1,Xr2,Xr3,Xr4])
		Yc = np.concatenate([Yr1,Yr2,Yr3,Yr4])
		#Road_c = np.transpose(np.array([Xr,Yr]))
		# CARRIL IZQUIERDO
		Xr1, Yr1 = self.rect_road(0.0,0.8,20.0,0.8,400)										#xi,yi,xf,yf,sub_div [m]
		Xr2, Yr2 = self.circ_road(20.0,5.0,4.2,-np.pi/2.0,np.pi/2.0,314)	#xc,yc,R,arci,arcf,sub_div [m],[rad]
		Xr3, Yr3 = self.rect_road(20.0,9.2,0.0,9.2,400)
		Xr4, Yr4 = self.circ_road(0.0,5.0,4.2,np.pi/2.0,3.0*np.pi/2.0,314)	
		Xl = np.concatenate([Xr1,Xr2,Xr3,Xr4])
		Yl = np.concatenate([Yr1,Yr2,Yr3,Yr4])
		#Road_l = np.transpose(np.array([Xl,Yl]))
		return Xr,Yr,Xc,Yc,Xl,Yl		

	# PISTA 02 // Construido con segmentos de recta de 5 [cm]
	def road_02(self):
		# CARRIL DERECHO
		Xr1, Yr1 = self.rect_road(0.0,0.0,20.0,0.0,400)										#xi,yi,xf,yf,sub_div [m]
		Xr2, Yr2 = self.circ_road(20.0,1.8,1.8,-np.pi/2.0,np.pi/2.0,113)	#xc,yc,R,arci,arcf,sub_div [m],[rad]
		Xr3, Yr3 = self.rect_road(20.0,3.6,0.0,3.6,400)
		Xr4, Yr4 = self.circ_road(0.0,1.8,1.8,np.pi/2.0,3.0*np.pi/2.0,113)	
		Xr = np.concatenate([Xr1,Xr2,Xr3,Xr4])
		Yr = np.concatenate([Yr1,Yr2,Yr3,Yr4])
		#Road_r = np.transpose(np.array([Xr,Yr]))
		# CARRIL CENTRAL
		Xr1, Yr1 = self.rect_road(0.0,0.4,20.0,0.4,400)										#xi,yi,xf,yf,sub_div [m]
		Xr2, Yr2 = self.circ_road(20.0,1.8,1.4,-np.pi/2.0,np.pi/2.0,113)	#xc,yc,R,arci,arcf,sub_div [m],[rad]
		Xr3, Yr3 = self.rect_road(20.0,3.2,0.0,3.2,400)
		Xr4, Yr4 = self.circ_road(0.0,1.8,1.4,np.pi/2.0,3.0*np.pi/2.0,314)	
		Xc = np.concatenate([Xr1,Xr2,Xr3,Xr4])
		Yc = np.concatenate([Yr1,Yr2,Yr3,Yr4])
		#Road_c = np.transpose(np.array([Xr,Yr]))
		# CARRIL IZQUIERDO
		Xr1, Yr1 = self.rect_road(0.0,0.8,20.0,0.8,400)										#xi,yi,xf,yf,sub_div [m]
		Xr2, Yr2 = self.circ_road(20.0,1.8,1.0,-np.pi/2.0,np.pi/2.0,314)	#xc,yc,R,arci,arcf,sub_div [m],[rad]
		Xr3, Yr3 = self.rect_road(20.0,2.8,0.0,2.8,400)
		Xr4, Yr4 = self.circ_road(0.0,1.8,1.0,np.pi/2.0,3.0*np.pi/2.0,314)	
		Xl = np.concatenate([Xr1,Xr2,Xr3,Xr4])
		Yl = np.concatenate([Yr1,Yr2,Yr3,Yr4])
		#Road_l = np.transpose(np.array([Xl,Yl]))
		return Xr,Yr,Xc,Yc,Xl,Yl		
		
	# PISTA 03 // Construido con segmentos de recta de 5 [cm]
	def road_03(self):
		# CARRIL DERECHO
		Xr1, Yr1 = self.rect_road(0.0,0.0,40.0,0.0,400)										#xi,yi,xf,yf,sub_div [m]
		Xr2, Yr2 = self.circ_road(40.0,1.8,1.8,-np.pi/2.0,0.0,57)	#xc,yc,R,arci,arcf,sub_div [m],[rad]
		Xr3, Yr3 = self.rect_road(41.8,1.8,41.8,8.6,400)
		Xr4, Yr4 = self.circ_road(43.6,8.6,1.8,np.pi,np.pi/2.0,57)	
		Xr5, Yr5 = self.rect_road(43.6,10.4,50.0,10.4,400)
		Xr6, Yr6 = self.circ_road(50.0,12.2,1.8,-np.pi/2.0,np.pi/2.0,113)	
		Xr7, Yr7 = self.rect_road(50.0,14.0,0.0,14.0,400)
		Xr8, Yr8 = self.circ_road(0.0,12.2,1.8,np.pi/2.0,np.pi,113)	
		Xr9, Yr9 = self.rect_road(-1.8,12.2,-1.8,1.8,400)
		Xr10, Yr10 = self.circ_road(0.0,1.8,1.8,-np.pi,-np.pi/2.0,113)	
		Xr = np.concatenate([Xr1,Xr2,Xr3,Xr4,Xr5,Xr6,Xr7,Xr8,Xr9,Xr10])
		Yr = np.concatenate([Yr1,Yr2,Yr3,Yr4,Yr5,Yr6,Yr7,Yr8,Yr9,Yr10])
		# CARRIL CENTRAL
		Xr1, Yr1 = self.rect_road(0.0,0.4,40.0,0.4,400)										#xi,yi,xf,yf,sub_div [m]
		Xr2, Yr2 = self.circ_road(40.0,1.8,1.4,-np.pi/2.0,0.0,57)	#xc,yc,R,arci,arcf,sub_div [m],[rad]
		Xr3, Yr3 = self.rect_road(41.4,1.8,41.4,8.6,400)
		Xr4, Yr4 = self.circ_road(43.6,8.6,2.2,np.pi,np.pi/2.0,57)	
		Xr5, Yr5 = self.rect_road(43.6,10.8,50.0,10.8,400)
		Xr6, Yr6 = self.circ_road(50.0,12.2,1.4,-np.pi/2.0,np.pi/2.0,113)	
		Xr7, Yr7 = self.rect_road(50.0,13.6,0.0,13.6,400)
		Xr8, Yr8 = self.circ_road(0.0,12.2,1.4,np.pi/2.0,np.pi,113)	
		Xr9, Yr9 = self.rect_road(-1.4,12.0,-1.4,1.8,400)
		Xr10, Yr10 = self.circ_road(0.0,1.8,1.4,-np.pi,-np.pi/2.0,113)	
		Xc = np.concatenate([Xr1,Xr2,Xr3,Xr4,Xr5,Xr6,Xr7,Xr8,Xr9,Xr10])
		Yc = np.concatenate([Yr1,Yr2,Yr3,Yr4,Yr5,Yr6,Yr7,Yr8,Yr9,Yr10])
		# CARRIL IZQUIERDO
		Xr1, Yr1 = self.rect_road(0.0,0.8,40.0,0.8,400)										#xi,yi,xf,yf,sub_div [m]
		Xr2, Yr2 = self.circ_road(40.0,1.8,1.0,-np.pi/2.0,0.0,57)	#xc,yc,R,arci,arcf,sub_div [m],[rad]
		Xr3, Yr3 = self.rect_road(41.0,1.8,41.0,8.6,400)
		Xr4, Yr4 = self.circ_road(43.6,8.6,2.6,np.pi,np.pi/2.0,57)	
		Xr5, Yr5 = self.rect_road(43.6,11.2,50.0,11.2,400)
		Xr6, Yr6 = self.circ_road(50.0,12.2,1.0,-np.pi/2.0,np.pi/2.0,113)	
		Xr7, Yr7 = self.rect_road(50.0,13.2,0.0,13.2,400)
		Xr8, Yr8 = self.circ_road(0.0,12.2,1.0,np.pi/2.0,np.pi,113)	
		Xr9, Yr9 = self.rect_road(-1.0,12.0,-1.0,1.8,400)
		Xr10, Yr10 = self.circ_road(0.0,1.8,1.0,-np.pi,-np.pi/2.0,113)	
		Xl = np.concatenate([Xr1,Xr2,Xr3,Xr4,Xr5,Xr6,Xr7,Xr8,Xr9,Xr10])
		Yl = np.concatenate([Yr1,Yr2,Yr3,Yr4,Yr5,Yr6,Yr7,Yr8,Yr9,Yr10])
		return Xr,Yr,Xc,Yc,Xl,Yl	
#************************************************************************************
#******************************* F. P. SIMULACION ***********************************
#************************************************************************************
	# Calculo de la distancia mas corta
	def get_point(self,L,p,theta):
		# Calcular la distancia euclidiana entre cada renglón de L y el punto p dado
		distances = np.linalg.norm(L-p, axis=1)
		# Encontrar el índice del renglon con la distancia euclidiana más corta
		index = np.argmin(distances)
		d = distances[index]
		return L[index][0],L[index][1],d

	# Calculo de la distancia mas corta a un obstaculo
	def v_lidar(self,L,x0,y0,theta0):
		p = [x0,y0]
		# Calcular la distancia euclidiana entre cada renglon de L y el punto p dado
		distances = np.linalg.norm(L-p, axis=1)
		# Encontrar el indice del renglon con la distancia euclidiana mas corta
		index = np.argmin(distances)
		rho = distances[index]
		if (rho>3.0): rho = 3.0
		xc = L[index][0]
		yc = L[index][1]
	  # Calculamos las coordenadas relativas del punto (xc,yc) con respecto a (x0,y0)
		x_rel = xc - x0
		y_rel = yc - y0
		# Rotamos las coordenadas relativas segun la orientacion theta_0
		x_rot = x_rel * np.cos(theta0) + y_rel * np.sin(theta0)
		y_rot = -x_rel * np.sin(theta0) + y_rel * np.cos(theta0)		
		gamma = np.arctan2(y_rot,x_rot)
		return rho, gamma

	# Deteccion de la linea que se debe seguir
	def v_homography(self,x, y, x0, y0, theta_r):	  
		  # Calculamos las coordenadas relativas del punto (x,y) con respecto a (x0,y0)
		  x_rel = x - x0
		  y_rel = y - y0
		  # Rotamos las coordenadas relativas segun la orientacion theta_0
		  x_rot = x_rel * np.cos(theta_r) + y_rel * np.sin(theta_r)
		  y_rot = -x_rel * np.sin(theta_r) + y_rel * np.cos(theta_r)
		  # Devolvemos las coordenadas rotadas
		  return x_rot, y_rot		
		  
	# Calculo de los errores relativos
	def calc_err(self,R,x0,y0,theta0,side):
		# Actualiza la posicion de la "camara"
		xh = x0+self.Lh*np.cos(theta0)
		yh = y0+self.Lh*np.sin(theta0)	
		xhl = x0+(self.Lh+self.l)*np.cos(theta0)
		yhl = y0+(self.Lh+self.l)*np.sin(theta0)
		# Detecta los puntos del camino que estan frente al coche
		x1,y1,d1 = self.get_point(R,np.array([xh, yh]),theta0)
		x2,y2,_ = self.get_point(R,np.array([xhl, yhl]),theta0)
		# Calcula la posicion de los puntos como si se hiciera con una homografia
		x1,y1 = self.v_homography(x1, y1, xh, yh, theta0) 
		x2,y2 = self.v_homography(x2, y2, xh, yh, theta0)
		# Verifica que los puntos esten en el campo de vision de la camara
		if (abs(d1)<1.0): in_cam = True
		else: in_cam = False
		# Calculo de los errores de posicion y orientacion relativos a la carretera
		if (side==1): y_ref = -0.2	# 1-C.Derecho
		else: y_ref = 0.2						# -1-C.Izquierda 
		e_y = y_ref-y1
		e_th = np.arctan2(y1-y2,self.l)
		return e_y, e_th, in_cam

	# Reset
	def reset(self):
		x_0 = 0.0												# Posicion x inicial [m]
		y_0 = random.uniform(-0.3, 0.3)	# Posicion y inicial [m]
		theta_0 = random.uniform(-0.52,0.52) 	# Orientación inicial [rad]
		Vmax = 0.25								# Velocidad [m/s]
		delta = 0.0								# Dirección inicial [rad]
		side = 1									# 1-C.Derecho // -1-C.Izquierda 
		return x_0, y_0, theta_0, Vmax, delta, side

	# Step
	def step(self,action,pose,R,Obj,Nrho,Ngamma): 
		Vmax, delta, side = action
		x_0,y_0,theta_0,h = pose
		# Manda las consignas al modelo de la bicicleta y obtine su pose
		x_k1, y_k1, theta_k1 = self.kin_model(Vmax,delta, x_0, y_0, theta_0, h)
		# Calculo de los errores relativos
		e_y, _, in_cam = self.calc_err(R,x_k1,y_k1,theta_k1,side)
		# Calculo de los estados, la recompensa y demas
		P = np.array([x_k1,y_k1])
		rho, gamma = self.v_lidar(Obj,x_k1,y_k1,theta_k1)
		if (side==1): p0 = 0.0
		else: p0 = 1.0
		fl = 10.0*(np.pi/180.0)
		fr = 349.0*(np.pi/180.0)
		bl = 170.0*(np.pi/180.0)
		br = 190.0*(np.pi/180.0)
		if (gamma<fl) or (gamma>fr): p2 = -5.0
		if (br<=gamma<=fr): p2 = 0.0
		if (bl<=gamma<=br): p2 = 5.0
		if (fl<=gamma<=bl): p2 = 0.0
		if (rho<=1.5): p1 = 1.0		
		else: p1 = 0.0
		reward = (1.0-p0)*p1*p2-(p0)*0.5
		rho_disc, gamma_disc = self.disc_states(rho,gamma,Nrho,Ngamma)		
		states_disc = [rho_disc, gamma_disc]
		if (abs(e_y)>1.0) or (in_cam==False) or (rho<0.10): 
			done = False
			reward = -50.0
		else: done = True
		pose_k1 = x_k1, y_k1, theta_k1
		return states_disc, reward, done, pose_k1

	# Render
	def render(self,X,Y,Xr,Yr,Xl,Yl,Xc,Yc,Scar1_p): #,Scar2_p
		fig = plt.figure(0)
		ax = plt.axes()
		ax.plot(Xr,Yr,'b',Xc,Yc,'--b',Xl,Yl,'b')
		ax.plot(Scar1_p[:,0],Scar1_p[:,1],'k.', markersize=5) #,Scar2_p[:,0],Scar2_p[:,1],'k.'
		line, = ax.plot([], [],'r.', markersize=2) 
		def init(): 
			line.set_data([], [])
			return line,
		def animate(i):
			line.set_data(X[:i], Y[:i])
			return line,
		anim = FuncAnimation(fig, animate, init_func = init, frames = len(X), interval = 10, blit = True)	
		plt.show()
	
	# Render
	def render2(self,X,Y,Xr,Yr,Xl,Yl,Xc,Yc,Scar1_p,Scar2_p):
		fig = plt.figure(0)
		ax = plt.axes()
		ax.plot(Xr,Yr,'b',Xc,Yc,'--b',Xl,Yl,'b')
		ax.plot(Scar1_p[:,0],Scar1_p[:,1],'k.',Scar2_p[:,0],Scar2_p[:,1],'k.', markersize=5)
		line, = ax.plot([], [],'r.', markersize=2) 
		def init(): 
			line.set_data([], [])
			return line,
		def animate(i):
			line.set_data(X[:i], Y[:i])
			return line,
		anim = FuncAnimation(fig, animate, init_func = init, frames = len(X), interval = 10, blit = True)	
		plt.show()
	
#************************************************************************************
#*********************** CONTROLADOR OPTIMO PROPORCIONAL ****************************
#************************************************************************************
# Controlador Optimo Proporcional
	def optimal_control(self,e_y,e_th):
		# Calcula la ley de control
		self.Ky = 0.5
		self.Kth = 0.5
		# delta<0  Giro a la derecha
		# delta>0  Giro a la izquierda
		delta = -np.arctan(self.Ky*e_y+self.Kth*e_th)
		return delta
#************************************************************************************
#********************************** Q-LEARNING **************************************
#************************************************************************************
	# Discretizacion de estados
	def disc_states(self,rho,gamma,Nrho,Ngamma):
		# rho <-> {0.0,...,3.0} [m]
		# gamma <-> {-pi,...,pi} [rad]
		rho = float(Nrho*(rho/3.0))
		rho = int(round(rho))
		gamma = float(Ngamma*(gamma+np.pi)/(2*np.pi))
		gamma = int(round(gamma))
		if (rho<0): rho=0
		if (rho>Nrho-1): rho=Nrho-1
		if (gamma<0): gamma=0
		if (gamma>Ngamma-1): gamma=Ngamma-1
		return rho, gamma
		
	# Des-discretizacion de las acciones	
	def undisc_act(self,lane,Nlane):
		# lane <-> {-1,1}
		# 0 lane der
		# 1 lane izq
		lane = 1-2*lane
		return lane

	def Q_update(self,states_disc0,states_disc, lane_disc, q_table, belman_P, update):
		reward,DISCOUNT,LEARNING_RATE = belman_P
		rho0_disc, gamma0_disc = states_disc0
		rho_disc, gamma_disc = states_disc
		# Actualiza el tensor Q
		if (update==True):
			# Adquiere el "value" del estado/accion anterior		
			current_q = q_table[rho0_disc,gamma0_disc,lane_disc]												
			# Adquiere el "value" maximo (a lo largo del eje a) del estado obtenido
			max_future_q = np.amax(q_table[rho_disc,gamma_disc,:]) 			
			# Ecuacion de Bellman iterativa que calcula el "value"
			new_q = (1-LEARNING_RATE)*current_q+LEARNING_RATE*(reward + DISCOUNT * max_future_q) 
			# Actualiza el tensor Q con el resultado de la ec. de Bellman			
			q_table[rho0_disc,gamma0_disc,lane_disc] = new_q
		return q_table
			
			
			
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
