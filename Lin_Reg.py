# a simple Linear Regression class
class LinReg:
    
    def __init__(self):
        
        # initialize all relevant parameters
        self.beta0 = None
        self.beta1 = None
        self.slope_err = None
        self.intcpt_err = None
        self.R_sqr = None
        self.obsv = None
        self.df = None
        self.tbeta1 = None
        self.sgfnt_beta1 = None
    
    def fit_data(self,Xlist,Ylist):
    	# do not proceed if the two lists are unequal, or either one is empty
		if (len(Xlist)!=len(Ylist)) or (Xlist==[]) or (Ylist==[]):
			pass
		# only proceed if the two non-empty lists are equal in length
		elif len(Xlist)==len(Ylist):
			# no. of observations is the length of either list
			self.obsv = len(Xlist)
			
			# calculate slope(beta1) and intercept(beta0)
			x_avg = sum(Xlist)/float(len(Xlist))
			y_avg = sum(Ylist)/float(len(Ylist))
			beta1_num_sum = 0
			beta1_den_sum = 0
			for i in range(len(Xlist)):
				beta1_num_sum += (Xlist[i]-x_avg)*(Ylist[i]-y_avg)
				beta1_den_sum += (Xlist[i]-x_avg)**2
			self.beta1 = beta1_num_sum/float(beta1_den_sum)
			self.beta0 = y_avg - self.beta1*x_avg
			
			# calculate standard error in slope
			Yhat_list = [self.beta0 + self.beta1*x for x in Xlist]
			e_list = []
			e_list_sqr = []
			for i in range(len(Ylist)):
				e_list.append(Ylist[i] - Yhat_list[i])
				e_list_sqr.append((Ylist[i] - Yhat_list[i])**2)
			e_avg = sum(e_list)/float(len(e_list))
			e_var_num = 0
			for i in e_list:
				e_var_num += (i - e_avg)**2
			# estimate error variance by sample variance s^2_(n-1)
			e_var = e_var_num/float(len(e_list)-1)
			slope_err_beta1_sqr = e_var/float(beta1_den_sum)
			self.slope_err = math.sqrt(slope_err_beta1_sqr)
			
			# calculate standard error in intercept
			beta0_err_sqr = e_var*((1/float(len(Ylist))) + ((x_avg)**2)/float(beta1_den_sum))
			self.intcpt_err = math.sqrt(beta0_err_sqr)
			
			# calculate R-squared
			RSS = 0
			for i in e_list_sqr:
				RSS += i
			TSS = 0
			for i in Ylist:
				TSS += (i-y_avg)**2
			self.R_sqr = (TSS - RSS)/float(TSS)
			
			# calculate t-statistic with n-2 degrees of freedom
			# null hypothesis=slope is zero, i.e. no relationship between Y and X
			self.df = len(Xlist) - 2
			self.tbeta1 = self.beta1/float(self.slope_err)
			# produce standard t-table
			t_table = {1:12.71, 2:4.303, 3:3.182, 4:2.776, 5:2.571,
			6:2.447, 7:2.365, 8:2.306, 9:2.262, 10:2.228,
			11:2.201, 12:2.179, 13:2.160, 14:2.145, 15:2.131,
			16:2.120, 17:2.110, 18: 2.101, 19:2.093, 20: 2.086,
			21:2.080, 22:2.074, 23:2.069, 24:2.064, 25:2.060,
			26:2.056, 27:2.052, 28:2.048, 29:2.045, 30:2.042,
			40:2.021, 60:2.000, 80:1.990, 100:1.984, 1000:1.962,
			10000:1.96}
			conf_intv = None
			# calculate 95% confidence length
			if self.df>0 and self.df<=30:
				conf_intv = t_table[self.df]
			elif self.df>30 and self.df<=40:
				conf_intv = t_table[40]
			elif self.df>40 and self.df<=60:
				conf_intv = t_table[60]
			elif self.df>60 and self.df<=80:
				conf_intv = t_table[80]
			elif self.df>80 and self.df<=300:
				conf_intv = t_table[100]
			elif self.df>300 and self.df<=3000:
				conf_intv = t_table[1000]
			elif self.df>3000:
				conf_intv = t_table[10000]
			
			# significant/insignificant t-statistic
			if abs(self.tbeta1)>conf_intv:
				self.sgfnt_beta1 = 'Significant'
			elif abs(self.tbeta1)<=conf_intv:
				self.sgfnt_beta1 = 'Insignificant'