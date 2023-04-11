import numpy as np

class Calibration():
    def __init__(self):
        self.name = 'calibration'
        self.C = 0.1

    def gen_eps(self, n, dist):
        if dist=='Uniform1':
            return np.random.uniform(0.05, 0.5, n)
        elif dist=='Uniform2':
            return np.random.uniform(0.05, 1, n)
        elif dist == 'Gauss1':
            eps0 = np.random.normal(0.1, 1, n)
            eps0 = np.maximum(eps0, 0.05)
            eps0 = np.minimum(eps0, 0.5)
            return eps0
        elif dist == 'Gauss2':
            eps0 = np.random.normal(0.2, 1, n)
            eps0 = np.maximum(eps0, 0.05)
            eps0 = np.minimum(eps0, 1)
            return eps0
        elif dist == 'MixGauss1':
            step = int(n*0.9)
            eps_low = np.random.normal(0.1, 1, step)
            eps_high = np.random.normal(0.5, 1, n-step)
            eps0 = np.concatenate((eps_low, eps_high))
            eps0 = np.maximum(eps0, 0.05)
            eps0 = np.minimum(eps0, 0.5)
            return eps0
        elif dist == 'MixGauss2':
            step = int(n*0.9)
            eps_low = np.random.normal(0.2, 1, step)
            eps_high = np.random.normal(1, 1, n-step)
            eps0 = np.concatenate((eps_low, eps_high))
            eps0 = np.maximum(eps0, 0.05)
            eps0 = np.minimum(eps0, 1)
            return eps0        
        else:
            return 0

    def clip_laplace(self, grad, epsilon, Clip_bound):
        C = Clip_bound
        b = 2*C/epsilon
        u = grad
        exp_part = np.exp((-C-u)/b)
        S = 1 - 0.5 * np.exp((-C+u)/b) - 0.5 * exp_part
        p = np.random.uniform(0, 1, grad.shape[0])
        sp = S*p
        
        step_point = np.sign(sp - (0.5 - 0.5*exp_part))
        X = u - step_point * b *np.log(1- 2*np.abs(sp - 0.5 + 0.5*exp_part))

        return X


    def laplace(self, updates, C, epsilon):
        '''
        inject laplacian noise to a vector
        '''
        b = C * 1.0 / epsilon
        updates += np.random.laplace(loc=0, scale=b, size=updates.shape)
        return updates


    def E_noisy_mean_g(self, eps, mean_g, C):
        b = 2*C/eps
        e1 = np.exp((-C-mean_g)/b)
        e2 = np.exp((-C+mean_g)/b)
        E = ((C+b)*(e1-e2)+2*mean_g) / (2-e1-e2)
        return np.mean(E)

    def MSE(self, n_g, n_time):
        # MSE = np.sum((xi-x_bar)**2) / n
        # return MSE
        sum_th = 0
        sum_exp = 0
        for i in range(n_time):
            # g = np.random.normal(-0.0,1, n_g)
            g = np.random.uniform(-self.C, self.C, n_g)
            g = np.maximum(g, -self.C)
            g = np.minimum(g, self.C)
            eps = self.gen_eps(0.05,1,n_g,'Uniform2')
            mean_g = np.mean(g)

            mean_g_arr = np.array([mean_g]*n_g)
            E_th = self.E_noisy_mean_g(mean_g, eps, 0.1)
            E_exp = np.mean(self.clip_laplace(mean_g_arr, eps, C))

            E_noisy_g = np.mean(self.clip_laplace(g, eps, C))

            sum_th += (E_th - E_noisy_g)**2
            sum_exp += (E_exp - E_noisy_g)**2
        return sum_th/n_time, sum_exp/n_time
    
    def Error(self, n_g, dist):
        # g = np.random.uniform(-self.C, self.C, n_g)
        # g = np.random.exponential(1, n_g)
        g = np.array([-self.C]*(int(n_g*0.1)) + [self.C]*(int(n_g*0.9)))
        g = np.maximum(g, -self.C)
        g = np.minimum(g, self.C)
        eps = self.gen_eps(n_g,dist)
        mean_g = np.mean(g)
        # print(mean_g)
        # mean_g = np.median(g) #use median value seems better for the settings where values does not concentrated around mean
        mean_g_arr = np.array([mean_g]*n_g)
        E_th = self.E_noisy_mean_g(mean_g, eps, 0.1)
        E_exp = np.mean(self.clip_laplace(mean_g_arr, eps, C))

        E_noisy_g = np.mean(self.clip_laplace(g, eps, C))
        return (E_th-E_noisy_g), (E_exp-E_noisy_g)


# n=10000
C= 0.1

### for one time test
# g = np.random.uniform(-0.08,0.04, n)
# g = np.random.normal(0.05,1, n)

# g = np.maximum(g, -C)
# g = np.minimum(g, C)
# eps = gen_eps(0.05,1,n,'Uniform')
# mean_g = np.mean(g)

# noisy_g = clip_laplace(g, eps, C)
# E_th = E_noisy_mean_g(mean_g, eps, 0.1)

# mean_g = np.array([mean_g]*n)
# E_exp = np.mean(clip_laplace(mean_g, eps, C))
# print('noisy_g:{}, E_theory:{}, E_emperical:{}'.format(np.mean(noisy_g), E_th, E_exp))

### for n times in one distribution
# MSE_th, MSE_exp = MSE(10000, 1000)
# print('n_time:{}, MSE_th:{}, MSE_exp:{}'.format(1000, MSE_th, MSE_exp))




