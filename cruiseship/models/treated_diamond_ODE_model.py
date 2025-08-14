import numpy as np
import pints
from scipy.integrate import odeint

class treatedDiamondODEModel(pints.ForwardModel):
    
    def simulate(self, parameters, times):
        # Unpack parameters
        bbarcp, cpp, b1, tpp, tcc, thetaa, thetap, chi, effp, effc, pintp, pintc = parameters

        Np = 2666
        Nc = 1045    
        # Initial conditions
        y0 = np.zeros(24)
        y0[0] = (1-pintp)*Np-1  # Initial no. susceptible passengers
        y0[1] = (1-pintc)*Nc    # Initial no. susceptible crew
        y0[10] = 1      # Initial no. symptomatic passengers with known onset time
        y0[22] = pintp*Np
        y0[23] = pintc*Nc

        # Define the ODE system for use with odeint
        # Note: odeint expects the function signature to be func(y, t, *args)
        def ode_system(y, t, bbarcp, cpp, b1, tpp, tcc, thetaa, thetap, chi, effp, effc):
            return self.the_ode_model(t, y, bbarcp, cpp, b1, tpp, tcc, thetaa, thetap, chi, effp, effc)
        
        # Solve the ODE using odeint
        sol = odeint(ode_system, y0, times, args=(bbarcp, cpp, b1, tpp, tcc, thetaa, thetap, chi, effp, effc))
        
        # Return only the values of interest, adjust indexing according to the output you're interested in
        # return sol[:,10:12] # for inference only
        return sol      # for modelling only
    
    def simulate_selected_outputs(self, parameters, times, selected_outputs):
        # Simulate the full model
        sol = self.simulate(parameters, times)
        # Return only the selected outputs
        return sol[:, selected_outputs]

    def n_parameters(self):
        # Return the dimension of the parameter vector (adjust this number according to your model)
        return 8  # bbarcp, cpp, b1, tpp, tcc, thetaa, thetap, chi
    
    def n_outputs(self):
        # Return the dimension of the parameter vector (adjust this number according to your model)
        return 2  #sol[:,10:12]

    @staticmethod
    def the_ode_model(t, y, bbarcp, cpp, b1, tpp, tcc, thetaa, thetap, chi, effp, effc):
        b2 = 80
        nu = 2 / 4.3
        ga = 1 / 5
        gp = 1 / 2.1
        gs = 1 / 2.9
        m = 0 if t < 17 else 0.7
        h = 1 / 7
        f = 199 / 314
        bbarcc = bbarcp
        bbarpc = bbarcp
        bbarpp = bbarcp
        ccc = 1
        cpc = cpp * 0.1
        ccp = cpc
        tpc = tpp
        tcp = tpp

        dy = np.zeros(24)
        a = +0.3
        a2 = -0.43
        bcp = (bbarcp + a2) * ccp * (1 - b1 / (1 + np.exp(-b2 * (t - tcp - a))))
        bcc = (bbarcc + a2) * ccc * (1 - b1 / (1 + np.exp(-b2 * (t - tcc - a))))
        bpc = (bbarpc + a2) * cpc * (1 - b1 / (1 + np.exp(-b2 * (t - tpc - a))))
        bpp = (bbarpp + a2) * cpp * (1 - b1 / (1 + np.exp(-b2 * (t - tpp - a))))

        # Define a dictionary with time intervals as keys and Ntests as values
        ntests_dict = {
            17: 0,
            18: 23,
            19: 64,
            20: 138,
            21: 3,
            22: 54,
            23: 43,
            24: 0,
            25: 17,
            26: 188,
            27: 0,
            28: 188,
            29: 257,
            30: 475,
            31: 658,
            32: 596,
            33: 45,
        }
        
        # Default value if t >= 33
        Ntests = 45
        
        # Find the correct Ntests value based on the time interval
        for key in sorted(ntests_dict.keys()):
            if t < key:
                Ntests = ntests_dict[key]
                break

        ft = Ntests / (np.sum(y[[0, 1, 2, 3, 6, 7, 8, 9, 16, 17, 18, 19, 22, 23]]))

        # Define dy equations
        dy[0] = -y[0] * (bpp * thetaa * y[8] + bpp * thetap * y[6] + bpp * y[10] + bpp * y[14]) / 2666 - \
                y[0] * (bpc * thetaa * y[9] + bpc * thetap * y[7] + bpc * y[11] + bpc * y[15]) / 1045
        dy[1] = -y[1] * (bcp * thetaa * y[8] + bcp * thetap * y[6] + bcp * y[10] + bcp * y[14]) / 2666 - \
                y[1] * (bcc * thetaa * y[9] + bcc * thetap * y[7] + bcc * y[11] + bcc * y[15]) / 1045
        dy[2] = y[0] * (bpp * thetaa * y[8] + bpp * thetap * y[6] + bpp * y[10] + bpp * y[14]) / 2666 + \
                y[0] * (bpc * thetaa * y[9] + bpc * thetap * y[7] + bpc * y[11] + bpc * y[15]) / 1045 + \
                y[22]*(1-effp)*(bpp*thetaa*y[8]+ bpp*thetap*y[6]+ bpp*y[10]+ bpp*y[14])/2666 + \
                y[22]*(1-effp)*(bpc*thetaa*y[9]+ bpc*thetap*y[7]+ bpc*y[11]+ bpc*y[15])/1045 - nu * y[2]
        dy[3] = y[1] * (bcp * thetaa * y[8] + bcp * thetap * y[6] + bcp * y[10] + bcp * y[14]) / 2666 + \
                y[1] * (bcc * thetaa * y[9] + bcc * thetap * y[7] + bcc * y[11] + bcc * y[15]) / 1045 + \
                y[23]*(1-effc)*(bcp*thetaa*y[8]+ bcp*thetap*y[6]+ bcp*y[10]+ bcp*y[14])/2666 + \
                y[23]*(1-effc)*(bcc*thetaa*y[9]+bcc*thetap*y[7]+bcc*y[11]+bcc*y[15])/1045 - nu * y[3]
        dy[4] = nu * y[2] - nu * y[4]
        dy[5] = nu * y[3] - nu * y[5]
        dy[6] = (1 - chi) * nu * y[4] - ft * y[6] - gp * y[6]
        dy[7] = (1 - chi) * nu * y[5] - ft * y[7] - gp * y[7]
        dy[8] = chi * nu * y[4] - ga * y[8] - ft * y[8]
        dy[9] = chi * nu * y[5] - ga * y[9] - ft * y[9]
        dy[10] = f * gp * y[6] - gs * y[10] - m * y[10]
        dy[11] = f * gp * y[7] - gs * y[11] - m * y[11]
        dy[12] = m * y[10] + m * y[14]
        dy[13] = m * y[11] + m * y[15]
        dy[14] = (1 - f) * gp * y[6] - m * y[14] - gs * y[14]
        dy[15] = (1 - f) * gp * y[7] - m * y[15] - gs * y[15]
        dy[16] = gs * y[10] + ga * y[8] + gs * y[14] - h * y[16] - ft * y[16]
        dy[17] = gs * y[11] + ga * y[9] + gs * y[15] - h * y[17] - ft * y[17]
        dy[18] = h * y[16]
        dy[19] = h * y[17]
        dy[20] = ft * y[6] + ft * y[16] + ft * y[8]
        dy[21] = ft * y[7] + ft * y[17] + ft * y[9]
        dy[22] = -y[22]*(1-effp)*(bpp*thetaa*y[8]+ bpp*thetap*y[6]+ bpp*y[10]+ bpp*y[14])/2666 - \
                y[22]*(1-effp)*(bpc*thetaa*y[9]+ bpc*thetap*y[7]+ bpc*y[11]+ bpc*y[15])/1045
        dy[23] = -y[23]*(1-effc)*(bcp*thetaa*y[8]+ bcp*thetap*y[6]+ bcp*y[10]+ bcp*y[14])/2666 - \
                y[23]*(1-effc)*(bcc*thetaa*y[9]+bcc*thetap*y[7]+bcc*y[11]+bcc*y[15])/1045
    
        return dy