import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import truncnorm
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix

def simulate_covid_spread(nr, dt, Tmax, A, s1, p_sev, g_mild, g_sev, efficacy, t_interv, R0, mean_prop, std_prop=None, t_end_conf=None, weighted='nonweighted', percentile=None):

    if t_end_conf == None:
        t_end_conf = Tmax

    if weighted == 'nonweighted':
        Tadj = unweighted_contact_matrix(A)
    elif weighted == 'weighted' and percentile != None:
        Tadj = weighted_contact_matrix(A, percentile)
    elif weighted == 'weighted' and percentile == None:
        ValueError('Percentile is unspecified. Specify a percentile between 0 and 100.')
    else:
        ValueError('Contact matrix weighting is specified incorrectly. Please specify weighted="weighted" and percentile="percentile", or weighted="unweighted".')

 

    # Probability to develop mild infection
    p_mild = 1 - p_sev

    # In treated individuals the infection rate is reduced by efficacy_BlockInf per cent
    efficacy_BlockInf = efficacy 

    # In treated individuals the probability of developing severe infection is reduced proportionally to the efficacy
    efficacy_BlockSevInf = efficacy

    # Probability to develop severe infection if protected
    pp_sev=(1-efficacy_BlockSevInf)*p_sev
    # Probability to develop mild infection if protected
    pp_mild=1-pp_sev
    b=R0*g_sev*g_mild/((1-p_mild)*g_mild+p_mild*g_sev)

    T=b*Tadj
    IsimMild=np.zeros((nr,int(Tmax/dt)))
    IsimSev=np.zeros((nr,int(Tmax/dt)))
    IsimTot=np.zeros((nr,int(Tmax/dt)))

    N=T.shape[0]

    sim = 1
    eps = np.finfo(float).eps

    new_time_points = np.arange(0, Tmax, dt)
    cumul_infections_all_sims = []
    cumul_mild_infections_all_sims = []
    cumul_sev_infections_all_sims = []
    cumul_unprotected_exposures_all_sims = []
    cumul_protected_exposures_all_sims = []

    if std_prop == None:
        prop = mean_prop
        props = np.repeat(prop, sim)
    else:
        lower_bound = 0
        upper_bound = 1
        l, u = (lower_bound - mean_prop) / std_prop, (upper_bound - mean_prop) / std_prop
        props = truncnorm(l, u, loc=mean_prop, scale=std_prop).rvs(nr)

    while sim <= nr:

        if std_prop != None:
            prop = props[sim-1]

        S_vec = np.ones(N)
        Sp_vec = np.zeros(N)
        E_vec = np.zeros(N)
        Ep_vec = np.zeros(N)
        Imild_vec = np.zeros(N)
        Isev_vec = np.zeros(N)
        R_vec = np.zeros(N)

        # Randomly select a proportion of the population to be protected from day 0
        randprotectVEC = np.random.choice(np.arange(0,N), size=int(prop*N), replace=False)
        S_vec[randprotectVEC] = 0
        Sp_vec[randprotectVEC] = 1


        InitInf=2;  # 'InitInf' initial individuals that get infection
        randpermVEC = np.random.choice(np.arange(0, N), size=InitInf, replace=False)

        InitInf_Mild=1 # Out of the total initial infected, InitInf_Mild are mild
        RI_mild = randpermVEC[:InitInf_Mild]

        InitInf_Sev=InitInf-InitInf_Mild # Out of the total initial infected, InitInf_Sev are severe
        RI_sev=randpermVEC[InitInf-InitInf_Mild:InitInf]   


        S_vec[RI_mild] = 0 
        S_vec[RI_sev] = 0 
        Imild_vec[RI_mild] = 1 
        Isev_vec[RI_sev] = 1

        # Total sum in each compartment
        S_tot = np.sum(S_vec)
        E_tot = np.sum(E_vec)
        Sp_tot = np.sum(Sp_vec)
        Ep_tot = np.sum(Ep_vec)
        Imild_tot = np.sum(Imild_vec)
        Isev_tot = np.sum(Isev_vec)
        new_I_tot = Imild_tot + Isev_tot
        new_Imild_tot = Imild_tot
        new_Isev_tot = Isev_tot
        new_E_tot = E_tot
        new_Ep_tot = Ep_tot
        R_tot = np.sum(R_vec)
        
        Out = []
        Out.append([np.sum(S_vec), np.sum(E_vec), np.sum(Sp_vec), np.sum(Ep_vec), np.sum(Imild_vec), np.sum(Isev_vec), np.sum(R_vec), 0, new_Imild_tot, new_Isev_tot, new_I_tot, new_E_tot, new_Ep_tot])
        present=0   
        event=0
        event_max=10000000000000   

        M = (Imild_vec + Isev_vec) * T
        SE_vec = S_vec * M
        EImild_vec = s1 * p_mild * E_vec
        EIsev_vec = s1 * p_sev * E_vec

        ImildR_vec = g_mild * Imild_vec
        IsevR_vec = g_sev * Isev_vec

        qs = 0
        qe = 0
        ls = 0
        le = 0

        SSp_vec = qs * S_vec
        EEp_vec = qe * E_vec 
        SpS_vec = ls * Sp_vec
        EpE_vec = le * Ep_vec
        SpEp_vec = Sp_vec * (1 - efficacy_BlockInf) * M
        EpImild_vec = s1 * pp_mild * Ep_vec
        EpIsev_vec = s1 * pp_sev * Ep_vec

        SErates = max(np.sum(SE_vec), eps)
        EImildrates = max(s1 * p_mild * np.sum(E_vec), eps)
        EIsevrates = max(s1 * p_sev * np.sum(E_vec), eps)
        ImildRrates = max(g_mild * np.sum(Imild_vec), eps)
        IsevRrates = max(g_sev * np.sum(Isev_vec), eps)
        SSprates = max(np.sum(SSp_vec), eps)
        EEprates = max(np.sum(EEp_vec), eps)
        SpSrates = max(np.sum(SpS_vec), eps)
        EpErates = max(np.sum(EpE_vec), eps)
        SpEprates = max(np.sum(SpEp_vec), eps)
        EpImildrates = max(s1 * pp_mild * np.sum(Ep_vec), eps)
        EpIsevrates = max(s1 * pp_sev * np.sum(Ep_vec), eps)

        infections=0
        present=0

        ind_SSp=0
        SSp_indices=[]

        ind_EEp=0
        EEp_indices=[]


        while event < event_max and present <= Tmax:

            # if Out[-1][7] <= t_interv:
            #     #Rate at which treated susceptibles become 'fully' susceptible
            #     ls = 0
            #     #Rate at which treated exposed become 'fully' exposed
            #     le = 0
            # else:  
            #     # Rate at which treated susceptibles become 'fully' susceptible
            #     ls=10000000000000000000
            #     # Rate at which treated exposed become 'fully' exposed
            #     le=10000000000000000000

            event += 1
            r2, r3, r4, r5 = np.random.rand(4)

            Sumrates = SErates+EImildrates+EIsevrates+ImildRrates+IsevRrates + SSprates+SpSrates+EEprates+EpErates+SpEprates+EpImildrates+EpIsevrates

            time=-((1/Sumrates)*np.log(r5))
            present=present+time

            v1=SErates/Sumrates
            v2=EImildrates/Sumrates
            v3=EIsevrates/Sumrates
            v4=ImildRrates/Sumrates
            v5=IsevRrates/Sumrates

            v6=SSprates/Sumrates
            v7=SpSrates/Sumrates
            v8=EEprates/Sumrates
            v9=EpErates/Sumrates
            v10=SpEprates/Sumrates
            v11=EpImildrates/Sumrates
            v12=EpIsevrates/Sumrates

            if r2 <= v1 and S_tot > 0 and present <= t_end_conf:
                A1=abs(SE_vec/SErates)
        
                edges = np.concatenate(([0], np.cumsum(A1)))
                f_index = np.digitize(r3, edges, right=True)
                f_index -= 1
                # Ensure f_index is within the valid range
                f_index = max(0, min(f_index, N - 1))
                S_vec[f_index] = 0
                E_vec[f_index] = 1

                S_tot -= 1
                E_tot += 1
                new_E_tot += 1
                infections += 1


            elif r2>v1 and r2<=v1+v2 and E_tot>0:
                
                A2=EImild_vec/EImildrates
        
                edges = np.concatenate(([0], np.cumsum(A2)))
                f_index = np.digitize(r3, edges, right=True)
                f_index -= 1
                # Ensure f_index is within the valid range
                f_index = max(0, min(f_index, N - 1))
                E_vec[f_index] = 0
                Imild_vec[f_index] = 1
                M += T[f_index,:]

                    
                E_tot -= 1
                Imild_tot += 1
                new_Imild_tot += 1
                new_I_tot += 1

            elif r2>v1 and r2>v1+v2 and r2<=v1+v2+v3 and E_tot>0:
                
                A3=EIsev_vec/EIsevrates
        
                edges = np.concatenate(([0], np.cumsum(A3)))
                f_index = np.digitize(r3, edges, right=True)
                f_index -= 1
                # Ensure f_index is within the valid range
                f_index = max(0, min(f_index, N - 1))
                E_vec[f_index] = 0
                Isev_vec[f_index] = 1
                M += T[f_index,:]
            
                E_tot -= 1
                Isev_tot += 1
                new_Isev_tot += 1
                new_I_tot += 1

            elif r2>v1 and r2>v1+v2 and r2>v1+v2+v3 and r2<=v1+v2+v3+v4 and Imild_tot>0:
                
                A4=ImildR_vec/ImildRrates
        
                edges = np.concatenate(([0], np.cumsum(A4)))
                f_index = np.digitize(r3, edges, right=True)
                f_index -= 1
                # Ensure f_index is within the valid range
                f_index = max(0, min(f_index, N - 1))                
                Imild_vec[f_index] = 0
                R_vec[f_index] = 1
                M -= T[f_index,:]
        
                Imild_tot -= 1
                R_tot += 1

            elif r2>v1 and r2>v1+v2 and r2>v1+v2+v3 and r2>v1+v2+v3+v4 and r2<=v1+v2+v3+v4+v5 and Isev_tot>0:
                
                A5=IsevR_vec/IsevRrates
        
                edges = np.concatenate(([0], np.cumsum(A5)))
                f_index = np.digitize(r3, edges, right=True)
                f_index -= 1
                # Ensure f_index is within the valid range
                f_index = max(0, min(f_index, N - 1))
                Isev_vec[f_index] = 0
                R_vec[f_index] = 1
                M -= T[f_index,:]
            
                Isev_tot -= 1
                R_tot += 1

            elif r2>v1 and r2>v1+v2 and r2>v1+v2+v3 and r2>v1+v2+v3+v4 and r2>v1+v2+v3+v4+v5 and r2<=v1+v2+v3+v4+v5+v6 and S_tot>0 and len(SSp_indices) + len(EEp_indices) < prop*N and Out[-1][7]<=t_interv:
                
                A6=SSp_vec/SSprates
        
                edges = np.concatenate(([0], np.cumsum(A6)))
                f_index = np.digitize(r3, edges, right=True)
                f_index -= 1
                # Ensure f_index is within the valid range
                f_index = max(0, min(f_index, N - 1))
                S_vec[f_index] = 0
                Sp_vec[f_index] = 1

                if (f_index not in SSp_indices) and (f_index not in EEp_indices): # we save only those that are treated for the first time
                    SSp_indices.append(f_index)
                    ind_SSp += 1
            
                S_tot -= 1
                Sp_tot += 1


            elif r2>v1 and r2>v1+v2 and r2>v1+v2+v3 and r2>v1+v2+v3+v4 and r2>v1+v2+v3+v4+v5 and r2>v1+v2+v3+v4+v5+v6 and r2<=v1+v2+v3+v4+v5+v6+v7 and Sp_tot>0:
                
                A7=SpS_vec/SpSrates
        
                edges = np.concatenate(([0], np.cumsum(A7)))
                f_index = np.digitize(r3, edges, right=True)
                f_index -= 1
                # Ensure f_index is within the valid range
                f_index = max(0, min(f_index, N - 1))
                Sp_vec[f_index] = 0
                S_vec[f_index] = 1
            
                Sp_tot -= 1
                S_tot += 1


            elif r2>v1 and r2>v1+v2 and r2>v1+v2+v3 and r2>v1+v2+v3+v4 and r2>v1+v2+v3+v4+v5 and r2>v1+v2+v3+v4+v5+v6 and r2>v1+v2+v3+v4+v5+v6+v7 and r2<=v1+v2+v3+v4+v5+v6+v7+v8 and E_tot>0 and  len(SSp_indices) + len(EEp_indices) < prop*N and Out[-1][7]<=t_interv:
                
                A8=EEp_vec/EEprates
        
                edges = np.concatenate(([0], np.cumsum(A8)))
                f_index = np.digitize(r3, edges, right=True)
                f_index -= 1
                # Ensure f_index is within the valid range
                f_index = max(0, min(f_index, N - 1))
                E_vec[f_index] = 0
                Ep_vec[f_index] = 1

                if (f_index not in SSp_indices) and (f_index not in EEp_indices): # we save only those that are treated for the first time
                    
                    EEp_indices.append(f_index)
                    ind_EEp += 1
            
                E_tot -= 1
                Ep_tot += 1

            elif r2>v1 and r2>v1+v2 and r2>v1+v2+v3 and r2>v1+v2+v3+v4 and r2>v1+v2+v3+v4+v5 and r2>v1+v2+v3+v4+v5+v6 and r2>v1+v2+v3+v4+v5+v6+v7 and r2>v1+v2+v3+v4+v5+v6+v7+v8 and r2<=v1+v2+v3+v4+v5+v6+v7+v8+v9 and Ep_tot>0:
                
                A9=EpE_vec/EpErates
        
                edges = np.concatenate(([0], np.cumsum(A9)))
                f_index = np.digitize(r3, edges, right=True)
                f_index -= 1
                # Ensure f_index is within the valid range
                f_index = max(0, min(f_index, N - 1))
                Ep_vec[f_index] = 0
                E_vec[f_index] = 1
            
                Ep_tot -= 1
                E_tot += 1

            elif r2>v1 and r2>v1+v2 and r2>v1+v2+v3 and r2>v1+v2+v3+v4 and r2>v1+v2+v3+v4+v5 and r2>v1+v2+v3+v4+v5+v6 and r2>v1+v2+v3+v4+v5+v6+v7 and r2>v1+v2+v3+v4+v5+v6+v7+v8 and r2>v1+v2+v3+v4+v5+v6+v7+v8+v9 and r2<=v1+v2+v3+v4+v5+v6+v7+v8+v9+v10 and Sp_tot>0 and present <= t_end_conf:
                
                A10 = abs(SpEp_vec/SpEprates)
        
                edges = np.concatenate(([0], np.cumsum(A10)))
                f_index = np.digitize(r3, edges, right=True)
                f_index -= 1
                # Ensure f_index is within the valid range
                f_index = max(0, min(f_index, N - 1))
                Sp_vec[f_index] = 0
                Ep_vec[f_index] = 1
            
                Sp_tot -= 1
                Ep_tot += 1
                new_Ep_tot += 1


            elif r2>v1 and r2>v1+v2 and r2>v1+v2+v3 and r2>v1+v2+v3+v4 and r2>v1+v2+v3+v4+v5 and r2>v1+v2+v3+v4+v5+v6 and r2>v1+v2+v3+v4+v5+v6+v7 and r2>v1+v2+v3+v4+v5+v6+v7+v8 and r2>v1+v2+v3+v4+v5+v6+v7+v8+v9 and r2>v1+v2+v3+v4+v5+v6+v7+v8+v9+v10 and r2<=v1+v2+v3+v4+v5+v6+v7+v8+v9+v10+v11 and Ep_tot>0:
                
                A11=EpImild_vec/EpImildrates
        
                edges = np.concatenate(([0], np.cumsum(A11)))
                f_index = np.digitize(r3, edges, right=True)
                f_index -= 1
                # Ensure f_index is within the valid range
                f_index = max(0, min(f_index, N - 1))
                Ep_vec[f_index] = 0
                Imild_vec[f_index] = 1
                M += T[f_index,:]
                    
                Ep_tot -= 1
                Imild_tot += 1
                new_Imild_tot += 1
                new_I_tot += 1

            elif r2>v1 and r2>v1+v2 and r2>v1+v2+v3 and r2>v1+v2+v3+v4 and r2>v1+v2+v3+v4+v5 and r2>v1+v2+v3+v4+v5+v6 and r2>v1+v2+v3+v4+v5+v6+v7 and r2>v1+v2+v3+v4+v5+v6+v7+v8 and r2>v1+v2+v3+v4+v5+v6+v7+v8+v9 and r2>v1+v2+v3+v4+v5+v6+v7+v8+v9+v10 and r2>v1+v2+v3+v4+v5+v6+v7+v8+v9+v10+v11 and r2<=v1+v2+v3+v4+v5+v6+v7+v8+v9+v10+v11+v12 and Ep_tot>0:
                
                A12=EpIsev_vec/EpIsevrates
        
                edges = np.concatenate(([0], np.cumsum(A12)))
                f_index = np.digitize(r3, edges, right=True)
                f_index -= 1
                # Ensure f_index is within the valid range
                f_index = max(0, min(f_index, N - 1))
                Ep_vec[f_index] = 0
                Isev_vec[f_index] = 1
                M += T[f_index,:]

                    
                Ep_tot -= 1
                Isev_tot += 1
                new_Isev_tot += 1
                new_I_tot += 1

            M = np.squeeze(np.asarray(M))
            SE_vec=S_vec*M
            EImild_vec=s1*p_mild*(E_vec)
            EIsev_vec=s1*p_sev*(E_vec)
            ImildR_vec=g_mild*(Imild_vec)
            IsevR_vec=g_sev*(Isev_vec)
            SSp_vec=qs*(S_vec)
            EEp_vec=qe*(E_vec)
            SpS_vec=ls*(Sp_vec)
            EpE_vec=le*(Ep_vec)
            SpEp_vec=Sp_vec*(1 - efficacy_BlockInf)*M
            EpImild_vec=s1*pp_mild*(Ep_vec)
            EpIsev_vec=s1*pp_sev*(Ep_vec)

            SErates=max(np.sum(SE_vec),eps)
            EImildrates=max(s1*p_mild*sum(E_vec),eps)
            EIsevrates=max(s1*p_sev*sum(E_vec),eps)
            ImildRrates=max(g_mild*sum(Imild_vec),eps)
            IsevRrates=max(g_sev*sum(Isev_vec),eps)

            SSprates=max(np.sum(SSp_vec),eps)
            EEprates=max(np.sum(EEp_vec),eps)
            SpSrates=max(np.sum(SpS_vec),eps)
            EpErates=max(np.sum(EpE_vec),eps)
            SpEprates=max(np.sum(SpEp_vec),eps)
            EpImildrates=max(s1*pp_mild*sum(Ep_vec),eps)
            EpIsevrates=max(s1*pp_sev*sum(Ep_vec),eps)

            QQ=[S_tot, E_tot, Sp_tot, Ep_tot, Imild_tot, Isev_tot, R_tot, present, new_Imild_tot, new_Isev_tot, new_I_tot, new_E_tot, new_Ep_tot]
            Out.append(QQ)
            new_Imild_tot = 0
            new_Isev_tot = 0
            new_I_tot = 0
            new_E_tot = 0
            new_Ep_tot = 0

        Out = np.asarray(Out)
        # Check if the last element in the 7th row (6th row in Python's 0-based indexing) is greater than N/5
        if Out[-1,6] > N / 5:
            # Crea te an interpolation function
            # 'linear' specifies linear interpolation; you can also use 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', etc.
            mild_interp_func = interp1d(Out[:, 7], Out[:, 4], kind='linear', bounds_error=False, fill_value=np.nan)
            sev_interp_func = interp1d(Out[:, 7], Out[:, 5], kind='linear', bounds_error=False, fill_value=np.nan)
            
            cumul_mild_infections = np.cumsum(Out[:,8])
            cumul_sev_infections = np.cumsum(Out[:,9])
            cumul_infections = np.cumsum(Out[:,10])
            cumul_unprotected_exposures = np.cumsum(Out[:,11])
            cumul_protected_exposures = np.cumsum(Out[:,12])

            cumul_interp_func = interp1d(Out[:, 7], cumul_infections, kind='linear', bounds_error=False, fill_value=np.nan)
            cumul_mild_interp_func = interp1d(Out[:, 7], cumul_mild_infections, kind='linear', bounds_error=False, fill_value=np.nan)
            cumul_sev_interp_func = interp1d(Out[:, 7], cumul_sev_infections, kind='linear', bounds_error=False, fill_value=np.nan)
            cumul_unprotected_interp_func = interp1d(Out[:, 7], cumul_unprotected_exposures, kind='linear', bounds_error=False, fill_value=np.nan)
            cumul_protected_interp_func = interp1d(Out[:, 7], cumul_protected_exposures, kind='linear', bounds_error=False, fill_value=np.nan)

            # Use the interpolation function to get resampled data at new time points
            mild_resampled_data = mild_interp_func(new_time_points)
            sev_resampled_data = sev_interp_func(new_time_points)
            cumul_resampled_data = cumul_interp_func(new_time_points)
            cumul_mild_resampled_data = cumul_mild_interp_func(new_time_points)
            cumul_sev_resampled_data = cumul_sev_interp_func(new_time_points)
            cumul_unprotected_resampled_data = cumul_unprotected_interp_func(new_time_points)
            cumul_protected_resampled_data = cumul_protected_interp_func(new_time_points)
            
            IsimMild[sim-1,:] = mild_resampled_data
            IsimSev[sim-1,:] = sev_resampled_data
            IsimTot[sim-1,:] = mild_resampled_data + sev_resampled_data
            cumul_infections_all_sims.append(cumul_resampled_data)
            cumul_mild_infections_all_sims.append(cumul_mild_resampled_data)
            cumul_sev_infections_all_sims.append(cumul_sev_resampled_data)
            cumul_unprotected_exposures_all_sims.append(cumul_unprotected_resampled_data)
            cumul_protected_exposures_all_sims.append(cumul_protected_resampled_data)
            sim += 1

    return IsimMild, IsimSev, IsimTot, cumul_mild_infections_all_sims, cumul_sev_infections_all_sims, cumul_infections_all_sims, cumul_unprotected_exposures_all_sims, cumul_protected_exposures_all_sims, props, N


def weighted_contact_matrix(data, percentile):

    # Get all unique IDs from the dataset
    all_ids = np.unique(data[['ID1', 'ID2']].values)

    # Count the number of interactions between each pair
    interaction_counts = data.groupby(['ID1', 'ID2']).size().reset_index(name='count')

    # Calculate the threshold based on the specified percentile
    threshold = np.percentile(interaction_counts['count'].values, percentile)

    # Initialize an empty matrix with all IDs
    full_matrix = pd.DataFrame(index=all_ids, columns=all_ids, dtype=float).fillna(0)

    # Apply the threshold to scale interaction counts and update the full matrix
    for _, row in interaction_counts.iterrows():
        weight = 1 if row['count'] >= threshold else row['count'] / threshold
        full_matrix.at[row['ID1'], row['ID2']] = float(weight)
        full_matrix.at[row['ID2'], row['ID1']] = float(weight)  # Ensure symmetry

    # Fill the diagonal with 1s as each node has maximum interaction with itself
    np.fill_diagonal(full_matrix.values, 1)

    return csr_matrix(full_matrix.values)

def unweighted_contact_matrix(data):
    # Get all unique IDs from the dataset
    all_ids = np.unique(data[['ID1', 'ID2']].values)

    interaction_pairs = data.groupby(['ID1', 'ID2']).size().reset_index(name='count')[['ID1', 'ID2']]

    # Initialize an empty matrix with all IDs
    full_matrix = pd.DataFrame(index=all_ids, columns=all_ids, dtype=int).fillna(0)

    # Update the full matrix to 1 if there has been any interaction
    for _, row in interaction_pairs.iterrows():
        full_matrix.at[row['ID1'], row['ID2']] = 1
        full_matrix.at[row['ID2'], row['ID1']] = 1  # Ensure symmetry

    # Fill the diagonal with 1s as each node has maximum interaction with itself
    np.fill_diagonal(full_matrix.values, 1)

    return csr_matrix(full_matrix.values)
