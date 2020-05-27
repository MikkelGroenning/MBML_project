def model(T, T_forecast, obs=None):
    beta1 = pyro.sample(name="beta_1", fn=dist.Normal(loc=0, scale=1.))
    beta2 = pyro.sample(name="beta_2", fn=dist.Normal(loc=0., scale=1.))
    tau = pyro.sample(name="tau", fn=dist.HalfCauchy(scale=3.))
    sigma = pyro.sample(name="sigma", fn=dist.HalfCauchy(scale=3.))
    z_prev1 = pyro.sample(name="z_1", fn=dist.Normal(loc=0, scale=3.))
    z_prev2 = pyro.sample(name="z_2", fn=dist.Normal(loc=0, scale=3.))
    
    Z = [z_prev1, z_prev2]
    Y = []
    for t in range(2, T):
        z_t_mean = beta1*z_prev1 + beta2*z_prev2
        z_t = pyro.sample(name="z_%d"%(t+1), fn=dist.Normal(loc=z_t_mean, scale=tau))
        Z.append(z_t)
        y_t = pyro.sample(name="y_%d"%(t+1), fn=dist.Normal(loc=Z[t], scale=sigma), obs=obs[t])
        Y.append(y_t)
        z_prev1 = z_prev2
        z_prev2 = Z[t]

    for t in range(T, T+T_forecast):
        z_t_mean = beta1*z_prev1 + beta2*z_prev2
        z_t = pyro.sample(name="z_%d"%(t+1), fn=dist.Normal(loc=z_t_mean, scale=tau))
        Z.append(z_t)
        y_t = pyro.sample(name="y_%d"%(t+1), fn=dist.Normal(loc=Z[t], scale=sigma), obs=None)
        Y.append(y_t)
        z_prev1 = z_prev2
        z_prev2 = Z[t]
    return Z