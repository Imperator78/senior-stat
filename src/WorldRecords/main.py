import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme

try:
    data = pd.read_csv('/Users/justin/Documents/4th-down/nfl-4th-down-heatmap/src/WorldRecords/FastestTimes.csv')
    

    #Data Pre-processing
    data['Negative_Time'] = -data['Best_Time']

    print(data.tail())

    # Get data for the model
    times_data = data['Negative_Time'].values
    years_since_start = data['Year'].values - data['Year'].min()

except FileNotFoundError:
    print("wrong path")
    exit()
except Exception as e:
    print(f"!!!{e}!!!")
    print("need correct headers")
    exit()


with pm.Model() as model:
    beta_0 = pm.Normal('beta_0', mu=np.mean(times_data), sigma=np.std(times_data) * 2)
    beta_1 = pm.Normal('beta_1', mu=0, sigma=1.0)
    sigma = pm.HalfNormal('sigma', sigma=np.std(times_data))
    xi = pm.Normal('xi', mu=0.0, sigma=0.1)
    mu = pm.Deterministic('mu', beta_0 + beta_1 * years_since_start)

    def gev_logp(value, mu, sigma, xi):
        z = (value - mu) / sigma
        t = 1.0 + xi * z

        gumbel_logp = -pm.math.log(sigma) - z - pm.math.exp(-z)
        gev_logp = -pm.math.log(sigma) - (1.0 + 1.0/xi) * pm.math.log(t) - t**(-1.0/xi)
        logp = pm.math.switch(
            pm.math.abs(xi) < 1e-8,  #if its 0 that's annoying
            gumbel_logp,              
            pm.math.switch(t > 0, gev_logp, -np.inf) 
        )
        
        return pm.math.sum(logp)

    y_obs = pm.DensityDist('y_obs', mu, sigma, xi, logp=gev_logp, observed=times_data)


n_draws = 2000      
n_tune = 1000 
n_chains = 4  
n_cores = 4   

print("MCMC")
print(f"(Draws={n_draws}, Tune={n_tune}, Chains={n_chains}, Cores={n_cores})")

with model:
    trace = pm.sample(
        n_draws,
        tune=n_tune,
        chains=n_chains,
        cores=n_cores,
        target_accept=0.9
    )

print("Results")
summary = az.summary(trace, var_names=['beta_0', 'beta_1', 'sigma', 'xi'])
print(summary)

print("Trace Plots")
az.plot_trace(trace, var_names=['beta_0', 'beta_1', 'sigma', 'xi'])
plt.tight_layout()
plt.show()

print(summary.loc['xi'])

print("\nDisplaying 'xi' posterior distribution (Close plot window to continue)...")
az.plot_posterior(trace, var_names=['xi'])
plt.show()


print("\n--- Next Year's Best Time ---")
# Get the 'year' value for next year (in years since start)
next_year_value = years_since_start[-1] + 1
next_year_actual = data['Year'].max() + 2

posterior = trace.posterior
beta_0_samples = posterior['beta_0'].values.flatten()
beta_1_samples = posterior['beta_1'].values.flatten()
sigma_samples = posterior['sigma'].values.flatten()
xi_samples = posterior['xi'].values.flatten()

mu_next_year = beta_0_samples + beta_1_samples * next_year_value

simulated_bests_neg = genextreme.rvs(
    c=-xi_samples,           # Shape parameter (negated)
    loc=mu_next_year,        # Location parameter
    scale=sigma_samples,     # Scale parameter
    size=len(xi_samples)     # One draw per posterior sample
)

#This is important cause we negated them earlier for some reason. Idk that's what the thing said to do
simulated_bests_real_time = -simulated_bests_neg

median_pred = np.percentile(simulated_bests_real_time, 50)
ci_pred = np.percentile(simulated_bests_real_time, [2.5, 97.5])

print(f"Predicted best time {next_year_actual}:")
print(f"Median: {median_pred:.2f} seconds")
print(f"95% confidence interval:       [{ci_pred[0]:.2f}, {ci_pred[1]:.2f}] seconds")

# Plot the distribution of predictions
az.plot_dist(simulated_bests_real_time, label=f"Predicted Best Time for {next_year_actual}")
plt.xlabel("Time (seconds)")
plt.title("Distribution of Possible Best Times Next Year")
plt.show()

current_record = 65.37 

# Get the minimum from our *data* for comparison
all_time_best_in_data = data['Best_Time'].min()
print(f"\n--- Probability of New World Record (WR) Next Year ---")
print(f"Set Current WR:    {current_record:.2f} seconds")
print(f"Fastest in dataset:  {all_time_best_in_data:.2f} seconds")

new_record_sims = simulated_bests_real_time[simulated_bests_real_time < current_record]

probability_new_record = len(new_record_sims) / len(simulated_bests_real_time)

print(f"\nEstimated probability of a new WR next year: {probability_new_record * 100:.2f}%")
