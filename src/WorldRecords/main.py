import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme

# =========================================================================
# === STEP 1: LOAD AND PREPARE YOUR DATA ==================================
# =========================================================================

###--- YOUR ACTION REQUIRED: Load Your Data ---###
# You must provide a dataset of *annual (or seasonal) bests*,
# NOT just the list of world records.
# The CSV file should have two columns: 'Year' and 'Best_Time' (in seconds).
#
# Here is an example of what your CSV data should look like:
# Year,Best_Time
# 1980,403.28
# 1981,401.11
# 1982,399.20
# ...
# 2023,358.92
# 2024,357.75

try:
    # --- UNCOMMENT THE LINE BELOW and replace with your file path ---
    data = pd.read_csv('/Users/justin/Documents/4th-down/nfl-4th-down-heatmap/src/WorldRecords/FastestTimes.csv')
    

    # --- Data Pre-processing ---
    # GEV models MAXIMA (largest values), but we have MINIMA (fastest times).
    # We must multiply our times by -1. The "largest" negative time
    # is the fastest (smallest positive) time.
    data['Negative_Time'] = -data['Best_Time']

    print("\nData successfully loaded and transformed:")
    print(data.tail())

    # Get data for the model (as numpy array for faster computation)
    times_data = data['Negative_Time'].values
    
    # Create a 'time' variable (years since first year in data)
    # This helps with model stability by centering the time variable around 0
    years_since_start = data['Year'].values - data['Year'].min()

except FileNotFoundError:
    print("\n!!! ERROR: Data file not found. !!!")
    print("Please update the 'pd.read_csv' line with the correct path.")
    exit()
except Exception as e:
    print(f"\n!!! ERROR: Could not load data: {e} !!!")
    print("Ensure your CSV has 'Year' and 'Best_Time' columns.")
    exit()


# =========================================================================
# === STEP 2: DEFINE THE BAYESIAN MODEL (NON-STATIONARY GEV) ============
# =========================================================================

# We will model the 'mu' (location) parameter as a linear trend over time:
# mu(t) = beta_0 + beta_1 * year
# This accounts for performance improvements over time.

with pm.Model() as model:
    
    ###--- ADJUST PRIORS (Optional, but recommended for advanced use) ---###
    # Priors are our "guesses" for the parameters before seeing the data.
    # The defaults here are "weakly informative" and should be a good start.

    # --- Priors for Non-Stationary Mu ---
    # beta_0: The intercept (baseline 'mu' at the start year)
    # We center it around the mean of the observed data with a wide variance
    beta_0 = pm.Normal('beta_0', mu=np.mean(times_data), sigma=np.std(times_data) * 2)
    
    # beta_1: The slope (how much 'mu' changes per year)
    # We expect this to be positive (negative times get "larger" / closer to 0)
    # Centered at 0 to be flexible about the direction and magnitude of trend
    beta_1 = pm.Normal('beta_1', mu=0, sigma=1.0)
    
    # --- Priors for other GEV parameters ---
    # sigma: The scale parameter (spread of extremes), must be positive.
    # HalfNormal ensures sigma > 0
    sigma = pm.HalfNormal('sigma', sigma=np.std(times_data))
    
    # xi: The shape parameter. This is the most important one!
    # xi < 0 -> Finite upper limit (ultimate record exists)
    # xi = 0 -> Gumbel distribution (no bound)
    # xi > 0 -> Heavy-tailed (Fréchet distribution)
    # We set a prior centered at 0, with a small standard deviation.
    xi = pm.Normal('xi', mu=0.0, sigma=0.1)

    # --- Define the changing Mu ---
    # mu changes linearly with time: mu(t) = beta_0 + beta_1 * t
    # Deterministic means it's fully determined by other variables (not random itself)
    mu = pm.Deterministic('mu', beta_0 + beta_1 * years_since_start)

    # --- Custom GEV Log-Probability Function ---
    def gev_logp(value, mu, sigma, xi):
        """
        Custom GEV log-probability function for PyMC.
        Uses pm.math operations for automatic differentiation.
        
        The GEV distribution has two cases:
        1. Gumbel (xi ≈ 0): Exponential tails
        2. General GEV (xi ≠ 0): Power-law tails
        
        Parameters:
        -----------
        value : array-like
            Observed data points
        mu : array-like
            Location parameter (can vary with time)
        sigma : float
            Scale parameter (spread)
        xi : float
            Shape parameter (tail behavior)
        """
        # Standardized value: z = (x - mu) / sigma
        z = (value - mu) / sigma
        # t is used in the GEV formula
        t = 1.0 + xi * z
        
        # Gumbel case (xi ≈ 0): Standard extreme value distribution
        # log-pdf: -log(sigma) - z - exp(-z)
        gumbel_logp = -pm.math.log(sigma) - z - pm.math.exp(-z)
        
        # General GEV case (xi ≠ 0)
        # log-pdf: -log(sigma) - (1 + 1/xi)*log(t) - t^(-1/xi)
        gev_logp = -pm.math.log(sigma) - (1.0 + 1.0/xi) * pm.math.log(t) - t**(-1.0/xi)
        
        # Use Gumbel when |xi| is very small, otherwise use GEV
        # Also enforce domain constraint: 1 + xi*z > 0 (ensures valid distribution)
        # If constraint violated, return -inf (zero probability)
        logp = pm.math.switch(
            pm.math.abs(xi) < 1e-8,  # If xi is close to 0
            gumbel_logp,              # Use Gumbel
            pm.math.switch(t > 0, gev_logp, -np.inf)  # Else use GEV if valid
        )
        
        # Sum log-probabilities across all data points
        return pm.math.sum(logp)

    # --- Likelihood ---
    # DensityDist allows us to specify a custom probability distribution
    # We pass mu, sigma, xi as positional arguments so they're tracked by PyMC
    # observed=times_data tells PyMC these are the actual observations
    y_obs = pm.DensityDist('y_obs', mu, sigma, xi, logp=gev_logp, observed=times_data)


# =========================================================================
# === STEP 3: RUN THE MODEL (MCMC SAMPLING) ===============================
# =========================================================================

###--- ADJUST SAMPLING (Optional) ---###
# These parameters control the MCMC sampler.
# If your model has issues (see Step 4), you may need to
# increase 'draws' and 'tune'. This will make it run slower.

n_draws = 2000      # Number of samples to keep per chain
n_tune = 1000       # Number of "warm-up" samples (discarded) to tune sampler
n_chains = 4        # Number of parallel chains to run (4 is standard for diagnostics)
n_cores = 4         # Number of CPU cores to use (parallel processing)

print(f"\n--- Starting MCMC sampling ---")
print(f"(Draws={n_draws}, Tune={n_tune}, Chains={n_chains}, Cores={n_cores})")
print("This may take a few minutes...")

with model:
    # Run the MCMC sampler
    # target_accept=0.9 means we aim for 90% acceptance rate (higher = more accurate but slower)
    trace = pm.sample(
        n_draws,
        tune=n_tune,
        chains=n_chains,
        cores=n_cores,
        target_accept=0.9
    )

print("--- Sampling complete ---")


# =========================================================================
# === STEP 4: CHECK THE MODEL DIAGNOSTICS =================================
# =========================================================================

print("\n--- Model Summary (Check 'r_hat' and 'ess') ---")
# 'r_hat' should be very close to 1.0 (e.g., < 1.01). If not, model did not converge.
# 'ess' (effective sample size) should be high (e.g., > 1000).
# Low ess_bulk or ess_tail indicates poor mixing of chains
summary = az.summary(trace, var_names=['beta_0', 'beta_1', 'sigma', 'xi'])
print(summary)

# --- Visual check ---
# This plot shows the "trace" (left) and posterior distribution (right).
# The trace plots (left) should look like "fuzzy caterpillars" with no
# clear drifts or patterns. This indicates good convergence.
# All chains should overlap and explore the same region.
print("\nDisplaying trace plot (Close plot window to continue)...")
az.plot_trace(trace, var_names=['beta_0', 'beta_1', 'sigma', 'xi'])
plt.tight_layout()
plt.show()


# =========================================================================
# === STEP 5: ANALYZE, PREDICT, AND INTERPRET =============================
# =========================================================================

# --- A. Does an ultimate record exist? Check 'xi' ---
print("\n--- Shape Parameter (xi) Analysis ---")
print("If the 95% Credible Interval is *entirely negative*,")
print("the model suggests a finite, ultimate world record exists.")
print(summary.loc['xi'])

# Plot the posterior distribution of xi
# If the distribution is mostly/entirely below 0, an ultimate record likely exists
print("\nDisplaying 'xi' posterior distribution (Close plot window to continue)...")
az.plot_posterior(trace, var_names=['xi'])
plt.show()


# --- B. Predict next year's best time ---
print("\n--- Predicting Next Year's Best Time ---")

# Get the 'year' value for next year (in years since start)
next_year_value = years_since_start[-1] + 1
next_year_actual = data['Year'].max() + 1

# Extract posterior samples from the trace
# We have 4 chains × 2000 draws = 8000 samples total
posterior = trace.posterior
beta_0_samples = posterior['beta_0'].values.flatten()
beta_1_samples = posterior['beta_1'].values.flatten()
sigma_samples = posterior['sigma'].values.flatten()
xi_samples = posterior['xi'].values.flatten()

# Calculate 'mu' for next year for all 8000 samples
# Each sample represents one possible "world state" given our data
mu_next_year = beta_0_samples + beta_1_samples * next_year_value

# Simulate 8000 possible "next year's bests"
# We use scipy's genextreme, noting that its 'c' parameter is -xi
# (scipy uses a different parameterization than the standard GEV)
simulated_bests_neg = genextreme.rvs(
    c=-xi_samples,           # Shape parameter (negated)
    loc=mu_next_year,        # Location parameter
    scale=sigma_samples,     # Scale parameter
    size=len(xi_samples)     # One draw per posterior sample
)

# CRITICAL: Flip the times back to positive (actual seconds)
# Remember we negated them earlier to convert minima to maxima
simulated_bests_real_time = -simulated_bests_neg

# Get the predicted range using percentiles
# Median = 50th percentile (middle value)
# 95% CI = 2.5th to 97.5th percentile (credible interval)
median_pred = np.percentile(simulated_bests_real_time, 50)
ci_pred = np.percentile(simulated_bests_real_time, [2.5, 97.5])

print(f"Predicted Best Time for {next_year_actual}:")
print(f"  Median (50%): {median_pred:.2f} seconds")
print(f"  95% CI:       [{ci_pred[0]:.2f}, {ci_pred[1]:.2f}] seconds")

# Plot the distribution of predictions
print("\nDisplaying prediction distribution (Close plot window to continue)...")
az.plot_dist(simulated_bests_real_time, label=f"Predicted Best Time for {next_year_actual}")
plt.xlabel("Time (seconds)")
plt.title("Distribution of Possible Best Times Next Year")
plt.show()


# --- C. What is the probability of a new world record? ---

###--- YOUR ACTION REQUIRED: Set Current World Record ---###
# You MUST set this value manually to the *current* official
# world record for the event you are modeling.

current_record = 65.37  # <--- EXAMPLE. REPLACE THIS VALUE!

# --- (End of required action) ---


if current_record == 356.78:
    print("\n!!! WARNING: Using example World Record value. !!!")
    print("!!! Please edit the script and set 'current_record' !!!")
    
# Get the minimum from our *data* for comparison
all_time_best_in_data = data['Best_Time'].min()
print(f"\n--- Probability of New World Record (WR) Next Year ---")
print(f"Set Current WR:    {current_record:.2f} seconds")
print(f"Fastest in dataset:  {all_time_best_in_data:.2f} seconds")

# Count how many of our simulations are faster than the current WR
# This gives us a probability estimate via Monte Carlo simulation
new_record_sims = simulated_bests_real_time[simulated_bests_real_time < current_record]

# Calculate probability as the fraction of simulations that beat the WR
probability_new_record = len(new_record_sims) / len(simulated_bests_real_time)

print(f"\nEstimated probability of a new WR next year: {probability_new_record * 100:.2f}%")
print("\n--- End of script ---")