# Unscented Kalman Filter (UKF) – Code Overview







---

##  Main state variables

### `self.x`
- Current **state mean**
- Shape: `(n_dim,)`
- Example: position, velocity etc.

---

### `self.p`
- Current **state covariance**
- Shape: `(n_dim, n_dim)`
- Represents uncertainty in the state

---

### `self.q`
- **Process noise covariance**
- Shape: `(n_dim, n_dim)`
- Added during prediction to model system uncertainty

---

### `self.sigmas`
- **Sigma points**
- Shape: `(n_dim, 2*n_dim + 1)`
- Deterministic points generated around `self.x`
- Used instead of random sampling

---

##  Size and tuning parameters

### `self.n_dim`
- Number of state variables

---

### `self.n_sig`
- Number of sigma points  
- Value: `2 * n_dim + 1`

---

### `self.alpha`, `self.beta`, `self.k`
- UKF tuning parameters
- Used only to:
  - control sigma point spread
  - compute weights



---

### `self.lambd`
- Scaling parameter computed from `alpha`, `k`, and `n_dim`
- Used in:
  - weight calculation
  - sigma point generation

---

##  Weights

### `self.mean_weights`
- Array of size `n_sig`
- Used when computing **means**

---

### `self.covar_weights`
- Array of size `n_sig`
- Used when computing **covariances**
- First weight includes the `beta` term

---

##  Helper function

### `__get_sigmas()`
- Generates sigma points from current `self.x` and `self.p`
- Uses matrix square root of `(n_dim + lambd) * P`
- Output shape: `(n_dim, n_sig)`

This function is called:
- during initialization
- after every predict
- after every update

---

##  Prediction step

### `predict(timestep, inputs)`

What it does (in order):

1. Propagates each sigma point through `iterate()`
2. Computes predicted mean `x_out`
3. Computes predicted covariance `p_out`
4. Adds process noise (`timestep * q`)
5. Updates:
   - `self.x`
   - `self.p`
   - `self.sigmas`



---

### `self.iterate`
- User-provided function
- Called as:
  ```python
  iterate(state, timestep, inputs)
