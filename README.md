# microlensing-dtw

Exploring Roman-like gravitational microlensing light curves with time-series ML methods (DTW, soft-DTW, shapelets, attention) from the ENSAI course *Machine Learning for Time Series*.

## Step 1: look at the data

`notebooks/01_look_at_the_data.ipynb` simulates five kinds of light curves (constant star, variable star, stellar microlensing event, free-floating-planet event, stellar flare) and makes four exploratory plots.

The simulator is `mldtw/simulate.py`.

## Run

```bash
pip install -r requirements.txt
cd notebooks
jupyter lab
```

Figures are saved to `figures/`.
