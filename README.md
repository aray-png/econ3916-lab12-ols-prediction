# Abigail Ray || OLS as a Line of Best Fit: Estimation versus Prediction

## Objective

I used California Housing data (20,640 districts) to show that the same OLS machinery can either estimate a coefficient or predict an outcome, and that each job needs a different scorecard.

## Dataset

- **Source:** California Housing, loaded from scikit-learn. Each row is one district (a census block group) from the 1990 census. There are 20,640 rows and 9 columns.
- **Outcome:** `value`, the district's median house value, in hundreds of thousands of dollars. It runs from 0.15 to 5.00 with a mean of 2.07, so the average house is worth about $207,000.
- **Main predictor:** `MedInc`, median household income, in tens of thousands of dollars. It runs from 0.50 to 15.00 with a mean of 3.87.
- **Other predictors:** `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude` and `Longitude`.
- **Known data issue:** house values above $500,000 were all recorded as 5.00001, so the outcome has a ceiling at 5.00.

## Methodology

1. **Three routes to one line.** I fit median house value on median income using the closed-form slope formula (covariance of x and y over the variance of x), statsmodels, and scikit-learn, and checked that all three agreed.
2. **Adding noise.** I added 0, 1, 5, 20 and 100 columns of random numbers to the regression and tracked R-squared and adjusted R-squared at each step.
3. **Residual diagnostics.** I plotted the fitted line and the residuals against the fitted values to look for patterns that a single R-squared number cannot show.
4. **Train/test split.** I split the data 80/20 with a fixed random state before fitting anything. I fit an eight-feature model on the 16,512 training rows only, then scored it on both the training rows and the 4,128 test rows.
5. **Two jobs, one specification.** I used median income and rooms per household for both:
   - Estimation: the AveRooms coefficient with its 95% confidence interval (statsmodels).
   - Prediction: test RMSE on rows the model never saw (scikit-learn).
6. **A reusable function.** I wrote `holdout_rmse(data, features)`, which splits a table the same way every time, fits on the training rows, and returns the test RMSE. I used it on median income alone and passed the lab's check cell.
7. **Take-home challenge.** I counted the districts at the 5.00 ceiling, refit the eight-feature model without them, compared coefficients and test RMSE, and checked what the trimmed model predicts for the ceiling districts.

## Results

| Step | Result |
|---|---|
| Intercept, all three routes | 0.450856 |
| Slope, all three routes | 0.417938 |
| R-squared, 0 noise columns | 0.4734 |
| R-squared, 100 noise columns | 0.4758 |
| Adjusted R-squared, 0 / 100 noise columns | 0.4734 / 0.4732 |
| Eight features, train RMSE / R-squared | 0.7197 / 0.6126 |
| Eight features, test RMSE / R-squared | 0.7456 / 0.5758 |
| AveRooms coefficient (with MedInc) | -0.0381 |
| AveRooms 95% CI | [-0.0430, -0.0333] |
| MedInc + AveRooms, test RMSE | 0.8379 |
| MedInc alone, test RMSE | 0.8421 |
| Districts at the ceiling | 965 of 20,640 (4.7%) |
| Eight features, test RMSE, ceiling removed | 0.6511 (n = 19,675) |

## Key Findings

- **Three routes, one line.** The closed form, statsmodels and scikit-learn all returned the same intercept and slope. A one-unit rise in median income (about $10,000) goes with a 0.418 rise in median house value, or about $41,800 on average. This step describes the data I already had. It does not test the line on new rows.
- **The intercept is not meaningful on its own.** It says a district with zero income would have a house value of about 0.45, but the lowest income in the data is 0.50. It still sets the starting height of the line.
- **R-squared rewards junk.** R-squared rose from 0.4734 with no noise columns to 0.4758 with 100, even though the noise has nothing to do with house value. Adjusted R-squared fell slightly, from 0.4734 to 0.4732, so it does penalise extra columns. It is still measured on the same data the model was fitted to, so I would not use it alone to choose a model.
- **The residual plot shows two problems.** The values are capped at 5.00, which makes a straight diagonal edge across the top of the residuals. The spread of the residuals is also wider at larger fitted values than at small ones, so the errors are not equally large across the range.
- **Train versus test.** The eight-feature model had a train RMSE of 0.7197 and a test RMSE of 0.7456. The gap is small, so I saw no sign of overfitting. A test RMSE of 0.7456 is about $74,600, which is large next to an average house value of about $207,000. I would use this model for rough patterns, not to price an individual home.
- **Estimation and prediction are different questions.** The AveRooms coefficient of -0.0381 is precisely estimated, and its interval does not include zero. The same two-variable specification still had a test RMSE of 0.8379, worse than the eight-feature model's 0.7456. A precise coefficient does not mean good predictions.
- **A fair comparison.** `holdout_rmse()` gave a test RMSE of 0.8421 for median income alone, against 0.7456 for all eight features. That is a gap of about 0.0965, or about $9,650 per house. Both models were scored on the same held-out rows, so neither gets credit for simply having more columns.
- **The ceiling.** 965 of 20,640 districts (4.7%) sit at the 5.00 ceiling. After dropping them, test RMSE fell from 0.7456 to 0.6511, and the coefficients moved toward zero. For example, MedInc went from 0.4487 to 0.4260 and AveBedrms from 0.7831 to 0.5753. The trimmed model predicted an average of 3.80 for the ceiling districts, whose actual value is 5.00.
- **Lower RMSE here does not mean a better model.** The districts I dropped were removed for having high values, not at random. The trimmed model only describes districts worth under $500,000, and it is scored on an easier set of rows.

## Limitations

- Every model here is a linear regression on 1990 census districts. I did not try other model types.
- The ceiling at 5.00 and the unequal spread of the residuals both break assumptions behind the standard errors, so I treat the confidence intervals with some caution.
- I used one train/test split with one random state, so the test RMSE values depend on which rows landed in the test set.
- Dropping the ceiling districts changed who the model is about. I did not try a method that keeps those districts in the data.

## What I Took Away

- R-squared on the training data can only rise as I add columns, so I need held-out rows to judge a model.
- I need to split the data before fitting, scaling or imputing, or information from the test rows leaks into the model.
- A coefficient and a prediction are judged differently: a coefficient by its interval, a prediction by its error on new data.
- A decision about how the outcome was recorded, like the 5.00 ceiling, changes what my model can claim, so I should state how I handled it when I report results.

## How to Run

1. Open `lab-ch12-guided.ipynb` in Google Colab.
2. Run the cells once, from the top, in order. Step 4 draws noise from one random generator, so if I run a cell twice I use Runtime → Restart session and start again.
3. The notebook saves one figure to a `figures/` folder: `ch12_fit_and_residuals.png`.
