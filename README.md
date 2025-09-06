7_STEP_XGBOOST.py

انتخاب پارامترها: پارامترهای مدل XGBoost (مثل n_estimators, max_depth, learning_rate) به صورت دستی تنظیم شده‌اند. اگرچه این پارامترها شروع خوبی هستند، اما برای یافتن بهترین ترکیب، می‌توانید از روش‌های بهینه‌سازی پارامتر (Hyperparameter Tuning) مانند Grid Search یا Random Search استفاده کنید.



------------------------
training 

Hyperparameter Tuning:
   · This is the biggest potential addition. The script uses a well-chosen set of fixed hyperparameters (max_depth=6, learning_rate=0.05, etc.). These are good sensible defaults, but they are unlikely to be optimal.
   · Suggestion: Integrate a tuning step (e.g., RandomizedSearchCV or BayesianOptimization) before the final training run with early stopping. You would use the X_tmp/y_tmp set for cross-validated tuning, find the best params, and then train the final model (with early stopping on X_dev/y_dev) using those best parameters.


Hyperparameter optimization (biggest immediate impact)

2. Cross-validation
