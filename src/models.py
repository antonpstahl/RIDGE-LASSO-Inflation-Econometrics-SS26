"""Stage 3a: Custom estimators."""
import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV


class AdaptiveLasso:
    """Adaptive LASSO via feature rescaling (sklearn-compatible).

    Ridge initialisation, because OLS is unstable at p/n≈0.69.
    Approximate adaptive weighting - the oracle prerequisite (√n consistency,
    Zou 2006) is not guaranteed at p/n≈0.69.
    """

    def __init__(self, gamma=1.0, eps=1e-6, alphas=None, cv=5, max_iter=10000):
        self.gamma    = gamma
        self.eps      = eps
        self.alphas   = alphas if alphas is not None else np.logspace(-4, 1, 50)
        self.cv       = cv
        self.max_iter = max_iter

    def fit(self, X, y):
        # Step 1: Ridge as initial estimator.
        # RidgeCV without a cv parameter uses sklearn LOO-GCV (efficient leave-one-out
        # estimation), which ignores the time structure. This is intentional here: the
        # Ridge coefficients only serve as initial weights for the adaptive
        # scaling (step 2), they do not enter the forecast. The actual
        # estimator (LassoCV, step 3) uses time-consistent TS-CV via self.cv.
        init = RidgeCV(alphas=np.logspace(-2, 4, 30)).fit(X, y)
        self._scale = np.abs(init.coef_) ** self.gamma + self.eps
        # Step 2: rescale features  X̃_j = X_j * scale_j
        X_tilde = X * self._scale
        # Step 3: standard LassoCV on X̃
        self._lasso = LassoCV(
            alphas=self.alphas, cv=self.cv,
            max_iter=self.max_iter, n_jobs=-1,
        ).fit(X_tilde, y)
        self.alpha_     = self._lasso.alpha_
        self.coef_      = self._lasso.coef_ * self._scale
        self.intercept_ = self._lasso.intercept_
        return self

    def predict(self, X):
        return self._lasso.predict(X * self._scale)
