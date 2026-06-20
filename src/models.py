"""Stage 3a: Custom estimators."""
import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV


class AdaptiveLasso:
    """Adaptive LASSO via feature rescaling (sklearn-compatible).

    Ridge-Initialisierung, da OLS bei p/n≈0.69 instabil ist.
    Approximative adaptive Gewichtung — Oracle-Voraussetzung (√n-Konsistenz,
    Zou 2006) ist bei p/n≈0.69 nicht garantiert.
    """

    def __init__(self, gamma=1.0, eps=1e-6, alphas=None, cv=5, max_iter=10000):
        self.gamma    = gamma
        self.eps      = eps
        self.alphas   = alphas if alphas is not None else np.logspace(-4, 1, 50)
        self.cv       = cv
        self.max_iter = max_iter

    def fit(self, X, y):
        # Schritt 1: Ridge als initialer Schätzer.
        # RidgeCV ohne cv-Parameter nutzt sklearn-LOO-GCV (effiziente Leave-One-Out-
        # Schätzung), das die Zeitstruktur ignoriert. Das ist hier bewusst: die
        # Ridge-Koeffizienten dienen nur als Initialgewichte fuer die adaptive
        # Skalierung (Schritt 2) — sie gehen nicht als Prognose ein. Der eigentliche
        # Schätzer (LassoCV, Schritt 3) nutzt zeitkonformes TS-CV ueber self.cv.
        init = RidgeCV(alphas=np.logspace(-2, 4, 30)).fit(X, y)
        self._scale = np.abs(init.coef_) ** self.gamma + self.eps
        # Schritt 2: Features rescalen  X̃_j = X_j · scale_j
        X_tilde = X * self._scale
        # Schritt 3: Standard-LassoCV auf X̃
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
