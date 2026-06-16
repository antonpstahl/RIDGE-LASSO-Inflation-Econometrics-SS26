"""Global configuration: paths, seeds, hyperparameter grids, CV objects, API constants."""
import pathlib

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT           = pathlib.Path(__file__).resolve().parent.parent
DATA_RAW       = ROOT / "data" / "raw"    / "data_raw.csv"
DATA_PROCESSED = ROOT / "data" / "processed" / "data_yoy.csv"
RESULTS_DIR    = ROOT / "results"
FIGURES_DIR    = ROOT / "results" / "figures"

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Feature engineering ───────────────────────────────────────────────────────
LAGS        = [1, 2, 3, 6, 12]
AR_LAGS     = [1, 2, 3, 6, 12]
TEST_MONTHS = 36
HORIZONS    = [1, 3, 6, 12]

# ── Reporting ─────────────────────────────────────────────────────────────────
WINDOW_ROLLING_RMSE = 12
TOP_N_STABILITY     = 25

# ── Hyperparameter grids ──────────────────────────────────────────────────────
ALPHAS_LASSO         = np.logspace(-4, 2, 200)
ALPHAS_RIDGE         = np.logspace(-2, 4, 200)
ALPHAS_LASSO_INNER   = np.logspace(-4, 1, 50)
ALPHAS_RIDGE_INNER   = np.logspace(-2, 4, 50)
L1_RATIOS_ENET       = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
L1_RATIOS_ENET_INNER = [0.5, 0.9, 0.99, 1.0]

# ── Cross-validation ──────────────────────────────────────────────────────────
TSCV       = TimeSeriesSplit(n_splits=10, test_size=12)
TSCV_INNER = TimeSeriesSplit(n_splits=3,  test_size=12)

# ── Plot colours ──────────────────────────────────────────────────────────────
COLORS = {
    "OLS": "#2196F3", "Ridge": "#FF9800",
    "LASSO": "#4CAF50", "ElasticNet": "#F44336",
}
COLORS_OOS = {
    "RW": "#9E9E9E",      "AR": "#795548",
    "OLS": "#2196F3",     "Ridge": "#FF9800",
    "LASSO": "#4CAF50",   "Elastic Net": "#F44336",
    "LASSO+HVPI": "#9C27B0",
}

# ── API endpoints ─────────────────────────────────────────────────────────────
ECB_BASE   = "https://data-api.ecb.europa.eu/service/data"
ESTAT_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

# ── Data-series definitions ───────────────────────────────────────────────────
PROD_SECTORS = {
    "IP_Verarbeitendes_Gew": "C",       "IP_Nahrungs_Genuss":    "C10-C12",
    "IP_Textil_Leder":       "C13-C15", "IP_Holz_Papier_Druck":  "C16-C18",
    "IP_Mineraloel":         "C19",     "IP_Chemie":             "C20",
    "IP_Pharma":             "C21",     "IP_Metall_Grundstoffe": "C24",
    "IP_Metallerzeugnisse":  "C25",     "IP_DV_Elektronik":      "C26",
    "IP_Elektrisch":         "C27",     "IP_Maschinenbau":       "C28",
    "IP_Kfz":                "C29",     "IP_Sonstiger_Fahrzeug": "C30",
    "IP_Energie":            "D",
}
BS_INDICATORS = {
    "BS_Konjunkturklima":    "BS-ICI",  "BS_Produktionserwart":  "BS-IEME",
    "BS_Produktionstendenz": "BS-IPT",  "BS_Absatzpreise":       "BS-ISFP",
    "BS_Auftragsbestand":    "BS-IOB",
}
PPI_SECTORS = {
    "PPI_Gesamt":        "B-E36",
    "PPI_Konsumgueter":  "MIG_COG",
    "PPI_Vorleistungen": "MIG_ING",
}
UNEMP_GROUPS = {
    "ALQ_Gesamt":   {"sex": "T", "age": "TOTAL"},
    "ALQ_Maenner":  {"sex": "M", "age": "TOTAL"},
    "ALQ_Frauen":   {"sex": "F", "age": "TOTAL"},
    "ALQ_Kern_Ges": {"sex": "T", "age": "Y25-74"},
    "ALQ_Kern_M":   {"sex": "M", "age": "Y25-74"},
    "ALQ_Kern_F":   {"sex": "F", "age": "Y25-74"},
}
LCI_SERIES = {
    "LCI_Lohnkosten_BN":  {"nace_r2": "B-N", "lcstruct": "D1_D4_MD5"},
    "LCI_Loehne_BN":      {"nace_r2": "B-N", "lcstruct": "D11"},
    "LCI_Lohnkosten_Ind": {"nace_r2": "B-F", "lcstruct": "D1_D4_MD5"},
    "LCI_Loehne_Ind":     {"nace_r2": "B-F", "lcstruct": "D11"},
}


def setup_environment():
    """Apply global numpy / matplotlib settings. Call once at pipeline start."""
    import matplotlib.pyplot as plt

    np.random.seed(SEED)
    # FP-Ausnahmen werden *lokal* mit np.errstate unterdrückt (training.py,
    # evaluation.py: rolling_origin mit suppress_fp=True), nicht global.
    plt.rcParams.update({
        "figure.dpi":        120,
        "savefig.dpi":       300,
        "font.size":         11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.figsize":    (12, 5),
    })
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
