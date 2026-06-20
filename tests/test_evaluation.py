"""Smoke-Tests: DM-Test und RMSE-Sanity fuer evaluation."""
import numpy as np
import pandas as pd
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.evaluation import (
    bonferroni_correct, clark_west, compute_giacomini_rossi,
    compute_regime_analysis, compute_robustness_mom,
    diebold_mariano, rolling_origin,
)


# ── Diebold-Mariano ───────────────────────────────────────────────────────────

def test_dm_perfect_model():
    """Perfektes Modell (e_mod=0) sollte positive DM-Statistik haben."""
    np.random.seed(0)
    e_rw  = np.random.randn(36)
    e_mod = np.zeros(36)
    dm, pv = diebold_mariano(e_rw, e_mod, h=1)
    assert dm > 0, "DM-Stat sollte positiv sein wenn Modell perfekt"
    assert 0 <= pv <= 1, f"p-Wert ausserhalb [0,1]: {pv}"


def test_dm_rw_is_rw():
    """Identische Fehler (Modell = RW) sollen DM ≈ 0 ergeben."""
    np.random.seed(1)
    e_rw = np.random.randn(36)
    dm, pv = diebold_mariano(e_rw, e_rw, h=1)
    assert abs(dm) < 1e-10, f"DM sollte 0 sein bei gleichen Fehlern, got {dm}"


def test_dm_worse_model():
    """Schlechteres Modell (groessere Fehler) soll negative DM-Statistik haben."""
    np.random.seed(2)
    e_rw  = np.random.randn(36) * 1.0
    e_mod = np.random.randn(36) * 2.0   # groessere Fehler
    dm, pv = diebold_mariano(e_rw, e_mod, h=1)
    assert dm < 0, "DM-Stat sollte negativ sein wenn Modell schlechter als RW"
    assert 0 <= pv <= 1


def test_dm_p_value_range():
    """p-Wert muss in [0,1] liegen."""
    e1 = np.linspace(-1, 1, 36)
    e2 = np.linspace(-2, 2, 36)
    _, pv = diebold_mariano(e1, e2, h=1)
    assert 0.0 <= pv <= 1.0


# ── Clark-West ────────────────────────────────────────────────────────────────

def test_cw_perfect_nested_model():
    """Perfektes geschachteltes Modell (e_mod=0) → CW-Stat positiv, p-Wert in [0,1]."""
    np.random.seed(10)
    e_rw  = np.random.randn(36)
    e_mod = np.zeros(36)
    cw, pv = clark_west(e_rw, e_mod, h=1)
    assert cw > 0, f"CW-Stat sollte positiv sein bei perfektem Modell, got {cw}"
    assert 0.0 <= pv <= 1.0, f"p-Wert ausserhalb [0,1]: {pv}"


def test_cw_equal_errors_zero_stat():
    """Wenn nested Modell = RW (e_mod = e_rw), dann f_t = 0 fuer alle t → CW ≈ 0."""
    np.random.seed(11)
    e_rw = np.random.randn(36)
    cw, pv = clark_west(e_rw, e_rw, h=1)
    assert abs(cw) < 1e-10, f"CW sollte 0 sein bei gleichen Fehlern, got {cw}"


def test_cw_p_value_one_sided():
    """CW p-Wert ist einseitig: fuer positiven CW-Stat ist p < 0.5."""
    np.random.seed(12)
    e_rw  = np.random.randn(100) * 2.0   # groessere RW-Fehler
    e_mod = np.random.randn(100) * 0.5   # kleinere Modellfehler
    cw, pv = clark_west(e_rw, e_mod, h=1)
    assert cw > 0, "CW-Stat sollte positiv sein wenn Modell klar besser"
    assert 0.0 <= pv < 0.5, f"Einseitiger p-Wert bei positivem CW sollte < 0.5 sein, got {pv}"


def test_cw_adjustment_exceeds_dm_numerator():
    """CW-Numerator (mean f_t^CW) ist >= DM-Numerator (mean f_t^DM).

    f_t^CW = f_t^DM + (e_mod - e_rw)^2  => mean(f^CW) >= mean(f^DM).
    """
    np.random.seed(13)
    e_rw  = np.random.randn(50)
    e_mod = np.random.randn(50)
    f_dm  = e_rw**2 - e_mod**2
    f_cw  = e_rw**2 - (e_mod**2 - (e_mod - e_rw)**2)
    assert f_cw.mean() >= f_dm.mean() - 1e-12, (
        f"CW-Numerator {f_cw.mean():.6f} sollte >= DM-Numerator {f_dm.mean():.6f}"
    )


# ── Bonferroni-Korrektur ──────────────────────────────────────────────────────

def test_bonferroni_correct_basic():
    """p_adj_i = min(n * p_i, 1.0) fuer alle nicht-NaN Eintraege."""
    p_values = [0.05, 0.10, 0.02]   # n = 3
    p_adj = bonferroni_correct(p_values)
    expected = [min(3 * p, 1.0) for p in p_values]
    np.testing.assert_allclose(p_adj, expected)


def test_bonferroni_correct_nan_passthrough():
    """NaN-Eintraege (Referenzmodell ohne Test) werden unveraendert weitergereicht."""
    p_values = [np.nan, 0.05, 0.10]   # n_valid = 2
    p_adj = bonferroni_correct(p_values)
    assert np.isnan(p_adj[0]), "NaN-Eintrag soll unveraendert bleiben"
    np.testing.assert_allclose(p_adj[1], min(2 * 0.05, 1.0))
    np.testing.assert_allclose(p_adj[2], min(2 * 0.10, 1.0))


def test_bonferroni_correct_clamp():
    """Korrigierte p-Werte werden auf 1.0 gedeckelt."""
    p_values = [0.8, 0.9]   # 2 * 0.8 = 1.6 > 1.0
    p_adj = bonferroni_correct(p_values)
    assert all(v <= 1.0 for v in p_adj), "Kein p_adj darf 1.0 ueberschreiten"
    np.testing.assert_allclose(p_adj, [1.0, 1.0])


# ── Rolling-Origin ────────────────────────────────────────────────────────────

def test_rolling_origin_length():
    """Rolling-Origin soll len(y) - start Prognosen liefern."""
    np.random.seed(3)
    n   = 50; start = 30
    X   = pd.DataFrame(np.random.randn(n, 3))
    y   = pd.Series(np.random.randn(n))
    from sklearn.linear_model import LinearRegression
    preds = rolling_origin(lambda: LinearRegression(), X, y, start)
    assert len(preds) == n - start, \
        f"Erwartet {n - start} Prognosen, erhalten {len(preds)}"


def test_rolling_origin_index_alignment():
    """Prognose-Index soll dem y-Index ab start entsprechen."""
    np.random.seed(4)
    idx = pd.date_range("2020-01", periods=40, freq="MS")
    X   = pd.DataFrame(np.random.randn(40, 2), index=idx)
    y   = pd.Series(np.random.randn(40), index=idx)
    from sklearn.linear_model import LinearRegression
    start = 25
    preds = rolling_origin(lambda: LinearRegression(), X, y, start)
    expected_idx = idx[start:]
    pd.testing.assert_index_equal(preds.index, expected_idx)


# ── Regime-Analyse ────────────────────────────────────────────────────────────

def test_regime_analysis_rw_reference():
    """RW hat RMSE/RW = 1.0 in beiden Regimen (Selbst-Referenz)."""
    idx = pd.date_range("2021-06", periods=40, freq="MS")
    rng = np.random.default_rng(42)
    n   = len(idx)
    y   = pd.Series(rng.standard_normal(n), index=idx)
    rw  = pd.Series(rng.standard_normal(n), index=idx)
    ar  = pd.Series(rng.standard_normal(n), index=idx)
    oos_ctx = {
        "oos_df":    pd.DataFrame({"RW": rw, "AR": ar}),
        "y_oos_ref": y,
    }
    result = compute_regime_analysis(oos_ctx, shock_end="2023-03")
    df     = result["df_regime"]
    assert np.isclose(df.loc["RW", "RMSE/RW Schock"],   1.0, atol=1e-10), \
        "RW RMSE/RW Schock sollte exakt 1.0 sein"
    assert np.isclose(df.loc["RW", "RMSE/RW Disinfl."], 1.0, atol=1e-10), \
        "RW RMSE/RW Disinfl. sollte exakt 1.0 sein"
    assert np.isclose(df.loc["RW", "RMSE/RW Gesamt"],   1.0, atol=1e-10), \
        "RW RMSE/RW Gesamt sollte exakt 1.0 sein"


def test_regime_analysis_disjoint_split():
    """n_shock + n_disfl == n (Regime-Split ist disjunkt und vollstaendig)."""
    idx = pd.date_range("2021-06", periods=40, freq="MS")
    rng = np.random.default_rng(7)
    n   = len(idx)
    y   = pd.Series(rng.standard_normal(n), index=idx)
    oos_ctx = {
        "oos_df": pd.DataFrame({
            "RW": rng.standard_normal(n),
            "AR": rng.standard_normal(n),
        }, index=idx),
        "y_oos_ref": y,
    }
    result = compute_regime_analysis(oos_ctx, shock_end="2023-03")
    assert result["n_shock"] + result["n_disfl"] == n, \
        "n_shock + n_disfl muss gleich Gesamt-n sein"
    assert result["n_shock"] > 0, "Schock-Regime darf nicht leer sein"
    assert result["n_disfl"] > 0, "Disinflations-Regime darf nicht leer sein"


# ── Giacomini-Rossi-Fluctuation-Test ─────────────────────────────────────────

def _make_gr_ctx(y_vals, rw_vals, model_vals, model_name="M",
                 start="2021-06", freq="MS"):
    """Hilfsfunktion: baut einen minimalen oos_ctx für compute_giacomini_rossi."""
    T   = len(y_vals)
    idx = pd.date_range(start, periods=T, freq=freq)
    return {
        "oos_df": pd.DataFrame(
            {"RW": rw_vals, model_name: model_vals}, index=idx
        ),
        "y_oos_ref": pd.Series(y_vals, index=idx),
    }


def test_gr_constant_advantage_always_positive():
    """Konstanter Modell-Vorteil (d_t > 0 stets) → alle GR-Statistiken positiv.

    DoD-Sanity 1: 'konstanter Vorteil ⇒ flache Statistik' (flach & positiv).
    """
    rng = np.random.default_rng(42)
    T   = 36
    y   = np.zeros(T)
    rw  = np.ones(T) * 1.0   # RW-Fehler = 1
    mod = np.ones(T) * 0.3   # Modell-Fehler = 0.3 (klar besser)
    # Kleines Rauschen, damit die HAC-Varianz > 0
    noise = rng.normal(0, 0.02, T)
    rw  = rw + noise
    mod = mod + noise * 0.5

    oos_ctx = _make_gr_ctx(y, rw, mod, model_name="AR")
    gr_ctx  = compute_giacomini_rossi(oos_ctx, m=12)
    gr_ar   = gr_ctx["gr_df"]["AR"]

    assert (gr_ar > 0).all(), (
        f"Alle GR-Statistiken sollten bei konstantem Vorteil positiv sein; "
        f"min={gr_ar.min():.3f}"
    )


def test_gr_regime_break_sign_change():
    """Vorzeichenwechsel an konstruiertem Regimebruch → GR_t wechselt das Vorzeichen.

    DoD-Sanity 2: 'Vorzeichenwechsel an konstruiertem Regimebruch ⇒ Bandüberschreitung'.
    Wir prüfen mindestens, dass die GR-Reihe sowohl positive als auch negative Werte
    enthält — ein Vorzeichenwechsel setzt einen Vorteils-/Nachteilswechsel voraus.
    """
    T   = 36
    y   = np.zeros(T)
    rw  = np.ones(T)
    # Erste Hälfte: Modell schlechter (d_t < 0); zweite Hälfte: Modell besser (d_t > 0)
    mod = np.empty(T)
    mod[:T // 2]  = 0.1   # Modell fast gleich wie RW (aber schlechter im Quadrat)
    mod[T // 2:]  = 2.0   # Modell viel schlechter → d_t negativ in 2. Hälfte

    # Wir drehen: d_t = e_RW^2 - e_M^2. Wenn RW=1 und mod=0.1 → d=1-0.01=0.99 (positiv).
    # Wenn RW=1 und mod=2 → d=1-4=-3 (negativ). So hat erste Hälfte d>0, zweite d<0.
    oos_ctx = _make_gr_ctx(y, rw, mod, model_name="LASSO+HVPI")
    gr_ctx  = compute_giacomini_rossi(oos_ctx, m=10)
    gr_vals = gr_ctx["gr_df"]["LASSO+HVPI"].values

    has_positive = bool((gr_vals > 0).any())
    has_negative = bool((gr_vals < 0).any())
    assert has_positive and has_negative, (
        f"GR-Reihe sollte bei Regimebruch Vorzeichen wechseln; "
        f"min={gr_vals.min():.3f}, max={gr_vals.max():.3f}"
    )


def test_rolling_origin_no_lookahead():
    """Prognose an Zeitpunkt t darf nur Daten bis t-1 nutzen (kein Look-ahead).

    Wir setzen y[0:start] = 0, y[start:] = 1. Ein korrektes Expanding-Window-Modell
    trainiert auf [0:start] und sagt fuer t=start einen Wert nahe 0 voraus,
    nicht den wahren Wert 1.
    """
    np.random.seed(5)
    n = 60; start = 40
    X = pd.DataFrame(np.random.randn(n, 1))
    y = pd.Series([0.0] * start + [1.0] * (n - start))
    from sklearn.linear_model import LinearRegression
    preds = rolling_origin(lambda: LinearRegression(), X, y, start)
    # Erste Prognose (trainiert nur auf Nullen) soll nahe 0 sein, nicht 1
    first_pred = preds.iloc[0]
    assert abs(first_pred) < 0.5, \
        f"Look-ahead-Verdacht: erste Prognose = {first_pred:.3f} (erwartet ≈ 0)"


# ── MoM-Robustheitsprüfung (AP29) ─────────────────────────────────────────────

def _make_raw_df(T=220, n_pred=2, seed=99):
    """Synthetisches df_raw: HVPI + n_pred Praediktor-Preisniveaureihen."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2004-01", periods=T, freq="MS")
    data = {"HVPI": 100 * np.exp(np.cumsum(rng.normal(0.002, 0.003, T)))}
    for i in range(n_pred):
        data[f"P{i}"] = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.004, T)))
    return pd.DataFrame(data, index=idx)


def test_robustness_mom_output_structure():
    """compute_robustness_mom gibt Dict mit DataFrame zurueck (Struktur-Smoke-Test)."""
    df_raw = _make_raw_df()
    result = compute_robustness_mom(df_raw, test_months=20)

    assert "df_robustness_mom" in result, "Schluessel 'df_robustness_mom' fehlt"
    df = result["df_robustness_mom"]
    assert isinstance(df, pd.DataFrame), "df_robustness_mom muss ein DataFrame sein"


def test_robustness_mom_expected_models():
    """Alle sechs Modelle (RW, AO, AR, Ridge, LASSO, LASSO+HVPI) sind vorhanden."""
    df_raw = _make_raw_df()
    df = compute_robustness_mom(df_raw, test_months=20)["df_robustness_mom"]

    expected = {"RW", "AO (Atkeson-Ohanian)", "AR", "Ridge", "LASSO", "LASSO+HVPI"}
    assert set(df.index) == expected, (
        f"Fehlende/unerwartete Modelle: {set(df.index).symmetric_difference(expected)}"
    )


def test_robustness_mom_rw_self_reference():
    """RW hat RMSE/RW = 1.0 (Selbstreferenz)."""
    df_raw = _make_raw_df()
    df = compute_robustness_mom(df_raw, test_months=20)["df_robustness_mom"]
    assert np.isclose(df.loc["RW", "RMSE/RW"], 1.0, atol=1e-10), (
        f"RW RMSE/RW sollte exakt 1.0 sein, got {df.loc['RW', 'RMSE/RW']}"
    )


def test_robustness_mom_ao_self_reference():
    """AO hat RMSE/AO = 1.0 (Selbstreferenz)."""
    df_raw = _make_raw_df()
    df = compute_robustness_mom(df_raw, test_months=20)["df_robustness_mom"]
    assert np.isclose(df.loc["AO (Atkeson-Ohanian)", "RMSE/AO"], 1.0, atol=1e-10), (
        f"AO RMSE/AO sollte exakt 1.0 sein, got {df.loc['AO (Atkeson-Ohanian)', 'RMSE/AO']}"
    )


def test_robustness_mom_positive_rmse():
    """Alle RMSE-Werte sind endlich und positiv."""
    df_raw = _make_raw_df()
    df = compute_robustness_mom(df_raw, test_months=20)["df_robustness_mom"]
    rmse_col = df["Test RMSE (MoM)"]
    assert rmse_col.notna().all(), "Keine NaN-Werte in RMSE-Spalte erwartet"
    assert (rmse_col > 0).all(), f"Alle RMSE-Werte muessen positiv sein:\n{rmse_col}"
