import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


DATA_PATH = "dataset.csv"
TARGET_COL = "Affects_Academic_Performance"


@dataclass(frozen=True)
class MLArtifacts:
    model: RandomForestClassifier
    expected_columns: list[str]
    imputations: dict[str, object]
    accuracy: float
    feature_importance: pd.Series


def inject_argon_css() -> None:
    st.markdown(
        """
        <style>
          /* Hide Streamlit chrome */
          #MainMenu {visibility: hidden;}
          footer {visibility: hidden;}
          header {visibility: hidden;}

          /* App background (Argon-like) */
          .stApp {
            background: linear-gradient(180deg, #f7fafc 0%, #eef2f7 60%, #e9edf5 100%);
          }

          /* Top navigation (real HTML bar) */
          .nf-topbar {
            background: #5e72e4;
            border-radius: 12px;
            padding: 10px 12px;
            box-shadow: 0 10px 24px rgba(23, 43, 77, 0.25);
            border: 1px solid rgba(255,255,255,0.10);
            margin: 10px 0 18px 0;
          }

          /* Streamlit topbar container styling (anchored) */
          [data-testid="stVerticalBlockBorderWrapper"]:has(#nf-topbar-anchor) {
            background: #5e72e4 !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255,255,255,0.10) !important;
            box-shadow: 0 10px 24px rgba(23, 43, 77, 0.25) !important;
          }
          [data-testid="stVerticalBlockBorderWrapper"]:has(#nf-topbar-anchor) > div {
            background: transparent !important;
          }
          [data-testid="stVerticalBlockBorderWrapper"]:has(#nf-topbar-anchor) * {
            color: rgba(255,255,255,0.92) !important;
          }
          [data-testid="stVerticalBlockBorderWrapper"]:has(#nf-topbar-anchor) .stButton > button {
            width: 100%;
            border-radius: 10px !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            background: rgba(255,255,255,0.06) !important;
            color: rgba(255,255,255,0.92) !important;
            font-weight: 750 !important;
            padding: 0.55rem 0.75rem !important;
          }
          [data-testid="stVerticalBlockBorderWrapper"]:has(#nf-topbar-anchor) .stButton > button:hover {
            background: rgba(255,255,255,0.10) !important;
            transform: translateY(-1px);
          }
          [data-testid="stVerticalBlockBorderWrapper"]:has(#nf-topbar-anchor) .stButton > button:focus {
            box-shadow: 0 0 0 3px rgba(94,114,228,0.25) !important;
          }
          /* Active tab gets a stronger tint */
          [data-testid="stVerticalBlockBorderWrapper"]:has(#nf-topbar-anchor) .nf-active .stButton > button {
            background: rgba(94,114,228,0.26) !important;
            border-color: rgba(94,114,228,0.55) !important;
          }
          #nf-topbar-anchor { display: none; }

          /* Cards */
          .card {
            background: #ffffff;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.10);
            padding: 20px;
            border: 1px solid rgba(0,0,0,0.04);
          }
          .dark-card {
            background: #172b4d;
            color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 10px 24px rgba(23, 43, 77, 0.35);
            border: 1px solid rgba(255,255,255,0.08);
          }

          .kpi-title { font-size: 13px; letter-spacing: 0.2px; color: rgba(0,0,0,0.55); margin: 0 0 6px 0; }
          .kpi-value { font-size: 26px; font-weight: 700; margin: 0; color: #32325d; }
          .kpi-sub { font-size: 12px; margin-top: 6px; color: rgba(0,0,0,0.55); }

          .dark-title { font-size: 14px; letter-spacing: 0.2px; color: rgba(255,255,255,0.78); margin: 0 0 6px 0; }
          .dark-value { font-size: 22px; font-weight: 700; margin: 0; color: #ffffff; }

          .alert {
            border-radius: 10px;
            padding: 14px 16px;
            background: rgba(255, 183, 77, 0.14);
            border: 1px solid rgba(255, 183, 77, 0.30);
            color: #172b4d;
          }
          .alert strong { font-weight: 750; }

          /* Tweak inputs a bit */
          .stSlider > div { padding-top: 0.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def _set_page(page_key: str) -> None:
    st.query_params["page"] = page_key
    st.rerun()


def plotly_transparent_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10),
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial", size=12),
    )
    return fig


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise

    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = (
        df[TARGET_COL]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
        .astype("Int64")
    )
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    if "Student_ID" in df.columns:
        df = df.drop(columns=["Student_ID"])

    return df


def _compute_imputations(x_train: pd.DataFrame) -> dict[str, object]:
    imps: dict[str, object] = {}
    for col in x_train.columns:
        if pd.api.types.is_numeric_dtype(x_train[col]):
            imps[col] = float(x_train[col].median())
        else:
            imps[col] = x_train[col].mode(dropna=True)[0]
    return imps


def _apply_imputations(df: pd.DataFrame, imps: dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    for col, val in imps.items():
        if col in out.columns:
            out[col] = out[col].fillna(val)
    return out


@st.cache_resource(show_spinner=True)
def train_model(df: pd.DataFrame) -> MLArtifacts:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Colonne cible manquante: {TARGET_COL}")

    y = df[TARGET_COL]
    x = df.drop(columns=[TARGET_COL])

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    imputations = _compute_imputations(x_train_raw)
    x_train_imp = _apply_imputations(x_train_raw, imputations)
    x_test_imp = _apply_imputations(x_test_raw, imputations)

    x_train = pd.get_dummies(x_train_imp, drop_first=False)
    expected_columns = list(x_train.columns)

    x_test = pd.get_dummies(x_test_imp, drop_first=False).reindex(columns=expected_columns, fill_value=0)

    model = RandomForestClassifier(class_weight="balanced", random_state=42)
    model.fit(x_train, y_train)

    accuracy = float(model.score(x_test, y_test))

    fi = pd.Series(model.feature_importances_, index=expected_columns).sort_values(ascending=True)

    return MLArtifacts(
        model=model,
        expected_columns=expected_columns,
        imputations=imputations,
        accuracy=accuracy,
        feature_importance=fi,
    )


def simulate_waves(minutes: int) -> pd.DataFrame:
    t = np.arange(0, 31)
    alpha = 70 - 0.9 * t - 12.0 * (t >= 10) - 0.25 * np.maximum(0, t - 10)
    delta = 25 + 0.6 * t + 12.0 * (t >= 10) + 0.35 * np.maximum(0, t - 10)
    alpha = np.clip(alpha, 0, 100)
    delta = np.clip(delta, 0, 100)
    df = pd.DataFrame({"minute": t, "Alpha": alpha, "Delta": delta})
    df["cursor"] = minutes
    return df


def kpi_card(title: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dark_kpi_card(title: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="dark-card">
          <div class="dark-title">{title}</div>
          <div class="dark-value">{value}</div>
          <div style="font-size:12px;color:rgba(255,255,255,0.75);margin-top:6px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_dashboard() -> None:
    st.markdown("## Dashboard Neuro‑Focus")

    minutes = st.slider("Temps de scroll actuel (minutes)", 0, 30, 0, 1)

    status = "Stable" if minutes < 10 else "Pause recommandée"
    alpha_now = int(simulate_waves(minutes).loc[simulate_waves(minutes)["minute"] == minutes, "Alpha"].iloc[0])

    k1, k2, k3 = st.columns(3, gap="large")
    with k1:
        kpi_card("Temps de Session", f"{minutes} min", "Fenêtre d’observation instantanée")
    with k2:
        kpi_card("Statut Cognitif", status, "Lecture indicative")
    with k3:
        kpi_card("Ondes Alpha", f"{alpha_now}", "Indice simulé")

    st.markdown("")
    waves = simulate_waves(minutes)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=waves["minute"],
            y=waves["Alpha"],
            mode="lines",
            name="Alpha",
            line=dict(color="#5e72e4", width=3, shape="spline", smoothing=1.2),
            fill="tozeroy",
            fillcolor="rgba(94,114,228,0.20)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=waves["minute"],
            y=waves["Delta"],
            mode="lines",
            name="Delta",
            line=dict(color="#2dce89", width=3, shape="spline", smoothing=1.2),
            fill="tozeroy",
            fillcolor="rgba(45,206,137,0.16)",
        )
    )
    fig.update_layout(
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Minutes", gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="Indice", gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.08)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(color="rgba(255,255,255,0.85)")
    fig.update_yaxes(color="rgba(255,255,255,0.85)")
    fig.update_layout(
        template="plotly_dark",
    )

    # Plotly charts are rendered outside of raw HTML blocks; wrapping them in a single
    # <div> can create a "empty bar" artifact. Use a Streamlit container and add
    # a dark card header + spacing for a clean Argon-style section.
    with st.container():
        st.markdown(
            """
            <div class="dark-card" style="padding-bottom:12px;">
              <div style="font-weight:650;margin:0 0 10px 0;">Évolution simulée des ondes (Area Chart)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    if minutes >= 10:
        st.markdown(
            """
            <div class="alert">
              <strong>Recommandation</strong><br/>
              Une pause courte est suggérée après ~10 minutes de scrolling continu (Satani et al.).
            </div>
            """,
            unsafe_allow_html=True,
        )


def breathing_478_component() -> None:
    html = """
    <div style="width:100%;max-width:720px;margin:0 auto;">
      <div style="background:#ffffff;border-radius:10px;box-shadow:0 4px 6px rgba(0,0,0,0.1);padding:20px;">
        <div style="font-size:14px;color:rgba(0,0,0,0.55);margin-bottom:10px;">Cohérence cardiaque 4‑7‑8</div>
        <div id="phase" style="font-size:22px;font-weight:700;color:#32325d;margin-bottom:12px;">Prêt</div>
        <div style="height:18px;background:rgba(50,50,93,0.08);border-radius:999px;overflow:hidden;">
          <div id="bar" style="height:100%;width:0%;background:#5e72e4;border-radius:999px;transition:width 0.1s linear;"></div>
        </div>
        <div style="display:flex;gap:12px;align-items:center;margin-top:14px;">
          <button id="start" style="background:#5e72e4;color:white;border:none;border-radius:10px;padding:10px 14px;font-weight:650;cursor:pointer;">
            Démarrer
          </button>
          <div id="timer" style="color:rgba(0,0,0,0.55);font-size:13px;">4s inspirer • 7s bloquer • 8s expirer</div>
        </div>
        <div style="margin-top:18px;display:flex;justify-content:center;">
          <div id="circleWrap" style="position:relative;width:130px;height:130px;display:flex;align-items:center;justify-content:center;">
            <div id="circle" style="position:absolute;inset:10px;border-radius:999px;background:rgba(94,114,228,0.18);border:2px solid rgba(94,114,228,0.35);"></div>
            <div id="circleText" style="position:relative;z-index:2;text-align:center;font-weight:750;color:#32325d;">
              Prêt
            </div>
          </div>
        </div>
      </div>
    </div>
    <script>
      const phases = [
        { label: "Inspirez (4s)", short: "Inspirez", seconds: 4, scaleFrom: 1.00, scaleTo: 1.45 },
        { label: "Retenez (7s)", short: "Retenez", seconds: 7, scaleFrom: 1.45, scaleTo: 1.45 },
        { label: "Expirez (8s)", short: "Expirez", seconds: 8, scaleFrom: 1.45, scaleTo: 1.00 },
      ];
      const phaseEl = document.getElementById("phase");
      const bar = document.getElementById("bar");
      const circle = document.getElementById("circle");
      const circleText = document.getElementById("circleText");
      const startBtn = document.getElementById("start");
      const timerEl = document.getElementById("timer");

      let running = false;

      function lerp(a,b,t){ return a + (b-a)*t; }

      async function run(){
        if(running) return;
        running = true;
        startBtn.disabled = true;
        startBtn.style.opacity = 0.6;

        const total = phases.reduce((s,p)=>s+p.seconds,0);
        let elapsed = 0;

        for(const p of phases){
          phaseEl.textContent = p.label;
          circleText.textContent = p.short;
          const t0 = performance.now();
          while(true){
            const now = performance.now();
            const dt = (now - t0) / 1000.0;
            const prog = Math.min(1.0, dt / p.seconds);

            const scale = lerp(p.scaleFrom, p.scaleTo, prog);
            circle.style.transform = `scale(${scale})`;
            circle.style.transition = "transform 0.08s linear";

            const globalProg = Math.min(1.0, (elapsed + dt) / total);
            bar.style.width = (globalProg * 100).toFixed(1) + "%";
            const remaining = Math.max(0, Math.ceil(p.seconds - dt));
            timerEl.textContent = remaining + "s";

            if(dt >= p.seconds) break;
            await new Promise(r => setTimeout(r, 50));
          }
          elapsed += p.seconds;
        }

        phaseEl.textContent = "Terminé";
        circleText.textContent = "Terminé";
        timerEl.textContent = "Respirez naturellement.";
        startBtn.disabled = false;
        startBtn.style.opacity = 1.0;
        running = false;
      }

      startBtn.addEventListener("click", run);
    </script>
    """
    components.html(html, height=400)


def memo_init() -> None:
    if "memo_cards" not in st.session_state:
        st.session_state.memo_cards = ["⬡", "⬡", "⚲", "⚲", "✧", "✧"]
        rng = np.random.default_rng(int(time.time() * 1000) % (2**32 - 1))
        rng.shuffle(st.session_state.memo_cards)
        st.session_state.memo_revealed = [False] * 6
        st.session_state.memo_matched = [False] * 6
        st.session_state.memo_selected = []
        st.session_state.memo_lock = False
        st.session_state.memo_preview_until = time.time() + 2.0
        st.session_state.memo_pending_hide = None  # tuple[int,int] | None
        st.session_state.memo_pending_until = 0.0


def memo_reset() -> None:
    st.session_state.memo_cards = ["⬡", "⬡", "⚲", "⚲", "✧", "✧"]
    rng = np.random.default_rng(int(time.time() * 1000) % (2**32 - 1))
    rng.shuffle(st.session_state.memo_cards)
    st.session_state.memo_revealed = [False] * 6
    st.session_state.memo_matched = [False] * 6
    st.session_state.memo_selected = []
    st.session_state.memo_lock = False
    st.session_state.memo_preview_until = time.time() + 2.0
    st.session_state.memo_pending_hide = None
    st.session_state.memo_pending_until = 0.0


def memo_on_click(idx: int) -> None:
    if st.session_state.memo_lock:
        return
    if st.session_state.memo_matched[idx] or st.session_state.memo_revealed[idx]:
        return
    if len(st.session_state.memo_selected) >= 2:
        return

    st.session_state.memo_revealed[idx] = True
    st.session_state.memo_selected.append(idx)

    if len(st.session_state.memo_selected) == 2:
        a, b = st.session_state.memo_selected
        if st.session_state.memo_cards[a] == st.session_state.memo_cards[b]:
            st.session_state.memo_matched[a] = True
            st.session_state.memo_matched[b] = True
            st.session_state.memo_selected = []
            return

        # Mismatch flow: lock, show both (via rerun), then hide after delay.
        st.session_state.memo_lock = True
        st.session_state.memo_pending_hide = (a, b)
        st.session_state.memo_pending_until = time.time() + 1.5


def render_memo_game() -> None:
    memo_init()

    st.markdown("### Jeu de Cartes Mémo")
    c_top = st.columns([1, 1, 1])
    with c_top[2]:
        if st.button("Mélanger et recommencer"):
            memo_reset()
            st.rerun()

    now = time.time()
    preview_mode = bool(st.session_state.memo_preview_until and now < float(st.session_state.memo_preview_until))
    observe_mode = bool(
        st.session_state.memo_pending_hide is not None and now < float(st.session_state.memo_pending_until)
    )

    if preview_mode:
        st.session_state.memo_lock = True
        st.session_state.memo_revealed = [True] * 6
        remaining = max(0.0, float(st.session_state.memo_preview_until) - now)
        st.caption(f"Mémorisez les symboles… ({remaining:.1f}s)")
    elif st.session_state.memo_preview_until:
        st.session_state.memo_preview_until = 0.0
        st.session_state.memo_lock = False
        st.session_state.memo_revealed = [False] * 6
        st.session_state.memo_selected = []

    if observe_mode:
        st.session_state.memo_lock = True
        st.caption("Observez…")
    elif st.session_state.memo_pending_hide is not None:
        a, b = st.session_state.memo_pending_hide
        st.session_state.memo_revealed[a] = False
        st.session_state.memo_revealed[b] = False
        st.session_state.memo_selected = []
        st.session_state.memo_pending_hide = None
        st.session_state.memo_pending_until = 0.0
        st.session_state.memo_lock = False

    clicked_idx: int | None = None
    grid = st.columns(3, gap="small")
    for i in range(6):
        with grid[i % 3]:
            face = (
                st.session_state.memo_cards[i]
                if (st.session_state.memo_revealed[i] or st.session_state.memo_matched[i])
                else "•"
            )
            disabled = st.session_state.memo_lock or st.session_state.memo_matched[i]
            pressed = st.button(face, key=f"memo_{i}", disabled=disabled, use_container_width=True)
            if pressed:
                clicked_idx = i

    if clicked_idx is not None:
        memo_on_click(clicked_idx)
        st.rerun()

    if all(st.session_state.memo_matched):
        st.success("Toutes les paires sont trouvées.")

    # Tick countdowns AFTER rendering so the browser actually shows the cards.
    if preview_mode:
        time.sleep(0.25)
        st.rerun()
    if observe_mode:
        time.sleep(0.25)
        st.rerun()


def page_recovery() -> None:
    st.markdown("## Centre de Récupération Cognitive")

    c1, c2 = st.columns([1.2, 1.0], gap="large")
    with c1:
        breathing_478_component()
    with c2:
        render_memo_game()


def page_technical() -> None:
    st.markdown("## Analyse Technique & Prédiction")

    try:
        df = load_dataset(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Fichier introuvable: `{DATA_PATH}`. Place `dataset.csv` dans le même dossier que `app.py`.")
        return

    artifacts = train_model(df)

    c1, c2 = st.columns([1, 1.3], gap="large")
    with c1:
        dark_kpi_card("Accuracy (test)", f"{artifacts.accuracy:.3f}", "Train/Test split 80/20")
    with c2:
        fi = artifacts.feature_importance.tail(15)
        fi_df = fi.reset_index()
        fi_df.columns = ["feature", "importance"]
        fig = px.bar(
            fi_df,
            x="importance",
            y="feature",
            orientation="h",
            title="Feature Importance (Top)",
        )
        fig = plotly_transparent_layout(fig)
        fig.update_layout(yaxis_title=None, xaxis_title=None)
        fig.update_traces(marker_color="#5e72e4")
        # Same issue as Dashboard: wrapping Plotly inside raw HTML can create empty "debug" cards.
        with st.container():
            st.markdown(
                """
                <div class="card" style="margin-bottom:10px;">
                  <div class="kpi-title">Feature Importance</div>
                  <div style="font-weight:750;color:#32325d;margin-top:2px;">Top variables</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Prédiction")

    with st.form("predict_form"):
        screen_hours = st.slider("Heures d'écran (par jour)", 0.0, 16.0, 4.0, 0.5)
        sleep_hours = st.slider("Sommeil (heures par nuit)", 0.0, 12.0, 7.0, 0.5)
        mental_health = st.slider("Santé mentale (0–100)", 0.0, 100.0, 60.0, 1.0)
        submit = st.form_submit_button("Lancer la prédiction")

    if not submit:
        return

    # Critique: initialiser toutes les valeurs depuis le dictionnaire d'imputation
    entry = dict(artifacts.imputations)
    entry["Avg_Daily_Usage_Hours"] = float(screen_hours)
    entry["Sleep_Hours_Per_Night"] = float(sleep_hours)
    entry["Mental_Health_Score"] = float(mental_health)

    row = pd.DataFrame([entry])
    row = pd.get_dummies(row, drop_first=False).reindex(columns=artifacts.expected_columns, fill_value=0)

    pred = int(artifacts.model.predict(row)[0])
    proba = float(artifacts.model.predict_proba(row)[0][1])

    label = "Yes" if pred == 1 else "No"
    color = "#fb6340" if pred == 1 else "#2dce89"
    st.markdown(
        f"""
        <div class="card" style="border-left:6px solid {color};">
          <div class="kpi-title">Résultat</div>
          <div class="kpi-value" style="font-size:22px;">{label}</div>
          <div class="kpi-sub">Score (probabilité classe 1): <strong>{proba:.2f}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    # Do NOT rely on native sidebar: some layouts/screens can hide it completely.
    st.set_page_config(layout="wide", page_title="Neuro-Focus PoC")
    inject_argon_css()

    qp = st.query_params
    page_key = (qp.get("page") or "dashboard").strip().lower()
    if page_key not in {"dashboard", "recovery", "tech"}:
        page_key = "dashboard"

    # Topbar: real Streamlit content (buttons) so it works reliably.
    topbar = st.container(border=True)
    with topbar:
        st.markdown('<div id="nf-topbar-anchor"></div>', unsafe_allow_html=True)
        left, right = st.columns([0.26, 0.74], vertical_alignment="center")
        with left:
            st.image("stopthis.png", width=160)
        with right:
            b1, b2, b3 = st.columns(3, gap="small")
            with b1:
                c = st.container()
                with c:
                    if st.button("Dashboard", key="nav_dashboard", use_container_width=True):
                        _set_page("dashboard")
                if page_key == "dashboard":
                    c.markdown('<div class="nf-active"></div>', unsafe_allow_html=True)
            with b2:
                c = st.container()
                with c:
                    if st.button("Récupération", key="nav_recovery", use_container_width=True):
                        _set_page("recovery")
                if page_key == "recovery":
                    c.markdown('<div class="nf-active"></div>', unsafe_allow_html=True)
            with b3:
                c = st.container()
                with c:
                    if st.button("Analyse technique", key="nav_tech", use_container_width=True):
                        _set_page("tech")
                if page_key == "tech":
                    c.markdown('<div class="nf-active"></div>', unsafe_allow_html=True)

    if page_key == "dashboard":
        page_dashboard()
    elif page_key == "recovery":
        page_recovery()
    else:
        page_technical()


if __name__ == "__main__":
    main()

