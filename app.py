
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import pickle

st.set_page_config(page_title="Football Predictor Pro", page_icon="⚽", layout="wide")

@st.cache_resource
def load_models():
    with open('all_leagues_models.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data(league_name):
    urls_map = {
        'Premier League': ["https://www.football-data.co.uk/mmz4281/2526/E0.csv",
                           "https://www.football-data.co.uk/mmz4281/2425/E0.csv"],
        'La Liga':        ["https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
                           "https://www.football-data.co.uk/mmz4281/2425/SP1.csv"],
        'Bundesliga':     ["https://www.football-data.co.uk/mmz4281/2526/D1.csv",
                           "https://www.football-data.co.uk/mmz4281/2425/D1.csv"],
        'Serie A':        ["https://www.football-data.co.uk/mmz4281/2526/I1.csv",
                           "https://www.football-data.co.uk/mmz4281/2425/I1.csv"],
        'Ligue 1':        ["https://www.football-data.co.uk/mmz4281/2526/F1.csv",
                           "https://www.football-data.co.uk/mmz4281/2425/F1.csv"],
    }
    dfs = []
    for url in urls_map.get(league_name, []):
        try: dfs.append(pd.read_csv(url))
        except: pass
    if not dfs: return pd.DataFrame()
    return pd.concat(dfs)[["HomeTeam","AwayTeam","FTHG","FTAG","FTR"]].dropna().reset_index(drop=True)

def predict(home, away, model, df, factors):
    ts = model["team_stats"]
    form = model["form"]
    ha = model["home_adv"]
    avg_h = model["avg_home"]
    avg_a = model["avg_away"]
    if home not in ts or away not in ts: return None
    h = ts[home]; a = ts[away]
    lh = h["attack_home"] * a["defense_away"] * avg_h
    la = a["attack_away"] * h["defense_home"] * avg_a
    lh *= 1 + 0.20*(form[home]["form_pts"]-1.5)/1.5
    la *= 1 + 0.20*(form[away]["form_pts"]-1.5)/1.5
    lh *= 1 + 0.15*(ha.get(home,1.0)-1.0)
    h2h = df[((df["HomeTeam"]==home)&(df["AwayTeam"]==away))|
              ((df["HomeTeam"]==away)&(df["AwayTeam"]==home))].tail(6)
    if len(h2h) > 0:
        hw = len(h2h[((h2h["HomeTeam"]==home)&(h2h["FTR"]=="H"))|
                      ((h2h["AwayTeam"]==home)&(h2h["FTR"]=="A"))])
        aw = len(h2h[((h2h["HomeTeam"]==away)&(h2h["FTR"]=="H"))|
                      ((h2h["AwayTeam"]==away)&(h2h["FTR"]=="A"))])
        lh *= 1 + 0.10*(hw/len(h2h)-0.45)
        la *= 1 + 0.10*(aw/len(h2h)-0.30)

    # Absences domicile — impact selon importance du joueur
    home_absence_impact = 1.0 - (factors["home_absence_level"] * 0.08)
    away_absence_impact = 1.0 - (factors["away_absence_level"] * 0.08)
    lh *= home_absence_impact
    la *= away_absence_impact

    # Gardien absent
    if factors["home_gk_out"]: lh *= 0.90; la *= 1.12
    if factors["away_gk_out"]: la *= 0.90; lh *= 1.12

    # Fatigue
    fatigue_map = {"Aucune": 1.0, "Légère (1 match)": 0.96, "Forte (2+ matchs)": 0.90}
    lh *= fatigue_map.get(factors["home_fatigue"], 1.0)
    la *= fatigue_map.get(factors["away_fatigue"], 1.0)

    # Motivation
    motivation_map = {"Normal": 1.0, "Élevée (derby/finale)": 1.08, "Faible (déjà relégué)": 0.88}
    lh *= motivation_map.get(factors["home_motivation"], 1.0)
    la *= motivation_map.get(factors["away_motivation"], 1.0)

    # Météo
    weather_map = {"Normale": 1.0, "Pluie": 0.93, "Vent fort": 0.90, "Neige": 0.85}
    weather_factor = weather_map.get(factors["weather"], 1.0)
    lh *= weather_factor
    la *= weather_factor

    probs = np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            probs[i,j] = poisson.pmf(i,lh)*poisson.pmf(j,la)
    ph = float(np.sum(np.tril(probs,-1)))
    pd_ = float(np.sum(np.diag(probs)))
    pa = float(np.sum(np.triu(probs,1)))
    bs = np.unravel_index(np.argmax(probs), probs.shape)
    scores = sorted([(i,j,probs[i,j]) for i in range(8) for j in range(8)],
                    key=lambda x: x[2], reverse=True)
    return {"p_home":ph,"p_draw":pd_,"p_away":pa,
            "score":f"{bs[0]}-{bs[1]}","xg_home":round(lh,2),
            "xg_away":round(la,2),"top_scores":scores[:5]}

saved = load_models()
all_models = saved["models"]
flags = {"Premier League":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","La Liga":"🇪🇸","Bundesliga":"🇩🇪","Serie A":"🇮🇹","Ligue 1":"🇫🇷"}
accuracy = {"Premier League":"55.1%","La Liga":"56.1%","Bundesliga":"55.3%","Serie A":"54.0%","Ligue 1":"57.2%"}

with st.sidebar:
    st.title("⚽ Football Predictor Pro")
    st.markdown("---")
    selected_league = st.selectbox("🏆 Ligue", list(all_models.keys()),
                                    format_func=lambda x: f"{flags.get(x,'')} {x}")
    st.markdown(f"**Accuracy : {accuracy.get(selected_league,'?')}**")
    st.markdown("---")
    for league, acc in accuracy.items():
        st.markdown(f"{flags.get(league,'')} {league}: **{acc}**")

model = all_models[selected_league]
df = load_data(selected_league)
teams = sorted(model["team_stats"].keys())

st.title(f"{flags.get(selected_league,'')} {selected_league} — Prédicteur")
st.caption(f"Poisson v4 · {accuracy.get(selected_league,'?')} · {len(teams)} équipes")
st.divider()

col1, col2, col3 = st.columns([2,1,2])
with col1:
    home = st.selectbox("🏠 Domicile", teams)
with col2:
    st.markdown("<br><h2 style=\'text-align:center;color:gray\'>VS</h2>", unsafe_allow_html=True)
with col3:
    away_opts = [t for t in teams if t != home]
    away = st.selectbox("✈️ Extérieur", away_opts)

st.divider()
st.subheader("⚙️ Facteurs humains")
st.caption("Ajuste les probabilités selon ce que tu sais du match que le modèle ne voit pas.")

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"### 🏠 {home}")

    home_absence_level = st.slider(
        "⭐ Joueurs absents (importance)",
        min_value=0, max_value=3, value=0,
        help="0 = aucune absence · 1 = joueur secondaire · 2 = joueur important · 3 = star absolue (Mbappe, Salah...)"
    )
    absence_labels = {0:"Effectif complet ✅", 1:"Absence mineure 🟡", 2:"Absence importante 🟠", 3:"Star absente 🔴"}
    st.caption(absence_labels[home_absence_level])

    home_gk_out = st.checkbox(f"🧤 Gardien titulaire absent ({home})")
    home_fatigue = st.selectbox(f"😴 Fatigue ({home})", 
                                 ["Aucune", "Légère (1 match)", "Forte (2+ matchs)"])
    home_motivation = st.selectbox(f"🔥 Motivation ({home})",
                                    ["Normal", "Élevée (derby/finale)", "Faible (déjà relégué)"])

with col2:
    st.markdown(f"### ✈️ {away}")

    away_absence_level = st.slider(
        "⭐ Joueurs absents (importance)",
        min_value=0, max_value=3, value=0,
        help="0 = aucune absence · 1 = joueur secondaire · 2 = joueur important · 3 = star absolue"
    )
    st.caption(absence_labels[away_absence_level])

    away_gk_out = st.checkbox(f"🧤 Gardien titulaire absent ({away})")
    away_fatigue = st.selectbox(f"😴 Fatigue ({away})",
                                 ["Aucune", "Légère (1 match)", "Forte (2+ matchs)"])
    away_motivation = st.selectbox(f"🔥 Motivation ({away})",
                                    ["Normal", "Élevée (derby/finale)", "Faible (déjà relégué)"])

weather = st.select_slider("🌤️ Météo", options=["Normale", "Pluie", "Vent fort", "Neige"])

factors = {
    "home_absence_level": home_absence_level,
    "away_absence_level": away_absence_level,
    "home_gk_out": home_gk_out,
    "away_gk_out": away_gk_out,
    "home_fatigue": home_fatigue,
    "away_fatigue": away_fatigue,
    "home_motivation": home_motivation,
    "away_motivation": away_motivation,
    "weather": weather,
}

st.divider()

if st.button("🔮 Prédire", use_container_width=True, type="primary"):
    r = predict(home, away, model, df, factors)
    if r is None:
        st.error("Données insuffisantes.")
    else:
        active_factors = []
        if home_absence_level > 0: active_factors.append(f"⭐ {absence_labels[home_absence_level]} ({home})")
        if away_absence_level > 0: active_factors.append(f"⭐ {absence_labels[away_absence_level]} ({away})")
        if home_gk_out: active_factors.append(f"🧤 Gardien absent ({home})")
        if away_gk_out: active_factors.append(f"🧤 Gardien absent ({away})")
        if home_fatigue != "Aucune": active_factors.append(f"😴 {home_fatigue} ({home})")
        if away_fatigue != "Aucune": active_factors.append(f"😴 {away_fatigue} ({away})")
        if home_motivation != "Normal": active_factors.append(f"🔥 {home_motivation} ({home})")
        if away_motivation != "Normal": active_factors.append(f"🔥 {away_motivation} ({away})")
        if weather != "Normale": active_factors.append(f"🌧️ {weather}")

        if active_factors:
            st.info("**Facteurs appliqués :**\n" + " · ".join(active_factors))

        st.divider()
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric(f"🏠 {home}", f"{r['p_home']:.1%}")
        with c2: st.metric("🤝 Nul", f"{r['p_draw']:.1%}")
        with c3: st.metric(f"✈️ {away}", f"{r['p_away']:.1%}")
        with c4: st.metric("⚽ Score prédit", r["score"])

        st.divider()
        c1,c2 = st.columns(2)
        with c1:
            st.subheader("📊 Expected Goals")
            st.progress(min(r["xg_home"]/4,1.0), text=f"{home}: {r['xg_home']} xG")
            st.progress(min(r["xg_away"]/4,1.0), text=f"{away}: {r['xg_away']} xG")
        with c2:
            st.subheader("📈 Forme récente")
            fh = model["form"][home]["form_pts"]
            fa = model["form"][away]["form_pts"]
            st.progress(min(fh/3,1.0), text=f"{home}: {fh:.2f} pts/match")
            st.progress(min(fa/3,1.0), text=f"{away}: {fa:.2f} pts/match")

        st.divider()
        st.subheader("🎯 Top 5 scores probables")
        cols = st.columns(5)
        for idx,(gh,ga,prob) in enumerate(r["top_scores"]):
            with cols[idx]: st.metric(f"{gh}-{ga}", f"{prob:.1%}")

        st.divider()
        if r["p_home"] > r["p_away"] and r["p_home"] > r["p_draw"]:
            st.success(f"**Favori : 🏠 {home}** ({r['p_home']:.1%})")
        elif r["p_away"] > r["p_home"] and r["p_away"] > r["p_draw"]:
            st.success(f"**Favori : ✈️ {away}** ({r['p_away']:.1%})")
        else:
            st.warning("**Match très serré**")

        st.caption("⚠️ Outil d\'analyse uniquement · Accuracy 54-57%")
