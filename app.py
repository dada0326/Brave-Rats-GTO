import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
import scipy.optimize as opt

# -----------------------------------------------------------------------------
# 1. 核心算力引擎 (Exact Linear Programming Solver)
# -----------------------------------------------------------------------------
class RealBraveRatsEngine:
    def __init__(self):
        self.memo = {}
        self.full_hand = [0, 1, 2, 3, 4, 5, 6, 7]

    def solve_matrix_game_exact(self, matrix):
        n, m = matrix.shape
        
        c = np.zeros(n + 1)
        c[-1] = -1
        A_ub = np.zeros((m, n + 1))
        A_ub[:, :n] = -matrix.T
        A_ub[:, n] = 1
        b_ub = np.zeros(m)
        A_eq = np.zeros((1, n + 1))
        A_eq[0, :n] = 1
        b_eq = np.array([1])
        bounds = [(0, None)] * n + [(None, None)]
        
        res1 = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res1.success:
            p1_strat = res1.x[:n]
            ev = res1.x[-1]
        else:
            p1_strat = np.ones(n) / n
            ev = 0.0
            
        c2 = np.zeros(m + 1)
        c2[-1] = 1
        A_ub2 = np.zeros((n, m + 1))
        A_ub2[:, :m] = matrix
        A_ub2[:, m] = -1
        b_ub2 = np.zeros(n)
        A_eq2 = np.zeros((1, m + 1))
        A_eq2[0, :m] = 1
        b_eq2 = np.array([1])
        bounds2 = [(0, None)] * m + [(None, None)]
        
        res2 = opt.linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2, bounds=bounds2, method='highs')
        if res2.success:
            p2_strat = res2.x[:m]
        else:
            p2_strat = np.ones(m) / m
            
        p1_strat = np.clip(p1_strat, 0, 1)
        if p1_strat.sum() > 0: p1_strat /= p1_strat.sum()
        p2_strat = np.clip(p2_strat, 0, 1)
        if p2_strat.sum() > 0: p2_strat /= p2_strat.sum()
        
        return ev, p1_strat, p2_strat

    def evaluate_matchup(self, c1, c2, p1_score, p2_score, pool, p1_mod, p2_mod, p1_hand, p2_hand):
        v1 = c1 + 2 if p1_mod == "general" else c1
        v2 = c2 + 2 if p2_mod == "general" else c2
        
        p1_active = True
        p2_active = True
        
        if c1 == 5: p2_active = False
        if c2 == 5: p1_active = False

        if c1 == 1 and p1_active and c2 == 7 and p2_active: return 1.0
        if c2 == 1 and p2_active and c1 == 7 and p1_active: return -1.0

        p1_win, p2_win, is_tie = False, False, False

        if c1 == 0 or c2 == 0:
            is_tie = True
        elif c1 == 7 and p1_active and c2 == 7 and p2_active:
            is_tie = True
        elif c1 == 7 and p1_active:
            p1_win = True
        elif c2 == 7 and p2_active:
            p2_win = True
        else:
            reverse = (c1 == 3 and p1_active) or (c2 == 3 and p2_active)
            if v1 == v2:
                is_tie = True
            elif reverse:
                if v1 < v2: p1_win = True
                else: p2_win = True
            else:
                if v1 > v2: p1_win = True
                else: p2_win = True

        next_p1_score = p1_score
        next_p2_score = p2_score
        next_pool = pool

        if is_tie:
            next_pool = pool + 1
        elif p1_win:
            points = pool * (2 if (c1 == 4 and p1_active) else 1)
            next_p1_score += points
            next_pool = 1
        elif p2_win:
            points = pool * (2 if (c2 == 4 and p2_active) else 1)
            next_p2_score += points
            next_pool = 1

        next_p1_mod = "general" if (c1 == 6 and p1_active) else ("spy" if (c1 == 2 and p1_active) else None)
        next_p2_mod = "general" if (c2 == 6 and p2_active) else ("spy" if (c2 == 2 and p2_active) else None)

        return self.get_ev([c for c in p1_hand if c != c1], [c for c in p2_hand if c != c2], next_p1_score, next_p2_score, next_pool, next_p1_mod, next_p2_mod)

    def get_ev(self, p1_hand, p2_hand, p1_score, p2_score, pool, p1_mod, p2_mod):
        if p1_score >= 4: return 1.0
        if p2_score >= 4: return -1.0
        if len(p1_hand) == 0:
            return 1.0 if p1_score > p2_score else (-1.0 if p2_score > p1_score else 0.0)

        state_key = (tuple(p1_hand), tuple(p2_hand), p1_score, p2_score, pool, p1_mod, p2_mod)
        if state_key in self.memo: return self.memo[state_key]

        sym_key = (tuple(p2_hand), tuple(p1_hand), p2_score, p1_score, pool, p2_mod, p1_mod)
        if sym_key in self.memo:
            return -self.memo[sym_key]

        n, m = len(p1_hand), len(p2_hand)
        matrix = np.zeros((n, m))
        for i, c1 in enumerate(p1_hand):
            for j, c2 in enumerate(p2_hand):
                matrix[i, j] = self.evaluate_matchup(c1, c2, p1_score, p2_score, pool, p1_mod, p2_mod, p1_hand, p2_hand)

        is_p1_spy = (p1_mod == "spy") and (p2_mod != "spy")
        is_p2_spy = (p2_mod == "spy") and (p1_mod != "spy")

        if is_p1_spy: ev = np.min(np.max(matrix, axis=0))
        elif is_p2_spy: ev = np.max(np.min(matrix, axis=1))
        else: ev, _, _ = self.solve_matrix_game_exact(matrix)

        self.memo[state_key] = ev
        return ev

DB_FILE = "braverats_gto_perfect.pkl" 

@st.cache_resource
def load_engine_smart():
    engine = RealBraveRatsEngine()
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f: engine.memo = pickle.load(f)
    else:
        engine.get_ev(engine.full_hand, engine.full_hand, 0, 0, 1, None, None)
        with open(DB_FILE, "wb") as f: pickle.dump(engine.memo, f)
    return engine

engine = load_engine_smart()

# -----------------------------------------------------------------------------
# 2. 狀態歷史模擬器 
# -----------------------------------------------------------------------------
def simulate_game_path(history):
    p1_hand, p2_hand = [0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7]
    p1_score, p2_score, pool = 0, 0, 1
    p1_mod, p2_mod = None, None
    game_over, winner = False, None

    for c1, c2 in history:
        v1 = c1 + 2 if p1_mod == "general" else c1
        v2 = c2 + 2 if p2_mod == "general" else c2
        
        p1_active = True
        p2_active = True
        
        if c1 == 5: p2_active = False
        if c2 == 5: p1_active = False

        if c1 == 1 and p1_active and c2 == 7 and p2_active: return [], [], 4, p2_score, 1, None, None, True, "Player 1 (公主狙擊成功)"
        if c2 == 1 and p2_active and c1 == 7 and p1_active: return [], [], p1_score, 4, 1, None, None, True, "Player 2 (公主狙擊成功)"

        p1_win, p2_win, is_tie = False, False, False

        if c1 == 0 or c2 == 0:
            is_tie = True
        elif c1 == 7 and p1_active and c2 == 7 and p2_active:
            is_tie = True
        elif c1 == 7 and p1_active:
            p1_win = True
        elif c2 == 7 and p2_active:
            p2_win = True
        else:
            reverse = (c1 == 3 and p1_active) or (c2 == 3 and p2_active)
            if v1 == v2:
                is_tie = True
            elif reverse:
                if v1 < v2: p1_win = True
                else: p2_win = True
            else:
                if v1 > v2: p1_win = True
                else: p2_win = True

        if is_tie:
            pool = pool + 1
        elif p1_win:
            p1_score += pool * (2 if (c1 == 4 and p1_active) else 1)
            pool = 1
        elif p2_win:
            p2_score += pool * (2 if (c2 == 4 and p2_active) else 1)
            pool = 1

        p1_mod = "general" if (c1 == 6 and p1_active) else ("spy" if (c1 == 2 and p1_active) else None)
        p2_mod = "general" if (c2 == 6 and p2_active) else ("spy" if (c2 == 2 and p2_active) else None)

        p1_hand = [c for c in p1_hand if c != c1]
        p2_hand = [c for c in p2_hand if c != c2]

        if p1_score >= 4: return p1_hand, p2_hand, p1_score, p2_score, pool, p1_mod, p2_mod, True, "Player 1"
        if p2_score >= 4: return p1_hand, p2_hand, p1_score, p2_score, pool, p1_mod, p2_mod, True, "Player 2"
        if len(p1_hand) == 0:
            win_str = "Player 1" if p1_score > p2_score else ("Player 2" if p2_score > p1_score else "平手")
            return p1_hand, p2_hand, p1_score, p2_score, pool, p1_mod, p2_mod, True, win_str

    return p1_hand, p2_hand, p1_score, p2_score, pool, p1_mod, p2_mod, game_over, winner

# -----------------------------------------------------------------------------
# 3. Streamlit 介面渲染
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Brave Rats GTO ")

if "history" not in st.session_state:
    st.session_state.history = []

p1_hand, p2_hand, p1_score, p2_score, pool, p1_mod, p2_mod, game_over, winner = simulate_game_path(st.session_state.history)

st.title(" 🐀 Brave Rats GTO ")

if st.session_state.history:
    path_str = " ➔ ".join([f"R{i+1}: [P1({c1}) vs P2({c2})]" for i, (c1, c2) in enumerate(st.session_state.history)])
    st.markdown(f"**📍 歷史路徑：** `開局` ➔ `{path_str}`")
else:
    st.markdown("**📍 歷史路徑：** `🏁 開局 (Opening Node)`")

card_slots = st.columns(5)
card_slots[0].metric("Player 1 分數", p1_score)
card_slots[1].metric("Player 2 分數", p2_score)
card_slots[2].metric("當前回合勝場價值", pool)
card_slots[3].metric("P1 效果狀態", str(p1_mod or "無").upper())
card_slots[4].metric("P2 效果狀態", str(p2_mod or "無").upper())

st.markdown("---")

if not game_over:
    n, m = len(p1_hand), len(p2_hand)
    matrix = np.zeros((n, m))
    for i, c1 in enumerate(p1_hand):
        for j, c2 in enumerate(p2_hand):
            matrix[i, j] = engine.evaluate_matchup(c1, c2, p1_score, p2_score, pool, p1_mod, p2_mod, p1_hand, p2_hand)

    is_p1_spy = (p1_mod == "spy") and (p2_mod != "spy")
    is_p2_spy = (p2_mod == "spy") and (p1_mod != "spy")

    if is_p1_spy:
        ev = np.min(np.max(matrix, axis=0))
        p1_responses = np.argmax(matrix, axis=0)
        p2_evs = np.max(matrix, axis=0)
        best_j = np.argmin(p2_evs)
        p2_strat = np.zeros(m); p2_strat[best_j] = 1.0
        p1_strat = np.zeros(n); p1_strat[p1_responses[best_j]] = 1.0
    elif is_p2_spy:
        ev = np.max(np.min(matrix, axis=1))
        p2_responses = np.argmin(matrix, axis=1)
        p1_evs = np.min(matrix, axis=1)
        best_i = np.argmax(p1_evs)
        p1_strat = np.zeros(n); p1_strat[best_i] = 1.0
        p2_strat = np.zeros(m); p2_strat[p2_responses[best_i]] = 1.0
    else:
        ev, p1_strat, p2_strat = engine.solve_matrix_game_exact(matrix)

    p1_equity = (ev + 1) / 2 * 100

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label="📊 節點 P1 期望值 (EV Map)", value=f"{ev:.4f}")
        st.metric(label="🏆 P1 當前勝率 (Equity)", value=f"{p1_equity:.2f} %")
        
        # 提示框已在此處移除

        fig_p1 = go.Figure(go.Bar(x=[f"C_{c}" for c in p1_hand], y=p1_strat * 100, marker_color='rgb(55, 83, 109)'))
        fig_p1.update_layout(title="Player 1 GTO 出牌頻率 (%)", yaxis=dict(range=[0, 101]), height=220, margin=dict(l=20,r=20,t=35,b=20))
        st.plotly_chart(fig_p1, use_container_width=True)

        fig_p2 = go.Figure(go.Bar(x=[f"C_{c}" for c in p2_hand], y=p2_strat * 100, marker_color='rgb(219, 64, 82)'))
        fig_p2.update_layout(title="Player 2 GTO 出牌頻率 (%)", yaxis=dict(range=[0, 101]), height=220, margin=dict(l=20,r=20,t=35,b=20))
        st.plotly_chart(fig_p2, use_container_width=True)

    with col2:
        fig_matrix = px.imshow(
            matrix, labels=dict(x="Player 2 Card", y="Player 1 Card", color="Node EV"),
            x=[f"C_{c}" for c in p2_hand], y=[f"C_{c}" for c in p1_hand],
            color_continuous_scale="RdBu_r", color_continuous_midpoint=0.0
        )
        fig_matrix.update_traces(text=np.round(matrix, 2), texttemplate="%{text}")
        fig_matrix.update_layout(height=520, margin=dict(l=40, r=40, t=20, b=20))
        st.plotly_chart(fig_matrix, use_container_width=True)

    st.markdown("### 🛠️ 動作樹節點推進")
    act_col1, act_col2 = st.columns(2)
    with act_col1:
        p1_select = st.selectbox("選擇 Player 1 打出的卡片", ["-- 請選擇 --"] + p1_hand)
    with act_col2:
        p2_select = st.selectbox("選擇 Player 2 打出的卡片", ["-- 請選擇 --"] + p2_hand)

    if p1_select != "-- 請選擇 --" and p2_select != "-- 請選擇 --":
        if st.button("⚔️ 執行此動作組合 (Advance Node)", type="primary"):
            st.session_state.history.append((int(p1_select), int(p2_select)))
            st.rerun()
else:
    st.success(f"🏆 動作樹終點！對局已結束。最終獲勝者：{winner}")

st.sidebar.header("🧭 樹狀路徑導航")
if st.sidebar.button("⏪ 撤銷上一步 (Undo Move)", disabled=len(st.session_state.history) == 0):
    st.session_state.history.pop()
    st.rerun()

if st.sidebar.button("🔄 重置回開局狀態 (Reset to Open)"):
    st.session_state.history = []
    st.rerun()
