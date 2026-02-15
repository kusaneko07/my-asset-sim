import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# ページ設定
st.set_page_config(page_title="資産運用シミュレーター Pro", layout="wide")

def run_simulation(params, life_events):
    years = params['end_age'] - params['age']
    n_sim = params['n_sim']
    
    # 資産データの初期化
    res_risk = np.zeros((n_sim, years + 1))
    res_safe = np.zeros((n_sim, years + 1))
    res_total = np.zeros((n_sim, years + 1))
    
    res_risk[:, 0] = params['init_risk']
    res_safe[:, 0] = params['init_safe']
    res_total[:, 0] = params['init_risk'] + params['init_safe']
    
    avg_withdraw_history = np.zeros(years + 1)
    mu, sigma, inf = params['mu']/100, params['sigma']/100, params['inflation']/100
    m_add = params['monthly_add'] * 12
    target_risk_ratio = params['risk_ratio'] / 100

    # イベントを辞書形式に変換 (年齢: 金額)
    event_dict = {e['age']: e['amount'] for e in life_events}

    for t in range(1, years + 1):
        curr_age = params['age'] + t
        Z = np.random.standard_normal(n_sim)
        
        # 1. 成長
        growth_rates = np.exp((mu - 0.5 * sigma**2) + sigma * Z)
        if params['use_stress'] and t % 10 == 0:
            growth_rates *= 0.7
        
        current_risk = res_risk[:, t-1] * growth_rates
        current_safe = res_safe[:, t-1]
        
        # 2. 定期積立 or 取り崩し
        actual_w = np.zeros(n_sim)
        if curr_age <= params['retire_age']:
            current_risk += m_add * target_risk_ratio
            current_safe += m_add * (1 - target_risk_ratio)
        else:
            if params['withdraw_type'] == "定額":
                base_w = params['withdraw_val'] * ((1 + inf) ** (curr_age - params['retire_age']))
            else:
                # 前年末の合計資産に対して定率
                base_w = (res_total[:, t-1]) * (params['withdraw_val'] / 100)
            
            # 加齢カット
            if params['cut_rate'] > 0 and curr_age >= params['cut_age']:
                base_w *= (1 - (params['cut_rate'] / 100))
            
            actual_w = np.full(n_sim, base_w)
            if params['use_guardrail'] and params['withdraw_type'] == "定額":
                actual_w[growth_rates < 0.9] *= (1 - (params['gr_cut_ratio'] / 100))

        # 3. ライフイベント / 追加投資
        event_val = event_dict.get(curr_age, 0)
        temp_total = current_risk + current_safe - actual_w + event_val
        temp_total = np.maximum(temp_total, 0)
        
        # 4. リバランス (比率を維持して次期へ)
        res_risk[:, t] = temp_total * target_risk_ratio
        res_safe[:, t] = temp_total * (1 - target_risk_ratio)
        res_total[:, t] = temp_total
        avg_withdraw_history[t] = np.mean(actual_w)
    
    return res_total, res_risk, res_safe, avg_withdraw_history

# --- UI構築 ---
st.title("🚀 資産運用シミュレーター Pro")

with st.sidebar:
    st.header("📋 基本設定")
    p = {
        'age': st.number_input("現在の年齢", 0, 100, 35),
        'retire_age': st.number_input("取り崩し開始年齢", 0, 100, 65),
        'end_age': st.number_input("終了年齢", 0, 120, 95),
        'init_risk': st.number_input("初期 運用資産 (万円)", 0, 100000, 700),
        'init_safe': st.number_input("初期 安全資産 (万円)", 0, 100000, 300),
        'risk_ratio': st.slider("目標運用比率 (%)", 0, 100, 70)
    }

    # バリデーション
    total_init = p['init_risk'] + p['init_safe']
    actual_ratio = (p['init_risk'] / total_init * 100) if total_init > 0 else 0
    if abs(actual_ratio - p['risk_ratio']) > 0.1:
        st.error(f"⚠️ 比率が不一致です (現在: {actual_ratio:.1f}%)")
        run_disabled = True
    else:
        run_disabled = False

    st.subheader("📈 運用・取り崩し")
    p['monthly_add'] = st.number_input("毎月の積立額 (万円)", 0, 100, 5)
    p['mu'] = st.slider("期待リターン (%)", 0.0, 15.0, 5.0)
    p['sigma'] = st.slider("リスク (%)", 0.0, 40.0, 15.0)
    p['withdraw_type'] = st.radio("取り崩し方法", ["定額", "定率"])
    p['withdraw_val'] = st.number_input("金額(万円) or 率(%)", 0.0, 2000.0, 300.0 if p['withdraw_type']=="定額" else 4.0)

    with st.expander("詳細オプション"):
        p.update({
            'inflation': st.slider("インフレ率 (%)", 0.0, 5.0, 2.0),
            'cut_age': st.number_input("支出カット開始年齢", 0, 100, 75),
            'cut_rate': st.slider("加齢カット率 (%)", 0, 50, 0),
            'use_stress': st.checkbox("10年毎に暴落"),
            'use_guardrail': st.checkbox("ガードレール(定額のみ)", value=True),
            'gr_cut_ratio': st.number_input("暴落時カット率 (%)", 0, 100, 20),
            'n_sim': st.select_slider("シミュ回数", options=[100, 500, 1000], value=500)
        })

# --- ライフイベントセクション ---
st.header("🗓 ライフイベント & 追加投資")
if 'events' not in st.session_state: st.session_state.events = []

c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
with c1: e_age = st.number_input("年齢", 0, 120, 50)
with c2: e_name = st.text_input("項目名", "退職金など")
with c3: e_amt = st.number_input("金額 (万円)", -10000, 10000, 1000)
with c4:
    if st.button("追加"):
        st.session_state.events.append({"age": e_age, "name": e_name, "amount": e_amt})

if st.session_state.events:
    df_ev = pd.DataFrame(st.session_state.events)
    st.table(df_ev)
    if st.button("リセット"):
        st.session_state.events = []; st.rerun()

# --- 実行 ---
if st.sidebar.button("シミュレーション実行", disabled=run_disabled):
    res_total, res_risk, res_safe, withdraw_hist = run_simulation(p, st.session_state.events)
    ages = np.arange(p['age'], p['end_age'] + 1)
    
    # メイン表示
    tab1, tab2 = st.tabs(["📊 資産推移", "📋 数値データ"])
    
    with tab1:
        view = st.radio("表示モード", ["合計資産の分布", "資産内訳(中央値)"], horizontal=True)
        fig = go.Figure()
        if view == "合計資産の分布":
            fig.add_trace(go.Scatter(x=ages, y=np.percentile(res_total, 50, axis=0), name="中央値", line=dict(color='red', width=3)))
            fig.add_trace(go.Scatter(x=ages, y=np.percentile(res_total, 25, axis=0), fill=None, name="下位25%", line=dict(color='rgba(100,100,100,0.3)')))
            fig.add_trace(go.Scatter(x=ages, y=np.percentile(res_total, 75, axis=0), fill='tonexty', name="上位25%", line=dict(color='rgba(100,100,100,0.3)')))
        else:
            fig.add_trace(go.Scatter(x=ages, y=np.median(res_risk, axis=0), name="運用資産", stackgroup='one', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=ages, y=np.median(res_safe, axis=0), name="安全資産", stackgroup='one', line=dict(color='lightblue')))
        
        for e in st.session_state.events:
            fig.add_vline(x=e['age'], line_dash="dash", line_color="green")
        
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # 中央値ベースの推移表
        summary_df = pd.DataFrame({
            "年齢": ages,
            "合計資産(中央値)": np.median(res_total, axis=0).astype(int),
            "運用資産(中央値)": np.median(res_risk, axis=0).astype(int),
            "安全資産(中央値)": np.median(res_safe, axis=0).astype(int),
            "平均取り崩し額": withdraw_hist.astype(int)
        })
        st.dataframe(summary_df)

    # 最終指標
    final_total = res_total[:, -1]
    col1, col2, col3 = st.columns(3)
    col1.metric("最終資産 (中央値)", f"{int(np.median(final_total)):,} 万円")
    col2.metric("資金枯渇回避率", f"{(np.sum(final_total > 0)/p['n_sim'])*100:.1f} %")
    col3.metric("平均支出額", f"{int(np.mean(withdraw_hist[withdraw_hist>0])):,} 万円")
