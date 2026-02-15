import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# ページ設定
st.set_page_config(page_title="資産運用シミュレーター Pro", layout="wide")

def run_simulation(params, life_events):
    years = params['end_age'] - params['age']
    n_sim = params['n_sim']
    
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

    event_dict = {e['age']: e['amount'] for e in life_events}

    for t in range(1, years + 1):
        curr_age = params['age'] + t
        Z = np.random.standard_normal(n_sim)
        
        growth_rates = np.exp((mu - 0.5 * sigma**2) + sigma * Z)
        if params['use_stress'] and t % 10 == 0:
            growth_rates *= 0.7
        
        current_risk = res_risk[:, t-1] * growth_rates
        current_safe = res_safe[:, t-1]
        
        actual_w = np.zeros(n_sim)
        if curr_age <= params['retire_age']:
            current_risk += m_add * target_risk_ratio
            current_safe += m_add * (1 - target_risk_ratio)
        else:
            if params['withdraw_type'] == "定額":
                base_w = params['withdraw_val'] * ((1 + inf) ** (curr_age - params['retire_age']))
            else:
                base_w = (res_total[:, t-1]) * (params['withdraw_val'] / 100)
            
            if params['cut_rate'] > 0 and curr_age >= params['cut_age']:
                base_w *= (1 - (params['cut_rate'] / 100))
            
            actual_w = np.full(n_sim, base_w)
            if params['use_guardrail'] and params['withdraw_type'] == "定額":
                actual_w[growth_rates < 0.9] *= (1 - (params['gr_cut_ratio'] / 100))

        event_val = event_dict.get(curr_age, 0)
        temp_total = current_risk + current_safe - actual_w + event_val
        temp_total = np.maximum(temp_total, 0)
        
        res_risk[:, t] = temp_total * target_risk_ratio
        res_safe[:, t] = temp_total * (1 - target_risk_ratio)
        res_total[:, t] = temp_total
        avg_withdraw_history[t] = np.mean(actual_w)
    
    return res_total, res_risk, res_safe, avg_withdraw_history

# --- UI構築 ---
st.title("🚀 資産運用シミュレーター Pro (統計詳細版)")

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
            'inflation': 2.0, 'cut_age': 75, 'cut_rate': 0, 'use_stress': False, 
            'use_guardrail': True, 'gr_cut_ratio': 20, 'n_sim': 1000
        })

if st.sidebar.button("シミュレーション実行", disabled=run_disabled):
    res_total, res_risk, res_safe, withdraw_hist = run_simulation(p, st.session_state.events if 'events' in st.session_state else [])
    ages = np.arange(p['age'], p['end_age'] + 1)
    
    # 統計計算
    m_total = np.median(res_total, axis=0)
    p70 = np.percentile(res_total, 70, axis=0)  # 上位30%の下限
    p30 = np.percentile(res_total, 30, axis=0)  # 下位30%の上限
    p10 = np.percentile(res_total, 10, axis=0)  # 下位10%の上限
    
    m_risk = np.median(res_risk, axis=0)
    m_safe = np.median(res_safe, axis=0)

    # ホバーテキスト作成
    custom_hover = [
        f"<b>年齢: {a}歳</b><br>" +
        f"合計資産(中央値): {int(t):,}万円<br>" +
        f"<span style='color:green'>上位30%下限: {int(up):,}万円</span><br>" +
        f"<span style='color:orange'>下位30%上限: {int(lo):,}万円</span><br>" +
        f"<span style='color:red'>下位10%上限: {int(cr):,}万円</span><br>" +
        f"--------------------<br>" +
        f"運用資産: {int(r):,}万円<br>" +
        f"安全資産: {int(s):,}万円<br>" +
        f"取り崩し額: {int(w):,}万円<extra></extra>"
        for a, t, up, lo, cr, r, s, w in zip(ages, m_total, p70, p30, p10, m_risk, m_safe, withdraw_hist)
    ]

    fig = go.Figure()
    
    # 中央値をメイン線として描画
    fig.add_trace(go.Scatter(
        x=ages, y=m_total, name="中央値",
        line=dict(color='red', width=3),
        hovertemplate="%{customdata}",
        customdata=custom_hover
    ))

    # 統計エリア（視覚的なガイドとして上位70%〜10%を薄く表示）
    fig.add_trace(go.Scatter(
        x=ages, y=p70, name="上位30%ライン",
        line=dict(color='rgba(0,128,0,0.2)', dash='dot'),
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=p10, name="下位10%ライン",
        line=dict(color='rgba(255,0,0,0.2)', dash='dot'),
        fill='tonexty', fillcolor='rgba(100,100,100,0.1)',
        hoverinfo='skip'
    ))

    fig.update_layout(
        hovermode="x unified",
        title="資産推移と統計的リスク分布",
        yaxis_title="金額 (万円)",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 結果サマリー
    st.info(f"💡 下位10%のケース（非常に不調）でも、{p['end_age']}歳時点で **{int(p10[-1]):,}万円** が残る計算です。")


