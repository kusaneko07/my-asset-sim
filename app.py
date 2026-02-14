import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ページ設定
st.set_page_config(page_title="資産運用シミュレーター", layout="wide")

def run_simulation(params):
    # パラメータ展開
    years = params['end_age'] - params['age']
    n_sim = params['n_sim']
    results = np.zeros((n_sim, years + 1))
    results[:, 0] = params['init_asset']
    avg_withdraw_history = np.zeros(years + 1)

    mu = params['mu'] / 100
    sigma = params['sigma'] / 100
    inf = params['inflation'] / 100
    m_add = params['monthly_add'] * 12
    
    for t in range(1, years + 1):
        curr_age = params['age'] + t - 1
        Z = np.random.standard_normal(n_sim)
        
        # 1. 成長 (GBM: 幾何ブラウン運動)
        # 期待リターンを対数収益率に変換して計算
        growth_rates = np.exp((mu - 0.5 * sigma**2) + sigma * Z)
        
        # 2. ストレステスト (10年ごとの暴落)
        if params['use_stress'] and t % 10 == 0:
            growth_rates *= 0.7

        prev_assets = results[:, t-1]
        
        # 3. 入出金計算
        if curr_age < params['retire_age']:
            results[:, t] = (prev_assets * growth_rates) + m_add
            avg_withdraw_history[t] = 0
        else:
            # 基本の取り崩し額 (インフレ+加齢カット考慮)
            base_w = params['withdraw_annual'] * ((1 + inf) ** (curr_age - params['age']))
            if params['cut_rate'] > 0 and curr_age >= params['cut_age']:
                base_w *= (1 - (params['cut_rate'] / 100))
            
            actual_withdrawals = np.full(n_sim, base_w)
            if params['use_guardrail']:
                # 運用成績が悪い（下落率10%以上）場合に支出をカット
                stress_mask = growth_rates < 0.9 
                actual_withdrawals[stress_mask] *= (1 - (params['gr_cut_ratio'] / 100))

            results[:, t] = (prev_assets * growth_rates) - actual_withdrawals
            avg_withdraw_history[t] = np.mean(actual_withdrawals)

        results[:, t] = np.maximum(results[:, t], 0)
    
    return results, avg_withdraw_history

# --- UI構築 ---
st.title("🚀 資産運用シミュレーター (ガードレール戦略対応)")
st.sidebar.header("📋 入力パラメータ")

# サイドバーに入力項目を配置
p = {}
p['age'] = st.sidebar.number_input("現在の年齢 (歳)", 0, 100, 35)
p['init_asset'] = st.sidebar.number_input("初期投資額 (万円)", 0, 100000, 500)
p['monthly_add'] = st.sidebar.number_input("毎月の積立額 (万円)", 0, 100, 5)
p['retire_age'] = st.sidebar.number_input("取り崩し開始年齢 (歳)", 0, 100, 65)
p['withdraw_annual'] = st.sidebar.number_input("年間取り崩し額 (万円/現在価値)", 0, 2000, 300)
p['mu'] = st.sidebar.slider("期待リターン (%)", 0.0, 15.0, 5.0)
p['sigma'] = st.sidebar.slider("リスク/ボラティリティ (%)", 0.0, 40.0, 15.0)
p['inflation'] = st.sidebar.slider("インフレ期待値 (%)", 0.0, 5.0, 2.0)
p['cut_age'] = st.sidebar.number_input("支出カット開始年齢 (歳)", 0, 100, 75)
p['cut_rate'] = st.sidebar.slider("支出カット率 (%) ※加齢による減少", 0, 50, 0)
p['n_sim'] = st.sidebar.select_slider("シミュレーション回数", options=[100, 500, 1000, 2000, 5000], value=1000)
p['end_age'] = st.sidebar.number_input("シミュレーション終了年齢", 0, 120, 95)

st.sidebar.subheader("🛡 戦略オプション")
p['use_stress'] = st.sidebar.checkbox("ストレステスト (10年毎に-30%暴落)")
p['use_guardrail'] = st.sidebar.checkbox("ガードレール戦略を発動")
p['gr_cut_ratio'] = st.sidebar.number_input("暴落時の支出削減率 (%)", 0, 100, 20)

# --- シミュレーション実行 ---
if st.sidebar.button("シミュレーションを実行"):
    results, withdraw_history = run_simulation(p)
    ages = np.arange(p['age'], p['end_age'] + 1)
    
    # グラフ作成
    fig = go.Figure()
    stats = [
        (np.max(results, axis=0), "最大値", "rgba(0, 200, 0, 0.2)", "dash"),
        (np.percentile(results, 75, axis=0), "上位25% (好調)", "rgba(0, 0, 255, 0.4)", "solid"),
        (np.percentile(results, 50, axis=0), "中央値 (標準)", "rgba(255, 0, 0, 1)", "solid"),
        (np.percentile(results, 25, axis=0), "下位25% (不調)", "rgba(100, 100, 100, 0.4)", "solid"),
        (np.percentile(results, 10, axis=0), "下位10% (危機)", "rgba(200, 0, 0, 0.5)", "solid"),
        (np.min(results, axis=0), "最小値", "rgba(0, 0, 0, 0.2)", "dash"),
    ]

    for val, name, color, dash in stats:
        fig.add_trace(go.Scatter(
            x=ages, y=val, name=name,
            line=dict(color=color, width=3 if name=="中央値 (標準)" else 1.5, dash=dash),
            customdata=withdraw_history,
            hovertemplate="<b>" + name + "</b><br>資産残高: %{y:,.0f}万円<br>平均取出額: %{customdata:,.0f}万円<extra></extra>"
        ))

    fig.update_layout(
        title="資産残高推移シミュレーション (モンテカルロ法)",
        xaxis_title="年齢",
        yaxis_title="資産残高 (万円)",
        hovermode="x unified",
        template="plotly_white",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # サマリー統計
    final_assets = results[:, -1]
    success_rate = np.sum(final_assets > 0) / p['n_sim'] * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("最終資産（中央値）", f"{int(np.median(final_assets)):,} 万円")
    col2.metric("資金枯渇回避率", f"{success_rate:.1f} %")
    col3.metric("平均年間取り崩し額", f"{int(np.mean(withdraw_history[withdraw_history>0])) if any(withdraw_history>0) else 0:,} 万円")

else:
    st.info("左側のサイドバーからパラメータを設定し、「シミュレーションを実行」ボタンを押してください。")