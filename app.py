
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# --------------------------------------------------------------------------------
# Page Configuration
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="資産運用シミュレーター Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------------
# Simulation Logic
# --------------------------------------------------------------------------------
def run_simulation(params, life_events):
    """
    Run Monte Carlo simulation for asset trajectory.
    """
    years = params['end_age'] - params['age']
    n_sim = params['n_sim']
    
    # Initialize arrays
    res_risk = np.zeros((n_sim, years + 1))
    res_safe = np.zeros((n_sim, years + 1))
    res_total = np.zeros((n_sim, years + 1))
    
    res_risk[:, 0] = params['init_risk']
    res_safe[:, 0] = params['init_safe']
    res_total[:, 0] = params['init_risk'] + params['init_safe']
    
    avg_withdraw_history = np.zeros(years + 1)
    
    # Parameters for simulation
    mu = params['mu'] / 100
    sigma = params['sigma'] / 100
    inf = params['inflation'] / 100
    m_add = params['monthly_add'] * 12
    pension = params['pension_amount']
    pension_start_age = params['pension_start_age']
    target_risk_ratio = params['risk_ratio'] / 100

    # Convert events to dictionary for quick lookup
    event_dict = {e['age']: e['amount'] for e in life_events}

    for t in range(1, years + 1):
        curr_age = params['age'] + t
        Z = np.random.standard_normal(n_sim)
        
        # 1. Growth (Geometric Brownian Motion)
        growth_rates = np.exp((mu - 0.5 * sigma**2) + sigma * Z)
        
        # Stress Test Logic (Every 10 years drop 30%)
        if params['use_stress'] and t % 10 == 0:
            growth_rates *= 0.7 
        
        current_risk = res_risk[:, t-1] * growth_rates
        current_safe = res_safe[:, t-1]
        
        # 2. Cash Flow (Contribution / Withdrawal + Pension)
        actual_w = np.zeros(n_sim)
        
        # Determine base cash flow requirement (positive = income, negative = expense)
        # Pension is always income if eligible
        annual_pension = pension if curr_age >= pension_start_age else 0
        
        if curr_age <= params['retire_age']:
            # Accumulation Phase
            net_cash_flow = m_add + annual_pension
            # Distribute to assets
            current_risk += net_cash_flow * target_risk_ratio
            current_safe += net_cash_flow * (1 - target_risk_ratio)
        else:
            # Decumulation Phase
            # Calculate Withdrawal Need
            if params['withdraw_type'] == "定額":
                # Inflation adjusted fixed amount
                base_w = params['withdraw_val'] * ((1 + inf) ** (curr_age - params['retire_age']))
            else: 
                # Percentage of total assets
                base_w = (res_total[:, t-1]) * (params['withdraw_val'] / 100)
            
            # Age-based spending cut
            if params['cut_rate'] > 0 and curr_age >= params['cut_age']:
                base_w *= (1 - (params['cut_rate'] / 100))
            
            # Pension offsets withdrawal need
            needed_from_assets = base_w - annual_pension
            
            # If pension covers withdrawal, remaining is treated as surplus (reinvested) or just zero withdrawal
            # Here we assume surplus is reinvested
            
            # Guardrail Strategy (for Fixed Amount Withdrawal only)
            if params['use_guardrail'] and params['withdraw_type'] == "定額":
                # Reduce spending if market return was poor (< -10%)
                # Only affects the withdrawal part, not pension
                mask_poor_performance = growth_rates < 0.9
                # Calculate reduced withdrawal for affected simulations
                reduced_asset_withdrawal = needed_from_assets * (1 - (params['gr_cut_ratio'] / 100))
                # Apply reduction only where needed > 0 (if pension covers all, no withdrawal anyway)
                final_asset_withdrawal = np.where(needed_from_assets > 0, needed_from_assets, needed_from_assets)
                final_asset_withdrawal[mask_poor_performance] = np.where(
                    needed_from_assets > 0, 
                    reduced_asset_withdrawal, 
                    needed_from_assets
                )[mask_poor_performance]
                
                # Actual spending from assets
                actual_w = final_asset_withdrawal
            else:
                actual_w = np.full(n_sim, needed_from_assets)

            # Apply cash flow
            # If actual_w is positive, we withdraw. If negative (pension > spending), we add.
            current_safe -= actual_w # Subtract from safe first? Or proportional?
            # Let's do proportional withdrawal/addition for simplicity to maintain ratio before rebalance
            # Actually, standard logic is often: withdraw from safe, or rebalance.
            # Here we subtract from total temp then rebalance.
            
        # 3. Events
        event_val = event_dict.get(curr_age, 0)
        
        # Combine everything
        # Current Total before rebalance
        # We handled cash flow by modifying current_risk/safe directly in accumulation, 
        # but for withdrawal we calculated `actual_w`. Let's unify.
        
        # Recalculate Total
        temp_total = current_risk + current_safe
        if curr_age > params['retire_age']:
            temp_total -= actual_w
            
        temp_total += event_val
        temp_total = np.maximum(temp_total, 0) # Assets cannot be negative
        
        # 4. Rebalance
        res_risk[:, t] = temp_total * target_risk_ratio
        res_safe[:, t] = temp_total * (1 - target_risk_ratio)
        res_total[:, t] = temp_total
        
        # Record withdrawal (only relevant for decumulation)
        if curr_age > params['retire_age']:
            # Recording true spending power (Withdrawal + Pension)
            # actual_w is amount taken from assets. 
            # Total spending = actual_w + annual_pension
            avg_withdraw_history[t] = np.mean(actual_w + annual_pension)
        else:
            avg_withdraw_history[t] = 0
            
    return res_total, res_risk, res_safe, avg_withdraw_history

# --------------------------------------------------------------------------------
# UI Components
# --------------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.header("🔧 シミュレーション設定")
    
    params = {}
    
    with st.sidebar.expander("👤 基本プロフィール", expanded=True):
        params['age'] = st.number_input("現在の年齢", 0, 100, 35)
        params['retire_age'] = st.number_input("引退/取崩し開始年齢", 0, 100, 65)
        params['end_age'] = st.number_input("シミュレーション終了年齢", 0, 120, 95)

    with st.sidebar.expander("💰 現在の資産", expanded=True):
        params['init_risk'] = st.number_input("リスク資産 (万円)", 0, 500000, 1000)
        params['init_safe'] = st.number_input("安全資産 (万円)", 0, 500000, 500)
        
        total_init = params['init_risk'] + params['init_safe']
        current_ratio = (params['init_risk'] / total_init * 100) if total_init > 0 else 0
        st.write(f"現在のリスク資産比率: **{current_ratio:.1f}%**")
        
        params['risk_ratio'] = st.slider("目標リスク資産比率 (%)", 0, 100, 70, help="リバランス時の目標比率")

    with st.sidebar.expander("📥 収入・積立"):
        params['monthly_add'] = st.number_input("毎月の積立額 (万円)", 0, 500, 5)
        st.markdown("---")
        st.caption("年金・その他収入")
        params['pension_start_age'] = st.number_input("受給開始年齢", 60, 80, 65)
        params['pension_amount'] = st.number_input("年間受給額 (万円)", 0, 1000, 200)

    with st.sidebar.expander("📤 取り崩し戦略"):
        params['withdraw_type'] = st.radio("取り崩し方法", ["定額", "定率"])
        if params['withdraw_type'] == "定額":
            default_val = 300.0
            label = "年間取り崩し額 (万円)"
        else:
            default_val = 4.0
            label = "年間取り崩し率 (%)"
        
        params['withdraw_val'] = st.number_input(label, 0.0, 5000.0, default_val)
        
        params['cut_rate'] = st.slider("加齢による支出カット率 (%)", 0, 50, 0, help="高齢になった際に支出を減らす割合")
        if params['cut_rate'] > 0:
            params['cut_age'] = st.number_input("カット開始年齢", params['retire_age'], 120, 80)
        else:
            params['cut_age'] = 80 # default

    with st.sidebar.expander("📈 市場・インフレ前提"):
        params['mu'] = st.slider("期待リターン (年率 %)", 0.0, 15.0, 5.0)
        params['sigma'] = st.slider("リスク (標準偏差 %)", 0.0, 40.0, 15.0)
        params['inflation'] = st.slider("インフレ率 (%)", -2.0, 10.0, 2.0)
        
        params['use_stress'] = st.checkbox("【ストレステスト】10年ごとに30%暴落", value=True)
        params['use_guardrail'] = st.checkbox("【ガードレール】暴落時に支出削減", value=True)
        if params['use_guardrail']:
            params['gr_cut_ratio'] = st.number_input("削減率 (%)", 0, 50, 20)
        else:
            params['gr_cut_ratio'] = 0

    params['n_sim'] = 500 # Fixed for performance, or add to advanced settings
    
    return params

def render_events():
    st.sidebar.markdown("---")
    with st.sidebar.expander("📅 ライフイベント (大きな出費)"):
        if 'events' not in st.session_state:
            st.session_state.events = []
        
        with st.form("event_form", clear_on_submit=True):
            e_age = st.number_input("年齢", 0, 120, 60)
            e_name = st.text_input("イベント名", "住宅リフォーム")
            e_amt = st.number_input("金額 (万円)", -10000, 50000, 300)
            submitted = st.form_submit_button("イベント追加")
            if submitted:
                st.session_state.events.append({"age": e_age, "name": e_name, "amount": -e_amt}) # Expense is negative
        
        if st.session_state.events:
            st.write("登録済みイベント:")
            for i, e in enumerate(st.session_state.events):
                st.text(f"{e['age']}歳: {e['name']} ({e['amount']}万円)")
            
            if st.button("イベントクリア"):
                st.session_state.events = []
                st.rerun()

    # Convert positive input for expense to negative for calculation logic if needed, 
    # but run_simulation expects signed amount. 
    # Let's standardize: User inputs positive for cost, we flip to negative. 
    # Wait, previous code used signed input. Let's keep it simple: 
    # Input box says "Amount (simulated add/subtract)", let user decide.
    # Actually, for better UX, usually "Expense" is positive input but subtracted.
    # Let's adjust: The form above does `-e_amt`.

    return st.session_state.events

# --------------------------------------------------------------------------------
# Main App
# --------------------------------------------------------------------------------
def main():
    st.title("🚀 資産運用シミュレーター Pro")
    st.markdown("長期間の資産推移をモンテカルロ・シミュレーションで可視化します。")
    
    params = render_sidebar()
    events = render_events()
    
    # Run Simulation
    if st.button("シミュレーションを実行", type="primary"):
        res_total, res_risk, res_safe, withdraw_hist = run_simulation(params, events)
        
        # Calculate stats
        ages = np.arange(params['age'], params['end_age'] + 1)
        median_total = np.median(res_total, axis=0)
        p10_total = np.percentile(res_total, 10, axis=0)
        p90_total = np.percentile(res_total, 90, axis=0)
        
        median_risk = np.median(res_risk, axis=0)
        median_safe = np.median(res_safe, axis=0)

        # -------------------
        # Metrics Area
        # -------------------
        final_median = median_total[-1]
        success_rate = (np.sum(res_total[:, -1] > 0) / params['n_sim']) * 100
        avg_monthly_spend = np.mean(withdraw_hist[withdraw_hist > 0]) / 12 if np.any(withdraw_hist > 0) else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("最終資産 (中央値)", f"{int(final_median):,} 万円", 
                  delta=f"{int(final_median - (params['init_risk'] + params['init_safe'])):,} 万円 (増減)")
        m2.metric("破綻しない確率", f"{success_rate:.1f} %", 
                  delta_color="normal" if success_rate > 80 else "inverse")
        m3.metric("老後平均月額支出", f"{int(avg_monthly_spend):,} 万円/月")
        
        # -------------------
        # Tabs for Analysis
        # -------------------
        tab1, tab2, tab3 = st.tabs(["📊 資産推移チャート", "📈 シナリオ比較", "📋 データ詳細"])
        
        with tab1:
            fig = go.Figure()
            # Range area
            fig.add_trace(go.Scatter(
                x=ages, y=p90_total, mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=ages, y=p10_total, mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(0, 100, 255, 0.1)',
                name='上位90% - 下位10% 範囲', hoverinfo='skip'
            ))
            
            # Median Line
            fig.add_trace(go.Scatter(
                x=ages, y=median_total, mode='lines', 
                line=dict(color='rgb(0, 100, 255)', width=3),
                name='資産中央値'
            ))

            # Events
            for e in events:
                fig.add_vline(x=e['age'], line_dash="dash", line_color="gray", annotation_text=e['name'])
            
            fig.update_layout(
                title="資産推移 (中央値 & 信頼区間)",
                xaxis_title="年齢",
                yaxis_title="資産額 (万円)",
                hovermode="x unified",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # Asset Allocation Chart
            st.subheader("資産構成 (中央値ベース)")
            fig_alloc = go.Figure()
            fig_alloc.add_trace(go.Scatter(
                x=ages, y=median_risk, mode='lines', stackgroup='one',
                name='リスク資産', line=dict(width=0, color='#ff9f43')
            ))
            fig_alloc.add_trace(go.Scatter(
                x=ages, y=median_safe, mode='lines', stackgroup='one',
                name='安全資産', line=dict(width=0, color='#2e86de')
            ))
            fig_alloc.update_layout(height=300, margin=dict(t=0, b=0), xaxis_title="年齢", yaxis_title="資産額")
            st.plotly_chart(fig_alloc, use_container_width=True)

        with tab2:
            st.info("現在の設定を「シナリオA」として保存し、設定を変更して再実行すると比較できます。")
            if st.button("現在の結果を保存 (シナリオA)"):
                st.session_state['scenario_a'] = {
                    'ages': ages,
                    'median': median_total,
                    'params': params.copy()
                }
                st.success("シナリオAを保存しました！設定を変更して再実行してください。")
            
            if 'scenario_a' in st.session_state:
                st.divider()
                st.subheader("シナリオ比較")
                sc_a = st.session_state['scenario_a']
                
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(
                    x=sc_a['ages'], y=sc_a['median'], 
                    name="シナリオA (保存済み)", line=dict(color='gray', dash='dash')
                ))
                fig_comp.add_trace(go.Scatter(
                    x=ages, y=median_total, 
                    name="現在のシナリオ (B)", line=dict(color='blue')
                ))
                fig_comp.update_layout(title="資産中央値の比較", hovermode="x unified")
                st.plotly_chart(fig_comp, use_container_width=True)

        with tab3:
            df_res = pd.DataFrame({
                "年齢": ages,
                "合計資産(中央値)": median_total.astype(int),
                "リスク資産": median_risk.astype(int),
                "安全資産": median_safe.astype(int),
                "年間支出(年金込)": withdraw_hist.astype(int)
            })
            st.dataframe(df_res, use_container_width=True)
            
            # CSV Download
            csv = df_res.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                "📥 結果をCSVでダウンロード",
                csv,
                "simulation_result.csv",
                "text/csv",
                key='download-csv'
            )

    else:
        st.info("👈 左側のサイドバーで設定を行い、「シミュレーションを実行」ボタンを押してください。")

if __name__ == "__main__":
    main()

