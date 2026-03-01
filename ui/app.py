import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import glob
import random
import time
import subprocess  # 引入子进程模块，用于调用你的 Jupyter Notebook

# ==========================================
# 0. 网页全局配置
# ==========================================
st.set_page_config(page_title="AI Agent 量化投研中控台", layout="wide", page_icon="📈")

# ==========================================
# 1. 核心数据加载引擎 (带缓存机制)
# ==========================================
@st.cache_data
def load_lstm_baseline():
    """读取 LSTM 跑出来的基准预测结果"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "lstm_ultimate_baseline.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame()

@st.cache_data
def load_historical_predictions():
    """读取每日保存的最终历史预测记录"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "daily_final_predictions.csv")
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if 'Symbol' in df.columns:
                df['Symbol'] = df['Symbol'].astype(str).str.split('.').str[0].str.strip().str.zfill(6)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            return df
        except Exception as e:
            print(f"解析历史预测记录失败: {e}")
    return pd.DataFrame()

@st.cache_data
def load_offline_agent_data():
    """读取预先抓取好的新闻文件和预先算好的Agent评分CSV"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ---------------- 极度强壮的新闻源读取 ----------------
    df_news = pd.DataFrame()
    news_files = glob.glob(os.path.join(base_dir, "*news*.csv")) + glob.glob(os.path.join(base_dir, "*news*.xlsx"))
    
    if news_files:
        news_file = news_files[0]
        try:
            if news_file.endswith('.csv'):
                try:
                    df_news = pd.read_csv(news_file, encoding='utf-8')
                except UnicodeDecodeError:
                    df_news = pd.read_csv(news_file, encoding='gbk')
            else:
                df_news = pd.read_excel(news_file)
            
            if 'stock_code' in df_news.columns:
                df_news['stock_code'] = df_news['stock_code'].astype(str).str.split('.').str[0].str.strip().str.zfill(6)
            
            if 'time' in df_news.columns:
                df_news['time'] = pd.to_datetime(df_news['time'], errors='coerce')
                latest_time = df_news['time'].max() 
                if pd.notna(latest_time):
                    three_months_ago = latest_time - pd.DateOffset(months=3)
                    df_news = df_news[df_news['time'] >= three_months_ago]
                df_news = df_news.sort_values(by='time', ascending=False)
        except Exception as e:
            st.error(f"解析新闻文件 {os.path.basename(news_file)} 失败: {e}")

    # ---------------- 极度强壮的评分 CSV 读取 ----------------
    score_file = os.path.join(base_dir, "historical_agent_scores.csv")
    df_scores = pd.DataFrame()
    if os.path.exists(score_file):
        try:
            df_scores = pd.read_csv(score_file)
            if 'stock_code' in df_scores.columns:
                df_scores['stock_code'] = df_scores['stock_code'].astype(str).str.split('.').str[0].str.strip().str.zfill(6)
            if 'date' in df_scores.columns:
                df_scores['date'] = pd.to_datetime(df_scores['date'], errors='coerce')
                df_scores = df_scores.sort_values(by='date', ascending=False)
        except Exception as e:
            st.error(f"解析 CSV 评分失败: {e}")
            
    return df_news, df_scores

def find_raw_csv_by_symbol(symbol):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(base_dir, "stock_data_csv")
    files = glob.glob(f"{csv_dir}/*_{symbol}_*.csv") + glob.glob(f"{csv_dir}/{symbol}_*.csv")
    if files:
        return files[0]
    return None

def load_kline_data(file_path):
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    df.sort_index(ascending=True, inplace=True)
    return df.tail(60)

# ==========================================
# 2. 真实 Agent 右脑推理与融合引擎 + 大模型分析
# ==========================================
from knowledge_graph.graph_adapter import RealGraphProvider
from agent.reasoning import FinancialAgent
from agent.llm_reporter import LLMReporter
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource
def init_agent_system():
    mock_a = RealGraphProvider()
    agent = FinancialAgent(decay_lambda=0.8)
    try:
        reporter = LLMReporter()
    except Exception as e:
        reporter = None
    return mock_a, agent, reporter

mock_a, financial_agent, llm_reporter = init_agent_system()

def get_real_agent_reasoning(symbol, name, df_news, df_scores):
    symbol_clean = str(symbol).strip().zfill(6)
    
    # ---------------- Step 1: 提取专门用于前端展示的新闻列表 ----------------
    news_items_for_display = []
    
    if not df_news.empty and 'stock_code' in df_news.columns:
        stock_news = df_news[df_news['stock_code'] == symbol_clean]
        if not stock_news.empty:
            for _, row in stock_news.head(3).iterrows(): # 最多取 3 条展示
                title = row.get('title', '无标题')
                time_val = row.get('time')
                time_str = time_val.strftime('%Y-%m-%d %H:%M') if pd.notna(time_val) else "未知时间"
                source = row.get('source', '网络资讯')
                news_items_for_display.append({"time": time_str, "title": title, "source": source})
                
    # ---------------- Step 2: 提取 Agent 评分 ----------------
    cached_score = None
    if not df_scores.empty and 'stock_code' in df_scores.columns:
        stock_scores = df_scores[df_scores['stock_code'] == symbol_clean]
        if not stock_scores.empty:
            cached_score = float(stock_scores.iloc[0]['total_score'])
            
    if cached_score is not None:
        agent_return = cached_score * 0.10
        agent_features = [agent_return]
        raw_agent_score = cached_score
        calc_mode = "CSV 预存结果"
    else:
        financial_agent.propagate_impact(target_stock=name, initial_power=0.0, graph_provider=mock_a)
        agent_features = financial_agent.get_feature_vectors(name)
        agent_return = agent_features[0]
        raw_agent_score = agent_return * 10.0 
        calc_mode = "现场多跳图谱推演"

    # ---------------- Step 3: 生成总结文案 ----------------
    if raw_agent_score > 0.2:
        reason = f"系统基于**{calc_mode}**判定：近期图谱传导对 {name} 形成结构性利好。"
    elif raw_agent_score < -0.2:
        reason = f"系统基于**{calc_mode}**判定：近期图谱传导对 {name} 构成短期利空。"
    else:
        reason = f"系统基于**{calc_mode}**判定：无重大异动，情绪维持中性。"
        
    return agent_features, reason, raw_agent_score, news_items_for_display

# ==========================================
# 3. 网站布局：侧边栏 (中控参数与算力引擎)
# ==========================================
df_baseline = load_lstm_baseline()
df_offline_news, df_offline_scores = load_offline_agent_data()
df_history_preds = load_historical_predictions()

with st.sidebar:
    st.title("⚙️ 系统控制台")
    st.markdown("### 🚀 后台算力引擎")
    if st.button("🔌 启动 collect.ipynb 重新训练", type="primary", use_container_width=True):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        collect_path = os.path.join(base_dir, "collect.ipynb")
        if not os.path.exists(collect_path):
            st.error("❌ 找不到 collect.ipynb 文件")
        else:
            with st.spinner("正在后台疯狂运转 collect.ipynb..."):
                try:
                    subprocess.run(["jupyter", "nbconvert", "--execute", "--inplace", collect_path], cwd=base_dir)
                    st.cache_data.clear()
                    st.success("✅ 训练大功告成！正在刷新面板...")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 系统调用失败: {e}")
    
    st.divider()

    if df_baseline.empty:
        st.warning("⚠️ 暂无基准数据。请先运行 collect.ipynb！")
        st.stop()

    st.markdown("### 🎯 标的与权重调参")
    df_baseline['Symbol'] = df_baseline['Symbol'].astype(str).str.zfill(6)
    df_baseline['Name'] = df_baseline['Name'].astype(str)
    
    stock_options = df_baseline['Symbol'] + " - " + df_baseline['Name']
    selected_option = st.selectbox("请选择目标标的:", stock_options)
    selected_symbol = selected_option.split(" - ")[0]
    selected_name = selected_option.split(" - ")[1]
    
    alpha = st.slider("左脑 (LSTM 技术面) 权重 α", min_value=0.0, max_value=1.0, value=0.6, step=0.1)

# ==========================================
# 4. 数据计算与仪表盘渲染
# ==========================================
stock_row = df_baseline[df_baseline['Symbol'] == selected_symbol].iloc[0]
last_close = stock_row['Last_Close']
lstm_return = stock_row['LSTM_Base_Return(%)'] / 100.0 

# 获取推理结果
agent_features, agent_reason, raw_agent_score, display_news_list = get_real_agent_reasoning(
    selected_symbol, selected_name, df_offline_news, df_offline_scores
)
agent_return = agent_features[0]

# 💡 核心修改点 1：将最终预测结果缩小 10 倍以贴近真实波动率
final_return = ((alpha * lstm_return) + ((1 - alpha) * agent_return)) / 10.0

trade_action = "买入/持有" if final_return > 0 else "卖出/观望"
fusion_status = f"✅ 人工滑块干预融合 (LSTM: {alpha*100:.0f}%, Agent: {(1-alpha)*100:.0f}%)"

fuse_result = {'final_score': final_return, 'action': trade_action, 'status': fusion_status}
predicted_price = last_close * (1 + final_return)

st.title(f"📊 {selected_name} ({selected_symbol}) - 复合量化预测看板")

col1, col2, col3, col4 = st.columns(4)
col1.metric("最新实际收盘价", f"¥ {last_close:.2f}")
col2.metric("LSTM 技术面预测", f"{lstm_return*100:+.2f}%")
agent_score_str = f"{raw_agent_score:.4f}" if raw_agent_score is not None else "未知"
col3.metric("Agent 图谱情绪评分", agent_score_str)
col4.metric(f"融合预测明日价 ({trade_action})", f"¥ {predicted_price:.2f}", f"{final_return*100:+.2f}%")

st.divider()

left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("📈 历史 K 线与明日落点")
    raw_file = find_raw_csv_by_symbol(selected_symbol)
    if raw_file:
        df_kline = load_kline_data(raw_file)
        fig = go.Figure(data=[go.Candlestick(x=df_kline.index, open=df_kline['open'], high=df_kline['high'], low=df_kline['low'], close=df_kline['close'], name="历史走势")])
        tomorrow = df_kline.index[-1] + pd.Timedelta(days=1)
        fig.add_trace(go.Scatter(x=[tomorrow], y=[predicted_price], mode='markers+text', marker=dict(color='red', size=14, symbol='star'), text=[f"预测: {predicted_price:.2f}"], textposition="top center", name="明日预测点"))
        fig.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"未能找到 {selected_symbol} 的原始 K 线文件，无法绘制走势图。")

with right_col:
    st.subheader("🧠 大模型 (Agent) 决策链")
    with st.container(border=True):
        st.markdown(f"**Agent 提取情绪评分:** `{agent_score_str}`")
        st.info(f"**引擎当前激活状态:** {fusion_status}")
        st.success(agent_reason)
        
        st.markdown("---")
        st.markdown("📰 **近 3 个月核心驱动资讯 (来自数据源)**")
        if display_news_list:
            for item in display_news_list:
                st.markdown(f"- **{item['time']}** | {item['title']} _({item['source']})_")
        else:
            st.warning(f"⚠️ 未找到近 3 个月新闻记录。")
        
    mermaid_code = f"""
    graph TD
        UP(产业链新闻/事件) -->|评分 {agent_score_str}| TARGET({selected_name})
        TARGET --> DOWN(最终资产定价)
        style TARGET fill:#f9f,stroke:#333,stroke-width:4px
    """
    st.markdown(f"```mermaid\n{mermaid_code}\n```")

# ==========================================
# 5. 往期预测历史记录 (新增模块)
# ==========================================
st.divider()
st.subheader("🕰️ 往期预测记录与回测走势")

if not df_history_preds.empty:
    stock_preds = df_history_preds[df_history_preds['Symbol'] == selected_symbol].copy()
    if not stock_preds.empty:
        # 💡 核心修改点 2：将历史记录表中的最终预测分数也统一缩小 10 倍
        if 'Final_Score' in stock_preds.columns:
            stock_preds['Final_Score'] = stock_preds['Final_Score'] / 10.0
            
        latest_date = stock_preds['Date'].max()
        three_months_ago = latest_date - pd.DateOffset(months=3)
        stock_preds = stock_preds[stock_preds['Date'] >= three_months_ago]
        stock_preds = stock_preds.sort_values(by='Date', ascending=False)
        
        with st.expander("📊 点击展开查看该标的近 3 个月融合预测历史 (CSV 数据)"):
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("##### 📅 预测记录明细表")
                display_df = stock_preds[['Date', 'LSTM_Pred_Return', 'Agent_Score', 'Final_Score', 'Action']].copy()
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                display_df['LSTM基准'] = (display_df['LSTM_Pred_Return'] * 100).map("{:+.2f}%".format)
                display_df['图谱评分'] = display_df['Agent_Score'].map("{:.4f}".format)
                display_df['最终预测'] = (display_df['Final_Score'] * 100).map("{:+.2f}%".format)
                st.dataframe(display_df[['Date', 'LSTM基准', '图谱评分', '最终预测', 'Action']], use_container_width=True, hide_index=True)
            
            with c2:
                st.markdown("##### 📈 预测分数波动趋势")
                plot_df = stock_preds.sort_values(by='Date', ascending=True)
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Final_Score'], mode='lines+markers', name='最终综合预测(Final)', line=dict(color='#E74C3C', width=2)))
                fig_hist.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['LSTM_Pred_Return'], mode='lines', name='纯技术面基准(LSTM)', line=dict(color='#3498DB', dash='dash')))
                fig_hist.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=300, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info(f"暂无 {selected_name} 的历史预测记录。")
else:
    st.info("未在目录下找到 `daily_final_predictions.csv` 历史回测文件。")

# ==========================================
# 6. 生成专业 AI 投资研报
# ==========================================
st.divider()
st.subheader("🤖 AI 投资研报生成器")
if st.button("✨ 生成最新个股研报", type="primary", use_container_width=True):
    if llm_reporter:
        with st.spinner(f"正在全网深度分析 {selected_name}..."):
            try:
                report_text = llm_reporter.get_natural_language_report(selected_symbol, fuse_result)
                st.markdown("### 📄 自动生成研报")
                with st.container(border=True):
                    st.markdown(report_text)
            except Exception as e:
                st.error(f"❌ 大模型调用失败: {e}")