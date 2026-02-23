import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import glob
import random
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
    file_path = "lstm_ultimate_baseline.csv" # 假设你的 collect.ipynb 最终会生成这个文件
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame()

def find_raw_csv_by_symbol(symbol):
    """智能匹配带标签的原始 CSV 文件 (如: 上游_AI芯片_300223_北京君正.csv)"""
    files = glob.glob(f"stock_data_csv/*_{symbol}_*.csv") + glob.glob(f"stock_data_csv/{symbol}_*.csv")
    if files:
        return files[0]
    return None

def load_kline_data(file_path):
    """读取历史 K 线用于画图"""
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    df.sort_index(ascending=True, inplace=True)
    return df.tail(60)

# ==========================================
# 2. 模拟 Agent 右脑 (后续可替换为大模型 API)
# ==========================================
def get_mock_agent_reasoning(symbol, name):
    """模拟大模型阅读新闻后的情绪打分 (-10% 到 +10%)"""
    mock_news_score = random.uniform(-0.05, 0.05) 
    if mock_news_score > 0.02:
        reason = f"【图谱监控】检测到 {name} 所在产业链板块有积极政策落地，Agent 判定为利好。"
    elif mock_news_score < -0.02:
        reason = f"【图谱监控】检测到 {name} 的上游原材料价格波动，Agent 判定为短期利空。"
    else:
        reason = f"【图谱监控】未发现 {name} 的核心产业链节点有重大突发新闻，情绪维持中性。"
    return mock_news_score, reason

# ==========================================
# 3. 网站布局：侧边栏 (中控参数与算力引擎)
# ==========================================
df_baseline = load_lstm_baseline()

with st.sidebar:
    st.title("⚙️ 系统控制台")
    
    # ----------------------------------------
    # 🌟 算力引擎：一键执行你的 Jupyter Notebook
    # ----------------------------------------
    st.markdown("### 🚀 后台算力引擎")
    if st.button("🔌 启动 collect.ipynb 重新训练", type="primary", use_container_width=True):
        
        # 检查你的笔记本文件是否存在
        if not os.path.exists("collect.ipynb"):
            st.error("❌ 找不到 collect.ipynb 文件，请确认它和 app.py 在同一个文件夹下！")
        else:
            with st.spinner("正在后台疯狂运转 collect.ipynb... 抓取数据与训练神经网络可能需要几分钟，请不要关闭页面！"):
                try:
                    # 核心魔法：用命令行强行无头执行 Jupyter Notebook
                    # --inplace 表示直接在原文件上运行，--execute 表示执行所有单元格
                    result = subprocess.run(
                        ["jupyter", "nbconvert", "--execute", "--inplace", "collect.ipynb"],
                        capture_output=True, text=True
                    )
                    
                    if result.returncode == 0:
                        # 执行成功！清除旧网页的数据缓存
                        st.cache_data.clear()
                        st.success("✅ 数据抓取与 LSTM 训练大功告成！正在刷新面板...")
                        time.sleep(1)
                        st.rerun() # 强制网页刷新，读取最新数据
                    else:
                        st.error(f"❌ 训练报错了！错误信息：\n{result.stderr[-500:]}") # 打印最后 500 个字符的报错信息
                except Exception as e:
                    st.error(f"❌ 系统调用失败: {e}\n(请确保你的环境中安装了 jupyter)")
    
    st.divider()

    # ----------------------------------------
    # 🌟 选股与参数调节
    # ----------------------------------------
    if df_baseline.empty:
        st.warning("⚠️ 暂无基准数据。请先点击上方的按钮运行 `collect.ipynb` 生成数据！")
        st.stop()

    st.markdown("### 🎯 标的与权重调参")
    # 强制将代码转换为 6 位字符串格式 (完美修复你之前的报错！)
    df_baseline['Symbol'] = df_baseline['Symbol'].astype(str).str.zfill(6)
    df_baseline['Name'] = df_baseline['Name'].astype(str)
    
    stock_options = df_baseline['Symbol'] + " - " + df_baseline['Name']
    selected_option = st.selectbox("请选择目标标的:", stock_options)
    
    selected_symbol = selected_option.split(" - ")[0]
    selected_name = selected_option.split(" - ")[1]
    
    alpha = st.slider("左脑 (LSTM 技术面) 权重 α", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
    st.caption(f"技术面占 {alpha*100:.0f}%, Agent 逻辑面占 {(1-alpha)*100:.0f}%")

# ==========================================
# 4. 数据计算与仪表盘渲染
# ==========================================
stock_row = df_baseline[df_baseline['Symbol'] == selected_symbol].iloc[0]
last_close = stock_row['Last_Close']
lstm_return = stock_row['LSTM_Base_Return(%)'] / 100.0 

agent_return, agent_reason = get_mock_agent_reasoning(selected_symbol, selected_name)

# 最终预测公式
final_return = (alpha * lstm_return) + ((1 - alpha) * agent_return)
predicted_price = last_close * (1 + final_return)

st.title(f"📊 {selected_name} ({selected_symbol}) - 复合量化预测看板")

col1, col2, col3 = st.columns(3)
col1.metric("最新实际收盘价", f"¥ {last_close:.2f}")
col2.metric("LSTM 纯技术面基准", f"{lstm_return*100:+.2f}%")
col3.metric("融合预测明日价", f"¥ {predicted_price:.2f}", f"{final_return*100:+.2f}%")

st.divider()

left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("📈 历史 K 线与明日落点")
    raw_file = find_raw_csv_by_symbol(selected_symbol)
    if raw_file:
        df_kline = load_kline_data(raw_file)
        fig = go.Figure(data=[go.Candlestick(
            x=df_kline.index, open=df_kline['open'], high=df_kline['high'], 
            low=df_kline['low'], close=df_kline['close'], name="历史走势"
        )])
        tomorrow = df_kline.index[-1] + pd.Timedelta(days=1)
        fig.add_trace(go.Scatter(
            x=[tomorrow], y=[predicted_price], mode='markers+text', 
            marker=dict(color='red', size=14, symbol='star'),
            text=[f"预测: {predicted_price:.2f}"], textposition="top center", name="明日预测点"
        ))
        fig.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"未能找到 {selected_symbol} 的原始 K 线文件，无法绘制走势图。")

with right_col:
    st.subheader("🧠 大模型 (Agent) 决策链")
    with st.container(border=True):
        st.markdown(f"**Agent 独立情绪打分:** `{agent_return*100:+.2f}%`")
        st.success(agent_reason)
        
    mermaid_code = f"""
    graph TD
        UP(上游节点) -->|传导| TARGET({selected_name})
        TARGET -->|传导| DOWN(下游节点)
        style TARGET fill:#f9f,stroke:#333,stroke-width:4px
    """
    st.markdown(f"```mermaid\n{mermaid_code}\n```")