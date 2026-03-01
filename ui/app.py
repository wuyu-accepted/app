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
    file_path = os.path.join(base_dir, "lstm_ultimate_baseline.csv") # 假设你的 collect.ipynb 最终会生成这个文件
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame()

def find_raw_csv_by_symbol(symbol):
    """智能匹配带标签的原始 CSV 文件 (如: 上游_AI芯片_300223_北京君正.csv)"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(base_dir, "stock_data_csv")
    files = glob.glob(f"{csv_dir}/*_{symbol}_*.csv") + glob.glob(f"{csv_dir}/{symbol}_*.csv")
    if files:
        return files[0]
    return None

def load_kline_data(file_path):
    """读取历史 K 线用于画图"""
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    df.sort_index(ascending=True, inplace=True)
    return df.tail(60)

# ==========================================
# 2. 真实 Agent 右脑推理与融合引擎 + 大模型分析
# ==========================================
from knowledge_graph.graph_adapter import RealGraphProvider
from agent.reasoning import FinancialAgent
from agent.dynamic_gating import FusionEngine
from agent.llm_reporter import LLMReporter
from dotenv import load_dotenv

# 加载 .env 文件中的 API Key
load_dotenv()

# 全局单例初始化缓存，防止 Streamlit 每次刷新都重头创建
@st.cache_resource
def init_agent_system():
    mock_a = RealGraphProvider()
    agent = FinancialAgent(decay_lambda=0.8)
    fusion_engine = FusionEngine(mode='math', k=2.0)
    
    try:
        reporter = LLMReporter()
    except Exception as e:
        reporter = None
        print(f"[警告] LLM 引擎初始化失败，请检查 .env 配置: {e}")
        
    return mock_a, agent, fusion_engine, reporter

mock_a, financial_agent, fusion_engine, llm_reporter = init_agent_system()

def get_real_agent_reasoning(symbol, name):
    """接入真正的多跳图谱推理引擎"""
    # 从图谱中获取当前股票的相关新闻事件
    news_dict = mock_a.get_latest_news(stock_name=name)

    # 确保新闻目标对准当前选择的股票
    news_dict['target_stock'] = name
    
    # 触发多跳推理
    financial_agent.propagate_impact(
        target_stock=news_dict['target_stock'],
        initial_power=news_dict['impact_score'],
        graph_provider=mock_a
    )
    
    # 获取该股票最终传导过来的特征分
    agent_features = financial_agent.get_feature_vectors(name)
    agent_return = agent_features[0]
    
    # 构造一条能够反映真实传导情况的原因文本
    impact_text = f"【图谱传导】引爆源：{news_dict['news_text']}。"
    
    if agent_return > 0.02:
        reason = f"{impact_text} 多跳网络最终判定对 {name} 为结构性利好。"
    elif agent_return < -0.02:
        reason = f"{impact_text} 供应链波动波及，最终判定对 {name} 为短期利空。"
    else:
        reason = f"{impact_text} 情绪衰减，未见重大突发波及，对 {name} 情绪维持中性。"
        
    return agent_features, reason

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
        base_dir = os.path.dirname(os.path.abspath(__file__))
        collect_path = os.path.join(base_dir, "collect.ipynb")
        if not os.path.exists(collect_path):
            st.error("❌ 找不到 collect.ipynb 文件，请确认它和 app.py 在同一个文件夹下！")
        else:
            with st.spinner("正在后台疯狂运转 collect.ipynb... 抓取数据与训练神经网络可能需要几分钟，请不要关闭页面！"):
                try:
                    # 核心魔法：用命令行强行无头执行 Jupyter Notebook
                    # --inplace 表示直接在原文件上运行，--execute 表示执行所有单元格
                    result = subprocess.run(
                        ["jupyter", "nbconvert", "--execute", "--inplace", collect_path],
                        cwd=base_dir,
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

agent_features, agent_reason = get_real_agent_reasoning(selected_symbol, selected_name)
agent_return = agent_features[0]

# 最终预测公式: 使用 FusionEngine 动态计算 (Math 模式)
lstm_features = [lstm_return] # Math数学门控模式下，仅需要基准分即可
fuse_result = fusion_engine.calculate_final_score(lstm_features, agent_features)

final_return = fuse_result['final_score']
trade_action = fuse_result['action']
fusion_status = fuse_result['status']

predicted_price = last_close * (1 + final_return)

st.title(f"📊 {selected_name} ({selected_symbol}) - 复合量化预测看板")

col1, col2, col3 = st.columns(3)
col1.metric("最新实际收盘价", f"¥ {last_close:.2f}")
col2.metric("LSTM 纯技术面基准", f"{lstm_return*100:+.2f}%")
col3.metric(f"融合预测明日价 ({trade_action})", f"¥ {predicted_price:.2f}", f"{final_return*100:+.2f}%")

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
        st.markdown(f"**Agent 情绪多跳推演因子:** `{agent_return*100:+.2f}%`")
        st.info(f"**引擎当前激活状态:** {fusion_status}")
        st.success(agent_reason)
        
    mermaid_code = f"""
    graph TD
        UP(上游节点) -->|传导| TARGET({selected_name})
        TARGET -->|传导| DOWN(下游节点)
        style TARGET fill:#f9f,stroke:#333,stroke-width:4px
    """
    st.markdown(f"```mermaid\n{mermaid_code}\n```")

# ==========================================
# 5. 生成专业 AI 投资研报
# ==========================================
st.divider()
st.subheader("🤖 AI 投资研报生成器")
st.caption("基于 Claude 大模型，结合量价左脑与图谱情绪右脑的综合评估结果，自动为您撰写投研报告。")

if st.button("✨ 生成最新个股研报", type="primary", use_container_width=True):
    if llm_reporter:
        with st.spinner(f"正在全网深度分析 {selected_name} 的异动传导与技术面，撰写专业报告中，请稍候..."):
            try:
                report_text = llm_reporter.get_natural_language_report(selected_symbol, fuse_result)
                st.markdown("### 📄 自动生成研报")
                with st.container(border=True):
                    st.markdown(report_text)
            except Exception as e:
                st.error(f"❌ 大模型接口调用失败或返回异常: {e}")
    else:
        st.error("⚠️ LLMReporter 组件未成功加载。请检查项目根目录下是否存在 `.env` 文件且含有正确的 `LLM_API_KEY`。")