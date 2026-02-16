import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import gumbel_r
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

# ---------------------------------------------------------
# 1. アプリのタイトルと設定
# ---------------------------------------------------------
st.set_page_config(page_title="Climate Risk App", page_icon="🌪️")
st.title("🌪️ Climate Risk Analyzer")
st.markdown("気象データをアップロードすると、**50年に1度の災害リスク**・**推定被害額**・**ハザードマップ**を表示します。")

# ---------------------------------------------------------
# 2. サイドバー：ファイルアップロード
# ---------------------------------------------------------
st.sidebar.header("データのアップロード")
uploaded_file = st.sidebar.file_uploader("気象庁のCSVファイル（日別）をドラッグ＆ドロップ", type=["csv"])

# ---------------------------------------------------------
# 3. 解析ロジック（関数）
# ---------------------------------------------------------
def analyze_data(file):
    try:
        # 3行目をヘッダーとして読み込み
        df = pd.read_csv(file, encoding="shift_jis", header=3)
        df = df.iloc[2:] # 余計な行をカット
        df = df.iloc[:, [0, 1]] # 日付と値
        df.columns = ["Date", "Value"]
        
        # データ型変換
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df = df.dropna()
        
        # 年最大値の集計
        df["Year"] = df["Date"].dt.year
        annual_max = df.groupby("Year")["Value"].max()
        
        return df, annual_max
    except Exception as e:
        return None, None

# ---------------------------------------------------------
# 4. メイン処理
# ---------------------------------------------------------
if uploaded_file is not None:
    st.info("データを解析中...")
    
    # 解析実行
    raw_df, annual_max = analyze_data(uploaded_file)
    
    if annual_max is not None:
        # --- 基本情報の表示 ---
        col1, col2 = st.columns(2)
        with col1:
            st.metric("データ期間", f"{annual_max.index.min()} - {annual_max.index.max()}年")
        with col2:
            st.metric("データ数", f"{len(annual_max)} 年分")
            
        # --- ガンベル分布によるリスク計算 ---
        loc, scale = gumbel_r.fit(annual_max)
        
        # スライダー（再現期間）
        st.write("---")
        return_period = st.slider("再現期間（年）を選択してください", min_value=10, max_value=200, value=50)
        
        # リスク値の計算
        risk_value = gumbel_r.ppf(1 - 1/return_period, loc, scale)
        
        # 結果表示
        st.success(f"📊 {return_period}年に1度の最大リスク予測値")
  st.markdown(f"<h1 style='text-align: center; color: crimson;'>{risk_value:.2f} m/s</h1>", unsafe_allow_html=True)

