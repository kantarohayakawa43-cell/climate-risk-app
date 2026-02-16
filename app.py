import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import gumbel_r
import matplotlib.pyplot as plt
import folium                                     # 追加
from streamlit_folium import st_folium            # 追加
# ---------------------------------------------------------
# 1. アプリのタイトルと設定
# ---------------------------------------------------------
st.set_page_config(page_title="Climate Risk App", page_icon="🌪️")
st.title("🌪️ Climate Risk Analyzer")
st.markdown("気象データをアップロードすると、**50年に1度の災害リスク**と**推定被害額**を自動計算します。")

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
        
        # =========================================================
        # ★ここが追加機能：損害額シミュレーション★
        # =========================================================
        st.write("---")
        st.subheader("💰 推定被害額シミュレーション")
        st.caption("※風速20m/sを超えると被害が急増するモデル（べき乗則）を使用")

        # 資産価値の入力欄（デフォルト10億円）
        asset_value = st.number_input("保有資産価値を入力してください (単位: 億円)", value=10, step=1)
        
        # 損害関数の計算ロジック
        if risk_value > 20:
            # 20m/sを超えた分について、被害率が3乗で増えると仮定
            damage_ratio = ((risk_value - 20) / 50) ** 3
            # 被害率は最大100%（1.0）で止める
            if damage_ratio > 1.0:
                damage_ratio = 1.0
            
            loss_amount = asset_value * damage_ratio
            
            # 結果表示（赤文字で警告）
            st.error(f"⚠️ 推定被害額: {loss_amount:.2f} 億円")
            st.write(f"(推定損害率: {damage_ratio*100:.1f}%)")
            
            # 損害レベルの可視化バー
            st.progress(damage_ratio)
        else:
            st.success("✅ この風速では、大きな構造的被害は想定されません（損害額 0円）")
        
        # =========================================================
        
        # --- グラフ描画 ---
        st.write("---")
        st.subheader("📈 リスクカーブ（再現期間曲線）")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # モデル線
        T_axis = np.logspace(0.1, 2.5, 100)
        wind_axis = gumbel_r.ppf(1 - 1/T_axis, loc, scale)
        ax.plot(T_axis, wind_axis, color="blue", label="Risk Model")
        
        # 実データプロット
        sorted_data = np.sort(annual_max)
        prob_obs = np.arange(1, len(sorted_data) + 1) / (len(sorted_data) + 1)
        T_obs = 1 / (1 - prob_obs)
        ax.scatter(T_obs, sorted_data, color="black", alpha=0.6, label="Observation")
        
        # ターゲットライン
        ax.axhline(y=risk_value, color="red", linestyle="--")
        ax.axvline(x=return_period, color="red", linestyle="--")
        
        ax.set_xscale("log")
        ax.set_xlabel("Return Period (Years)")
        ax.set_ylabel("Value (m/s)")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend()
        
        st.pyplot(fig)
        
    else:
        st.error("データの読み込みに失敗しました。CSVの中身を確認してください。")
else:
    st.info("👈 左のサイドバーから CSVファイルをアップロードしてください")
    # --- (既存のコードの続き) ---
        
        # =========================================================
        # ★追加機能：リスクマップの表示★
        # =========================================================
        st.write("---")
        st.subheader("🗺️ リスク・マッピング")
        st.caption("対象地点のリスクを地図上で可視化します")

        # ユーザーに緯度経度を入力させる（デフォルトは東京駅周辺）
        col_lat, col_lon = st.columns(2)
        with col_lat:
            input_lat = st.number_input("緯度 (Latitude)", value=35.6895, format="%.4f")
        with col_lon:
            input_lon = st.number_input("経度 (Longitude)", value=139.6917, format="%.4f")

        # 地図の作成
        m = folium.Map(location=[input_lat, input_lon], zoom_start=10, tiles="CartoDB positron")

        # 円の色判定
        if risk_value >= 25:
            color = "crimson"
            fill_color = "red"
        elif risk_value >= 20:
            color = "orange"
            fill_color = "orange"
        else:
            color = "blue"
            fill_color = "cyan"

        # 円を描画
        folium.CircleMarker(
            location=[input_lat, input_lon],
            radius=risk_value * 2.0,  # 風速に応じて大きく
            color=color,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.6,
            popup=f"Risk: {risk_value:.2f} m/s"
        ).add_to(m)

        # Streamlitで地図を表示
        st_folium(m, width=700, height=500)
