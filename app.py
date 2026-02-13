import streamlit as st
import pandas as pd

st.title("セットアップ完了！")
st.write("ついにStreamlitが動きました。おめでとうございます！")

# 試しにデータフレームを表示
df = pd.DataFrame({
    '名前': ['Aさん', 'Bさん', 'Cさん'],
    '点数': [80, 95, 70]
})

st.write("▼ データの表示テスト")
st.dataframe(df)

st.write("▼ グラフの表示テスト")
st.bar_chart(df.set_index('名前'))