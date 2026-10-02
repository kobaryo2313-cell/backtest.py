import streamlit as st
import pandas as pd
import re
import os
import zipfile
import unicodedata
import numpy as np
from datetime import datetime

st.set_page_config(page_title="ロジック検証エンジン (Backtest)", layout="wide")

st.title("🧪 血統×展開バイアス ロジック検証エンジン")
st.write("※指定された「学習用データ」で傾向を分析し、「テスト用データ」で回収率をシミュレーションします。血統マスタによるクロス加点も統合済みです。")

# --- サイドバーで芝・ダート切替 ---
st.sidebar.markdown("---")
track_type = st.sidebar.radio("📁 読み込むデータベースを選択", ["芝", "ダート"])

# ==========================================
# 1. データ読み込み ＆ クレンジング関数
# ==========================================
def clean_name(name):
    if pd.isna(name): return ""
    return unicodedata.normalize('NFKC', str(name)).replace(' ', '').replace(' ', '').strip()

def clean_date(d_str):
    if pd.isna(d_str): return "20000101"
    d_str = str(d_str).replace('.', '').replace('/', '').replace(' ', '')
    m = re.search(r'\d{8}', d_str)
    return m.group() if m else "20000101"

def calc_days_between(date_past, date_now):
    try:
        dp = datetime.strptime(str(date_past), "%Y%m%d")
        dn = datetime.strptime(str(date_now), "%Y%m%d")
        return (dn - dp).days
    except:
        return 999

def process_dataframe(df):
    """読み込んだレースデータの共通前処理"""
    df['計算用_着順'] = pd.to_numeric(df.get('着順', 99), errors='coerce').fillna(99)
    df['有効出走'] = df['計算用_着順'].apply(lambda x: 1 if x != 99 else 0)
    
    date_col = next((c for c in df.columns if '日付' in c), None)
    df['検索用_日付'] = df[date_col].apply(clean_date) if date_col else "20000101"
    df['検索用_日付_数値'] = pd.to_numeric(df['検索用_日付'], errors='coerce')
    
    sire_col = next((c for c in df.columns if c in ['父', '父馬', '種牡馬']), None)
    bms_col = next((c for c in df.columns if c in ['母父馬', '母父', 'BMS']), None)
    if sire_col and bms_col:
        df['検索用_父'] = df[sire_col].apply(clean_name)
        df['検索用_母父'] = df[bms_col].apply(clean_name)
        df['ニックス'] = df['検索用_父'] + " × " + df['検索用_母父']
    else:
        df['検索用_父'] = ""
        df['検索用_母父'] = ""
        df['ニックス'] = "不明"
    return df

@st.cache_data
def load_all_data(track):
    current_files = os.listdir()
    keyword = "芝" if track == "芝" else "ダート"
    
    train_zip, test_zip = None, None
    pedigree_dfs = []
    
    for f in current_files:
        norm_f = unicodedata.normalize('NFC', f)
        if norm_f.endswith('.zip'):
            # 学習・テスト用レースデータの判別
            if keyword in norm_f:
                if '学習用' in norm_f: train_zip = f
                elif 'テスト用' in norm_f: test_zip = f
            
            # 血統マスタデータの読み込み
            if any(k in norm_f for k in ['血統', 'マスタ', '種牡馬', '母父']):
                with zipfile.ZipFile(f, 'r') as z:
                    for c_file in z.namelist():
                        if c_file.endswith('.csv') and '__MACOSX' not in c_file:
                            try: temp_df = pd.read_csv(z.open(c_file), encoding="utf-8-sig")
                            except: temp_df = pd.read_csv(z.open(c_file), encoding="cp932")
                            if '馬名' in temp_df.columns:
                                temp_df['検索用馬名'] = temp_df['馬名'].apply(clean_name)
                                pedigree_dfs.append(temp_df)
                
    if not train_zip or not test_zip:
        st.error(f"【エラー】'{track}' のレースデータが見つかりません。")
        st.write(f"必要なファイル: `学習用データ_{track}.zip` および `テスト用データ_{track}.zip`")
        st.stop()
        
    def load_zip_to_df(zip_filename):
        with zipfile.ZipFile(zip_filename, 'r') as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv') and '__MACOSX' not in f]
            with z.open(csv_files[0]) as f:
                try: df = pd.read_csv(f, encoding="utf-8-sig")
                except:
                    f.seek(0)
                    df = pd.read_csv(f, encoding="cp932")
        return process_dataframe(df)

    df_train = load_zip_to_df(train_zip)
    df_test = load_zip_to_df(test_zip)
    df_pedigree = pd.concat(pedigree_dfs, ignore_index=True).drop_duplicates(subset=['検索用馬名']) if pedigree_dfs else pd.DataFrame()
    
    return df_train, df_test, df_pedigree

df_train, df_test, df_pedigree = load_all_data(track_type)

# テストデータの馬が過去にどんなレースをしたか検索できるよう、全レースデータを統合
df_master_all = pd.concat([df_train, df_test], ignore_index=True)

# ==========================================
# 2. 検証条件の設定
# ==========================================
st.sidebar.markdown("### 🔍 バックテスト条件")
place = st.sidebar.selectbox("検証する競馬場", df_test.get('場所', pd.Series(['不明'])).dropna().unique())
distances = df_test.get('距離', pd.Series(['芝1200'])).dropna().unique()
distance = st.sidebar.selectbox("検証する距離", distances)

train_cond = (df_train['場所'] == place) & (df_train['距離'] == distance)
test_cond = (df_test['場所'] == place) & (df_test['距離'] == distance)

df_train_f = df_train[train_cond].copy()
df_test_f = df_test[test_cond].copy().sort_values('検索用_日付_数値')

if len(df_test_f) < 10:
    st.warning("テスト用データのレース数が少なすぎます。")
    st.stop()

# ==========================================
# 3. バックテストエンジンの実行
# ==========================================
if st.button("🚀 バックテスト実行 (血統マスタクロス判定 統合版)", type="primary"):
    with st.spinner("血統クロスを判定し、展開スコアと合算してシミュレーション中..."):
        
        # --- 学習フェーズ ---
        nicks_stats = df_train_f.groupby('ニックス').agg(出走=('有効出走', 'sum'), 複勝=('計算用_着順', lambda x: (x<=3).sum()))
        nicks_stats['複勝率'] = nicks_stats['複勝'] / nicks_stats['出走']
        
        results = {'A: 血統のみ(ベース+クロス)': [], 'B: 血統＋過去3走展開ペナルティ': [], 'C: 血統×展開係数': []}
        
        # --- テストフェーズ ---
        df_test_f['レースキー'] = df_test_f['検索用_日付'].astype(str) + "_" + df_test_f.get('馬場状態', '').astype(str)
        
        for race_key, race_df in df_test_f.groupby('レースキー'):
            if len(race_df) < 5: continue 
            
            race_date = race_df['検索用_日付_数値'].iloc[0]
            scores_A, scores_B, scores_C = {}, {}, {}
            
            for idx, horse in race_df.iterrows():
                horse_name = horse.get('馬名', '')
                nicks = horse['ニックス']
                e_sire_name = horse.get('検索用_父', '')
                e_bms_name = horse.get('検索用_母父', '')
                
                # --- 1. 血統ベーススコア算出 ---
                n_data = nicks_stats.loc[nicks] if nicks in nicks_stats.index else None
                runs = n_data['出走'] if n_data is not None else 0
                place_rate = n_data['複勝率'] if n_data is not None else 0
                
                base_score = 0
                if runs >= 3 and place_rate >= 0.3: base_score = 30
                
                # --- 2. 5代血統マスタ連動: クロス判定による加点 ---
                crosses_found = []
                s_row = df_pedigree[df_pedigree['検索用馬名'] == e_sire_name].iloc[0] if not df_pedigree.empty and e_sire_name in df_pedigree['検索用馬名'].values else None
                b_row = df_pedigree[df_pedigree['検索用馬名'] == e_bms_name].iloc[0] if not df_pedigree.empty and e_bms_name in df_pedigree['検索用馬名'].values else None
                
                if s_row is not None and b_row is not None:
                    s_ancestors, b_ancestors = {}, {}
                    for c in [col for col in s_row.index if re.match(r'^[父母]+$', str(col))]:
                        name = clean_name(s_row[c])
                        if name and name != 'nan': s_ancestors.setdefault(name, set()).add(len(str(c)))
                    for c in [col for col in b_row.index if re.match(r'^[父母]+$', str(col))]:
                        name = clean_name(b_row[c])
                        if name and name != 'nan': b_ancestors.setdefault(name, set()).add(len(str(c)) + 1)
                            
                    for ancestor in set(s_ancestors.keys()).intersection(b_ancestors.keys()):
                        for s_gen in s_ancestors[ancestor]:
                            for b_gen in b_ancestors[ancestor]:
                                cross_pair = {s_gen, b_gen}
                                if cross_pair in [{3, 4}, {4, 3}, {4, 5}, {5, 4}, {4, 4}]:
                                    crosses_found.append(ancestor)
                
                # クロス1本につき10点の加点
                cross_points = len(set(crosses_found)) * 10
                base_score += cross_points
                
                # --- 3. 過去3走の展開動的抽出 (新馬・休養明け対応) ---
                past_races = df_master_all[(df_master_all['馬名'] == horse_name) & (df_master_all['検索用_日付_数値'] < race_date)].sort_values('検索用_日付_数値', ascending=False).head(3)
                
                score_B = base_score
                multiplier_C = 1.0 
                
                if len(past_races) == 0:
                    pass # 新馬戦はベーススコアのみで勝負
                else:
                    avg_corner = past_races['4角'].mean() if '4角' in past_races.columns else 99
                    latest_past_date = past_races['検索用_日付_数値'].iloc[0]
                    days_absent = calc_days_between(latest_past_date, race_date)
                    
                    if days_absent > 180:
                        if avg_corner >= 10: score_B -= 10
                        if avg_corner <= 4: multiplier_C = 1.25
                        elif avg_corner >= 10: multiplier_C = 0.75
                    else:
                        if avg_corner >= 10: score_B -= 20
                        if avg_corner <= 4: multiplier_C = 1.5
                        elif avg_corner >= 10: multiplier_C = 0.5
                
                scores_A[idx] = base_score
                scores_B[idx] = score_B
                scores_C[idx] = base_score * multiplier_C

            # 最もスコアが高い馬を抽出
            def get_best(scores):
                if not scores: return None
                best = max(scores, key=scores.get)
                return best if scores[best] > 0 else None

            picks = {'A: 血統のみ(ベース+クロス)': get_best(scores_A), 'B: 血統＋過去3走展開ペナルティ': get_best(scores_B), 'C: 血統×展開係数': get_best(scores_C)}
            
            for log_name, picked_idx in picks.items():
                if picked_idx is not None:
                    res = race_df.loc[picked_idx]
                    payout_win = float(str(res.get('単勝配当', 0)).replace(',','')) if res['計算用_着順'] == 1 else 0
                    payout_place = float(str(res.get('複勝配当', 0)).replace(',','')) if res['計算用_着順'] <= 3 else 0
                    
                    results[log_name].append({
                        'win': 1 if res['計算用_着順'] == 1 else 0,
                        'place': 1 if res['計算用_着順'] <= 3 else 0,
                        'win_ret': payout_win,
                        'place_ret': payout_place
                    })

        # --- 集計と画面表示 ---
        summary = []
        for name, log in results.items():
            cnt = len(log)
            if cnt > 0:
                summary.append({
                    "ロジック": name,
                    "テスト対象R数": cnt,
                    "勝率(%)": round(sum(r['win'] for r in log)/cnt*100, 1),
                    "複勝率(%)": round(sum(r['place'] for r in log)/cnt*100, 1),
                    "単回値(%)": round(sum(r['win_ret'] for r in log)/(cnt*100)*100, 1),
                    "複回値(%)": round(sum(r['place_ret'] for r in log)/(cnt*100)*100, 1)
                })
        
        st.markdown("#### 📊 検証結果（指定されたテストデータでの集計）")
        if summary:
            st.dataframe(pd.DataFrame(summary).style.highlight_max(subset=['単回値(%)', '複回値(%)'], color='lightgreen'), use_container_width=True)
            st.success("検証が完了しました。血統クロス加点が組み込まれた上で、どの展開ロジックが最も高い回収率になるかご確認ください。")
        else:
            st.warning("推奨馬を抽出できたレースがありませんでした。")