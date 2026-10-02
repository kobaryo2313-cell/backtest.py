import streamlit as st
import pandas as pd
import re
import os
import zipfile
import unicodedata
import numpy as np
from datetime import datetime
import math

st.set_page_config(page_title="ロジック検証エンジン (Backtest)", layout="wide")

st.title("🧪 血統×展開バイアス ロジック一括検証エンジン")
st.write("※ワンクリックでテストデータ内の「全競馬場・全距離」を自動巡回し、4つのロジックの総合回収率をシミュレーションします。")

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
    # 浮動小数点で読み込まれた際の ".0" を削除し、記号を取り除く
    d_str = str(d_str).split('.')[0].replace('/', '').replace('-', '').replace(' ', '')
    # 6桁（例: 260922）の場合は頭に20をつけて8桁にする
    if len(d_str) == 6 and d_str.isdigit():
        d_str = "20" + d_str
    
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
    if df.empty: return df
    
    # 修正: 着順の全角文字を半角に正規化してから数値化（数字以外はNaNになり99着にフォールバック）
    def parse_rank(x):
        if pd.isna(x): return 99
        val = unicodedata.normalize('NFKC', str(x))
        nums = re.findall(r'\d+', val)
        return int(nums[0]) if nums else 99

    df['計算用_着順'] = df.get('着順', 99).apply(parse_rank)
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
            if keyword in norm_f:
                if '学習用' in norm_f: train_zip = f
                elif 'テスト用' in norm_f: test_zip = f
            
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
        st.error(f"【エラー】'{track}' の検証用ZIPデータが見つかりません。")
        st.stop()
        
    def load_zip_to_df(zip_filename):
        dfs = []
        with zipfile.ZipFile(zip_filename, 'r') as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv') and '__MACOSX' not in f]
            for c_file in csv_files:
                with z.open(c_file) as f:
                    try: df = pd.read_csv(f, encoding="utf-8-sig")
                    except:
                        f.seek(0)
                        df = pd.read_csv(f, encoding="cp932")
                    dfs.append(df)
        if not dfs: return pd.DataFrame()
        combined_df = pd.concat(dfs, ignore_index=True)
        return process_dataframe(combined_df)

    df_train = load_zip_to_df(train_zip)
    df_test = load_zip_to_df(test_zip)
    df_pedigree = pd.concat(pedigree_dfs, ignore_index=True).drop_duplicates(subset=['検索用馬名']) if pedigree_dfs else pd.DataFrame()
    
    return df_train, df_test, df_pedigree

df_train, df_test, df_pedigree = load_all_data(track_type)

if df_test.empty:
    st.error("テスト用データが空です。")
    st.stop()

# ==========================================
# 2. 高速化のための事前キャッシュ構築
# ==========================================
with st.spinner("シミュレーション準備中（データ辞書構築）..."):
    pedigree_dict = df_pedigree.set_index('検索用馬名').to_dict('index') if not df_pedigree.empty else {}
    
    df_master_all = pd.concat([df_train, df_test], ignore_index=True)
    df_master_all = df_master_all.sort_values('検索用_日付_数値')
    corner_col = next((c for c in df_master_all.columns if '4角' in c), None)
    
    history_dict = {}
    for _, row in df_master_all.iterrows():
        name = row.get('馬名', '')
        date_val = row.get('検索用_日付_数値', 0)
        corner = row.get(corner_col, np.nan) if corner_col else np.nan
        if name and not pd.isna(corner):
            history_dict.setdefault(name, []).append({'date': date_val, 'corner': corner})

    nicks_stats = df_train.groupby(['場所', '距離', 'ニックス']).agg(出走=('有効出走', 'sum'), 複勝=('計算用_着順', lambda x: (x<=3).sum())).reset_index()
    nicks_stats['複勝率'] = nicks_stats['複勝'] / nicks_stats['出走']
    
    nicks_dict = {}
    for _, row in nicks_stats.iterrows():
        key = f"{row['場所']}_{row['距離']}_{row['ニックス']}"
        nicks_dict[key] = {'runs': row['出走'], 'place_rate': row['複勝率']}

# ==========================================
# 3. 全自動 バックテストエンジン実行
# ==========================================
st.markdown("### 🚀 全条件一括 バックテスト実行")
st.write(f"テストデータ内の全出走馬を対象に、4パターンのロジックを一気に検証します。")

if st.button("全条件 一括シミュレーションを開始する", type="primary"):
    with st.spinner("全競馬場・全距離のレースを自動巡回中..."):
        
        results = {
            'A: 減点方式 (前に行けない馬をペナルティ)': [], 
            'B: 係数方式 (展開一致でスコア1.5倍)': [], 
            'C: Z値(統計)方式 (出走数×複勝率で評価)': [],
            'D: 足切り方式 (出走数不足・後方馬は無条件カット)': []
        }
        
        # テストデータをレース単位（日付・場所・距離）で正確にグループ化
        df_test['レースキー'] = df_test['検索用_日付_数値'].astype(str) + "_" + df_test['場所'].astype(str) + "_" + df_test['距離'].astype(str)
        
        for race_key, race_df in df_test.groupby('レースキー'):
            if len(race_df) < 5: continue 
            
            race_date = race_df['検索用_日付_数値'].iloc[0]
            race_place = race_df['場所'].iloc[0]
            race_dist = race_df['距離'].iloc[0]
            
            scores_A, scores_B, scores_C, scores_D = {}, {}, {}, {}
            
            for idx, horse in race_df.iterrows():
                horse_name = horse.get('馬名', '')
                nicks = horse['ニックス']
                e_sire_name = horse.get('検索用_父', '')
                e_bms_name = horse.get('検索用_母父', '')
                
                # --- 1. 条件別(コース専用) 血統ベーススコア算出 ---
                n_key = f"{race_place}_{race_dist}_{nicks}"
                n_data = nicks_dict.get(n_key, {'runs': 0, 'place_rate': 0})
                runs = n_data['runs']
                place_rate = n_data['place_rate']
                
                base_score = 30 if runs >= 3 and place_rate >= 0.3 else 0
                
                # --- 2. Z値(統計スコア)の算出 ---
                z_base_score = place_rate * math.log10(runs + 1) * 30 if runs > 0 else 0
                
                # --- 3. 5代血統マスタ連動: クロス判定 ---
                cross_points = 0
                s_row = pedigree_dict.get(e_sire_name)
                b_row = pedigree_dict.get(e_bms_name)
                
                if s_row and b_row:
                    s_ancestors, b_ancestors = {}, {}
                    for k, v in s_row.items():
                        if re.match(r'^[父母]+$', str(k)) and str(v) != 'nan':
                            s_ancestors.setdefault(clean_name(v), set()).add(len(str(k)))
                    for k, v in b_row.items():
                        if re.match(r'^[父母]+$', str(k)) and str(v) != 'nan':
                            b_ancestors.setdefault(clean_name(v), set()).add(len(str(k)) + 1)
                            
                    crosses_found = set()
                    for ancestor in set(s_ancestors.keys()).intersection(b_ancestors.keys()):
                        for s_gen in s_ancestors[ancestor]:
                            for b_gen in b_ancestors[ancestor]:
                                if {s_gen, b_gen} in [{3,4}, {4,3}, {4,5}, {5,4}, {4,4}]:
                                    crosses_found.add(ancestor)
                    cross_points = len(crosses_found) * 10
                
                base_score += cross_points
                z_base_score += cross_points
                
                # --- 4. 過去3走の展開抽出 ---
                past = [p for p in history_dict.get(horse_name, []) if p['date'] < race_date]
                past_3 = sorted(past, key=lambda x: x['date'], reverse=True)[:3]
                
                score_A, score_B, score_C, score_D = base_score, base_score, z_base_score, base_score
                
                if len(past_3) == 0:
                    pass 
                else:
                    avg_corner = sum(p['corner'] for p in past_3) / len(past_3)
                    days_absent = calc_days_between(past_3[0]['date'], race_date)
                    
                    if days_absent > 180:
                        if avg_corner >= 10: score_A -= 10
                        if avg_corner <= 4: score_B *= 1.25
                        elif avg_corner >= 10: score_B *= 0.75
                    else:
                        if avg_corner >= 10: score_A -= 20
                        if avg_corner <= 4: score_B *= 1.5
                        elif avg_corner >= 10: score_B *= 0.5
                    
                    if runs < 5 or avg_corner >= 10:
                        score_D = 0
                
                scores_A[idx] = score_A
                scores_B[idx] = score_B
                scores_C[idx] = score_C
                scores_D[idx] = score_D

            def get_best(scores):
                if not scores: return None
                best = max(scores, key=scores.get)
                return best if scores[best] > 0 else None

            picks = {
                'A: 減点方式 (前に行けない馬をペナルティ)': get_best(scores_A), 
                'B: 係数方式 (展開一致でスコア1.5倍)': get_best(scores_B), 
                'C: Z値(統計)方式 (出走数×複勝率で評価)': get_best(scores_C),
                'D: 足切り方式 (出走数不足・後方馬は無条件カット)': get_best(scores_D)
            }
            
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
                    "対象R数": cnt,
                    "勝率(%)": round(sum(r['win'] for r in log)/cnt*100, 1),
                    "複勝率(%)": round(sum(r['place'] for r in log)/cnt*100, 1),
                    "単回値(%)": round(sum(r['win_ret'] for r in log)/(cnt*100)*100, 1),
                    "複回値(%)": round(sum(r['place_ret'] for r in log)/(cnt*100)*100, 1)
                })
        
        st.markdown("#### 📊 全条件 総合バックテスト結果")
        if summary:
            st.dataframe(pd.DataFrame(summary).style.highlight_max(subset=['単回値(%)', '複回値(%)'], color='lightgreen'), use_container_width=True)
            st.success("すべての条件（競馬場・距離）を網羅した総合結果です。最もトータルで優秀だったロジックを採用してください。")
        else:
            st.warning("推奨馬を抽出できたレースがありませんでした。")