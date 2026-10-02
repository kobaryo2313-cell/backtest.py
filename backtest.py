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
st.write("※ワンクリックでテストデータ内の「全競馬場・全距離」を自動巡回し、6つのロジックの総合回収率をシミュレーションします。")

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
    d_str = str(d_str).split('.')[0].replace('/', '').replace('-', '').replace(' ', '')
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
    
    def parse_rank(x):
        if pd.isna(x): return 99
        val = unicodedata.normalize('NFKC', str(x))
        nums = re.findall(r'\d+', val)
        return int(nums[0]) if nums else 99

    df['計算用_着順'] = df.get('着順', 99).apply(parse_rank)
    df['有効出走'] = df['計算用_着順'].apply(lambda x: 1 if x != 99 else 0)
    
    # PCIと上がり3Fを数値化
    df['計算用_PCI'] = pd.to_numeric(df.get('PCI', np.nan), errors='coerce')
    df['計算用_上り3F'] = pd.to_numeric(df.get('上り3F', np.nan), errors='coerce')
    
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
    
    # --- 新規: コース別（場所×距離）のPCI・上がり3Fの平均と標準偏差を事前計算 ---
    course_stats = df_train.groupby(['場所', '距離'])[['計算用_PCI', '計算用_上り3F']].agg(['mean', 'std'])
    course_stats.columns = ['pci_mean', 'pci_std', 'f3_mean', 'f3_std']
    course_dict = course_stats.to_dict('index')
    
    history_dict = {}
    for _, row in df_master_all.iterrows():
        name = row.get('馬名', '')
        date_val = row.get('検索用_日付_数値', 0)
        corner = row.get(corner_col, np.nan) if corner_col else np.nan
        pci = row.get('計算用_PCI', np.nan)
        f3 = row.get('計算用_上り3F', np.nan)
        place = row.get('場所', '')
        dist = row.get('距離', '')
        
        if name and not pd.isna(corner):
            history_dict.setdefault(name, []).append({
                'date': date_val, 'corner': corner, 'pci': pci, 'f3': f3, 'place': place, 'dist': dist
            })

    nicks_stats = df_train.groupby(['場所', '距離', 'ニックス']).agg(出走=('有効出走', 'sum'), 複勝=('計算用_着順', lambda x: (x<=3).sum())).reset_index()
    nicks_stats['複勝率'] = nicks_stats['複勝'] / nicks_stats['出走']
    nicks_dict = {f"{r['場所']}_{r['距離']}_{r['ニックス']}": {'runs': r['出走'], 'place_rate': r['複勝率']} for _, r in nicks_stats.iterrows()}

# ==========================================
# 3. 全自動 バックテストエンジン実行
# ==========================================
st.markdown("### 🚀 PCI・上がり3F 反映版 バックテスト実行")

if st.button("全条件 一括シミュレーションを開始する", type="primary"):
    with st.spinner("全競馬場・全距離のレースを自動巡回中..."):
        
        logic_names = [
            '1_ベース: 展開係数',
            '2_候補1: 瞬発力特化 (PCI51超×3F)',
            '3_候補2: 基礎能力加算 (絶対値)',
            '4_候補3: 厳格フィルター (足切り)',
            '5_候補4: メンバー内偏差値',
            '6_推奨: 過去コース乖離Z値'
        ]
        
        results = {name: [] for name in logic_names}
        detailed_logs = []
        
        df_test['レースキー'] = df_test['検索用_日付_数値'].astype(str) + "_" + df_test['場所'].astype(str) + "_" + df_test['距離'].astype(str)
        
        for race_key, race_df in df_test.groupby('レースキー'):
            if len(race_df) < 5: continue 
            
            race_date = race_df['検索用_日付_数値'].iloc[0]
            race_place = race_df['場所'].iloc[0]
            race_dist = race_df['距離'].iloc[0]
            
            # メンバー内偏差値計算用（ロジック5）
            member_stats = []
            
            # 各ロジックのスコア格納用
            scores = {name: {} for name in logic_names}
            horse_histories = {}
            
            # 事前ループ: 各馬の基本スコアと過去履歴の取得
            for idx, horse in race_df.iterrows():
                horse_name = horse.get('馬名', '')
                nicks = horse['ニックス']
                n_key = f"{race_place}_{race_dist}_{nicks}"
                n_data = nicks_dict.get(n_key, {'runs': 0, 'place_rate': 0})
                runs, place_rate = n_data['runs'], n_data['place_rate']
                
                base_score = 30 if runs >= 3 and place_rate >= 0.3 else 0
                
                past = [p for p in history_dict.get(horse_name, []) if p['date'] < race_date]
                past_3 = sorted(past, key=lambda x: x['date'], reverse=True)[:3]
                
                avg_pci, avg_f3 = np.nan, np.nan
                if past_3:
                    valid_pci = [p['pci'] for p in past_3 if not pd.isna(p['pci'])]
                    valid_f3 = [p['f3'] for p in past_3 if not pd.isna(p['f3'])]
                    avg_pci = sum(valid_pci)/len(valid_pci) if valid_pci else np.nan
                    avg_f3 = sum(valid_f3)/len(valid_f3) if valid_f3 else np.nan
                
                horse_histories[idx] = {
                    'name': horse_name, 'base_score': base_score, 'past_3': past_3, 
                    'avg_pci': avg_pci, 'avg_f3': avg_f3, 'runs': runs
                }
                
                if not pd.isna(avg_pci) and not pd.isna(avg_f3):
                    member_stats.append({'idx': idx, 'pci': avg_pci, 'f3': avg_f3})
                    
            # ロジック5用のメンバー内平均・標準偏差
            mem_pci_mean = np.mean([m['pci'] for m in member_stats]) if member_stats else 0
            mem_pci_std = np.std([m['pci'] for m in member_stats]) if member_stats else 1
            mem_f3_mean = np.mean([m['f3'] for m in member_stats]) if member_stats else 0
            mem_f3_std = np.std([m['f3'] for m in member_stats]) if member_stats else 1
            
            # スコア計算ループ
            for idx, data in horse_histories.items():
                bs = data['base_score']
                past_3 = data['past_3']
                avg_pci = data['avg_pci']
                avg_f3 = data['avg_f3']
                runs = data['runs']
                
                s_1, s_2, s_3, s_4, s_5, s_6 = bs, bs, bs, bs, bs, bs
                
                if len(past_3) > 0:
                    avg_corner = sum(p['corner'] for p in past_3) / len(past_3)
                    
                    # ロジック1 (ベース)
                    if avg_corner <= 4: s_1 *= 1.5
                    elif avg_corner >= 10: s_1 *= 0.5
                    
                    # ロジック2 (瞬発力)
                    shunpatsu_count = sum(1 for p in past_3 if p['pci'] >= 51 and p['f3'] <= 34.9)
                    s_2 = bs * (1.5 ** shunpatsu_count)
                    
                    # ロジック3 (基礎能力加算)
                    if not pd.isna(avg_pci) and not pd.isna(avg_f3):
                        pci_pts = max(0, (avg_pci - 45)) * 2 
                        f3_pts = max(0, (36.0 - avg_f3)) * 10
                        s_3 = bs + pci_pts + f3_pts
                    
                    # ロジック4 (厳格フィルター)
                    if not pd.isna(avg_pci) and not pd.isna(avg_f3):
                        if avg_pci < 45 or avg_f3 >= 36.0 or avg_corner >= 10:
                            s_4 = 0
                    else:
                        s_4 = 0
                        
                    # ロジック5 (メンバー内偏差値)
                    if not pd.isna(avg_pci) and not pd.isna(avg_f3) and mem_pci_std > 0 and mem_f3_std > 0:
                        z_pci = (avg_pci - mem_pci_mean) / mem_pci_std * 10 + 50
                        z_f3 = (mem_f3_mean - avg_f3) / mem_f3_std * 10 + 50 # タイムは低い方が良い
                        s_5 = bs + (z_pci + z_f3)
                        
                    # ロジック6 (コース乖離Z値 - プロ推奨)
                    dev_scores = []
                    for p in past_3:
                        c_stats = course_dict.get((p['place'], p['dist']))
                        if c_stats and not pd.isna(p['pci']) and not pd.isna(p['f3']):
                            c_pci_std = c_stats['pci_std'] if c_stats['pci_std'] > 0 else 1
                            c_f3_std = c_stats['f3_std'] if c_stats['f3_std'] > 0 else 1
                            
                            z_pci_c = (p['pci'] - c_stats['pci_mean']) / c_pci_std * 10 + 50
                            z_f3_c = (c_stats['f3_mean'] - p['f3']) / c_f3_std * 10 + 50
                            dev_scores.append(z_pci_c + z_f3_c)
                    
                    if dev_scores:
                        s_6 = bs + (sum(dev_scores) / len(dev_scores))
                    else:
                        s_6 = 0 # データ不足は推奨外
                        
                else:
                    s_1 = s_2 = s_3 = s_4 = s_5 = s_6 = 0 # 過去走がない馬は一旦0
                
                scores[logic_names[0]][idx] = s_1
                scores[logic_names[1]][idx] = s_2
                scores[logic_names[2]][idx] = s_3
                scores[logic_names[3]][idx] = s_4
                scores[logic_names[4]][idx] = s_5
                scores[logic_names[5]][idx] = s_6

            def get_best(s_dict):
                if not s_dict: return None
                best = max(s_dict, key=s_dict.get)
                return best if s_dict[best] > 0 else None

            picks = {name: get_best(scores[name]) for name in logic_names}
            
            # --- ログ記録 ---
            race_log = {'日付': race_date, '場所': race_place, '距離': race_dist}
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
                    
                    race_log[f"{log_name}_推奨馬"] = res.get('馬名', '')
                    race_log[f"{log_name}_着順"] = res['計算用_着順']
                    race_log[f"{log_name}_単勝"] = payout_win
                    race_log[f"{log_name}_複勝"] = payout_place
                else:
                    race_log[f"{log_name}_推奨馬"] = "見送り"
                    race_log[f"{log_name}_着順"] = "-"
                    race_log[f"{log_name}_単勝"] = 0
                    race_log[f"{log_name}_複勝"] = 0
                    
            detailed_logs.append(race_log)

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
        
        st.markdown("#### 📊 総合バックテスト結果")
        if summary:
            st.dataframe(pd.DataFrame(summary).style.highlight_max(subset=['単回値(%)', '複回値(%)'], color='lightgreen'), use_container_width=True)
            
            st.markdown("#### 🔍 全レース詳細結果（横並び比較用）")
            df_details = pd.DataFrame(detailed_logs)
            st.dataframe(df_details, use_container_width=True)
            
            csv = df_details.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 全レース比較データをCSVでダウンロード",
                data=csv,
                file_name="backtest_details_all_logics.csv",
                mime="text/csv",
            )
        else:
            st.warning("推奨馬を抽出できたレースがありませんでした。")