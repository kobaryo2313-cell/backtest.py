import streamlit as st
import pandas as pd
import re
import os
import zipfile
import unicodedata
import numpy as np
from datetime import datetime

st.set_page_config(page_title="ロジック検証エンジン (Backtest Matrix)", layout="wide")

st.title("🧪 血統(4) × 展開(6) ＝ 24パターン マトリックス検証 ＋ 馬場適性補正")
st.write("芝とダートの特性分析に加え、**「クッション値・含水率」の適性**をスコアに反映し、より高精度な24パターンの組み合わせ検証を実行します。")

st.sidebar.markdown("---")
track_type = st.sidebar.radio("📁 読み込むデータベースを選択", ["芝", "ダート"])
st.sidebar.info("💡 芝はクッション値、ダートは含水率を中心に、当日の馬場状態と過去の好走馬場を比較して適性補正を行います。")

# ==========================================
# 1. データ読み込み ＆ クレンジング
# ==========================================
def clean_name(name):
    if pd.isna(name): return ""
    return unicodedata.normalize('NFKC', str(name)).replace(' ', '').replace(' ', '').strip()

def clean_date(d_str):
    if pd.isna(d_str): return "20000101"
    d_str = str(d_str).split('.')[0].replace('/', '').replace('-', '').replace(' ', '')
    if len(d_str) == 6 and d_str.isdigit(): d_str = "20" + d_str
    m = re.search(r'\d{8}', d_str)
    return m.group() if m else "20000101"

def process_dataframe(df, df_baba=None):
    if df.empty: return df
    
    def parse_rank(x):
        if pd.isna(x): return 99
        val = unicodedata.normalize('NFKC', str(x))
        nums = re.findall(r'\d+', val)
        return int(nums[0]) if nums else 99

    df['計算用_着順'] = df.get('着順', 99).apply(parse_rank)
    df['有効出走'] = df['計算用_着順'].apply(lambda x: 1 if x != 99 else 0)
    
    df['単勝配当_数値'] = pd.to_numeric(df.get('単勝配当', '0').astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    df['複勝配当_数値'] = pd.to_numeric(df.get('複勝配当', '0').astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    df['計算用_PCI'] = pd.to_numeric(df.get('PCI', np.nan), errors='coerce')
    df['計算用_上り3F'] = pd.to_numeric(df.get('上り3F', np.nan), errors='coerce')
    
    date_col = next((c for c in df.columns if '日付' in c), None)
    df['検索用_日付'] = df[date_col].apply(clean_date) if date_col else "20000101"
    df['検索用_日付_数値'] = pd.to_numeric(df['検索用_日付'], errors='coerce')
    
    # 血統情報の整備
    sire_col = next((c for c in df.columns if c in ['父', '父馬', '種牡馬']), None)
    bms_col = next((c for c in df.columns if c in ['母父馬', '母父', 'BMS']), None)
    if sire_col and bms_col:
        df['検索用_父'] = df[sire_col].apply(clean_name)
        df['検索用_母父'] = df[bms_col].apply(clean_name)
        df['ニックス'] = df['検索用_父'] + " × " + df['検索用_母父']
    else:
        df['検索用_父'], df['検索用_母父'], df['ニックス'] = "", "", "不明"

    # --- 🌟 馬場データの結合処理 ---
    if df_baba is not None and not df_baba.empty:
        # 結合キーを揃える
        df_baba['検索用_日付'] = pd.to_datetime(df_baba['日付']).dt.strftime('%Y%m%d')
        # 数値型に変換
        for col in ['クッション値', '芝_含水率_ゴール前', 'ダート_含水率_ゴール前']:
            if col in df_baba.columns:
                df_baba[col] = pd.to_numeric(df_baba[col], errors='coerce')
        
        # マージ（左結合）
        df = pd.merge(
            df, 
            df_baba[['検索用_日付', '場所', 'クッション値', '芝_含水率_ゴール前', 'ダート_含水率_ゴール前']], 
            on=['検索用_日付', '場所'], 
            how='left'
        )
    else:
        # 馬場データが無い場合は空列を作成
        df['クッション値'] = np.nan
        df['芝_含水率_ゴール前'] = np.nan
        df['ダート_含水率_ゴール前'] = np.nan

    return df

@st.cache_data
def load_all_data(track):
    current_files = os.listdir()
    keyword = "芝" if track == "芝" else "ダート"
    train_zip, test_zip = None, None
    pedigree_dfs = []
    
    # 🌟 馬場情報CSVを探す
    baba_file = next((f for f in current_files if f == 'baba_info_complete.csv'), None)
    df_baba = pd.read_csv(baba_file) if baba_file else pd.DataFrame()
    if df_baba.empty:
        st.warning("⚠️ 同一フォルダに `baba_info_complete.csv` が見つからないため、馬場適性補正はスキップされます。")
    
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
        return process_dataframe(pd.concat(dfs, ignore_index=True), df_baba)

    df_train = load_zip_to_df(train_zip)
    df_test = load_zip_to_df(test_zip)
    df_pedigree = pd.concat(pedigree_dfs, ignore_index=True).drop_duplicates(subset=['検索用馬名']) if pedigree_dfs else pd.DataFrame()
    return df_train, df_test, df_pedigree

df_train, df_test, df_pedigree = load_all_data(track_type)

# ==========================================
# 2. 高速化データ辞書構築 (24パターン用)
# ==========================================
with st.spinner("シミュレーション準備中（馬場適性・血統・展開辞書構築）..."):
    pedigree_dict = df_pedigree.set_index('検索用馬名').to_dict('index') if not df_pedigree.empty else {}
    
    df_master_all = pd.concat([df_train, df_test], ignore_index=True).sort_values('検索用_日付_数値')
    corner_col = next((c for c in df_master_all.columns if '4角' in c), None)
    
    # 🌟 過去履歴辞書に馬場状態を記憶させる
    history_dict = {}
    for _, row in df_master_all.iterrows():
        name = row.get('馬名', '')
        date_val = row.get('検索用_日付_数値', 0)
        rank = row.get('計算用_着順', 99)
        if name:
            history_dict.setdefault(name, []).append({
                'date': date_val, 'rank': rank, 'corner': row.get(corner_col, np.nan), 
                'pci': row.get('計算用_PCI', np.nan), 'f3': row.get('計算用_上り3F', np.nan),
                'place': row.get('場所', ''), 'dist': row.get('距離', ''),
                'cushion': row.get('クッション値', np.nan),
                'moisture_turf': row.get('芝_含水率_ゴール前', np.nan),
                'moisture_dirt': row.get('ダート_含水率_ゴール前', np.nan)
            })

    # 血統用 A, B: ニックスの回収率・複勝率
    df_train['単回値_獲得'] = df_train.apply(lambda x: x['単勝配当_数値'] if x['計算用_着順'] == 1 else 0, axis=1)
    df_train['複回値_獲得'] = df_train.apply(lambda x: x['複勝配当_数値'] if x['計算用_着順'] <= 3 else 0, axis=1)
    
    nicks_stats = df_train.groupby(['場所', '距離', 'ニックス']).agg(
        出走=('有効出走', 'sum'),
        複勝=('計算用_着順', lambda x: (x<=3).sum()),
        単勝回収=('単回値_獲得', 'sum'),
        複勝回収=('複回値_獲得', 'sum')
    ).reset_index()
    
    nicks_stats['複勝率'] = nicks_stats['複勝'] / nicks_stats['出走'].replace(0, 1)
    nicks_stats['単回値'] = nicks_stats['単勝回収'] / (nicks_stats['出走'] * 100).replace(0, 1) * 100
    nicks_stats['複回値'] = nicks_stats['複勝回収'] / (nicks_stats['出走'] * 100).replace(0, 1) * 100
    
    course_nicks_mean = nicks_stats.groupby(['場所', '距離'])['複勝率'].agg(['mean', 'std']).reset_index()
    nicks_stats = pd.merge(nicks_stats, course_nicks_mean, on=['場所', '距離'], suffixes=('', '_c'))
    
    nicks_dict = {}
    for _, r in nicks_stats.iterrows():
        nicks_dict[f"{r['場所']}_{r['距離']}_{r['ニックス']}"] = {
            'runs': r['出走'], 'place_rate': r['複勝率'], 'tan_ret': r['単回値'], 'fuku_ret': r['複回値'],
            'z_score': (r['複勝率'] - r['mean']) / (r['std'] if r['std'] > 0 else 1) * 10 + 50
        }

    # 血統C: 種牡馬の全体的なPCI/上り3F平均ポテンシャル
    sire_stats = df_train.groupby('検索用_父')[['計算用_PCI', '計算用_上り3F']].mean()
    sire_dict = sire_stats.to_dict('index')

    # 展開ロジック用: コース平均
    course_stats = df_train.groupby(['場所', '距離'])[['計算用_PCI', '計算用_上り3F']].agg(['mean', 'std'])
    course_stats.columns = ['pci_mean', 'pci_std', 'f3_mean', 'f3_std']
    course_dict = course_stats.to_dict('index')

# ==========================================
# 3. 24パターン + 絶対来ない3頭 バックテスト実行
# ==========================================
st.markdown("### 🚀 マトリックス・シミュレーション実行")

blood_logics = ['A:回収率重視', 'B:適性Z値', 'C:展開ポテンシャル', 'D:配合洗練']
pace_logics = ['1:ベース(馬場適性加味)', '2:瞬発力(PCI×3F)', '3:基礎能力加算', '4:厳格フィルター', '5:メンバー内偏差値', '6:過去コース乖離Z値']

if st.button("24パターン＋ワースト3頭 一括シミュレーションを開始", type="primary"):
    with st.spinner("馬場適性を加味して全競馬場・全距離を解析中..."):
        
        results = {f"血統{b}_展開{p}": [] for b in blood_logics for p in pace_logics}
        results["🚨消し馬:絶対に来ないワースト3頭"] = []
        detailed_logs = []
        
        df_test['レースキー'] = df_test['検索用_日付_数値'].astype(str) + "_" + df_test['場所'].astype(str) + "_" + df_test['距離'].astype(str)
        
        for race_key, race_df in df_test.groupby('レースキー'):
            if len(race_df) < 8: continue 
            
            race_date = race_df['検索用_日付_数値'].iloc[0]
            race_place = race_df['場所'].iloc[0]
            race_dist = race_df['距離'].iloc[0]
            
            # 🌟 当日の馬場状態を取得
            current_cushion = race_df['クッション値'].iloc[0]
            current_turf_m = race_df['芝_含水率_ゴール前'].iloc[0]
            current_dirt_m = race_df['ダート_含水率_ゴール前'].iloc[0]
            
            horse_scores = {idx: {} for idx in race_df.index}
            member_stats = []
            
            for idx, horse in race_df.iterrows():
                h_name = horse.get('馬名', '')
                nicks = horse['ニックス']
                e_sire = horse.get('検索用_父', '')
                e_bms = horse.get('検索用_母父', '')
                
                # --- 過去成績 ---
                past = [p for p in history_dict.get(h_name, []) if p['date'] < race_date]
                past_3 = sorted(past, key=lambda x: x['date'], reverse=True)[:3]
                past_good = [p for p in past if p['rank'] <= 3] # 🌟 過去好走（3着以内）したレースを抽出
                
                valid_pci = [p['pci'] for p in past_3 if not pd.isna(p['pci'])]
                valid_f3 = [p['f3'] for p in past_3 if not pd.isna(p['f3'])]
                avg_pci = sum(valid_pci)/len(valid_pci) if valid_pci else np.nan
                avg_f3 = sum(valid_f3)/len(valid_f3) if valid_f3 else np.nan
                avg_corner = sum(p['corner'] for p in past_3 if not pd.isna(p['corner'])) / len(past_3) if past_3 else np.nan
                
                if not pd.isna(avg_pci) and not pd.isna(avg_f3):
                    member_stats.append({'idx': idx, 'pci': avg_pci, 'f3': avg_f3})
                
                # --- 🌟 馬場適性スコアの算出 ---
                baba_bonus = 1.0
                if track_type == "芝" and not pd.isna(current_cushion) and past_good:
                    # 過去に好走したクッション値の平均を算出
                    good_cushions = [p['cushion'] for p in past_good if not pd.isna(p['cushion'])]
                    if good_cushions:
                        avg_good_cushion = sum(good_cushions) / len(good_cushions)
                        diff = abs(current_cushion - avg_good_cushion)
                        if diff <= 0.3: baba_bonus = 1.3  # 当日の馬場と得意な馬場が合致（大加点）
                        elif diff >= 1.0: baba_bonus = 0.7 # 得意な馬場と全く違う（減点）
                
                elif track_type == "ダート" and not pd.isna(current_dirt_m) and past_good:
                    # ダートは含水率で判定
                    good_moisture = [p['moisture_dirt'] for p in past_good if not pd.isna(p['moisture_dirt'])]
                    if good_moisture:
                        avg_good_m = sum(good_moisture) / len(good_moisture)
                        diff = abs(current_dirt_m - avg_good_m)
                        if diff <= 1.5: baba_bonus = 1.3
                        elif diff >= 4.0: baba_bonus = 0.7

                # --- 血統ベーススコア(4種) ---
                b_scores = {}
                n_data = nicks_dict.get(f"{race_place}_{race_dist}_{nicks}", {'runs': 0, 'z_score': 50, 'tan_ret': 0, 'fuku_ret': 0})
                runs = n_data['runs']
                
                b_scores['A:回収率重視'] = (n_data['tan_ret'] + n_data['fuku_ret']) / 2 if runs >= 3 else 10
                b_scores['B:適性Z値'] = n_data['z_score'] if runs >= 3 else 30
                
                s_data = sire_dict.get(e_sire, {'計算用_PCI': 45, '計算用_上り3F': 36.0})
                b_scores['C:展開ポテンシャル'] = max(0, s_data['計算用_PCI'] - 40) + max(0, 36.5 - s_data['計算用_上り3F']) * 10
                
                d_score = 30
                s_row = pedigree_dict.get(e_sire)
                b_row = pedigree_dict.get(e_bms)
                if s_row and b_row:
                    s_anc, b_anc = {}, {}
                    for k, v in s_row.items():
                        if re.match(r'^[父母]+$', str(k)) and str(v) != 'nan': s_anc.setdefault(clean_name(v), set()).add(len(str(k)))
                    for k, v in b_row.items():
                        if re.match(r'^[父母]+$', str(k)) and str(v) != 'nan': b_anc.setdefault(clean_name(v), set()).add(len(str(k)) + 1)
                    
                    for ancestor in set(s_anc.keys()).intersection(b_anc.keys()):
                        for s_gen in s_anc[ancestor]:
                            for b_gen in b_anc[ancestor]:
                                cross = {s_gen, b_gen}
                                if cross in [{3,4}, {4,3}, {4,4}, {4,5}, {5,4}]: d_score += 15 
                                elif cross in [{2,3}, {3,2}, {3,3}]: d_score -= 20 
                b_scores['D:配合洗練'] = max(0, d_score)
                
                # 展開スコアと合成 (6種)
                for b_key, base_s in b_scores.items():
                    # 🌟 血統ベーススコア自体に馬場適性ボーナスを掛ける
                    base_s = base_s * baba_bonus

                    for p_key in pace_logics:
                        combo_key = f"血統{b_key}_展開{p_key}"
                        final_score = base_s
                        
                        if len(past_3) > 0 and not pd.isna(avg_corner):
                            if p_key == '1:ベース(馬場適性加味)':
                                if avg_corner <= 4: final_score *= 1.5
                                elif avg_corner >= 10: final_score *= 0.5
                                
                            elif p_key == '2:瞬発力(PCI×3F)':
                                sh_cnt = sum(1 for p in past_3 if p['pci'] >= 51 and p['f3'] <= 34.9)
                                final_score = base_s * (1.5 ** sh_cnt)
                                
                            elif p_key == '3:基礎能力加算':
                                if not pd.isna(avg_pci) and not pd.isna(avg_f3):
                                    final_score = base_s + max(0, (avg_pci - 45))*2 + max(0, (36.0 - avg_f3))*10
                                    
                            elif p_key == '4:厳格フィルター':
                                if not pd.isna(avg_pci) and not pd.isna(avg_f3):
                                    if avg_pci < 45 or avg_f3 >= 36.0 or avg_corner >= 10: final_score = 0
                                else: final_score = 0
                                
                            elif p_key in ['5:メンバー内偏差値', '6:過去コース乖離Z値']:
                                final_score = base_s 
                                
                        else:
                            final_score = 0 
                            
                        horse_scores[idx][combo_key] = final_score

                # 消し馬スコア（馬場不適性で減点）
                if len(past_3) >= 3 and not pd.isna(avg_pci) and not pd.isna(avg_f3):
                    worst_score = avg_pci + (36.0 - avg_f3)*5 - avg_corner*2 + b_scores['A:回収率重視']*0.1
                    # 🌟 馬場が合わない馬はさらにワースト評価を下げる（消しやすくする）
                    worst_score = worst_score * (1.0 / baba_bonus)
                    horse_scores[idx]['worst_eval'] = worst_score
                else:
                    horse_scores[idx]['worst_eval'] = 999 

            # 展開5と6の事後計算
            mem_pci_mean = np.mean([m['pci'] for m in member_stats]) if member_stats else 0
            mem_pci_std = np.std([m['pci'] for m in member_stats]) if member_stats else 1
            mem_f3_mean = np.mean([m['f3'] for m in member_stats]) if member_stats else 0
            mem_f3_std = np.std([m['f3'] for m in member_stats]) if member_stats else 1

            for idx, scores_dict in horse_scores.items():
                horse = race_df.loc[idx]
                past = [p for p in history_dict.get(horse.get('馬名', ''), []) if p['date'] < race_date][:3]
                
                valid_pci = [p['pci'] for p in past if not pd.isna(p['pci'])]
                valid_f3 = [p['f3'] for p in past if not pd.isna(p['f3'])]
                avg_pci = sum(valid_pci)/len(valid_pci) if valid_pci else np.nan
                avg_f3 = sum(valid_f3)/len(valid_f3) if valid_f3 else np.nan

                for b_key in blood_logics:
                    # baba_bonusを加味したベーススコアを取得（展開1〜4で計算済みの値）
                    base_s = horse_scores[idx].get(f"血統{b_key}_展開1:ベース(馬場適性加味)", 30)
                    
                    if not pd.isna(avg_pci) and not pd.isna(avg_f3) and mem_pci_std > 0 and mem_f3_std > 0:
                        z_pci = (avg_pci - mem_pci_mean) / mem_pci_std * 10 + 50
                        z_f3 = (mem_f3_mean - avg_f3) / mem_f3_std * 10 + 50
                        horse_scores[idx][f"血統{b_key}_展開5:メンバー内偏差値"] = base_s + z_pci + z_f3
                    else:
                        horse_scores[idx][f"血統{b_key}_展開5:メンバー内偏差値"] = 0

                    dev_scores = []
                    for p in past:
                        c_stats = course_dict.get((p['place'], p['dist']))
                        if c_stats and not pd.isna(p['pci']) and not pd.isna(p['f3']):
                            c_pci_std, c_f3_std = max(c_stats['pci_std'], 1), max(c_stats['f3_std'], 1)
                            dev_scores.append(((p['pci'] - c_stats['pci_mean']) / c_pci_std * 10 + 50) + ((c_stats['f3_mean'] - p['f3']) / c_f3_std * 10 + 50))
                    
                    horse_scores[idx][f"血統{b_key}_展開6:過去コース乖離Z値"] = base_s + (sum(dev_scores)/len(dev_scores)) if dev_scores else 0

            # 各ロジックの1位を抽出
            picks = {}
            for combo_key in results.keys():
                if combo_key != "🚨消し馬:絶対に来ないワースト3頭":
                    best_idx = max(horse_scores.keys(), key=lambda i: horse_scores[i][combo_key])
                    picks[combo_key] = [best_idx] if horse_scores[best_idx][combo_key] > 0 else []

            # ワースト3頭を抽出 (worst_evalが低い順)
            worst_sorted = sorted([idx for idx in horse_scores if horse_scores[idx]['worst_eval'] != 999], key=lambda i: horse_scores[i]['worst_eval'])
            picks["🚨消し馬:絶対に来ないワースト3頭"] = worst_sorted[:3]

            # 結果格納
            race_log = {'日付': race_date, '場所': race_place, '距離': race_dist}
            
            for logic_name, picked_indices in picks.items():
                for picked_idx in picked_indices:
                    res = race_df.loc[picked_idx]
                    p_win = res['単勝配当_数値'] if res['計算用_着順'] == 1 else 0
                    p_place = res['複勝配当_数値'] if res['計算用_着順'] <= 3 else 0
                    
                    results[logic_name].append({
                        'win': 1 if res['計算用_着順'] == 1 else 0,
                        'place': 1 if res['計算用_着順'] <= 3 else 0,
                        'win_ret': p_win, 'place_ret': p_place
                    })
                
                # 詳細ログ用
                if logic_name != "🚨消し馬:絶対に来ないワースト3頭" and picked_indices:
                    res = race_df.loc[picked_indices[0]]
                    race_log[logic_name] = f"{res.get('馬名', '')}({int(res['計算用_着順'])}着)"
                elif logic_name != "🚨消し馬:絶対に来ないワースト3頭":
                    race_log[logic_name] = "-"
                    
            detailed_logs.append(race_log)

        # --- 集計と画面表示 ---
        summary = []
        for name, log in results.items():
            cnt = len(log)
            if cnt > 0:
                summary.append({
                    "解析ロジック": name,
                    "対象R数/頭数": cnt,
                    "勝率(%)": round(sum(r['win'] for r in log)/cnt*100, 1),
                    "複勝率(%)": round(sum(r['place'] for r in log)/cnt*100, 1),
                    "単回値(%)": round(sum(r['win_ret'] for r in log)/(cnt*100)*100, 1),
                    "複回値(%)": round(sum(r['place_ret'] for r in log)/(cnt*100)*100, 1)
                })
        
        st.markdown("#### 📊 24パターン＋消し馬 総合結果")
        if summary:
            df_summary = pd.DataFrame(summary)
            def highlight_worst(s):
                if '消し馬' in str(s['解析ロジック']): return ['background-color: #ffcccc'] * len(s)
                return [''] * len(s)
            
            st.dataframe(df_summary.style.apply(highlight_worst, axis=1).highlight_max(subset=['単回値(%)', '複回値(%)'], color='lightgreen'), use_container_width=True)
            st.info("🚨 『消し馬』の回収率が低ければ低いほど（0%に近いほど）、ロジックが『負ける馬』を正確に見抜けている証明になります。")
            
            st.markdown("#### 🔍 全レース推奨馬一覧（横スクロール対応）")
            df_details = pd.DataFrame(detailed_logs)
            df_details.set_index(['日付', '場所', '距離'], inplace=True)
            st.dataframe(df_details, use_container_width=True)
            
            csv = df_details.reset_index().to_csv(index=False, encoding='utf-8-sig')
            st.download_button(label="📥 推奨馬一覧データをCSVでダウンロード", data=csv, file_name="backtest_matrix_details.csv", mime="text/csv")
        else:
            st.warning("検証可能なデータがありませんでした。")