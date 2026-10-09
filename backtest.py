# -*- coding: utf-8 -*-
"""
ロジック検証エンジン (Backtest Matrix) 修正版 v2

【v1からの主な修正点】
 1. 馬場データ結合の診断を追加（一致率・キー重複・不一致サンプル・場所名の差分）
 2. 直近3走の取り方を全展開ロジックで統一（v1は展開5・6だけ最古3走になっていた）
 3. 4角平均の分母バグ修正（有効値のみで平均）
 4. 同一日・同一場所・同一距離の別レースが1レースに混ざる問題への対策（レース識別列の指定）
 5. 展開5・6のベーススコアを「血統スコア×馬場補正」に整理（v1は脚質補正後の値を流用していた）
 6. 消し馬スコアの馬場補正の符号バグ修正（負の値のとき逆方向に効いていた）
 7. 同点時に馬番順で選ばれる偏りを、基礎能力(PCI/上り3F)での決着に変更
 8. 学習データ→選定期間→検証期間 の3分割集計、信頼区間、最高配当除外回収率、ベースライン比較を追加
 9. ニックスZ値の母集団を出走数下限以上に限定／馬場補正の最小サンプル数を設定
10. 血統の世代計算を実際の世代(父=1, 母父=2)に補正（CONFIGで旧仕様に戻せる）
11. マジックナンバーをCONFIGに集約、キャッシュの更新検知、スクリプト基準のファイル探索

※ 学習用/テスト用のZIPと baba_info_complete.csv は、このファイルと同じフォルダに置いてください。
"""
import io
import os
import re
import zipfile
import unicodedata
from collections import namedtuple
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ロジック検証エンジン (Backtest Matrix) v2", layout="wide")

# ==========================================================
# 設定（マジックナンバーはここに集約。変更したら結果の再現性のため記録しておくこと）
# ==========================================================
CONFIG = {
    "MIN_FIELD": 8,              # これ未満の頭数のレースは除外
    "MAX_FIELD": 18,             # これを超える頭数のグループは「別レース混在」とみなして除外
    "NICKS_MIN_RUNS": 3,         # ニックス統計を採用する最低出走数
    "NICKS_DEFAULT_A": 10,       # A: 出走数不足時のスコア
    "NICKS_DEFAULT_B": 30,       # B: 出走数不足時のスコア
    "CUSHION_NEAR": 0.3,         # 芝: 好走時クッション値との差がこれ以下なら加点
    "CUSHION_FAR": 1.0,          # 芝: 差がこれ以上なら減点
    "MOISTURE_NEAR": 1.5,        # ダート: 好走時含水率との差がこれ以下なら加点
    "MOISTURE_FAR": 4.0,         # ダート: 差がこれ以上なら減点
    "BABA_BONUS_GOOD": 1.3,
    "BABA_BONUS_BAD": 0.7,
    "BABA_MIN_GOOD_RUNS": 2,     # 馬場適性判定に必要な「馬場データ付き好走(3着内)」の最低回数（v1は1回）
    "CORNER_FRONT": 4,           # 4角平均がこれ以下なら先行評価
    "CORNER_REAR": 10,           # 4角平均がこれ以上なら後方評価
    "FRONT_MULT": 1.5,
    "REAR_MULT": 0.5,
    "SHUNSOKU_PCI": 51,          # 展開2: 瞬発力判定のPCI下限
    "SHUNSOKU_F3": 34.9,         # 展開2: 瞬発力判定の上り3F上限
    "SHUNSOKU_MULT": 1.5,
    "BASE_PCI": 45,              # 展開3: 基礎能力のPCI基準
    "BASE_F3": 36.0,             # 展開3: 基礎能力の上り3F基準
    "STRICT_PCI": 45,            # 展開4: 厳格フィルター
    "STRICT_F3": 36.0,
    "STRICT_CORNER": 10,
    "WORST_MIN_PAST": 3,         # 消し馬判定に必要な過去走数
    "WORST_N": 3,                # 消し馬の頭数
    "PEDIGREE_TRUE_GEN": True,   # True: 実際の世代(父=1,母父=2)で配合判定 / False: v1の計算(1つ若い世代)
}

BLOOD_LOGICS = ['A:回収率重視', 'B:適性Z値', 'C:展開ポテンシャル', 'D:配合洗練']
PACE_LOGICS = ['1:ベース(馬場適性加味)', '2:瞬発力(PCI×3F)', '3:基礎能力加算',
               '4:厳格フィルター', '5:メンバー内偏差値', '6:過去コース乖離Z値']
WORST_NAME = "🚨消し馬:絶対に来ないワースト3頭"
BASE_ALL = "📏ベースライン:全頭均等買い"
BASE_POP1 = "📏ベースライン:1番人気"
BABA_FILE = "baba_info_complete.csv"
BABA_VALUE_COLS = ['クッション値', '芝_含水率_ゴール前', 'ダート_含水率_ゴール前']
REQUIRED_COLS = ['馬名', '場所', '距離', '着順']

Past = namedtuple('Past', 'date rank corner pci f3 place dist cushion m_turf m_dirt')

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()


def combo_name(b, p):
    return f"血統{b}_展開{p}"


ALL_COMBOS = [combo_name(b, p) for b in BLOOD_LOGICS for p in PACE_LOGICS]


# ==========================================================
# 1. ユーティリティ
# ==========================================================
def isn(x):
    return x is None or x != x


def clean_name(name):
    if pd.isna(name):
        return ""
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(name)))


def clean_date(d):
    """様々な日付表記を YYYYMMDD 文字列へ。失敗時は '20000101'（診断で失敗率を表示）"""
    if pd.isna(d):
        return "20000101"
    s = unicodedata.normalize('NFKC', str(d)).strip()
    m = re.match(r'^(\d{4})[/\-.年](\d{1,2})[/\-.月](\d{1,2})', s)
    if m:
        return f"{int(m.group(1)):04d}{int(m.group(2)):02d}{int(m.group(3)):02d}"
    digits = re.sub(r'\D', '', s.split('.')[0])
    if len(digits) == 6:
        digits = "20" + digits
    m = re.search(r'\d{8}', digits)
    return m.group() if m else "20000101"


def parse_rank(x):
    if pd.isna(x):
        return 99
    nums = re.findall(r'\d+', unicodedata.normalize('NFKC', str(x)))
    return int(nums[0]) if nums else 99


def col_or(df, name, default=np.nan):
    return df[name] if name in df.columns else pd.Series(default, index=df.index)


def to_money(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '', regex=False), errors='coerce').fillna(0)


def read_csv_bytes(data):
    last = None
    for enc in ("utf-8-sig", "cp932"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except UnicodeDecodeError as e:
            last = e
    raise last


def file_signature():
    items = []
    for f in sorted(os.listdir(BASE_DIR)):
        if f.endswith('.zip') or f == BABA_FILE:
            p = os.path.join(BASE_DIR, f)
            items.append((f, os.path.getmtime(p), os.path.getsize(p)))
    return tuple(items)


# ==========================================================
# 2. データ読み込み ＆ 前処理
# ==========================================================
def prepare_baba(df_baba):
    """馬場CSVを (日付, 場所) キーで一意化。重複は平均化。"""
    info = {'rows': 0, 'dup': 0, 'places': [], 'date_min': None, 'date_max': None, 'error': None}
    if df_baba is None or df_baba.empty:
        return None, info
    b = df_baba.copy()
    for need in ['日付', '場所']:
        if need not in b.columns:
            info['error'] = f"馬場CSVに『{need}』列がありません（列: {list(b.columns)}）"
            return None, info
    info['rows'] = len(b)
    b['場所'] = b['場所'].map(clean_name)
    b['検索用_日付'] = b['日付'].map(clean_date)
    present = [c for c in BABA_VALUE_COLS if c in b.columns]
    for c in present:
        b[c] = pd.to_numeric(b[c], errors='coerce')
    keys = ['検索用_日付', '場所']
    info['dup'] = int(b.duplicated(subset=keys).sum())
    info['places'] = sorted(b['場所'].unique().tolist())
    ok = b['検索用_日付'] != '20000101'
    if ok.any():
        info['date_min'], info['date_max'] = b.loc[ok, '検索用_日付'].min(), b.loc[ok, '検索用_日付'].max()
    b = b.groupby(keys, as_index=False)[present].mean()
    return b, info


def process_dataframe(df, baba):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"必須列がありません: {missing} / 実際の列: {list(df.columns)}")
    df = df.copy()

    df['馬名_key'] = df['馬名'].map(clean_name)
    df['場所'] = df['場所'].map(clean_name)
    df['計算用_着順'] = df['着順'].map(parse_rank)
    df['有効出走'] = (df['計算用_着順'] != 99).astype(int)
    df['単勝配当_数値'] = to_money(col_or(df, '単勝配当', 0))
    df['複勝配当_数値'] = to_money(col_or(df, '複勝配当', 0))
    df['計算用_PCI'] = pd.to_numeric(col_or(df, 'PCI'), errors='coerce')
    df['計算用_上り3F'] = pd.to_numeric(col_or(df, '上り3F'), errors='coerce')
    corner_col = next((c for c in df.columns if '4角' in str(c)), None)
    df['計算用_4角'] = pd.to_numeric(df[corner_col], errors='coerce') if corner_col else np.nan
    df['計算用_人気'] = col_or(df, '人気', 99).map(parse_rank) if '人気' in df.columns else 99

    date_col = next((c for c in df.columns if '日付' in str(c)), None)
    if date_col is None:
        raise ValueError("日付列（列名に『日付』を含む列）がありません")
    df['検索用_日付'] = df[date_col].map(clean_date)
    df['検索用_日付_数値'] = pd.to_numeric(df['検索用_日付'], errors='coerce')
    df['日付パース失敗'] = df['検索用_日付'] == '20000101'

    sire_col = next((c for c in df.columns if c in ['父', '父馬', '種牡馬']), None)
    bms_col = next((c for c in df.columns if c in ['母父馬', '母父', 'BMS']), None)
    if sire_col and bms_col:
        df['検索用_父'] = df[sire_col].map(clean_name)
        df['検索用_母父'] = df[bms_col].map(clean_name)
        df['ニックス'] = df['検索用_父'] + " × " + df['検索用_母父']
    else:
        df['検索用_父'], df['検索用_母父'], df['ニックス'] = "", "", "不明"

    n0 = len(df)
    if baba is not None and not baba.empty:
        df = df.merge(baba, on=['検索用_日付', '場所'], how='left')
        if len(df) != n0:
            raise RuntimeError(f"馬場マージで行数が変化しました: {n0} → {len(df)}")
    for c in BABA_VALUE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _read_zip_csvs(path, require_name_col=False):
    dfs = []
    with zipfile.ZipFile(path, 'r') as z:
        for name in z.namelist():
            if not name.lower().endswith('.csv') or '__MACOSX' in name or os.path.basename(name).startswith('.'):
                continue
            try:
                d = read_csv_bytes(z.read(name))
            except pd.errors.EmptyDataError:
                continue
            if require_name_col and '馬名' not in d.columns:
                continue
            dfs.append(d)
    return dfs


@st.cache_resource(show_spinner="データ読み込み・辞書構築中（初回のみ）...")
def load_and_build(track, sig):
    C = CONFIG
    notes = []
    files = sorted(os.listdir(BASE_DIR))
    keyword = "芝" if track == "芝" else "ダート"
    train_c, test_c, ped_files = [], [], []
    for f in files:
        nf = unicodedata.normalize('NFC', f)
        if not nf.endswith('.zip'):
            continue
        if keyword in nf:
            if '学習用' in nf:
                train_c.append(f)
            elif 'テスト用' in nf:
                test_c.append(f)
        if any(k in nf for k in ['血統', 'マスタ', '種牡馬', '母父']):
            ped_files.append(f)
    if not train_c or not test_c:
        raise FileNotFoundError(f"『{track}』の学習用/テスト用ZIPが見つかりません（探索フォルダ: {BASE_DIR}）")
    if len(train_c) > 1 or len(test_c) > 1:
        notes.append(f"学習用ZIP候補={train_c} / テスト用ZIP候補={test_c} → 名前順で最後のものを使用")
    train_zip, test_zip = train_c[-1], test_c[-1]

    # 馬場CSV
    baba_path = os.path.join(BASE_DIR, BABA_FILE)
    df_baba_raw = pd.DataFrame()
    if os.path.exists(baba_path):
        with open(baba_path, 'rb') as fh:
            df_baba_raw = read_csv_bytes(fh.read())
    else:
        notes.append(f"{BABA_FILE} が見つかりません。馬場適性補正は無効になります。")
    baba, baba_info = prepare_baba(df_baba_raw)
    if baba_info['error']:
        notes.append(baba_info['error'])

    def load_zip(path):
        dfs = _read_zip_csvs(os.path.join(BASE_DIR, path))
        if not dfs:
            raise ValueError(f"{path} にCSVがありません")
        return process_dataframe(pd.concat(dfs, ignore_index=True), baba)

    df_train = load_zip(train_zip)
    df_test = load_zip(test_zip)

    # 血統マスタ
    ped_dfs = []
    for f in ped_files:
        for d in _read_zip_csvs(os.path.join(BASE_DIR, f), require_name_col=True):
            d = d.copy()
            d['検索用馬名'] = d['馬名'].map(clean_name)
            ped_dfs.append(d)
    df_ped = (pd.concat(ped_dfs, ignore_index=True).drop_duplicates(subset=['検索用馬名'])
              if ped_dfs else pd.DataFrame())
    pedigree = df_ped.set_index('検索用馬名').to_dict('index') if not df_ped.empty else {}

    # ---- 過去走履歴（新しい順）。学習+テストを結合するが、参照時に date < race_date で絞る ----
    k = ['検索用_日付', '場所', '距離', '馬名_key', '計算用_着順']
    master = pd.concat([df_train, df_test], ignore_index=True).drop_duplicates(subset=k)
    master = master.sort_values('検索用_日付_数値', ascending=False, kind='mergesort')
    hcols = ['馬名_key', '検索用_日付_数値', '計算用_着順', '計算用_4角', '計算用_PCI', '計算用_上り3F',
             '場所', '距離', 'クッション値', '芝_含水率_ゴール前', 'ダート_含水率_ゴール前']
    history = {}
    for vals in zip(*(master[c].tolist() for c in hcols)):
        if vals[0]:
            history.setdefault(vals[0], []).append(Past(*vals[1:]))

    # ---- ニックス統計（学習データのみ）----
    t = df_train.copy()
    t['tan_pay'] = np.where(t['計算用_着順'] == 1, t['単勝配当_数値'], 0.0)
    t['fuku_pay'] = np.where(t['計算用_着順'].between(1, 3), t['複勝配当_数値'], 0.0)
    t['is_place'] = t['計算用_着順'].between(1, 3).astype(int)
    g = t.groupby(['場所', '距離', 'ニックス'], sort=False).agg(
        runs=('有効出走', 'sum'), place=('is_place', 'sum'),
        tan=('tan_pay', 'sum'), fuku=('fuku_pay', 'sum')).reset_index()
    den = g['runs'].clip(lower=1)
    g['place_rate'] = g['place'] / den
    g['tan_ret'] = g['tan'] / den        # 配当は100円あたり円 → そのまま回収率(%)
    g['fuku_ret'] = g['fuku'] / den
    q = g[g['runs'] >= C['NICKS_MIN_RUNS']]
    cm = q.groupby(['場所', '距離'])['place_rate'].agg(['mean', 'std']).reset_index()
    g = g.merge(cm, on=['場所', '距離'], how='left')
    std = g['std'].where(g['std'] > 0, 1.0).fillna(1.0)
    g['z'] = ((g['place_rate'] - g['mean']) / std * 10 + 50).where(g['mean'].notna(), 50.0)
    nicks = {(p, d, n): (r, z, tr, fr) for p, d, n, r, z, tr, fr in
             zip(g['場所'], g['距離'], g['ニックス'], g['runs'], g['z'], g['tan_ret'], g['fuku_ret'])}

    sire_stats = t.groupby('検索用_父')[['計算用_PCI', '計算用_上り3F']].mean()
    sire = {k_: (r['計算用_PCI'], r['計算用_上り3F']) for k_, r in sire_stats.iterrows()}

    cs = t.groupby(['場所', '距離'])[['計算用_PCI', '計算用_上り3F']].agg(['mean', 'std'])
    cs.columns = ['pci_mean', 'pci_std', 'f3_mean', 'f3_std']
    course = cs.to_dict('index')

    # ---- 診断情報 ----
    key4 = ['検索用_日付', '場所', '距離', '馬名_key']
    overlap = len(pd.merge(df_train[key4].drop_duplicates(), df_test[key4].drop_duplicates(), on=key4))

    def rng(d):
        ok = d.loc[~d['日付パース失敗'], '検索用_日付']
        return (ok.min(), ok.max()) if len(ok) else (None, None)

    baba_rel = 'クッション値' if track == '芝' else 'ダート_含水率_ゴール前'
    any_baba = lambda d: d[BABA_VALUE_COLS].notna().any(axis=1)
    unmatched = (df_test.loc[~any_baba(df_test), ['検索用_日付', '場所']]
                 .value_counts().head(15).reset_index(name='行数'))
    win_pay = df_train.loc[df_train['計算用_着順'] == 1, '単勝配当_数値']
    plc_pay = df_train.loc[df_train['計算用_着順'].between(1, 3), '複勝配当_数値']
    diag = {
        'notes': notes,
        'files': {'train': train_zip, 'test': test_zip, 'pedigree': ped_files, 'baba': BABA_FILE if os.path.exists(baba_path) else None},
        'rows': {'train': len(df_train), 'test': len(df_test)},
        'range': {'train': rng(df_train), 'test': rng(df_test)},
        'overlap': overlap,
        'date_fail': {'train': float(df_train['日付パース失敗'].mean()), 'test': float(df_test['日付パース失敗'].mean())},
        'baba_info': baba_info,
        'baba_rel': baba_rel,
        'baba_match': {
            'train_any': float(any_baba(df_train).mean()), 'test_any': float(any_baba(df_test).mean()),
            'train_rel': float(df_train[baba_rel].notna().mean()), 'test_rel': float(df_test[baba_rel].notna().mean()),
        },
        'baba_unmatched': unmatched,
        'race_places': sorted(df_test['場所'].unique().tolist()),
        'payout': {
            'win_mean': float(win_pay.mean()) if len(win_pay) else float('nan'),
            'win_median': float(win_pay.median()) if len(win_pay) else float('nan'),
            'place_mean': float(plc_pay.mean()) if len(plc_pay) else float('nan'),
        },
        'null_rates': {c: float(df_test[c].isna().mean()) for c in
                       ['計算用_PCI', '計算用_上り3F', '計算用_4角']},
        'sire_missing': float((df_test['検索用_父'] == '').mean()),
        'ped_n': len(pedigree),
        'rank99_test': float((df_test['計算用_着順'] == 99).mean()),
        'has_pop': bool((df_test['計算用_人気'] != 99).any()),
    }
    return {'df_train': df_train, 'df_test': df_test, 'history': history, 'nicks': nicks,
            'sire': sire, 'course': course, 'pedigree': pedigree, 'diag': diag}


def detect_race_col(df):
    for p in ['レースID', 'レースId', 'race_id', 'RaceID', 'レースＩＤ', 'レース番号', 'R', 'Ｒ', 'R番号', 'レース名', 'レース']:
        if p in df.columns:
            return p
    return None


def show_diagnostics(A, track, race_col):
    d = A['diag']
    df_test = A['df_test']
    with st.expander("🩺 データ診断（実行前に必ず確認）", expanded=True):
        for n in d['notes']:
            st.warning(n)

        c1, c2, c3 = st.columns(3)
        c1.metric("学習データ行数", f"{d['rows']['train']:,}")
        c2.metric("テストデータ行数", f"{d['rows']['test']:,}")
        c3.metric("学習⇔テスト重複行(リーク)", f"{d['overlap']:,}")
        st.write(f"学習期間: {d['range']['train'][0]} 〜 {d['range']['train'][1]} ／ "
                 f"テスト期間: {d['range']['test'][0]} 〜 {d['range']['test'][1]}")
        st.caption(f"使用ファイル: 学習={d['files']['train']} / テスト={d['files']['test']} / "
                   f"血統={d['files']['pedigree']} / 馬場={d['files']['baba']}")
        if d['overlap'] > 0:
            st.error("🚨 学習データとテストデータに同一レースが含まれています。結果は信用できません。")
        tr_end, te_start = d['range']['train'][1], d['range']['test'][0]
        if tr_end and te_start and tr_end >= te_start:
            st.error(f"🚨 学習期間の終了({tr_end})がテスト期間の開始({te_start})以降です。時系列分割になっていません。")
        if d['date_fail']['train'] > 0 or d['date_fail']['test'] > 0:
            st.error(f"🚨 日付パース失敗率 学習={d['date_fail']['train']:.1%} / テスト={d['date_fail']['test']:.1%}")

        st.markdown("**🌦 馬場データ結合**")
        bm, bi = d['baba_match'], d['baba_info']
        rel = d['baba_rel']
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"テスト: {rel} 一致率", f"{bm['test_rel']:.1%}")
        m2.metric(f"学習: {rel} 一致率", f"{bm['train_rel']:.1%}")
        m3.metric("馬場CSV行数", f"{bi['rows']:,}")
        m4.metric("馬場CSVキー重複(平均化済)", f"{bi['dup']:,}")
        if bi['rows'] and bm['test_rel'] < 0.8:
            st.error("🚨 馬場データの一致率が低く、馬場補正はほぼ効いていません。下の『場所名の差』『不一致サンプル』を確認してください。")
        if bi['rows']:
            st.write(f"馬場CSV期間: {bi['date_min']} 〜 {bi['date_max']}")
            only_race = sorted(set(d['race_places']) - set(bi['places']))
            only_baba = sorted(set(bi['places']) - set(d['race_places']))
            st.write(f"レース側の場所: {d['race_places']}")
            st.write(f"馬場CSV側の場所: {bi['places']}")
            if only_race:
                st.warning(f"馬場CSVに存在しない場所（表記ゆれの可能性）: {only_race}")
            if only_baba:
                st.caption(f"レース側に存在しない馬場CSVの場所: {only_baba}")
            if len(d['baba_unmatched']):
                st.write("馬場が付かなかった(日付,場所)の上位:")
                st.dataframe(d['baba_unmatched'])

        st.markdown("**💴 配当・欠損の妥当性**")
        p = d['payout']
        st.write(f"学習データ: 勝ち馬の単勝配当 平均={p['win_mean']:.0f} / 中央値={p['win_median']:.0f}、"
                 f"3着内の複勝配当 平均={p['place_mean']:.0f}")
        if p['win_median'] < 100:
            st.error("🚨 単勝配当が『100円あたりの円』ではなくオッズ表記の可能性があります（中央値が100未満）。回収率計算が狂います。")
        nr = d['null_rates']
        st.write(f"テストデータ欠損率: PCI={nr['計算用_PCI']:.1%} / 上り3F={nr['計算用_上り3F']:.1%} / "
                 f"4角={nr['計算用_4角']:.1%} / 父欠損={d['sire_missing']:.1%} / 着順99(除外・中止等)={d['rank99_test']:.1%} / "
                 f"血統マスタ頭数={d['ped_n']:,}")
        if nr['計算用_4角'] > 0.5:
            st.error("🚨 4角列がほとんど数値化できていません（脚質補正・消し馬判定が機能しません）。")

        st.markdown("**🏇 レース単位の分割**")
        key_cols = ['検索用_日付_数値', '場所', '距離'] + ([race_col] if race_col else [])
        sizes = df_test.groupby(key_cols).size()
        c1, c2, c3 = st.columns(3)
        c1.metric("レース(グループ)数", f"{len(sizes):,}")
        c2.metric(f"{CONFIG['MAX_FIELD']}頭超のグループ", f"{int((sizes > CONFIG['MAX_FIELD']).sum()):,}")
        c3.metric(f"{CONFIG['MIN_FIELD']}頭未満のグループ", f"{int((sizes < CONFIG['MIN_FIELD']).sum()):,}")
        if (sizes > CONFIG['MAX_FIELD']).any():
            st.error("🚨 同一日・同一場所・同一距離の別レースが混ざっています。サイドバーの『レース識別列』でR番号やレースID列を指定してください。"
                     "（該当グループは集計から除外されます）")
        if not race_col:
            st.info("レース識別列が未指定です。同日同場所同距離に複数レースがあると、1レースに1頭しか選べない誤集計になります。")


# ==========================================================
# 3. バックテスト本体
# ==========================================================
GOOD_CROSS = {frozenset({3, 4}), frozenset({4}), frozenset({4, 5})}
BAD_CROSS = {frozenset({2, 3}), frozenset({3})}


def _ancestors(row, offset):
    out = {}
    for k, v in row.items():
        if re.fullmatch(r'[父母]+', str(k)) and not pd.isna(v) and str(v) not in ('nan', ''):
            out.setdefault(clean_name(v), set()).add(len(str(k)) + offset)
    return out


def pedigree_score(sire, bms, pedigree, cache):
    key = (sire, bms)
    if key in cache:
        return cache[key]
    score = 30
    s_row, b_row = pedigree.get(sire), pedigree.get(bms)
    if s_row and b_row:
        s_off, b_off = (1, 2) if CONFIG['PEDIGREE_TRUE_GEN'] else (0, 1)
        s_anc, b_anc = _ancestors(s_row, s_off), _ancestors(b_row, b_off)
        for anc in set(s_anc) & set(b_anc):
            for sg in s_anc[anc]:
                for bg in b_anc[anc]:
                    cr = frozenset({sg, bg})
                    if cr in GOOD_CROSS:
                        score += 15
                    elif cr in BAD_CROSS:
                        score -= 20
    cache[key] = max(0, score)
    return cache[key]


def run_backtest(A, track, race_col, progress=None):
    C = CONFIG
    df = A['df_test']
    history, nicks, sire_d, course_d, pedigree = A['history'], A['nicks'], A['sire'], A['course'], A['pedigree']
    key_cols = ['検索用_日付_数値', '場所', '距離'] + ([race_col] if race_col else [])
    grouped = df.groupby(key_cols, sort=True)
    total = grouped.ngroups
    ped_cache = {}
    records, race_logs = [], []
    stat = dict(races=0, skip_small=0, skip_large=0, horses=0, baba_applied=0, baba_checked=0, no_pick=0)

    def std_floor(s):
        return s if (not isn(s) and s > 1.0) else 1.0

    for gi, (gkey, rdf) in enumerate(grouped):
        if progress is not None and gi % 40 == 0:
            progress.progress(min(gi / max(total, 1), 1.0), text=f"解析中... {gi}/{total} レース")
        n = len(rdf)
        if n < C['MIN_FIELD']:
            stat['skip_small'] += 1
            continue
        if n > C['MAX_FIELD']:
            stat['skip_large'] += 1
            continue

        race_date, place, dist = int(gkey[0]), gkey[1], gkey[2]
        rid = gkey[3] if race_col else ''
        cur_cush = rdf['クッション値'].mean()
        cur_dirt = rdf['ダート_含水率_ゴール前'].mean()

        # ---------- パス1: 馬ごとの特徴量 ----------
        feats = []
        for rec in rdf.to_dict('records'):
            past = [p for p in history.get(rec['馬名_key'], ()) if p.date < race_date]   # 新しい順
            past3 = past[:3]
            good = [p for p in past if 1 <= p.rank <= 3]

            v_pci = [p.pci for p in past3 if not isn(p.pci)]
            v_f3 = [p.f3 for p in past3 if not isn(p.f3)]
            v_cor = [p.corner for p in past3 if not isn(p.corner)]
            avg_pci = sum(v_pci) / len(v_pci) if v_pci else np.nan
            avg_f3 = sum(v_f3) / len(v_f3) if v_f3 else np.nan
            avg_cor = sum(v_cor) / len(v_cor) if v_cor else np.nan

            # 馬場適性ボーナス
            bonus = 1.0
            if track == '芝' and not isn(cur_cush):
                vals = [p.cushion for p in good if not isn(p.cushion)]
                stat['baba_checked'] += 1
                if len(vals) >= C['BABA_MIN_GOOD_RUNS']:
                    diff = abs(cur_cush - sum(vals) / len(vals))
                    if diff <= C['CUSHION_NEAR']:
                        bonus = C['BABA_BONUS_GOOD']
                    elif diff >= C['CUSHION_FAR']:
                        bonus = C['BABA_BONUS_BAD']
            elif track == 'ダート' and not isn(cur_dirt):
                vals = [p.m_dirt for p in good if not isn(p.m_dirt)]
                stat['baba_checked'] += 1
                if len(vals) >= C['BABA_MIN_GOOD_RUNS']:
                    diff = abs(cur_dirt - sum(vals) / len(vals))
                    if diff <= C['MOISTURE_NEAR']:
                        bonus = C['BABA_BONUS_GOOD']
                    elif diff >= C['MOISTURE_FAR']:
                        bonus = C['BABA_BONUS_BAD']
            if bonus != 1.0:
                stat['baba_applied'] += 1
            stat['horses'] += 1

            # 血統スコア4種
            runs, z, tan_r, fuku_r = nicks.get((place, dist, rec['ニックス']), (0, 50.0, 0.0, 0.0))
            blood = {}
            blood['A:回収率重視'] = (tan_r + fuku_r) / 2 if runs >= C['NICKS_MIN_RUNS'] else C['NICKS_DEFAULT_A']
            blood['B:適性Z値'] = z if runs >= C['NICKS_MIN_RUNS'] else C['NICKS_DEFAULT_B']
            sp, sf = sire_d.get(rec['検索用_父'], (45.0, 36.0))
            sp = 45.0 if isn(sp) else sp
            sf = 36.0 if isn(sf) else sf
            blood['C:展開ポテンシャル'] = max(0, sp - 40) + max(0, 36.5 - sf) * 10
            blood['D:配合洗練'] = pedigree_score(rec['検索用_父'], rec['検索用_母父'], pedigree, ped_cache)

            # 過去コース乖離Z（展開6用）
            devs = []
            for p in past3:
                cs = course_d.get((p.place, p.dist))
                if cs and not isn(p.pci) and not isn(p.f3) and not isn(cs['pci_mean']) and not isn(cs['f3_mean']):
                    devs.append(((p.pci - cs['pci_mean']) / std_floor(cs['pci_std']) * 10 + 50) +
                                ((cs['f3_mean'] - p.f3) / std_floor(cs['f3_std']) * 10 + 50))
            dev_avg = sum(devs) / len(devs) if devs else np.nan

            sh_cnt = sum(1 for p in past3 if not isn(p.pci) and not isn(p.f3)
                         and p.pci >= C['SHUNSOKU_PCI'] and p.f3 <= C['SHUNSOKU_F3'])
            ability = (avg_pci + (36.0 - avg_f3) * 5) if (not isn(avg_pci) and not isn(avg_f3)) else -1e9

            # 消し馬スコア（低いほど消し）
            worst = np.nan
            if len(past3) >= C['WORST_MIN_PAST'] and not isn(avg_pci) and not isn(avg_f3) and not isn(avg_cor):
                w = avg_pci + (36.0 - avg_f3) * 5 - avg_cor * 2 + blood['A:回収率重視'] * 0.1
                w = w / bonus if w >= 0 else w * bonus      # 馬場不適(bonus<1)なら必ず「より悪く」なる向きに補正
                worst = w

            feats.append(dict(rec=rec, n_past=len(past3), avg_pci=avg_pci, avg_f3=avg_f3, avg_cor=avg_cor,
                              bonus=bonus, blood=blood, dev_avg=dev_avg, sh_cnt=sh_cnt,
                              ability=ability, worst=worst))

        # ---------- メンバー内統計（展開5用）----------
        mp = [f['avg_pci'] for f in feats if not isn(f['avg_pci']) and not isn(f['avg_f3'])]
        mf = [f['avg_f3'] for f in feats if not isn(f['avg_pci']) and not isn(f['avg_f3'])]
        use_mem = len(mp) >= 2 and np.std(mp) > 0 and np.std(mf) > 0
        if use_mem:
            m_pci, s_pci, m_f3, s_f3 = np.mean(mp), np.std(mp), np.mean(mf), np.std(mf)

        # ---------- パス2: 24パターンのスコア ----------
        def score(f, b_key, p_key):
            base = f['blood'][b_key] * f['bonus']
            ap, af, ac = f['avg_pci'], f['avg_f3'], f['avg_cor']
            if p_key in ('1:ベース(馬場適性加味)', '2:瞬発力(PCI×3F)', '3:基礎能力加算', '4:厳格フィルター'):
                if f['n_past'] == 0 or isn(ac):
                    return 0.0
                if p_key.startswith('1'):
                    if ac <= C['CORNER_FRONT']:
                        return base * C['FRONT_MULT']
                    if ac >= C['CORNER_REAR']:
                        return base * C['REAR_MULT']
                    return base
                if p_key.startswith('2'):
                    return base * (C['SHUNSOKU_MULT'] ** f['sh_cnt'])
                if p_key.startswith('3'):
                    if not isn(ap) and not isn(af):
                        return base + max(0, ap - C['BASE_PCI']) * 2 + max(0, C['BASE_F3'] - af) * 10
                    return base
                if isn(ap) or isn(af):
                    return 0.0
                if ap < C['STRICT_PCI'] or af >= C['STRICT_F3'] or ac >= C['STRICT_CORNER']:
                    return 0.0
                return base
            if p_key.startswith('5'):
                if use_mem and not isn(ap) and not isn(af):
                    return base + ((ap - m_pci) / s_pci * 10 + 50) + ((m_f3 - af) / s_f3 * 10 + 50)
                return 0.0
            if isn(f['dev_avg']):
                return 0.0
            return base + f['dev_avg']

        race_key = f"{race_date}_{place}_{dist}_{rid}"
        stat['races'] += 1
        log = {'日付': race_date, '場所': place, '距離': dist}
        if race_col:
            log['レース識別'] = rid

        def add(logic, kind, rec):
            rk = int(rec['計算用_着順'])
            records.append({
                'logic': logic, 'kind': kind, 'date': race_date, 'race': race_key,
                'horse': rec.get('馬名', ''), 'rank': rk,
                'win': 1 if rk == 1 else 0, 'place': 1 if 1 <= rk <= 3 else 0,
                'win_ret': rec['単勝配当_数値'] if rk == 1 else 0.0,
                'place_ret': rec['複勝配当_数値'] if 1 <= rk <= 3 else 0.0,
            })

        for b_key in BLOOD_LOGICS:
            for p_key in PACE_LOGICS:
                name = combo_name(b_key, p_key)
                best, best_key = None, None
                for f in feats:
                    s = score(f, b_key, p_key)
                    if s > 0 and (best_key is None or (s, f['ability']) > best_key):
                        best, best_key = f, (s, f['ability'])
                if best is None:
                    log[name] = "-"
                    stat['no_pick'] += 1
                else:
                    add(name, 'pattern', best['rec'])
                    rk = int(best['rec']['計算用_着順'])
                    log[name] = f"{best['rec'].get('馬名', '')}({rk if rk != 99 else '外'}着)"

        worst_list = sorted([f for f in feats if not isn(f['worst'])], key=lambda f: f['worst'])[:C['WORST_N']]
        for f in worst_list:
            add(WORST_NAME, 'worst', f['rec'])

        for f in feats:
            add(BASE_ALL, 'baseline', f['rec'])
        pops = [f for f in feats if f['rec']['計算用_人気'] == 1]
        if pops:
            add(BASE_POP1, 'baseline', pops[0]['rec'])

        race_logs.append(log)

    if progress is not None:
        progress.progress(1.0, text="解析完了")
    return {'picks': pd.DataFrame(records), 'logs': pd.DataFrame(race_logs), 'stat': stat}


# ==========================================================
# 4. 集計
# ==========================================================
ORDER = {n: i for i, n in enumerate(ALL_COMBOS + [WORST_NAME, BASE_ALL, BASE_POP1])}
KIND_JP = {'pattern': '24パターン', 'worst': '消し馬', 'baseline': 'ベースライン'}


def _ci(x):
    return 1.96 * np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def summarize(picks):
    rows = []
    if picks.empty:
        return pd.DataFrame()
    for (kind, logic), g in picks.groupby(['kind', 'logic'], sort=False):
        n = len(g)
        wr, pr = g['win_ret'].to_numpy(float), g['place_ret'].to_numpy(float)
        rows.append({
            '区分': KIND_JP[kind], '解析ロジック': logic, '対象数': n,
            '勝率(%)': g['win'].mean() * 100, '複勝率(%)': g['place'].mean() * 100,
            '単回値(%)': wr.mean(), '単回値±95%': _ci(wr),
            '単回値(最高配当除外)': (wr.sum() - wr.max()) / (n - 1) if n > 1 else np.nan,
            '複回値(%)': pr.mean(), '複回値±95%': _ci(pr),
            '複回値(最高配当除外)': (pr.sum() - pr.max()) / (n - 1) if n > 1 else np.nan,
        })
    out = pd.DataFrame(rows)
    out['_o'] = out['解析ロジック'].map(ORDER)
    return out.sort_values('_o').drop(columns='_o').reset_index(drop=True)


def style_summary(df):
    def row_style(row):
        if row['区分'] == '消し馬':
            return ['background-color:#ffd6d6'] * len(row)
        if row['区分'] == 'ベースライン':
            return ['background-color:#e6e6e6'] * len(row)
        return [''] * len(row)

    sty = df.style.apply(row_style, axis=1)
    pat = df[df['区分'] == '24パターン']
    for c in ['単回値(%)', '複回値(%)']:
        if len(pat) and pat[c].notna().any():
            sty = sty.set_properties(subset=pd.IndexSlice[[pat[c].idxmax()], [c]], **{'background-color': '#90ee90'})
    fmt = {c: '{:.1f}' for c in df.columns if c not in ('区分', '解析ロジック', '対象数')}
    return sty.format(fmt, na_rep='-')


# ==========================================================
# 5. 画面
# ==========================================================
st.title("🧪 血統(4) × 展開(6) ＝ 24パターン マトリックス検証 ＋ 馬場適性補正（v2 / 診断つき）")
st.write("学習データ（〜2025年）の統計で、テスト期間（2026年）の推奨馬を選び、回収率を検証します。"
         "**24パターンから最良を選ぶと偶然の高回収率を拾いやすい**ため、『選定期間』で選んで『検証期間』で再現するかを確認する設計です。")

st.sidebar.markdown("---")
track_type = st.sidebar.radio("📁 読み込むデータベースを選択", ["芝", "ダート"])
split_date = st.sidebar.date_input("選定期間の終了日（これより後＝検証期間）", value=date(2026, 6, 30))
split_int = int(split_date.strftime('%Y%m%d'))
st.sidebar.info("💡 芝はクッション値、ダートは含水率で当日馬場と過去好走時の馬場を比較して補正します。")

try:
    assets = load_and_build(track_type, file_signature())
except Exception as e:
    st.error("データ読み込みでエラーが発生しました。")
    st.exception(e)
    st.stop()

cols = list(assets['df_test'].columns)
auto = detect_race_col(assets['df_test'])
opts = ["(指定なし: 日付+場所+距離のみ)"] + cols
race_sel = st.sidebar.selectbox("レース識別列（R番号・レースID等）", opts,
                                index=(opts.index(auto) if auto in opts else 0))
race_col = None if race_sel == opts[0] else race_sel

with st.sidebar.expander("現在の設定値 (CONFIG)"):
    st.json(CONFIG)

show_diagnostics(assets, track_type, race_col)

state_key = f"bt::{track_type}::{race_col}"
if st.button("24パターン＋ワースト3頭 一括シミュレーションを開始", type="primary"):
    bar = st.progress(0.0, text="準備中...")
    st.session_state[state_key] = run_backtest(assets, track_type, race_col, bar)

res = st.session_state.get(state_key)
if res is None:
    st.info("診断結果に問題がないことを確認してから、上のボタンで実行してください。")
    st.stop()

picks, logs, stat = res['picks'], res['logs'], res['stat']
if picks.empty:
    st.warning("検証可能なデータがありませんでした。診断結果を確認してください。")
    st.stop()

picks_sel = picks[picks['date'] <= split_int]
picks_val = picks[picks['date'] > split_int]

tab1, tab2, tab3, tab4 = st.tabs(["📊 総合結果", "🔬 選定→検証", "🔍 推奨馬一覧", "🧾 実行ログ"])

with tab1:
    period = st.radio("集計期間", ["全期間", f"選定期間(〜{split_int})", f"検証期間({split_int}より後)"], horizontal=True)
    target = picks if period == "全期間" else (picks_sel if period.startswith("選定") else picks_val)
    summ = summarize(target)
    st.dataframe(style_summary(summ))
    st.caption("回収率は1点100円購入換算(%)。『±95%』は回収率の95%信頼区間の半幅（対象数が少ないほど広い）。"
               "『最高配当除外』は最も高い配当1本を除いた回収率で、大穴1本に依存していないかの確認用です。")
    if not summ.empty:
        wst = summ[summ['区分'] == '消し馬']
        base = summ[summ['解析ロジック'] == BASE_ALL]
        if len(wst) and len(base):
            st.info(f"🚨 消し馬の複勝率 {wst['複勝率(%)'].iloc[0]:.1f}% ／ 全頭平均の複勝率 {base['複勝率(%)'].iloc[0]:.1f}%。"
                    "消し馬が全頭平均より明確に低く、かつ対象数が十分にある場合のみ『見抜けている』と判断できます。"
                    "（回収率が低いだけでは根拠になりません：多くの馬券は元々70〜80%程度です）")

with tab2:
    s_sel, s_val = summarize(picks_sel), summarize(picks_val)
    if s_sel.empty or s_val.empty:
        st.warning("選定期間または検証期間のデータがありません。サイドバーの分割日を調整してください。")
    else:
        keep = ['解析ロジック', '対象数', '単回値(%)', '複回値(%)', '複回値±95%']
        m = (s_sel[s_sel['区分'] == '24パターン'][keep]
             .merge(s_val[s_val['区分'] == '24パターン'][keep], on='解析ロジック', suffixes=('_選定', '_検証')))
        m = m.sort_values('複回値(%)_選定', ascending=False).reset_index(drop=True)
        st.markdown("#### 選定期間の複勝回収率が高い順 → 検証期間で再現したか")
        st.dataframe(m.style.format({c: '{:.1f}' for c in m.columns if c not in ('解析ロジック', '対象数_選定', '対象数_検証')}, na_rep='-'))
        top5 = m.head(5)
        c1, c2, c3 = st.columns(3)
        c1.metric("選定期間 上位5の複回値(平均)", f"{top5['複回値(%)_選定'].mean():.1f}%")
        c2.metric("同じ5パターンの検証期間 複回値(平均)", f"{top5['複回値(%)_検証'].mean():.1f}%")
        c3.metric("全24パターンの検証期間 複回値(平均)", f"{m['複回値(%)_検証'].mean():.1f}%")
        st.caption("選定期間で良かったパターンが検証期間でも全体平均を上回らなければ、選定期間の好成績は偶然（過学習）の可能性が高いです。")
        bs = s_val[s_val['区分'] == 'ベースライン'][['解析ロジック', '対象数', '単回値(%)', '複回値(%)']]
        st.markdown("#### 参考：検証期間のベースライン")
        st.dataframe(bs.style.format({'単回値(%)': '{:.1f}', '複回値(%)': '{:.1f}'}, na_rep='-'))

with tab3:
    st.markdown("#### 🔍 全レース推奨馬一覧（横スクロール対応）")
    st.dataframe(logs)
    st.download_button("📥 推奨馬一覧をCSVでダウンロード", logs.to_csv(index=False, encoding='utf-8-sig'),
                       file_name="backtest_matrix_details.csv", mime="text/csv")
    st.download_button("📥 総合結果(全期間)をCSVでダウンロード", summarize(picks).to_csv(index=False, encoding='utf-8-sig'),
                       file_name="backtest_matrix_summary.csv", mime="text/csv")

with tab4:
    st.write(f"解析レース数: {stat['races']:,} ／ 頭数不足で除外: {stat['skip_small']:,} ／ "
             f"{CONFIG['MAX_FIELD']}頭超(別レース混在)で除外: {stat['skip_large']:,}")
    chk = max(stat['baba_checked'], 1)
    st.write(f"当日馬場データがあり判定対象になった馬: {stat['baba_checked']:,} 頭 ／ "
             f"そのうち馬場補正(加点/減点)が実際に発動: {stat['baba_applied']:,} 頭 ({stat['baba_applied'] / chk:.1%})")
    st.write(f"推奨馬が出なかった(全馬スコア0)パターン×レースの件数: {stat['no_pick']:,}")
    if stat['baba_applied'] == 0:
        st.error("🚨 馬場補正が1度も発動していません。馬場データの結合を確認してください。")
    st.json(CONFIG)