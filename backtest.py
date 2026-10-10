# -*- coding: utf-8 -*-
"""
血統分析エンジン v3（先祖内包・クロス別 的中率／回収率）

【やること】
  新形式CSV（日付・確定着順・単勝オッズ・複勝配当・枠番など）と、現行の血統マスターを使い、
  「どの先祖を内包しているか」「何代前でクロスしているか（例: 3×4）」ごとに
  勝率・複勝率・単勝回収率・複勝回収率を、学習期間と検証期間を並べて出力します。

【フォルダに置くもの（このファイルと同じ場所）】
  ・芝_学習用.zip / 芝_テスト用.zip / ダート_学習用.zip / ダート_テスト用.zip （名前に「芝」or「ダート」と「学習用」or「テスト用」を含む）
  ・血統マスターのZIP（名前に「血統」「マスタ」「種牡馬」「母父」のいずれかを含む。中のCSVに『馬名』列と『父』『母』『父父』…列）
  ・baba_info_complete.csv（日付, 場所, クッション値, 芝_含水率_ゴール前, ダート_含水率_ゴール前）

【回収率の定義】 1点100円購入換算(%)。単勝＝オッズ×100（単勝配当列があればそちら）、複勝＝『複勝』列の配当。
【統計上の注意】 先祖は数百種あり、学習期間の回収率が高い先祖は偶然である可能性が高いです。
                必ず「検証期間で再現したか」を見てください（サマリー欄に自動集計します）。
"""
import io
import os
import re
import zipfile
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="血統分析エンジン v3", layout="wide")

CONFIG = {
    "NON_STARTER_CODES": {1, 2, 3},   # 異常コードがこれなら『走っていない馬(取消・除外)』として集計から除外。他の異常(中止など)は着外として集計
    "MAX_GEN": 5,                     # 展開する血統の最大世代
    "USE_MOTHER_ROW": True,           # 母馬が血統マスターにあれば、母系の先祖もそこから補う
}

COLS = {
    'place': ['場所', '競馬場', '開催場'], 'dist': ['距離'], 'surface': ['芝・ダ', '芝ダ', '種別', '芝ダート'],
    'going': ['馬場状態'], 'horse': ['馬名'], 'rank': ['確定着順', '着順'], 'abn': ['異常コード'],
    'pop': ['人気順', '人気'], 'odds': ['単勝オッズ', 'オッズ'], 'win_pay': ['単勝配当'],
    'place_pay': ['複勝配当', '複勝'], 'field': ['頭数'], 'umaban': ['馬番'], 'waku': ['枠番', '枠'],
    'rno': ['レース番号', 'レースNo', 'レースNO', 'Ｒ', 'R番号', 'R', 'Race', 'race_no'],
    'rid': ['レースID', 'レースId', 'レースコード', 'レースキー', 'race_id', 'RaceID'],
    'rname': ['レース名', 'レース'],
    'sire': ['父馬名', '父馬', '種牡馬', '父'], 'dam': ['母馬名', '母'],
    'bms': ['母の父馬名', '母父馬名', '母父馬', '母父'],
}
BABA_FILE = "baba_info_complete.csv"
BABA_VALUE_COLS = ['クッション値', '芝_含水率_ゴール前', 'ダート_含水率_ゴール前']

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()


# ==========================================================
# ユーティリティ
# ==========================================================
def clean_name(x):
    if pd.isna(x):
        return ""
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(x)))


def ped_name(x):
    """血統の名前照合用: 括弧(国名)除去・空白除去・大文字化"""
    if pd.isna(x):
        return ""
    s = unicodedata.normalize('NFKC', str(x))
    s = re.sub(r'[\(（][^\)）]*[\)）]', '', s)
    return re.sub(r'\s+', '', s).upper()


def clean_date(d):
    if pd.isna(d):
        return "20000101"
    s = re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(d)))   # '2025.12. 7' のような空白埋めを除去
    m = re.match(r'^(\d{4})[/\-.年](\d{1,2})[/\-.月](\d{1,2})', s)
    if m:
        return f"{int(m.group(1)):04d}{int(m.group(2)):02d}{int(m.group(3)):02d}"
    m = re.match(r'^(\d{2})[/\-.](\d{1,2})[/\-.](\d{1,2})$', s)
    if m:
        return f"20{int(m.group(1)):02d}{int(m.group(2)):02d}{int(m.group(3)):02d}"
    if re.fullmatch(r'\d{5}(\.0+)?', s):                      # Excelのシリアル値
        v = int(float(s))
        if 30000 < v < 70000:
            return (pd.Timestamp('1899-12-30') + pd.Timedelta(days=v)).strftime('%Y%m%d')
    digits = re.sub(r'\D', '', s.split('.')[0])
    if len(digits) == 6:
        digits = "20" + digits
    m = re.search(r'\d{8}', digits)
    return m.group() if m else "20000101"


def map_unique(series, fn):
    u = series.unique()
    mp = {v: fn(v) for v in u}
    return series.map(mp)


def parse_int_raw(x):
    if pd.isna(x):
        return -1
    nums = re.findall(r'\d+', unicodedata.normalize('NFKC', str(x)))
    return int(nums[0]) if nums else -1


def to_money(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '', regex=False), errors='coerce').fillna(0.0)


def read_csv_bytes(data):
    last = None
    for enc in ("utf-8-sig", "cp932"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except UnicodeDecodeError as e:
            last = e
    raise last


def read_zip_csvs(path):
    dfs = []
    with zipfile.ZipFile(path, 'r') as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if not name.lower().endswith('.csv') or '__MACOSX' in name or base.startswith('.'):
                continue
            try:
                dfs.append(read_csv_bytes(z.read(name)))
            except pd.errors.EmptyDataError:
                continue
    return dfs


def file_signature():
    items = []
    for f in sorted(os.listdir(BASE_DIR)):
        if f.endswith('.zip') or f == BABA_FILE:
            p = os.path.join(BASE_DIR, f)
            items.append((f, os.path.getmtime(p), os.path.getsize(p)))
    return tuple(items)


def find_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    for n in names:                                   # 『レースID(新)』のように後ろに付記がある列名
        if len(n) >= 4:
            for col in df.columns:
                if str(col).startswith(n):
                    return col
    return None


def resolve_date(df):
    for c in df.columns:
        if '日付' in str(c):
            return map_unique(df[c], clean_date), str(c)
    if all(c in df.columns for c in ['年', '月', '日']):
        y = pd.to_numeric(df['年'], errors='coerce')
        y = y.where(y >= 100, y + 2000)
        m = pd.to_numeric(df['月'], errors='coerce')
        d = pd.to_numeric(df['日'], errors='coerce')
        s = (y.fillna(2000).astype(int) * 10000 + m.fillna(1).astype(int) * 100 + d.fillna(1).astype(int)).astype(str)
        return s, '年/月/日'
    c0 = df.columns[0]
    s = map_unique(df[c0], clean_date)
    if (s != '20000101').mean() > 0.9:
        return s, str(c0)
    raise ValueError("日付列が見つかりません（列名に『日付』を含む列、年/月/日列、または先頭列が日付である必要があります）")


def norm_surface(x):
    s = unicodedata.normalize('NFKC', str(x)).strip()
    if s.startswith('芝'):
        return '芝'
    if s.startswith('ダ'):
        return 'ダート'
    if s.startswith('障'):
        return '障害'
    return 'その他'


# ==========================================================
# 馬場CSV
# ==========================================================
def prepare_baba(df_baba):
    info = {'rows': 0, 'dup': 0, 'places': [], 'date_min': None, 'date_max': None, 'error': None}
    if df_baba is None or df_baba.empty:
        return None, info
    b = df_baba.copy()
    b.columns = [str(c).strip() for c in b.columns]
    for need in ['日付', '場所']:
        if need not in b.columns:
            info['error'] = f"馬場CSVに『{need}』列がありません（列: {list(b.columns)}）"
            return None, info
    info['rows'] = len(b)
    b['place'] = b['場所'].map(clean_name)
    b['date_str'] = map_unique(b['日付'], clean_date)
    present = [c for c in BABA_VALUE_COLS if c in b.columns]
    for c in present:
        b[c] = pd.to_numeric(b[c], errors='coerce')
    keys = ['date_str', 'place']
    info['dup'] = int(b.duplicated(subset=keys).sum())
    info['places'] = sorted(b['place'].unique().tolist())
    ok = b['date_str'] != '20000101'
    if ok.any():
        info['date_min'], info['date_max'] = b.loc[ok, 'date_str'].min(), b.loc[ok, 'date_str'].max()
    return b.groupby(keys, as_index=False)[present].mean(), info


# ==========================================================
# レースCSV → 標準形式
# ==========================================================
def prep_race_df(raw, baba, track):
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    c = {k: find_col(df, v) for k, v in COLS.items()}
    missing = [k for k in ['place', 'dist', 'horse', 'rank', 'sire', 'bms'] if c[k] is None]
    if missing:
        raise ValueError(f"必須列が見つかりません: {missing} / 実際の列: {list(df.columns)}")
    if c['odds'] is None and c['win_pay'] is None:
        raise ValueError("単勝オッズ（または単勝配当）列が見つかりません")

    info = {'raw_rows': len(df)}
    date_s, date_col = resolve_date(df)
    out = pd.DataFrame(index=df.index)
    out['date_str'] = date_s
    out['date'] = pd.to_numeric(date_s, errors='coerce').fillna(20000101).astype(int)
    info['date_fail'] = float((date_s == '20000101').mean())
    info['date_col'] = date_col
    bad = (date_s == '20000101')
    info['date_fail_samples'] = (df.loc[bad, date_col].astype(str).unique()[:5].tolist()
                                 if bad.any() and date_col in df.columns else [])

    out['place'] = map_unique(df[c['place']], clean_name)
    dist_raw = df[c['dist']]
    out['dist'] = pd.to_numeric(dist_raw, errors='coerce')
    if out['dist'].isna().mean() > 0.01:
        out['dist'] = pd.to_numeric(dist_raw.astype(str).str.extract(r'(\d{3,4})')[0], errors='coerce')
    out['surface'] = map_unique(df[c['surface']], norm_surface) if c['surface'] else track
    out['going'] = df[c['going']].astype(str).str.strip() if c['going'] else '不明'
    out['horse'] = df[c['horse']].map(clean_name)

    rr = df[c['rank']].map(parse_int_raw)
    if c['abn']:
        abn = pd.to_numeric(df[c['abn']], errors='coerce').fillna(0).astype(int)
        nonstart = abn.isin(CONFIG['NON_STARTER_CODES']) | (rr < 0)
    else:
        abn = pd.Series(0, index=df.index)
        nonstart = rr <= 0
    info['abn_counts'] = {int(k): int(v) for k, v in abn.value_counts().items() if k != 0}
    info['nonstart_rows'] = int(nonstart.sum())
    info['dnf_rows'] = int((~nonstart & (rr <= 0)).sum())          # 着順0だが出走扱い（中止等）
    out['rank'] = np.where(rr <= 0, 99, rr)

    odds = pd.to_numeric(df[c['odds']], errors='coerce') if c['odds'] else pd.Series(np.nan, index=df.index)
    out['odds'] = odds
    if c['win_pay']:
        wp = to_money(df[c['win_pay']])
        out['win_pay'] = np.where(out['rank'] == 1, wp, 0.0)
    else:
        out['win_pay'] = np.where(out['rank'] == 1, odds.fillna(0) * 100, 0.0)
    placed = out['rank'].between(1, 3)
    info['has_place_pay'] = c['place_pay'] is not None
    if c['place_pay']:
        pp = to_money(df[c['place_pay']])
        out['place_pay'] = np.where(placed, pp, 0.0)
        info['placers_zero_pay'] = int((placed & (pp <= 0) & ~nonstart).sum())
    else:
        out['place_pay'] = np.nan
        info['placers_zero_pay'] = 0

    out['pop'] = df[c['pop']].map(parse_int_raw).where(lambda s: s > 0) if c['pop'] else np.nan
    out['umaban'] = pd.to_numeric(df[c['umaban']], errors='coerce') if c['umaban'] else np.nan
    out['waku'] = pd.to_numeric(df[c['waku']], errors='coerce') if c['waku'] else np.nan
    out['field'] = pd.to_numeric(df[c['field']], errors='coerce') if c['field'] else np.nan

    # レースの区切り: ①レース番号 → ②レースID → ③レース名 → ④頭数による連続ブロック分割（最終手段）
    info['race_key_method'] = ''
    if c['rno']:
        out['race_key'] = out['date_str'] + '_' + out['place'] + '_' + df[c['rno']].astype(str).str.strip()
        info['race_key_method'] = f"日付＋場所＋『{c['rno']}』"
    elif c['rid']:
        ids = df[c['rid']].astype(str).str.strip()
        dp = out['date_str'] + '_' + out['place'] + '_'
        cand = {f"『{c['rid']}』の末尾2桁(馬番)を除いたもの": dp + ids.str[:-2],
                f"『{c['rid']}』そのもの": dp + ids}
        if c['field']:
            fld = pd.to_numeric(df[c['field']], errors='coerce')
            score = {k: float((v.groupby(v).transform('size') == fld).mean()) for k, v in cand.items()}
            info['race_key_method'] = max(score, key=score.get)       # 頭数と最も一致する方
        else:
            def plaus(v):
                sz = v.groupby(v).transform('size')
                return float(sz.between(5, 18).mean())
            score = {k: plaus(v) for k, v in cand.items()}
            best = max(score, key=score.get)
            if score[best] < 0.9:
                raise ValueError(f"『{c['rid']}』でレースを区切れません（1レース5〜18頭に収まる割合が {score[best]:.0%}）。"
                                 f" IDの値の例: {ids.head(6).tolist()} ／ 行数={len(ids)} ・ ID種類数={ids.nunique()}。"
                                 f" この例を教えてください。")
            info['race_key_method'] = best
        out['race_key'] = cand[info['race_key_method']]
    elif c['rname']:
        out['race_key'] = (out['date_str'] + '_' + out['place'] + '_' + out['dist'].astype(str) + '_'
                           + df[c['rname']].astype(str).str.strip())
        info['race_key_method'] = f"日付＋場所＋距離＋『{c['rname']}』（代用）"
    elif c['field']:
        grp = pd.DataFrame({'d': out['date_str'], 'p': out['place'], 'k': out['dist'], 'f': pd.to_numeric(df[c['field']], errors='coerce')})
        pos = grp.groupby(['d', 'p', 'k', 'f'], sort=False).cumcount()
        blk = (pos // grp['f'].clip(lower=1)).astype(int)
        out['race_key'] = grp['d'] + '_' + grp['p'] + '_' + grp['k'].astype(str) + '_' + grp['f'].astype(str) + '_' + blk.astype(str)
        info['race_key_method'] = "日付＋場所＋距離＋頭数で、頭数ごとの連続行をレースとみなす（代用。行がレース順に並んでいる前提）"
    else:
        raise ValueError("レースを区切る列が見つかりません（レース番号・レースID・レース名・頭数のいずれかが必要です）。"
                         f" 実際の列: {list(df.columns)}")

    out['sire_k'] = map_unique(df[c['sire']], ped_name)
    out['bms_k'] = map_unique(df[c['bms']], ped_name)
    out['dam_k'] = map_unique(df[c['dam']], ped_name) if c['dam'] else ""
    out['sire'] = map_unique(df[c['sire']], clean_name)
    out['bms'] = map_unique(df[c['bms']], clean_name)

    # レース分割の妥当性（頭数列との一致率）
    if c['field']:
        sz = out.groupby('race_key')['race_key'].transform('size')
        info['size_match'] = float((sz == out['field']).mean())
    else:
        info['size_match'] = None

    info['other_surface'] = int((out['surface'] != track).sum())
    keep = (~nonstart) & (out['surface'] == track)
    out = out[keep].reset_index(drop=True)

    n0 = len(out)
    if baba is not None and not baba.empty:
        out = out.merge(baba, on=['date_str', 'place'], how='left')
        if len(out) != n0:
            raise RuntimeError(f"馬場マージで行数が変化しました: {n0} → {len(out)}")
    for col in BABA_VALUE_COLS:
        if col not in out.columns:
            out[col] = np.nan
    out['baba_val'] = out['クッション値'] if track == '芝' else out['ダート_含水率_ゴール前']
    info['rows'] = len(out)
    return out, info


# ==========================================================
# 血統展開
# ==========================================================
def build_pedigree_long(base, ped_rows, max_gen, use_mother):
    """各馬の血統を『父/母/父父/…』の位置パス単位に展開し、(行, 先祖コード, 世代, 系統) の縦持ちにする"""
    tri = base[['sire_k', 'dam_k', 'bms_k']].copy()
    tid_row = tri.groupby(['sire_k', 'dam_k', 'bms_k'], sort=False).ngroup().to_numpy()
    uniq = tri.drop_duplicates().reset_index(drop=True)   # ngroup(sort=False)は初出順
    name2code, names = {}, []

    def code(n):
        v = name2code.get(n)
        if v is None:
            v = len(names)
            name2code[n] = v
            names.append(n)
        return v

    P_code, P_gen, P_side, starts, lens = [], [], [], [], []
    for sire, dam, bms in zip(uniq['sire_k'], uniq['dam_k'], uniq['bms_k']):
        paths = {}

        def put(path, name):
            if name and len(path) <= max_gen and path not in paths:
                paths[path] = name

        put('父', sire)
        for ln, anc, k in ped_rows.get(sire, ()):
            put('父' + k, anc)
        put('母', dam)
        if use_mother and dam in ped_rows:
            for ln, anc, k in ped_rows[dam]:
                put('母' + k, anc)
        put('母父', bms)
        for ln, anc, k in ped_rows.get(bms, ()):
            put('母父' + k, anc)
        starts.append(len(P_code))
        for path, name in paths.items():
            P_code.append(code(name))
            P_gen.append(len(path))
            P_side.append(0 if path[0] == '父' else 1)
        lens.append(len(paths))

    P_code = np.array(P_code, dtype=np.int32)
    P_gen = np.array(P_gen, dtype=np.int8)
    P_side = np.array(P_side, dtype=np.int8)
    starts = np.array(starts, dtype=np.int64)
    lens = np.array(lens, dtype=np.int64)

    lens_r = lens[tid_row]
    total = int(lens_r.sum())
    n = len(base)
    row_idx = np.repeat(np.arange(n, dtype=np.int32), lens_r)
    offs = np.cumsum(lens_r) - lens_r
    within = np.arange(total, dtype=np.int64) - np.repeat(offs, lens_r)
    pidx = np.repeat(starts[tid_row], lens_r) + within
    return {'row': row_idx, 'code': P_code[pidx], 'gen': P_gen[pidx], 'side': P_side[pidx],
            'names': np.array(names, dtype=object)}


@st.cache_resource(show_spinner="データ読み込み・血統展開中（初回のみ。数十秒かかることがあります）...")
def load_assets(track, sig):
    C = CONFIG
    notes = []
    files = sorted(os.listdir(BASE_DIR))
    keyword = "芝" if track == "芝" else "ダート"
    train_c, test_c, ped_files = [], [], []
    for f in files:
        nf = unicodedata.normalize('NFC', f)
        if not nf.endswith('.zip'):
            continue
        if any(k in nf for k in ['血統', 'マスタ', '種牡馬', '母父']):
            ped_files.append(f)
            continue
        if keyword in nf:
            if '学習用' in nf:
                train_c.append(f)
            elif 'テスト用' in nf:
                test_c.append(f)
    if not train_c or not test_c:
        raise FileNotFoundError(f"『{track}』の学習用/テスト用ZIPが見つかりません（探索フォルダ: {BASE_DIR}）")
    if len(train_c) > 1 or len(test_c) > 1:
        notes.append(f"学習用ZIP候補={train_c} / テスト用ZIP候補={test_c} → 名前順で最後のものを使用")
    if not ped_files:
        raise FileNotFoundError("血統マスターのZIP（名前に『血統』『マスタ』『種牡馬』『母父』を含む）が見つかりません")

    baba_path = os.path.join(BASE_DIR, BABA_FILE)
    baba, baba_info = None, {'rows': 0, 'dup': 0, 'places': [], 'date_min': None, 'date_max': None, 'error': None}
    if os.path.exists(baba_path):
        with open(baba_path, 'rb') as fh:
            baba, baba_info = prepare_baba(read_csv_bytes(fh.read()))
        if baba_info['error']:
            notes.append(baba_info['error'])
    else:
        notes.append(f"{BABA_FILE} が見つかりません。馬場による絞り込みは使えません。")

    parts, infos = [], {}
    for role, fn, is_test in (('train', train_c[-1], 0), ('test', test_c[-1], 1)):
        dfs = read_zip_csvs(os.path.join(BASE_DIR, fn))
        if not dfs:
            raise ValueError(f"{fn} にCSVがありません")
        d, info = prep_race_df(pd.concat(dfs, ignore_index=True), baba, track)
        d['is_test'] = is_test
        parts.append(d)
        infos[role] = info
        if info['date_fail'] > 0.01:
            raise ValueError(
                f"{fn} の日付列『{info['date_col']}』が {info['date_fail']:.1%} 読み取れません。"
                f"読めなかった値の例: {info['date_fail_samples']} ／ この形式を教えてください（日付が読めないと、"
                f"学習/テストの重複・馬場の突合・レースの区切りがすべて崩れます）。")
    base = pd.concat(parts, ignore_index=True)

    # 馬場帯（学習データの3分位）
    tr = base.loc[base['is_test'] == 0, 'baba_val'].dropna()
    labels = ['低', '中', '高']
    if len(tr) > 30:
        q1, q2 = tr.quantile([1 / 3, 2 / 3]).tolist()
        base['baba_band'] = np.select([base['baba_val'] <= q1, base['baba_val'] <= q2, base['baba_val'] > q2],
                                      labels, default='不明')
        base.loc[base['baba_val'].isna(), 'baba_band'] = '不明'
        baba_cuts = (q1, q2)
    else:
        base['baba_band'] = '不明'
        baba_cuts = None
    base['dist_band'] = np.select([base['dist'] < 1400, base['dist'] <= 1800, base['dist'] <= 2200, base['dist'] > 2200],
                                  ['短距離(〜1300)', 'マイル(1400-1800)', '中距離(1900-2200)', '長距離(2300〜)'], default='不明')
    base['pop_band'] = np.select([base['pop'] <= 3, base['pop'] <= 6, base['pop'] > 6],
                                 ['1-3番人気', '4-6番人気', '7番人気以下'], default='不明')

    # 血統マスター
    needed = set(base['sire_k']) | set(base['bms_k']) | set(base['dam_k'])
    needed.discard('')
    ped_rows, master_n, master_cols, maxdepth = {}, 0, [], 0
    for f in ped_files:
        for d in read_zip_csvs(os.path.join(BASE_DIR, f)):
            if '馬名' not in d.columns:
                continue
            d = d.copy()
            d.columns = [str(c).strip() for c in d.columns]
            pcols = [c for c in d.columns if re.fullmatch(r'[父母]+', c)]
            if not pcols:
                continue
            master_cols = master_cols or pcols
            maxdepth = max(maxdepth, max(len(c) for c in pcols))
            keys = d['馬名'].map(ped_name)
            master_n += len(d)
            sel = keys.isin(needed) & ~keys.isin(ped_rows.keys())
            for key, rowvals in zip(keys[sel], d.loc[sel, pcols].itertuples(index=False, name=None)):
                if key in ped_rows:
                    continue
                lst = []
                for col, v in zip(pcols, rowvals):
                    if pd.isna(v) or str(v).strip() in ('', 'nan'):
                        continue
                    lst.append((len(col), ped_name(v), col))
                ped_rows[key] = lst
    if not ped_rows:
        raise ValueError("血統マスターから、レースデータの父・母父に一致する馬が1頭も見つかりませんでした（馬名の表記を確認してください）")

    L = build_pedigree_long(base, ped_rows, C['MAX_GEN'], C['USE_MOTHER_ROW'])
    base['has_ped'] = base['sire_k'].isin(ped_rows.keys())

    B = {
        'is_test': base['is_test'].to_numpy(np.int8),
        'win': (base['rank'] == 1).to_numpy(np.int8),
        'plc': base['rank'].between(1, 3).to_numpy(np.int8),
        'wp': base['win_pay'].to_numpy(np.float64),
        'pp': base['place_pay'].to_numpy(np.float64),
    }
    has_place = bool(infos['train']['has_place_pay'] and infos['test']['has_place_pay'])

    ok_tr = base.loc[base['is_test'] == 0, 'date']
    ok_te = base.loc[base['is_test'] == 1, 'date']
    k = ['race_key', 'horse']
    overlap = len(pd.merge(base.loc[base['is_test'] == 0, k].drop_duplicates(),
                           base.loc[base['is_test'] == 1, k].drop_duplicates(), on=k))
    diag = {
        'notes': notes, 'files': {'train': train_c[-1], 'test': test_c[-1], 'pedigree': ped_files},
        'infos': infos, 'rows': {'train': int((base['is_test'] == 0).sum()), 'test': int((base['is_test'] == 1).sum())},
        'range': {'train': (int(ok_tr.min()), int(ok_tr.max())), 'test': (int(ok_te.min()), int(ok_te.max()))},
        'overlap': overlap, 'baba_info': baba_info, 'baba_cuts': baba_cuts,
        'baba_match': float(base['baba_val'].notna().mean()),
        'baba_unmatched': (base.loc[base['baba_val'].isna(), ['date_str', 'place']].value_counts().head(15)
                           .reset_index(name='頭数')),
        'race_places': sorted(base['place'].unique().tolist()),
        'ped': {'master_n': master_n, 'depth': maxdepth, 'cols': master_cols, 'rows_used': len(ped_rows),
                'sire_found': float(base['sire_k'].isin(ped_rows.keys()).mean()),
                'bms_found': float(base['bms_k'].isin(ped_rows.keys()).mean()),
                'dam_found': float(base['dam_k'].isin(ped_rows.keys()).mean()),
                'long_rows': len(L['row']), 'n_anc': len(L['names'])},
        'has_place': has_place,
        'win_check': {'odds_win_mean': float(base.loc[base['rank'] == 1, 'win_pay'].mean()),
                      'place_pay_mean': float(base.loc[base['rank'].between(1, 3), 'place_pay'].mean())
                      if has_place else float('nan')},
    }
    return {'base': base, 'B': B, 'L': L, 'diag': diag}


# ==========================================================
# 集計
# ==========================================================
@st.cache_resource(show_spinner="先祖・クロス情報を集計中...")
def anc_agg(_A, track, sig, G):
    """(馬, 先祖) 単位に集約: 最小世代・出現回数・クロス型('3×4')・父系/母系フラグ"""
    L = _A['L']
    m = L['gen'] <= G
    df = pd.DataFrame({'r': L['row'][m], 'c': L['code'][m], 'g': L['gen'][m], 's': L['side'][m]})
    g = df.groupby(['r', 'c'], sort=False).agg(gmin=('g', 'min'), npos=('g', 'size'),
                                               smin=('s', 'min'), smax=('s', 'max')).reset_index()
    dup = g.loc[g['npos'] >= 2, ['r', 'c']]
    if len(dup):
        sub = df.merge(dup, on=['r', 'c']).sort_values(['r', 'c', 'g'])
        pat = (sub.groupby(['r', 'c'], sort=False)['g']
               .agg(lambda x: '×'.join(map(str, x))).reset_index(name='pat'))
        g = g.merge(pat, on=['r', 'c'], how='left')
    else:
        g['pat'] = ''
    g['pat'] = g['pat'].fillna('')
    g['both'] = g['smin'] != g['smax']
    g['hasS'] = g['smin'] == 0
    g['hasD'] = g['smax'] == 1
    return g


def group_stats(keys, r_idx, mask, B, key_names):
    sel = mask[r_idx]
    d = keys[sel].copy()
    r = r_idx[sel]
    d['test'] = B['is_test'][r]
    d['win'] = B['win'][r]
    d['plc'] = B['plc'][r]
    d['wp'] = B['wp'][r]
    d['pp'] = B['pp'][r]
    d['pp2'] = d['pp'] ** 2
    g = d.groupby(key_names + ['test'], observed=True).agg(
        n=('win', 'size'), w=('win', 'sum'), p=('plc', 'sum'), wp=('wp', 'sum'),
        pp=('pp', 'sum'), pp2=('pp2', 'sum'), ppmax=('pp', 'max'))
    w = g.unstack('test')
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    for t in (0, 1):
        for a in ['n', 'w', 'p', 'wp', 'pp', 'pp2', 'ppmax']:
            if f"{a}_{t}" not in w.columns:
                w[f"{a}_{t}"] = 0.0
    return w.fillna(0.0).reset_index()


def metrics(w, has_place):
    out = pd.DataFrame(index=w.index)
    for t, lab in ((0, '学習'), (1, '検証')):
        n = w[f'n_{t}'].replace(0, np.nan)
        out[f'{lab}_頭数'] = w[f'n_{t}'].astype(int)
        out[f'{lab}_勝率'] = w[f'w_{t}'] / n * 100
        out[f'{lab}_複勝率'] = w[f'p_{t}'] / n * 100
        out[f'{lab}_単回値'] = w[f'wp_{t}'] / n
        if has_place:
            mean = w[f'pp_{t}'] / n
            var = (w[f'pp2_{t}'] / n - mean ** 2).clip(lower=0) * n / (n - 1)
            out[f'{lab}_複回値'] = mean
            out[f'{lab}_複回±95%'] = 1.96 * np.sqrt(var / n)
            if t == 1:
                out['検証_複回値(最高配当除外)'] = (w['pp_1'] - w['ppmax_1']) / (n - 1)
        else:
            out[f'{lab}_複回値'] = np.nan
            out[f'{lab}_複回±95%'] = np.nan
    return out


def base_row(label, mask, B, has_place, extra=None):
    m = mask if extra is None else (mask & extra)
    row = {'区分': label}
    for t, lab in ((0, '学習'), (1, '検証')):
        s = m & (B['is_test'] == t)
        n = int(s.sum())
        row[f'{lab}_頭数'] = n
        row[f'{lab}_勝率'] = B['win'][s].mean() * 100 if n else np.nan
        row[f'{lab}_複勝率'] = B['plc'][s].mean() * 100 if n else np.nan
        row[f'{lab}_単回値'] = B['wp'][s].mean() if n else np.nan
        row[f'{lab}_複回値'] = B['pp'][s].mean() if (n and has_place) else np.nan
    return row


def render_view(title, raw, name_cols, has_place, base_ref, min_n, sort_by, top_n, view_key):
    """raw: group_stats の結果。name_cols: 表示するキー列"""
    if raw.empty:
        st.warning("該当データがありません（絞り込み条件を緩めてください）。")
        return
    met = metrics(raw, has_place)
    tab = pd.concat([raw[name_cols].reset_index(drop=True), met.reset_index(drop=True)], axis=1)
    tab = tab[tab['学習_頭数'] >= min_n]
    if tab.empty:
        st.warning(f"学習期間の頭数が{min_n}頭以上の項目がありません。最小頭数を下げてください。")
        return
    st.markdown(f"##### {title}")
    # 学習上位→検証の再現性チェック
    if has_place:
        ref = tab.sort_values('学習_複回値', ascending=False).head(20)
        te_n = ref['検証_頭数'].sum()
        te_ret = (ref['検証_複回値'] * ref['検証_頭数']).sum() / te_n if te_n else np.nan
        tr_ret = (ref['学習_複回値'] * ref['学習_頭数']).sum() / ref['学習_頭数'].sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("学習期間の複回値が高い上位20項目：学習期間の平均", f"{tr_ret:.1f}%")
        c2.metric("同じ20項目の検証期間の平均（再現性）", f"{te_ret:.1f}%" if te_n else "-")
        c3.metric("検証期間の全体ベースライン", f"{base_ref:.1f}%" if base_ref == base_ref else "-")
        st.caption("学習期間で高かった項目が検証期間でベースラインを上回らない場合、その高回収率は偶然（過学習）の可能性が高いです。"
                   "（1頭が複数の先祖に重複して数えられるため、合計は参考値です）")
    tab = tab.sort_values(sort_by, ascending=False).head(top_n).reset_index(drop=True)
    fmt = {c: '{:.1f}' for c in tab.columns if c not in name_cols and not c.endswith('頭数')}
    st.dataframe(tab.style.format(fmt, na_rep='-'))
    st.download_button("📥 CSVダウンロード", tab.to_csv(index=False, encoding='utf-8-sig'),
                       file_name=f"pedigree_{view_key}.csv", mime="text/csv", key=f"dl_{view_key}")


# ==========================================================
# 診断
# ==========================================================
def show_diagnostics(A, track):
    d = A['diag']
    st.markdown("### 🩺 データ診断")
    for n in d['notes']:
        st.warning(n)
    c1, c2, c3 = st.columns(3)
    c1.metric("学習データ(出走馬)", f"{d['rows']['train']:,}")
    c2.metric("テストデータ(出走馬)", f"{d['rows']['test']:,}")
    c3.metric("学習⇔テスト重複(リーク)", f"{d['overlap']:,}")
    st.write(f"学習期間: {d['range']['train'][0]} 〜 {d['range']['train'][1]} ／ テスト期間: {d['range']['test'][0]} 〜 {d['range']['test'][1]}")
    st.caption(f"使用ファイル: 学習={d['files']['train']} / テスト={d['files']['test']} / 血統マスター={d['files']['pedigree']}")
    if d['overlap'] > 0:
        st.error("🚨 学習とテストに同一レースの同一馬が含まれています。")
    if d['range']['train'][1] >= d['range']['test'][0]:
        st.error("🚨 学習期間の終了がテスト期間の開始以降です。時系列分割になっていません。")

    st.markdown("**📄 CSVの読み取り結果**")
    for role, lab in (('train', '学習'), ('test', 'テスト')):
        i = d['infos'][role]
        sm = f"{i['size_match']:.1%}" if i['size_match'] is not None else "頭数列なし"
        st.write(f"{lab}: レースの区切り方=【{i['race_key_method']}】")
        st.write(f"{lab}: 元{i['raw_rows']:,}行 → 集計対象{i['rows']:,}行 ／ 日付列=『{i['date_col']}』(失敗率{i['date_fail']:.1%}) ／ "
                 f"除外(取消・除外等){i['nonstart_rows']:,}行 異常コード内訳={i['abn_counts']} ／ "
                 f"着順0の出走扱い(中止等){i['dnf_rows']:,}行 ／ 他路面・障害で除外{i['other_surface']:,}行 ／ "
                 f"レース分割の頭数一致率={sm}")
        if i['date_fail'] > 0:
            st.error(f"🚨 {lab}: 日付が読めない行があります。")
        if i['size_match'] is not None and i['size_match'] < 0.95:
            st.error(f"🚨 {lab}: レース分割が頭数と合いません（レース番号列の確認が必要です）。")
        if not i['has_place_pay']:
            st.error(f"🚨 {lab}: 『複勝』『複勝配当』列がありません。複勝回収率は計算できません。")
        elif i['placers_zero_pay'] > 0:
            st.warning(f"{lab}: 3着以内なのに複勝配当が0の馬が {i['placers_zero_pay']:,} 頭あります（配当の欠落の可能性）。")

    st.markdown("**💴 配当の妥当性**")
    w = d['win_check']
    st.write(f"勝ち馬の単勝配当 平均={w['odds_win_mean']:.0f}円 ／ 3着以内の複勝配当 平均={w['place_pay_mean']:.0f}円")
    if w['odds_win_mean'] < 150:
        st.error("🚨 単勝配当の平均が100円台前半です。オッズの単位（倍率）や列の取り違えの可能性があります。")

    st.markdown("**🌦 馬場データの突合（日付＋場所）**")
    bi = d['baba_info']
    m1, m2, m3 = st.columns(3)
    m1.metric("馬場値の一致率(全体)", f"{d['baba_match']:.1%}")
    m2.metric("馬場CSV行数", f"{bi['rows']:,}")
    m3.metric("馬場CSVのキー重複(平均化済)", f"{bi['dup']:,}")
    if bi['rows']:
        st.write(f"馬場CSV期間: {bi['date_min']} 〜 {bi['date_max']} ／ 馬場CSVの場所: {bi['places']} ／ レース側の場所: {d['race_places']}")
        only_race = sorted(set(d['race_places']) - set(bi['places']))
        if only_race:
            st.warning(f"馬場CSVに存在しない場所（表記ゆれの可能性）: {only_race}")
        if d['baba_match'] < 0.8:
            st.error("🚨 馬場の一致率が低いです。下の『一致しなかった日付・場所』を確認してください。")
            st.dataframe(d['baba_unmatched'])
    if d['baba_cuts']:
        lab = 'クッション値' if track == '芝' else 'ダート含水率(ゴール前)'
        st.caption(f"馬場帯の区切り（学習データの3分位）: {lab} ≦{d['baba_cuts'][0]:.2f} →低 / ≦{d['baba_cuts'][1]:.2f} →中 / それ超 →高")

    st.markdown("**🧬 血統マスターの突合**")
    p = d['ped']
    st.write(f"マスター {p['master_n']:,}頭 ／ 認識した血統列: {p['cols'][:12]}{'…' if len(p['cols']) > 12 else ''}（最大 {p['depth']} 文字＝{p['depth'] + 1}代まで）")
    c1, c2, c3 = st.columns(3)
    c1.metric("父がマスターに存在", f"{p['sire_found']:.1%}")
    c2.metric("母父がマスターに存在", f"{p['bms_found']:.1%}")
    c3.metric("母がマスターに存在", f"{p['dam_found']:.1%}")
    st.write(f"展開した血統の行数: {p['long_rows']:,} ／ 先祖の種類: {p['n_anc']:,}")
    if p['sire_found'] < 0.9:
        st.error("🚨 父の一致率が低いです。馬名の表記ゆれ（国名・カナ表記）を確認してください。")


# ==========================================================
# 画面
# ==========================================================
st.title("🧬 血統分析エンジン v3：先祖内包・クロス別の的中率／回収率")
st.write("血統マスターから各馬の血統を展開し、**どの先祖を持つか・何代前でクロスするか**ごとに、"
         "学習期間と検証期間の成績を並べて表示します。")

track = st.sidebar.radio("📁 データベース", ["芝", "ダート"])
try:
    A = load_assets(track, file_signature())
except Exception as e:
    st.error("データ読み込みでエラーが発生しました。")
    st.exception(e)
    st.stop()

base, B, diag = A['base'], A['B'], A['diag']
has_place = diag['has_place']

st.sidebar.markdown("---")
st.sidebar.markdown("**絞り込み（両期間に適用）**")
f_place = st.sidebar.multiselect("場所", sorted(base['place'].unique()), default=sorted(base['place'].unique()))
f_dist = st.sidebar.multiselect("距離帯", sorted(base['dist_band'].unique()), default=sorted(base['dist_band'].unique()))
f_going = st.sidebar.multiselect("馬場状態", sorted(base['going'].unique()), default=sorted(base['going'].unique()))
f_baba = st.sidebar.multiselect(("クッション値帯" if track == '芝' else "含水率帯") + "（学習データの3分位）",
                                sorted(base['baba_band'].unique()), default=sorted(base['baba_band'].unique()))
f_pop = st.sidebar.multiselect("人気帯", sorted(base['pop_band'].unique()), default=sorted(base['pop_band'].unique()))
mask = (base['place'].isin(f_place) & base['dist_band'].isin(f_dist) & base['going'].isin(f_going)
        & base['baba_band'].isin(f_baba) & base['pop_band'].isin(f_pop)).to_numpy()

st.sidebar.markdown("---")
G = st.sidebar.slider("先祖を探す世代の上限", 3, CONFIG['MAX_GEN'], 4)
min_n = st.sidebar.slider("表示する最小頭数（学習期間）", 30, 1000, 150, step=10)
top_n = st.sidebar.slider("表示行数", 20, 500, 100, step=10)
sort_by = st.sidebar.selectbox("並べ替え", ['学習_複回値', '学習_単回値', '学習_複勝率', '学習_頭数', '検証_複回値', '検証_複勝率'])
if not has_place and sort_by.endswith('複回値'):
    sort_by = '学習_複勝率'

tab_diag, tab_anc, tab_cross, tab_anccross, tab_sire = st.tabs(
    ["🩺 診断", "先祖別（内包する馬）", "クロス型別", "先祖×クロス型", "父・母父・ニックス"])

with tab_diag:
    show_diagnostics(A, track)

N = len(base)
all_rows = np.arange(N, dtype=np.int32)
with tab_diag:
    st.markdown("**📏 現在の絞り込み条件でのベースライン**")
    rows_ = [base_row("全頭均等買い", mask, B, has_place),
             base_row("1番人気", mask, B, has_place, extra=(base['pop'] == 1).to_numpy())]
    bt = pd.DataFrame(rows_)
    st.dataframe(bt.style.format({c: '{:.1f}' for c in bt.columns if c != '区分' and not c.endswith('頭数')}, na_rep='-'))
te_all = mask & (B['is_test'] == 1)
base_ref = (B['pp'][te_all].mean() if has_place and te_all.any() else float('nan'))

ra = anc_agg(A, track, file_signature(), G)
names = A['L']['names']

with tab_anc:
    c1, c2 = st.columns(2)
    side = c1.radio("先祖がいる系統", ["どちらでも", "父系", "母系"], horizontal=True)
    only_cross = c2.checkbox("クロス(同じ先祖が2回以上)を持つ場合だけ", value=False)
    r = ra
    if side == "父系":
        r = r[r['hasS']]
    elif side == "母系":
        r = r[r['hasD']]
    if only_cross:
        r = r[r['npos'] >= 2]
    keys = pd.DataFrame({'先祖コード': r['c'].to_numpy()})
    raw = group_stats(keys, r['r'].to_numpy(), mask, B, ['先祖コード'])
    raw['先祖'] = names[raw['先祖コード'].to_numpy()]
    render_view(f"先祖別（{G}代以内）", raw, ['先祖'], has_place, base_ref, min_n, sort_by, top_n, "ancestor")

with tab_cross:
    c1, c2 = st.columns(2)
    scope = c1.radio("クロスの種類", ["すべて", "父系×母系にまたがる", "片系のみ"], horizontal=True)
    r = ra[ra['npos'] >= 2]
    if scope == "父系×母系にまたがる":
        r = r[r['both']]
    elif scope == "片系のみ":
        r = r[~r['both']]
    keys = pd.DataFrame({'クロス型': r['pat'].to_numpy()})
    r_all = r['r'].to_numpy()
    # クロスなし
    covered = base['has_ped'].to_numpy()
    cross_rows = np.unique(ra.loc[ra['npos'] >= 2, 'r'].to_numpy())
    nocross = np.setdiff1d(all_rows[covered], cross_rows)
    keys_all = pd.concat([keys, pd.DataFrame({'クロス型': ['（クロスなし）'] * len(nocross)})], ignore_index=True)
    r_all = np.concatenate([r_all, nocross.astype(np.int32)])
    raw = group_stats(keys_all, r_all, mask, B, ['クロス型'])
    render_view(f"クロス型別（{G}代以内。『3×4』＝同じ先祖が3代前と4代前にいる）", raw, ['クロス型'],
                has_place, base_ref, min(min_n, 100), sort_by, top_n, "crosspattern")

with tab_anccross:
    r = ra[ra['npos'] >= 2]
    keys = pd.DataFrame({'先祖コード': r['c'].to_numpy(), 'クロス型': r['pat'].to_numpy()})
    raw = group_stats(keys, r['r'].to_numpy(), mask, B, ['先祖コード', 'クロス型'])
    raw['先祖'] = names[raw['先祖コード'].to_numpy()]
    render_view(f"先祖×クロス型（{G}代以内）", raw, ['先祖', 'クロス型'], has_place, base_ref,
                min(min_n, 60), sort_by, top_n, "ancestor_cross")

with tab_sire:
    which = st.radio("単位", ["父", "母父", "父×母父（ニックス）"], horizontal=True)
    if which == "父":
        keys = pd.DataFrame({'父': base['sire'].to_numpy()})
        cols_ = ['父']
    elif which == "母父":
        keys = pd.DataFrame({'母父': base['bms'].to_numpy()})
        cols_ = ['母父']
    else:
        keys = pd.DataFrame({'父': base['sire'].to_numpy(), '母父': base['bms'].to_numpy()})
        cols_ = ['父', '母父']
    raw = group_stats(keys, all_rows, mask, B, cols_)
    render_view(which, raw, cols_, has_place, base_ref, min_n, sort_by, top_n, "sire_" + which)