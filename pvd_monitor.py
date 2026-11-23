"""
PVD 모니터링 분석 모듈 (6단계)

pvd_engine.Engine 위에서 대시보드에 들어갈 숫자를 계산한다. 모델 재학습은 하지 않는다.
(롤링 재학습은 rolling.py가 따로 돌리고, 여기서는 그 결과 CSV를 읽기만 한다.)

1. load_rolling()          rolling.py 결과(월별 재학습 → 다음 달 추천)를 읽어 월별 요약
2. quality_table()         규격(목표 ± 반폭)을 가정하고 현재 / 추천 적용 / 보수적 추천의 Cpk·규격 내 비율
3. cv_policy()             챔버별 타깃 사용량-CV 곡선 → 교체 제안 시점, 최근 CV로 점검 우선순위
4. lot_view() / chamber_status()   로트·챔버를 고르면 추천 증착 시간, 예측 두께·CV, 챔버 경고

'보수적 추천'의 뜻
  엔진의 기본 추천은 현재 예측(pred_now, 그 웨이퍼까지 학습한 모델)으로 Δ를 고른다. 실제 운영에서는
  처리 전 웨이퍼의 두께를 모르므로, 그 웨이퍼를 학습에 쓰지 않은 예측으로 Δ를 골라야 한다.
  - 'oof'     : 로트 교차검증 예측(*_oof)으로 Δ를 고름 (시간 순서는 무시)
  - 'rolling' : rolling.py 결과. 과거 데이터로만 학습한 모델로 Δ를 고름 (시간 순서까지 지킴, 가장 엄격)
  두 경우 모두 평가는 '실측값 + 보정 효과'로 같다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pvd_engine import Engine, GRID, THK, CV, _natural

ROLLING_FILE = 'rolling_wafers.csv'


# ============================================================ 공통
def eq_order(names):
    return sorted(names, key=_natural)


def recommend_from(E: Engine, base_thk: np.ndarray, base_cv: np.ndarray, target: float, mask=None):
    """엔진과 같은 목적함수로 Δ를 고르되, 기준 예측을 바꿔 넣을 수 있게 한 버전
    base_thk, base_cv: 현재 증착 시간에서의 예측 두께·CV (웨이퍼 수,)
    반환: x1_rec, shift(dict y→배열)"""
    m = E.meta
    n = len(E.w)
    mask = np.ones(n, bool) if mask is None else np.asarray(mask)
    G = np.broadcast_to(GRID, (n, len(GRID)))
    r_now = {y: E.rsm_wafer(y, E.x1_now) for y in E.ys}
    r_grid = {y: E.rsm_wafer(y, G) for y in (THK, CV)}
    t_hat = base_thk[:, None] + r_grid[THK] - r_now[THK][:, None]
    cv_hat = base_cv[:, None] + r_grid[CV] - r_now[CV][:, None]
    f = m['weights']['w1'] * np.abs(t_hat - target) / m['y_std'][THK] + m['weights']['w2'] * cv_hat / m['y_std'][CV]
    x1 = GRID[f.argmin(axis=1)]
    x1 = np.where(mask, x1, E.x1_now)
    shift = {y: E.rsm_wafer(y, x1) - r_now[y] for y in E.ys}
    return x1, shift


# ============================================================ 1. 롤링 재학습 결과
def load_rolling(E: Engine):
    """rolling.py가 만든 pvd_model/data/rolling_wafers.csv를 읽는다. 없으면 None"""
    f = E.dir / 'data' / ROLLING_FILE
    if not f.exists():
        return None
    r = pd.read_csv(f)
    return r


def rolling_monthly(E: Engine, r: pd.DataFrame, tol: float = 1.0) -> pd.DataFrame:
    """월별 '현재 / 고정 모델 추천 / 롤링 재학습 추천'의 ±tol 이내 비율과 두께 오차"""
    rows = []
    for (month, mode), g in r.groupby(['month', 'mode']):
        t = g['thk_true'].values
        tgt = g['target'].values
        a = t + g['shift_thk'].values
        rows.append({'월': month, '방식': mode, '웨이퍼 수': len(g),
                     '현재 ±tol': (np.abs(t - tgt) <= tol).mean(), '추천 ±tol': (np.abs(a - tgt) <= tol).mean(),
                     '현재 오차': np.abs(t - tgt).mean(), '추천 오차': np.abs(a - tgt).mean(),
                     '예측 오차(MAE)': np.abs(g['pred_thk'].values - t).mean()})
    return pd.DataFrame(rows)


def rolling_summary(E: Engine, r: pd.DataFrame, tol: float = 1.0) -> pd.DataFrame:
    rows = {}
    first = r[r['mode'] == r['mode'].iloc[0]]
    t, tgt = first['thk_true'].values, first['target'].values
    rows['현재'] = {'웨이퍼 수': len(first), '두께 오차': np.abs(t - tgt).mean(), f'±{tol:g} 이내': (np.abs(t - tgt) <= tol).mean(),
                  '예측 오차(MAE)': np.nan}
    for mode, g in r.groupby('mode'):
        a = g['thk_true'].values + g['shift_thk'].values
        tg = g['target'].values
        rows[mode] = {'웨이퍼 수': len(g), '두께 오차': np.abs(a - tg).mean(), f'±{tol:g} 이내': (np.abs(a - tg) <= tol).mean(),
                      '예측 오차(MAE)': np.abs(g['pred_thk'].values - g['thk_true'].values).mean()}
    return pd.DataFrame(rows).T


# ============================================================ 2. 품질 지표
def capability(x: np.ndarray, lsl: float, usl: float) -> dict:
    x = np.asarray(x, float)
    mu, sd = x.mean(), x.std(ddof=1)
    cp = (usl - lsl) / (6 * sd)
    cpk = min(usl - mu, mu - lsl) / (3 * sd)
    inside = ((x >= lsl) & (x <= usl)).mean()
    return {'평균': mu, '표준편차': sd, 'Cp': cp, 'Cpk': cpk, '규격 내 비율': inside,
            '규격 밖(장)': int(((x < lsl) | (x > usl)).sum()), '규격 밖 ppm': (1 - inside) * 1e6}


def scenario_thickness(E: Engine, target: float, mask=None) -> dict:
    """규격 평가에 쓰는 웨이퍼별 두께: 현재 / 추천 적용(엔진 기본) / 보수적(교차검증 예측으로 Δ 선택)
    + (있으면) 롤링 재학습 추천. 각 값 = 실측값 + 보정 효과"""
    w = E.w
    mask = np.ones(len(w), bool) if mask is None else np.asarray(mask)
    t_true = w[f'{THK}_true'].values
    rec = E.recommend(target)
    _, sh_oof = recommend_from(E, w[f'{THK}_oof'].values, w[f'{CV}_oof'].values, target)
    out = {'현재': t_true[mask],
           '추천 적용': (t_true + rec.shift[THK])[mask],
           '추천 적용(보수적)': (t_true + sh_oof[THK])[mask]}
    return out


def quality_table(E: Engine, target: float, half: float = 2.0, mask=None, rolling: pd.DataFrame | None = None):
    """규격 = 목표 ± half. 현재 / 추천 적용 / 보수적 추천(교차검증) / 롤링 재학습(있으면, 2024-01~ 구간)"""
    lsl, usl = target - half, target + half
    th = scenario_thickness(E, target, mask)
    rows = {k: capability(v, lsl, usl) for k, v in th.items()}
    tb = pd.DataFrame(rows).T
    if rolling is not None and abs(rolling['target'].iloc[0] - target) < 1e-9 and mask is None:
        g = rolling[rolling['mode'] == '롤링 재학습']
        idx = g['row'].values
        sub = pd.DataFrame({'현재(롤링 구간)': capability(g['thk_true'].values, lsl, usl),
                            '롤링 재학습 추천': capability(g['thk_true'].values + g['shift_thk'].values, lsl, usl)}).T
        tb = pd.concat([tb, sub])
    return tb


def quality_by_chamber(E: Engine, target: float, half: float = 2.0) -> pd.DataFrame:
    lsl, usl = target - half, target + half
    th = scenario_thickness(E, target)
    rows = []
    for e in eq_order(pd.unique(E.eq_short)):
        mk = E.eq_short == e
        r = {'챔버': e, '웨이퍼 수': int(mk.sum())}
        for k, v in th.items():
            c = capability(v[mk], lsl, usl)
            r[f'{k} Cpk'] = c['Cpk']
            r[f'{k} 규격 내'] = c['규격 내 비율']
        rows.append(r)
    return pd.DataFrame(rows).set_index('챔버')


# ============================================================ 3. CV 관리 정책
def cv_curve(E: Engine, eq: str, n: int = 41) -> pd.DataFrame:
    """챔버 eq(equipment_N)의 RSM 기준 타깃 사용량-CV 곡선 (보정량은 그 챔버의 평균 현재 보정량에 고정)"""
    lr = E.meta['life_range'][eq]
    x2 = np.linspace(-1, 1, n)
    x1 = np.full(n, E.x1_now[E.w['equipment'].values == eq].mean())
    cv = E.rsm_eq(eq, CV, x1, x2)
    return pd.DataFrame({'타깃 사용량': lr['center'] + x2 * lr['half'], 'x2': x2, 'RSM CV': cv})


def cv_actual_bins(E: Engine, edges=(0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500)) -> pd.DataFrame:
    w = E.w
    b = pd.cut(w[E.life_col], list(edges), include_lowest=True)
    mid = b.apply(lambda iv: iv.mid).astype(float)
    df = pd.DataFrame({'챔버': E.eq_short, '구간 중앙': mid, 'CV': w[f'{CV}_true'].values})
    g = df.groupby(['챔버', '구간 중앙']).CV.agg(['mean', 'count']).reset_index()
    g.columns = ['챔버', '타깃 사용량', '실측 CV 평균', '웨이퍼 수']
    return g


def cv_policy(E: Engine, cv_limit: float = 3.0, rise: float = 0.3, recent_months: int = 6) -> pd.DataFrame:
    """챔버별 교체 제안 시점과 점검 우선순위
    교체 제안 시점: RSM CV 곡선이 최저점을 지난 뒤 (최저 + rise) 또는 cv_limit 중 먼저 넘는 타깃 사용량
                    관측 범위(타깃 사용량 5~95%) 안에서 안 넘으면 '범위 내 초과 없음'
    점검 우선순위: 최근 recent_months개월 실측 CV 평균이 높은 순 (기준 초과 비율·추세 함께 표시)"""
    w = E.w
    t = pd.to_datetime(w['runstart'])
    cut = t.max() - pd.DateOffset(months=recent_months)
    prev = cut - pd.DateOffset(months=12)
    cv_true = w[f'{CV}_true'].values
    life_now_max = w.groupby('equipment')[E.life_col].quantile(0.98)
    rows = []
    for e in E.eqs:
        cur = cv_curve(E, e, n=201)
        i0 = int(cur['RSM CV'].values.argmin())
        tail = cur.iloc[i0:]
        cv_min = cur['RSM CV'].iloc[i0]
        thr = min(cv_limit, cv_min + rise)
        hit = tail[tail['RSM CV'] >= thr]
        if cv_min >= cv_limit:
            # 새 타깃이어도 기준을 넘음 → 타깃 교체로는 개선되지 않는 챔버 요인
            life_rep, why = np.nan, '교체로 개선 안 됨 (챔버 점검)'
        elif len(hit):
            life_rep = float(hit['타깃 사용량'].iloc[0])
            why = '절대 한계' if thr == cv_limit else '최저 대비 상승'
        else:
            life_rep, why = np.nan, '범위 내 초과 없음'
        mk = w['equipment'].values == e
        rec = mk & (t >= cut).values
        old = mk & (t >= prev).values & (t < cut).values
        rows.append({'챔버': e.replace('equipment_', 'eq'),
                     'CV 최저(RSM)': cur['RSM CV'].iloc[i0], '최저 지점 타깃 사용량': cur['타깃 사용량'].iloc[i0],
                     '교체 기준 CV': thr, '교체 제안 타깃 사용량': life_rep, '판단 근거': why,
                     '현재 교체 시점(실측 98%)': float(life_now_max[e]),
                     '최근 CV 평균': cv_true[rec].mean() if rec.any() else np.nan,
                     '이전 CV 평균': cv_true[old].mean() if old.any() else np.nan,
                     '최근 기준 초과 비율': (cv_true[rec] > cv_limit).mean() if rec.any() else np.nan,
                     '최근 웨이퍼 수': int(rec.sum())})
    df = pd.DataFrame(rows)
    df['추세(최근−이전)'] = df['최근 CV 평균'] - df['이전 CV 평균']
    df['최근 가동'] = np.where(df['최근 웨이퍼 수'] > 0, '가동', '최근 가동 없음')
    df['앞당김'] = df['현재 교체 시점(실측 98%)'] - df['교체 제안 타깃 사용량']
    df = df.sort_values('최근 CV 평균', ascending=False).reset_index(drop=True)
    df.insert(0, '점검 우선순위', np.arange(1, len(df) + 1))
    return df.set_index('챔버')


# ============================================================ 4. 로트·챔버 모니터링
def chamber_status(E: Engine, target: float, tol: float = 1.0, cv_limit: float = 3.0, rise: float = 0.3,
                   top_k: int = 3, policy: pd.DataFrame | None = None) -> pd.DataFrame:
    """챔버별 경고. 평소 '이상 없음', 경고가 하나라도 있으면 '조치 필요'
    - 목표 달성 불가 (보정만으로 목표 ±tol에 닿는 웨이퍼 절반 미만)
    - CV 점검 우선순위 상위 top_k 이면서 최근 CV 평균이 기준 초과
    - 마지막 웨이퍼의 타깃 사용량이 교체 제안 시점을 넘음"""
    pol = cv_policy(E, cv_limit, rise) if policy is None else policy
    fz = E.feasibility(target, tol)['by_chamber']
    w = E.w
    last = w.assign(_t=pd.to_datetime(w['runstart'])).sort_values('_t').groupby('equipment').tail(1)
    last_life = dict(zip(last['equipment'].str.replace('equipment_', 'eq'), last[E.life_col]))
    rows = []
    for c in eq_order(pol.index):
        p = pol.loc[c]
        warns = []
        if fz.loc[c, '판정'] == '달성 불가':
            warns.append(f'목표 {target:.1f} 달성 불가 (도달 가능 {fz.loc[c, "도달가능비율"]:.0%}) → 로트를 다른 챔버로 배정')
        if p['점검 우선순위'] <= top_k and p['최근 CV 평균'] > cv_limit:
            warns.append(f'CV 점검 필요 (최근 평균 {p["최근 CV 평균"]:.2f} > 기준 {cv_limit:g}, 우선순위 {int(p["점검 우선순위"])}위)')
        if p['판단 근거'].startswith('교체로 개선 안 됨') and not (p['점검 우선순위'] <= top_k and p['최근 CV 평균'] > cv_limit):
            warns.append(f'새 타깃에서도 CV {p["CV 최저(RSM)"]:.2f} ≥ 기준 {cv_limit:g} → 챔버 점검')
        if np.isfinite(p['교체 제안 타깃 사용량']) and last_life[c] >= p['교체 제안 타깃 사용량']:
            warns.append(f'타깃 교체 권장 (사용량 {last_life[c]:.0f} ≥ 제안 {p["교체 제안 타깃 사용량"]:.0f})')
        rows.append({'챔버': c, '상태': '조치 필요' if warns else '이상 없음', '경고': ' / '.join(warns),
                     '최근 CV 평균': p['최근 CV 평균'], '점검 우선순위': int(p['점검 우선순위']),
                     '마지막 타깃 사용량': last_life[c], '교체 제안 타깃 사용량': p['교체 제안 타깃 사용량'],
                     '도달 가능 비율': fz.loc[c, '도달가능비율']})
    return pd.DataFrame(rows).set_index('챔버')


def wafer_table(E: Engine, target: float) -> pd.DataFrame:
    """웨이퍼별 모니터링 표 (모든 로트). 예측은 그 웨이퍼를 학습하지 않은 교차검증 예측 기준"""
    w = E.w
    m = E.meta
    x1, sh = recommend_from(E, w[f'{THK}_oof'].values, w[f'{CV}_oof'].values, target)
    d_rec = E.dc + x1 * E.dh
    t_now = w[E.tdep_col].values
    return pd.DataFrame({
        'row': w['row'].values, '로트': w['lot'].values, '챔버': E.eq_short, '웨이퍼': w['wafer_id'].values,
        '처리 시각': w['runstart'].values, '타깃 사용량': w[E.life_col].values,
        '현재 증착 시간': t_now, '추천 증착 시간': t_now + (d_rec - w['delta_now'].values),
        '보정량 Δ(추천−현재)': d_rec - w['delta_now'].values, '범위 끝': np.abs(x1) > 0.99,
        '예측 두께(현재)': w[f'{THK}_oof'].values, '예측 두께(추천)': w[f'{THK}_oof'].values + sh[THK],
        '예측 CV(현재)': w[f'{CV}_oof'].values, '예측 CV(추천)': w[f'{CV}_oof'].values + sh[CV],
        '실측 두께': w[f'{THK}_true'].values, '실측+보정 두께': w[f'{THK}_true'].values + sh[THK],
        '실측 CV': w[f'{CV}_true'].values,
    })


def lot_view(E: Engine, target: float, lot: str, chamber: str | None = None, table: pd.DataFrame | None = None):
    """로트(와 챔버)를 고르면 해당 웨이퍼들의 추천 증착 시간·예측 두께·CV"""
    tb = wafer_table(E, target) if table is None else table
    g = tb[tb['로트'] == lot]
    if chamber is not None:
        g = g[g['챔버'] == chamber]
    return g
