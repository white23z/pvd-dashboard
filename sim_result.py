"""'목표 두께 변경 시뮬레이션' 화면에 넣을 값 계산 (pvd_sim + 엔진)

target_change(E, sim, target2, day, tol) → dict (JSON으로 화면에 그대로 넘김)
- KPI·일별·챔버별: SimPy 가상 공정 30일 (pvd_sim, '목표 두께 변경' 시나리오와 같은 방식)
- '목표를 어디까지 바꿀 수 있나': 엔진 RSM 기준 evaluate (목표 149.5~156, 0.5 간격)
- 적용 가능 챔버: 엔진 feasibility (도달 가능 웨이퍼 50% 이상)
"""
import numpy as np

from pvd_sim import Scenario, GRID

TARGET = 152.2                       # 현재 목표
SWEEP = [round(149.5 + 0.5 * k, 1) for k in range(14)]
CV_MAIN = 0.64                       # 152.2 교차검증 ±1 이내 (메인 숫자) → 그래프의 파란 구간 기준


def short(e):
    return 'eq' + str(e).split('_')[-1]


def sweep(E, tol):
    cur, rec = [], []
    for t in SWEEP:
        s = E.evaluate(t, tol)['summary']
        row = [i for i in s.index if '이내' in i][0]
        cur.append(float(s.loc[row, '현재']))
        rec.append(float(s.loc[row, '추천 적용']))
    return {'t': SWEEP, 'cur': cur, 'rec': rec}


def rec_at(E, target, tol):
    s = E.evaluate(target, tol)['summary']
    row = [i for i in s.index if '이내' in i][0]
    return float(s.loc[row, '추천 적용'])


def target_change(E, sim, target2, day, tol, sweep_data=None):
    sc = Scenario(name='목표 두께 변경', target=TARGET, target2=target2, target2_day=day, tol=tol)
    out = sim.run(sc)
    log = out['log']
    log = log.assign(eq=log['챔버'].map(short))
    after, before = log[log['시각(일)'] >= day], log[log['시각(일)'] < day]
    er = (after['두께_추천'] - after['목표']).abs()
    ec = (after['두께_현재'] - after['목표']).abs()

    order = sorted(log['eq'].unique(), key=lambda s: int(s[2:]))
    feas = E.feasibility(target2, tol)
    bc = feas['by_chamber']
    bad = [e for e in order if e in bc.index and bc.loc[e, '판정'] != '가능']

    chambers = []
    for e in order:
        a, b = after[after['eq'] == e], before[before['eq'] == e]
        chambers.append({
            'name': e, 'ok': e not in bad,
            'dB': float(b['보정량Δ'].mean()) if len(b) else None,
            'dA': float(a['보정량Δ'].mean()) if len(a) else None,
            'cur': float(((a['두께_현재'] - a['목표']).abs() <= tol).mean()) if len(a) else None,
            'rec': float(((a['두께_추천'] - a['목표']).abs() <= tol).mean()) if len(a) else None,
        })

    d = log.assign(일=np.floor(log['시각(일)']).astype(int))
    daily = []
    for k, g in d.groupby('일'):
        daily.append({'d': int(k), 'cur': float(g['두께_현재'].mean()), 'rec': float(g['두께_추천'].mean()),
                      'cin': float(((g['두께_현재'] - g['목표']).abs() <= tol).mean()),
                      'rin': float(((g['두께_추천'] - g['목표']).abs() <= tol).mean())})

    # 하루씩 재생용: (일, 챔버)별 합계 [일, 챔버 번호, 장 수, 그대로 ±tol 안, 보정 ±tol 안, 그대로 두께 합, 보정 두께 합, Δ 합]
    d = d.assign(_c=(d['두께_현재'] - d['목표']).abs() <= tol, _r=(d['두께_추천'] - d['목표']).abs() <= tol,
                 _e=d['eq'].map({e: k for k, e in enumerate(order)}))
    cells = [[int(k[0]), int(k[1]), int(len(g)), int(g['_c'].sum()), int(g['_r'].sum()),
              round(float(g['두께_현재'].sum()), 4), round(float(g['두께_추천'].sum()), 4), round(float(g['보정량Δ'].sum()), 5)]
             for k, g in d.groupby(['일', '_e'])]

    dB = np.nanmean([c['dB'] for c in chambers if c['dB'] is not None])
    dA = np.nanmean([c['dA'] for c in chambers if c['dA'] is not None])
    lo, hi = feas['observed_range']
    return {
        'req': {'target': TARGET, 'target2': target2, 'day': day, 'tol': tol},
        'days': sc.days,
        'n_after': int(len(after)),
        'rec_in': float((er <= tol).mean()), 'cur_in': float((ec <= tol).mean()),
        'rec_out': int((er > tol).sum()), 'cur_out': int((ec > tol).sum()),
        'mean_after': float(after['두께_추천'].mean()),
        'd_before': float(dB), 'd_after': float(dA),
        'feas_ok': len(order) - len(bad), 'feas_total': len(order), 'feas_bad': bad,
        'observed': [float(lo), float(hi)], 'in_obs': bool(lo <= target2 <= hi),
        'drange': [float(E.dc + GRID.min() * E.dh), float(E.dc + GRID.max() * E.dh)],
        'chambers': chambers, 'daily': daily, 'eqs': order, 'cells': cells,
        'sweep': sweep_data if sweep_data is not None else sweep(E, tol),
        'rec_new': rec_at(E, target2, tol), 'cv_main': CV_MAIN,
    }
