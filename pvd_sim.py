"""
PVD 가상 공정 시뮬레이터 (5단계, SimPy)

무엇을 흉내 내나
- 로트가 도착하고(포아송), 챔버 16대 중 하나에 배정되어 웨이퍼를 한 장씩 처리한다.
- 웨이퍼를 처리할 때마다 그 챔버의 타깃 사용량이 늘고, 한계에 닿으면 타깃을 교체한다(정지 시간 포함).
- 웨이퍼 한 장의 '실측 두께'는 그 챔버·타깃 사용량과 비슷한 실제 웨이퍼를 하나 뽑아 그 실측값을 쓴다.
  여기에 시나리오(드리프트 등)의 영향을 더한다. → 가상 실측값 = 실제 웨이퍼 실측 + 시나리오 영향

두 가지 운영 방식을 '같은 사건 흐름(공통 난수)' 위에서 동시에 계산한다
- 현재(APC만): 실제 데이터의 증착 시간을 그대로 쓴다. 주입한 드리프트에는 반응하지 않는다고 가정
- 추천 적용  : 웨이퍼마다 증착 시간 보정량 Δ를 추천해 적용한다.
    예상 두께 = 교차검증 예측(그 웨이퍼를 학습에 안 쓴 모델) + 챔버별 피드백 보정 + RSM 보정 효과
    피드백 보정 = 그 챔버에서 최근 측정한 '실측 − 예상'의 EWMA (run-to-run 제어)
    피드백이 alarm_run장 연속 허용 범위를 넘으면 '엔지니어 호출' 경보를 남기고 그 챔버를 '경보 중'으로 둔다.
    경보 중에는 다시 울리지 않고, 피드백이 alarm_run장 연속 허용 범위 안에 머물러야 '경보 해제'된다.

시나리오 (Scenario 필드)
- drift_eq / drift_start / drift_rate : 특정 챔버 두께가 drift_start일부터 하루 drift_rate씩 변함
- replace_all_day                      : 그날 모든 챔버의 타깃을 한꺼번에 교체 (새 타깃 구간 몰림)
- down_eq / down_start / down_end      : 특정 챔버 정지 → 로트가 다른 챔버로 재배정
- target / target2 / target2_day       : 목표 두께 (target2_day부터 target2로 변경)

시간 단위: 1 = 1시간. 기본 30일, 실제 데이터와 비슷한 웨이퍼 수(약 5,200장)가 나오도록 도착률을 맞췄다.
※ 가상 환경 결과이며 실제 공정 실험이 아니다.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import simpy

from pvd_engine import Engine, GRID, THK, CV

HOURS_PER_DAY = 24


@dataclass
class Scenario:
    name: str = '기본'
    days: int = 30
    seed: int = 0
    lots_per_hour: float = 4.1           # 도착률 (30일 ≈ 웨이퍼 5,200장)
    target: float = 152.2
    tol: float = 1.0
    target2: float | None = None         # 목표 변경 시나리오
    target2_day: float | None = None
    drift_eq: str | None = None          # 'equipment_6' 등
    drift_start: float = 10.0            # 일
    drift_rate: float = -0.15            # 하루당 두께 변화
    replace_all_day: float | None = None
    down_eq: str | None = None
    down_start: float = 10.0
    down_end: float = 15.0
    feedback: bool = True                # 추천 적용에서 피드백(EWMA) 사용
    ewma_lambda: float = 0.1
    alarm_run: int = 8                   # 피드백이 tol을 넘은 채 연속 몇 장이면 경보 / tol 안에 연속 몇 장이면 해제
    wafer_hours: float = 0.5             # 웨이퍼 1장 처리 시간
    replace_hours: float = 1.0           # 타깃 교체 정지 시간
    label: str = ''

    def target_at(self, day):
        if self.target2 is not None and self.target2_day is not None and day >= self.target2_day:
            return self.target2
        return self.target


PRESETS = {
    '기본': Scenario(name='기본'),
    '목표 두께 변경': Scenario(name='목표 두께 변경', target2=153.0, target2_day=10),
    '챔버 드리프트': Scenario(name='챔버 드리프트', drift_eq='equipment_6', drift_start=10, drift_rate=-0.15),
    '타깃 일괄 교체': Scenario(name='타깃 일괄 교체', replace_all_day=10),
    '챔버 정지': Scenario(name='챔버 정지', down_eq='equipment_9', down_start=10, down_end=15),
}


class Simulator:
    def __init__(self, engine: Engine):
        E = self.E = engine
        w = E.w
        self.eqs = E.eqs
        self.oof = w[f'{THK}_oof'].values
        self.true = {y: w[f'{y}_true'].values for y in E.ys}
        self.life = w[E.life_col].values
        # 챔버별: 웨이퍼 인덱스를 타깃 사용량 순으로 정렬 (비슷한 웨이퍼 뽑기용)
        self.pool = {}
        for e in self.eqs:
            idx = np.where(w['equipment'].values == e)[0]
            idx = idx[np.argsort(self.life[idx])]
            self.pool[e] = (idx, self.life[idx])
        share = w['equipment'].value_counts()
        self.share = np.array([share[e] for e in self.eqs], float)
        self.share /= self.share.sum()
        # 타깃 사용량: 교체 한계(챔버별 98%), 웨이퍼당 증가량(실제 데이터의 양의 증가분)
        self.life_max = {e: float(np.quantile(self.pool[e][1], 0.98)) for e in self.eqs}
        tt = pd.to_datetime(w['runstart'])
        d = w.assign(_t=tt).sort_values(['equipment', '_t']).groupby('equipment')[E.life_col].diff()
        self.life_step = d[d > 0].values
        self.wafers_per_lot = w.groupby('lot').size().values

    # ---------- 웨이퍼 한 장 ----------
    def _pick(self, rng, eq, life, k=15):
        idx, lv = self.pool[eq]
        j = np.searchsorted(lv, life)
        lo, hi = max(0, j - k), min(len(idx), j + k)
        return int(idx[rng.integers(lo, hi)])

    def _recommend(self, i, target, bias):
        """웨이퍼 i에 대한 추천 x1 (엔진과 같은 목적함수, 예측은 교차검증 + 피드백)"""
        E = self.E
        c_t, c_cv = E.coef[THK][i], E.coef[CV][i]
        x2 = E.x2[i]
        T = np.stack([GRID, np.full_like(GRID, x2), GRID ** 2, GRID * x2, np.full_like(GRID, x2 ** 2)], 1)
        T0 = np.array([E.x1_now[i], x2, E.x1_now[i] ** 2, E.x1_now[i] * x2, x2 ** 2])
        d_t = T @ c_t[1:] - T0 @ c_t[1:]
        d_cv = T @ c_cv[1:] - T0 @ c_cv[1:]
        t_hat = self.oof[i] + bias + d_t
        cv_hat = self.true[CV][i] + d_cv
        m = E.meta
        f = m['weights']['w1'] * np.abs(t_hat - target) / m['y_std'][THK] + m['weights']['w2'] * cv_hat / m['y_std'][CV]
        k = int(f.argmin())
        return GRID[k], d_t[k], d_cv[k]

    # ---------- 실행 ----------
    def run(self, sc: Scenario) -> dict:
        rng = np.random.default_rng(sc.seed)
        env = simpy.Environment()
        res = {e: simpy.Resource(env, capacity=1) for e in self.eqs}
        life = {e: float(rng.uniform(0, self.life_max[e])) for e in self.eqs}     # 시작 시점 타깃 상태는 무작위
        bias = {e: 0.0 for e in self.eqs}
        over = {e: 0 for e in self.eqs}          # 피드백이 ±tol 밖에 연속으로 있은 장 수
        inside = {e: 0 for e in self.eqs}        # 피드백이 ±tol 안에 연속으로 있은 장 수
        alarmed = {e: False for e in self.eqs}   # 챔버별 '경보 중' 상태
        logs, events = [], []
        horizon = sc.days * HOURS_PER_DAY

        def is_down(e, day):
            return sc.down_eq == e and sc.down_start <= day < sc.down_end

        def replace(e, why):
            yield env.timeout(sc.replace_hours)
            life[e] = float(rng.uniform(0, 5))
            events.append({'시각(일)': env.now / HOURS_PER_DAY, '챔버': e, '사건': why})

        def process_lot(lot_id, e, n):
            with res[e].request() as req:
                yield req
                for _ in range(n):
                    day = env.now / HOURS_PER_DAY
                    if life[e] >= self.life_max[e]:
                        yield env.process(replace(e, '타깃 교체'))
                        day = env.now / HOURS_PER_DAY
                    i = self._pick(rng, e, life[e])
                    target = sc.target_at(day)
                    drift = sc.drift_rate * (day - sc.drift_start) if (sc.drift_eq == e and day >= sc.drift_start) else 0.0
                    t_apc = self.true[THK][i] + drift
                    b = bias[e] if sc.feedback else 0.0
                    x1, d_t, d_cv = self._recommend(i, target, b)
                    t_rec = self.true[THK][i] + drift + d_t
                    # 피드백 갱신 (추천 적용 쪽 측정값으로)
                    err = t_rec - (self.oof[i] + d_t)
                    bias[e] = (1 - sc.ewma_lambda) * bias[e] + sc.ewma_lambda * err
                    if abs(bias[e]) > sc.tol:
                        over[e] += 1; inside[e] = 0
                    else:
                        inside[e] += 1; over[e] = 0
                    alarm = (not alarmed[e]) and over[e] >= sc.alarm_run
                    if alarm:
                        alarmed[e] = True
                        events.append({'시각(일)': day, '챔버': e, '사건': f'엔지니어 호출 (피드백 보정 {bias[e]:+.2f}가 {sc.alarm_run}장 연속 ±{sc.tol:g} 초과)'})
                    elif alarmed[e] and inside[e] >= sc.alarm_run:
                        alarmed[e] = False
                        events.append({'시각(일)': day, '챔버': e, '사건': f'경보 해제 (피드백 보정 {bias[e]:+.2f}가 {sc.alarm_run}장 연속 ±{sc.tol:g} 이내)'})
                    logs.append({'시각(일)': day, '로트': lot_id, '챔버': e, '타깃 사용량': life[e], '원본 웨이퍼': i,
                                 '목표': target, '드리프트': drift, '두께_현재': t_apc, '두께_추천': t_rec,
                                 'CV_현재': self.true[CV][i], 'CV_추천': self.true[CV][i] + d_cv,
                                 '보정량Δ': self.E.dc + x1 * self.E.dh, '범위끝': abs(x1) > 0.99,
                                 '피드백': bias[e], '경보': alarm, '경보 중': alarmed[e]})
                    yield env.timeout(sc.wafer_hours)
                    life[e] += float(rng.choice(self.life_step))

        def arrivals():
            lot = 0
            while True:
                yield env.timeout(rng.exponential(1 / sc.lots_per_hour))
                day = env.now / HOURS_PER_DAY
                p = self.share.copy()
                for k, e in enumerate(self.eqs):
                    if is_down(e, day):
                        p[k] = 0
                p /= p.sum()
                e = self.eqs[int(rng.choice(len(self.eqs), p=p))]
                n = int(rng.choice(self.wafers_per_lot))
                lot += 1
                env.process(process_lot(lot, e, n))

        def scheduled():
            if sc.replace_all_day is not None:
                yield env.timeout(sc.replace_all_day * HOURS_PER_DAY)
                for e in self.eqs:
                    life[e] = self.life_max[e]          # 다음 웨이퍼 전에 교체되도록
                events.append({'시각(일)': sc.replace_all_day, '챔버': '전체', '사건': '타깃 일괄 교체 지시'})
            if sc.down_eq:
                yield env.timeout(max(0, sc.down_start * HOURS_PER_DAY - env.now))
                events.append({'시각(일)': sc.down_start, '챔버': sc.down_eq, '사건': '챔버 정지 (로트 재배정 시작)'})
                yield env.timeout(max(0, sc.down_end * HOURS_PER_DAY - env.now))
                events.append({'시각(일)': sc.down_end, '챔버': sc.down_eq, '사건': '챔버 복귀'})

        env.process(arrivals())
        env.process(scheduled())
        env.run(until=horizon)
        log = pd.DataFrame(logs)
        ev = pd.DataFrame(events, columns=['시각(일)', '챔버', '사건']).sort_values('시각(일)').reset_index(drop=True)
        return {'scenario': sc, 'log': log, 'events': ev, 'summary': summarize(log, sc), 'daily': daily(log, sc)}


# ---------- 요약 ----------
def _m(t, target, tol):
    e = np.abs(t - target)
    return e.mean(), (e <= tol).mean(), int((e > tol).sum())


def summarize(log: pd.DataFrame, sc: Scenario) -> pd.DataFrame:
    rows = {}
    for name, col, cvc in [('현재', '두께_현재', 'CV_현재'), ('추천 적용', '두께_추천', 'CV_추천')]:
        err, within, bad = _m(log[col].values, log['목표'].values, sc.tol)
        rows[name] = {'웨이퍼 수': len(log), '두께 오차': err, f'±{sc.tol:g} 이내': within, '불량(장)': bad,
                      '평균 두께': log[col].mean(), '두께 표준편차': log[col].std(), '두께 CV': log[cvc].mean()}
    tb = pd.DataFrame(rows)
    tb['변화'] = tb['추천 적용'] - tb['현재']
    return tb


def daily(log: pd.DataFrame, sc: Scenario) -> pd.DataFrame:
    d = log.assign(일=np.floor(log['시각(일)']).astype(int))
    out = []
    for day, g in d.groupby('일'):
        r = {'일': day, '웨이퍼 수': len(g), '목표': g['목표'].iloc[-1]}
        for name, col in [('현재', '두께_현재'), ('추천 적용', '두께_추천')]:
            e = np.abs(g[col] - g['목표'])
            r[f'{name} ±{sc.tol:g}'] = (e <= sc.tol).mean()
            r[f'{name} 불량'] = int((e > sc.tol).sum())
            r[f'{name} 평균 두께'] = g[col].mean()
        out.append(r)
    df = pd.DataFrame(out)
    df['현재 누적 불량'] = df['현재 불량'].cumsum()
    df['추천 적용 누적 불량'] = df['추천 적용 불량'].cumsum()
    return df


def validate(sim: Simulator, seed: int = 0) -> dict:
    """기본 시나리오의 '현재'가 실제 데이터 분포를 재현하는지"""
    out = sim.run(Scenario(name='기본', seed=seed))
    log = out['log']
    real = sim.E.w
    t_real = real[f'{THK}_true']
    by = pd.DataFrame({'실제 평균': real.groupby(sim.E.eq_short)[f'{THK}_true'].mean(),
                       '가상 평균': log.groupby(log['챔버'].str.replace('equipment_', 'eq'))['두께_현재'].mean(),
                       '실제 웨이퍼 비율': real['equipment'].str.replace('equipment_', 'eq').value_counts(normalize=True),
                       '가상 웨이퍼 비율': log['챔버'].str.replace('equipment_', 'eq').value_counts(normalize=True)})
    overall = pd.DataFrame({'실제': [len(real), t_real.mean(), t_real.std(), real[sim.E.life_col].mean()],
                            '가상(현재)': [len(log), log['두께_현재'].mean(), log['두께_현재'].std(), log['타깃 사용량'].mean()]},
                           index=['웨이퍼 수', '평균 두께', '두께 표준편차', '평균 타깃 사용량'])
    return {'overall': overall, 'by_chamber': by, 'run': out}
