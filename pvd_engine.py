"""
PVD 증착 시간 보정 엔진 (3단계)

노트북(0~2단계 + RSM + 1·2단계)에서 저장한 pvd_model/ 결과물을 불러와
목표 두께가 바뀌었을 때 '현재 → 추천 적용'을 계산한다. 학습은 하지 않는다.

핵심 함수
- Engine.recommend(target)      웨이퍼별 추천 보정량 Δ (노트북 5번과 같은 방식)
- Engine.evaluate(target)       실측 기준 '현재 → 추천 적용 → 변화' (전체·챔버별·타깃 구간별)
- Engine.feasibility(target)    목표에 닿을 수 있는 웨이퍼 비율 (챔버별) → 달성 불가 판단
- Engine.feasible_map()         챔버 × 타깃 사용량 구간별 도달 가능한 두께 범위 (RSM 기준)
- Engine.rule_table(target)     챔버 × 타깃 사용량 구간별 추천 보정량 (보조 자료, 노트북 4번)
- Engine.xgb_shift(target)      (선택) XGBoost로 보정 효과 재확인. xgboost 설치 필요

보정 효과(shift)는 두 가지
- 'rsm' : RSM 반응표면에서 계산 (빠름, 어떤 목표값이든 바로 계산)
- 'xgb' : 추천 증착 시간을 XGBoost에 넣어 다시 예측 (노트북의 A 방식, 느림)
실측 기준 평가 = 실측값 + shift  (모델이 설명 못 하는 잔차는 그대로 남는다고 봄)

코드값: x1 = (Δ − delta.center) / delta.half,  x2 = (타깃 사용량 − life_range[eq].center) / life_range[eq].half
RSM 항 순서: x1, x2, x1², x1·x2, x2²
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

THK, CV, RES = 'mean_thickness', 'thickness_cv', 'mean_resistance'
GRID = np.linspace(-1, 1, 201)          # 보정량 x1 탐색 격자 (노트북과 동일)


def _terms(x1, x2):
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    return np.stack([x1, x2, x1 ** 2, x1 * x2, x2 ** 2], axis=-1)


@dataclass
class Recommendation:
    target: float
    x1_now: np.ndarray
    x1_rec: np.ndarray
    delta_rec: np.ndarray
    shift: dict             # y → 웨이퍼별 보정 효과 (RSM 기준)
    at_bound: np.ndarray    # 추천값이 탐색 범위 끝에 걸렸는지


class Engine:
    def __init__(self, model_dir: str | Path = 'pvd_model'):
        d = Path(model_dir)
        self.dir = d
        self.meta = json.loads((d / 'meta.json').read_text(encoding='utf-8'))
        self.w = pd.read_csv(d / 'data' / 'wafers.csv')
        self._X = None
        m = self.meta
        self.ys = m['ys']
        self.y_names = m['y_names']
        self.life_col, self.tdep_col = m['cols']['life'], m['cols']['tdep']
        self.eqs = m['equipments']
        self.target_default = m['target_default']
        dc, dh = m['delta']['center'], m['delta']['half']
        self.dc, self.dh = dc, dh
        # 웨이퍼별 코드값
        eq = self.w['equipment'].values
        lc = np.array([m['life_range'][e]['center'] for e in eq])
        lh = np.array([m['life_range'][e]['half'] for e in eq])
        self.x2 = np.clip((self.w[self.life_col].values - lc) / lh, -1, 1)
        self.x1_now = np.clip((self.w['delta_now'].values - dc) / dh, -1, 1)
        # RSM 계수: (웨이퍼 수, 6) 행렬로 펼쳐 둠 → 벡터 계산
        self.coef = {}
        for y in self.ys:
            c = {e: np.r_[m['rsm']['coef'][e][y]['intercept'], m['rsm']['coef'][e][y]['coef']] for e in self.eqs}
            self.coef[y] = np.stack([c[e] for e in eq])
        self.eq_coef = {y: {e: np.r_[m['rsm']['coef'][e][y]['intercept'], m['rsm']['coef'][e][y]['coef']]
                            for e in self.eqs} for y in self.ys}
        self.eq_short = self.w['equipment'].str.replace('equipment_', 'eq').values
        self._rec_cache = {}

    # ---------------- RSM ----------------
    def rsm_wafer(self, y, x1):
        """웨이퍼마다 자기 챔버 RSM으로 계산. x1: (웨이퍼 수,) 또는 (웨이퍼 수, k)"""
        x1 = np.asarray(x1, float)
        c = self.coef[y]
        if x1.ndim == 1:
            return c[:, 0] + (_terms(x1, self.x2) * c[:, 1:]).sum(-1)
        T = _terms(x1, np.broadcast_to(self.x2[:, None], x1.shape))     # (n, k, 5)
        return c[:, [0]] + (T * c[:, None, 1:]).sum(-1)

    def rsm_eq(self, eq, y, x1, x2):
        c = self.eq_coef[y][eq]
        return c[0] + _terms(x1, x2) @ c[1:]

    # ---------------- 추천 ----------------
    def recommend(self, target: float | None = None, w1=None, w2=None) -> Recommendation:
        """노트북 5번과 같은 방식: 현재 예측 + [RSM(후보) − RSM(현재)] 로 목적함수를 계산해 격자에서 최소값 선택
        목적함수 = w1·|두께 − 목표| / 두께 표준편차 + w2·CV / CV 표준편차"""
        m = self.meta
        target = self.target_default if target is None else float(target)
        w1 = m['weights']['w1'] if w1 is None else w1
        w2 = m['weights']['w2'] if w2 is None else w2
        key = (round(target, 6), w1, w2)
        if key in self._rec_cache:
            return self._rec_cache[key]
        G = np.broadcast_to(GRID, (len(self.w), len(GRID)))
        r_now = {y: self.rsm_wafer(y, self.x1_now) for y in self.ys}
        r_grid = {y: self.rsm_wafer(y, G) for y in (THK, CV)}
        t_hat = self.w[f'{THK}_pred_now'].values[:, None] + r_grid[THK] - r_now[THK][:, None]
        cv_hat = self.w[f'{CV}_pred_now'].values[:, None] + r_grid[CV] - r_now[CV][:, None]
        f = w1 * np.abs(t_hat - target) / m['y_std'][THK] + w2 * cv_hat / m['y_std'][CV]
        x1_rec = GRID[f.argmin(axis=1)]
        shift = {y: self.rsm_wafer(y, x1_rec) - r_now[y] for y in self.ys}
        rec = Recommendation(target, self.x1_now, x1_rec, self.dc + x1_rec * self.dh, shift, np.abs(x1_rec) > 0.99)
        self._rec_cache[key] = rec
        return rec

    # ---------------- 평가 ----------------
    @staticmethod
    def _metrics(t, cv, r, target, tol):
        e = np.abs(t - target)
        return {'두께 오차': e.mean(), f'±{tol:g} 이내': (e <= tol).mean(), '불량 예상(장)': float((e > tol).sum()),
                '두께 CV': cv.mean(), '면저항 표준편차': r.std()}

    def evaluate(self, target: float | None = None, tol: float = 1.0, shift: str = 'rsm', mask=None):
        """실측 기준 '현재 → 추천 적용 → 변화'
        shift: 'rsm' (기본, 어떤 목표든) / 'B' / 'A' (노트북 저장값, 목표 152.2 전용) / 'xgb' (XGBoost 재확인)
        반환: dict(summary, by_chamber, by_life, rec)"""
        target = self.target_default if target is None else float(target)
        rec = self.recommend(target)
        if shift == 'rsm':
            sh = rec.shift
        elif shift in ('A', 'B'):
            sh = {y: self.w[f'{y}_shift_{shift}'].values for y in self.ys}
        elif shift == 'xgb':
            sh = self.xgb_shift(target)
        else:
            raise ValueError(shift)
        mask = np.ones(len(self.w), bool) if mask is None else np.asarray(mask)
        true = {y: self.w[f'{y}_true'].values for y in self.ys}

        def table(mk):
            now = self._metrics(*(true[y][mk] for y in self.ys), target, tol)
            aft = self._metrics(*(true[y][mk] + sh[y][mk] for y in self.ys), target, tol)
            tb = pd.DataFrame({'현재': now, '추천 적용': aft})
            tb['변화'] = tb['추천 적용'] - tb['현재']
            return tb

        summary = table(mask)
        rows = []
        for key, col in [('챔버', self.eq_short), ('타깃 구간', self.w['life_bin'].values)]:
            for g in sorted(pd.unique(col[mask]), key=_natural):
                mk = mask & (col == g)
                tb = table(mk)
                rows.append({'구분': key, '그룹': g, '웨이퍼 수': int(mk.sum()),
                             '현재 오차': tb.loc['두께 오차', '현재'], '추천 오차': tb.loc['두께 오차', '추천 적용'],
                             f'현재 ±{tol:g}': tb.loc[f'±{tol:g} 이내', '현재'], f'추천 ±{tol:g}': tb.loc[f'±{tol:g} 이내', '추천 적용'],
                             '현재 CV': tb.loc['두께 CV', '현재'], '추천 CV': tb.loc['두께 CV', '추천 적용'],
                             '평균 보정량 Δ': rec.delta_rec[mk].mean(), '범위 끝 비율': rec.at_bound[mk].mean()})
        g = pd.DataFrame(rows)
        return {'summary': summary,
                'by_chamber': g[g['구분'] == '챔버'].drop(columns='구분').set_index('그룹'),
                'by_life': g[g['구분'] == '타깃 구간'].drop(columns='구분').set_index('그룹'),
                'rec': rec, 'shift': sh}

    # ---------------- 달성 가능성 ----------------
    def wafer_reach(self):
        """웨이퍼별로 보정량을 범위 안에서 움직였을 때 도달 가능한 두께 [최소, 최대] (실측 기준)"""
        G = np.broadcast_to(GRID, (len(self.w), len(GRID)))
        d = self.rsm_wafer(THK, G) - self.rsm_wafer(THK, self.x1_now)[:, None]
        t = self.w[f'{THK}_true'].values[:, None]
        return (t + d).min(1), (t + d).max(1)

    def feasibility(self, target: float | None = None, tol: float = 1.0, min_share: float = 0.5):
        """목표 ±tol 구간에 닿을 수 있는 웨이퍼 비율. 챔버별로 min_share 미만이면 '달성 불가'
        observed: 목표가 실측 두께의 관측 범위(5~95%) 안인지 → 밖이면 외삽이라 모델로 판단 불가"""
        target = self.target_default if target is None else float(target)
        lo, hi = self.wafer_reach()
        ok = (hi >= target - tol) & (lo <= target + tol)
        df = pd.DataFrame({'챔버': self.eq_short, 'ok': ok, 'lo': lo, 'hi': hi})
        by = df.groupby('챔버').agg(도달가능비율=('ok', 'mean'), 도달하한_중앙값=('lo', 'median'),
                                   도달상한_중앙값=('hi', 'median'))
        by = by.loc[sorted(by.index, key=_natural)]
        by['판정'] = np.where(by['도달가능비율'] >= min_share, '가능', '달성 불가')
        q = np.quantile(self.w[f'{THK}_true'], [0.05, 0.95])
        return {'overall_share': ok.mean(), 'by_chamber': by, 'observed_range': tuple(q),
                'in_observed_range': bool(q[0] <= target <= q[1])}

    def feasible_map(self, levels=(-1, -0.5, 0, 0.5, 1)):
        """챔버 × 타깃 사용량 구간별 RSM 두께의 [최소, 최대] (그 챔버 '평균 웨이퍼' 기준)"""
        rows = []
        for e in self.eqs:
            for x2 in levels:
                t = self.rsm_eq(e, THK, GRID, np.full_like(GRID, x2))
                rows.append({'챔버': e.replace('equipment_', 'eq'), 'x2': x2,
                             '타깃 사용량': self.meta['life_range'][e]['center'] + x2 * self.meta['life_range'][e]['half'],
                             '최소 두께': t.min(), '최대 두께': t.max()})
        return pd.DataFrame(rows)

    def rule_table(self, target: float | None = None, levels=(-1, -0.5, 0, 0.5, 1)):
        """챔버 × 타깃 사용량 구간별 추천 보정량 (RSM '평균 웨이퍼' 기준, 보조 자료)"""
        m = self.meta
        target = self.target_default if target is None else float(target)
        rows = []
        for e in self.eqs:
            for x2 in levels:
                x2v = np.full_like(GRID, x2)
                t = self.rsm_eq(e, THK, GRID, x2v); cv = self.rsm_eq(e, CV, GRID, x2v)
                f = m['weights']['w1'] * np.abs(t - target) / m['y_std'][THK] + m['weights']['w2'] * cv / m['y_std'][CV]
                i = f.argmin()
                rows.append({'챔버': e.replace('equipment_', 'eq'), 'x2': x2, '추천 보정량 Δ': self.dc + GRID[i] * self.dh,
                             'RSM 두께': t[i], 'RSM CV': cv[i], '범위 끝에 걸림': abs(GRID[i]) > 0.99})
        return pd.DataFrame(rows)

    # ---------------- (선택) XGBoost 재확인 ----------------
    @property
    def X(self):
        if self._X is None:
            self._X = pd.read_csv(self.dir / 'data' / 'X.csv', index_col='row')[self.meta['features']]
        return self._X

    def xgb_shift(self, target: float | None = None):
        """추천 증착 시간을 XGBoost에 넣어 다시 예측한 보정 효과 (노트북 A 방식). xgboost 필요"""
        import xgboost as xgb
        m = self.meta
        rec = self.recommend(target)
        X0 = self.X
        Xr = X0.copy()
        eq = self.w['equipment'].values
        L = self.w[self.life_col].values
        base = np.array([np.polyval(m['base_time_coef'][e], l) for e, l in zip(eq, L)])
        t_new = base + rec.delta_rec
        if m['cols']['recipe']:
            Xr[m['cols']['recipe']] = X0[m['cols']['recipe']].values + m['b_recipe'] * (t_new - X0[self.tdep_col].values)
        Xr[self.tdep_col] = t_new
        out = {}
        for y in self.ys:
            b = xgb.XGBRegressor(); b.load_model(str(self.dir / 'models' / f'xgb_{y}.json'))
            s = m['y_scale'][y]
            out[y] = (b.predict(Xr) - b.predict(X0)) * s['sd']
        return out


def _natural(s):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(s))]
