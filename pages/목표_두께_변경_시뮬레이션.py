"""목표 두께 변경 시뮬레이션 (디자인 ui/Simulation.dc.html)

- 화면의 새 목표(150.0~155.0)·적용일·허용 범위 → '시뮬레이션 실행' → pvd_sim 30일 계산 → 화면에서 하루씩 재생
- 실행하면 그 순간의 실시간 모니터링 시각부터 첫 화면 목표도 새 목표로 바뀐다 (replay_store, 여러 번 가능)
- 계산은 sim_result.target_change (SimPy 30일 + 엔진 RSM·feasibility), 같은 입력은 캐시
"""
import streamlit as st

import replay_store as RS
import sim_result
from common import load_engine
from dc_page import render
from pvd_sim import Simulator

DEFAULT = {'target2': 153.0, 'day': 10, 'tol': 1.0}
T_MIN, T_MAX = 150.0, 155.0        # 첫 화면 데이터(mon.json)에 보정 효과가 있는 목표 범위


@st.cache_resource
def simulator():
    return Simulator(load_engine())


@st.cache_data(show_spinner=False)
def sweep(tol):
    return sim_result.sweep(load_engine(), tol)


@st.cache_data(show_spinner=False)
def result(target2, day, tol):
    return sim_result.target_change(load_engine(), simulator(), target2, day, tol, sweep(tol))


sent = st.session_state.get('dc_sim')          # 화면에서 보낸 마지막 실행 요청
if isinstance(sent, dict) and sent.get('n') is not None and sent.get('n') != st.session_state.get('sim_last_n'):
    try:
        req = {'target2': round(min(T_MAX, max(T_MIN, float(sent['target2']))), 1),
               'day': int(min(29, max(1, round(float(sent['day']))))),
               'tol': round(min(3.0, max(0.1, float(sent['tol']))), 1)}
        st.session_state.sim_req = req
        st.session_state.sim_last_n = sent['n']
        h = RS.add_recipe(req['target2'])        # 첫 화면에 '실행한 순간'부터 반영
        st.session_state.sim_applied = (
            {'ok': True, 'msg': f"실시간 공정 모니터링에 {RS.fmt(h)}부터 목표 {req['target2']:.1f} 적용됨"} if h is not None else
            {'ok': False, 'msg': '실시간 공정 모니터링 재생이 끝나서 반영하지 못했습니다 · 처음부터 재생한 뒤 다시 실행하세요'})
    except (KeyError, TypeError, ValueError):
        pass

req = st.session_state.get('sim_req', DEFAULT)
R = {**result(req['target2'], req['day'], req['tol']), 'n': st.session_state.get('sim_last_n')}
render('Simulation.dc.html', '목표 두께 변경 시뮬레이션', height=1640, key='dc_sim',
       props={'R': R, 'applied': st.session_state.get('sim_applied'), 'tmin': T_MIN, 'tmax': T_MAX})
