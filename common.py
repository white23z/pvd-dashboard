"""대시보드 페이지 공통: 엔진 로드(1번만), 사이드바 설정 공유, 표시 도우미"""
from pathlib import Path

import pandas as pd
import streamlit as st

from pvd_engine import Engine
import pvd_monitor as M

MODEL_DIR = Path(__file__).parent / 'pvd_model'

DEFAULTS = {'target': 152.2, 'tol': 1.0, 'spec_half': 2.0, 'cv_limit': 3.0, 'cv_rise': 0.3}


@st.cache_resource
def load_engine():
    return Engine(MODEL_DIR)


@st.cache_data
def cached_policy(cv_limit, rise):
    return M.cv_policy(load_engine(), cv_limit, rise)


@st.cache_data
def cached_wafer_table(target):
    return M.wafer_table(load_engine(), target)


@st.cache_data
def cached_rolling(mtime):
    """rolling_wafers.csv (파일 수정 시각이 바뀌면 다시 읽음)"""
    return M.load_rolling(load_engine())


def rolling_mtime():
    f = MODEL_DIR / 'data' / M.ROLLING_FILE
    return f.stat().st_mtime if f.exists() else None


def setting(key, label, step, fmt='%.1f', min_value=None, max_value=None, help=None):
    """페이지를 옮겨도 값이 유지되는 사이드바 숫자 입력"""
    val = st.sidebar.number_input(label, value=float(st.session_state.get(key, DEFAULTS[key])), step=step,
                                  format=fmt, min_value=min_value, max_value=max_value, key=f'_{key}', help=help)
    val = round(float(val), 2)
    st.session_state[key] = val
    return val


def pct(x):
    return '—' if pd.isna(x) else f'{x:.1%}'


def flow_metric(col, label, now, after, fmt='{:.3f}', better='low', neutral_eps=None):
    """'현재 → 추천 적용 → 변화' 지표 카드 (app.py와 같은 모양)"""
    chg = after - now
    if fmt == 'pct':
        s_now, s_aft, s_chg = pct(now), pct(after), f'{chg * 100:+.1f}%p'
    else:
        s_now, s_aft, s_chg = fmt.format(now), fmt.format(after), fmt.replace(':', ':+').format(chg)
    good = chg < 0 if better == 'low' else chg > 0
    color = '#09ab3b' if good else '#d62728'
    if neutral_eps is not None and abs(chg) < neutral_eps:
        color = '#808080'
    col.markdown(
        f'<div style="line-height:1.35">'
        f'<div style="font-size:0.9rem;opacity:0.75">{label}</div>'
        f'<div><span style="font-size:1.15rem;opacity:0.6">{s_now}</span>'
        f'<span style="opacity:0.6"> → </span>'
        f'<span style="font-size:1.9rem;font-weight:600">{s_aft}</span></div>'
        f'<div style="color:{color};font-size:0.95rem;font-weight:600">변화 {s_chg}</div></div>',
        unsafe_allow_html=True)


def status_tile(col, name, status, sub=''):
    bad = status == '조치 필요'
    bg = 'rgba(214,39,40,0.16)' if bad else 'rgba(9,171,59,0.10)'
    bd = '#d62728' if bad else '#09ab3b'
    col.markdown(
        f'<div style="border:1.5px solid {bd};background:{bg};border-radius:8px;padding:6px 8px;margin-bottom:6px">'
        f'<div style="font-weight:700">{name}</div>'
        f'<div style="color:{bd};font-weight:600;font-size:0.9rem">{status}</div>'
        f'<div style="font-size:0.75rem;opacity:0.7">{sub}</div></div>', unsafe_allow_html=True)
