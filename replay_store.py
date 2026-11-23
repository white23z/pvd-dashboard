"""실시간 공정 모니터링 재생 상태 (페이지를 옮겨도 이어지도록 session_state에 보관)

- 화면(JS)이 재생·일시정지·대응 조치 때마다 상태 전체를 보내고(seq 증가), 여기서 받아 둔다
- 시간은 기준점(aW = 서버 시각 ms, aH = 그때 재생 시각 h)에서 계산 → 다른 페이지에 가 있어도 계속 흐른다
- events: 화면에서 일어난 일 [{h, k, t?}]  k = recipe(목표 변경, t=새 목표) / ack / a1~a4 / back
- 시뮬레이션 페이지가 '실행한 순간'의 h에 recipe 이벤트를 더한다
"""
import time
from datetime import datetime, timedelta

import streamlit as st

END = 1104                 # 재생 끝 (2025.03.15 00:00 + 1104h = 04.30 24:00)
SECONDS_TOTAL = 80         # 전체 재생 시간(초), Replay.dc.html의 secondsTotal 기본값과 같게
T0 = datetime(2025, 3, 15)
KEY = 'replay_state'


def now_ms():
    return time.time() * 1000


def get():
    if KEY not in st.session_state:
        st.session_state[KEY] = {'seq': 0, 'spd': SECONDS_TOTAL, 'started': False, 'running': False,
                                 'aW': 0, 'aH': 0, 'events': []}
    return st.session_state[KEY]


def accept(sent):
    """화면에서 보낸 상태가 더 새것이면 저장"""
    if isinstance(sent, dict) and isinstance(sent.get('seq'), (int, float)) and sent['seq'] > get()['seq']:
        st.session_state[KEY] = sent


def h_now(S=None):
    S = S or get()
    h = S['aH']
    if S['running']:
        h += (now_ms() - S['aW']) / 1000 * END / S['spd']
    return min(END, h)


def fmt(h):
    return (T0 + timedelta(hours=h)).strftime('%m.%d %H:%M')


def add_recipe(target):
    """지금 재생 시각부터 목표 변경. 재생이 끝났으면 None"""
    S = get()
    h = h_now(S)
    if h >= END:
        return None
    S['events'].append({'h': h, 'k': 'recipe', 't': float(target)})
    S['seq'] += 1
    return h
