"""실시간 공정 모니터링 (디자인 ui/Replay.dc.html)

- 화면·문구·재생 로직은 디자인 그대로, 데이터는 mon.json
- 재생 상태는 replay_store(session_state)에 보관 → 시뮬레이션 페이지에 다녀와도 시간이 계속 흐르고,
  시뮬레이션 페이지에서 실행한 목표 변경이 그 시각부터 반영된다
"""
import streamlit as st

import replay_store as RS
from dc_page import render

RS.accept(st.session_state.get('dc_replay'))
render('Replay.dc.html', '실시간 공정 모니터링', height=1300, data=True,
       props={'state': RS.get(), 'now': RS.now_ms(), 'secondsTotal': RS.SECONDS_TOTAL}, key='dc_replay')
