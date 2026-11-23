"""PVD 대시보드 진입점: `streamlit run app.py`

메뉴에는 디자인 화면 2개만 둔다 (이전 단계 페이지들은 legacy/로 옮김).
url_path는 dc_page.NAV(디자인 아이콘 메뉴 링크)와 같아야 한다.
"""
import streamlit as st

pg = st.navigation([
    st.Page('pages/실시간_공정_모니터링.py', title='실시간 공정 모니터링', url_path='실시간_공정_모니터링', default=True),
    st.Page('pages/목표_두께_변경_시뮬레이션.py', title='목표 두께 변경 시뮬레이션', url_path='목표_두께_변경_시뮬레이션'),
])
pg.run()
