"""디자인 원본(ui/*.dc.html)을 그대로 Streamlit 페이지로 띄우는 도우미

- 디자인 마크업·스타일·로직(JS)은 원본 파일 그대로 쓴다
- {{값}} / <sc-if> / <sc-for> / onClick 같은 디자인 문법은 컴포넌트(ui/_component/index.html)가 처리
- 양방향: Python → 화면은 props(매 실행마다 갱신), 화면 → Python은 디자인 JS의 window.dcSend(값) → render()의 반환값
- 데이터는 mon.json 값을 넣는다 (원본에 박혀 있는 `const D = ...;`를 교체, 값은 같음)
- 왼쪽 아이콘 메뉴의 링크는 Streamlit 페이지 이동으로 바꾼다
"""
import json
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).parent
UI_DIR = ROOT / 'ui'

# 디자인 파일 링크 → Streamlit 페이지 주소(pages/ 파일 이름)
NAV = {
    'Main.dc.html': '실시간_공정_모니터링',
    'Replay.dc.html': '실시간_공정_모니터링',
    'Simulation.dc.html': '목표_두께_변경_시뮬레이션',
}

_component = components.declare_component('dc_page', path=str(UI_DIR / '_component'))


def mon_data():
    """mon.json 위치: design/mon.json → 없으면 프로젝트 맨 위 mon.json"""
    for p in (ROOT / 'design' / 'mon.json', ROOT / 'mon.json'):
        if p.exists():
            return json.loads(p.read_text(encoding='utf-8'))
    raise FileNotFoundError('mon.json을 찾을 수 없습니다')


def parse(name: str, data_json: str | None = None) -> dict:
    """ui/<name> 디자인 파일 → {head, body, logic, props}"""
    src = (UI_DIR / name).read_text(encoding='utf-8')

    helmet = re.search(r'<helmet>(.*?)</helmet>', src, re.S)
    body = re.search(r'<x-dc>(.*?)</x-dc>', src, re.S).group(1)
    body = re.sub(r'<helmet>.*?</helmet>', '', body, flags=re.S)

    sm = re.search(r"<script type=\"text/x-dc\" data-dc-script data-props='(.*?)'>(.*?)</script>", src, re.S)
    props_def = json.loads(sm.group(1)) if sm else {}
    logic = sm.group(2) if sm else 'class Component extends DCLogic { renderVals() { return {}; } }'
    props = {k: v['default'] for k, v in props_def.items() if isinstance(v, dict) and 'default' in v}

    if data_json is not None:  # 디자인에 박힌 데이터 대신 mon.json (값은 같음)
        logic = re.sub(r'^const D = .*?;$', lambda _: 'const D = ' + data_json + ';', logic, count=1, flags=re.M)

    # 이벤트 속성: 템플릿 안에서 실행되지 않도록 이름만 바꿔 둔다 (React식 onChange = 입력할 때마다)
    body = body.replace('onClick="', 'data-on-click="').replace('onChange="', 'data-on-input="')
    for f, slug in NAV.items():
        body = body.replace(f'href="{f}"', f'href="#" data-nav="{slug}"')

    return {'head': helmet.group(1) if helmet else '', 'body': body, 'logic': logic, 'props': props}


def render(name: str, page_title: str, height: int, data: bool = False, props: dict | None = None, key: str | None = None):
    """디자인 한 장을 페이지 전체 폭으로 띄운다.
    height = 처음 높이(이후 내용에 맞춰 자동 조절), props = 디자인 JS의 this.props에 더할 값.
    반환값 = 화면에서 window.dcSend(...)로 보낸 마지막 값 (없으면 None)"""
    st.set_page_config(page_title=page_title, layout='wide', initial_sidebar_state='collapsed')
    st.markdown("""<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #E9EBEF; }
[data-testid="stHeader"] { background: transparent; }
.stMain .block-container { padding: 0 !important; max-width: none !important; }
.stMain [data-testid="stVerticalBlock"] { gap: 0; }
.stMain iframe { display: block; border: 0; }
</style>""", unsafe_allow_html=True)
    data_json = json.dumps(mon_data(), ensure_ascii=False, separators=(',', ':')) if data else None
    a = parse(name, data_json)
    a['props'] = {**a['props'], **(props or {})}
    return _component(**a, key=key or name, default=None, height=height)
