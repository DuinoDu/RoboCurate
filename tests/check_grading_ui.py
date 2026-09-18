"""Grading-specific acceptance; all test state stays in an isolated workspace."""
import json
from functools import partial
from pathlib import Path
import tempfile
import threading

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait, Select

import server as viewer
from studio import APP, StudioLibrary, Handler


def main():
    evidence = APP/'workspace/evidence'; profiles = evidence/'profiles'; profiles.mkdir(parents=True,exist_ok=True)
    before = StudioLibrary(APP/'workspace').store.all()
    checks=[]
    def check(name, value):
        assert value,name
        checks.append(name);print('PASS',name,flush=True)
    with tempfile.TemporaryDirectory(prefix='grade-ui-',dir=evidence) as temp:
        lib=StudioLibrary(Path(temp));(Path(temp)/'cache').symlink_to(APP/'workspace/cache',target_is_directory=True)
        lib.add_root(APP.parent/'2026_09_10-17_40_44')
        lib.add_root(APP.parent/'mcap_viz.tar(1)/mcap_viz/mcap_0.mcap')
        viewer.STATIC_DIR=APP/'static'
        http=viewer.ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,library=lib,robot=viewer.Robot(APP/'robot/g1_29dof_rev_1_0.urdf')))
        threading.Thread(target=http.serve_forever,daemon=True).start()
        url=f'http://127.0.0.1:{http.server_address[1]}'
        options=Options();options.add_argument('-headless')
        service=Service('/snap/bin/geckodriver',service_args=['--profile-root',str(profiles)],log_output=str(evidence/'grading-geckodriver.log'))
        d=webdriver.Firefox(options=options,service=service);wait=WebDriverWait(d,50)
        def el(css):return d.find_element('css selector',css)
        def click(css):el(css).click()
        def select(css,value):Select(el(css)).select_by_value(value)
        try:
            d.set_window_size(1580,1100);d.get(url)
            wait.until(lambda _:len(d.find_elements('css selector','.episode-row'))==19)
            d.execute_script("window.gradingErrors=[];addEventListener('error',e=>gradingErrors.push(e.message));addEventListener('unhandledrejection',e=>gradingErrors.push(String(e.reason)))")
            check('automatic grade counts',len(d.find_elements('css selector','[data-auto-grade="B"]'))==18 and len(d.find_elements('css selector','[data-auto-grade="D"]'))==1)
            check('best first default',el('.auto-grade-cell').get_attribute('data-auto-grade')=='B')
            check('grade distribution visible','数据质量' in el('#data-grade-summary').text)
            d.save_screenshot(str(evidence/'grading-library.png'))
            select('#filter-auto','B');check('B filter',len(d.find_elements('css selector','.episode-row'))==18)
            select('#filter-auto','D');check('D filter',len(d.find_elements('css selector','.episode-row'))==1)
            select('#filter-auto','A');check('empty top grade is honest',not d.find_elements('css selector','.episode-row'))
            select('#filter-auto','');select('#filter-grade','B')
            check('manual approval is independent',not d.find_elements('css selector','.episode-row'))
            select('#filter-grade','');select('#filter-sort','score')
            check('worst first sort',el('.auto-grade-cell').get_attribute('data-auto-grade')=='D')
            select('#filter-sort','quality')
            el('#search').send_keys('episode_000140');click('.ep-name')
            wait.until(lambda _:len(d.find_elements('css selector','#auto-grade-review'))==1)
            check('review shows measured grade',el('#auto-grade-review .data-grade').get_attribute('data-level')=='B')
            click('[data-action="show-quality"]')
            check('five dimensions explained',len(d.find_elements('css selector','.grade-check'))==5)
            check('coverage and action visible','99.71%' in el('.grade-report').text and '建议' in el('.grade-report').text)
            check('review todo still visible','需要补充具体任务指令' in el('#detail-body').text)
            click('.reference-checks summary')
            wait.until(lambda _:len(d.find_elements('css selector','.reference-timing tbody tr'))==9)
            check('source checks show native stream evidence',len(d.find_elements('css selector','.reference-values tbody tr'))==6)
            check('reference evidence is advisory','不自动改变数据等级' in el('#reference-checks-body').text)
            d.save_screenshot(str(evidence/'reference-checks-review.png'))
            click('.reference-checks summary')
            d.save_screenshot(str(evidence/'grading-review.png'))
            click('a[href="#rules"]');wait.until(lambda _:len(d.find_elements('css selector','.grade-standard'))==4)
            check('rules explain best through unusable','当前不可用' in el('.grade-standards').text and '99%' in el('.grade-standards').text)
            d.save_screenshot(str(evidence/'grading-rules.png'))
            d.set_window_size(820,1100)
            check('rules fit narrow screen',d.execute_script('return document.documentElement.scrollWidth <= innerWidth'))
            check('no JavaScript errors',d.execute_script('return window.gradingErrors')==[])
            check('real reviews preserved',StudioLibrary(APP/'workspace').store.all()==before)
            (evidence/'grading-ui-result.json').write_text(json.dumps(dict(passed=True,checks=checks),ensure_ascii=False,indent=2))
        finally:
            d.quit();http.shutdown();http.server_close();lib.pool.shutdown();lib.export_pool.shutdown()


if __name__=='__main__':main()
