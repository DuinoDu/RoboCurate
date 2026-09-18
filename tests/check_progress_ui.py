"""Progress review/export browser acceptance with real media and labelled test scores."""
import json
from functools import partial
from pathlib import Path
import sys
import tempfile
import threading
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
import server as viewer
from studio import APP,StudioLibrary,Handler
from warp_fixtures import install_profile,install_score,MODEL_ID


def main():
    evidence=APP/'workspace/evidence/warp';evidence.mkdir(parents=True,exist_ok=True)
    profiles=evidence/'browser-profiles';profiles.mkdir(exist_ok=True)
    checks=[]
    def check(name,condition):
        assert condition,name
        checks.append(name);print('PASS',name,flush=True)
    with tempfile.TemporaryDirectory(prefix='warp-ui-',dir=evidence) as folder:
        workspace=Path(folder);lib=StudioLibrary(workspace)
        (workspace/'cache').symlink_to(APP/'workspace/cache',target_is_directory=True)
        source=APP.parent/'2026_09_10-17_40_44/data/lenovo/v141_smoke/episode_000131/mcap/mcap_0.mcap'
        lib.add_root(source);ref=lib.ref(lib.order[0])
        install_profile(lib.progress_service,APP/'workspace/warp-env/bin/python')
        install_score(lib.progress_service,ref.id)
        review=lib.validate_review(ref.id,dict(revision=0,grade='B',instruction='自动化界面验收，不代表人工任务判定',
                                            segments=[dict(start=2,end=4,label='验收片段')]))
        lib.store.save_many([(ref.id,review)])
        before=lib.store.all()
        viewer.STATIC_DIR=APP/'static';robot=viewer.Robot(APP/'robot/g1_29dof_rev_1_0.urdf')
        http=viewer.ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,library=lib,robot=robot))
        threading.Thread(target=http.serve_forever,daemon=True).start()
        opts=Options();opts.add_argument('-headless')
        driver=webdriver.Firefox(options=opts,service=Service('/snap/bin/geckodriver',
                                  service_args=['--profile-root',str(profiles)],log_output=str(evidence/'gecko.log')))
        wait=WebDriverWait(driver,60);url=f'http://127.0.0.1:{http.server_address[1]}'
        def el(css):return driver.find_element('css selector',css)
        def click(css):el(css).click()
        def js(code,*args):return driver.execute_script(code,*args)
        try:
            driver.set_window_size(1580,1050)
            driver.get(url+'/#review/'+ref.id)
            wait.until(lambda _:len(driver.find_elements('css selector','[data-detail="progress"]'))>0)
            js("window.warpUiErrors=[];addEventListener('error',e=>warpUiErrors.push(e.message));addEventListener('unhandledrejection',e=>warpUiErrors.push(String(e.reason)))")
            click('[data-detail="progress"]')
            wait.until(lambda _:len(driver.find_elements('css selector','.warp-candidate'))>0)
            js("document.querySelector('.warp-settings').open=true")
            check('test curves are explicitly labelled', '非模型输出' in el('#warp-model').text and '合成曲线' in el('.warp-domain').text)
            click('.warp-settings summary')
            check('one synchronized timeline shows the curve without a duplicate',len(driver.find_elements('css selector','.warp-chart'))==1)
            check('missing coverage is a visible gap',el('#progress-timeline .warp-chart path').get_attribute('d').count('M')>=2)
            check('no completion percentage is invented','不表示任务完成百分比' in el('.warp-panel').text)
            t=float(el('[data-warp-seek]').get_attribute('data-warp-seek'))
            click('[data-warp-seek]')
            check('candidate seeks the shared video clock',abs(float(el('#timeline').get_attribute('value'))-t)<1e-6)
            click('[data-warp-propose]')
            check('candidate fills bounded editable range',float(el('#clip-end').get_attribute('value'))>float(el('#clip-start').get_attribute('value')) and '候选' in el('#clip-label').get_attribute('value'))
            check('candidate leaves saved human review untouched',lib.store.all()==before)
            click('[data-warp-filter="regression"]')
            check('reviewer can isolate regression candidates',len(driver.find_elements('css selector','.warp-candidate'))>0 and
                  len(driver.find_elements('css selector','.warp-candidate:not(.regression)'))==0)
            click('[data-warp-filter="all"]')
            check('candidate filters preserve the full score curve',len(driver.find_elements('css selector','.warp-chart'))==1)
            js("document.querySelector('#warp-min-duration').value='2';document.querySelector('#warp-min-duration').dispatchEvent(new Event('change',{bubbles:true}))")
            spans=js("return [...document.querySelectorAll('[data-warp-seek]')].map(e=>e.textContent.split('–').map(parseFloat))")
            check('minimum duration suppresses brief candidates without changing the curve',all(b-a>=1.98 for a,b in spans) and len(driver.find_elements('css selector','.warp-chart'))==1)
            click('.detail-more>summary');click('[data-detail="signals"]');click('[data-detail="progress"]')
            check('returning to progress restores controls and filtering',el('#warp-min-duration').get_attribute('value')=='2' and len(driver.find_elements('css selector','.warp-chart'))==1)
            driver.save_screenshot(str(evidence/'progress-dark-test-fixture.png'))
            check('desktop has no page overflow',not js('return document.documentElement.scrollWidth>innerWidth'))
            click('[data-action="theme"]')
            driver.save_screenshot(str(evidence/'progress-light-test-fixture.png'))
            driver.set_window_size(850,1100)
            check('tablet has no page overflow',not js('return document.documentElement.scrollWidth>innerWidth'))
            driver.set_window_size(1580,1050)
            driver.get(url+'/#exports')
            wait.until(lambda _:len(driver.find_elements('css selector','[data-action="new-export"]'))>0)
            click('[data-action="new-export"]')
            wait.until(lambda _:len(driver.find_elements('css selector','#export-warp'))>0)
            click('.warp-export summary');click('#export-warp')
            horizon=el('#export-warp-horizon');horizon.clear();horizon.send_keys('10')
            js("document.querySelector('#export-warp-horizon').dispatchEvent(new Event('change',{bubbles:true}))")
            wait.until(lambda _:'进展覆盖' in el('#preflight-result').text)
            wait.until(lambda _:el('[data-action="confirm-export"]').is_enabled())
            check('export preflight reports progress retention','完整动作块' in el('#preflight-result').text and '保留' in el('#preflight-result').text)
            driver.save_screenshot(str(evidence/'progress-export-test-fixture.png'))
            check('no browser errors',js('return window.warpUiErrors||[]')==[])
            check('saved review still unchanged',lib.store.all()==before)
            (evidence/'ui-report.json').write_text(json.dumps(dict(passed=True,checks=checks,
                data='real MCAP; isolated synthetic score fixture, not a model result'),ensure_ascii=False,indent=2))
        finally:
            driver.quit();http.shutdown();http.server_close()
            lib.progress_service.close();lib.pool.shutdown();lib.export_pool.shutdown()


if __name__=='__main__':main()
