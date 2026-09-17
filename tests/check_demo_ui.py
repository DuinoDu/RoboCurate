"""Portable browser acceptance using only a generated MCAP and isolated reviews."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import threading
import zipfile

from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait

import server as viewer
from studio import APP, StudioLibrary, Handler
from scripts.create_demo import create_demo


def run(args):
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    profiles=out/'browser-profiles';profiles.mkdir(exist_ok=True)
    checks=[]
    def check(name,ok):
        assert ok,name
        checks.append(name);print('PASS',name,flush=True)
    with tempfile.TemporaryDirectory(prefix='synthetic-ui-',dir=out) as folder:
        root=Path(folder);source=create_demo(root/'recordings')
        original=hashlib.sha256(source.read_bytes()).hexdigest()
        lib=StudioLibrary(root/'workspace');lib.add_root(source)
        ep=lib.order[0];lib.build_now(ep)
        viewer.STATIC_DIR=APP/'static'
        http=viewer.ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,library=lib,
            robot=viewer.Robot(APP/'robot/g1_29dof_rev_1_0.urdf')))
        threading.Thread(target=http.serve_forever,daemon=True).start()
        options=Options();options.add_argument('-headless')
        driver=None
        try:
            driver=webdriver.Firefox(options=options,service=Service(args.geckodriver,
                service_args=['--profile-root',str(profiles)],log_output=str(out/'gecko.log')))
            wait=WebDriverWait(driver,60)
            def el(css):return driver.find_element('css selector',css)
            def js(code,*values):return driver.execute_script(code,*values)
            def click(css):
                def attempt(_):
                    try:
                        target=el(css);js("arguments[0].scrollIntoView({block:'center'})",target)
                        target.click();return True
                    except StaleElementReferenceException:return False
                wait.until(attempt)
            def fill(css,value):
                target=el(css);target.clear();target.send_keys(str(value))
            url=f'http://127.0.0.1:{http.server_address[1]}'
            driver.set_window_size(1580,1050);driver.get(url+'/#review/'+ep)
            wait.until(lambda _:bool(driver.find_elements('css selector','.progress-empty')))
            js("window.demoErrors=[];addEventListener('error',e=>demoErrors.push(e.message));addEventListener('unhandledrejection',e=>demoErrors.push(String(e.reason)))")
            check('fresh source uses RoboCurate branding',driver.title.startswith('RoboCurate'))
            check('task has an honest uninstalled-model state','尚未安装任务分析模型' in el('.progress-empty').text and not lib.progress_service.models()['models'])
            check('three primary tabs and folded advanced navigation',len(driver.find_elements('css selector','.detail-tabs>button[role=tab]'))==3 and not js("return document.querySelector('.detail-more').open"))
            check('ten-point score matches the API',float(el('.rating-number strong').text)==lib.record(lib.ref(ep))['quality']['rating']['score'] and '/ 10' in el('.rating-number').text)
            click('[data-stage-layout=video]')
            wait.until(lambda _:js("return [...document.querySelectorAll('[data-camera]')].length===3 && [...document.querySelectorAll('[data-camera]')].every(i=>i.complete&&i.naturalWidth>0&&i.dataset.frameIndex!==undefined)"))
            check('three synthetic camera views decode',True)
            click('[data-detail=diagnostics]')
            wait.until(lambda _:bool(driver.find_elements('css selector','.event-list-row')))
            check('known synthetic faults are available for review',len(lib.diagnostics(ep)['events'])>=2)
            click('.event-list-row')
            check('quality timeline is the active analysis track',el('#event-timeline').is_displayed() and not el('#progress-timeline').is_displayed())
            check('no horizontal overflow',not js('return document.documentElement.scrollWidth>innerWidth'))
            js("document.querySelector('#detail-body').scrollTop=0;document.querySelector('.playback-panel').scrollTop=0;window.scrollTo(0,0)")
            driver.save_screenshot(str(out/'synthetic-demo.png'))
            click('[data-action=show-quality]')
            check('all four score dimensions are still inspectable',len(driver.find_elements('css selector','.rating-dimension'))==4 and '/ 10' in el('.rating-dimension summary').text)
            click('.detail-more>summary');click('[data-detail=streams]')
            check('raw channels are reachable through More',el('.technical-table').is_displayed())
            click('[data-detail=review]');click('[data-grade=B]')
            fill('#task-instruction','将黄色方块从左桌移至右侧容器，合成演示。')
            click('#clip-editor>summary');fill('#clip-start',.5);fill('#clip-end',1.5)
            fill('#clip-label','合成演示有效片段');click('[data-action=add-clip]')
            check('draft does not save itself',not lib.store.all())
            click('[data-action=save]')
            wait.until(lambda _:lib.store.all().get(ep,{}).get('revision')==1)
            check('explicit save persists the reviewed clip',lib.store.all()[ep]['segments'][0]['start']==.5 and lib.store.all()[ep]['segments'][0]['end']==1.5)
            click('[data-nav=exports]')
            wait.until(lambda _:bool(driver.find_elements('css selector','[data-action=new-export]')))
            click('[data-action=new-export]');wait.until(lambda _:el('[data-action=confirm-export]').is_enabled())
            check('synthetic training export passes preflight','检查通过' in el('#preflight-result').text)
            click('[data-action=confirm-export]')
            wait.until(lambda _:any(j['state']=='done' for j in lib.exports()))
            result=next(j for j in lib.exports() if j['state']=='done')
            check('UI creates an actual 30-sample training bundle',result['samples']==30 and result['valid_samples']==30)
            with zipfile.ZipFile(lib.export_root/result['file']) as archive:
                row=json.loads(archive.read('manifest.json'))['episodes'][0]
                check('export preserves the ten-point scale',row['quality']['rating']['max_score']==10 and 0<=row['quality']['score']<=10)
            check('generated source remains unchanged',hashlib.sha256(source.read_bytes()).hexdigest()==original)
            check('no uncaught browser errors',js('return window.demoErrors')==[])
            (out/'report.json').write_text(json.dumps(dict(passed=True,checks=checks,data='Generated synthetic MCAP only; no model predictions or private recordings'),ensure_ascii=False,indent=2)+'\n')
        except Exception:
            if driver:driver.save_screenshot(str(out/'failure.png'))
            raise
        finally:
            if driver:driver.quit()
            http.shutdown();http.server_close();lib.progress_service.close()
            lib.pool.shutdown();lib.export_pool.shutdown()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(APP/'workspace/evidence/public-demo'))
    parser.add_argument('--geckodriver',default=shutil.which('geckodriver'))
    args=parser.parse_args()
    if not args.geckodriver:parser.error('Install Firefox and geckodriver or specify --geckodriver')
    run(args)
