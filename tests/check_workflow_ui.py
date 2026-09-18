"""Workflow / Holo component acceptance in an isolated review workspace."""
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
    evidence=APP/'workspace/evidence';profiles=evidence/'profiles';profiles.mkdir(parents=True,exist_ok=True)
    real=StudioLibrary(APP/'workspace');before=real.store.all();checks=[]
    def check(name,ok):
        assert ok,name
        checks.append(name);print('PASS',name,flush=True)
    with tempfile.TemporaryDirectory(prefix='workflow-ui-',dir=evidence) as tmp:
        lib=StudioLibrary(Path(tmp));(Path(tmp)/'cache').symlink_to(APP/'workspace/cache',target_is_directory=True)
        lib.add_root(APP.parent/'2026_09_10-17_40_44');lib.add_root(APP.parent/'mcap_viz.tar(1)/mcap_viz/mcap_0.mcap')
        original_grades={r['id']:r['quality']['grading']['grade'] for r in lib.catalog()['episodes']}
        viewer.STATIC_DIR=APP/'static'
        http=viewer.ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,library=lib,robot=viewer.Robot(APP/'robot/g1_29dof_rev_1_0.urdf')))
        threading.Thread(target=http.serve_forever,daemon=True).start();url=f'http://127.0.0.1:{http.server_address[1]}'
        opts=Options();opts.add_argument('-headless')
        service=Service('/snap/bin/geckodriver',service_args=['--profile-root',str(profiles)],log_output=str(evidence/'workflow-geckodriver.log'))
        d=webdriver.Firefox(options=opts,service=service);wait=WebDriverWait(d,60)
        def el(css):return d.find_element('css selector',css)
        def click(css):el(css).click()
        def fill(css,value):el(css).clear();el(css).send_keys(str(value))
        def js(code,*args):return d.execute_script(code,*args)
        try:
            d.set_window_size(1580,1100);d.get(url)
            wait.until(lambda _:len(d.find_elements('css selector','.episode-row'))==19)
            js("window.workflowErrors=[];addEventListener('error',e=>workflowErrors.push(e.message));addEventListener('unhandledrejection',e=>workflowErrors.push(String(e.reason)))")
            js("window.workflowClicks=[];document.addEventListener('click',e=>{const b=e.target.closest('button');if(b)workflowClicks.push({action:b.dataset.action,step:b.dataset.nextStep,text:b.textContent,next:document.querySelector('#review-next')?.textContent});},true)")
            check('grade filters use plain text and Holo toolbar icons',js('return !document.querySelector(".quality-card svg") && document.querySelector("#sidebar-toggle use").getAttribute("href")==="/sidebar-control.svg#panel"'))
            check('quality overview has five clear entries',len(d.find_elements('css selector','.quality-card'))==5)
            check('work queues reflect prerequisites','18' in el('[data-queue-filter="review"]').text and '1' in el('[data-queue-filter="excluded"]').text and '0' in el('[data-queue-filter="preflight"]').text)
            d.save_screenshot(str(evidence/'workflow-library-dark.png'))
            click('[data-quality-filter="B"]');check('quality card filters real records',len(d.find_elements('css selector','.episode-row'))==18)
            click('[data-quality-filter="A"]');check('empty quality band is honest',not d.find_elements('css selector','.episode-row'))
            click('[data-action="clear-filters"]');check('clear filter restores catalog',len(d.find_elements('css selector','.episode-row'))==19)
            click('[data-queue-filter="excluded"]');check('excluded queue is actionable',len(d.find_elements('css selector','.episode-row'))==1 and '查看原因' in el('.workflow-open').text)
            click('[data-queue-filter="review"]');check('queue entry clears conflicting filters',len(d.find_elements('css selector','.episode-row'))==18 and el('#filter-auto').get_attribute('value')=='')
            fill('#search','episode_000140');click('.workflow-open')
            wait.until(lambda _:len(d.find_elements('css selector','#review-next'))==1)
            wait.until(lambda _:js('return [...document.querySelectorAll("[data-camera]")].length===3 && [...document.querySelectorAll("[data-camera]")].every(i=>i.complete&&i.naturalWidth>0)'))
            click('button[data-stage-layout=balanced]')
            check('desktop prioritizes visualization beside evidence and review tabs',js('const a=document.querySelector(".playback-panel").getBoundingClientRect(),b=document.querySelector(".details-panel").getBoundingClientRect();return a.right<=b.left+1&&a.width>b.width&&Math.abs(a.top-b.top)<3'))
            check('manual decisions have meanings instead of a second letter scale',js('return [...document.querySelectorAll("[data-grade]")].map(b=>b.querySelector("span").textContent).join("|")==="通过|处理后保留|待复核|排除"'))
            click('[data-detail="review"]')
            check('next step asks for review without auto approval','填写审核结论' in el('#review-next').text and not lib.store.all())
            click('[data-action="show-quality"]')
            check('grade reasons precede technical details',len(d.find_elements('css selector','.grade-attention'))>=1 and el('.grade-all-checks').get_attribute('open') is None)
            d.save_screenshot(str(evidence/'workflow-review-dark.png'))
            click('.grade-attention [data-grade-evidence="motion"]')
            wait.until(lambda _:len(d.find_elements('css selector','.event-list-row.selected'))==1)
            check('grade evidence locates a real issue',float(el('#timeline').get_attribute('value'))>0)
            check('locating evidence keeps video visible',js('const r=document.querySelector(".camera-grid").getBoundingClientRect();return r.top>=0&&r.bottom<=innerHeight'))
            click('[data-detail="review"]');click('[data-action="workflow-next"]')
            check('review prompt focuses decision without changing it',js('return document.activeElement.hasAttribute("data-grade")') and not lib.store.all())
            click('[data-grade="B"]')
            check('next step identifies missing task','填写任务指令' in el('#review-next').text)
            click('[data-action="workflow-next"]');check('task action focuses correct field',js('return document.activeElement.id')=='task-instruction')
            fill('#task-instruction','把桌上的物体拿起并放入容器（界面隔离验收）')
            check('unsaved work requires save before preflight','保存审核' in el('#review-next').text)
            check('clip editor starts collapsed',el('#clip-editor').get_attribute('open') is None)
            click('#clip-editor > summary');fill('#clip-start',2);fill('#clip-end',3);fill('#clip-label','界面验收片段');click('[data-action="add-clip"]')
            check('clip summary reflects retained range','1 个保留片段' in el('#clip-summary').text)
            click('[data-action="workflow-next"]')
            wait.until(lambda _:lib.store.all().get('2b38f251e50e',{}).get('revision')==1)
            wait.until(lambda _:js('const b=document.querySelector("[data-action=workflow-next]");return b?.dataset.nextStep==="preflight"&&!b.disabled'))
            saved=lib.store.all()['2b38f251e50e']
            check('semantic decision preserves storage compatibility',saved['grade']=='B' and saved['segments'][0]['start']==2)
            click('[data-action="workflow-next"]');wait.until(lambda _:el('[data-action="confirm-export"]').is_enabled())
            check('guided preflight targets this saved record',el('input[name="export-kind"]:checked').get_attribute('value')=='dataset' and 'episode_000140' in el('#preflight-result').text)
            d.save_screenshot(str(evidence/'workflow-preflight.png'))
            click('[data-action="confirm-export"]')
            wait.until(lambda _:len(d.find_elements('css selector','a[download]'))==1)
            job=next(j for j in lib.exports() if j['state']=='done')
            check('workflow exports the selected clip',job['samples']==30 and job['valid_samples']>=27)
            check('export page distinguishes preflight from completion','可预检 1 条' in el('.export-readiness').text)
            d.save_screenshot(str(evidence/'workflow-exports.png'))
            click('[data-queue-nav="preflight"]');wait.until(lambda _:len(d.find_elements('css selector','.episode-row'))==1)
            check('export readiness returns to correct queue','episode_000140' in el('.ep-name').text)
            click('[data-queue-filter="review"]');fill('#search','episode_000131');click('.select-ep');click('[data-action="export-selected"]')
            click('input[name="export-kind"][value="dataset"]')
            wait.until(lambda _:len(d.find_elements('css selector','[data-preflight-review]'))==1)
            check('blocked preflight cannot submit',not el('[data-action="confirm-export"]').is_enabled())
            check('preflight uses review language','审核结论需为' in el('#preflight-result').text)
            click('[data-preflight-review]')
            wait.until(lambda _:js('return document.querySelector(".review-head h1")?.textContent.includes("episode_000131")'))
            check('preflight repair returns to record without modifying it',el('#dialog').get_attribute('open') is None and len(lib.store.all())==1)
            click('#theme-toggle')
            wait.until(lambda _:js('return document.documentElement.dataset.theme')=='light')
            d.save_screenshot(str(evidence/'workflow-review-light.png'))
            check('light theme retained',js('return document.documentElement.dataset.theme')=='light')
            click('#theme-toggle');d.set_window_size(1280,1000);click('#sidebar-toggle')
            check('1280 collapsed sidebar retains simultaneous evidence',js('const a=document.querySelector(".playback-panel").getBoundingClientRect(),b=document.querySelector(".details-panel").getBoundingClientRect();return a.right<=b.left+1'))
            d.save_screenshot(str(evidence/'workflow-review-1280.png'))
            d.set_window_size(820,1050)
            check('compact review does not overflow',js('return document.documentElement.scrollWidth<=innerWidth'))
            d.save_screenshot(str(evidence/'workflow-review-compact.png'))
            check('no JavaScript errors through the workflow',js('return window.workflowErrors')==[])
            d.get(url+'/#library');wait.until(lambda _:len(d.find_elements('css selector','.quality-card'))==5)
            check('compact overview does not overflow',js('return document.documentElement.scrollWidth<=innerWidth'))
            d.save_screenshot(str(evidence/'workflow-library-compact.png'))
            check('automatic grades unchanged',{r['id']:r['quality']['grading']['grade'] for r in lib.catalog()['episodes']}==original_grades)
            check('real annotations preserved',real.store.all()==before)
            (evidence/'workflow-ui-result.json').write_text(json.dumps(dict(passed=True,checks=checks,isolated_workspace=True),ensure_ascii=False,indent=2))
        finally:
            try:
                d.save_screenshot(str(evidence/'workflow-last-state.png'))
                (evidence/'workflow-clicks.json').write_text(json.dumps(js('return {clicks:window.workflowClicks,errors:window.workflowErrors}'),ensure_ascii=False,indent=2))
            except Exception:pass
            d.quit();http.shutdown();http.server_close();lib.pool.shutdown();lib.export_pool.shutdown()

if __name__=='__main__':main()
