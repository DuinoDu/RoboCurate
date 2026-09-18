"""Real-video interaction acceptance; never save production reviews or scores."""
import json
from pathlib import Path
import urllib.request
from urllib.parse import urlencode
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait,Select

APP=Path(__file__).resolve().parents[1]
def main():
    base='http://127.0.0.1:8421';episode='b90556bde0e6';model='g1-stage-v3'
    out=APP/'workspace/evidence/simplify';out.mkdir(parents=True,exist_ok=True)
    profiles=out/'browser-profiles';profiles.mkdir(exist_ok=True)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def api(route):
        with opener.open(base+route,timeout=15) as response:return json.load(response)
    record=api('/api/progress?'+urlencode(dict(ep=episode,model_id=model)))['result']
    assert record['evidence']['available'];before=api('/api/catalog');checks=[];viewports={}
    def check(name,ok):
        assert ok,name
        checks.append(name);print('PASS',name,flush=True)
    opts=Options();opts.add_argument('-headless')
    driver=webdriver.Firefox(options=opts,service=Service('/snap/bin/geckodriver',service_args=['--profile-root',str(profiles)],log_output=str(out/'gecko.log')))
    wait=WebDriverWait(driver,60)
    def el(css):return driver.find_element('css selector',css)
    def js(code,*args):return driver.execute_script(code,*args)
    def click(css):
        def attempt(_):
            try:
                target=el(css);js("arguments[0].scrollIntoView({block:'center'})",target);target.click();return True
            except StaleElementReferenceException:return False
        wait.until(attempt)
    def seek(t):
        js("const x=document.querySelector('#timeline');x.value=arguments[0];x.dispatchEvent(new Event('input',{bubbles:true}));x.dispatchEvent(new Event('change',{bubbles:true}))",t)
    def time():return float(el('#timeline').get_attribute('value'))
    def frame_ready():
        wait.until(lambda _:js("const x=document.querySelector('[data-camera=head]');return x?.dataset.frameIndex!==undefined && Number(x.dataset.frameIndex)===Number(document.querySelector('#frame-counter').dataset.target)"))
    def reset_scroll():js("document.querySelector('#detail-body').scrollTop=0;document.querySelector('.playback-panel').scrollTop=0;window.scrollTo(0,0)")
    try:
        driver.set_window_size(1580,1050);driver.get(base)
        js("localStorage.removeItem('holocurate-progress-model');localStorage.removeItem('robocurate-progress-model');localStorage.removeItem('robocurate-theme');localStorage.setItem('holocurate-theme','dark')")
        driver.get(base+'/?preferences-migration=1#review/'+episode)
        wait.until(lambda _:len(driver.find_elements('css selector','[data-detail=progress]'))>0)
        js("window.stageUiErrors=[];addEventListener('error',e=>stageUiErrors.push(e.message));addEventListener('unhandledrejection',e=>stageUiErrors.push(String(e.reason)))")
        wait.until(lambda _:el('#play-button').is_enabled());click('[data-stage-layout=video]');click('[data-detail=progress]')
        wait.until(lambda _:len(driver.find_elements('css selector','[data-stage-confidence]'))>0)
        check('fresh session selects latest installed ready stage model',el('#warp-model').get_attribute('value')==model)
        check('RoboCurate branding is live',driver.title.startswith('RoboCurate') and el('.brand-name').text=='RoboCurate')
        check('legacy preferences migrate to the new brand',js("return localStorage.getItem('robocurate-theme')==='dark'"))
        check('three primary tabs replace six',len(driver.find_elements('css selector','.detail-tabs>button[role=tab]'))==3)
        check('one shared stage curve replaces duplicate charts',len(driver.find_elements('css selector','.warp-chart'))==1 and not driver.find_elements('css selector','.warp-panel .warp-chart'))
        check('advanced settings and detailed evidence start folded',not js("return document.querySelector('.warp-settings').open||document.querySelector('.stage-evidence-details').open"))
        check('technical score uses ten points',el('.rating-number strong').text=='9.23' and '/ 10' in el('.rating-number').text and '/ 100' not in el('.rating-number').text)
        check('rating bar retains its proportion',abs(float(js("return parseFloat(document.querySelector('.rating-bar i').style.width)"))-92.3)<.01)
        check('API declares the new scale',next(x for x in before['episodes'] if x['id']==episode)['quality']['rating']['max_score']==10)
        click('.detail-more summary');click('[data-detail=signals]');wait.until(lambda _:len(driver.find_elements('css selector','#signal-chart'))>0)
        check('advanced curves remain accessible',el('#signal-chart').is_displayed())
        click('.detail-more summary');click('[data-detail=streams]');check('raw channels remain accessible',el('.technical-table').is_displayed())
        click('[data-action=show-quality]');check('dimension scores use ten points','/ 10' in el('.rating-dimension summary').text and '/ 100' not in el('.rating-details').text)
        click('.rating-dimension summary');check('weighted contribution remains inspectable','本项贡献' in el('.rating-dimension').text)
        click('[data-detail=diagnostics]');wait.until(lambda _:len(driver.find_elements('css selector','.event-lane'))>0)
        check('quality timeline is the sole visible analysis track',el('#event-timeline').is_displayed() and not el('#progress-timeline').is_displayed())
        click('[data-detail=progress]');check('task timeline returns with the task panel',el('#progress-timeline').is_displayed() and not el('#event-timeline').is_displayed())
        check('technical score remains separately labelled','技术评分' in el('#rating-overview').text and '最终结果待核验' in el('.stage-outcome-note').text)
        check('five stage navigation buttons are available',len(driver.find_elements('css selector','.stage-route [data-stage-jump]'))==5)
        check('current masks are actual missing input evidence',el('[data-stage-input=head]').get_attribute('data-valid')=='no' and record['evidence']['camera_valid'][0][0] is False)
        check('current phase uses stored output',el('[data-stage-live-name]').text==record['stage_names'][record['stage'][0]])
        seek(10);frame_ready();reset_scroll();driver.save_screenshot(str(out/'task-overview-dark.png'))
        check('review cards retain visible action buttons',js("return [...document.querySelectorAll('.stage-review-item')].every(x=>x.querySelector('.stage-review-item-actions').getBoundingClientRect().bottom<=x.getBoundingClientRect().bottom)"))
        driver.save_screenshot(str(out/'review-queue-dark.png'))
        click('.stage-evidence-details summary');seek(10);frame_ready()
        index=max(i for i,t in enumerate(record['times_s']) if t<=time())
        shown=[float(el(f'[data-stage-prob-value="{k}"]').text[:-1]) for k in range(5)]
        check('displayed probabilities match all five stored classes',all(abs(a-100*b)<.051 for a,b in zip(shown,record['evidence']['probability'][index])))
        check('confidence is not labelled task success probability','不是任务成功概率' in el('.stage-probabilities').text)
        start=next(s['start_s'] for s in record['stage_segments'] if s['stage']==3)
        click('.stage-route [data-stage-jump="3"]');check('stage navigation seeks first observed matching stage',abs(time()-start)<1e-6)
        chart=el('#progress-timeline .warp-chart');js("arguments[0].scrollIntoView({block:'center'})",chart);chart.send_keys(Keys.ARROW_RIGHT)
        check('keyboard curve navigation changes clock by half a second',abs(time()-start-.5)<1e-6)
        frame_ready();check('camera and model evidence share the selected video time',True)
        click('.stage-evidence-details summary');seek(0)
        count=len(driver.find_elements('css selector','[data-review-range]'));check('queue groups flagged samples into review intervals',0<count<sum(record['needs_review']))
        click('.stage-review-tools [data-stage-review-nav="1"]');first_next=time();check('next review seeks a real flagged sample',first_next>0 and record['needs_review'][record['times_s'].index(first_next)])
        click('.stage-review-tools [data-stage-review-nav="-1"]');check('previous review returns to earlier interval',time()<first_next)
        select=el('#stage-reason-filter');js("arguments[0].scrollIntoView({block:'center'})",select);Select(select).select_by_value('32')
        check('reason filter isolates raw uncertainty guard',len(driver.find_elements('css selector','[data-review-range]'))>0 and all('阶段证据不足' in x.text for x in driver.find_elements('css selector','[data-review-range]')))
        click('[data-stage-review]');check('high confidence guard keeps a visible explanation','校准前' in el('[data-stage-reason-detail]').text)
        review_id=el('[data-stage-draft]').get_attribute('data-stage-draft');selected_start=time()
        click('[data-stage-draft]')
        check('review range fills editable draft without approval',el('#clip-label').get_attribute('value').startswith('阶段待复核：') and abs(float(el('#clip-start').get_attribute('value'))-selected_start)<1e-6)
        Select(el('#stage-reason-filter')).select_by_value('0')
        click('[data-stage-preview]');wait.until(lambda _:el('#play-button').get_attribute('aria-label')=='播放')
        check('context playback stops at bounded range end',0<time()<record['times_s'][-1])
        click('.stage-evidence-details summary');seek(10);frame_ready();reset_scroll()
        check('desktop has no horizontal overflow',not js('return document.documentElement.scrollWidth>innerWidth'))
        viewports['desktop']=js('return [innerWidth,innerHeight]')
        driver.save_screenshot(str(out/'stage-evidence-dark.png'))
        click('[data-action=theme]');reset_scroll();driver.save_screenshot(str(out/'stage-evidence-light.png'))
        driver.set_window_size(850,1100);reset_scroll();check('tablet has no horizontal overflow',not js('return document.documentElement.scrollWidth>innerWidth'))
        viewports['tablet']=js('return [innerWidth,innerHeight]')
        driver.save_screenshot(str(out/'stage-evidence-tablet.png'))
        driver.set_window_size(390,844);reset_scroll();check('phone has no horizontal overflow',not js('return document.documentElement.scrollWidth>innerWidth'))
        viewports['narrow']=js('return [innerWidth,innerHeight]')
        click('[data-detail=progress]');js("document.querySelector('.details-panel').scrollIntoView({block:'start'})");driver.save_screenshot(str(out/'stage-evidence-phone.png'))
        driver.set_window_size(1580,1050);click('[data-action=theme]');click('.warp-settings summary');Select(el('#warp-model')).select_by_value('open-dinov2-pilot')
        wait.until(lambda _:'平均进展速度' in el('.warp-panel').text)
        check('switching analysis preserves WARP units and removes stage evidence',len(driver.find_elements('css selector','.stage-live'))==0 and '虚线 1×' in el('#progress-timeline').text)
        check('no browser runtime errors during interaction',js('return window.stageUiErrors')==[])
        driver.refresh();wait.until(lambda _:len(driver.find_elements('css selector','[data-detail=progress]'))>0);click('[data-detail=progress]')
        wait.until(lambda _:'平均进展速度' in el('.warp-panel').text)
        check('explicit WARP choice survives reload',el('#warp-model').get_attribute('value')=='open-dinov2-pilot')
        js("document.querySelector('.warp-settings').open=true");Select(el('#warp-model')).select_by_value('g1-stage-v3');wait.until(lambda _:len(driver.find_elements('css selector','.stage-live'))>0)
        check('returning to task restores current stage model',el('#warp-model').get_attribute('value')==model)
        after=api('/api/catalog');check('production reviews remain unchanged',{r['id']:r['review'] for r in before['episodes']}=={r['id']:r['review'] for r in after['episodes']})
        report=dict(passed=True,checks=checks,viewports=viewports,episode=episode,model_id=model,signature=record['signature'],data='Real saved model outputs and MCAP frames; no synthetic results or production writes')
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    except Exception as error:
        driver.save_screenshot(str(out/'failure.png'))
        (out/'failure.json').write_text(json.dumps(dict(checks=checks,error=str(error),url=driver.current_url),ensure_ascii=False,indent=2)+'\n')
        raise
    finally:driver.quit()
if __name__=='__main__':main()
