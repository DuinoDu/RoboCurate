"""Read-only browser acceptance on a real, already-scored production recording.

Unlike check_progress_ui.py, this script installs no fixtures or reviews.
It may seek media and fill a draft range, but never saves or exports anything.
"""
import argparse
import json
from pathlib import Path
import urllib.request
from urllib.parse import urlencode

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait

APP = Path(__file__).resolve().parents[1]


def run(args):
    output = Path(args.output).resolve() if args.output else APP/'workspace/evidence/warp/live'
    output.mkdir(parents=True, exist_ok=True)
    profiles = output/'browser-profiles'; profiles.mkdir(exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def api(route):
        with opener.open(args.server+route, timeout=15) as response:
            return json.load(response)
    current = api('/api/progress?'+urlencode(dict(ep=args.episode, model_id=args.model_id)))
    if current['state'] != 'ready':
        raise ValueError('Score this real recording before running live acceptance')
    result = current['result']
    if result['validation_status'] == 'test_fixture' or result['feature_valid_count'] < 32:
        raise ValueError('A real visual-model result is required')
    before = api('/api/catalog')
    checks = []
    def check(name, condition):
        assert condition, name
        checks.append(name); print('PASS', name, flush=True)
    opts = Options(); opts.add_argument('-headless')
    driver = webdriver.Firefox(options=opts, service=Service('/snap/bin/geckodriver',
        service_args=['--profile-root', str(profiles)], log_output=str(output/'gecko.log')))
    wait = WebDriverWait(driver, 60)
    def el(css):
        return driver.find_element('css selector', css)
    try:
        driver.set_window_size(1580, 1050)
        driver.get(args.server)
        driver.execute_script('localStorage.setItem("robocurate-progress-model",arguments[0])', args.model_id)
        driver.get(args.server+'/#review/'+args.episode)
        wait.until(lambda _:len(driver.find_elements('css selector','[data-detail="progress"]'))>0)
        wait.until(lambda _:el('#play-button').is_enabled())
        el('[data-stage-layout="video"]').click()
        el('[data-detail="progress"]').click()
        wait.until(lambda _:len(driver.find_elements('css selector','#progress-timeline .warp-chart path'))>0)
        check('real registered model selected', el('#warp-model').get_attribute('value') == args.model_id)
        check('curve is an actual model result', '测试曲线' not in el('.warp-panel').text)
        check('model provenance is inspectable', result['checkpoint_sha256'] in el('.warp-provenance').get_attribute('textContent'))
        wait.until(lambda _:driver.execute_script("return document.querySelector('.warp-domain')?.textContent.includes('尚未验证')"))
        check('domain transfer remains visible', True)
        check('one synchronized timeline shows the real curve',len(driver.find_elements('css selector','.warp-chart'))==1)
        if result.get('kind')=='stage_progress':
            check('stage result has dedicated chart',len(driver.find_elements('css selector','.stage-chart'))==1)
            check('stage progress never displayed as WARP velocity','平均进展速度' not in el('.warp-panel').text and '虚线 1×' not in el('.warp-panel').text)
            check('post-release is not automatic success','最终结果待核验' in el('.warp-panel').text and '待核验' in el('.warp-panel').text)
            check('stage segments are actionable',len(driver.find_elements('css selector','[data-stage-propose]'))>0)
            if result.get('stage_version',1)>=2:
                el('.warp-settings summary').click()
                el('.stage-evidence-details>summary').click()
                check('measured state modality is visible','身体与双手状态' in el('.warp-source').text)
                check('per-modality coverage is visible','关节状态' in el('.stage-inputs').text and '右腕' in el('.stage-inputs').text)
                check('incomplete inputs carry a review explanation',not result['summary']['degraded_fraction'] or '输入不完整' in el('.warp-panel').text)
                el('.warp-settings summary').click()
                el('.stage-evidence-details>summary').click()
            if result.get('stage_version',1)>=3:
                check('review reasons are present in model output',len(result['review_reasons'])==result['frame_count'])
                check('boundary warnings are disclosed if enabled',not result.get('boundary_review_enabled') or '阶段切换' in el('.warp-provenance').get_attribute('textContent'))
                check('review guard is disclosed if enabled',not result.get('review_guard_enabled') or '阶段证据不足' in el('.warp-provenance').get_attribute('textContent'))
            if result.get('stage_version')==4:
                check('learned boundary signal has explicit availability',len(result['boundary_probability'])==result['frame_count'] and (result['boundary_head_available'] or all(x is None for x in result['boundary_probability'])))
                check('learned boundary review is disclosed if enabled',not result['learned_boundary_review_enabled'] or '阶段过渡待核查' in el('.warp-provenance').get_attribute('textContent'))
        before_seek = float(el('#timeline').get_attribute('value'))
        chart = el('#progress-timeline .warp-chart')
        driver.execute_script("arguments[0].dispatchEvent(new MouseEvent('click',{bubbles:true,clientX:arguments[0].getBoundingClientRect().left+arguments[0].getBoundingClientRect().width*.4}))", chart)
        check('real curve seeks video', float(el('#timeline').get_attribute('value')) != before_seek)
        if result.get('kind')=='stage_progress':
            check('stage readout follows playback',any(name in el('.warp-panel .warp-now').text for name in result['stage_names']))
            el('.stage-evidence-details>summary').click()
            el('.stage-segments summary').click()
            el('[data-stage-propose]').click()
            check('stage segment fills draft only',el('#clip-label').get_attribute('value').startswith('阶段待复核：'))
            el('.stage-segments summary').click()
        candidates = driver.find_elements('css selector','[data-warp-propose]')
        if candidates:
            candidates[0].click()
            check('real candidate creates an editable draft',float(el('#clip-end').get_attribute('value'))>float(el('#clip-start').get_attribute('value')))
        wait.until(lambda _:driver.execute_script("const img=document.querySelector('[data-camera=\"head\"]'),counter=document.querySelector('#frame-counter');return img?.dataset.frameIndex!==undefined && Number(img.dataset.frameIndex)===Number(counter.dataset.target)"))
        check('native camera frame reaches the selected time',True)
        check('desktop has no page overflow',not driver.execute_script('return document.documentElement.scrollWidth>innerWidth'))
        driver.execute_script("document.querySelector('#detail-body').scrollTop=0")
        driver.save_screenshot(str(output/'progress-real-video.png'))
        after = api('/api/catalog')
        def reviews(catalog):
            assert len(catalog['episodes'])>0
            return {r['id']:r['review'] for r in catalog['episodes']}
        check('saved production reviews unchanged',reviews(before)==reviews(after))
        (output/'report.json').write_text(json.dumps(dict(passed=True,checks=checks,
            model_id=args.model_id,episode_id=args.episode,score_signature=result['signature'],
            checkpoint_sha256=result['checkpoint_sha256'],data='Actual MCAP frames, official pretrained DINOv2, locally trained progress head; no synthetic score fixtures'),ensure_ascii=False,indent=2))
    finally:
        driver.quit()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server',default='http://127.0.0.1:8421')
    parser.add_argument('--episode',required=True);parser.add_argument('--model-id',required=True)
    parser.add_argument('--output',default=None)
    run(parser.parse_args())
