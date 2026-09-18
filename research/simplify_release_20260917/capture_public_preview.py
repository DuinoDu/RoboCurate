"""Capture the current application UI using real, already-scored recordings.

Only presentation controls are used. Production annotations and model results
are read before/after to ensure the capture does not alter them.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait, Select


BASE = 'http://127.0.0.1:8421'
EPISODE = 'b90556bde0e6'
MODEL = 'g1-stage-v3'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--ghosts', action='store_true')
parser.add_argument('--time', type=float, default=12)
args = parser.parse_args()
OUTPUT = Path(__file__).resolve().parent / ('public_preview_ghosts' if args.ghosts else 'public_preview')
OUTPUT.mkdir(exist_ok=True)
PROFILES = OUTPUT / 'browser-profiles'
PROFILES.mkdir(exist_ok=True)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def api(path):
    with opener.open(BASE + path, timeout=30) as response:
        return json.load(response)


def reviews():
    return {row['id']: row['review'] for row in api('/api/catalog')['episodes']}


before = reviews()
progress = api(f'/api/progress?ep={EPISODE}&model_id={MODEL}')
assert progress['state'] == 'ready'
assert progress['result']['validation_status'] != 'test_fixture'
assert progress['result']['feature_valid_count'] >= 32
opts = Options()
opts.add_argument('-headless')
driver = webdriver.Firefox(options=opts, service=Service(
    '/snap/bin/geckodriver', service_args=['--profile-root', str(PROFILES)],
    log_output=str(OUTPUT/'gecko.log')))
wait = WebDriverWait(driver, 60)


def el(selector):
    return driver.find_element('css selector', selector)


def capture(name):
    assert not driver.execute_script('return document.documentElement.scrollWidth > innerWidth')
    driver.execute_script("document.querySelector('#detail-body').scrollTop=0")
    time.sleep(0.5)
    driver.save_screenshot(str(OUTPUT/name))
    print('CAPTURED', name, flush=True)


def seek(seconds):
    driver.execute_script("const e=arguments[0];e.value=String(arguments[1]);e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));", el('#timeline'), seconds)
    wait.until(lambda _: driver.execute_script('const img=document.querySelector("[data-camera=head]"),c=document.querySelector("#frame-counter");return Number(img.dataset.frameIndex)===Number(c.dataset.target)'))


try:
    driver.set_window_size(1800, 1240)
    driver.get(BASE)
    driver.execute_script('localStorage.setItem("robocurate-progress-model",arguments[0])', MODEL)
    driver.get(BASE+'/#review/'+EPISODE)
    wait.until(lambda _: len(driver.find_elements('css selector', '[data-detail="progress"]')) > 0)
    wait.until(lambda _: el('#play-button').is_enabled())
    el('[data-detail="progress"]').click()
    wait.until(lambda _: len(driver.find_elements('css selector', '#progress-timeline .stage-chart')) > 0)
    assert el('#warp-model').get_attribute('value') == MODEL
    wait.until(lambda _: el('#robot-host').get_attribute('data-ready') == 'true'
               or bool(driver.find_elements('css selector', '[data-action="robot-compat"]')))
    if driver.find_elements('css selector', '[data-action="robot-compat"]'):
        el('[data-action="robot-compat"]').click()
    wait.until(lambda _: el('#robot-host').get_attribute('data-ready') == 'true')
    el('[data-stage-layout="balanced"]').click()
    if args.ghosts:
        el('[data-scene-toggle="show-pose-trail"]').click()
        el('[data-scene-toggle="show-targets"]').click()
        el('.scene-settings>summary').click()
        el('#robot-overlay').click()
        Select(el('#past-window')).select_by_value('5')
        Select(el('#trail-window')).select_by_value('5')
        Select(el('#trail-part')).select_by_value('all')
        driver.execute_script("const e=arguments[0];e.value='0.30';e.dispatchEvent(new Event('change',{bubbles:true}));", el('#pose-opacity'))
        el('.scene-settings>summary').click()
        for control in ['show-pose-trail', 'show-targets', 'robot-overlay']:
            assert el('#'+control).is_selected()
        for seconds in [6,12,24]:
            seek(seconds)
            wait.until(lambda _:len(driver.find_elements('css selector','.pose-stamp.past'))==3)
            capture(f'task-workspace-{seconds}.png')
    seek(args.time if args.ghosts else 10)
    wait.until(lambda _: driver.execute_script('return [...document.querySelectorAll("[data-camera]")].every(i=>i.complete&&i.naturalWidth>0&&i.dataset.frameIndex!==undefined)'))
    wait.until(lambda _: driver.execute_script('const img=document.querySelector("[data-camera=head]"),c=document.querySelector("#frame-counter");return Number(img.dataset.frameIndex)===Number(c.dataset.target)'))
    capture('task-workspace.png')
    el('[data-stage-layout="video"]').click()
    capture('task-cameras.png')
    el('[data-stage-layout="motion"]').click()
    el('[data-detail="diagnostics"]').click()
    wait.until(lambda _: len(driver.find_elements('css selector', '.event-list-row')) > 0)
    capture('motion-quality.png')
    assert reviews() == before
    after = api(f'/api/progress?ep={EPISODE}&model_id={MODEL}')
    assert after['result']['signature'] == progress['result']['signature']
    (OUTPUT/'capture-report.json').write_text(json.dumps({
        'episode_id': EPISODE, 'model_id': MODEL,
        'score_signature': progress['result']['signature'],
        'checkpoint_sha256': progress['result']['checkpoint_sha256'],
        'capture': 'Unmodified browser screenshots of the actual local RoboCurate application',
        'recording': 'Real G1 banana-transfer MCAP; no synthetic image or score fixtures',
        'review_state_unchanged': True, 'model_result_unchanged': True,
        'display_time_s': args.time if args.ghosts else 10,
        'recorded_pose_ghosts': args.ghosts, 'control_target_overlay': args.ghosts,
        'screenshots': ['task-workspace.png', 'task-cameras.png', 'motion-quality.png']
    }, ensure_ascii=False, indent=2)+'\n')
finally:
    driver.quit()
