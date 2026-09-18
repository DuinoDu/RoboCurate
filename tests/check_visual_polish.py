"""Same-record visual captures and playback measurements, no review writes."""
import argparse,json,threading,tempfile,time,os,io
from PIL import Image
from selenium.webdriver.common.action_chains import ActionChains
from pathlib import Path
from functools import partial
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait,Select
import server as viewer
from studio import APP,StudioLibrary,Handler


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--headed',action='store_true');parser.add_argument('--phase',default='after');parser.add_argument('--static-dir',type=Path);parser.add_argument('--seconds',type=float,default=15);args=parser.parse_args()
    out=APP/'workspace/evidence/visual-polish';out.mkdir(parents=True,exist_ok=True);profiles=APP/'workspace/evidence/profiles';profiles.mkdir(exist_ok=True)
    original=StudioLibrary(APP/'workspace');reviews=original.store.all()
    with tempfile.TemporaryDirectory(prefix='visual-',dir=out) as tmp:
        lib=StudioLibrary(Path(tmp));(Path(tmp)/'cache').symlink_to(APP/'workspace/cache',target_is_directory=True);lib.add_root(APP.parent/'2026_09_10-17_40_44')
        viewer.STATIC_DIR=args.static_dir or APP/'static';http=viewer.ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,library=lib,robot=viewer.Robot(APP/'robot/g1_29dof_rev_1_0.urdf')));threading.Thread(target=http.serve_forever,daemon=True).start()
        o=Options();
        if not args.headed:o.add_argument('-headless')
        else:o.binary_location='/snap/firefox/current/usr/lib/firefox/firefox'
        d=webdriver.Firefox(options=o,service=Service('/snap/bin/geckodriver',env=dict(os.environ),service_args=['--profile-root',str(profiles)],log_output=str(out/(args.phase+'-gecko.log'))));wait=WebDriverWait(d,50)
        def el(s):return d.find_element('css selector',s)
        def click(s):el(s).click()
        def js(code,*values):return d.execute_script(code,*values)
        def settings():
            if el('.scene-settings').get_attribute('open') is None:click('.scene-settings>summary')
        def toggle(selector,on):
            settings()
            if el(selector).is_selected()!=on:click(selector)
        def close_settings():
            if el('.scene-settings').get_attribute('open') is not None:click('.scene-settings>summary')
        def capture(name):
            close_settings();time.sleep(.15);d.save_screenshot(str(out/(args.phase+'-'+name+'.png')));el('#robot-host canvas').screenshot(str(out/(args.phase+'-'+name+'-scene.png')))
        try:
            d.set_window_size(1580,1050);d.get(f'http://127.0.0.1:{http.server_address[1]}/#review/988e894e8afb')
            wait.until(lambda _:el('#robot-host').get_attribute('data-ready')=='true' or bool(d.find_elements('css selector','[data-action=robot-compat]')))
            if not args.headed and d.find_elements('css selector','[data-action=robot-compat]'):
                click('[data-action=robot-compat]');wait.until(lambda _:el('#robot-host').get_attribute('data-ready')=='true')
            assert el('#robot-host').get_attribute('data-ready')=='true',el('#robot-host').text
            if args.headed:assert el('#robot-host canvas').get_attribute('data-renderer')=='webgl'
            wait.until(lambda _:'个显示点' in el('#trajectory-status').text)
            js("window.visualErrors=[];addEventListener('error',e=>visualErrors.push(e.message));addEventListener('unhandledrejection',e=>visualErrors.push(String(e.reason)))")
            click('#filmstrip button:nth-child(4)');click('button[data-stage-layout=motion]')
            capture('default')
            toggle('#show-pose-trail',False);toggle('#show-targets',False);toggle('#show-trails',False);capture('solid')
            toggle('#show-trails',True);toggle('#show-pose-trail',True);toggle('#show-future',True);toggle('#show-targets',True);capture('comparison')
            if args.headed:
                canvas=el('#robot-host canvas');im=Image.open(io.BytesIO(canvas.screenshot_as_png)).convert('RGB');w,h=im.size
                candidates=[(x,y) for y in range(int(h*.18),int(h*.7),2) for x in range(int(w*.35),int(w*.65),2) if min(im.getpixel((x,y)))>170 and max(im.getpixel((x,y)))-min(im.getpixel((x,y)))<25]
                assert candidates,'No clear opaque body pixels were found'
                x,y=min(candidates,key=lambda p:(p[0]-w/2)**2+(p[1]-h*.38)**2)
                old_time=float(el('#timeline').get_attribute('value'));size=canvas.size
                ActionChains(d).move_to_element_with_offset(canvas,int(x*size['width']/w-size['width']/2),int(y*size['height']/h-size['height']/2)).click().perform()
                assert abs(float(el('#timeline').get_attribute('value'))-old_time)<1e-8,'Opaque body selected a hidden ghost'

            if d.find_elements('css selector','#scene-focus'):
                Select(el('#scene-focus')).select_by_value('upper');capture('upper')
                Select(el('#scene-focus')).select_by_value('lower');Select(el('#trail-part')).select_by_value('foot');capture('lower')
                Select(el('#scene-focus')).select_by_value('all');Select(el('#trail-part')).select_by_value('wrist')
            click('button[data-stage-layout=balanced]');click('[data-action=reset-playback]')
            wait.until(lambda _:el('[data-camera=head]').get_attribute('data-frame-index')=='0')
            js("window.renderStamps=[];window.renderCosts=[];window.drawObserver=new MutationObserver(rs=>{for(const r of rs)if(r.attributeName==='data-rendered-faces'){renderStamps.push(performance.now());let m=Number(r.target.dataset.renderMs);if(Number.isFinite(m))renderCosts.push(m)}});drawObserver.observe(document.querySelector('#robot-host canvas'),{attributes:true});window.basePaints=[...document.querySelectorAll('[data-camera]')].map(i=>Number(i.dataset.paintCount||0));window.measureStart=performance.now()")
            click('[data-action=play]');time.sleep(args.seconds);click('[data-action=play]')
            result=js("const elapsed=(performance.now()-measureStart)/1000,sort=renderCosts.toSorted?renderCosts.toSorted((a,b)=>a-b):[...renderCosts].sort((a,b)=>a-b);return {seconds:elapsed,playhead:Number(document.querySelector('#timeline').value),renderer:document.querySelector('#robot-host canvas').dataset.renderer,render_fps:renderStamps.length/elapsed,render_p95_ms:sort.length?sort[Math.floor(sort.length*.95)]:null,render_max_ms:sort.length?sort.at(-1):null,camera_paints:[...document.querySelectorAll('[data-camera]')].map((i,n)=>Number(i.dataset.paintCount||0)-basePaints[n]),errors:visualErrors}")
            js('drawObserver.disconnect()');assert not result['errors'],result['errors'];assert result['playhead']>args.seconds*.65,result
            capture('playback')
            # Model lifetime and exact paused time are retained through layout changes.
            canvas=el('#robot-host canvas');position=el('#timeline').get_attribute('value');click('button[data-stage-layout=video]');click('button[data-stage-layout=motion]');assert el('#robot-host canvas')==canvas and el('#timeline').get_attribute('value')==position
            result['same_canvas_and_time_after_layout_switch']=True
            assert original.store.all()==reviews and not lib.store.all();result['reviews_unchanged']=True
            (out/(args.phase+'-result.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
        finally:
            d.save_screenshot(str(out/(args.phase+'-last.png')));d.quit();http.shutdown();http.server_close();lib.pool.shutdown();lib.export_pool.shutdown()

if __name__=='__main__':main()
