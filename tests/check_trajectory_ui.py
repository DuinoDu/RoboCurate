"""Humanoid trajectory workspace acceptance; real media, isolated annotations."""
import json, threading, tempfile, time
from pathlib import Path
from functools import partial
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import server as viewer
from studio import APP, StudioLibrary, Handler


def main():
    evidence=APP/'workspace/evidence';profiles=evidence/'profiles';profiles.mkdir(parents=True,exist_ok=True)
    real=StudioLibrary(APP/'workspace');before=real.store.all();checks=[]
    def check(name,ok):
        assert ok,name
        checks.append(name);print('PASS',name,flush=True)
    with tempfile.TemporaryDirectory(prefix='trajectory-ui-',dir=evidence) as temp:
        lib=StudioLibrary(Path(temp));(Path(temp)/'cache').symlink_to(APP/'workspace/cache',target_is_directory=True)
        lib.add_root(APP.parent/'2026_09_10-17_40_44');lib.add_root(APP.parent/'mcap_viz.tar(1)/mcap_viz/mcap_0.mcap')
        viewer.STATIC_DIR=APP/'static';robot=viewer.Robot(APP/'robot/g1_29dof_rev_1_0.urdf')
        http=viewer.ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,library=lib,robot=robot));threading.Thread(target=http.serve_forever,daemon=True).start()
        options=Options();options.add_argument('-headless')
        d=webdriver.Firefox(options=options,service=Service('/snap/bin/geckodriver',service_args=['--profile-root',str(profiles)],log_output=str(evidence/'trajectory-gecko.log')))
        wait=WebDriverWait(d,75);url=f'http://127.0.0.1:{http.server_address[1]}'
        def el(s):return d.find_element('css selector',s)
        def click(s):el(s).click()
        def js(s,*args):return d.execute_script(s,*args)
        def visible(s):return el(s).is_displayed()
        def overflow():return js('return document.documentElement.scrollWidth>innerWidth')
        try:
            d.set_window_size(1580,1050);d.get(url)
            wait.until(lambda _:len(d.find_elements('css selector','.episode-row'))==19)
            check('workspace styles with scoped progress extension',js('return [...document.styleSheets].length===3&&[...document.styleSheets].every(s=>/workspace.css|progress.css|simplify.css/.test(s.href))'))
            check('continuous scores in catalog',len(d.find_elements('css selector','.row-score'))==18)
            d.save_screenshot(str(evidence/'trajectory-library.png'))
            d.get(url+'/#review/988e894e8afb')
            wait.until(lambda _:len(d.find_elements('css selector','[data-detail="diagnostics"]'))>0)
            click('[data-detail="diagnostics"]')
            wait.until(lambda _:len(d.find_elements('css selector','.event-list-row'))>0)
            js("window.workspaceErrors=[];addEventListener('error',e=>workspaceErrors.push(e.message));addEventListener('unhandledrejection',e=>workspaceErrors.push(String(e.reason)))")
            wait.until(lambda _:el('#robot-host').get_attribute('data-ready')=='true' or bool(d.find_elements('css selector','[data-action=robot-compat]')))
            if d.find_elements('css selector','[data-action=robot-compat]'):
                check('full model never silently degrades',not d.find_elements('css selector','#robot-host canvas'))
                click('[data-action=robot-compat]');wait.until(lambda _:el('#robot-host').get_attribute('data-ready')=='true')
            check('entity renderer loads successfully',el('#robot-host').get_attribute('data-ready')=='true')
            wait.until(lambda _:'个显示点' in el('#trajectory-status').text)
            check('default scene avoids overlay clutter',not el('#show-pose-trail').is_selected() and not el('#show-targets').is_selected())
            click('button[data-stage-layout=balanced]')
            wait.until(lambda _:js('return [...document.querySelectorAll("[data-camera]")].every(i=>i.complete&&i.naturalWidth>0&&i.dataset.frameIndex!==undefined)'))
            check('persistent 3D and three cameras',len(d.find_elements('css selector','#robot-host canvas'))==1 and len(d.find_elements('css selector','[data-camera]'))==3)
            check('visual stage dominates evidence column',js('const a=document.querySelector(".playback-panel").getBoundingClientRect(),b=document.querySelector(".details-panel").getBoundingClientRect();return a.width>b.width*1.8&&a.right<=b.left+1'))
            check('filmstrip eight navigable previews',len(d.find_elements('css selector','[data-film-time]'))==8)
            check('one score-detail entry replaces four repeated dimension buttons',len(d.find_elements('css selector','[data-action="show-quality"]'))==1 and not d.find_elements('css selector','.rating-dimensions>button'))
            print('LAYOUT',js('return [".review-layout",".camera-grid",".scene-stage",".player-controls",".details-panel"].map(s=>{let e=document.querySelector(s),r=e.getBoundingClientRect();return {s,x:r.x,y:r.y,w:r.width,h:r.height,scroll:e.scrollHeight}})'),flush=True)
            canvas=el('#robot-host canvas');renderer=canvas.get_attribute('data-renderer')
            click('#filmstrip [data-film-time]:nth-child(4)');t=float(el('#timeline').get_attribute('value'))
            check('filmstrip seeks the shared playhead',t>15 and t<30)
            wait.until(lambda _:el('#wrist-left').text not in ['—','此刻无有效采样'])
            click('[data-scene-toggle=show-pose-trail]')
            check('past pose stamps are recorded and causal',len(d.find_elements('css selector','.pose-stamp.past'))>0 and all(float(p.get_attribute('data-film-time'))<t for p in d.find_elements('css selector','.pose-stamp')))
            check('stereo image retains native aspect ratio',js('const i=document.querySelector("[data-camera=head]");const r=i.getBoundingClientRect();return Math.abs(r.width/r.height-i.naturalWidth/i.naturalHeight)<.01'))
            click('[data-stage-layout="motion"]');check('3D layout enlarges stage and retains timeline',not visible('.camera-grid') and visible('#timeline') and el('#robot-host canvas')==canvas)
            click('.scene-settings>summary');click('#show-future')
            wait.until(lambda _:len(d.find_elements('css selector','.pose-stamp.future'))>0)
            check('future recorded poses distinguished from prediction','非预测' in el('#scene-pose-times').text)
            Select(el('#pose-scope')).select_by_value('lower')
            check('humanoid lower-body ghost scope',el('#pose-scope').get_attribute('value')=='lower')
            Select(el('#pose-scope')).select_by_value('all');click('.scene-settings>summary')
            d.save_screenshot(str(evidence/'trajectory-recorded-poses.png'))
            pose_time=float(el('.pose-stamp.future').get_attribute('data-film-time'));click('.pose-stamp.future')
            check('recorded pose time seeks all views',abs(float(el('#timeline').get_attribute('value'))-pose_time)<1e-8)
            click('[data-stage-layout="video"]');check('video layout preserves time and model',not visible('.trajectory-panel') and visible('.camera-grid') and abs(float(el('#timeline').get_attribute('value'))-pose_time)<1e-8 and el('#robot-host canvas')==canvas)
            d.save_screenshot(str(evidence/'trajectory-camera-mode.png'))
            click('[data-stage-layout="balanced"]');check('return to 3D keeps the existing renderer',visible('.trajectory-panel') and el('#robot-host canvas')==canvas)
            ray=d.execute_async_script("""const done=arguments[0],channel='trajectory-math-'+Date.now();
              window.addEventListener(channel,e=>done(JSON.parse(e.detail)),{once:true});
              const script=document.createElement('script');script.type='module';
              script.textContent=`import * as THREE from '/viewer/vendor/three.module.js';
                import {TrailLayer} from '/viewer/trails.js';
                import {RobotView} from '/viewer/robot3d.js';
                (async()=>{
                  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(45,1,.1,10);
                  camera.position.set(1,0,4);camera.lookAt(1,0,0);camera.updateMatrixWorld(true);
                  const payload={max_gap_s:.1,tracks:[{source:'state.q',side:'left',part:'wrist',segments:[{times_s:[0,1,2],positions:[[0,0,0],[1,0,0],[2,0,0]]}]}]};
                  const layer=new TrailLayer(scene,payload,0),opts={showTrails:true,showTargets:true,trailSide:'both',trailPart:'wrist',trailWindow:5};scene.updateMatrixWorld(true);
                  layer.update(0,opts);const hidden=layer.pick(camera,new THREE.Vector2(0,0));
                  layer.update(1,opts);const shown=layer.pick(camera,new THREE.Vector2(0,0));layer.destroy();
                  const desc=await fetch('/api/robot').then(r=>r.json()),record=await fetch('/api/trajectories?ep=988e894e8afb').then(r=>r.json());
                  const view=new RobotView(document.createElement('div'),{}, {},desc);
                  view.scene=new THREE.Scene();view.meshGeometries=new Map();view.standHeight=.8;view.recordedPoses=record.poses;
                  Object.assign(view.opts,{showPoseTrail:true,showPast:true,showFuture:true,pastWindow:2,futureWindow:2,poseCount:3,poseOpacity:.2,poseScope:'all'});
                  view._updateRecordedGhosts(23.4);view.scene.updateMatrixWorld(true);
                  let compared=0,maxError=0;
                  for(const ghost of view.poseGhosts) {
                    for(const track of record.tracks.filter(t=>t.source==='state.q')) {
                      for(const segment of track.segments) {
                        const index=segment.times_s.indexOf(ghost.root.userData.recordedTime);if(index<0)continue;
                        const point=ghost.links.get(track.link).group.getWorldPosition(new THREE.Vector3());point.z-=.8;
                        maxError=Math.max(maxError,point.distanceTo(new THREE.Vector3(...segment.positions[index])));compared++;
                      }
                    }
                  }
                  const pool=view.poseGhosts.length;view.opts.showPoseTrail=false;view._updateRecordedGhosts(23.4);
                  const hiddenGhosts=view.poseGhosts.every(g=>!g.root.visible);view.destroy();
                  window.dispatchEvent(new CustomEvent('${channel}',{detail:JSON.stringify({hidden,shown,compared,maxError,pool,hiddenGhosts})}));
                })().catch(e=>window.dispatchEvent(new CustomEvent('${channel}',{detail:JSON.stringify({error:String(e)})})));`;
              script.onerror=()=>done({error:'module loading failed'});document.head.append(script);""")
            print('GEOMETRY_RESULT',ray,flush=True)
            check('3D point picking excludes hidden future points',ray.get('hidden') is None and ray.get('shown')==1)
            check('native wrist and ankle FK matches Three.js posed G1',ray.get('compared',0)>=24 and ray['maxError']<.00002)
            check('full body ghost pool bounded and hidden when disabled',ray.get('pool')==6 and ray.get('hiddenGhosts'))
            d.save_screenshot(str(evidence/'trajectory-review.png'))
            click('.event-list-row');check('evidence label synchronized',el('#scene-evidence').get_attribute('data-active')=='true')
            click('[data-decision="accepted"]')
            click('.detail-more>summary');click('[data-detail="signals"]');wait.until(lambda _:bool(d.find_elements('css selector','#signal-chart')))
            check('switching evidence preserves 3D canvas',el('#robot-host canvas')==canvas)
            click('[data-action="show-quality"]')
            check('rating includes seven humanoid groups',len(d.find_elements('css selector','.body-coordination>div'))==7 and len(d.find_elements('css selector','.rating-dimension'))==4)
            click('.rating-dimension>summary');check('weight and penalty auditable','本项贡献' in el('.rating-dimension').text)
            d.save_screenshot(str(evidence/'trajectory-score.png'))
            Select(el('#trail-part')).select_by_value('foot');Select(el('#trail-window')).select_by_value('0');Select(el('#robot-view')).select_by_value('side')
            check('lower-body mode and full trails',el('#point-left-label').text=='左踝' and el('#robot-view').get_attribute('value')=='side')
            d.save_screenshot(str(evidence/'trajectory-feet.png'))
            Select(el('#trail-part')).select_by_value('all');Select(el('#robot-view')).select_by_value('iso')
            click('[data-detail="review"]');check('manual decision pane accessible',visible('#task-instruction'))
            el('#task-instruction').send_keys('移动到桌前，拿起物体并放入容器。');click('[data-grade="B"]')
            click('[data-detail="diagnostics"]');click('#sidebar-toggle');click('[data-detail="review"]')
            check('draft survives tabs and sidebar','移动到桌前' in el('#task-instruction').get_attribute('value'))
            click('[data-action="save"]');wait.until(lambda _:el('#save-state').text=='已保存')
            saved=lib.store.all()['988e894e8afb']
            check('saved review preserves event decisions',saved['grade']=='B' and saved['instruction'].startswith('移动到桌前') and 'accepted' in saved['event_decisions'].values())
            click('[data-action="reset-playback"]');click('[data-action="frame-next"]')
            wait.until(lambda _:el('[data-camera="head"]').get_attribute('data-frame-index')=='1')
            check('native frame stepping preserved',True)
            click('[data-action="play"]');time.sleep(1.2);click('[data-action="play"]')
            check('playback advances with trajectories',float(el('#timeline').get_attribute('value'))>.5)
            click('[data-camera-expand="left_wrist"]');check('camera enlargement visible',visible('[data-cam-card="left_wrist"]'))
            click('[data-camera-expand="left_wrist"]')
            click('#theme-toggle');check('light theme complete',js('return document.documentElement.dataset.theme')=='light' and not overflow())
            d.save_screenshot(str(evidence/'trajectory-light.png'))
            click('#theme-toggle');d.set_window_size(1366,850);click('[data-detail="diagnostics"]');click('.event-list-row')
            check('laptop workspace no horizontal overflow',not overflow())
            d.save_screenshot(str(evidence/'trajectory-laptop.png'))
            d.set_window_size(700,1000);check('compact layout no overflow',not overflow())
            click('#sidebar-toggle');check('mobile drawer explicit',visible('.sidebar-scrim'))
            ActionChains(d).send_keys(Keys.ESCAPE).perform();check('Escape closes mobile navigation',not visible('.sidebar-scrim'))
            d.save_screenshot(str(evidence/'trajectory-compact.png'))
            check('no uncaught script errors',js('return window.workspaceErrors')==[])
            check('real reviews untouched',real.store.all()==before)
            (evidence/'trajectory-ui-result.json').write_text(json.dumps(dict(passed=True,checks=checks,renderer=renderer),ensure_ascii=False,indent=2))
        finally:
            d.save_screenshot(str(evidence/'trajectory-last-state.png'));d.quit();http.shutdown();http.server_close();lib.pool.shutdown();lib.export_pool.shutdown()

if __name__=='__main__':main()
