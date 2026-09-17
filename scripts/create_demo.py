"""Generate a small, explicitly synthetic G1-shaped MCAP without private data."""
import argparse,io,json,math,sys
from pathlib import Path
from PIL import Image,ImageDraw
from mcap_ros2.writer import Writer

APP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(APP))
from extract import T_LOWSTATE,T_ACTION,T_INSTRUCTION,T_POLICY_MODE,T_ROBOT_STATE,HAND_STATE,HAND_CMD

LOWSTATE='''demo_msgs/Motor[29] motor_state
demo_msgs/Imu imu_state
================================================================================
MSG: demo_msgs/Motor
float32 q
float32 dq
float32 tau_est
float32[2] temperature
float32 vol
================================================================================
MSG: demo_msgs/Imu
float32[3] rpy
float32[3] gyroscope
float32[3] accelerometer
float32[4] quaternion
'''
HAND_STATE_SCHEMA='''demo_msgs/Finger[6] states
================================================================================
MSG: demo_msgs/Finger
float32 q
float32 tau_est
'''
HAND_CMD_SCHEMA='''demo_msgs/Target[6] cmds
================================================================================
MSG: demo_msgs/Target
float32 q
'''
IMAGE_SCHEMA='''std_msgs/Header header
string format
uint8[] data
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
'''

def frame(camera,t):
    """A diagram generated from coordinates, not an edited camera recording."""
    canvas=Image.new('RGB',(640,360),(21,24,31));draw=ImageDraw.Draw(canvas)
    draw.text((24,20),'RoboCurate / SYNTHETIC DEMO',fill=(187,166,240))
    draw.text((24,44),f'{camera}    {t:04.1f} s    NOT REAL ROBOT FOOTAGE',fill=(160,169,188))
    draw.line((25,305,615,305),fill=(67,74,88),width=2)
    draw.rectangle((65,230,210,245),fill=(77,84,101))
    draw.rectangle((425,230,575,245),fill=(77,84,101))
    for x in [80,195,440,560]:draw.line((x,245,x,304),fill=(77,84,101),width=7)
    draw.rectangle((470,188,550,229),outline=(77,175,158),width=4)
    phase=min(1,max(0,(t-1)/4));x=145+365*phase;y=214-65*math.sin(math.pi*phase)
    draw.line((320,110,int(x),int(y)-16),fill=(173,158,219),width=7)
    draw.rounded_rectangle((x-17,y-12,x+17,y+12),radius=5,fill=(228,187,82))
    draw.text((24,325),'Controlled faults: arm offset 1.8-2.4 s; head gap 3.0-3.4 s',fill=(160,169,188))
    out=io.BytesIO();canvas.save(out,format='JPEG',quality=82);return out.getvalue()

def pose(t):
    q=[0.0]*29
    for start in [0,6]:q[start+2]=-.14;q[start+3]=.28;q[start+4]=-.14
    q[15]=.18*math.sin(t);q[19]=.35+.12*math.sin(t)
    q[22]=-.25*math.sin(t);q[26]=.35+.2*math.sin(t)
    return q

def create_demo(output):
    episode=Path(output)/'synthetic_pick_place'/'episode_000001'
    path=episode/'mcap'/'demo.mcap';metadata=episode/'episode_meta.json'
    if path.exists():
        if metadata.is_file() and json.loads(metadata.read_text()).get('generated_by')=='robocurate-synthetic-v1':return path
        raise FileExistsError('Refusing to replace an existing recording')
    path.parent.mkdir(parents=True,exist_ok=True)
    t0=1_700_000_000_123_456_789;duration=6
    instruction='合成演示：将黄色方块从左桌移至右侧容器；不是真实机器人采集。'
    with path.open('xb') as stream,Writer(stream) as writer:
        low=writer.register_msgdef('demo_msgs/msg/LowState',LOWSTATE)
        action=writer.register_msgdef('demo_msgs/msg/Action','float32[] data\n')
        hand_state=writer.register_msgdef('demo_msgs/msg/HandState',HAND_STATE_SCHEMA)
        hand_cmd=writer.register_msgdef('demo_msgs/msg/HandCommand',HAND_CMD_SCHEMA)
        text=writer.register_msgdef('std_msgs/msg/String','string data\n')
        image=writer.register_msgdef('sensor_msgs/msg/CompressedImage',IMAGE_SCHEMA)
        def write(topic,schema,message,t):
            ns=t0+round(t*1e9);writer.write_message(topic,schema,message,log_time=ns,publish_time=ns)
        write(T_INSTRUCTION,text,dict(data=instruction),0)
        write(T_POLICY_MODE,text,dict(data='motion_enabled'),0)
        write(T_ROBOT_STATE,text,dict(data='SYNTHETIC_DEMO'),0)
        for index in range(duration*500+1):
            t=index/500;q=pose(t)
            if 1.8<=t<2.4:q[19]+=.9
            motors=[dict(q=v,dq=0.,tau_est=0.,temperature=[35.,35.],vol=48.) for v in q]
            write(T_LOWSTATE,low,dict(motor_state=motors,imu_state=dict(rpy=[0.,0.,0.],gyroscope=[0.,0.,0.],accelerometer=[0.,0.,9.81],quaternion=[1.,0.,0.,0.])),t)
        for index in range(duration*50+1):
            t=index/50;write(T_ACTION,action,dict(data=pose(t)),t)
            closure=.65 if 1<t<5 else .05
            for side in ['left','right']:
                write(HAND_STATE[side],hand_state,dict(states=[dict(q=closure,tau_est=0.) for _ in range(6)]),t)
                write(HAND_CMD[side],hand_cmd,dict(cmds=[dict(q=closure) for _ in range(6)]),t)
        for camera in ['head','left_wrist','right_wrist']:
            for index in range(duration*30+1):
                t=index/30
                if camera=='head' and 3<=t<3.4:continue
                ns=t0+round(t*1e9)
                write(f'/observations/camera/{camera}/color/image',image,dict(header=dict(stamp=dict(sec=ns//1_000_000_000,nanosec=ns%1_000_000_000),frame_id=camera),format='jpeg',data=frame(camera,t)),t)
    metadata.write_text(json.dumps(dict(generated_by='robocurate-synthetic-v1',task='synthetic_pick_place',
        instruction=instruction,episode_id='episode_000001',metas=dict(head_camera_mode='mono',
        wrist_cameras=['left_wrist','right_wrist']),synthetic=True,license='Apache-2.0',
        known_faults=[dict(kind='tracking',start_s=1.8,end_s=2.4,joint_index=19),
                      dict(kind='head_camera_gap',start_s=3.0,end_s=3.4)],
        limitations='Procedural test fixture; no task-success label and no learned-model performance claim.'),ensure_ascii=False,indent=2)+'\n')
    return path

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=APP/'workspace/demo')
    args=parser.parse_args();print(create_demo(args.output))
