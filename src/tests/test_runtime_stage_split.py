import importlib.util
import io
from pathlib import Path

from src.vision.audit import read_json,save_json


def load_runner(tmp_path,monkeypatch):
    script=Path(__file__).resolve().parents[2]/'tools/run_shared_groups29.py'
    spec=importlib.util.spec_from_file_location('runtime_runner_fixture',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'PROJECT',tmp_path)
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    return module


def test_runtime_split_resumes_exact_same_command_after_verified_initialization(tmp_path,monkeypatch):
    runner=load_runner(tmp_path,monkeypatch)
    generation=tmp_path/'groups/02_chair/methods/PerLAGE/generation'
    commands=[]
    class Child:
        def __init__(self,index):self.index=index;self.pid=900+index;self.calls=0;self.returncode=None
        def poll(self):
            self.calls+=1
            if self.index==0 and self.calls==1:
                save_json(generation/'initialization/ready.json',dict(status='completed',method='perlage_generation_initialization'))
                return None
            if self.index==0:
                assert (generation/'PAUSE_REQUESTED').is_file()
                path=generation/'latent_generation_refinement'
                save_json(path/'progress.json',dict(status='paused',step=3))
                (path/'checkpoint.pt').write_bytes(b'verified checkpoint fixture')
            self.returncode=1 if self.index==0 else 0
            return self.returncode
    def launch(command,**kwargs):
        commands.append(list(command));return Child(len(commands)-1)
    monkeypatch.setattr(runner.subprocess,'Popen',launch)
    command=['python','-m','pipeline','generate','--config','same_frozen_config.json']
    assert runner.training_child(command,io.StringIO(),{},'02_chair')==0
    assert commands==[command,command]
    assert not (generation/'PAUSE_REQUESTED').exists()
    proof=read_json(generation/'RUNTIME_PROCESS_SPLIT.json')
    assert proof['status']=='completed_in_fresh_process'
    assert proof['numeric_defaults_unchanged'] is True and proof['completed_latent_steps']==3


def test_cached_initialization_failure_is_not_retried_as_a_runtime_split(tmp_path,monkeypatch):
    runner=load_runner(tmp_path,monkeypatch)
    generation=tmp_path/'groups/02_chair/methods/PerLAGE/generation'
    save_json(generation/'initialization/ready.json',dict(status='completed'))
    launches=[]
    class Failed:
        pid=1;returncode=17
        def poll(self):return self.returncode
    def launch(command,**kwargs):launches.append(command);return Failed()
    monkeypatch.setattr(runner.subprocess,'Popen',launch)
    assert runner.training_child(['same command'],io.StringIO(),{},'02_chair')==17
    assert len(launches)==1 and not (generation/'PAUSE_REQUESTED').exists()
