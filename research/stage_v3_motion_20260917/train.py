"""Use the exact shared experiment engine with this separately frozen protocol."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parent
source=ROOT.parent/'stage_v3_20260917/train.py'
spec=importlib.util.spec_from_file_location('v3_experiment',source)
experiment=importlib.util.module_from_spec(spec);spec.loader.exec_module(experiment)
experiment.ROOT=ROOT
if __name__=='__main__':experiment.main()
