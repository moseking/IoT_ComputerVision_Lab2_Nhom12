"""Export a trusted YOLO checkpoint to a real Python pickle artifact."""
import argparse
import hashlib
import os
import pickle
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.runtime/ultralytics'))

def export_pickle(source, output):
    from ultralytics import YOLO, __version__
    from server.inference import CLASSES
    source, output = Path(source), Path(output)
    if source.suffix != '.pt' or not source.is_file():
        raise ValueError('Source must be an existing trusted .pt checkpoint')
    if output.suffix != '.pkl':
        raise ValueError('Output must use .pkl')
    model = YOLO(str(source)).to('cpu')
    if {int(k):v for k,v in model.names.items()} != CLASSES:
        raise ValueError('Checkpoint class mapping mismatch')
    artifact = {'format':'facemask-yolo-pickle-v1','ultralytics_version':__version__,
                'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'model':model}
    output.parent.mkdir(parents=True,exist_ok=True)
    # Refuse to overwrite a previous exported model.
    with output.open('xb') as stream:
        pickle.dump(artifact,stream,protocol=pickle.HIGHEST_PROTOCOL)
    print('Exported:',output)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    export_pickle(args.source,args.output)
