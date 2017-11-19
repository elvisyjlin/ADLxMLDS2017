import subprocess
import sys

def bleu_eval(name):
    try:
        out = subprocess.check_output(['python', 
                                       'bleu_eval.py', 
                                       '{}.txt'.format(name)])
        print(out)
    except:
        print(sys.exc_info()[0])

def load_predictions(name, original=False):
    preds = {}
    if not original:
        name = '{}.txt'.format(name)
    with open(name) as f:
        for line in f.readlines():
            id, pred = line.strip().split(',', 1)
            preds[id] = pred
    return preds

def show_predictions(name):
    preds = load_predictions(name)
    for id, pred in preds.items():
        print('{}\t=> {}'.format(id, pred))
            
def special_mission(name, file=None, original=False):
    specials = ['klteYv1Uv9A_27_33.avi', 
                '5YJaS2Eswg0_22_26.avi', 
                'UbmZAe5u5FI_132_141.avi', 
                'JntMAcTlOF0_50_70.avi', 
                'tJHUH9tpqPg_113_118.avi']
    preds = load_predictions(name, original)
    for id in specials:
        print('{}\t=> {}'.format(id, preds[id]))
    if file:
        with open(file, 'w') as f:
            for id in specials:
                f.write('{},{}\n'.format(id, preds[id]))

def forEachEpoch(f, name, start, end, step):
    epochs = range(start, end+1, step)
    for epoch in epochs:
        f('{}_epoch_{}'.format(name, epoch))