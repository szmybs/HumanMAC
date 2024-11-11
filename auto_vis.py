import os
import sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())


actions = [
    'walking', 'sitting', 'eating', 'posing', 'smoking', 'phoning', 
    'directions', 'discussion', 'greeting', 'purchases', 'sittingdown',
    'takingphoto', 'waiting', 'walkingdog', 'walkingtogether',
]

for action in actions:
    vis = os.path.join(os.getcwd(), 'main.py')
    cmd = 'python ' + vis + ' --action ' + action + ' --mode ' + 'vis'
    print(cmd)
    os.system(cmd)