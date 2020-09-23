import olive
import sys
import os

if __name__ == "__main__":
    t = olive.Tree('output', create_folders=False, verbose=False)
    print('loading temp load file')
    sys.stdout.flush()
    t.load_branch('temp\\load')
    while os.path.exists('temp\\load.pkl'):
        try:
            if os.path.isfile('temp\\load.pkl'):
                os.remove('temp\\load.pkl')
        except PermissionError:
            time.sleep(3)
    t.actually_build_output()
    b = t.branches[list(t.branches.keys())[0]]
    print('saving temp reload file')
    sys.stdout.flush()
    b.save_branch('temp\\reload')
