import os, shlex, shutil, argparse, subprocess, json, sys, contextlib, shelve

@contextlib.contextmanager
def null_resource():
    yield None

def show(cmd):
    if isinstance(cmd, str):
        print(cmd)
    else:
        print(shlex.join(cmd))
    sys.stdout.flush()

def run(cmd, **kwargs):
    show(cmd)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kwargs)

def eval(cmd, check=True, **kwargs):
    show(cmd)
    return subprocess.run(cmd,
                          capture_output=True,
                          shell=isinstance(cmd, str),
                          check=check,
                          **kwargs).stdout.decode('utf-8').strip()

def run_perf_report(driver, f, mlir=False, env=None):
    print(json.dumps(env, indent=4))
    cmd = [driver, 'time', f, '--exhaustive-tune']
    if mlir:
        cmd.append('--mlir')
    r = eval(cmd, check=False, env=env)
    for line in r.splitlines():
        if not 'Total time:' in line:
            continue
        t = float(line.replace('Total time:', '').replace('ms', '').strip())
        print('{}ms'.format(t))
        return t
    return 0.0

def compare_perf(driver, f, env1, env2):
    print("Compare perf for {}".format(f))
    p1 = run_perf_report(driver, f, mlir=True, env=env1)
    p2 = run_perf_report(driver, f, mlir=False, env=env2)
    return [p1,p2]

def compare_mlir(driver, f, env, db):
    p1 = {'MIGRAPHX_MLIR_USE_SPECIFIC_OPS': 'dot,fused,attention,convolution','MIGRAPHX_MLIR_TUNE_EXHAUSTIVE' : '1'} | env
    p2 = {'MIGRAPHX_DISABLE_MLIR': '1'} | env
    if db != None:
        if f in db:
            print(f"Skipping {f}")
            return db[f]
    result = compare_perf(driver, f, p1, p2)
    if db != None:
        print(f"Saving {f}")
        db[f] = result
    return result

def optional_open(f):
    if f:
        return shelve.open(f)
    else:
        return null_resource()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='*')
    parser.add_argument('-d', '--driver-path', default='./bin/driver')
    parser.add_argument('--problem-cache', '-c', nargs='?')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    env = {'MIGRAPHX_DISABLE_PASSES': 'auto_contiguous'}
    if(args.problem_cache):
        env['MIGRAPHX_PROBLEM_CACHE'] = os.path.abspath(args.problem_cache)
    
    ts = []
    with optional_open(args.output) as db:
        ts = [compare_mlir(args.driver_path, i, env, db) for i in args.inputs]

    for i,t in zip(args.inputs, ts):
        print(i, t[0], t[1], t[1]/t[0])

if __name__ == "__main__":
    main()
