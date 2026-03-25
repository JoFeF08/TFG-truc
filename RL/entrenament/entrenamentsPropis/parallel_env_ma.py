import multiprocessing as mp
import sys
import os

try:
    if '__file__' in globals():
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        sys.path.insert(0, root_path)
except Exception:
    pass

from joc.entorn_ma.env_ma import TrucEnvMa

def worker(remote, parent_remote, env_config, seed):
    parent_remote.close()

    cfg = env_config.copy()
    if seed is not None:
        cfg['seed'] = seed
        import numpy as np
        np.random.seed(seed)

    env = TrucEnvMa(cfg)

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                action = data
                next_s, next_p_id = env.step(action)

                raw = next_s['raw_obs']
                ri = raw.get('reward_intermedis', [0.0, 0.0])
                done = (next_p_id is None)

                if done:
                    next_s, next_p_id = env.reset()

                env.action_recorder = []
                remote.send(((next_s, next_p_id), ri, done))

            elif cmd == 'reset':
                state, player_id = env.reset()
                remote.send((state, player_id))

            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError(f"Comanda '{cmd}' no reconeguda.")
    except BaseException as e:
        print(f"Error al worker SubprocVecEnvMa: {e}")
    finally:
        remote.close()


class SubprocVecEnvMa:
    """
    Versio de SubprocVecEnv per a entrenament per mans (TrucEnvMa).
    """
    def __init__(self, num_envs, env_config):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])

        self.processes = []
        base_seed = env_config.get('seed', 42)

        for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            p = mp.Process(target=worker, args=(work_remote, remote, env_config, base_seed + i))
            p.daemon = True
            p.start()
            self.processes.append(p)
            work_remote.close()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        states_players, rewards, dones = zip(*results)
        return list(states_players), list(rewards), list(dones)

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset_async(self, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        for i in indices:
            self.remotes[i].send(('reset', None))

    def reset_wait(self, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        results = []
        for i in indices:
            results.append(self.remotes[i].recv())
        return results

    def reset_all(self):
        self.reset_async()
        return self.reset_wait()

    def close(self):
        # Enviar comanda de tancament a cada worker
        for remote in self.remotes:
            try:
                remote.send(('close', None))
            except (BrokenPipeError, OSError):
                pass

        # Esperar amb timeout, forçar kill si no responen
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=3)
            if p.is_alive():
                p.kill()
                p.join(timeout=2)

        # Tancar pipes del costat pare
        for remote in self.remotes:
            try:
                remote.close()
            except OSError:
                pass

        self.processes.clear()

    def __del__(self):
        if self.processes:
            self.close()
