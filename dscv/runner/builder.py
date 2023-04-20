from dscv.utils import Registry, build_from_cfg


RUNNERS = Registry('runners')


def build_runner(cfg, model, optimizer, distributed=False):
    runner = build_from_cfg(cfg, RUNNERS, model=model, optimizer=optimizer, distributed=distributed)
    return runner
