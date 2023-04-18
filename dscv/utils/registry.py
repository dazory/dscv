
class Registry(object):
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def register_module(self, module_class):
        self._module_dict[module_class.__name__] = module_class
        return module_class


def build_from_cfg(cfg, registry):
    if isinstance(cfg, list):
        modules = []
        for c in cfg:
            modules.append(build_from_cfg(c, registry))
        return modules
    elif isinstance(cfg, dict):
        _cfg = cfg.copy()
        obj_type = _cfg.pop('type')
        if obj_type not in registry._module_dict:
            raise KeyError('Unrecognized {} type "{}"'.format(
                registry._name, obj_type))
        return registry._module_dict[obj_type](**_cfg)
    else:
        raise TypeError('cfg must be a dict or list, but got {}'.format(type(cfg)))
