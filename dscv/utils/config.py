"""
REF: https://github.com/maet3608/config-dict/blob/master/config.py
Configuration dictionary
"""
import copy
import os
import collections


class Config(dict):
    def __init__(self, *args, **kwargs):
        """ Construct the same way a plain Python dict is created. """
        wrap = lambda v: Config(v) if type(v) is dict else v
        kvdict = {k: wrap(v) for k, v in dict(*args, **kwargs).items()}
        super(Config, self).update(kvdict)

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value

    def copy(self, del_type=False):
        data = Config()
        for k, v in self.items():
            if isinstance(v, Config):
                data[k] = v.copy()
            else:
                data[k] = copy.deepcopy(v)
        if del_type:
            data.pop('type')
        return data

    def _update(self, d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = self._update(d.get(k, {}), v)
            else:
                if not d.get(k):
                    d[k] = v
        return d

    def load(self, filepath):
        """Load configuration from given JSON filepath"""
        items = dict()
        cfg_txt = open(filepath).read()
        exec(cfg_txt, items)

        if items.get('_base_'):
            for _module in items['_base_']:
                parent_path = '/'.join(filepath.split('/')[:-1])
                data = self.load(os.path.join(parent_path, _module))
                items = self._update(items, data)

        if items.get('__builtins__'):
            del items['__builtins__']

        wrap = lambda v: Config(v) if type(v) is dict else v
        kvdict = {k: wrap(v) for k, v in items.items()}
        super(Config, self).__init__(kvdict)
        return kvdict
