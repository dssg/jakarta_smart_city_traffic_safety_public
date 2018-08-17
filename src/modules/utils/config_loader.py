# ============ Base imports ======================
import os
import glob
# ====== External package imports ================
import yaml
# ====== Internal package imports ================
# ============== Logging  ========================
# =========== Config File Loading ================
# ================================================

jakartapath = os.environ['JAKARTAPATH']
config_path = os.path.join(jakartapath, "config/")


# class that lets us access config parameters using conf.param.param2 instead of conf["param"]["param2"]
class Configuration:
    def __init__(self, conf=None, key=None):
        self.atts = list()
        self.conf = {}
        if conf is not None:
            self.add_keys(conf, key)
            self.conf = conf

    def add_keys(self, conf, key=None):
        if conf is None:
            return
        if key is not None:
            self.atts.append(key)
            self.__setattr__(key, Configuration(conf))
            self.conf[key] = conf
            return
        for key, val in conf.items():
            self.atts.append(key)
            self.conf[key] = val
            if type(val) is dict:
                self.__setattr__(key, Configuration(val))
            else:
                self.__setattr__(key, val)

    def __str__(self):
        s = ""
        for attr in self.atts:
            attr_string = str(self.__getattribute__(attr))
            if "\n" in attr_string:
                for line in str(self.__getattribute__(attr)).split("\n"):
                    s += f"{attr}.{line}\n"
            else:
                s += f"{attr}: {attr_string}\n"
        return s[:-1]  # omit last newline

    def __repr__(self):
        return self.__str__()


def get_config():
    conf = Configuration()
    for config_file in glob.glob(f"{config_path}*.yml") + glob.glob(f"{config_path}*.yaml"):
        conf.add_keys(yaml.load(open(config_file)))
    return conf


def main():
    test = get_config()
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
