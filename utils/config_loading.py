from config.config import cfg


def load_config(config_file: str):
    cfg.merge_from_file(config_file)
    return cfg
