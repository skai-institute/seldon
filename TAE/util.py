def fill_config(config):
    # TODO: not finished yet
    config["exp_params"]["config"]["params"]["dataset"] = config["dataset"]["name"]
    config["logging_params"]["name"] = config["model_params"]["name"]
    return config
