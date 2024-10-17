from transbymep import tools, optimize_MEP


if __name__ == "__main__":
    ###############################
    #####  Setup environment  #####
    ###############################

    # Import configuration files
    config = tools.import_run_config('configs/wolfe.yaml')

    output = optimize_MEP(**config)
