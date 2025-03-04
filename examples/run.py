from ase.io import read
from Popcornn import tools, optimize_MEP


if __name__ == "__main__":
    ###############################
    #####  Setup environment  #####
    ###############################

    # Import configuration files
    args = tools.build_default_arg_parser().parse_args()
    config = tools.import_run_config(args.config)
    
    # Parse images
    if isinstance(config["images"], str) and config["images"].endswith(".xyz"):
        config["images"] = read(config["images"], ":")
        
    output = optimize_MEP(**config)
