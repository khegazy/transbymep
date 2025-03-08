from ase.io import read, write
from popcornn import tools, optimize_MEP


if __name__ == "__main__":
    ###############################
    #####  Setup environment  #####
    ###############################

    # Import configuration files
    args = tools.build_default_arg_parser().parse_args()
    config = tools.import_run_config(args.config)
    
    # Run the optimization
    final_images, ts_image = optimize_MEP(**config)
    
    # Write the final images
    write('configs/6445_popcornn.xyz', final_images)
