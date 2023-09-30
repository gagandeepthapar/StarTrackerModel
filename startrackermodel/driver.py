"""
driver

Used for user-facing tool interaction.
Handles:
    - cmd flags/CLI tool version
    - dialog box
    - reformatting data into singular style
    - pass user-provided data into computation tool
    - I/O

startrackermodel
"""

import argparse
import logging

import data.CONSTANTS as const

# Set project level defaults
logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s] %(levelname)s [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
)
logger = logging.getLogger("driver")


def parse_arguments() -> argparse.Namespace:
    """
    Parse arguments from command line. If none are recevied, "
    "open dialog box for user to enter in information.

    Returns:
        arparse.Namespace: struct containing command line flags and arguments
    """

    # program info
    parser = argparse.ArgumentParser(
        prog="python -m driver",
        usage="python -m driver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""\
Estimate accuracy and precision of a star tracker 
via error propagation from hardware, software, and environmental sources."
                                                                
                                                            
                                                            
                                                            
                                                  +         
                                              .-#@@         
                                   @@+           @@ .::     
                                  :=-=           .          
                                                            
                         @@                                 
                 +@@@   :@@@                                
              #@@@@@@@: @@@@@            @@                 
           +@@@@@@@@@@@@@@@@@@*         : .-                
        #@@@@@@@@@@@@@@@@@@@@@@%                            
     -@@@@@@@@@@@@@@@@@@@@@@@@@@@                           
   @@@@@@@@@@@@@@@@@@@@@@@@                                 
    @@@@@@@@@@@@@@@@@@@@@@@@+                               
     @@@@@@@@@@@@@@@@@@@@@@@@                               
      @@@@@@@@@@@@@@@@@@@@%                                 
       #@@@@@@@@@@@@@@@#                                    
        -@@@@@@@@@@@*                                       
          @@@@@@@+                                          
           @@@+                                             
                                                            
                                                            
                                                            """,
        epilog="Source and bug tracker found at the following site:\n\t"
        "https://github.com/gagandeepthapar/StarTrackerModel"
        "",
    )

    # logger level set
    parser.add_argument(
        "-log",
        "--logger",
        metavar="",
        type=str,
        help="Set logging level: "
        "(D)ebug, (I)nfo, (W)arning, (E)rror, or (C)ritical. Default (D)ebug.",
        default="Debug",
    )

    # plotting arguments
    parser.add_argument(
        "--showplot",
        metavar="",
        type=bool,
        help="Show results of accuracy and precision in plots. Defaults to true.",
        default=True,
    )

    parser.add_argument(
        "--saveplot",
        metavar="",
        type=bool,
        help="Save all results of accuracy and precision in plots. "
        "Saves in media/ folder in root directory. Defaults to false.",
        default=False,
    )

    args = parser.parse_args()

    # get logging level from dict stored in CONSTANTS
    logger.setLevel(const.level_hash.get((args.logger).upper()))
    logger.debug("CMD Arguments: %s", args)

    return args


if __name__ == "__main__":
    arg = parse_arguments()
