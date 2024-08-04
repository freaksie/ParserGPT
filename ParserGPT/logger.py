import logging
import os
from datetime import datetime


# get custom path from user
def set_logger(dir_loc:str = os.getcwd()):
    LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y')}.log"
    logs_path=os.path.join(dir_loc,"logs",LOG_FILE)
    os.makedirs(logs_path, exist_ok=True)
    LOG_FILE_PATH=os.path.join(logs_path, LOG_FILE)

    logging.basicConfig(
        filename=LOG_FILE_PATH,
        format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        filemode='w'
    )

if __name__=="__main__":
    set_logger('/home/neelvora/Projects')
    logging.info("Logging has started")
    