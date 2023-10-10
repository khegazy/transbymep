
class logging():
    def __init__(self):
        print("init logging")
    
    def training_logger(self, step, val):
        step_string = ("step: " + str(step)).ljust(15)
        val_string = "val: " + str(val)
        print(step_string, val_string)