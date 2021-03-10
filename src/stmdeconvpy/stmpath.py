import os
binpath = os.path.dirname(os.path.realpath(__file__)) + "/../../bin"

def binexecute(name):
    os.system("python "+binpath+"/"+name)

