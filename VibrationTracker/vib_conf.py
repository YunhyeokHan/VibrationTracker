LISTBOX_MIMETYPE = "application/x-item"

OP_NODE_INPUT = 1
OP_NODE_OUTPUT = 2
OP_NODE_ADD = 3
OP_NODE_SUB = 4
OP_NODE_MUL = 5
OP_NODE_DIV = 6

# add node op codes here
# OP_NODE_...
OP_NODE_IMPORTIMAGES = 7

# node for calibrating camera
OP_NODE_CALIBRATECAMERA = 8
OP_NODE_INITIALIZETARGET = 9
OP_NODE_TRACKTARGET = 10
OP_NODE_ESTIMATEPOSE = 11
OP_NODE_CALCULATEDISPLACEMENT = 12

# node for preprocessing dic images
OP_NODE_PREPROCESSDIC = 13
OP_NODE_PROCESSDIC = 14
OP_NODE_POSTPROCESSDIC = 15

# node for animating the vibration
OP_NODE_ANIMATEVIBRATION = 16

VIB_NODES = {
}


class ConfException(Exception): pass
class InvalidNodeRegistration(ConfException): pass
class OpCodeNotRegistered(ConfException): pass


def register_node_now(op_code, class_reference):
    if op_code in VIB_NODES:
        raise InvalidNodeRegistration("Duplicate node registration of '%s'. There is already %s" %(
            op_code, VIB_NODES[op_code]
        ))
    VIB_NODES[op_code] = class_reference


def register_node(op_code):
    def decorator(original_class):
        register_node_now(op_code, original_class)
        return original_class
    return decorator

def get_class_from_opcode(op_code):
    if op_code not in VIB_NODES: raise OpCodeNotRegistered("OpCode '%d' is not registered" % op_code)
    return VIB_NODES[op_code]





# import all nodes and register them
from VibrationTracker.nodes import *