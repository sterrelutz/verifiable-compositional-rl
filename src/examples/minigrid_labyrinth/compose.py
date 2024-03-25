
def compose(controller_list, objective):
    """
    Function that composes subcontrollers into a meta-policy that achieves the given objective.

    Inputs
    controller_list : list of subcontrollers
    objective : objective:
    """

    policy = controller_list

    return policy