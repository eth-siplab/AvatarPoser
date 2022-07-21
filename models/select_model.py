
"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model_AMASS(opt):
    model = opt['model']      # one input: L

    if model == 'plain':
        from models.model_plain import ModelPlain as M


    elif model == 'amass':  # two inputs: L, C
        from models.model_amass import ModelAMASS as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
