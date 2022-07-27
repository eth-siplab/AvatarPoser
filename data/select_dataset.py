def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()

    if dataset_type in ['amass']:
        from data.dataset_amass import AMASS_Dataset as D


    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
