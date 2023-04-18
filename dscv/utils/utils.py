
def get_dir_and_file_name(full_path):
    '''
    Examples:
        '/ws/data', None = get_dir_and_file_name('/ws/data')
        '/ws/data', 'img.png' = get_dir_and_file_name('/ws/data/img.png')
        'data', None = get_dir_and_file_name('data')
        None, 'img.png' = get_dir_and_file_name('img.png')
    '''
    name_dir = full_path.split('/')
    name_file = full_path.split('.')

    file = name_dir[-1] if len(name_file) > 1 else None
    dir = full_path if file is None else '/'.join(name_dir[:-1])
    if len(name_dir) == 1 and (not file is None):
        dir = None

    return dir, file

