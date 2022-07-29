import arrow


def write_info(path, option):
    info_file = open(path/'info.txt', 'w')
    time = arrow.now()
    print(f'Current time: {time}', file = info_file)
    option_dict = vars(option)
    for key, value in option_dict.items():
        print(f'{key}: {value}', file = info_file)
        
    