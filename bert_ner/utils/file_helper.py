# coding=utf8

import os


def mk_data_folder():
    data_folder = get_data_folder()
    os.makedirs(data_folder, exist_ok=True)


def get_data_folder():
    cur_dir, _ = os.path.split(__file__)
    return os.path.join(cur_dir, os.pardir, "data")


def get_data_file(file_name):
    data_dir = get_data_folder()
    return os.path.join(data_dir, file_name)


def get_project_file(file_name):
    cur_dir, _ = os.path.split(__file__)
    return os.path.join(cur_dir, os.pardir, file_name)


def is_file_exists(file_name):
    return os.path.exists(file_name)


def mk_folder_for_file(file_name):
    """
    为file_name创建其文件夹
    :param file_name:
    :return:
    """
    dir_name = os.path.dirname(file_name)
    os.makedirs(dir_name, exist_ok=True)


if __name__ == "__main__":
    print(("data_folder", get_data_folder()))
    print(("config_file", get_project_file("config.ini")))
