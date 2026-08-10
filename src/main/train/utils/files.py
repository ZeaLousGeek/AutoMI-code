import errno
import glob
import os
import time

from src.main.train.utils.results import matching_files


def is_folder_accessible(folder_path):
    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在。")
        return False

    try:
        os.listdir(folder_path)

        test_file = os.path.join(folder_path, '.test_access')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)

        return True

    except PermissionError as e:
        print(f"PermissionError: {e}. 文件夹可能被占用或权限不足。")
        return False
    except OSError as e:
        if e.errno == errno.EACCES:
            print(f"权限被拒绝: {e}")
        else:
            print(f"OSError: {e}")
        return False
    except Exception as e:
        print(f"发生未知错误: {e}")
        return False




def wait_until_folder_available(folder_path, check_interval=5):
    while not is_folder_accessible(folder_path):
        print(f"文件夹 {folder_path} 被占用，等待 {check_interval} 秒后重试...")
        time.sleep(check_interval)
    print(f"文件夹 {folder_path} 可用。")


def replace_first_zero(s: str) -> str:
    if '0' in s:
        return s.replace('0', '/', 1)
    return s


def find_directory(directories=None, target_folder='bcicIV2a'):
    if directories is None:
        root = os.environ.get("AUTOMI_DATA_ROOT")
        directories = [root] if root else []
    for directory in directories:
        if os.path.isdir(os.path.join(directory, target_folder)):
            return directory
    return None


def delete_pth_files(directory):
    pth_files = glob.glob(os.path.join(directory, "*.pth"))

    for file in pth_files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except OSError as e:
            print(f"Error deleting {file}: {e}")


def remove_last_path_component(file_path):
    last_slash_index = file_path.rfind('\\')

    if last_slash_index != -1:
        return file_path[:last_slash_index]
    else:
        return file_path


def delete_multi_pth_files(find_path="logs",
                           start_date='2022090101',
                           end_date='2026063099',
                           datasets=['bcicIV2a', 'OpenBMI'],
                           modes=['Cl_FS'],
                           models=['SKLightNet']):
    from src.main.train.utils.configs import get_project_rootpath
    find_path = os.path.join(get_project_rootpath(), find_path)
    matching_excel_paths = matching_files(find_path, 'subject_mean_max_acc_of_', start_date, end_date, datasets, modes,
                                          models)

    for matching_excel_path in matching_excel_paths:
        matching_excel_path = remove_last_path_component(matching_excel_path)
        delete_pth_files(matching_excel_path)


def replace_specific_in_folder(folder_path, old_str, new_str):
    parent_dir = os.path.dirname(folder_path)
    folder_name = os.path.basename(folder_path)

    if old_str in folder_name:
        new_folder_name = folder_name.replace(old_str, new_str)
        new_folder_path = os.path.join(parent_dir, new_folder_name)

        os.rename(folder_path, new_folder_path)
        print(f"文件夹名称已修改为: {new_folder_name}")
    else:
        new_folder_path = folder_path
        print(f"文件夹名称不包含 '{old_str}'，无需修改。")

    for filename in os.listdir(new_folder_path):
        file_path = os.path.join(new_folder_path, filename)

        if os.path.isfile(file_path):
            if old_str in filename:
                new_filename = filename.replace(old_str, new_str)
                new_file_path = os.path.join(new_folder_path, new_filename)
                os.rename(file_path, new_file_path)
                print(f"文件名称已修改为: {new_filename}")
        elif os.path.isdir(file_path):
            replace_specific_in_folder(file_path, old_str, new_str)


def multi_replace_specific_in_folder(find_path="logs",
                           start_date='2022091101',
                           end_date='2026121799',
                           datasets=['bcicIV2a', 'OpenBMI'],
                           modes=['Cl_FS'],
                           models=['TSNet0TSNet'],
                           old_str = 'TSNet0TSNet',
                           new_str = 'TSNet0D1Conv'):
    from src.main.train.utils.configs import get_project_rootpath
    find_path = os.path.join(get_project_rootpath(), find_path)
    matching_excel_paths = matching_files(find_path, 'subject_mean_max_acc_of_', start_date, end_date, datasets, modes,
                                          models)

    for matching_excel_path in matching_excel_paths:
        matching_excel_path = remove_last_path_component(matching_excel_path)
        replace_specific_in_folder(matching_excel_path, old_str, new_str)



