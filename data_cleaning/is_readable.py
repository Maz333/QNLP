from PIL import Image, UnidentifiedImageError
import shutil
import os


def is_image_readable(image_path):
    """检查图片是否可识别、是否可转化RGB"""
    try:
        with Image.open(image_path) as img:
            img.verify()  # 验证图片完整性
            img.convert("RGB")
        print(f"图片可识别：{image_path}")
        return True
    except (UnidentifiedImageError, IOError):
        print(f"图片不可识别：{image_path}")
        return False


def delete_folder(folder_path):
    """删除文件夹及其内容"""
    try:
        shutil.rmtree(folder_path)
        print(f"已删除文件夹：{folder_path}")
    except Exception as e:
        print(f"删除文件夹 {folder_path} 时发生错误：{e}")


def process_folder(folder_path):
    """检查文件夹中的图片是否可读，不可读则删除文件夹"""
    print(f"正在处理文件夹：{folder_path}")
    for file_name in os.listdir(folder_path):
        if file_name.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, file_name)
            if not is_image_readable(image_path):
                print(f"发现不可识别图片，删除文件夹：{folder_path}")
                delete_folder(folder_path)
                return  # 停止处理当前文件夹
    print(f"文件夹 {folder_path} 中图片均可识别。")


def process_all_folders(base_directory):
    """遍历根目录中的所有文件夹并处理"""
    print(f"开始处理根目录：{base_directory}")
    for root, dirs, _ in os.walk(base_directory):
        for folder_name in dirs:
            folder_path = os.path.join(root, folder_name)
            process_folder(folder_path)
    print("所有文件夹处理完成。")


# 使用当前脚本所在的目录作为根目录
base_directory = os.path.abspath(os.path.dirname(__file__))

# 获取根目录下的所有子文件夹
subdirectories = [
    d
    for d in os.listdir(base_directory)
    if os.path.isdir(os.path.join(base_directory, d))
]

# 分别对两个文件夹执行
for subdir in subdirectories:
    folder_path = os.path.join(base_directory, subdir)
    print(f"正在处理文件夹：{folder_path}")
    process_all_folders(folder_path)

print("图片可读性验证完成！")
