import os
import json
import requests
import csv

# 计数器用于记录已删除的空文件夹数
deleted_folders_count = 0


def log_error(folder_path, reason, error_log="get_image_error.csv"):
    """将下载失败的文件夹路径和原因记录到 CSV 文件"""
    file_exists = os.path.isfile(error_log)
    with open(error_log, mode="a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Folder Path", "Reason"])
        writer.writerow([folder_path, reason])


def clear_line():
    """清除终端上一行内容"""
    print("\r", end="", flush=True)
    print(" " * 80, end="\r", flush=True)


import shutil


def delete_folder(folder_path):
    """删除非空文件夹"""
    try:
        shutil.rmtree(folder_path)
        global deleted_folders_count
        deleted_folders_count += 1
        print(
            f"已删除文件夹：{folder_path}，当前已删除空文件夹数：{deleted_folders_count}"
        )
    except Exception as e:
        print(f"删除文件夹 {folder_path} 时发生错误：{e}")


def process_folder(parent_path, folder_name):
    global deleted_folders_count
    folder_path = os.path.join(parent_path, folder_name)
    relative_path = os.path.relpath(folder_path)
    print(f"正在处理文件夹：{relative_path}", end="\r", flush=True)

    try:
        # 查找 JSON 文件
        json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
        if not json_files:
            clear_line()
            print(
                f"文件夹 {relative_path} 中未找到 JSON 文件，正在删除此文件夹...",
                end="\r",
                flush=True,
            )
            delete_folder(folder_path)
            return

        json_path = os.path.join(folder_path, json_files[0])
        clear_line()
        print(f"找到 JSON 文件：{relative_path}/{json_files[0]}", end="\r", flush=True)

        # 加载 JSON 文件
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # 检查 text 字段
        text_field = data.get("text")
        if not text_field:
            clear_line()
            print(
                f"文件夹 {relative_path} 的 JSON 文件中缺少 'text' 字段，正在删除文件夹...",
                end="\r",
                flush=True,
            )
            delete_folder(folder_path)
            return

        # 检查文件夹内是否已有图片
        if any(f.endswith((".jpg", ".jpeg", ".png")) for f in os.listdir(folder_path)):
            clear_line()
            print(f"文件夹 {relative_path} 内已有图片，跳过下载。已完成。")
            return

        # 提取 top_img URL
        top_img_url = data.get("top_img")
        if not top_img_url:
            images = data.get("images", [])
            if images:
                top_img_url = images[0]
            else:
                clear_line()
                print(
                    f"文件夹 {relative_path} 的 JSON 文件中 'top_img' 和 'images' 字段都缺失，正在删除文件夹...",
                    end="\r",
                    flush=True,
                )
                delete_folder(folder_path)
                return

        if not top_img_url.startswith("http"):
            clear_line()
            print(f"无效的图片 URL：{top_img_url}，正在删除文件夹...")
            log_error(relative_path, "无效的图片 URL")
            delete_folder(folder_path)
            return

        # 根据链接动态设置文件扩展名
        file_extension = os.path.splitext(top_img_url)[1].lower()
        if file_extension not in [".jpg", ".jpeg", ".png"]:
            file_extension = ".jpg"  # 默认扩展名

        save_path = os.path.join(
            folder_path, f"top_image_{folder_name}{file_extension}"
        )
        clear_line()
        print(f"开始下载图片：{top_img_url}", end="\r", flush=True)
        if not download_image(top_img_url, save_path):
            clear_line()
            print(
                f"图片下载失败，正在删除文件夹：{relative_path}...",
                end="\r",
                flush=True,
            )
            log_error(relative_path, "图片下载失败")
            delete_folder(folder_path)

    except json.JSONDecodeError as e:
        clear_line()
        print(f"文件夹 {relative_path} 中的 JSON 文件解析失败：{e}，正在删除文件夹...")
        log_error(relative_path, "JSON 解析失败")
        delete_folder(folder_path)


def download_image(image_url, save_path):
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()

        with open(save_path, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

        clear_line()
        print(f"图片已成功保存到：{save_path}")
        return True
    except requests.exceptions.RequestException as e:
        clear_line()
        print(f"下载图片失败：{e}")
        return False


def process_all_folders(base_path):
    print(f"开始处理根目录：{base_path}")
    for root_folder in ["fake", "real"]:
        root_path = os.path.join(base_path, root_folder)
        if not os.path.isdir(root_path):
            print(f"未找到顶层目录：{root_path}")
            continue
        for folder_name in os.listdir(root_path):
            folder_path = os.path.join(root_path, folder_name)
            if os.path.isdir(folder_path):
                process_folder(root_path, folder_name)
    print("所有文件夹处理完成！")


# main
# 使用当前脚本所在的目录作为根目录
base_directory = os.path.abspath(os.path.dirname(__file__))
print(f"根目录已设置为：{base_directory}")

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

print("图片下载完成！")
