import os
import shutil
import csv
import json

# 设置根目录路径和目标文件夹路径
base_directory = os.path.abspath(
    os.path.dirname(__file__)
)  # 使用当前所在的目录作为根目录
img_directory = os.path.join(base_directory, "img")  # 存放图片的文件夹
os.makedirs(img_directory, exist_ok=True)  # 创建 img 文件夹（如果不存在）
data_directory = os.path.join(base_directory, "data")  # 存放csv的文件夹
os.makedirs(data_directory, exist_ok=True)  # 创建 data 文件夹（如果不存在）


def get_image_file(folder_path):
    """从文件夹中找到第一张图片"""
    for file in os.listdir(folder_path):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")):
            return file
    return None


def process_folder(parent_path, folder_name, label, train_data, test_data, total_count):
    """处理单个文件夹，并分配到训练集和测试集"""
    folder_path = os.path.join(parent_path, folder_name)
    relative_path = os.path.relpath(folder_path)

    # 查找图片文件
    image_filename = get_image_file(folder_path)
    if not image_filename:
        print(f"文件夹 {relative_path} 缺少图片，跳过处理。")
        return

    image_source_path = os.path.join(folder_path, image_filename)
    image_target_path = os.path.join(img_directory, image_filename)

    # 复制图片到 img 文件夹
    shutil.copy(image_source_path, image_target_path)

    # 查找 JSON 文件
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    for json_file in json_files:
        json_path = os.path.join(folder_path, json_file)

        # 加载 JSON 文件
        with open(json_path, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
                text = data.get("text", "").strip()

                # 如果文本内容为空，跳过处理
                if not text:
                    print(f"JSON 文件 {json_path} 的 'text' 字段为空，跳过处理。")
                    continue

                # 添加数据到相应的列表
                content_data = {
                    "content": text,
                    "label": label,
                    "image": image_filename,
                }

                # 按比例分配到测试集或训练集
                if len(test_data) < total_count * 0.1:  # 测试集比例10%
                    test_data.append(content_data)
                else:
                    train_data.append(content_data)

            except json.JSONDecodeError as e:
                print(f"文件 {json_path} 解析失败：{e}")
            except Exception as e:
                print(f"处理文件 {json_path} 时发生错误：{e}")


def write_to_csv(csv_file, data):
    """将数据写入 CSV 文件"""
    with open(csv_file, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["content", "label", "image"])
        writer.writeheader()
        writer.writerows(data)


def count_csv_data(csv_file):
    """统计 CSV 文件中 fake 和 real 的数量"""
    fake_count = 0
    real_count = 0
    with open(csv_file, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["label"] == "0":
                fake_count += 1
            elif row["label"] == "1":
                real_count += 1
    return fake_count, real_count


def process_dataset(dataset_name):
    """处理单个数据集"""
    train_data = []
    test_data = []
    total_data_count = 0

    dataset_path = os.path.join(base_directory, dataset_name)

    # 遍历 fake 和 real 两个顶层目录
    for sub_folder, label in [("fake", 0), ("real", 1)]:
        sub_path = os.path.join(dataset_path, sub_folder)
        if not os.path.isdir(sub_path):
            print(f"未找到目录：{sub_path}")
            continue

        # 统计文件夹总数
        total_data_count += len(
            [
                folder
                for folder in os.listdir(sub_path)
                if os.path.isdir(os.path.join(sub_path, folder))
            ]
        )

        for folder_name in os.listdir(sub_path):
            folder_path = os.path.join(sub_path, folder_name)
            if os.path.isdir(folder_path):
                process_folder(
                    sub_path,
                    folder_name,
                    label,
                    train_data,
                    test_data,
                    total_data_count,
                )

    # 写入 CSV 文件
    os.chdir(data_directory)  # 切换当前工作目录到 data 文件夹
    train_csv = f"{dataset_name}_train.csv"
    test_csv = f"{dataset_name}_test.csv"
    write_to_csv(train_csv, train_data)
    write_to_csv(test_csv, test_data)

    # 统计并打印每个 CSV 的 fake 和 real 数据量
    train_fake_count, train_real_count = count_csv_data(train_csv)
    test_fake_count, test_real_count = count_csv_data(test_csv)

    print(f"{dataset_name} 数据集处理完成！")
    print(
        f"训练集: 总数={train_fake_count + train_real_count} (fake={train_fake_count}, real={train_real_count})"
    )
    print(
        f"测试集: 总数={test_fake_count + test_real_count} (fake={test_fake_count}, real={test_real_count})"
    )


def process_all_datasets():
    """处理所有数据集"""
    for dataset_name in ["politifact", "gossip"]:
        process_dataset(dataset_name)


# 执行文件处理
process_all_datasets()
print("所有数据集处理完成！")
