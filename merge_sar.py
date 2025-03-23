import os
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
from tqdm import tqdm


def merge(folder_A, folder_B, file, files_B, output_folder):
    if file not in files_B:
        print("文件名不匹配 跳过")

    path_A = os.path.join(folder_A, file)
    path_B = os.path.join(folder_B, file)

    if not (
        path_A.endswith((".png", ".jpg", ".jpeg", ".tif"))
        and path_B.endswith((".png", ".jpg", ".jpeg", ".tif"))
    ):
        print(f"跳过非图片文件: {file}, {file}")

    # 打开图片
    img_A = Image.open(path_A)
    img_B = Image.open(path_B)

    # 拼接
    merged = Image.new("RGB", (img_A.width + img_B.width, img_A.height))
    merged.paste(img_A, (0, 0))  # 左边放 A
    merged.paste(img_B, (img_A.width, 0))  # 右边放 B

    # 保存结果
    merged.save(os.path.join(output_folder, file))


def merge_images(folder_A, folder_B, output_folder):
    os.makedirs(output_folder, exist_ok=True)  # 确保输出文件夹存在

    # 获取文件列表（假设 A 和 B 里的文件名完全相同）
    files_A = sorted(os.listdir(folder_A))
    files_B = sorted(os.listdir(folder_B))

    # for file in tqdm((files_A), total=len(files_A)):

    task = []
    with ProcessPoolExecutor() as extc:
        for file in files_A:
            task.append(
                extc.submit(merge, folder_A, folder_B, file, files_B, output_folder)
            )
            # merge(file, files_B, output_folder)
        for t in tqdm(task):
            t.result()
    print(f"拼接完成！拼接结果保存在: {output_folder}")


if __name__ == "__main__":
    merge_images(
        "/home/flyingbucket/sr3/dataset/WHU_512", "/home/flyingbucket/sr3/dataset/opt_512", "/home/flyingbucket/cg_pix/datasets/sar/train"
    )
