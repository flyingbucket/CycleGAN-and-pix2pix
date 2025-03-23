import os
from PIL import Image
from tqdm import tqdm


def merge_images(folder_A, folder_B, output_folder):
    os.makedirs(output_folder, exist_ok=True)  # 确保输出文件夹存在

    # 获取文件列表（假设 A 和 B 里的文件名完全相同）
    files_A = sorted(os.listdir(folder_A))
    files_B = sorted(os.listdir(folder_B))

    # 确保 A 和 B 里图片数量相同
    assert len(files_A) == len(files_B), "A 和 B 目录中的文件数不匹配！"

    for file_A, file_B in tqdm(zip(files_A, files_B), total=len(files_A)):
        if file_A != file_B:  # 确保文件名匹配
            print(f"文件名不匹配: {file_A} vs {file_B}, 跳过")
            continue

        path_A = os.path.join(folder_A, file_A)
        path_B = os.path.join(folder_B, file_B)

        if not (
            path_A.endswith((".png", ".jpg", ".jpeg", ".tif"))
            and path_B.endswith((".png", ".jpg", ".jpeg", ".tif"))
        ):
            print(f"跳过非图片文件: {file_A}, {file_B}")
            continue

        # 打开图片
        img_A = Image.open(path_A)
        img_B = Image.open(path_B)

        # 拼接
        merged = Image.new("RGB", (img_A.width + img_B.width, img_A.height))
        merged.paste(img_A, (0, 0))  # 左边放 A
        merged.paste(img_B, (img_A.width, 0))  # 右边放 B

        # 保存结果
        merged.save(os.path.join(output_folder, file_A))

    print(f"拼接完成！拼接结果保存在: {output_folder}")


if __name__ == "__main__":
    merge_images(
        "~/sr3/dataset/WHU_512", "~/sr3/dataset/opt_512", "~/cg_pix/datasets/sar/train"
    )
merge_images("path/to/A", "path/to/B", "path/to/output")
