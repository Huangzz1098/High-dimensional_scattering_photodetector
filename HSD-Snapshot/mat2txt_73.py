import os
import h5py
import numpy as np


def find_largest_dataset(h5_file):
    """查找MAT文件中最大的数值数据集。"""
    datasets = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append((name, obj.size))

    h5_file.visititems(visitor)

    if not datasets:
        raise ValueError("MAT文件中没有找到数据变量")

    datasets.sort(key=lambda x: x[1], reverse=True)
    return datasets[0][0]


def convert_mat_to_txt(
    mat_path,
    txt_path,
    expected_samples
):
    with h5py.File(mat_path, "r") as mat_file:
        variable_name = find_largest_dataset(mat_file)
        data = mat_file[variable_name]

        print("读取变量：", variable_name)
        print("HDF5中尺寸：", data.shape)

        os.makedirs(
            os.path.dirname(txt_path),
            exist_ok=True
        )

        with open(
            txt_path,
            "w",
            encoding="utf-8",
            buffering=1024 * 1024
        ) as f:

            for index in range(expected_samples):

                # MATLAB v7.3中的维度通常反向存储：
                # MATLAB: [N, 64, 64, 100]
                # h5py:   [100, 64, 64, N]
                if (
                    data.ndim == 4
                    and data.shape[-1] == expected_samples
                ):
                    cube = np.asarray(
                        data[:, :, :, index],
                        dtype=np.float32
                    )

                    # [光谱, 宽, 高] → [高, 宽, 光谱]
                    cube = np.transpose(
                        cube,
                        (2, 1, 0)
                    )

                # 防止MAT文件本身就是[N,64,64,100]
                elif (
                    data.ndim == 4
                    and data.shape[0] == expected_samples
                ):
                    cube = np.asarray(
                        data[index, :, :, :],
                        dtype=np.float32
                    )

                else:
                    raise ValueError(
                        f"无法识别标签尺寸 {data.shape}，"
                        f"期望包含 {expected_samples} 个样本"
                    )

                if cube.shape[0] != 64 or cube.shape[1] != 64:
                    raise ValueError(
                        f"第 {index + 1} 个标签空间尺寸错误："
                        f"{cube.shape}"
                    )

                cube[cube < 0] = 0

                # 按照 [64,64,光谱点] 展平，
                label = cube.reshape(-1, order="C")

                # 与图片名严格对应
                image_name = f"{index + 1:012d}"

                f.write(image_name)
                f.write(" ")
                f.write(
                    " ".join(
                        format(value, ".8g")
                        for value in label
                    )
                )
                f.write("\n")

                if (
                    (index + 1) % 10 == 0
                    or index + 1 == expected_samples
                ):
                    print(
                        f"{txt_path}: "
                        f"{index + 1}/{expected_samples}"
                    )

    print("转换完成：", txt_path)


if __name__ == "__main__":
    dataset_dir = "./dataset/datasets_color_plate_high_dim"

    convert_mat_to_txt(
        mat_path=os.path.join(dataset_dir, "train.mat"),
        txt_path=os.path.join(dataset_dir, "train.txt"),
        expected_samples=8000
    ) # 训练集图片数量

    convert_mat_to_txt(
        mat_path=os.path.join(dataset_dir, "test.mat"),
        txt_path=os.path.join(dataset_dir, "test.txt"),
        expected_samples=2000
    ) # 测试集图片数量

