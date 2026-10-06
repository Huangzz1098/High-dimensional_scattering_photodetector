import os
import h5py
import numpy as np
import scipy.io as sio


def find_largest_hdf5_dataset(h5_file):
    datasets = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            if np.issubdtype(obj.dtype, np.number):
                datasets.append((name, obj.size))

    h5_file.visititems(visitor)

    if not datasets:
        raise ValueError("MAT文件中没有找到数值数据变量")

    datasets.sort(key=lambda x: x[1], reverse=True)
    return datasets[0][0]


def find_largest_mat_array(mat_dict):
    arrays = []

    for name, value in mat_dict.items():
        if name.startswith("__"):
            continue

        if isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.number):
                arrays.append((name, value.size))

    if not arrays:
        raise ValueError("MAT文件中没有找到数值数据变量")

    arrays.sort(key=lambda x: x[1], reverse=True)
    return arrays[0][0]


def load_mat_data(mat_path):
    try:
        mat_dict = sio.loadmat(mat_path)
        variable_name = find_largest_mat_array(mat_dict)
        data = mat_dict[variable_name]

        print("读取普通MAT变量：", variable_name)
        print("MATLAB中尺寸：", data.shape)

        return data, "normal", None

    except NotImplementedError:
        h5_file = h5py.File(mat_path, "r")
        variable_name = find_largest_hdf5_dataset(h5_file)
        data = h5_file[variable_name]

        print("读取HDF5 MAT变量：", variable_name)
        print("HDF5中尺寸：", data.shape)

        return data, "hdf5", h5_file


def get_cube(data, index, expected_samples, mat_type):

    if data.ndim != 4:
        raise ValueError(
            f"标签必须是4维数据，当前尺寸为 {data.shape}"
        )

    if mat_type == "normal":
        # 普通MAT常见格式：[N, 64, 64, C]
        if data.shape[0] == expected_samples:
            cube = np.asarray(data[index, :, :, :], dtype=np.float32)

        # 兼容格式：[64, 64, C, N]
        elif data.shape[-1] == expected_samples:
            cube = np.asarray(data[:, :, :, index], dtype=np.float32)

        else:
            raise ValueError(
                f"无法识别普通MAT标签尺寸 {data.shape}，"
                f"期望样本数为 {expected_samples}"
            )

    elif mat_type == "hdf5":
        # MATLAB v7.3常见反向格式：
        # MATLAB: [N, 64, 64, C]
        # h5py:   [C, 64, 64, N]
        if data.shape[-1] == expected_samples:
            cube = np.asarray(data[:, :, :, index], dtype=np.float32)
            cube = np.transpose(cube, (2, 1, 0))

        # 兼容已经是：[N, 64, 64, C]
        elif data.shape[0] == expected_samples:
            cube = np.asarray(data[index, :, :, :], dtype=np.float32)

        else:
            raise ValueError(
                f"无法识别HDF5标签尺寸 {data.shape}，"
                f"期望样本数为 {expected_samples}"
            )

    else:
        raise ValueError(f"未知MAT类型：{mat_type}")

    if cube.shape[0] != 64 or cube.shape[1] != 64:
        raise ValueError(
            f"第 {index + 1} 个标签空间尺寸错误：{cube.shape}，"
            f"期望为 [64, 64, C]"
        )

    return cube


def convert_mat_to_txt(mat_path, txt_path, expected_samples):
    data, mat_type, h5_file = load_mat_data(mat_path)

    try:
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)

        with open(
            txt_path,
            "w",
            encoding="utf-8",
            buffering=1024 * 1024
        ) as f:
            for index in range(expected_samples):
                cube = get_cube(
                    data=data,
                    index=index,
                    expected_samples=expected_samples,
                    mat_type=mat_type
                )

                cube[cube < 0] = 0

                label = cube.reshape(-1, order="C")

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

                if (index + 1) % 10 == 0 or index + 1 == expected_samples:
                    print(f"{txt_path}: {index + 1}/{expected_samples}")

        print("转换完成：", txt_path)

    finally:
        if h5_file is not None:
            h5_file.close()


if __name__ == "__main__":
    dataset_dir = "./dataset/polarization_caco3"

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