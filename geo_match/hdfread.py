# -*- coding: utf-8 -*-
"""
读取 HDF 文件基本信息
支持:
    - HDF5 (.h5/.hdf/.HDF)
    - HDF4 (MODIS/FY等)

输出:
    - 文件类型
    - 全局属性
    - 数据集名称
    - shape
    - 数据类型
"""
import os
import h5py
try:
    from pyhdf.SD import SD, SDC
    HAS_PYHDF = True
except:
    HAS_PYHDF = False
def read_hdf5(file_path):
    """读取HDF5"""

    print("=" * 80)
    print(f"文件: {file_path}")
    print("类型: HDF5")

    with h5py.File(file_path, 'r') as f:

        print("\n【全局属性】")
        for k, v in f.attrs.items():
            print(f"{k}: {v}")

        print("\n【数据集】")

        def print_structure(name, obj):

            if isinstance(obj, h5py.Dataset):
                print(f"\n{name}")
                print(f" shape : {obj.shape}")
                print(f" dtype : {obj.dtype}")

                if obj.attrs:
                    print(" attrs:")
                    for k, v in obj.attrs.items():
                        print(f"   {k}: {v}")

        f.visititems(print_structure)


def read_hdf4(file_path):
    """读取HDF4"""

    print("=" * 80)
    print(f"文件: {file_path}")
    print("类型: HDF4")

    hdf = SD(file_path, SDC.READ)

    print("\n【全局属性】")
    attrs = hdf.attributes()

    for k, v in attrs.items():
        print(f"{k}: {v}")

    print("\n【数据集】")

    datasets = hdf.datasets()

    for name, info in datasets.items():

        # info结构:
        # (dim_sizes, data_type, n_attrs)

        dim_sizes = info[0]
        data_type = info[1]

        print(f"\n{name}")
        print(f" shape : {dim_sizes}")
        print(f" dtype : {data_type}")

        ds = hdf.select(name)

        attrs = ds.attributes()

        if attrs:
            print(" attrs:")
            for k, v in attrs.items():
                print(f"   {k}: {v}")


def read_hdf(file_path):

    try:
        read_hdf5(file_path)

    except:

        if HAS_PYHDF:
            try:
                read_hdf4(file_path)
            except Exception as e:
                print(f"读取失败: {e}")
        else:
            print("未安装 pyhdf，无法读取 HDF4")


if __name__ == "__main__":

    hdf_dir = "/data/Data_yuq/testdata/modis"
    for file in os.listdir(hdf_dir):

        if file.endswith((".hdf", ".HDF", ".h5")):

            file_path = os.path.join(hdf_dir, file)

            read_hdf(file_path)