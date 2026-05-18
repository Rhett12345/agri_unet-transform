# -*- coding: utf-8 -*-
"""
读取 NetCDF(.nc) 文件基本信息
包括：
1. 全局属性
2. 维度信息
3. 变量信息
4. 每个变量的属性
"""

from netCDF4 import Dataset
import sys


def read_nc_info(nc_file):
    # 打开文件
    ds = Dataset(nc_file, 'r')

    print("=" * 80)
    print("文件:", nc_file)
    print("=" * 80)

    # =====================
    # 全局属性
    # =====================
    print("\n【全局属性】")

    attrs = ds.ncattrs()

    if attrs:
        for attr in attrs:
            value = getattr(ds, attr)
            print(f"{attr}: {value}")
    else:
        print("无全局属性")

    # =====================
    # 维度信息
    # =====================
    print("\n【维度信息】")

    for dim_name, dim in ds.dimensions.items():
        print(
            f"{dim_name:<20} "
            f"size={len(dim):<10} "
            f"unlimited={dim.isunlimited()}"
        )

    # =====================
    # 变量信息
    # =====================
    print("\n【变量信息】")

    for var_name, var in ds.variables.items():

        print("\n" + "-" * 60)
        print(f"变量名: {var_name}")
        print(f"数据类型: {var.dtype}")
        print(f"维度: {var.dimensions}")
        print(f"形状: {var.shape}")

        # 变量属性
        attrs = var.ncattrs()

        if attrs:
            print("\n属性:")
            for attr in attrs:
                value = getattr(var, attr)

                # 太长则截断
                text = str(value)
                if len(text) > 200:
                    text = text[:200] + " ..."

                print(f"  {attr}: {text}")
        else:
            print("无属性")

    ds.close()

    print("\n完成")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("使用方式:")
        print("python read_nc.py your_file.nc")
        sys.exit()

    nc_file = sys.argv[1]

    read_nc_info(nc_file)