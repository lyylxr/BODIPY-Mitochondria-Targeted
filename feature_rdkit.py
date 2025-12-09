import os
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
import pandas as pd
from collections import defaultdict


def create_unified_fingerprint_dict(smiles_list, output_folder):
    """创建统一的摩根指纹字典，并将特征的SVG图像保存到指定文件夹"""
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 存储所有物质中出现的所有特征
    all_features = defaultdict(int)
    feature_idx_map = {}
    next_idx = 0

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # 获取摩根指纹和特征信息
        fp_info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=327680, bitInfo=fp_info)

        # 收集当前分子的所有特征ID
        for feature_id, feature_info in fp_info.items():
            # 如果是新特征，添加到字典
            if feature_id not in feature_idx_map:
                # 获取特征对应的子结构
                atom_idx, radius = feature_info[0]

                # 使用Draw.DrawMorganBit绘制SVG图像并保存到文件
                svg_filename = os.path.join(output_folder, f"feature_{next_idx}.svg")
                img = Draw.DrawMorganBit(mol, feature_id, fp_info, useSVG=True)
                with open(svg_filename, 'w') as f:
                    f.write(img)

                feature_idx_map[feature_id] = {
                    'idx': next_idx,
                    'svg_file': svg_filename  # 存储SVG文件路径
                }
                next_idx += 1
            all_features[feature_id] += 1
    return feature_idx_map


def generate_fingerprint(smiles, feature_idx_map):
    """基于统一字典生成单个物质的指纹"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 获取分子特征
    fp_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2,nBits=327680, bitInfo=fp_info)

    # 生成指纹向量
    fingerprint = [0] * len(feature_idx_map)

    # 在统一字典中设置出现的特征
    for feature_id in fp_info.keys():
        if feature_id in feature_idx_map:
            idx = feature_idx_map[feature_id]['idx']
            fingerprint[idx] = 1

    return fingerprint


def extract_descriptors(mol):
    """提取一些常见的分子描述符作为特征"""
    features = {
        'MolecularWeight': Descriptors.MolWt(mol),  # 分子量
        'NumHDonors': Descriptors.NumHDonors(mol),  # 氢键供体数量
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),  # 氢键受体数量
        'LogP': Descriptors.MolLogP(mol),  # LogP值
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),  # 可旋转键数量
        'TPSA': Descriptors.TPSA(mol),  # 拓扑极性表面积
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),  # 芳香环数量
        'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),  # 饱和环数量
        'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),  # 杂原子数量
        'FractionCSP3': Descriptors.FractionCSP3(mol),  # 三级碳原子比例
        'HeavyAtomCount': mol.GetNumHeavyAtoms(),  # 重原子数量
        'RingCount': Descriptors.RingCount(mol)  # 环数量
    }
    return features


# 主程序
if __name__ == "__main__":
    # 参数设置
    INPUT_FILE = "Data-target.xlsx"  # 输入Excel文件名
    OUTPUT_FILE = "compound_features.xlsx"  # 输出文件名
    SHEET_NAME = "Sheet1"  # Excel工作表名
    SMILES_COL = "SMILES"  # SMILES列名
    OUTPUT_FOLDER = "output_svgs"  # 输出SVG图像的文件夹

    # 1. 读取Excel文件
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

    # 2. 创建统一的指纹字典
    feature_idx_map = create_unified_fingerprint_dict(df[SMILES_COL], OUTPUT_FOLDER)
    print(f"创建的统一字典包含 {len(feature_idx_map)} 个特征")

    # 3. 为每个物质生成指纹和提取描述符
    fingerprints = []
    descriptors_list = []
    for smiles in df[SMILES_COL]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fingerprints.append([0] * len(feature_idx_map))
            descriptors_list.append({key: None for key in extract_descriptors(mol).keys()})
            continue

        fp = generate_fingerprint(smiles, feature_idx_map)
        fingerprints.append(fp if fp is not None else [0] * len(feature_idx_map))

        descriptors = extract_descriptors(mol)
        descriptors_list.append(descriptors)

    # 4. 将指纹和描述符添加到DataFrame，但DataFrame只保留Number和FL两列
    df = df[['Number', 'FL']]
    feature_columns = [f'Feature_{idx}' for idx in range(len(feature_idx_map))]
    fingerprint_df = pd.DataFrame(fingerprints, columns=feature_columns)
    descriptors_df = pd.DataFrame(descriptors_list)
    df = pd.concat([df, descriptors_df,fingerprint_df], axis=1)

    # 5. 保存结果
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"结果已保存至 {OUTPUT_FILE}")