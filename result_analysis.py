import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score
import datetime

# 获取当前时间

def process_file(filepath, model_name):
    """
    解析单个知识追踪结果txt文件。
    
    假定文件是空白符（如tab或空格）分隔的。
    将每个学生的时间步数据聚合成单行，包含完整的序列。
    """
    try:
        # 使用 \s+ 作为分隔符来匹配一个或多个空白字符
        df = pd.read_csv(filepath, sep=r'\s+', engine='python')
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return pd.DataFrame()

    # 1. 清理数据：过滤掉混入的表头行
    # 我们通过qidx列是否为数字来判断。如果不是数字（如 'qidx' 字符串），则丢弃
    df = df[pd.to_numeric(df['qidx'], errors='coerce').notnull()]

    # 2. 转换必要的数据类型
    df['qidx'] = df['qidx'].astype(int)
    df['late_trues'] = pd.to_numeric(df['late_trues'], errors='coerce')
    df['late_mean'] = pd.to_numeric(df['late_mean'], errors='coerce')
    
    # 确保 'questions' 和 'concepts' 是字符串
    df['questions'] = df['questions'].astype(str)
    df['concepts'] = df['concepts'].astype(str)

    # 3. 按 'orirow' 分组，并聚合每个学生的完整序列
    
    # 先按 orirow 和 qidx 排序，确保时间步正确
    df = df.sort_values(by=['orirow', 'qidx'])
    
    grouped = df.groupby('orirow')
    
    student_data = []
    # 遍历所有分组（每个学生）
    for orirow_id, group in grouped:
        # 拼接字符串序列
        questions_seq = ",".join(group['questions'])
        concepts_seq = ",".join(group['concepts'])
        
        # 收集 trues 和 preds 列表用于计算AUC
        trues_seq = list(group['late_trues'])
        preds_seq = list(group['late_mean'])
        
        student_data.append({
            'questions': questions_seq,
            'concepts': concepts_seq,
            f'trues_{model_name}': trues_seq,
            f'preds_{model_name}': preds_seq
        })

    return pd.DataFrame(student_data)

def calculate_auc(trues, preds):
    """
    安全地计算AUC分数，处理只有一个类别的情况。
    """
    try:
        # 检查真实标签是否只有单一类别（如全0或全1）
        if len(set(trues)) < 2:
            return np.nan
        return roc_auc_score(trues, preds)
    except Exception:
        return np.nan

def merge_kt_results(file_paths, model_names):
    """
    融合多个KT结果文件，计算每个学生每个模型的AUC。
    
    :param file_paths: txt文件的路径列表 (由您手动传入)
    :param model_names: 对应的模型名称列表 (例如 ['model_A', 'model_B'])
    :return: 一个融合后的DataFrame
    """
    if len(file_paths) != len(model_names):
        raise ValueError("文件路径列表和模型名称列表的长度必须一致")
        
    if not file_paths:
        return pd.DataFrame()

    all_model_data = []
    for filepath, model_name in zip(file_paths, model_names):
        print(f"正在处理: {filepath} (模型: {model_name})")
        model_df = process_file(filepath, model_name)
        if not model_df.empty:
            all_model_data.append(model_df)

    if not all_model_data:
        print("未成功解析任何文件。")
        return pd.DataFrame()

    # 1. 将第一个DataFrame作为基础
    result_df = all_model_data[0]

    # 2. 迭代地将其余的DataFrame合并进来
    if len(all_model_data) > 1:
        for i in range(1, len(all_model_data)):
            result_df = pd.merge(
                result_df, 
                all_model_data[i], 
                on=['questions', 'concepts'],
                how='outer'
            )

    # 3. 添加全局学生ID
    result_df.insert(0, 'student_id', range(len(result_df)))

    # 4. 计算每个模型的AUC
    for model_name in model_names:
        trues_col = f'trues_{model_name}'
        preds_col = f'preds_{model_name}'
        auc_col = f'auc_{model_name}'
        
        if trues_col in result_df and preds_col in result_df:
            
            # =================== 关键修改在这里 ===================
            # 我们使用 isinstance(..., list) 来检查单元格是否包含列表
            # 而不是 pd.notna()
            # ========================================================
            result_df[auc_col] = result_df.apply(
                lambda row: calculate_auc(row[trues_col], row[preds_col])
                            if isinstance(row[trues_col], list) and isinstance(row[preds_col], list)
                            else np.nan,
                axis=1
            )
        else:
            print(f"警告: 找不到 {trues_col} 或 {preds_col}，跳过 {auc_col} 的计算")

    # 5. 调整列顺序
    ordered_cols = ['student_id', 'questions', 'concepts']
    for name in model_names:
        ordered_cols.extend([col for col in [f'trues_{name}', f'preds_{name}', f'auc_{name}'] if col in result_df])
    
    final_cols = [col for col in ordered_cols if col in result_df.columns]
    
    return result_df[final_cols]
# ==================================================================
#                       *** 如何使用 ***
# ==================================================================

if __name__ == "__main__":
    now = datetime.datetime.now()
    file_timestamp = now.strftime("%Y%m%d_%H%M%S")
    print(f"格式 (时间戳): {file_timestamp}")
    # 1. 在这里手动定义您的文件路径列表
    # (请确保使用正确的路径分隔符，或者在字符串前加 r)
    my_file_paths = [
        r"/data/pykt-toolkit/examples/saved_model/assist2009_akt_qid_saved_model_3407_0_0.1_64_256_4_4_0.001_0_1_1/qid_test_question_window_predictions.txt",
        r"/data/pykt-toolkit/examples/saved_model/assist2009_0_0.0001_3407_32_200_0_1_saved_model_qikt_mamba_attn_0.5_256_1_2.0_gru_0/attn_test_window_predictions.txt",
        # ... 您可以添加任意多个文件
    ]
    
    # 2. 为每个文件定义一个唯一的模型名称 (将用于DataFrame的列名)
    my_model_names = [
        "akt",
        "qikt",
        # ... 确保名称与文件一一对应
    ]

    # 3. 运行融合与分析
    print("开始进行知识追踪结果融合分析...")
    final_merged_df = merge_kt_results(my_file_paths, my_model_names)

    # 4. 打印结果或保存到文件
    if not final_merged_df.empty:
        print("\n--- 融合分析完成 ---")
        print("最终DataFrame的前5行:")
        print(final_merged_df.head())

        # (可选) 打印所有学生的平均AUC
        print("\n--- 模型平均AUC (在所有学生上) ---")
        for name in my_model_names:
            if f'auc_{name}' in final_merged_df.columns:
                mean_auc = final_merged_df[f'auc_{name}'].mean()
                print(f"{name} Mean AUC: {mean_auc:.5f}")
        file_model_names = '_'.join(my_model_names)
        # (可选) 将最终结果保存到 Excel 或 CSV
        output_csv_path = f"/data/pykt-results/analysis_result/kt_merged_analysis_{file_model_names}_{file_timestamp}.csv"
        final_merged_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {output_csv_path}")
        
        # output_excel_path = "kt_merged_analysis.xlsx"
        # final_merged_df.to_excel(output_excel_path, index=False)
        # print(f"\n结果已保存到: {output_excel_path}")

    else:
        print("处理失败，未生成任何数据。请检查您的文件路径和文件内容。")