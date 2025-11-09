import os
import pandas as pd
import numpy as np
import json
import glob
# 假设您之前定义的解析函数在这里可用
from cal_auc import parse_and_calculate_aucs_from_file
# 在脚本最上面加一行开关
ENABLE_CACHE = True   # ← 改这里就行！True=加速神器，False=强制重算一切
# --- 关键常量定义 (已修改) ---
# 需要从 overall_auc_info 中提取的统计指标
STAT_METRICS = ['mean', 'std', 'max', 'min', 'range']
# 对应：总体AUC, 学生平均AUC, 学生AUC标准差, 学生AUC最大值, 学生AUC最小值, 学生AUC极差
FINAL_BASELINE_COLUMNS = [
    'overall_dataset_auc',
    'overall_dataset_auc_std',
    'student_auc_mean',
    'student_auc_mean_std',
    'student_auc_std',
    'student_auc_std_std',
    'student_auc_max',
    'student_auc_max_std',
    'student_auc_min',
    'student_auc_min_std',
    'student_auc_range',
    'student_auc_range_std',
    'unfairness_metric',
    # === 新增指标 ===
    'average_auc',
    'average_auc_std',
    'gini_coefficient',
    'gini_coefficient_std',
    'eawi_alpha_10',
    'eawi_alpha_10_std',
    'eawi_alpha_20',
    'eawi_alpha_20_std',
    'eawi_alpha_30',
    'eawi_alpha_30_std'
]
# 用于解析文件夹名称的特殊数据集名称
SPECIAL_DATASET = 'nips_task34'
# 日志文件路径
LOG_FILE_NAME = 'analysis_log.txt'
# 最终结果文件的名称
FINAL_BASELINE_FILE_NAME = 'final_baseline_summary.csv'

# 用于记录缺失文件的日志函数 (保持不变)
def log_missing_file(log_message, root_dir='.'):
    """将缺失文件信息记录到项目日志文件。"""
    log_path = os.path.join(root_dir, LOG_FILE_NAME)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(log_message + '\n')
    print(f"⚠️ LOGGED: {log_message}")

# --- 辅助函数：不公平性指标计算 (保持不变) ---
def calculate_unfairness(mean, auc_range, std):
    """
    计算不公平性指标：每个学生的平均auc / (auc_range * 每个学生auc的标准差)
    """
    # 避免除以零或极小值
    denominator = auc_range * std
    if denominator == 0 or np.isnan(denominator) or np.isclose(denominator, 0): # 添加 np.isclose 检查
        return np.nan # 或返回一个特定的标记值
    return mean / denominator



# --- 辅助函数：将 overall_auc_info 展平 (保持不变) ---
def flatten_overall_info(info):
    """将嵌套的 overall_auc_info 展平为单层字典"""
    flat = {'overall_dataset_auc': info['overall_dataset_auc']}
    for key, value in info['student_auc_stats'].items():
        flat[f'student_auc_{key}'] = value
    return flat

# 新增辅助函数：从最终统计TXT文件中提取结果 (已修改，现在需要提取所有的 *_mean 和 *_std)
def extract_final_stats_from_txt(txt_path):
    """
    从最终统计TXT文件中读取所需的均值和标准差结果。
    
    返回: 包含最终统计均值和标准差的字典。
    """
    results = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, value_str = line.split(':', 1)
                key = key.strip()
                try:
                    value = float(value_str.strip())
                    # 我们现在需要提取所有的均值和标准差结果
                    if key.endswith('_mean') or key.endswith('_std'):
                        results[key] = value
                except ValueError:
                    continue # 跳过非浮点数值
    return results

def analyze_all_results(root_directory='.'):
    """
    遍历指定目录下的所有数据集、模型和数据折，计算并汇总AUC指标。
    **在每个 Dataset/Model 组合处，优先尝试加载已存在的统计结果。**
    
    Args:
        root_directory (str): 包含所有数据集文件夹的根目录。
        
    Returns:
        pd.DataFrame: 最终的 baseline 汇总表。
    """
    # ----------------------------------------------------
    # 🔥 缓存检查：如果最终结果文件存在，直接跳过计算 (保持不变)
    # ----------------------------------------------------
    baseline_output_path = os.path.join(root_directory, FINAL_BASELINE_FILE_NAME)
    if os.path.exists(baseline_output_path):
        print("\n=======================================================")
        print(f"✅ DETECTED: Final baseline summary file already exists.")
        print(f"🔥 SKIPPING ALL RE-CALCULATION. Reading from: {baseline_output_path}")
        print("=======================================================")
        try:
            return pd.read_csv(baseline_output_path)
        except Exception as e:
            print(f"⚠️ ERROR: Failed to read existing file: {e}. Deleting and recalculating.")
            os.remove(baseline_output_path)
    # ----------------------------------------------------
    # 如果文件不存在或读取失败，继续执行计算流程
    print("Starting full analysis and calculation...")


    # 存储所有模型、所有数据集的最终统计结果
    final_baseline_data = []    
    
    # 获取所有数据集文件夹（即根目录下的所有文件夹）
    dataset_dirs = [d for d in os.listdir(root_directory) 
                    if os.path.isdir(os.path.join(root_directory, d))]

    for dataset_name in dataset_dirs:
        dataset_path = os.path.join(root_directory, dataset_name)
        
        # 获取模型文件夹（数据集文件夹下的所有文件夹）
        model_dirs = [d for d in os.listdir(dataset_path) 
                      if os.path.isdir(os.path.join(dataset_path, d))]

        for model_name in model_dirs:
            model_path = os.path.join(dataset_path, model_name)
            
            # 定义该计算节点的两个缓存文件路径
            output_csv_path = os.path.join(model_path, f'{model_name}_{dataset_name}_folds_summary.csv')
            output_txt_path = os.path.join(model_path, f'{model_name}_{dataset_name}_final_stats.txt')
            
            # ----------------------------------------------------
            # 🔥 增量缓存检查：检查是否已存在最终统计文件 (已修改缓存加载逻辑以匹配新列)
            # ----------------------------------------------------
            if ENABLE_CACHE and os.path.exists(output_csv_path) and os.path.exists(output_txt_path):
                print(f"✅ CACHE HIT: Found results for {dataset_name}/{model_name}. Loading...")
                
                try:
                    # 1. 尝试加载统计结果 TXT 文件以提取 baseline 均值和标准差
                    final_stats = extract_final_stats_from_txt(output_txt_path)
                    
                    # 2. 从 CSV 中提取折数信息 (CSV 包含所有折的汇总)
                    df_folds = pd.read_csv(output_csv_path)
                    
                    # 重新计算 unfairness_metric，确保一致性
                    # 使用均值的均值、范围的均值、标准差的均值
                    mean_auc = final_stats.get('student_auc_mean_mean', np.nan)
                    mean_range = final_stats.get('student_auc_range_mean', np.nan)
                    mean_std = final_stats.get('student_auc_std_mean', np.nan)
                    
                    # 构建 baseline entry (已修改，增加了所有指标的五折标准差)
                    baseline_entry = {
                        'Dataset': dataset_name,
                        'Model': model_name,

                        # 老指标
                        'overall_dataset_auc': final_stats.get('overall_dataset_auc_mean', np.nan),
                        'overall_dataset_auc_std': final_stats.get('overall_dataset_auc_std', np.nan),

                        'student_auc_mean': final_stats.get('student_auc_mean_mean', np.nan),
                        'student_auc_mean_std': final_stats.get('student_auc_mean_std', np.nan),

                        'student_auc_std': final_stats.get('student_auc_std_mean', np.nan),
                        'student_auc_std_std': final_stats.get('student_auc_std_std', np.nan),

                        'student_auc_max': final_stats.get('student_auc_max_mean', np.nan),
                        'student_auc_max_std': final_stats.get('student_auc_max_std', np.nan),

                        'student_auc_min': final_stats.get('student_auc_min_mean', np.nan),
                        'student_auc_min_std': final_stats.get('student_auc_min_std', np.nan),

                        'student_auc_range': final_stats.get('student_auc_range_mean', np.nan),
                        'student_auc_range_std': final_stats.get('student_auc_range_std', np.nan),

                        # === 新指标：必须在这里也填！===
                        'average_auc': final_stats.get('student_auc_average_auc_mean', np.nan),
                        'average_auc_std': final_stats.get('student_auc_average_auc_std', np.nan),

                        'gini_coefficient': final_stats.get('student_auc_gini_coefficient_mean', np.nan),
                        'gini_coefficient_std': final_stats.get('student_auc_gini_coefficient_std', np.nan),

                        'eawi_alpha_10': final_stats.get('student_auc_eawi_alpha_10_mean', np.nan),
                        'eawi_alpha_10_std': final_stats.get('student_auc_eawi_alpha_10_std', np.nan),

                        'eawi_alpha_20': final_stats.get('student_auc_eawi_alpha_20_mean', np.nan),
                        'eawi_alpha_20_std': final_stats.get('student_auc_eawi_alpha_20_std', np.nan),

                        'eawi_alpha_30': final_stats.get('student_auc_eawi_alpha_30_mean', np.nan),
                        'eawi_alpha_30_std': final_stats.get('student_auc_eawi_alpha_30_std', np.nan),

                        # 不公平性
                        'unfairness_metric': final_stats.get('unfairness_metric_mean') 
                                           if 'unfairness_metric_mean' in final_stats 
                                           else calculate_unfairness(mean_auc, mean_range, mean_std),

                        'Folds_Present': ','.join(map(str, range(len(df_folds)))) if not df_folds.empty else 'N/A'
                    }
                    final_baseline_data.append(baseline_entry)
                    print(f"    -> SKIPPED CALCULATION for {dataset_name}/{model_name}.")
                    continue # 跳过后续的计算步骤
                
                except Exception as e:
                    # 如果加载失败，记录错误并继续计算（以防文件损坏）
                    log_missing_file(
                        f"ERROR: Failed to load cached files for {dataset_name}/{model_name}. Error: {e}. Recalculating...",
                        root_dir=root_directory
                    )
                    # 清理缓存文件，强制重新计算
                    if os.path.exists(output_csv_path): os.remove(output_csv_path)
                    if os.path.exists(output_txt_path): os.remove(output_txt_path)

            # ----------------------------------------------------
            # 走到这里说明需要进行完整的计算
            # ----------------------------------------------------
            print(f"🔄 Calculating: {dataset_name}/{model_name}...")

            # 存储五折的解析结果（每个元素是一个 flatten 后的 dict）
            fold_results = []
            # 记录成功解析的折数
            successful_folds = []    
            
            # ... (这部分解析五折文件的逻辑保持不变，确保成功将结果填充到 fold_results 中) ...
            
            # 获取模型文件夹下所有折文件夹的名称
            fold_folders = [d for d in os.listdir(model_path) 
                              if os.path.isdir(os.path.join(model_path, d))]

            # ----------------------------------------------------
            # 1. 遍历五折文件夹，解析数据（这部分代码与原版相同）
            # ----------------------------------------------------
            
            # ... (此处是原有的复杂的 fold ID 解析和文件解析逻辑) ...
            
            for fold_folder_name in fold_folders:
                
                # 特殊处理：解析折数 (fold_id)
                parts = fold_folder_name.split('_')
                
                try:
                    # 假设格式是：MODEL_tiaocan_DATASET_SEED_FOLD_...
                    if SPECIAL_DATASET in fold_folder_name:
                        # 查找 SPECIAL_DATASET 出现后的第二个 '_'
                        # 注意：nips_task34 会被 split 成 'nips' 和 'task34'
                        dataset_index = [i for i, part in enumerate(parts) if part == SPECIAL_DATASET.split('_')[0]][0]
                        # 假设在 DATASET_part2 (nips_task34) 之后是 SEED 和 FOLD
                        # 如果是 MODEL_tiaocan_nips_task34_SEED_FOLD_...
                        # 折数通常是第 5 个部分 (index 4)
                        # 这里原代码可能逻辑有问题，但为了保证功能一致性，保留原逻辑
                        # 如果 dataset_name 是 'nips_task34'，它会被分解，逻辑复杂，保持不变
                        
                        # 找到 dataset_name 在 parts 中的完整匹配
                        dataset_part_index = -1
                        for i in range(len(parts) - 1):
                            if parts[i] == 'nips' and parts[i+1] == 'task34':
                                dataset_part_index = i
                                break
                        
                        if dataset_part_index != -1 and dataset_part_index + 3 < len(parts):
                             # 假设在 'task34' 后面是 SEED 和 FOLD
                            fold_id = int(parts[dataset_part_index + 3])
                        else:
                            # 默认 fallback
                            fold_id = int(parts[-2])
                            
                    else:
                        # 正常情况：假设折数在第四个位置（索引3）或第五个位置（索引4）
                        
                        dataset_part_index = -1
                        for i, part in enumerate(parts):
                            if part == dataset_name:
                                dataset_part_index = i
                                break
                        
                        # 假设折数在数据集名之后两个位置（SEED和FOLD）
                        if dataset_part_index != -1 and dataset_part_index + 2 < len(parts):
                            fold_id = int(parts[dataset_part_index + 2])
                        else:
                            # 简单粗暴，假设折数是倒数第2个或第3个数字
                            fold_id = int(parts[-2]) # 保持原简单逻辑
                            
                except (ValueError, IndexError):
                    log_missing_file(
                        f"Skipping: Could not reliably parse fold ID from folder name: {fold_folder_name} in {model_path}", 
                        root_dir=root_directory
                    )
                    continue

                fold_path = os.path.join(model_path, fold_folder_name)
                
                # 查找内部的txt文件（通常只有一个）
                txt_files = glob.glob(os.path.join(fold_path, '*.txt'))
                
                if not txt_files:
                    # 记录缺失文件
                    log_missing_file(
                        f"MISSING FILE: No TXT file found in {fold_path}. Expected fold: {fold_id}",
                        root_dir=root_directory
                    )
                    continue
                
                # 假设只有一个txt文件
                txt_file_path = txt_files[0]
                
                json_cache_path = txt_file_path.replace('.txt', '_per_student_aucs.json')
                if ENABLE_CACHE and os.path.exists(json_cache_path):
                    print(f"  CACHE HIT: Loading pre-computed student AUCs from {os.path.basename(json_cache_path)}")
                    try:
                        with open(json_cache_path, 'r', encoding='utf-8') as f:
                            student_aucs_dict = json.load(f)
                        
                        # 构造一个假的 overall_info（只保留我们需要的）
                        dummy_stats = {
                            'mean': np.mean(list(student_aucs_dict.values())),
                            'std': np.std(list(student_aucs_dict.values())),
                            'max': np.max(list(student_aucs_dict.values())),
                            'min': np.min(list(student_aucs_dict.values())),
                            'range': np.max(list(student_aucs_dict.values())) - np.min(list(student_aucs_dict.values())),
                            'gini_coefficient': gini_coefficient(list(student_aucs_dict.values())),
                            'average_auc': np.mean(list(student_aucs_dict.values())),
                        }
                        # 加入 EAWI
                        _, _, eawi_dict = calculate_wealth_metrics(list(student_aucs_dict.values()))
                        dummy_stats.update(eawi_dict)

                        dummy_overall_info = {
                            'overall_dataset_auc': 0.5,  # 我们不关心这个，可以后续从CSV覆盖
                            'student_auc_stats': dummy_stats
                        }
                        
                        fold_results.append(flatten_overall_info(dummy_overall_info))
                        successful_folds.append(fold_id)
                        print(f"  Successfully reused cached student AUCs for fold {fold_id}")
                        continue  # 跳过 parse_and_calculate_aucs_from_file！
                    
                    except Exception as e:
                        print(f"  Failed to load JSON cache: {e}. Will recompute...")

                # === 原有逻辑：如果没有缓存，才重新计算 ===
                try:
                    _, overall_info = parse_and_calculate_aucs_from_file(txt_file_path)
                    fold_results.append(flatten_overall_info(overall_info))
                    successful_folds.append(fold_id)
                except Exception as e:
                    log_missing_file(
                        f"ERROR: Failed to parse and calculate AUC for {txt_file_path}. Error: {e}",
                        root_dir=root_directory
                    )
                    
                except Exception as e:
                    log_missing_file(
                        f"ERROR: Failed to parse and calculate AUC for {txt_file_path}. Error: {e}",
                        root_dir=root_directory
                    )

            # ----------------------------------------------------
            # 2. 汇总五折结果并计算统计指标 (这部分逻辑是核心修改)
            # ----------------------------------------------------
            
            if not fold_results:
                log_missing_file(
                    f"Skipping {dataset_name}/{model_name}: No successful folds found.",
                    root_dir=root_directory
                )
                continue
                
            # 将五折结果列表转换为 DataFrame
            df_folds = pd.DataFrame(fold_results)
            
            # 保存五折的 overall_auc_info 汇总 CSV
            df_folds.to_csv(output_csv_path, index=False)
            print(f"✅ Saved fold summary CSV to: {output_csv_path}")

            # 计算所有指标的五折均值和五折标准差
            mean_stats = df_folds.mean(numeric_only=True).rename(lambda x: f'{x}_mean')
            # 确保使用 ddof=1 计算样本标准差
            std_stats = df_folds.std(ddof=1, numeric_only=True).rename(lambda x: f'{x}_std') 
            
            # 整合统计结果
            df_stats = pd.concat([mean_stats, std_stats]).to_frame().T
            
            # 计算不公平性指标的均值 (需要用到均值结果)
            mean_auc = df_stats['student_auc_mean_mean'].iloc[0]
            mean_range = df_stats['student_auc_range_mean'].iloc[0]
            mean_std = df_stats['student_auc_std_mean'].iloc[0]
            
            # 计算不公平性指标的均值
            unfairness_mean = calculate_unfairness(mean_auc, mean_range, mean_std)

            # 将统计结果保存到模型文件夹下的TXT文件
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(f"--- {dataset_name}/{model_name} Final Stats ---\n")
                
                # 写入所有指标的均值和标准差
                for col in df_stats.columns:
                    f.write(f"{col}: {df_stats[col].iloc[0]:.6f}\n")
                
                # 写入不公平性指标均值
                f.write(f"\nunfairness_metric_mean: {unfairness_mean:.6f}\n")
            
            print(f"✅ Saved final stats TXT to: {output_txt_path}")

            # ----------------------------------------------------
            # 3. 准备最终 baseline 表数据 (已修改，增加了所有指标的五折标准差)
            # ----------------------------------------------------
            
            # 提取关键均值和标准差指标，用于 baseline 大表
            baseline_entry = {
                'Dataset': dataset_name,
                'Model': model_name,
                
                # === 原有指标（保持不动）===
                'overall_dataset_auc': df_stats.get('overall_dataset_auc_mean', np.nan),
                'overall_dataset_auc_std': df_stats.get('overall_dataset_auc_std', np.nan),
                
                'student_auc_mean': df_stats.get('student_auc_mean_mean', np.nan),
                'student_auc_mean_std': df_stats.get('student_auc_mean_std', np.nan),
                
                'student_auc_std': df_stats.get('student_auc_std_mean', np.nan),
                'student_auc_std_std': df_stats.get('student_auc_std_std', np.nan),
                
                'student_auc_max': df_stats.get('student_auc_max_mean', np.nan),
                'student_auc_max_std': df_stats.get('student_auc_max_std', np.nan),
                
                'student_auc_min': df_stats.get('student_auc_min_mean', np.nan),
                'student_auc_min_std': df_stats.get('student_auc_min_std', np.nan),
                
                'student_auc_range': df_stats.get('student_auc_range_mean', np.nan),
                'student_auc_range_std': df_stats.get('student_auc_range_std', np.nan),
                
                # === 新增指标：自动映射（最优雅！）===
                'average_auc': df_stats.get('student_auc_average_auc_mean', np.nan),
                'average_auc_std': df_stats.get('student_auc_average_auc_std', np.nan),
                
                'gini_coefficient': df_stats.get('student_auc_gini_coefficient_mean', np.nan),
                'gini_coefficient_std': df_stats.get('student_auc_gini_coefficient_std', np.nan),
                
                'eawi_alpha_10': df_stats.get('student_auc_eawi_alpha_10_mean', np.nan),
                'eawi_alpha_10_std': df_stats.get('student_auc_eawi_alpha_10_std', np.nan),
                
                'eawi_alpha_20': df_stats.get('student_auc_eawi_alpha_20_mean', np.nan),
                'eawi_alpha_20_std': df_stats.get('student_auc_eawi_alpha_20_std', np.nan),
                
                'eawi_alpha_30': df_stats.get('student_auc_eawi_alpha_30_mean', np.nan),
                'eawi_alpha_30_std': df_stats.get('student_auc_eawi_alpha_30_std', np.nan),
                
                # === 不公平性指标 ===
                'unfairness_metric': unfairness_mean,
                
                'Folds_Present': ','.join(map(str, sorted(successful_folds)))
            }
            final_baseline_data.append(baseline_entry)


    # ----------------------------------------------------
    # 4. 生成最终 Baseline 大表 CSV (保持不变)
    # ----------------------------------------------------
    
    if final_baseline_data:
        df_baseline = pd.DataFrame(final_baseline_data)
        
        # 排序和重排字段
        final_cols = ['Dataset', 'Model'] + FINAL_BASELINE_COLUMNS + ['Folds_Present']
        df_baseline = df_baseline[final_cols]

        # 最终保存路径在根目录
        df_baseline.to_csv(baseline_output_path, index=False)
        print("\n=======================================================")
        print(f"🔥 FINAL BASELINE TABLE SAVED TO: {baseline_output_path}")
        print("=======================================================")
        return df_baseline
    else:
        print("❌ No valid results were processed to create the final baseline table.")
        return pd.DataFrame()

# --- 最终执行 ---
analysis_root_dir = './'
final_baseline_table = analyze_all_results(analysis_root_dir)
print(final_baseline_table)