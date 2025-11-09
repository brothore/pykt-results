import pandas as pd
import numpy as np
import re

def extract_value_from_series_string(series_string):
    """
    从Pandas Series的字符串表示中提取数值
    
    参数:
    series_string: 包含Series表示的字符串
    
    返回:
    float或np.nan
    """
    if pd.isna(series_string) or not isinstance(series_string, str):
        return np.nan
    
    try:
        # 匹配模式: 数字 数值\nName: 列名, dtype: 类型
        pattern = r'(\d+)\s+([\d.]+)\\nName:\s+\w+,\s+dtype:\s+\w+'
        match = re.search(pattern, series_string)
        
        if match:
            # 提取数值部分
            value_str = match.group(2)
            return float(value_str)
        else:
            # 尝试其他可能的模式
            # 直接匹配浮点数
            float_pattern = r'(\d+\.\d+)'
            float_match = re.search(float_pattern, series_string)
            if float_match:
                return float(float_match.group(1))
            
            # 如果都失败，返回NaN
            return np.nan
            
    except (ValueError, TypeError, IndexError) as e:
        print(f"解析错误: {series_string} -> {e}")
        return np.nan

def safe_float_conversion(value):
    """
    安全地将字符串转换为浮点数
    
    参数:
    value: 要转换的值
    
    返回:
    float或np.nan
    """
    if pd.isna(value):
        return np.nan
    
    # 如果已经是数值类型，直接返回
    if isinstance(value, (int, float)):
        return float(value)
    
    # 如果是字符串，进行处理
    if isinstance(value, str):
        # 首先尝试提取Series字符串中的数值
        extracted_value = extract_value_from_series_string(value)
        if not np.isnan(extracted_value):
            return extracted_value
        
        # 如果提取失败，尝试直接转换
        cleaned_value = value.strip()
        
        # 处理空字符串
        if cleaned_value == '':
            return np.nan
        
        # 处理百分号
        if '%' in cleaned_value:
            cleaned_value = cleaned_value.replace('%', '')
            try:
                return float(cleaned_value) / 100
            except (ValueError, TypeError):
                return np.nan
        
        # 处理科学计数法
        if 'e' in cleaned_value.lower():
            try:
                return float(cleaned_value)
            except (ValueError, TypeError):
                return np.nan
        
        # 处理逗号分隔的数字
        if ',' in cleaned_value:
            cleaned_value = cleaned_value.replace(',', '')
        
        # 尝试直接转换
        try:
            return float(cleaned_value)
        except (ValueError, TypeError):
            return np.nan
    
    return np.nan

def process_metrics_data(csv_file_path):
    """
    处理指标数据并生成LaTeX表格
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 筛选指定数据集的行
    target_datasets = ['assist2009', 'nips_task34', 'peiyou']
    filtered_df = df[df['Dataset'].isin(target_datasets)].copy()
    
    print(f"筛选后数据形状: {filtered_df.shape}")
    print(f"找到的数据集: {filtered_df['Dataset'].unique()}")
    
    # 找出同时拥有3个数据集的模型
    model_counts = filtered_df.groupby('Model')['Dataset'].nunique()
    valid_models = model_counts[model_counts == 3].index.tolist()
    
    print(f"有效模型 ({len(valid_models)}个): {valid_models}")
    
    # 筛选有效模型的数据
    final_df = filtered_df[filtered_df['Model'].isin(valid_models)]
    
    print(f"最终数据形状: {final_df.shape}")
    
    # 定义需要处理的指标列对
    metric_pairs = [
        ('student_auc_mean', 'student_auc_mean_std'),
        ('student_auc_std', 'student_auc_std_std'),
        ('student_auc_range', 'student_auc_range_std'),
        ('gini_coefficient', 'gini_coefficient_std'),
        ('eawi_alpha_10', 'eawi_alpha_10_std'),
        ('eawi_alpha_20', 'eawi_alpha_20_std'),
        ('eawi_alpha_30', 'eawi_alpha_30_std')
    ]
    
    # 首先转换所有指标列为数值类型
    print("开始数据类型转换...")
    all_metric_columns = []
    for mean_col, std_col in metric_pairs:
        if mean_col in final_df.columns:
            all_metric_columns.append(mean_col)
        if std_col in final_df.columns:
            all_metric_columns.append(std_col)
    
    # 转换数值列
    conversion_stats = {}
    for col in all_metric_columns:
        if col in final_df.columns:
            original_dtype = final_df[col].dtype
            print(f"转换列 {col} (原始类型: {original_dtype})")
            
            # 显示转换前的样本
            sample_before = final_df[col].head(3).tolist()
            print(f"  转换前样本: {sample_before}")
            
            # 应用转换
            final_df[col] = final_df[col].apply(safe_float_conversion)
            
            # 统计转换结果
            non_na_count = final_df[col].notna().sum()
            conversion_stats[col] = non_na_count
            
            # 显示转换后的样本
            sample_after = final_df[col].head(3).tolist()
            print(f"  转换后样本: {sample_after}")
            print(f"  成功转换: {non_na_count}/{len(final_df)} 行")
    
    print(f"转换统计: {conversion_stats}")
    
    # 处理指标列，合并为mean±std格式
    processed_data = []
    
    for model in valid_models:
        model_data = final_df[final_df['Model'] == model]
        
        # 确保每个模型都有3个数据集的数据
        if len(model_data) != 3:
            print(f"警告: 模型 {model} 只有 {len(model_data)} 个数据集的数据")
            continue
            
        for dataset in target_datasets:
            dataset_data = model_data[model_data['Dataset'] == dataset]
            
            if len(dataset_data) == 1:
                row_data = {'Model': model, 'Dataset': dataset}
                row_has_data = False
                
                for mean_col, std_col in metric_pairs:
                    if mean_col in dataset_data.columns and std_col in dataset_data.columns:
                        mean_val = dataset_data[mean_col].values[0]
                        std_val = dataset_data[std_col].values[0]
                        
                        # 格式化数值，保留合适的小数位数
                        if pd.notna(mean_val) and pd.notna(std_val):
                            # 根据数值大小决定小数位数
                            if abs(mean_val) < 0.01 or abs(std_val) < 0.01:
                                formatted_value = f"{mean_val:.4f}±{std_val:.4f}"
                            else:
                                formatted_value = f"{mean_val:.3f}±{std_val:.3f}"
                            row_has_data = True
                        else:
                            formatted_value = "N/A"
                            if pd.isna(mean_val) or pd.isna(std_val):
                                print(f"警告: 模型 {model} 在数据集 {dataset} 的 {mean_col}/{std_col} 有缺失值")
                        
                        row_data[f"{mean_col}_formatted"] = formatted_value
                        # 同时保存原始数值用于检查
                        row_data[f"{mean_col}_value"] = mean_val
                        row_data[f"{std_col}_value"] = std_val
                
                if row_has_data:
                    processed_data.append(row_data)
                else:
                    print(f"警告: 模型 {model} 在数据集 {dataset} 没有有效数据")
    
    # 创建处理后的DataFrame
    processed_df = pd.DataFrame(processed_data)
    
    print(f"处理后的数据形状: {processed_df.shape}")
    
    if len(processed_df) == 0:
        print("错误: 没有成功处理任何数据！")
        return pd.DataFrame(), ""
    
    # 生成LaTeX表格
    latex_table = generate_latex_table(processed_df, valid_models, target_datasets, metric_pairs)
    
    return processed_df, latex_table

def generate_latex_table(df, models, datasets, metric_pairs):
    """
    生成LaTeX格式的表格 - 新布局：行为模型，列为指标
    """
    
    # LaTeX表格开始
    latex_code = "\\begin{table}[htbp]\n"
    latex_code += "\\centering\n"
    latex_code += "\\caption{模型性能指标比较 (格式: mean±std)}\n"
    latex_code += "\\label{tab:model_comparison}\n"
    
    # 计算列数 (1个模型列 + 数据集数量 * 指标数量)
    num_metrics = len(metric_pairs)
    total_columns = 1 + len(datasets) * num_metrics
    
    # 修复这里：正确的字符串拼接方式
    column_spec = "l" + "c" * (total_columns - 1)
    
    # 开始tabular环境
    latex_code += f"\\begin{{tabular}}{{{column_spec}}}\n"
    latex_code += "\\toprule\n"
    
    # 第一行表头: 数据集名称（跨列）
    latex_code += " & "
    for dataset in datasets:
        latex_code += f"\\multicolumn{{{num_metrics}}}{{c}}{{\\textbf{{{dataset}}}}} & "
    latex_code = latex_code.rstrip(" & ") + "\\\\\n"
    
    # 第二行表头: 具体指标名称
    latex_code += "\\cmidrule(lr){2-" + str(total_columns) + "}\n"
    latex_code += "Model "
    for dataset in datasets:
        for mean_col, std_col in metric_pairs:
            metric_name = get_metric_display_name(mean_col)
            latex_code += f"& {metric_name} "
    latex_code += "\\\\\n"
    latex_code += "\\midrule\n"
    
    # 预先计算每个数据集每个指标的最佳值
    best_values = {}
    for dataset in datasets:
        best_values[dataset] = {}
        for mean_col, std_col in metric_pairs:
            values = []
            for model in models:
                cell_data = df[(df['Model'] == model) & (df['Dataset'] == dataset)]
                if not cell_data.empty and f"{mean_col}_value" in cell_data.columns:
                    mean_val = cell_data[f"{mean_col}_value"].values[0]
                    if pd.notna(mean_val):
                        values.append(mean_val)
            
            if values:
                metric_name = get_metric_display_name(mean_col)
                # 确定最佳值的方向（越大越好还是越小越好）
                if metric_name in ['AUC Std', 'AUC Range', 'Gini']:
                    # 越小越好
                    best_value = min(values)
                else:
                    # 越大越好
                    best_value = max(values)
                best_values[dataset][mean_col] = best_value
            else:
                best_values[dataset][mean_col] = None
    
    # 数据行：每个模型一行
    for model in models:
        latex_code += f"{model} "
        
        for dataset in datasets:
            # 查找该模型在该数据集下的所有数据
            cell_data = df[(df['Model'] == model) & (df['Dataset'] == dataset)]
            
            for mean_col, std_col in metric_pairs:
                if not cell_data.empty:
                    formatted_col = f"{mean_col}_formatted"
                    value_col = f"{mean_col}_value"
                    
                    if formatted_col in cell_data.columns and value_col in cell_data.columns:
                        formatted_value = cell_data[formatted_col].values[0]
                        mean_val = cell_data[value_col].values[0]
                        
                        # 检查是否为最佳值
                        if (pd.notna(mean_val) and 
                            best_values[dataset][mean_col] is not None and 
                            abs(mean_val - best_values[dataset][mean_col]) < 1e-6):
                            latex_code += f"& \\textbf{{{formatted_value}}} "
                        else:
                            latex_code += f"& {formatted_value} "
                    else:
                        latex_code += "& N/A "
                else:
                    latex_code += "& N/A "
        
        latex_code += "\\\\\n"
    
    latex_code += "\\bottomrule\n"
    latex_code += "\\end{tabular}\n"
    latex_code += "\\end{table}"
    
    return latex_code


def get_metric_display_name(metric_col):
    """
    获取指标的显示名称
    """
    display_names = {
        'student_auc_mean': 'AUC Mean',
        'student_auc_std': 'AUC Std',
        'student_auc_range': 'AUC Range',
        'gini_coefficient': 'Gini',
        'eawi_alpha_10': 'EAWI-10',
        'eawi_alpha_20': 'EAWI-20',
        'eawi_alpha_30': 'EAWI-30'
    }
    return display_names.get(metric_col, metric_col)

def generate_alternative_layout(df, models, datasets, metric_pairs):
    """
    生成备选布局的LaTeX表格 - 每个数据集一个子表格
    """
    
    latex_code = "\\begin{table}[htbp]\n"
    latex_code += "\\centering\n"
    latex_code += "\\caption{模型性能指标分数据集比较 (格式: mean±std)}\n"
    latex_code += "\\label{tab:model_comparison_by_dataset}\n"
    
    for dataset in datasets:
        latex_code += f"\\subsection*{{{dataset}}}\n"
        latex_code += "\\begin{tabular}{l" + "c" * len(metric_pairs) + "}\n"
        latex_code += "\\toprule\n"
        
        # 表头
        latex_code += "Model & " + " & ".join([get_metric_display_name(m[0]) for m in metric_pairs]) + " \\\\\n"
        latex_code += "\\midrule\n"
        
        # 预先计算该数据集每个指标的最佳值
        best_values = {}
        for mean_col, std_col in metric_pairs:
            values = []
            for model in models:
                cell_data = df[(df['Model'] == model) & (df['Dataset'] == dataset)]
                if not cell_data.empty and f"{mean_col}_value" in cell_data.columns:
                    mean_val = cell_data[f"{mean_col}_value"].values[0]
                    if pd.notna(mean_val):
                        values.append(mean_val)
            
            if values:
                metric_name = get_metric_display_name(mean_col)
                # 确定最佳值的方向（越大越好还是越小越好）
                if metric_name in ['AUC Std', 'AUC Range', 'Gini']:
                    # 越小越好
                    best_value = min(values)
                else:
                    # 越大越好
                    best_value = max(values)
                best_values[mean_col] = best_value
            else:
                best_values[mean_col] = None
        
        # 数据行
        for model in models:
            latex_code += f"{model} "
            
            cell_data = df[(df['Model'] == model) & (df['Dataset'] == dataset)]
            
            for mean_col, std_col in metric_pairs:
                if not cell_data.empty:
                    formatted_col = f"{mean_col}_formatted"
                    value_col = f"{mean_col}_value"
                    
                    if formatted_col in cell_data.columns and value_col in cell_data.columns:
                        formatted_value = cell_data[formatted_col].values[0]
                        mean_val = cell_data[value_col].values[0]
                        
                        # 检查是否为最佳值
                        if (pd.notna(mean_val) and 
                            best_values[mean_col] is not None and 
                            abs(mean_val - best_values[mean_col]) < 1e-6):
                            latex_code += f"& \\textbf{{{formatted_value}}} "
                        else:
                            latex_code += f"& {formatted_value} "
                    else:
                        latex_code += "& N/A "
                else:
                    latex_code += "& N/A "
            
            latex_code += "\\\\\n"
        
        latex_code += "\\bottomrule\n"
        latex_code += "\\end{tabular}\n"
        latex_code += "\\vspace{1em}\n"
    
    latex_code += "\\end{table}"
    
    return latex_code
def debug_specific_columns(csv_file_path):
    """
    调试特定列的转换
    """
    df = pd.read_csv(csv_file_path)
    
    # 测试几个具体的列
    test_columns = ['student_auc_mean', 'student_auc_mean_std', 'eawi_alpha_10']
    
    for col in test_columns:
        if col in df.columns:
            print(f"\n=== 调试列: {col} ===")
            sample_data = df[col].head(5)
            print("原始数据:")
            for i, value in enumerate(sample_data):
                print(f"  {i}: {repr(value)}")
                converted = safe_float_conversion(value)
                print(f"      -> {converted}")

# 使用示例
if __name__ == "__main__":
    # 替换为您的CSV文件路径
    csv_file_path = "/data/pykt-results/no_window/final_baseline_summary.csv"
    import os
    csv_dir = os.path.dirname(csv_file_path)
    
    # 拼接输出文件路径
    processed_csv_path = os.path.join(csv_dir, "processed_metrics.csv")
    latex_table_path = os.path.join(csv_dir, "metrics_table.tex")
    alt_latex_path = os.path.join(csv_dir, "metrics_table_alternative.tex")

    # 首先调试特定列的转换
    print("=== 数据转换调试 ===")
    debug_specific_columns(csv_file_path)
    
    print("\n=== 开始处理数据 ===")
    try:
        # 处理数据并生成表格
        processed_df, latex_table = process_metrics_data(csv_file_path)
        
        if not processed_df.empty:
            # 保存处理后的数据
            processed_df.to_csv(processed_csv_path, index=False)
            print("处理后的数据已保存到 processed_metrics.csv")
            
            # 保存主LaTeX表格
            with open(latex_table_path, "w", encoding="utf-8") as f:
                f.write(latex_table)
            print(f"LaTeX表格已保存到 {latex_table_path}")
            
            # 生成并保存备选布局
            alt_latex = generate_alternative_layout(processed_df, 
                                                  processed_df['Model'].unique().tolist(),
                                                  processed_df['Dataset'].unique().tolist(),
                                                  [('student_auc_mean', 'student_auc_mean_std'),
                                                   ('student_auc_std', 'student_auc_std_std'),
                                                   ('student_auc_range', 'student_auc_range_std'),
                                                   ('gini_coefficient', 'gini_coefficient_std'),
                                                   ('eawi_alpha_10', 'eawi_alpha_10_std'),
                                                   ('eawi_alpha_20', 'eawi_alpha_20_std'),
                                                   ('eawi_alpha_30', 'eawi_alpha_30_std')])
            with open(alt_latex_path, "w", encoding="utf-8") as f:
                f.write(alt_latex)
            print(f"备选布局LaTeX表格已保存到  {alt_latex_path}")
            
            # 打印表格预览
            print("\n生成的LaTeX表格预览:")
            print("=" * 50)
            print(latex_table)
        else:
            print("没有生成有效数据，请检查原始数据格式")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_file_path}")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()