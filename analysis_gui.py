import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import ast  # 用于将字符串列表转为真实列表
from sklearn.metrics import roc_auc_score, accuracy_score

# 导入matplotlib相关库
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class KTAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("知识追踪 (KT) 坏例分析工具 (Knowledge Tracing Bad Case Analysis Tool)")
        self.root.geometry("1600x900")

        self.df = None
        self.model_names = []
        self.student_cols = [] # 用于筛选
        self.current_student_data = []  # 存储当前选中学生的序列（已解析）

        self.current_clicked_timestep_data = None  
        self.analysis_mode = "Question"  

        self.student_table_headings = {} 
        self.last_sort_col = None
        self.last_sort_reverse = False

        self.create_widgets()

    def create_widgets(self):
        # --- 顶部菜单 ---
        top_frame = ttk.Frame(self.root, padding=5)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_load = ttk.Button(top_frame, text="加载 CSV 分析文件 (Load CSV)", command=self.load_csv)
        self.btn_load.pack(side=tk.LEFT)
        self.lbl_file = ttk.Label(top_frame, text="未加载文件 (No file loaded)")
        self.lbl_file.pack(side=tk.LEFT, padx=10)

        # --- 主内容区 (使用 PanedWindow 分割) ---
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # --- 左侧面板：学生列表 ---
        left_frame = ttk.Frame(main_paned, padding=5)
        main_paned.add(left_frame, weight=3) 

        ttk.Label(left_frame, text="学生总览 (Student Overview)", font=("-weight bold", 12)).pack(fill=tk.X)

        # --- 修改点 2: 筛选功能 ---
        filter_frame = ttk.Frame(left_frame, padding=(0, 5))
        filter_frame.pack(fill=tk.X)

        ttk.Label(filter_frame, text="筛选:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.filter_col_var = tk.StringVar()
        self.filter_col_menu = ttk.OptionMenu(filter_frame, self.filter_col_var, "选择列")
        self.filter_col_menu.pack(side=tk.LEFT, padx=2)
        
        self.filter_op_var = tk.StringVar(value=">")
        operators = ['>', '<', '>=', '<=', '==', '!=']
        self.filter_op_menu = ttk.OptionMenu(filter_frame, self.filter_op_var, *operators)
        self.filter_op_menu.pack(side=tk.LEFT, padx=2)
        
        self.filter_val_entry = ttk.Entry(filter_frame, width=10)
        self.filter_val_entry.pack(side=tk.LEFT, padx=2)
        
        self.btn_apply_filter = ttk.Button(filter_frame, text="筛选", command=self.apply_student_filter)
        self.btn_apply_filter.pack(side=tk.LEFT, padx=2)
        
        self.btn_clear_filter = ttk.Button(filter_frame, text="重置", command=self.reset_student_table)
        self.btn_clear_filter.pack(side=tk.LEFT, padx=2)
        # --- 筛选结束 ---

        student_cols = ['student_id', 'auc']
        self.student_table = ttk.Treeview(left_frame, columns=student_cols, show='headings', height=25)
        self.student_table.pack(fill=tk.BOTH, expand=True)

        vsb_student = ttk.Scrollbar(self.student_table, orient="vertical", command=self.student_table.yview)
        self.student_table.configure(yscrollcommand=vsb_student.set)
        vsb_student.pack(side='right', fill='y')

        self.student_table.bind("<<TreeviewSelect>>", self.on_student_select)


        # --- 右侧面板 (再上下分割) ---
        right_paned = ttk.PanedWindow(main_paned, orient=tk.VERTICAL)
        main_paned.add(right_paned, weight=7) 

        # --- 右上：序列详情 ---
        seq_frame = ttk.Frame(right_paned, padding=5)
        right_paned.add(seq_frame, weight=4) 

        ttk.Label(seq_frame, text="学生序列详情 (Student Sequence Detail)", font=("-weight bold", 12)).pack(fill=tk.X)

        seq_cols = ['timestep', 'q_id', 'c_id', 'true']
        self.sequence_table = ttk.Treeview(seq_frame, columns=seq_cols, show='headings', height=10)
        self.sequence_table.pack(fill=tk.BOTH, expand=True)

        vsb_seq = ttk.Scrollbar(self.sequence_table, orient="vertical", command=self.sequence_table.yview)
        self.sequence_table.configure(yscrollcommand=vsb_seq.set)
        vsb_seq.pack(side='right', fill='y')
        
        hsb_seq = ttk.Scrollbar(self.sequence_table, orient="horizontal", command=self.sequence_table.xview)
        self.sequence_table.configure(xscrollcommand=hsb_seq.set)
        hsb_seq.pack(side='bottom', fill='x')

        self.sequence_table.bind("<<TreeviewSelect>>", self.on_timestep_select)

        # --- 右下：分析区 (指标 + 图表) ---
        analysis_frame = ttk.Frame(right_paned, padding=5)
        right_paned.add(analysis_frame, weight=6) 

        analysis_paned = ttk.PanedWindow(analysis_frame, orient=tk.HORIZONTAL)
        analysis_paned.pack(fill=tk.BOTH, expand=True)

        # --- 右下左：指标表 ---
        metrics_frame = ttk.Frame(analysis_paned, padding=5)
        analysis_paned.add(metrics_frame, weight=5) 

        metrics_control_frame = ttk.Frame(metrics_frame)
        metrics_control_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(metrics_control_frame, text="历史累积指标分析 (Historical Metrics)", font=("-weight bold", 12)).pack(side=tk.LEFT)

        self.btn_toggle_analysis = ttk.Button(metrics_control_frame, text="切换到 Concept 分析", command=self.toggle_analysis_mode)
        self.btn_toggle_analysis.pack(side=tk.RIGHT)
        self.btn_toggle_analysis.config(state=tk.DISABLED)
        
        metrics_cols = ['模型', '分析实体', 'ID', '历史总数', '真实正确率', '预测ACC', '预测AUC']
        self.metrics_table = ttk.Treeview(metrics_frame, columns=metrics_cols, show='headings', height=8) # 限制高度
        self.metrics_table.pack(fill=tk.BOTH, expand=True)

        for col in metrics_cols:
            self.metrics_table.heading(col, text=col)
            self.metrics_table.column(col, width=100, anchor='center')
        self.metrics_table.column('模型', width=120)
        
        # --- 修改点 1: 新增历史序列 ---
        ttk.Separator(metrics_frame, orient='horizontal').pack(fill=tk.X, pady=(10, 5))
        ttk.Label(metrics_frame, text="选中实体历史序列 (Selected Entity Sequence)", font=("-weight bold", 12)).pack(fill=tk.X)
        
        hist_seq_frame = ttk.Frame(metrics_frame)
        hist_seq_frame.pack(fill=tk.BOTH, expand=True)
        
        hist_cols = ['timestep', 'true']
        self.history_sequence_table = ttk.Treeview(hist_seq_frame, columns=hist_cols, show='headings', height=8)
        self.history_sequence_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        vsb_hist = ttk.Scrollbar(hist_seq_frame, orient="vertical", command=self.history_sequence_table.yview)
        self.history_sequence_table.configure(yscrollcommand=vsb_hist.set)
        vsb_hist.pack(side='right', fill='y')
        
        hsb_hist = ttk.Scrollbar(hist_seq_frame, orient="horizontal", command=self.history_sequence_table.xview)
        self.history_sequence_table.configure(xscrollcommand=hsb_hist.set)
        hsb_hist.pack(side='bottom', fill='x')
        # --- 新增结束 ---


        # --- 右下右：图表 ---
        plot_frame = ttk.Frame(analysis_paned, padding=5)
        analysis_paned.add(plot_frame, weight=5) 

        ttk.Label(plot_frame, text="历史预测可视化 (Historical Visualization)", font=("-weight bold", 12)).pack(fill=tk.X)

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.ax.set_title("Please select a timestep to analyze")
        self.ax.set_xlabel("Time Step")
        self.ax.set_ylabel("Prediction / True Value")
        self.plot_canvas.draw()


    def clear_treeview(self, tree):
        for item in tree.get_children():
            tree.delete(item)

    def load_csv(self):
        filepath = filedialog.askopenfilename(
            title="选择分析文件",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if not filepath:
            return

        try:
            self.df = pd.read_csv(filepath)
            self.lbl_file.config(text=filepath.split('/')[-1])

            self.model_names = []

            for col in self.df.columns:
                if col.startswith('trues_'):
                    model_name = col.replace('trues_', '')
                    self.model_names.append(model_name)

            if not self.model_names:
                raise Exception("未找到 'trues_' 或 'preds_' 领衔的列")
            
            base_model = self.model_names[0]

            for model in self.model_names:
                # self.df[f'trues_{model}'] = self.df[f'trues_{model}'].apply(ast.literal_eval)
                # self.df[f'preds_{model}'] = self.df[f'preds_{model}'].apply(ast.literal_eval)
                self.df[f'trues_{model}'] = self.df[f'trues_{model}'].apply(self._parse_list_string)
                self.df[f'preds_{model}'] = self.df[f'preds_{model}'].apply(self._parse_list_string)
        
            # --- 修改点 2: 预计算列, 存入 self.df ---
            auc_cols = [f'auc_{model}' for model in self.model_names]
            delta_auc_cols = [f'delta_{model}' for model in self.model_names[1:]]
            
            # 预计算 seq_len
            self.df['seq_len'] = self.df[f'trues_{base_model}'].apply(len)
            
            # 预计算 delta_auc
            base_auc_col = self.df[f'auc_{base_model}']
            for model in self.model_names[1:]:
                delta_col_name = f'delta_{model}'
                try:
                    self.df[delta_col_name] = self.df[f'auc_{model}'] - base_auc_col
                except Exception:
                    self.df[delta_col_name] = np.nan # 使用 np.nan 便于后续处理

            # 存储列名
            self.student_cols = ['student_id', 'seq_len'] + auc_cols + delta_auc_cols
            self.student_table.config(columns=self.student_cols)
            
            self.student_table_headings = {} 
            self.last_sort_col = None
            self.last_sort_reverse = False

            for col in self.student_cols:
                col_name = col
                if col.startswith('delta_'):
                    col_name = f"Δ_auc_{col.replace('delta_', '')}"
                
                self.student_table_headings[col] = col_name 
                self.student_table.heading(col, text=col_name, 
                                           command=lambda c=col: self.sort_by_column(self.student_table, c))
                width = 80
                if col == 'student_id': width = 60
                elif col == 'seq_len': width = 50
                elif col.startswith('auc_'): width = 90
                self.student_table.column(col, width=width, anchor='center')

            # --- 修改点 2: 更新筛选菜单 ---
            self.filter_col_menu['menu'].delete(0, 'end')
            for col in self.student_cols:
                self.filter_col_menu['menu'].add_command(label=col, command=tk._setit(self.filter_col_var, col))
            self.filter_col_var.set(self.student_cols[0]) # 默认选中第一个
            # --- 结束 ---
            
            # 重置表格
            self.reset_student_table()

            print(f"加载成功! 识别到 {len(self.model_names)} 个模型: {self.model_names}")

        except Exception as e:
            messagebox.showerror("加载失败 (Load Failed)", f"无法解析CSV文件 (Failed to parse CSV): {e}\n\n请确保文件是上一步生成的CSV，且包含 `trues_` 和 `preds_` 列。")

    # --- 新函数: (重构) 填充学生表 ---
    def _populate_student_table(self, df_to_render):
        self.clear_treeview(self.student_table)
        
        auc_cols = [f'auc_{model}' for model in self.model_names]
        delta_auc_cols = [f'delta_{model}' for model in self.model_names[1:]]

        for _, row in df_to_render.iterrows():
            auc_values = [f"{row[col]:.4f}" if pd.notna(row[col]) else 'N/A' for col in auc_cols]
            delta_auc_values = [f"{row[col]:+.4f}" if pd.notna(row[col]) else 'N/A' for col in delta_auc_cols]
            
            values = [row['student_id'], row['seq_len']] + auc_values + delta_auc_values
            self.student_table.insert('', tk.END, values=values)
            
    # --- 新函数: 重置学生表 (筛选用) ---
    def reset_student_table(self):
        if self.df is None:
            return
        self._populate_student_table(self.df)
        # 重置排序箭头
        self.last_sort_col = None
        for col in self.student_cols:
            self.student_table.heading(col, text=self.student_table_headings[col])

    # --- 新函数: 应用筛选 (筛选用) ---
    def apply_student_filter(self):
        if self.df is None:
            return
            
        col = self.filter_col_var.get()
        op = self.filter_op_var.get()
        val_str = self.filter_val_entry.get()
        
        if col == "选择列":
            messagebox.showwarning("筛选错误", "请选择一个筛选列。")
            return
            
        try:
            val = float(val_str)
        except ValueError:
            messagebox.showwarning("筛选错误", f"值 '{val_str}' 不是一个有效的数字。")
            return

        try:
            # 执行筛选
            if op == '>': filtered_df = self.df[self.df[col] > val].copy()
            elif op == '<': filtered_df = self.df[self.df[col] < val].copy()
            elif op == '>=': filtered_df = self.df[self.df[col] >= val].copy()
            elif op == '<=': filtered_df = self.df[self.df[col] <= val].copy()
            elif op == '==': filtered_df = self.df[self.df[col] == val].copy()
            elif op == '!=': filtered_df = self.df[self.df[col] != val].copy()
            else:
                filtered_df = self.df.copy()
                
            self._populate_student_table(filtered_df)
            
        except Exception as e:
            messagebox.showerror("筛选失败", f"无法应用筛选: {e}\n请确保所选列是数值型。")

    # --- 修改点 3: 修复排序逻辑 ---
    def safe_float(self, val, default):
        """辅助函数：安全地将字符串转为浮点数，处理N/A"""
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def sort_by_column(self, tree, col):
        if col == self.last_sort_col:
            reverse = not self.last_sort_reverse
        else:
            reverse = False

        # 根据排序方向，为 N/A 或无效值设置默认值
        # 升序时, N/A 放在最前 (负无穷)
        # 降序时, N/A 放在最后 (负无穷)
        default_val = float('-inf') 
        
        # 从 treeview 中获取数据，使用 safe_float 转换
        l = [(self.safe_float(tree.set(k, col), default=default_val), k) 
             for k in tree.get_children('')]

        # 按数值排序
        l.sort(key=lambda t: t[0], reverse=reverse)

        # 重新排列
        for index, (val, k) in enumerate(l):
            tree.move(k, '', index)

        # 更新标题箭头
        for c in self.student_table_headings:
            tree.heading(c, text=self.student_table_headings[c]) 
            
        new_heading = self.student_table_headings[col] + (' ▼' if reverse else ' ▲')
        tree.heading(col, text=new_heading)

        self.last_sort_col = col
        self.last_sort_reverse = reverse
    # --- 排序修复结束 ---

    def on_student_select(self, event):
        if not self.student_table.selection():
            return

        selected_item = self.student_table.selection()[0]
        selected_student_id = int(self.student_table.item(selected_item)['values'][0])

        self.clear_treeview(self.sequence_table)
        self.clear_treeview(self.metrics_table)
        self.clear_treeview(self.history_sequence_table) # 修改点 1
        self.ax.clear()

        self.ax.set_title("Please select a timestep to analyze")
        self.ax.set_xlabel("Time Step")
        self.ax.set_ylabel("Prediction / True Value")
        self.plot_canvas.draw()

        self.current_clicked_timestep_data = None
        self.btn_toggle_analysis.config(state=tk.DISABLED, text="切换到 Concept 分析")
        self.analysis_mode = "Question"

        student_row = self.df[self.df['student_id'] == selected_student_id].iloc[0]

        q_list = str(student_row['questions']).split(',') 
        c_list = str(student_row['concepts']).split(',') 

        trues_list = student_row[f'trues_{self.model_names[0]}']

        if not (len(q_list) == len(c_list) == len(trues_list)):
            print(f"警告: 学生 {selected_student_id} 数据长度不一致! Qs:{len(q_list)}, Cs:{len(c_list)}, Trues:{len(trues_list)}")
            min_len = min(len(q_list), len(c_list), len(trues_list))
            q_list = q_list[:min_len]
            c_list = c_list[:min_len]
            trues_list = trues_list[:min_len]

        pred_cols = [f'pred_{model}' for model in self.model_names]
        seq_cols = ['timestep', 'q_id', 'c_id', 'true'] + pred_cols
        self.sequence_table.config(columns=seq_cols)

        for col in seq_cols:
            self.sequence_table.heading(col, text=col)
            self.sequence_table.column(col, width=70, anchor='center')

        self.current_student_data = []

        for i in range(len(q_list)):
            preds_map = {}
            row_values = [i, q_list[i], c_list[i], trues_list[i]]

            for model in self.model_names:
                pred_val = student_row[f'preds_{model}'][i]
                preds_map[model] = pred_val
                row_values.append(f"{pred_val:.3f}")

            self.sequence_table.insert('', tk.END, values=row_values, iid=str(i))

            self.current_student_data.append({
                'timestep': i,
                'q_id': q_list[i],
                'c_id': c_list[i],
                'true': trues_list[i],
                'preds': preds_map
            })
        
        try:
            for _ in range(3):
                self.sequence_table.insert('', tk.END, values=[""] * len(seq_cols))
        except Exception as e:
            print(f"添加空行失败: {e}")

    def on_timestep_select(self, event):
        if not self.sequence_table.selection() or not self.current_student_data:
            return
            
        try:
            selected_iid = self.sequence_table.selection()[0]
            clicked_timestep = int(selected_iid)
            clicked_data = self.current_student_data[clicked_timestep]
        except (IndexError, ValueError):
            return 

        self.current_clicked_timestep_data = {
            'q_id': clicked_data['q_id'],
            'c_id': clicked_data['c_id'],
            'timestep': clicked_timestep
        }

        self.btn_toggle_analysis.config(state=tk.NORMAL)
        
        self.run_analysis_and_plot()

    def toggle_analysis_mode(self):
        if self.analysis_mode == "Question":
            self.analysis_mode = "Concept"
            self.btn_toggle_analysis.config(text="切换到 Question 分析")
        else:
            self.analysis_mode = "Question"
            self.btn_toggle_analysis.config(text="切换到 Concept 分析")
        
        self.run_analysis_and_plot()

    def run_analysis_and_plot(self):
        if not self.current_clicked_timestep_data:
            return

        self.clear_treeview(self.metrics_table)
        self.clear_treeview(self.history_sequence_table) # 修改点 1

        q_id = self.current_clicked_timestep_data['q_id']
        c_id = self.current_clicked_timestep_data['c_id']
        plot_max_timestep = self.current_clicked_timestep_data['timestep']

        entity_type = self.analysis_mode
        entity_id = q_id if entity_type == 'Question' else c_id
        
        # --- 1. 为(Table)计算(全部)历史指标 ---
        hist_trues_all, hist_preds_all, hist_ts_all = self.get_historical_data(entity_type, entity_id)
        metrics_list = self.calculate_metrics_list(entity_type, entity_id, hist_trues_all, hist_preds_all)
        
        for metrics in metrics_list:
            if metrics:
                self.metrics_table.insert('', tk.END, values=[
                    metrics['model'], metrics['type'], metrics['id'],
                    metrics['count'], metrics['true_acc'],
                    metrics['pred_acc'], metrics['auc']
                ])
                
        # --- 2. 填充(全部)历史序列 (修改点 1) ---
        pred_cols = [f'pred_{model}' for model in self.model_names]
        hist_seq_cols = ['timestep', 'true'] + pred_cols
        self.history_sequence_table.config(columns=hist_seq_cols)
        for col in hist_seq_cols:
            self.history_sequence_table.heading(col, text=col)
            self.history_sequence_table.column(col, width=70, anchor='center')
        
        for i, ts in enumerate(hist_ts_all):
            true_val = hist_trues_all[i]
            pred_vals = [f"{hist_preds_all[model][i]:.3f}" for model in self.model_names]
            row_values = [ts, true_val] + pred_vals
            self.history_sequence_table.insert('', tk.END, values=row_values)
        
                
        # --- 3. 为(Plot)计算(截至到)点击时间的数据 ---
        hist_trues_part, hist_preds_part, hist_ts_part = self.get_historical_data(entity_type, entity_id, plot_max_timestep)
        plot_data = self.calculate_plot_data(hist_trues_part, hist_preds_part, hist_ts_part)
        
        # 更新图表
        if entity_type == "Question":
            title = f"Question ID: {q_id} (Up to timestep {plot_max_timestep})"
        else:
            title = f"Concept ID: {c_id} (Up to timestep {plot_max_timestep})"
        
        self.update_plot(plot_data, title)


    def get_historical_data(self, entity_type, entity_id, max_timestep=None):
        history_trues = []
        history_preds = {model: [] for model in self.model_names}
        history_timesteps = []  

        for item in self.current_student_data:
            if max_timestep is not None and item['timestep'] > max_timestep:
                break 

            current_entity_id = item['q_id'] if entity_type == 'Question' else item['c_id']
            
            if current_entity_id == entity_id:
                history_trues.append(item['true'])
                history_timesteps.append(item['timestep'])
                for model in self.model_names:
                    history_preds[model].append(item['preds'][model])
        
        return history_trues, history_preds, history_timesteps
    def _parse_list_string(self, s):
        """
        一个更健壮的列表解析函数，用于替换 ast.literal_eval。
        它可以处理字符串中的 'nan' 以及单元格本身就是 'nan' 的情况。
        """
        # 1. 处理输入不是字符串的情况 (例如，pd.read_csv 已经将其读为 np.nan)
        if not isinstance(s, str):
            if pd.isna(s):
                return [] # 如果是 np.nan，返回空列表
            return s if isinstance(s, list) else [] # 其他情况

        # 2. 处理单元格内容就是 'nan' 字符串
        if s == 'nan':
            return []

        # 3. 处理列表字符串中的 'nan'，例如 "[0.1, nan, 0.8]"
        try:
            # 关键：将 'nan' 替换为 'None'，ast.literal_eval 可以解析 'None'
            s_safe = s.replace('nan', 'None')
            
            parsed_list = ast.literal_eval(s_safe)
            
            # 将 'None' 转换回 np.nan 以便进行数学计算
            if isinstance(parsed_list, list):
                return [np.nan if x is None else x for x in parsed_list]
            else:
                return parsed_list # 理论上应该总是列表
                
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing string '{s}': {e}. Returning empty list.")
            return []
        except Exception as e:
            print(f"Unexpected error parsing string '{s}': {e}. Returning empty list.")
            return []
    # ^^^^^^ 新函数结束 ^^^^^^
    def calculate_metrics_list(self, entity_type, entity_id, history_trues, history_preds):
        final_metrics_list = []
        
        if len(history_trues) < 2 or len(set(history_trues)) < 2:
            for model in self.model_names:
                final_metrics_list.append({
                    'model': model, 'type': entity_type, 'id': entity_id,
                    'count': len(history_trues),
                    'true_acc': f"{(sum(history_trues) / len(history_trues)):.3f}" if len(history_trues) > 0 else "N/A",
                    'pred_acc': "N/A (No Preds)", 'auc': "N/A (Need >1 Class)"
                })
            return final_metrics_list

        for model in self.model_names:
            model_preds = history_preds[model]
            
            try:
                auc = roc_auc_score(history_trues, model_preds)
                auc_str = f"{auc:.3f}"
            except ValueError:
                auc_str = "N/A"
            
            pred_labels = [1 if p > 0.5 else 0 for p in model_preds]
            acc = accuracy_score(history_trues, pred_labels)
            true_acc = sum(history_trues) / len(history_trues)

            final_metrics_list.append({
                'model': model,
                'type': entity_type,
                'id': entity_id,
                'count': len(history_trues),
                'true_acc': f"{true_acc:.3f}",
                'pred_acc': f"{acc:.3f}",
                'auc': auc_str
            })
            
        return final_metrics_list

    def calculate_plot_data(self, history_trues, history_preds, history_timesteps):
        cumulative_accuracy = []
        if history_trues:
            cumulative_trues = np.cumsum(history_trues)
            cumulative_counts = np.arange(1, len(history_trues) + 1)
            cumulative_accuracy = (cumulative_trues / cumulative_counts).tolist()

        cumulative_preds_map = {}
        if history_trues:
            for model in self.model_names:
                model_preds_list = history_preds[model]
                if model_preds_list:
                    cumulative_sum = np.cumsum(model_preds_list)
                    cumulative_counts = np.arange(1, len(model_preds_list) + 1)
                    cumulative_preds_map[model] = (cumulative_sum / cumulative_counts).tolist()
                else:
                    cumulative_preds_map[model] = []
        
        plot_data = {
            'timesteps': history_timesteps,
            'trues': history_trues,
            'preds': history_preds, 
            'cumulative_accuracy': cumulative_accuracy,
            'cumulative_preds': cumulative_preds_map 
        }
        return plot_data

    def update_plot(self, plot_data, title):
        self.ax.clear()

        if not plot_data or not plot_data['timesteps']:
            self.ax.set_title("No historical data to plot")
            self.ax.set_xlabel("Time Step")
            self.ax.set_ylabel("Prediction / True Value")
            self.plot_canvas.draw()
            return

        timesteps = plot_data['timesteps']
        trues = plot_data['trues']
        
        cumulative_accuracy = plot_data.get('cumulative_accuracy')
        cumulative_preds_map = plot_data.get('cumulative_preds')

        true_plot_vals = [0.95 if t == 1 else 0.05 for t in trues]
        self.ax.scatter(timesteps, true_plot_vals, label="True Answer (0=False, 1=True)",
                        color='black', marker='x', s=50, zorder=3) 

        for model in self.model_names:
            preds = plot_data['preds'][model]
            if preds:
                line, = self.ax.plot(timesteps, preds, label=f"Pred ({model})", marker='.', zorder=2)
                
                if cumulative_preds_map:
                    cum_preds = cumulative_preds_map.get(model)
                    if cum_preds:
                        self.ax.plot(timesteps, cum_preds, 
                                     label=f"Cum. Pred ({model})", 
                                     linestyle=':', 
                                     color=line.get_color(), 
                                     zorder=2, marker='_') 


        if cumulative_accuracy:
            self.ax.plot(timesteps, cumulative_accuracy, 
                         label="Cumulative True Acc", 
                         color='green', linestyle='--', marker='o', 
                         markersize=3, zorder=1)

        self.ax.set_title(title)
        self.ax.set_xlabel("Time Step")
        self.ax.set_ylabel("Prediction Probability / True Answer")
        self.ax.set_ylim(-0.1, 1.1)

        self.ax.legend(loc='lower center', 
                       bbox_to_anchor=(0.5, 1.15), 
                       ncol=4,  
                       fontsize='small') 

        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        self.fig.subplots_adjust(left=0.12, bottom=0.15, right=0.98, top=0.82)

        self.plot_canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = KTAnalysisGUI(root)
    root.mainloop()