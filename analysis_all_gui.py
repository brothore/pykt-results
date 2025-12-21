import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import ast
from sklearn.metrics import roc_auc_score

# 导入matplotlib相关库
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

#以此优化图表显示中文字体 (如果有需要可开启，否则使用默认)
plt.rcParams['axes.unicode_minus'] = False 

class KTAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("知识追踪 (KT) 序列可视化工具 (Knowledge Tracing Sequence Visualization)")
        self.root.geometry("1600x900")

        self.df = None
        self.model_names = []
        self.student_cols = [] 
        self.current_student_data = [] 

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

        # ================== 左侧面板：学生列表 ==================
        left_frame = ttk.Frame(main_paned, padding=5)
        main_paned.add(left_frame, weight=3) 

        ttk.Label(left_frame, text="学生总览 (Student Overview)", font=("-weight bold", 12)).pack(fill=tk.X)

        # --- 筛选功能 ---
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

        # ================== 右侧面板 (再上下分割) ==================
        right_paned = ttk.PanedWindow(main_paned, orient=tk.VERTICAL)
        main_paned.add(right_paned, weight=7) 

        # --- 右上：序列详情表格 ---
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

        # --- 右下：全局预测曲线图 (改动最大区域) ---
        plot_frame = ttk.Frame(right_paned, padding=5)
        right_paned.add(plot_frame, weight=6) 

        ttk.Label(plot_frame, text="全序列预测概率 vs 真实作答 (Full Sequence Prediction vs True)", font=("-weight bold", 12)).pack(fill=tk.X)

        # 创建图表对象
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._init_empty_plot()

    def _init_empty_plot(self):
        self.ax.clear()
        self.ax.set_title("Please select a student to visualize trajectory")
        self.ax.set_xlabel("Time Step")
        self.ax.set_ylabel("Probability / Correctness")
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.plot_canvas.draw()

    def clear_treeview(self, tree):
        for item in tree.get_children():
            tree.delete(item)

    # ================== 数据加载与处理 ==================
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
                self.df[f'trues_{model}'] = self.df[f'trues_{model}'].apply(self._parse_list_string)
                self.df[f'preds_{model}'] = self.df[f'preds_{model}'].apply(self._parse_list_string)
        
            # 预计算列
            auc_cols = [f'auc_{model}' for model in self.model_names]
            delta_auc_cols = [f'delta_{model}' for model in self.model_names[1:]]
            
            self.df['seq_len'] = self.df[f'trues_{base_model}'].apply(len)
            
            base_auc_col = self.df[f'auc_{base_model}']
            for model in self.model_names[1:]:
                delta_col_name = f'delta_{model}'
                try:
                    self.df[delta_col_name] = self.df[f'auc_{model}'] - base_auc_col
                except Exception:
                    self.df[delta_col_name] = np.nan 

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

            # 更新筛选菜单
            self.filter_col_menu['menu'].delete(0, 'end')
            for col in self.student_cols:
                self.filter_col_menu['menu'].add_command(label=col, command=tk._setit(self.filter_col_var, col))
            self.filter_col_var.set(self.student_cols[0]) 
            
            self.reset_student_table()
            print(f"加载成功! 识别到 {len(self.model_names)} 个模型: {self.model_names}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("加载失败", f"无法解析CSV文件: {e}\n\n请查看控制台输出详情。")

    def _parse_list_string(self, s):
        if not isinstance(s, str):
            if pd.isna(s): return [] 
            return s if isinstance(s, list) else [] 
        if s == 'nan': return []
        try:
            s_safe = s.replace('nan', 'None')
            parsed_list = ast.literal_eval(s_safe)
            if isinstance(parsed_list, list):
                return [np.nan if x is None else x for x in parsed_list]
            else:
                return parsed_list 
        except Exception:
            return []

    def _populate_student_table(self, df_to_render):
        self.clear_treeview(self.student_table)
        auc_cols = [f'auc_{model}' for model in self.model_names]
        delta_auc_cols = [f'delta_{model}' for model in self.model_names[1:]]

        for _, row in df_to_render.iterrows():
            auc_values = [f"{row[col]:.4f}" if pd.notna(row[col]) else 'N/A' for col in auc_cols]
            delta_auc_values = [f"{row[col]:+.4f}" if pd.notna(row[col]) else 'N/A' for col in delta_auc_cols]
            values = [row['student_id'], row['seq_len']] + auc_values + delta_auc_values
            self.student_table.insert('', tk.END, values=values)

    def reset_student_table(self):
        if self.df is None: return
        self._populate_student_table(self.df)
        self.last_sort_col = None
        for col in self.student_cols:
            self.student_table.heading(col, text=self.student_table_headings[col])

    def apply_student_filter(self):
        if self.df is None: return
        col = self.filter_col_var.get()
        op = self.filter_op_var.get()
        val_str = self.filter_val_entry.get()
        if col == "选择列": return
        try:
            val = float(val_str)
        except ValueError:
            messagebox.showwarning("筛选错误", f"值 '{val_str}' 不是有效的数字。")
            return

        try:
            if op == '>': filtered_df = self.df[self.df[col] > val].copy()
            elif op == '<': filtered_df = self.df[self.df[col] < val].copy()
            elif op == '>=': filtered_df = self.df[self.df[col] >= val].copy()
            elif op == '<=': filtered_df = self.df[self.df[col] <= val].copy()
            elif op == '==': filtered_df = self.df[self.df[col] == val].copy()
            elif op == '!=': filtered_df = self.df[self.df[col] != val].copy()
            else: filtered_df = self.df.copy()
            self._populate_student_table(filtered_df)
        except Exception as e:
            messagebox.showerror("筛选失败", f"{e}")

    def safe_float(self, val, default):
        try: return float(val)
        except (ValueError, TypeError): return default

    def sort_by_column(self, tree, col):
        if col == self.last_sort_col:
            reverse = not self.last_sort_reverse
        else:
            reverse = False
        default_val = float('-inf') 
        l = [(self.safe_float(tree.set(k, col), default=default_val), k) for k in tree.get_children('')]
        l.sort(key=lambda t: t[0], reverse=reverse)
        for index, (val, k) in enumerate(l):
            tree.move(k, '', index)
        for c in self.student_table_headings:
            tree.heading(c, text=self.student_table_headings[c]) 
        new_heading = self.student_table_headings[col] + (' ▼' if reverse else ' ▲')
        tree.heading(col, text=new_heading)
        self.last_sort_col = col
        self.last_sort_reverse = reverse

    # ================== 核心交互逻辑 ==================

    def on_student_select(self, event):
        """当选中学生时，加载序列详情，并直接绘制整条曲线"""
        if not self.student_table.selection():
            return

        selected_item = self.student_table.selection()[0]
        selected_student_id = int(self.student_table.item(selected_item)['values'][0])

        self.clear_treeview(self.sequence_table)
        self.ax.clear()

        # 获取学生数据
        student_row = self.df[self.df['student_id'] == selected_student_id].iloc[0]

        q_list = str(student_row['questions']).split(',') 
        c_list = str(student_row['concepts']).split(',') 
        
        # 基础真实标签 (取第一个模型的真实标签即可，因为真实标签是一样的)
        trues_list = student_row[f'trues_{self.model_names[0]}']

        # 安全截断
        min_len = min(len(q_list), len(c_list), len(trues_list))
        q_list = q_list[:min_len]
        c_list = c_list[:min_len]
        trues_list = trues_list[:min_len]

        # 收集所有模型的预测值
        all_preds_map = {} # { 'DKT': [0.1, 0.2...], 'AKT': [...] }
        for model in self.model_names:
            p_list = student_row[f'preds_{model}']
            all_preds_map[model] = p_list[:min_len]

        # 1. 填充序列详情表
        pred_cols = [f'pred_{model}' for model in self.model_names]
        seq_cols = ['timestep', 'q_id', 'c_id', 'true'] + pred_cols
        self.sequence_table.config(columns=seq_cols)

        for col in seq_cols:
            self.sequence_table.heading(col, text=col)
            self.sequence_table.column(col, width=70, anchor='center')

        self.current_student_data = []

        for i in range(min_len):
            row_values = [i, q_list[i], c_list[i], trues_list[i]]
            preds_at_t = {}
            for model in self.model_names:
                val = all_preds_map[model][i]
                row_values.append(f"{val:.3f}")
                preds_at_t[model] = val
            
            self.sequence_table.insert('', tk.END, values=row_values, iid=str(i))
            
            self.current_student_data.append({
                'timestep': i, 'q_id': q_list[i], 'c_id': c_list[i],
                'true': trues_list[i], 'preds': preds_at_t
            })

        # 2. 直接绘制整条曲线
        self.plot_full_student_trajectory(
            student_id=selected_student_id,
            timesteps=list(range(min_len)),
            trues=trues_list,
            preds_map=all_preds_map
        )

    def plot_full_student_trajectory(self, student_id, timesteps, trues, preds_map):
        """绘制所有模型的预测曲线以及学生的实际作答情况"""
        self.ax.clear()

        # 1. 绘制真实作答 (散点)
        # 正确(1)显示为绿色点，错误(0)显示为红色点
        true_indices = [i for i, val in enumerate(trues) if val == 1]
        false_indices = [i for i, val in enumerate(trues) if val == 0]
        
        self.ax.scatter(true_indices, [1]*len(true_indices), color='green', marker='o', s=40, label='True: Correct', zorder=5)
        self.ax.scatter(false_indices, [0]*len(false_indices), color='red', marker='x', s=40, label='True: Incorrect', zorder=5)

        # 2. 绘制各模型预测曲线
        colors = plt.cm.get_cmap('tab10') # 获取一组颜色
        for idx, model in enumerate(self.model_names):
            preds = preds_map[model]
            # 处理可能的 nan
            safe_preds = [p if pd.notna(p) else 0.5 for p in preds]
            
            color = colors(idx % 10)
            self.ax.plot(timesteps, safe_preds, label=f'{model} Pred', 
                         color=color, marker='.', markersize=4, linewidth=1.5, alpha=0.8)

        # 3. 图表设置
        self.ax.set_title(f"Student {student_id}: Full Trajectory Analysis")
        self.ax.set_xlabel("Time Step (Question Sequence)")
        self.ax.set_ylabel("Probability / Correctness")
        self.ax.set_ylim(-0.1, 1.1) 
        
        # 图例放置在上方，避免遮挡曲线
        self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(self.model_names)+2, fontsize='small')
        
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        # 调整布局以适应图例
        self.fig.subplots_adjust(left=0.08, bottom=0.15, right=0.98, top=0.85)
        
        self.plot_canvas.draw()

    def on_timestep_select(self, event):
        """
        点击序列详情的某一行时，在图表上高亮该时间步。
        （可选功能，这里简单实现画一条竖线）
        """
        if not self.sequence_table.selection():
            return
        
        try:
            selected_iid = self.sequence_table.selection()[0]
            clicked_timestep = int(selected_iid)
            
            # 清除旧的竖线 (如果有)
            for line in self.ax.lines:
                if line.get_label() == '_highlight_line':
                    line.remove()
            
            # 绘制新的竖线
            self.ax.axvline(x=clicked_timestep, color='orange', linestyle='--', alpha=0.8, label='_highlight_line')
            self.plot_canvas.draw()
            
        except (ValueError, IndexError):
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = KTAnalysisGUI(root)
    root.mainloop()