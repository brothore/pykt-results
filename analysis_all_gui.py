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

# 优化字体
plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 如需中文支持请取消注释

class KTAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("知识追踪 (KT) 序列深度分析工具 - 复杂度增强版")
        self.root.geometry("1600x900")

        self.df = None
        self.model_names = []
        self.student_cols = [] 
        self.current_student_data = [] 
        self.model_vars = {} 
        self.model_checkbuttons = {} 

        self.student_table_headings = {} 
        self.last_sort_col = None
        self.last_sort_reverse = False

        self.selected_student_id = None
        self.current_timesteps = []
        self.current_trues = []
        self.current_preds_map = {}

        self.create_widgets()

    def create_widgets(self):
        # --- 顶部菜单 ---
        top_frame = ttk.Frame(self.root, padding=5)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_load = ttk.Button(top_frame, text="加载 CSV 分析文件 (Load CSV)", command=self.load_csv)
        self.btn_load.pack(side=tk.LEFT)
        self.lbl_file = ttk.Label(top_frame, text="未加载文件")
        self.lbl_file.pack(side=tk.LEFT, padx=10)

        # --- 主内容区 ---
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # ================== 左侧面板：学生列表 ==================
        left_frame = ttk.Frame(main_paned, padding=5)
        main_paned.add(left_frame, weight=3) 

        ttk.Label(left_frame, text="学生序列特征总览 (Student Features)", font=("-weight bold", 12)).pack(fill=tk.X)

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

        # 初始化列 (稍后在load_csv中动态调整)
        student_cols = ['student_id', 'seq_len']
        self.student_table = ttk.Treeview(left_frame, columns=student_cols, show='headings', height=25)
        self.student_table.pack(fill=tk.BOTH, expand=True)

        vsb_student = ttk.Scrollbar(self.student_table, orient="vertical", command=self.student_table.yview)
        self.student_table.configure(yscrollcommand=vsb_student.set)
        vsb_student.pack(side='right', fill='y')

        self.student_table.bind("<<TreeviewSelect>>", self.on_student_select)

        # ================== 右侧面板 ==================
        right_paned = ttk.PanedWindow(main_paned, orient=tk.VERTICAL)
        main_paned.add(right_paned, weight=7) 

        # --- 右上：序列详情表格 ---
        seq_frame = ttk.Frame(right_paned, padding=5)
        right_paned.add(seq_frame, weight=3) 

        ttk.Label(seq_frame, text="学生序列逐题详情 (Step-by-Step Detail)", font=("-weight bold", 12)).pack(fill=tk.X)

        seq_cols = ['timestep', 'q_id', 'c_id', 'true']
        self.sequence_table = ttk.Treeview(seq_frame, columns=seq_cols, show='headings', height=8)
        self.sequence_table.pack(fill=tk.BOTH, expand=True)

        vsb_seq = ttk.Scrollbar(self.sequence_table, orient="vertical", command=self.sequence_table.yview)
        self.sequence_table.configure(yscrollcommand=vsb_seq.set)
        vsb_seq.pack(side='right', fill='y')
        
        hsb_seq = ttk.Scrollbar(self.sequence_table, orient="horizontal", command=self.sequence_table.xview)
        self.sequence_table.configure(xscrollcommand=hsb_seq.set)
        hsb_seq.pack(side='bottom', fill='x')

        self.sequence_table.bind("<<TreeviewSelect>>", self.on_timestep_select)

        # --- 右下：全局预测曲线图 ---
        plot_container = ttk.Frame(right_paned, padding=5)
        right_paned.add(plot_container, weight=7) 

        self.model_control_frame = ttk.LabelFrame(plot_container, text="模型显示控制", padding=5)
        self.model_control_frame.pack(fill=tk.X, side=tk.TOP, pady=(0, 5))

        ttk.Label(plot_container, text="预测概率 & 累计AUC (Probability & Cumulative AUC)", font=("-weight bold", 12)).pack(fill=tk.X)

        self.plot_scroll_frame = ttk.Frame(plot_container)
        self.plot_scroll_frame.pack(fill=tk.BOTH, expand=True)

        self.plot_h_scrollbar = ttk.Scrollbar(self.plot_scroll_frame, orient=tk.HORIZONTAL)
        self.plot_h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.plot_canvas_wrapper = tk.Canvas(self.plot_scroll_frame, bg="white", xscrollcommand=self.plot_h_scrollbar.set)
        self.plot_canvas_wrapper.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.plot_h_scrollbar.config(command=self.plot_canvas_wrapper.xview)

        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=self.plot_canvas_wrapper)
        self.chart_widget = self.chart_canvas.get_tk_widget()
        self.canvas_window_id = self.plot_canvas_wrapper.create_window(0, 0, window=self.chart_widget, anchor="nw")

        self.plot_canvas_wrapper.bind("<Configure>", self._on_canvas_configure)
        self._init_empty_plot()

    def _on_canvas_configure(self, event):
        self.plot_canvas_wrapper.itemconfig(self.canvas_window_id, height=event.height)

    def _init_empty_plot(self):
        self.ax.clear()
        self.ax.text(0.5, 0.5, "Please load data and select a student", ha='center', va='center')
        self.chart_canvas.draw()

    def clear_treeview(self, tree):
        for item in tree.get_children():
            tree.delete(item)

    # ================== 核心算法：复杂度指标计算 ==================
    def _calculate_complexity_metrics(self, row, base_model):
        """
        计算三个核心指标：
        1. LZ Complexity (Lempel-Ziv): 序列的可压缩性/规律性
        2. Switch Rate (SR): 状态反转频率 (0->1 或 1->0)
        3. Conditional Entropy: 已知前一次作答，下一次作答的不确定性
        """
        seq = row[f'trues_{base_model}']
        
        # 异常处理：如果没有数据或长度太短
        if not isinstance(seq, list) or len(seq) < 2:
            return 0, 0.0, 0.0

        # --- 1. Lempel-Ziv Complexity (LZ78 Set-based implementation) ---
        # 逻辑：从左到右扫描，如果当前子串在之前出现过，就延长子串；没出现过，就记为一个新 pattern
        s_str = ''.join(map(str, map(int, seq))) # 转为字符串 "10110..."
        seen_patterns = set()
        i = 0
        lz_count = 0
        while i < len(s_str):
            sub = s_str[i]
            j = i + 1
            # 贪婪匹配：寻找未见过的最短子串
            while sub in seen_patterns and j < len(s_str):
                sub += s_str[j]
                j += 1
            seen_patterns.add(sub)
            lz_count += 1
            i = j
        
        # --- 2. Switch Rate (SR) ---
        # 计算相邻两次作答不一致的比例
        arr = np.array(seq)
        changes = np.sum(arr[:-1] != arr[1:])
        sr = changes / (len(seq) - 1)

        # --- 3. Conditional Entropy H(Y_t | Y_{t-1}) ---
        # H(Next|Prev) = P(Prev=0)*H(Next|Prev=0) + P(Prev=1)*H(Next|Prev=1)
        
        # 构造 (前一个, 后一个) 的配对
        pairs = list(zip(seq[:-1], seq[1:]))
        n_pairs = len(pairs)
        
        if n_pairs == 0:
            entropy = 0.0
        else:
            # 统计
            count_prev_0 = seq[:-1].count(0)
            count_prev_1 = seq[:-1].count(1)
            
            # 转移统计
            c00 = pairs.count((0, 0)) # 0 -> 0
            c01 = pairs.count((0, 1)) # 0 -> 1
            c10 = pairs.count((1, 0)) # 1 -> 0
            c11 = pairs.count((1, 1)) # 1 -> 1
            
            # 辅助函数：计算二元熵
            def binary_entropy(p):
                if p <= 1e-9 or p >= 1 - 1e-9: return 0.0
                return -p * np.log2(p) - (1-p) * np.log2(1-p)
            
            # 计算条件概率的熵
            h_given_0 = binary_entropy(c01 / count_prev_0) if count_prev_0 > 0 else 0
            h_given_1 = binary_entropy(c11 / count_prev_1) if count_prev_1 > 0 else 0
            
            # 加权求和
            p_prev_0 = count_prev_0 / n_pairs
            p_prev_1 = count_prev_1 / n_pairs
            
            entropy = p_prev_0 * h_given_0 + p_prev_1 * h_given_1

        return lz_count, sr, entropy

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

            # 解析字符串列表
            for model in self.model_names:
                self.df[f'trues_{model}'] = self.df[f'trues_{model}'].apply(self._parse_list_string)
                self.df[f'preds_{model}'] = self.df[f'preds_{model}'].apply(self._parse_list_string)
        
            # --- 预计算列 ---
            self.df['seq_len'] = self.df[f'trues_{base_model}'].apply(len)
            
            # 计算复杂度指标
            print("正在计算序列复杂度指标 (LZ, SR, Entropy)...")
            complexity_data = self.df.apply(lambda row: self._calculate_complexity_metrics(row, base_model), axis=1)
            
            self.df['lz_complexity'] = complexity_data.apply(lambda x: x[0])
            self.df['switch_rate'] = complexity_data.apply(lambda x: x[1])
            self.df['cond_entropy'] = complexity_data.apply(lambda x: x[2])
            
            # --- 【新增】计算 LZ Ratio (归一化复杂度) ---
            # 避免除以0，虽然 seq_len 一般>=1
            self.df['lz_ratio'] = self.df.apply(lambda x: x['lz_complexity'] / x['seq_len'] if x['seq_len'] > 0 else 0, axis=1)

            auc_cols = [f'auc_{model}' for model in self.model_names]
            delta_auc_cols = [f'delta_{model}' for model in self.model_names[1:]]
            
            base_auc_col = self.df[f'auc_{base_model}']
            for model in self.model_names[1:]:
                delta_col_name = f'delta_{model}'
                try:
                    self.df[delta_col_name] = self.df[f'auc_{model}'] - base_auc_col
                except Exception:
                    self.df[delta_col_name] = np.nan 

            # *** 更新表格列定义：加入 lz_ratio ***
            # 顺序：ID, Len, LZ数值, LZ比率, SR, Entropy, AUCs...
            self.student_cols = ['student_id', 'seq_len', 'lz_complexity', 'lz_ratio', 'switch_rate', 'cond_entropy'] + auc_cols + delta_auc_cols
            self.student_table.config(columns=self.student_cols)
            
            self.student_table_headings = {} 
            self.last_sort_col = None
            self.last_sort_reverse = False

            for col in self.student_cols:
                col_name = col
                # 美化列名显示
                if col == 'lz_complexity': col_name = 'LZ (Val)'
                elif col == 'lz_ratio': col_name = 'LZ Rate' # 新增显示的列名
                elif col == 'switch_rate': col_name = 'SR (Switch)'
                elif col == 'cond_entropy': col_name = 'Entropy (H)'
                elif col.startswith('delta_'): col_name = f"Δ_{col.replace('delta_', '')}"
                elif col.startswith('auc_'): col_name = f"AUC_{col.replace('auc_', '')}"
                
                self.student_table_headings[col] = col_name 
                self.student_table.heading(col, text=col_name, 
                                     command=lambda c=col: self.sort_by_column(self.student_table, c))
                
                # 设置列宽
                width = 80
                if col == 'student_id': width = 60
                elif col == 'seq_len': width = 50
                elif col == 'lz_ratio': width = 60 # 新列宽度
                elif col in ['lz_complexity', 'switch_rate', 'cond_entropy']: width = 70 
                elif col.startswith('auc_'): width = 90
                self.student_table.column(col, width=width, anchor='center')

            # 更新筛选菜单
            self.filter_col_menu['menu'].delete(0, 'end')
            for col in self.student_cols:
                self.filter_col_menu['menu'].add_command(label=col, command=tk._setit(self.filter_col_var, col))
            self.filter_col_var.set(self.student_cols[0]) 
            
            # --- 生成模型复选框 ---
            for widget in self.model_control_frame.winfo_children():
                widget.destroy()
            
            self.model_vars = {}
            self.model_checkbuttons = {}
            
            for model in self.model_names:
                var = tk.BooleanVar(value=True) 
                cb = ttk.Checkbutton(self.model_control_frame, text=model, variable=var, 
                                     command=self.refresh_plot) 
                cb.pack(side=tk.LEFT, padx=10)
                self.model_vars[model] = var
                self.model_checkbuttons[model] = cb

            self.reset_student_table()
            print(f"加载成功! 指标计算完毕。")

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
            # 格式化 AUC
            auc_values = [f"{row[col]:.4f}" if pd.notna(row[col]) else 'N/A' for col in auc_cols]
            delta_auc_values = [f"{row[col]:+.4f}" if pd.notna(row[col]) else 'N/A' for col in delta_auc_cols]
            
            # 格式化新指标
            lz = int(row['lz_complexity']) if pd.notna(row['lz_complexity']) else 0
            
            # --- 【新增】格式化 LZ Ratio ---
            lz_ratio_str = f"{row['lz_ratio']:.3f}" if pd.notna(row['lz_ratio']) else '0.000'
            
            sr = f"{row['switch_rate']:.3f}" if pd.notna(row['switch_rate']) else '0.000'
            ent = f"{row['cond_entropy']:.3f}" if pd.notna(row['cond_entropy']) else '0.000'

            # 组合数据：ID, Len, LZ数值, 【LZ比率】, SR, Ent, AUCs...
            values = [row['student_id'], row['seq_len'], lz, lz_ratio_str, sr, ent] + auc_values + delta_auc_values
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
        
        # 针对不同列设置默认值（防止None导致排序报错）
        default_val = float('-inf')
        
        l = []
        for k in tree.get_children(''):
            val_str = tree.set(k, col)
            # 处理 N/A 或空值
            if val_str == 'N/A' or val_str == '':
                val = default_val
            else:
                val = self.safe_float(val_str, default_val)
            l.append((val, k))
            
        l.sort(key=lambda t: t[0], reverse=reverse)
        
        for index, (val, k) in enumerate(l):
            tree.move(k, '', index)
            
        for c in self.student_table_headings:
            tree.heading(c, text=self.student_table_headings[c]) 
        new_heading = self.student_table_headings[col] + (' ▼' if reverse else ' ▲')
        tree.heading(col, text=new_heading)
        self.last_sort_col = col
        self.last_sort_reverse = reverse

    # ================== 交互逻辑 (保持不变) ==================
    def on_student_select(self, event):
        if not self.student_table.selection(): return
        selected_item = self.student_table.selection()[0]
        self.selected_student_id = int(self.student_table.item(selected_item)['values'][0])
        self.clear_treeview(self.sequence_table)
        
        student_row = self.df[self.df['student_id'] == self.selected_student_id].iloc[0]
        q_list = str(student_row['questions']).split(',') 
        c_list = str(student_row['concepts']).split(',') 
        trues_list = student_row[f'trues_{self.model_names[0]}']
        min_len = min(len(q_list), len(c_list), len(trues_list))
        
        q_list = q_list[:min_len]
        c_list = c_list[:min_len]
        trues_list = trues_list[:min_len]
        self.current_timesteps = list(range(min_len))
        self.current_trues = trues_list
        self.current_preds_map = {}
        for model in self.model_names:
            p_list = student_row[f'preds_{model}']
            self.current_preds_map[model] = p_list[:min_len]

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
                val = self.current_preds_map[model][i]
                row_values.append(f"{val:.3f}")
                preds_at_t[model] = val
            self.sequence_table.insert('', tk.END, values=row_values, iid=str(i))
            self.current_student_data.append({'timestep': i, 'q_id': q_list[i], 'c_id': c_list[i], 'true': trues_list[i], 'preds': preds_at_t})
        self.refresh_plot()

    def refresh_plot(self):
        if self.selected_student_id is None: return
        self.ax.clear()
        
        points_per_view = 30 
        default_width_inches = 10 
        dpi = 100
        pixels_per_step = (default_width_inches * dpi) / points_per_view
        points_count = len(self.current_timesteps)
        required_width_inches = max(default_width_inches, (points_count * pixels_per_step) / dpi)
        
        self.fig.set_size_inches(required_width_inches, 5) 
        self.chart_canvas.draw()
        self.plot_canvas_wrapper.itemconfig(self.canvas_window_id, width=required_width_inches*dpi, height=500) 
        self.plot_canvas_wrapper.configure(scrollregion=(0, 0, required_width_inches*dpi, 500))

        true_indices = [i for i, val in enumerate(self.current_trues) if val == 1]
        false_indices = [i for i, val in enumerate(self.current_trues) if val == 0]
        
        self.ax.scatter(true_indices, [1.02]*len(true_indices), color='green', marker='o', s=40, label='True: Correct', zorder=5, clip_on=False)
        self.ax.scatter(false_indices, [-0.02]*len(false_indices), color='red', marker='x', s=40, label='True: Incorrect', zorder=5, clip_on=False)

        colors = plt.cm.get_cmap('tab10') 
        y_true_full = np.array(self.current_trues)
        for idx, model in enumerate(self.model_names):
            if not self.model_vars[model].get(): continue
            preds = self.current_preds_map[model]
            safe_preds = [p if pd.notna(p) else 0.5 for p in preds]
            color = colors(idx % 10)
            self.ax.plot(self.current_timesteps, safe_preds, label=f'{model} Pred', color=color, marker='.', markersize=4, linewidth=1.5, alpha=0.8)
            
            cum_auc_list = []
            for t in range(len(self.current_timesteps)):
                current_sub_true = y_true_full[:t+1]
                current_sub_pred = safe_preds[:t+1]
                if len(np.unique(current_sub_true)) < 2:
                    cum_auc_list.append(np.nan)
                else:
                    try:
                        score = roc_auc_score(current_sub_true, current_sub_pred)
                        cum_auc_list.append(score)
                    except:
                        cum_auc_list.append(np.nan)
            self.ax.plot(self.current_timesteps, cum_auc_list, label=f'{model} Cum. AUC', color=color, linestyle='--', linewidth=1.5, alpha=0.6)

        self.ax.set_title(f"Student {self.selected_student_id}: Trajectory & Cumulative AUC")
        self.ax.set_xlabel("Time Step")
        self.ax.set_ylabel("Probability / AUC")
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.set_xlim(-0.5, len(self.current_timesteps) - 0.5)
        self.ax.set_xticks(self.current_timesteps) 
        self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(self.model_names)*2 + 2, fontsize='small')
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.85, bottom=0.15) 
        self.chart_canvas.draw()

    def on_timestep_select(self, event):
        if not self.sequence_table.selection(): return
        try:
            selected_iid = self.sequence_table.selection()[0]
            clicked_timestep = int(selected_iid)
            for line in self.ax.lines:
                if line.get_label() == '_highlight_line':
                    line.remove()
            self.ax.axvline(x=clicked_timestep, color='orange', linestyle='-', alpha=0.5, linewidth=20, label='_highlight_line')
            self.chart_canvas.draw()
        except (ValueError, IndexError):
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = KTAnalysisGUI(root)
    root.mainloop()