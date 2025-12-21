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

# 优化字体 (Windows下通常用SimHei，Mac/Linux可能需要调整)
plt.rcParams['axes.unicode_minus'] = False 
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 如果中文显示方框，请取消此行注释

class KTAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("知识追踪 (KT) 序列可视化工具 - 增强版")
        self.root.geometry("1600x900")

        self.df = None
        self.model_names = []
        self.student_cols = [] 
        self.current_student_data = [] 
        self.model_vars = {} # 存储模型复选框状态 {model_name: tk.BooleanVar}
        self.model_checkbuttons = {} # 存储复选框组件

        self.student_table_headings = {} 
        self.last_sort_col = None
        self.last_sort_reverse = False

        # 当前选中的数据缓存，用于刷新图表
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
        right_paned.add(seq_frame, weight=3) 

        ttk.Label(seq_frame, text="学生序列详情 (Student Sequence Detail)", font=("-weight bold", 12)).pack(fill=tk.X)

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

        # --- 右下：全局预测曲线图 (包含滚动条和复选框) ---
        plot_container = ttk.Frame(right_paned, padding=5)
        right_paned.add(plot_container, weight=7) 

        # 1. 模型控制区域 (复选框)
        self.model_control_frame = ttk.LabelFrame(plot_container, text="模型显示控制 (Model Visibility)", padding=5)
        self.model_control_frame.pack(fill=tk.X, side=tk.TOP, pady=(0, 5))
        # (复选框将在加载CSV后动态生成)

        ttk.Label(plot_container, text="全序列预测概率 & 累计AUC (Full Sequence Prediction & Cumulative AUC)", font=("-weight bold", 12)).pack(fill=tk.X)

        # 2. 滚动图表区域
        # 使用 Canvas + Scrollbar 实现 matplotlib 图表的滚动
        self.plot_scroll_frame = ttk.Frame(plot_container)
        self.plot_scroll_frame.pack(fill=tk.BOTH, expand=True)

        self.plot_h_scrollbar = ttk.Scrollbar(self.plot_scroll_frame, orient=tk.HORIZONTAL)
        self.plot_h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.plot_canvas_wrapper = tk.Canvas(self.plot_scroll_frame, bg="white", xscrollcommand=self.plot_h_scrollbar.set)
        self.plot_canvas_wrapper.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.plot_h_scrollbar.config(command=self.plot_canvas_wrapper.xview)

        # 初始化 Figure
        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # 将 Figure 放入 FigureCanvasTkAgg
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=self.plot_canvas_wrapper)
        self.chart_widget = self.chart_canvas.get_tk_widget()

        # 将 chart_widget 放入 wrapper canvas 的窗口中
        self.canvas_window_id = self.plot_canvas_wrapper.create_window(0, 0, window=self.chart_widget, anchor="nw")

        # 绑定大小变化事件，确保高度自适应
        self.plot_canvas_wrapper.bind("<Configure>", self._on_canvas_configure)

        self._init_empty_plot()

    def _on_canvas_configure(self, event):
        """当外层 Canvas 大小改变时，更新图表高度以填满"""
        current_width = self.chart_widget.winfo_width()
        # 保持当前宽度，但高度跟随容器
        self.plot_canvas_wrapper.itemconfig(self.canvas_window_id, height=event.height)

    def _init_empty_plot(self):
        self.ax.clear()
        self.ax.text(0.5, 0.5, "Please load data and select a student", ha='center', va='center')
        self.chart_canvas.draw()

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
            
            # --- 生成模型复选框 ---
            for widget in self.model_control_frame.winfo_children():
                widget.destroy()
            
            self.model_vars = {}
            self.model_checkbuttons = {}
            
            for model in self.model_names:
                var = tk.BooleanVar(value=True) # 默认选中
                cb = ttk.Checkbutton(self.model_control_frame, text=model, variable=var, 
                                     command=self.refresh_plot) # 勾选变化时刷新图
                cb.pack(side=tk.LEFT, padx=10)
                self.model_vars[model] = var
                self.model_checkbuttons[model] = cb

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
        """当选中学生时，加载数据并刷新图表"""
        if not self.student_table.selection():
            return

        selected_item = self.student_table.selection()[0]
        self.selected_student_id = int(self.student_table.item(selected_item)['values'][0])

        self.clear_treeview(self.sequence_table)
        
        # 获取学生数据
        student_row = self.df[self.df['student_id'] == self.selected_student_id].iloc[0]

        q_list = str(student_row['questions']).split(',') 
        c_list = str(student_row['concepts']).split(',') 
        trues_list = student_row[f'trues_{self.model_names[0]}']

        min_len = min(len(q_list), len(c_list), len(trues_list))
        q_list = q_list[:min_len]
        c_list = c_list[:min_len]
        trues_list = trues_list[:min_len]

        # 保存到类变量，方便 refresh_plot 使用
        self.current_timesteps = list(range(min_len))
        self.current_trues = trues_list
        self.current_preds_map = {}

        for model in self.model_names:
            p_list = student_row[f'preds_{model}']
            self.current_preds_map[model] = p_list[:min_len]

        # 1. 填充序列详情表 (不受复选框影响，始终显示所有列)
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
            self.current_student_data.append({
                'timestep': i, 'q_id': q_list[i], 'c_id': c_list[i],
                'true': trues_list[i], 'preds': preds_at_t
            })

        # 2. 绘制图表
        self.refresh_plot()

    def refresh_plot(self):
        """根据当前数据和复选框状态重绘图表"""
        """根据当前数据和复选框状态重绘图表"""
        if self.selected_student_id is None:
            return

        self.ax.clear()
        
        # 1. 处理画布尺寸 (横向滚动逻辑)
        
        # --- 修改开始：调整视窗大小逻辑 ---
        
        # 设定你希望在当前视窗(不滚动)内显示多少个点
        points_per_view = 30  # <--- 这里改为 30
        
        # 设定视窗的基础宽度（英寸），这里保持跟 Figure 初始化时一致
        default_width_inches = 10 
        dpi = 100
        
        # 计算每个点应该占多少像素：总像素宽度 / 想要显示的点数
        # 10英寸 * 100dpi = 1000像素。 1000 / 30 ≈ 33.3 像素/点
        pixels_per_step = (default_width_inches * dpi) / points_per_view
        
        points_count = len(self.current_timesteps)
        
        # 计算所需的总宽度（英寸）
        # 逻辑：总点数 * 每点宽度 / DPI
        # 并使用 max 确保如果点很少，图表至少也有 default_width_inches 那么宽
        required_width_inches = max(default_width_inches, (points_count * pixels_per_step) / dpi)
        
        # --- 修改结束 ---
        
        # 设置 Figure 大小
        self.fig.set_size_inches(required_width_inches, 5) # 高度固定 5
        self.chart_canvas.draw()
        
        # 更新 Tkinter Canvas 的 scrollregion
        # 需要更新 chart_widget 在 wrapper canvas 中的大小
        self.plot_canvas_wrapper.itemconfig(self.canvas_window_id, width=required_width_inches*dpi, height=500) # 高度大致估算或自适应
        self.plot_canvas_wrapper.configure(scrollregion=(0, 0, required_width_inches*dpi, 500))

        # 2. 绘制真实作答 (散点) - 始终显示
        true_indices = [i for i, val in enumerate(self.current_trues) if val == 1]
        false_indices = [i for i, val in enumerate(self.current_trues) if val == 0]
        
        self.ax.scatter(true_indices, [1.02]*len(true_indices), color='green', marker='o', s=40, label='True: Correct', zorder=5, clip_on=False)
        self.ax.scatter(false_indices, [-0.02]*len(false_indices), color='red', marker='x', s=40, label='True: Incorrect', zorder=5, clip_on=False)

        # 3. 绘制模型曲线 (根据复选框)
        colors = plt.cm.get_cmap('tab10') 
        
        # 预计算真实标签用于 AUC
        y_true_full = np.array(self.current_trues)

        for idx, model in enumerate(self.model_names):
            if not self.model_vars[model].get(): # 如果未勾选，跳过
                continue

            preds = self.current_preds_map[model]
            safe_preds = [p if pd.notna(p) else 0.5 for p in preds]
            color = colors(idx % 10)
            
            # --- 绘制预测概率曲线 (实线) ---
            self.ax.plot(self.current_timesteps, safe_preds, label=f'{model} Pred', 
                         color=color, marker='.', markersize=4, linewidth=1.5, alpha=0.8)

            # --- 计算并绘制累计 AUC (虚线) ---
            cum_auc_list = []
            for t in range(len(self.current_timesteps)):
                # 取截止当前时刻的数据 (包含当前时刻)
                current_sub_true = y_true_full[:t+1]
                current_sub_pred = safe_preds[:t+1]
                
                # 至少要有两个类别才能计算 AUC
                if len(np.unique(current_sub_true)) < 2:
                    cum_auc_list.append(np.nan)
                else:
                    try:
                        score = roc_auc_score(current_sub_true, current_sub_pred)
                        cum_auc_list.append(score)
                    except:
                        cum_auc_list.append(np.nan)
            
            self.ax.plot(self.current_timesteps, cum_auc_list, label=f'{model} Cum. AUC',
                         color=color, linestyle='--', linewidth=1.5, alpha=0.6)

        # 4. 图表修饰
        self.ax.set_title(f"Student {self.selected_student_id}: Trajectory & Cumulative AUC")
        self.ax.set_xlabel("Time Step")
        self.ax.set_ylabel("Probability / AUC")
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.set_xlim(-0.5, len(self.current_timesteps) - 0.5) # 紧凑显示
        self.ax.set_xticks(self.current_timesteps) # 显示每个时间步的刻度

        # 图例放置
        self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(self.model_names)*2 + 2, fontsize='small')
        self.ax.grid(True, linestyle='--', alpha=0.5)
        
        self.fig.tight_layout()
        # 重新调整边距以适应外部图例
        self.fig.subplots_adjust(top=0.85, bottom=0.15) 

        self.chart_canvas.draw()

    def on_timestep_select(self, event):
        """点击序列详情高亮"""
        if not self.sequence_table.selection():
            return
        
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