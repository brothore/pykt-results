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
        self.current_student_data = []  # 存储当前选中学生的序列（已解析）

        # --- 新增：用于存储Q/C切换和点击状态 ---
        self.current_clicked_timestep_data = None  # 存储当前点击的q_id, c_id, timestep
        self.analysis_mode = "Question"  # 默认分析模式

        # --- 新增：用于排序状态 ---
        self.student_table_headings = {} # 存储原始标题
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
        main_paned.add(left_frame, weight=3) # 调整了权重

        ttk.Label(left_frame, text="学生总览 (Student Overview)", font=("-weight bold", 12)).pack(fill=tk.X)

        student_cols = ['student_id', 'auc']
        self.student_table = ttk.Treeview(left_frame, columns=student_cols, show='headings', height=25)
        self.student_table.pack(fill=tk.BOTH, expand=True)

        vsb_student = ttk.Scrollbar(self.student_table, orient="vertical", command=self.student_table.yview)
        self.student_table.configure(yscrollcommand=vsb_student.set)
        vsb_student.pack(side='right', fill='y')

        # --- 修改点: 排序功能在 load_csv 中动态绑定 ---

        self.student_table.bind("<<TreeviewSelect>>", self.on_student_select)


        # --- 右侧面板 (再上下分割) ---
        right_paned = ttk.PanedWindow(main_paned, orient=tk.VERTICAL)
        main_paned.add(right_paned, weight=7) # 调整了权重

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
        
        # 水平滚动条 (如你所愿，保持“横板”布局)
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
        self.btn_toggle_analysis.config(state=tk.DISABLED)  # 初始禁用
        
        metrics_cols = ['模型', '分析实体', 'ID', '历史总数', '真实正确率', '预测ACC', '预测AUC']
        self.metrics_table = ttk.Treeview(metrics_frame, columns=metrics_cols, show='headings')
        self.metrics_table.pack(fill=tk.BOTH, expand=True)

        for col in metrics_cols:
            self.metrics_table.heading(col, text=col)
            self.metrics_table.column(col, width=100, anchor='center')
        self.metrics_table.column('模型', width=120)

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
                self.df[f'trues_{model}'] = self.df[f'trues_{model}'].apply(ast.literal_eval)
                self.df[f'preds_{model}'] = self.df[f'preds_{model}'].apply(ast.literal_eval)

            # --- 填充学生总览表 (修改点 1) ---
            self.clear_treeview(self.student_table)
            
            auc_cols = [f'auc_{model}' for model in self.model_names]
            
            # --- 新增：计算差值列 ---
            delta_auc_cols = [f'delta_{model}' for model in self.model_names[1:]]
            
            # --- 新增：序列长度列 和 差值列 ---
            student_cols = ['student_id', 'seq_len'] + auc_cols + delta_auc_cols
            self.student_table.config(columns=student_cols)
            
            self.student_table_headings = {} # 重置标题

            for col in student_cols:
                col_name = col
                if col.startswith('delta_'):
                    col_name = f"Δ_auc_{col.replace('delta_', '')}"
                
                self.student_table_headings[col] = col_name # 存储原始名称
                
                self.student_table.heading(col, text=col_name, 
                                           command=lambda c=col: self.sort_by_column(self.student_table, c))
                
                # 设置列宽
                width = 80
                if col == 'student_id':
                    width = 60
                elif col == 'seq_len':
                    width = 50
                elif col.startswith('auc_'):
                    width = 90
                self.student_table.column(col, width=width, anchor='center')

            # 重置排序状态
            self.last_sort_col = None
            self.last_sort_reverse = False

            for _, row in self.df.iterrows():
                # --- 新增：计算 seq_len 和 delta_aucs ---
                seq_len = len(row[f'trues_{base_model}'])
                base_auc = row[f'auc_{base_model}']
                
                auc_values = [f"{row[col]:.4f}" if col in row else 'N/A' for col in auc_cols]
                
                delta_auc_values = []
                for model in self.model_names[1:]:
                    try:
                        delta = row[f'auc_{model}'] - base_auc
                        delta_auc_values.append(f"{delta:+.4f}") # 带符号
                    except:
                        delta_auc_values.append("N/A")

                values = [row['student_id'], seq_len] + auc_values + delta_auc_values
                self.student_table.insert('', tk.END, values=values)

            print(f"加载成功! 识别到 {len(self.model_names)} 个模型: {self.model_names}")

        except Exception as e:
            messagebox.showerror("加载失败 (Load Failed)", f"无法解析CSV文件 (Failed to parse CSV): {e}\n\n请确保文件是上一步生成的CSV，且包含 `trues_` 和 `preds_` 列。")

    # --- 新增：排序功能 (修改点 1) ---
    def sort_by_column(self, tree, col):
        """点击列标题时对 Treeview 进行排序"""
        
        # 判断排序方向
        if col == self.last_sort_col:
            reverse = not self.last_sort_reverse
        else:
            reverse = False

        # 尝试将值转为浮点数排序
        try:
            l = [(float(tree.set(k, col)), k) for k in tree.get_children('')]
        except (ValueError, TypeError):
            # 如果失败，则按字符串排序
            l = [(tree.set(k, col), k) for k in tree.get_children('')]

        l.sort(key=lambda t: t[0], reverse=reverse)

        # 重新排列
        for index, (val, k) in enumerate(l):
            tree.move(k, '', index)

        # --- 更新标题箭头 ---
        # 移除所有旧箭头
        for c in self.student_table_headings:
            tree.heading(c, text=self.student_table_headings[c]) 
            
        # 添加新箭头
        new_heading = self.student_table_headings[col] + (' ▼' if reverse else ' ▲')
        tree.heading(col, text=new_heading)

        # 保存状态
        self.last_sort_col = col
        self.last_sort_reverse = reverse

    def on_student_select(self, event):
        if not self.student_table.selection():
            return

        selected_item = self.student_table.selection()[0]
        selected_student_id = int(self.student_table.item(selected_item)['values'][0])

        # 清空旧数据
        self.clear_treeview(self.sequence_table)
        self.clear_treeview(self.metrics_table)
        self.ax.clear()

        self.ax.set_title("Please select a timestep to analyze")
        self.ax.set_xlabel("Time Step")
        self.ax.set_ylabel("Prediction / True Value")
        self.plot_canvas.draw()

        self.current_clicked_timestep_data = None
        self.btn_toggle_analysis.config(state=tk.DISABLED, text="切换到 Concept 分析")
        self.analysis_mode = "Question"

        student_row = self.df[self.df['student_id'] == selected_student_id].iloc[0]

        q_list = str(student_row['questions']).split(',') # 增加str()确保安全
        c_list = str(student_row['concepts']).split(',') # 增加str()确保安全

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

    # 重构 on_timestep_select
    def on_timestep_select(self, event):
        if not self.sequence_table.selection() or not self.current_student_data:
            return

        selected_iid = self.sequence_table.selection()[0]
        clicked_timestep = int(selected_iid)

        clicked_data = self.current_student_data[clicked_timestep]

        # 存储当前点击的数据，供切换按钮使用
        self.current_clicked_timestep_data = {
            'q_id': clicked_data['q_id'],
            'c_id': clicked_data['c_id'],
            'timestep': clicked_timestep
        }

        # 启用切换按钮
        self.btn_toggle_analysis.config(state=tk.NORMAL)

        # 默认以 "Question" 模式运行分析
        self.analysis_mode = "Question"
        self.btn_toggle_analysis.config(text="切换到 Concept 分析")

        # 运行分析
        self.run_analysis_and_plot()

    # 新增函数: 切换分析模式
    def toggle_analysis_mode(self):
        if self.analysis_mode == "Question":
            self.analysis_mode = "Concept"
            self.btn_toggle_analysis.config(text="切换到 Question 分析")
        else:
            self.analysis_mode = "Question"
            self.btn_toggle_analysis.config(text="切换到 Concept 分析")

        # 重新运行分析
        self.run_analysis_and_plot()

    # 新增函数: 运行分析与绘图
    def run_analysis_and_plot(self):
        if not self.current_clicked_timestep_data:
            return

        # 清空指标表
        self.clear_treeview(self.metrics_table)

        q_id = self.current_clicked_timestep_data['q_id']
        c_id = self.current_clicked_timestep_data['c_id']
        max_timestep = self.current_clicked_timestep_data['timestep']

        if self.analysis_mode == "Question":
            metrics_list, plot_data = self.calculate_history_metrics('Question', q_id, max_timestep)
            title = f"Question ID: {q_id} (Up to timestep {max_timestep})"

        else:  # self.analysis_mode == "Concept"
            metrics_list, plot_data = self.calculate_history_metrics('Concept', c_id, max_timestep)
            title = f"Concept ID: {c_id} (Up to timestep {max_timestep})"

        # 填充指标表
        for metrics in metrics_list:
            if metrics:
                self.metrics_table.insert('', tk.END, values=[
                    metrics['model'], metrics['type'], metrics['id'],
                    metrics['count'], metrics['true_acc'],
                    metrics['pred_acc'], metrics['auc']
                ])

        # 更新图表
        self.update_plot(plot_data, title)

    def calculate_history_metrics(self, entity_type, entity_id, max_timestep):
        """
        核心分析函数：计算指定实体(Q或C)在指定时间步之前的累积指标
        """
        history_trues = []
        history_preds = {model: [] for model in self.model_names}
        history_timesteps = []  # 用于绘图X轴

        for item in self.current_student_data:
            if item['timestep'] > max_timestep:
                break

            current_entity_id = item['q_id'] if entity_type == 'Question' else item['c_id']

            if current_entity_id == entity_id:
                history_trues.append(item['true'])
                history_timesteps.append(item['timestep'])
                for model in self.model_names:
                    history_preds[model].append(item['preds'][model])
        
        # --- 修改点 3: 计算累计准确率 ---
        cumulative_accuracy = []
        if history_trues:
            cumulative_trues = np.cumsum(history_trues)
            cumulative_counts = np.arange(1, len(history_trues) + 1)
            cumulative_accuracy = (cumulative_trues / cumulative_counts).tolist()
        # -------------------------------

        final_metrics_list = []
        
        plot_data = {
            'timesteps': history_timesteps,
            'trues': history_trues,
            'preds': history_preds,
            'cumulative_accuracy': cumulative_accuracy # 新增
        }

        if len(history_trues) < 2 or len(set(history_trues)) < 2:
            for model in self.model_names:
                final_metrics_list.append({
                    'model': model, 'type': entity_type, 'id': entity_id,
                    'count': len(history_trues),
                    'true_acc': f"{(sum(history_trues) / len(history_trues)):.3f}" if len(history_trues) > 0 else "N/A",
                    'pred_acc': "N/A (No Preds)", 'auc': "N/A (Need >1 Class)"
                })
            return final_metrics_list, plot_data

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

        return final_metrics_list, plot_data

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
        
        # --- 修改点 3: 获取累计准确率 ---
        cumulative_accuracy = plot_data.get('cumulative_accuracy')

        true_plot_vals = [0.95 if t == 1 else 0.05 for t in trues]
        self.ax.scatter(timesteps, true_plot_vals, label="True Answer (0=False, 1=True)",
                        color='black', marker='x', s=50, zorder=3) # zorder提高图层

        for model in self.model_names:
            preds = plot_data['preds'][model]
            if preds:
                self.ax.plot(timesteps, preds, label=f"Pred ({model})", marker='.', zorder=2)

        # --- 修改点 3: 绘制累计准确率 ---
        if cumulative_accuracy:
            self.ax.plot(timesteps, cumulative_accuracy, 
                         label="Cumulative True Acc", 
                         color='green', linestyle='--', marker='o', 
                         markersize=3, zorder=1)
        # -----------------------------

        self.ax.set_title(title)
        self.ax.set_xlabel("Time Step")
        self.ax.set_ylabel("Prediction Probability / True Answer")
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.legend(loc='best', fontsize='small')
        self.ax.grid(True, linestyle='--', alpha=0.6)

        self.plot_canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = KTAnalysisGUI(root)
    root.mainloop()