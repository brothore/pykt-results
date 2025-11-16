import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import numpy as np
import ast # 用于将字符串列表转为真实列表
from sklearn.metrics import roc_auc_score, accuracy_score

# 导入matplotlib相关库
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class KTAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("知识追踪 (KT) 坏例分析工具")
        self.root.geometry("1600x900")

        self.df = None
        self.model_names = []
        self.current_student_data = [] # 存储当前选中学生的序列（已解析）

        self.create_widgets()

    def create_widgets(self):
        # --- 顶部菜单 ---
        top_frame = ttk.Frame(self.root, padding=5)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_load = ttk.Button(top_frame, text="加载 CSV 分析文件", command=self.load_csv)
        self.btn_load.pack(side=tk.LEFT)
        self.lbl_file = ttk.Label(top_frame, text="未加载文件")
        self.lbl_file.pack(side=tk.LEFT, padx=10)

        # --- 主内容区 (使用 PanedWindow 分割) ---
        # 左右分割
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # --- 左侧面板：学生列表 ---
        left_frame = ttk.Frame(main_paned, padding=5)
        main_paned.add(left_frame, weight=2) # 占 20% 宽度

        ttk.Label(left_frame, text="学生总览 (点击选择)", font=("-weight bold", 12)).pack(fill=tk.X)
        
        student_cols = ['student_id', 'auc'] # 初始列，加载后会动态添加
        self.student_table = ttk.Treeview(left_frame, columns=student_cols, show='headings', height=25)
        self.student_table.pack(fill=tk.BOTH, expand=True)
        
        # 设置滚动条
        vsb_student = ttk.Scrollbar(self.student_table, orient="vertical", command=self.student_table.yview)
        self.student_table.configure(yscrollcommand=vsb_student.set)
        vsb_student.pack(side='right', fill='y')

        self.student_table.bind("<<TreeviewSelect>>", self.on_student_select)


        # --- 右侧面板 (再上下分割) ---
        right_paned = ttk.PanedWindow(main_paned, orient=tk.VERTICAL)
        main_paned.add(right_paned, weight=8) # 占 80% 宽度

        # --- 右上：序列详情 ---
        seq_frame = ttk.Frame(right_paned, padding=5)
        right_paned.add(seq_frame, weight=4) # 占 40% 高度

        ttk.Label(seq_frame, text="学生序列详情 (点击分析)", font=("-weight bold", 12)).pack(fill=tk.X)
        
        seq_cols = ['timestep', 'q_id', 'c_id', 'true'] # 初始列
        self.sequence_table = ttk.Treeview(seq_frame, columns=seq_cols, show='headings', height=10)
        self.sequence_table.pack(fill=tk.BOTH, expand=True)
        
        vsb_seq = ttk.Scrollbar(self.sequence_table, orient="vertical", command=self.sequence_table.yview)
        self.sequence_table.configure(yscrollcommand=vsb_seq.set)
        vsb_seq.pack(side='right', fill='y')

        self.sequence_table.bind("<<TreeviewSelect>>", self.on_timestep_select)

        # --- 右下：分析区 (指标 + 图表) ---
        analysis_frame = ttk.Frame(right_paned, padding=5)
        right_paned.add(analysis_frame, weight=6) # 占 60% 高度

        # 右下再左右分割
        analysis_paned = ttk.PanedWindow(analysis_frame, orient=tk.HORIZONTAL)
        analysis_paned.pack(fill=tk.BOTH, expand=True)

        # --- 右下左：指标表 ---
        metrics_frame = ttk.Frame(analysis_paned, padding=5)
        analysis_paned.add(metrics_frame, weight=5) # 占 50% 宽度

        ttk.Label(metrics_frame, text="历史累积指标分析", font=("-weight bold", 12)).pack(fill=tk.X)
        
        metrics_cols = ['模型', '分析实体', 'ID', '历史总数', '真实正确率', '预测ACC', '预测AUC']
        self.metrics_table = ttk.Treeview(metrics_frame, columns=metrics_cols, show='headings')
        self.metrics_table.pack(fill=tk.BOTH, expand=True)

        for col in metrics_cols:
            self.metrics_table.heading(col, text=col)
            self.metrics_table.column(col, width=100, anchor='center')
        self.metrics_table.column('模型', width=120)

        # --- 右下右：图表 ---
        plot_frame = ttk.Frame(analysis_paned, padding=5)
        analysis_paned.add(plot_frame, weight=5) # 占 50% 宽度

        ttk.Label(plot_frame, text="历史预测可视化", font=("-weight bold", 12)).pack(fill=tk.X)
        
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax.set_title("请点击序列中的Question/Concept进行分析")
        self.ax.set_xlabel("时间步 (Time Step)")
        self.ax.set_ylabel("预测值 / 真实值")
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
            
            # --- 关键步骤：解析数据 ---
            self.model_names = []
            
            # 动态识别模型
            for col in self.df.columns:
                if col.startswith('trues_'):
                    model_name = col.replace('trues_', '')
                    self.model_names.append(model_name)
            
            if not self.model_names:
                raise Exception("未找到 'trues_' 或 'preds_' 领衔的列")

            # 转换字符串列表为真实列表 (非常重要!)
            # 您的CSV保存的列表是 "[0, 0, 1]" 这种字符串
            for model in self.model_names:
                self.df[f'trues_{model}'] = self.df[f'trues_{model}'].apply(ast.literal_eval)
                self.df[f'preds_{model}'] = self.df[f'preds_{model}'].apply(ast.literal_eval)

            # --- 填充学生总览表 ---
            self.clear_treeview(self.student_table)
            
            # 动态设置学生表的列
            auc_cols = [f'auc_{model}' for model in self.model_names]
            student_cols = ['student_id'] + auc_cols
            self.student_table.config(columns=student_cols)
            
            for col in student_cols:
                self.student_table.heading(col, text=col)
                self.student_table.column(col, width=100, anchor='center')
            self.student_table.column('student_id', width=60)

            # 插入数据
            for _, row in self.df.iterrows():
                values = [row['student_id']] + [f"{row[col]:.4f}" if col in row else 'N/A' for col in auc_cols]
                self.student_table.insert('', tk.END, values=values)
                
            print(f"加载成功! 识别到 {len(self.model_names)} 个模型: {self.model_names}")

        except Exception as e:
            tk.messagebox.showerror("加载失败", f"无法解析CSV文件: {e}\n\n请确保文件是上一步生成的CSV，且包含 `trues_` 和 `preds_` 列。")

    def on_student_select(self, event):
        if not self.student_table.selection():
            return
        
        selected_item = self.student_table.selection()[0]
        selected_student_id = int(self.student_table.item(selected_item)['values'][0])
        
        # 清空旧数据
        self.clear_treeview(self.sequence_table)
        self.clear_treeview(self.metrics_table)
        self.ax.clear()
        self.ax.set_title("请点击序列中的Question/Concept进行分析")
        self.plot_canvas.draw()

        # 获取学生数据行
        student_row = self.df[self.df['student_id'] == selected_student_id].iloc[0]
        
        # --- 关键步骤：解析序列 ---
        q_list = student_row['questions'].split(',')
        c_list = student_row['concepts'].split(',')
        
        # (我们假设所有模型的 trues 列表都一样，取第一个)
        trues_list = student_row[f'trues_{self.model_names[0]}']
        
        # 检查数据一致性
        if not (len(q_list) == len(c_list) == len(trues_list)):
            print(f"警告: 学生 {selected_student_id} 数据长度不一致! Qs:{len(q_list)}, Cs:{len(c_list)}, Trues:{len(trues_list)}")
            # 尝试使用最短的长度
            min_len = min(len(q_list), len(c_list), len(trues_list))
            q_list = q_list[:min_len]
            c_list = c_list[:min_len]
            trues_list = trues_list[:min_len]
        
        # 动态设置序列表的列
        pred_cols = [f'pred_{model}' for model in self.model_names]
        seq_cols = ['timestep', 'q_id', 'c_id', 'true'] + pred_cols
        self.sequence_table.config(columns=seq_cols)
        
        for col in seq_cols:
            self.sequence_table.heading(col, text=col)
            self.sequence_table.column(col, width=70, anchor='center')
        
        # 准备数据以供后续点击分析
        self.current_student_data = []
        
        # 填充序列详情表
        for i in range(len(q_list)):
            preds_map = {}
            row_values = [i, q_list[i], c_list[i], trues_list[i]]
            
            for model in self.model_names:
                pred_val = student_row[f'preds_{model}'][i]
                preds_map[model] = pred_val
                row_values.append(f"{pred_val:.3f}")
            
            self.sequence_table.insert('', tk.END, values=row_values, iid=str(i))
            
            # 存储解析后的数据
            self.current_student_data.append({
                'timestep': i,
                'q_id': q_list[i],
                'c_id': c_list[i],
                'true': trues_list[i],
                'preds': preds_map
            })

    def on_timestep_select(self, event):
        if not self.sequence_table.selection() or not self.current_student_data:
            return

        selected_iid = self.sequence_table.selection()[0]
        clicked_timestep = int(selected_iid)
        
        clicked_data = self.current_student_data[clicked_timestep]
        clicked_q_id = clicked_data['q_id']
        clicked_c_id = clicked_data['c_id']

        # --- 1. 计算指标 ---
        # 清空指标表
        self.clear_treeview(self.metrics_table)

        # 分析 Question
        q_metrics_list, q_plot_data = self.calculate_history_metrics('Question', clicked_q_id, clicked_timestep)
        
        # 分析 Concept
        c_metrics_list, c_plot_data = self.calculate_history_metrics('Concept', clicked_c_id, clicked_timestep)
        
        # 填充指标表
        for metrics in (q_metrics_list + c_metrics_list):
            if metrics:
                self.metrics_table.insert('', tk.END, values=[
                    metrics['model'], metrics['type'], metrics['id'], 
                    metrics['count'], metrics['true_acc'], 
                    metrics['pred_acc'], metrics['auc']
                ])
                
        # --- 2. 更新图表 (我们优先绘制 Question 的) ---
        self.update_plot(q_plot_data, f"Question ID: {clicked_q_id} (截至时间步 {clicked_timestep})")


    def calculate_history_metrics(self, entity_type, entity_id, max_timestep):
        """
        核心分析函数：计算指定实体(Q或C)在指定时间步之前的累积指标
        """
        history_trues = []
        history_preds = {model: [] for model in self.model_names}
        history_timesteps = [] # 用于绘图X轴

        # 遍历当前学生的所有历史数据
        for item in self.current_student_data:
            if item['timestep'] > max_timestep:
                break # 只分析到当前点击的时间步
            
            current_entity_id = item['q_id'] if entity_type == 'Question' else item['c_id']
            
            # 如果是我们要找的实体
            if current_entity_id == entity_id:
                history_trues.append(item['true'])
                history_timesteps.append(item['timestep'])
                for model in self.model_names:
                    history_preds[model].append(item['preds'][model])

        # --- 开始计算指标 ---
        final_metrics_list = []
        
        # 如果历史记录太少，无法计算AUC
        if len(history_trues) < 2 or len(set(history_trues)) < 2:
             # 即使无法计算AUC，也返回基本信息
            for model in self.model_names:
                final_metrics_list.append({
                    'model': model, 'type': entity_type, 'id': entity_id,
                    'count': len(history_trues),
                    'true_acc': f"{(sum(history_trues) / len(history_trues)):.3f}" if len(history_trues) > 0 else "N/A",
                    'pred_acc': "N/A (No Preds)", 'auc': "N/A (Need >1 Class)"
                })
            plot_data = {'timesteps': history_timesteps, 'trues': history_trues, 'preds': history_preds}
            return final_metrics_list, plot_data


        # 对于每个模型，计算累积指标
        for model in self.model_names:
            model_preds = history_preds[model]
            
            # 1. 累计预测AUC
            try:
                auc = roc_auc_score(history_trues, model_preds)
                auc_str = f"{auc:.3f}"
            except ValueError:
                auc_str = "N/A"
            
            # 2. 累计预测ACC (使用0.5阈值)
            pred_labels = [1 if p > 0.5 else 0 for p in model_preds]
            acc = accuracy_score(history_trues, pred_labels)
            
            # 3. 累计准确率 (真实)
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
            
        plot_data = {
            'timesteps': history_timesteps,
            'trues': history_trues,
            'preds': history_preds # 包含所有模型的预测
        }

        return final_metrics_list, plot_data

    def update_plot(self, plot_data, title):
        self.ax.clear()
        
        if not plot_data or not plot_data['timesteps']:
            self.ax.set_title("无历史数据可供绘制")
            self.plot_canvas.draw()
            return

        timesteps = plot_data['timesteps']
        trues = plot_data['trues']
        
        # 1. 绘制真实答案 (使用散点)
        # 将 0 映射到 0.05, 1 映射到 0.95，防止与坐标轴重叠
        true_plot_vals = [0.95 if t == 1 else 0.05 for t in trues]
        self.ax.scatter(timesteps, true_plot_vals, label="True Answer (0=False, 1=True)", 
                        color='black', marker='x', s=50) # s=50 增大标记

        # 2. 绘制每个模型的预测 (使用折线)
        for model in self.model_names:
            preds = plot_data['preds'][model]
            if preds:
                self.ax.plot(timesteps, preds, label=f"Pred ({model})", marker='.')

        self.ax.set_title(title)
        self.ax.set_xlabel("序列中的时间步 (Time Step)")
        self.ax.set_ylabel("预测概率 / 真实答案")
        self.ax.set_ylim(-0.1, 1.1) # Y轴固定在 0-1
        self.ax.legend(loc='best', fontsize='small')
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        self.plot_canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = KTAnalysisGUI(root)
    root.mainloop()