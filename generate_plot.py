import os
import pandas as pd
import numpy as np
import argparse
import re
import plotly.express as px
import plotly.graph_objects as go

parser = argparse.ArgumentParser()
parser.add_argument("--sharp_dir",
                    type=str,
                   default="./out_report_post/batch_4096_lr_0.01_epochs_1000_freewarm_40_freelambda_4096.0/")
parser.add_argument("--bpr_dir",
                    type=str,
                   default="./out_report_post/batch_4096_lr_0.01_epochs_1000_freewarm_1000_freelambda_4096.0/")
parser.add_argument("--slow_dir",
                    type=str,
                   default="./out_slow_report_post/batch_4096_lr_0.01_epochs_1000_freewarm_40_freelambda_4096.0/")

parser.add_argument("--out_dir",
                    type=str,
                   default="./out_plot")


def main(args):
	df_stat_sharp = pd.read_pickle(os.path.join(args.sharp_dir, "stat.pkl"))
	df_stat_bpr = pd.read_pickle(os.path.join(args.bpr_dir, "stat.pkl"))
	df_stat_slow = pd.read_pickle(os.path.join(args.slow_dir, "stat.pkl"))
	
	li_cols_time = ["01.train()", "02.neg_sample", "03.get_batch", "04.zero_grad", "05.backward", "05.forward", "06.step", "07.eval()",	"08.hr_ndcg"]
	s_fast_mean = df_stat_sharp[li_cols_time].mean(axis=0)
	s_slow_mean = df_stat_slow[li_cols_time].mean(axis=0)
	li_state = [ "slow(origin)", "fast(optimized)"]
	li_data = [ s_slow_mean, s_fast_mean,]
	li_task = list()
	li_time = list()
	li_s = list()

	# Plot Entire Bar Plot
	for series, s in zip(li_data, li_state):
	    for idx, seconds in series.items():
	        li_task.append(idx)
	        li_time.append(float(seconds))
	        li_s.append(s)
	df_time_mean = pd.DataFrame(dict(task=li_task, time_seconds=li_time, state=li_s))
	df_time_mean["warm"] = df_time_mean["state"].apply(lambda x: x.split("_")[-1])
	df_time_mean["optimization"] = df_time_mean["state"].apply(lambda x: x.split("_")[0])
	
	_df_time = df_time_mean.rename(columns = {"time_seconds": "Elapsed Time(s)"})
	fig = px.bar(_df_time, x="state", y="Elapsed Time(s)", color="task", title="Running(Train,Eval) Time Improvements (Pre/Post Processing w/ GPU)", orientation="v", text="Elapsed Time(s)")
	fig.update_layout(width=1000, height=700,yaxis={'categoryorder':'total ascending'} )
	fig.update_traces(texttemplate='%{text:.3s}', textposition='outside')
	os.makedirs(args.out_dir, exist_ok=True)
	fig.write_html(os.path.join(args.out_dir, "entire.html"))
	#fig.show()
	
	# Show Latency Breakdown Tables
	_df_report = _df_time.pivot(index="optimization", columns="task", values="Elapsed Time(s)").sort_index(ascending=False)
	_df_report["total"] = _df_report.sum(axis=1)
	_df_report.loc["improvements(fast over slow)"] = _df_report.loc["slow(origin)"] / _df_report.loc["fast(optimized)"]
	_li_paper = [None] * (len(_df_report.columns))
	_li_paper[-1] = 17.1
	_df_report.loc["paper"] = _li_paper
	_li_paper = [None] * (len(_df_report.columns))
	_li_paper[-1] = 17.1 / _df_report.loc["fast(optimized)", "total"]
	_df_report.loc["improvements(fast over paper)"] = _li_paper
	_df_report.index.name = ""
	print(_df_report)

	# Plot Latency Break Down
	_df_time = df_time_mean.rename(columns = {"time_seconds": "Elapsed Time(s)"})
	_df_time = _df_time.sort_values(by="state", ascending=False).sort_values(by="task", kind="stable")
	x = [
	    _df_time["task"].tolist(),
	    _df_time["state"].tolist(),
	]
	li_colors = ["lightslategray", "crimson"] * (len(_df_time) // 2)
	fig = go.Figure(go.Bar(x=x,y=_df_time["Elapsed Time(s)"], name="Elapsed Time(s)", marker_color=li_colors, textposition='outside'))
	
	fig.update_xaxes(ticks="", tickfont=dict(size=10))
	fig.update_layout(width=1000, height=700, yaxis_title="Elapsed Time(s)",)
	fig.update_layout(title="Running(Train,Eval) Time Improvements (Pre nad Pos Processing w/ GPU)")
	fig.update_traces(texttemplate='%{y:.3s}', textposition='outside')
	fig.write_html(os.path.join(args.out_dir, "latency_breakdown.html"))
	#fig.show()


	df_stat_sharp["method"] = "SharpCF"
	df_stat_bpr["method"] = "BPR"
	df_to_plot = pd.concat([ df_stat_bpr, df_stat_sharp], axis=0)
	
	#df_stat_bpr.head()
	fig = px.line(df_to_plot, x="epoch", y="ndcg", color="method")
	fig.update_layout(title="nDCG over Epochs (BPR vs SharpCF)")
	fig.update_layout(width=1000, height=700)
	fig.write_html(os.path.join(args.out_dir, "ndcg.html"))
	#fig.show()


if __name__ == "__main__":
	args = parser.parse_args()
	main(args)
