
WINDOW_BEFORE = pd.Timedelta("5min")
WINDOW_AFTER  = pd.Timedelta("5min")

# 计算每个开始时间前后的功耗均值，二者相减得到前后变化
def compute_job_power_change(power_series, start_time):
    before = power_series.loc[start_time - WINDOW_BEFORE : start_time].mean()
    after  = power_series.loc[start_time : start_time + WINDOW_AFTER].mean()
    return after - before


# comm_job 为最开始job的属性数据，comm是指选择了和metric共有时间段，因为最初job信息包含了很长时间。筛选好的文件名为：surf_job_info.csv
# df表示metric文件
comm_job['power_delta'] = comm_job['start_date'].apply(
    lambda t: compute_job_power_change(df['nvidia_gpu_power_usage_milliwatts_mean'], t)
)

# 选择变化显著的
threshold = comm_job['power_delta'].quantile(0.90)
significant_jobs = comm_job[comm_job['power_delta'] > threshold]
significant_jobs[['id', 'submit_date', 'start_date', 'end_date', 'power_delta', 'duration_min', 'state', 'nodetypes']]

# nodetype统计
significant_jobs['nodetypes'].value_counts()

# 可视化，验证是否发生了突变
anchor = '2022-08-27 03:17:04'
anchor = pd.to_datetime(anchor)
df.loc[anchor - WINDOW_BEFORE : anchor + WINDOW_AFTER]['nvidia_gpu_power_usage_milliwatts_mean'].plot()