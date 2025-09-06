from data_provider.timeseries_data_loader import  Dataset_common,Dataset_Electricity,Dataset_common2 #,Dataset_Custom
from torch.utils.data import DataLoader

data_dict = {
    # 'ETTh1': Dataset_ETT_hour,
    # 'ETTh2': Dataset_ETT_hour,
    # 'ETTm1': Dataset_ETT_minute,
    # 'ETTm2': Dataset_ETT_minute,
    'Electricity_shanxi': Dataset_Electricity,
    'Electricity_shandong': Dataset_Electricity,
    'ETTh1': Dataset_common,
    'ETTh2': Dataset_common,
    'ETTm1': Dataset_common,
    'ETTm2': Dataset_common,
    # 'custom': Dataset_Custom,
    'weather': Dataset_common,
    'illness': Dataset_common,
    'traffic': Dataset_common,
    'exchange_rate': Dataset_common,
    'wind': Dataset_common,
    'UTS_Dataset': Dataset_common,
}

# # # TMDM
# def data_provider(args, flag):
#     Data = data_dict[args.data]
#     timeenc = 0 if args.embed != 'timeF' else 1

#     if flag == 'test':
#         shuffle_flag = False
#         drop_last = True
#         batch_size = args.test_batch_size
#         freq = args.freq
#     elif flag == 'pred':
#         shuffle_flag = False
#         drop_last = False
#         batch_size = 1
#         freq = args.freq
#         Data = Dataset_Pred
#     else:
#         shuffle_flag = True
#         drop_last = True
#         batch_size = args.batch_size
#         freq = args.freq

#     data_set = Data(
#         root_path=args.root_path,
#         data_path=args.data_path,
#         flag=flag,
#         size=[args.past_len, args.label_len, args.pred_len],
#         features=args.features,
#         target=args.target,
#         timeenc=timeenc,
#         freq=freq
#     )
#     print(flag, len(data_set))
#     data_loader = DataLoader(
#         data_set,
#         batch_size=batch_size,
#         shuffle=shuffle_flag,
#         num_workers=args.num_workers,
#         drop_last=drop_last)
#     return data_set, data_loader

def data_provider(args, flag):
    Data = data_dict[args.data]
    if args.data=='wind':
        target='target'
    else:
        target='OT'
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size

    freq = args.freq

    if args.task_name == 'anomaly_detection':
        return None
    else:
        data_set = Data(
            args = args,
            flag=flag,
            size=[args.past_len, args.label_len, args.pred_len],
            features=args.features,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        # print(flag, len(data_set))
        if flag=='test' and args.model_name=='tmdm':
            test_batch_size = args.test_batch_size
            data_loader = DataLoader(
                data_set,
                batch_size=test_batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last)
        else:
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last)
        return data_set, data_loader