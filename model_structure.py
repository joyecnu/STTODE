import os
import sys
sys.path.append(os.getcwd())

def calculate_floats_from_sentence(sentence, float_type='float32'):
    """
    判断句子中是否包含 weight 或 bias，计算对应的浮点数数量。

    参数：
    sentence (str): 输入的句子。
    float_type (str): 浮点数类型，默认为 'float32'，可以是 'float64'。

    返回：
    int: 浮点数数量。
    """
    # 设置浮点数的字节大小
    if float_type == 'float64':
        bytes_per_float = 8  # float64 每个占 8 字节
    else:
        bytes_per_float = 4  # 默认是 float32，每个占 4 字节

    # 判断句子中是否包含 'weight' 或 'bias'
    if 'weight' in sentence:
        print("The sentence contains 'weight'. This is a weight parameter.")
        # 假设我们知道权重的形状和元素数量，比如 64 * 256
        num_floats = 64 * 256  # 示例大小

    elif 'bias' in sentence:
        print("The sentence contains 'bias'. This is a bias parameter.")
        # 假设偏置只有 64 个元素
        num_floats = 64  # 示例大小

    else:
        print("The sentence does not contain 'weight' or 'bias'.")
        return 0,0

    # 计算浮点数的存储大小
    # 计算浮点数的存储大小
    total_size_bytes = num_floats * bytes_per_float
    total_size_mb = total_size_bytes / (1024 * 1024)  # 转换为 MB

    return num_floats, total_size_mb

def model_structure(model):
    '''
    model :输入模型
    input_tensor：输入模型的张量
    '''
    # print('计算参数...','***'*50)
    # blank = ' '
    # print('-' * 120)
    # print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
    #       + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
    #       + ' ' * 10 + 'number' + ' ' * 10 + '|' \
    #       + ' ' * 10 + 'FLOPs (M)' + ' ' * 8 + '|')
    # print('-' * 120)
    #
    # num_params = 0
    # total_flops = 0
    # type_size = 4  # 如果是浮点数，则每个参数占 4 字节
    #
    # # 模拟输入张量
    # input_tensor = torch.randn(input_shape)
    #
    # for index, (key, w_variable) in enumerate(model.named_parameters()):
    #     if len(key) <= 30:
    #         key = key + (30 - len(key)) * blank
    #     shape = str(w_variable.shape)
    #     if len(shape) <= 40:
    #         shape = shape + (40 - len(shape)) * blank
    #     each_para = 1
    #     for k in w_variable.shape:
    #         each_para *= k
    #     num_params += each_para
    #     str_num = str(each_para)
    #     if len(str_num) <= 10:
    #         str_num = str_num + (10 - len(str_num)) * blank
    #
    #     # 根据层类型计算 FLOPs
    #     flops = 0
    #     for layer in model.modules():
    #         if isinstance(layer, torch.nn.Conv2d):
    #             # Conv2D FLOPs
    #             output_size = (
    #                 input_tensor.size(2) // layer.stride[0],
    #                 input_tensor.size(3) // layer.stride[1]
    #             )
    #             flops = (layer.in_channels * layer.out_channels *
    #                      layer.kernel_size[0] * layer.kernel_size[1] *
    #                      output_size[0] * output_size[1]) * 2  # 乘法和加法
    #             input_tensor = torch.randn(1, layer.out_channels, *output_size)  # 更新输入张量形状
    #         elif isinstance(layer, torch.nn.Linear):
    #             # Linear FLOPs
    #             flops = 2 * layer.in_features * layer.out_features
    #
    #     total_flops += flops
    #     flops_in_m = flops / 1e6  # 转换为 M 单位
    #     str_flops = f"{flops_in_m:.2f}"
    #     if len(str_flops) <= 10:
    #         str_flops = str_flops + (10 - len(str_flops)) * blank
    #
    #     print('| {} | {} | {} | {} |'.format(key, shape, str_num, str_flops))
    #
    # print('-' * 120)
    # print('The total number of parameters: ' + str(num_params))
    # print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_params * type_size / 1024 / 1024))
    # print('The total FLOPs of Model {}: {:4f} MFLOPs'.format(model._get_name(), total_flops / 1e6))
    # print('-' * 120)
    #


    # for cnt, batch in enumerate(loader_test):
    #     seq_name = batch.pop()[0]
    #     frame_idx = int(batch.pop()[0])
    #     batch = [tensor[0].cpu() for tensor in batch]
    #     # batch = batch[0]
    #     obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, \
    #     non_linear_ped, valid_ped, obs_loss_mask, pred_loss_mask = batch
    #     with torch.no_grad():
    #         model.set_data(obs_traj, pred_traj_gt, obs_loss_mask, pred_loss_mask)
    #     break
    # # 定义输入张量形状
    # input_size = (1, 1, 1)
    # model.sample_num = 1
    # # 使用 ptflops 计算 FLOPs 和参数量
    # macs, params = get_model_complexity_info(model, input_size, as_strings=True,
    #                                          print_per_layer_stat=True)
    #
    # print(f"FLOPs (MACs): {macs}")
    # print(f"Parameters: {params}")


    # blank = ' '
    # print('-' * 90)
    # print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
    #       + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
    #       + ' ' * 3 + 'number' + ' ' * 3 + '|')
    # print('-' * 90)
    # num_para = 0
    # type_size = 1  # 如果是浮点数就是4
    #
    # for index, (key, w_variable) in enumerate(model.named_parameters()):
    #     if len(key) <= 30:
    #         key = key + (30 - len(key)) * blank
    #     shape = str(w_variable.shape)
    #     if len(shape) <= 40:
    #         shape = shape + (40 - len(shape)) * blank
    #     each_para = 1
    #     for k in w_variable.shape:
    #         each_para *= k
    #     num_para += each_para
    #     str_num = str(each_para)
    #     if len(str_num) <= 10:
    #         str_num = str_num + (10 - len(str_num)) * blank
    #
    #     print('| {} | {} | {} |'.format(key, shape, str_num))
    # print('-' * 90)
    # print('The total number of parameters: ' + str(num_para))
    # print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    # print('-' * 90)


    # 打印模型结构并计算浮点数
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|' \
          + ' ' * 3 + 'float size (MB)' + ' ' * 3 + '|')  # 添加浮点数大小 (MB) 列
    print('-' * 90)

    num_para = 0
    type_size = 4  # 默认为浮点数 float32，4 字节。如果需要使用 float64，将其设置为 8
    Floats = 0
    global total_size_bytes
    total_size_bytes = 0
    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank

        # 计算当前层的参数数量
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para

        # 使用句子来判断类型并计算浮点数
        sentence = key  # 假设权重名就是句子

        num_floats,float_size_mb = calculate_floats_from_sentence(sentence, float_type='float32')  # 计算浮点数
        Floats += float_size_mb
        # 格式化输出每一层的权重、形状、参数数量和浮点数
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank
        str_floats = str(num_floats)
        if len(str_floats) <= 10:
            str_floats = str_floats + (10 - len(str_floats)) * blank
        str_float_size = f"{Floats:.6f}"  # 格式化为小数点后 6 位
        if len(str_float_size) <= 10:
            str_float_size = str_float_size + (10 - len(str_float_size)) * blank

        print('| {} | {} | {} | {} | {} |'.format(key, shape, str_num, str_floats, str_float_size))  # 打印浮点数和大小 (MB)

    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The total number of Floats: {:4f}M'.format(float(str_float_size)))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * 1 / 1000 / 1000))
    print('-' * 90)

# def model_structure(model):
#     blank = ' '
#     print('-' * 90)
#     print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
#           + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
#           + ' ' * 3 + 'number' + ' ' * 3 + '|')
#     print('-' * 90)
#     num_para = 0
#     type_size = 1  # 如果是浮点数就是4
#
#     for index, (key, w_variable) in enumerate(model.named_parameters()):
#         if len(key) <= 30:
#             key = key + (30 - len(key)) * blank
#         shape = str(w_variable.shape)
#         if len(shape) <= 40:
#             shape = shape + (40 - len(shape)) * blank
#         each_para = 1
#         for k in w_variable.shape:
#             each_para *= k
#         num_para += each_para
#         str_num = str(each_para)
#         if len(str_num) <= 10:
#             str_num = str_num + (10 - len(str_num)) * blank
#
#         print('| {} | {} | {} |'.format(key, shape, str_num))
#     print('-' * 90)
#     print('The total number of parameters: ' + str(num_para))
#     print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
#     print('-' * 90)

