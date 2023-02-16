with torch.no_grad():
    input_data, ans_data = next(iter(train_dataloader))
    ans_data = ans_data.cuda()
    print(input_data.shape)
    print(ans_data.shape)
    print(label_to_string(ans_data, id2char))
    
    result_logit = model(input_data.cuda(), ground_truth=ans_data, teacher_forcing_rate=0.6)

    result_logit = result_logit[:,:ans_data.size(1),:].contiguous()
    ans_data = ans_data[:,:result_logit.size(1)].contiguous()

    loss_check = criterion(result_logit.view(-1, result_logit.size(-1)), ans_data.view(-1))
    print(loss_check)
    y_pred = torch.max(result_logit, dim=-1)[1]
    print(label_to_string(y_pred, id2char))