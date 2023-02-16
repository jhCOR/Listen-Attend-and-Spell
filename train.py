from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
#scheduler = ReduceLROnPlateau(optimizer, 'min')

epochs = 1
checkpoint = 1
print_step = 200
save_epoch = 1
checkpoint_path = './'

loss_history =[]
for epoch in range(epochs):
    total_loss = 0.
    num_samples = 0
    cer = 0.
    wer = 0.
    step= 0
    if epoch==5:
        set_lr(optimizer, 1e-5)
    total_step = len(train_dataloader)
    for batch in train_dataloader:
        model.train()
        optimizer.zero_grad()

        x, y = batch
        x = x.cuda()
        y = y.cuda()
        batch_size = x.size(0)
        
        target = y[:, :].contiguous().cuda()
        teacher_forcing_rate = scheduler_sampling(epoch)
        
        logits = model(x, ground_truth=y, teacher_forcing_rate=teacher_forcing_rate)
        
        if logits.dim() == 2:
            logits = logits.T
            logits = logits.unsqueeze(0)
            
        y_hats = torch.max(logits, dim=-1)[1]
        loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))

        total_loss += loss.item()
        num_samples += batch_size

        loss.backward()
        loss_history.append(loss.item())
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=400)
        optimizer.step()
        
        cer_, wer_ = score(y_hats.long(), target)
        cer += cer_
        wer += wer_
        if step%print_step==0:
            print('timestep: {:4d}/{:4d}, loss: {:.4f}, cer: {:.2f}, wer: {:.2f}, tf_rate: {:.2f}'.format(
                step, total_step, total_loss/num_samples, cer/num_samples, wer/num_samples, teacher_forcing_rate))
            with open('aihub-4.log', 'at') as f:
                f.write('timestep: {:4d}/{:4d}, loss: {:.4f}, cer: {:.2f}, wer: {:.2f}, tf_rate: {:.2f}\n'.format(
                step, total_step, total_loss/num_samples, cer/num_samples, wer/num_samples, teacher_forcing_rate))
        step += 1
        
    total_loss /= num_samples
    cer /= num_samples
    wer /= num_samples
    print('Epoch %d (Training) Total Loss %0.4f CER %0.4f WER %0.4f' % (epoch, total_loss, cer, wer))
    with open('las.log', 'at') as f:
        f.write('Epoch %d (Training) Total Loss %0.4f CER %0.4f WER %0.4f\n' % (epoch, total_loss, cer, wer))

    total_loss = 0.
    num_samples = 0
    cer = 0.
    wer = 0.
    with torch.no_grad():
        model.eval()
        for batch in tqdm(test_dataloader):
            x, y = batch

            batch_size = x.size(0)
            
            x = x.cuda()
            y = y.cuda()
            target = y[:, :].contiguous().cuda()
            
            logits = model(x, ground_truth=None, teacher_forcing_rate=0.0)
            if logits.dim() == 2:
                logits = logits.T
                logits = logits.unsqueeze(0)
            y_hats = torch.max(logits, dim=-1)[1]
            
            logits = logits[:,:target.size(1),:].contiguous() # cut over length to calculate loss
            target = target[:,:logits.size(1)].contiguous()

            loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))

            total_loss += loss.item()

            cer_, wer_ = score(y_hats.long(), target)
            cer += cer_
            wer += wer_
            num_samples += batch_size

    val_loss = total_loss/num_samples
    
    #scheduler.step(val_loss)
    val_loss = total_loss/num_samples
    cer /= num_samples
    wer /= num_samples
    #scheduler.step(val_loss)
    print('Epoch %d (Evaluate) Total Loss %0.4f CER %0.4f WER %0.4f' % (epoch, val_loss, cer, wer))
    with open('las.log', 'at') as f:
        f.write('Epoch %d (Evaluate) Total Loss %0.4f CER %0.4f WER %0.4f\n' % (epoch, val_loss, cer, wer))
    last_checkpoint(checkpoint_path+'/', epoch, model, optimizer, loss)
    if epoch%save_epoch==0:
        torch.save(model, "{}/epoch{}-cer{:.2f}-wer{:.2f}.pt".format(checkpoint_path, epoch, cer, wer))