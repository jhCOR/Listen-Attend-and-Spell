def main():
    train_dataset = SpeechDataset(char2id, split=CFG.dataset_list[0], specaugment=True, max_len=0)
    test_dataset = SpeechDataset(char2id, split="test-clean", max_len=0)
    train_dataloader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, pin_memory=True,
                                collate_fn=_collate_fn, num_workers=CFG.worker)
    test_dataloader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=True, pin_memory=True,
                                collate_fn=_collate_fn, num_workers=CFG.worker)
    
    listener = Listener(CFG.num_channels, 256)
    speller = Speller(len(id2char), 512, num_heads=4, dropout=0.3)
    model = LAS(listener, speller).cuda()

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index = PAD_TOKEN).cuda()


if __name__ == '__main__':
    main()
