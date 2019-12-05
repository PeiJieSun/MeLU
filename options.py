config = {
    # item & user
    'num_item': 3706,
    'num_user': 6040,
    # network parameter
    'embedding_dim': 32,
    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 64,
    # cuda setting
    'use_cuda': True,
    # model setting
    'inner': 1,
    'lr': 5e-5,
    'local_lr': 5e-6,
    'batch_size': 32,
    'num_epoch': 20,
    # candidate selection
    'num_candidate': 20,
}

states = ["warm_state", "user_cold_state", "item_cold_state", "user_and_item_cold_state"]
