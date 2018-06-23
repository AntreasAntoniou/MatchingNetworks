import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Welcome to the DAGAN training and inference system')
    parser.add_argument('--batch_size', nargs="?", type=int, default=32, help='Batch_size for experiment')
    parser.add_argument('--experiment_title', nargs="?", type=str, default="experiment-title", help='Experiment name')
    parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='Continue from checkpoint of epoch')
    parser.add_argument('--dropout_rate_value', type=float, default=0.3, help='Dropout_rate_value')
    parser.add_argument('--total_epochs', type=int, default=200, help='Number of epochs per experiment')
    parser.add_argument('--total_iter_per_epoch', type=int, default=1000, help='Number of iters per epoch')
    parser.add_argument('--full_context_unroll_k', type=int, default=5, help='Unroll levels K for attLSTM, used only'
                                                                             ' when use_full_context_embeddings is set'
                                                                             ' to True')
    parser.add_argument('--classes_per_set', type=int, default=5, help='Number of classes to sample per set')
    parser.add_argument('--samples_per_class', type=int, default=5, help='Number of samples per set to sample')
    parser.add_argument('--use_full_context_embeddings', type=str, default="False", help="Whether to use full context "
                                                                                         "embeddings (i.e. bidirLSTM "
                                                                                         "for g_embed and attLSTM for "
                                                                                         "f_embed)")
    parser.add_argument('--use_mean_per_class_embeddings', type=str, default="False", help="Whether to take the mean of"
                                                                                           "the CNN embeddings "
                                                                                           "classwise (i.e. produce one"
                                                                                           " embedding per class, "
                                                                                           "similar to prototypical"
                                                                                           " networks paper)")

    args = parser.parse_args()
    args.use_full_context_embeddings = True if args.use_full_context_embeddings=="True" else False
    args.use_mean_per_class_embeddings = True if args.use_mean_per_class_embeddings == "True" else False

    return args