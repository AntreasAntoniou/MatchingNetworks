from one_shot_learning_network import *
from experiment_builder import ExperimentBuilder
import tensorflow.contrib.slim as slim
import data as dataset
import tqdm

from utils.parser_utils import get_args
from utils.storage import save_statistics, build_experiment_folder

tf.reset_default_graph()
args = get_args()
# Experiment builder
data = dataset.FolderDatasetLoader(num_of_gpus=1, batch_size=args.batch_size, image_height=28, image_width=28,
                                   image_channels=1,
                                   train_val_test_split=(1200/1622, 211/1622, 211/162),
                                   samples_per_iter=1, num_workers=4,
                                   data_path="datasets/omniglot_data", name="omniglot_data",
                                   index_of_folder_indicating_class=-2, reset_stored_filepaths=False,
                                   num_samples_per_class=args.samples_per_class, num_classes_per_set=args.classes_per_set)

experiment = ExperimentBuilder(data)
one_shot_omniglot, losses, c_error_opt_op, init = experiment.build_experiment(args.batch_size,
                                                                              args.classes_per_set,
                                                                              args.samples_per_class,
                                                                              args.use_full_context_embeddings,
                                                                              full_context_unroll_k=
                                                                              args.full_context_unroll_k,
                                                                              args=args)
total_train_batches = args.total_iter_per_epoch
total_val_batches = args.total_iter_per_epoch
total_test_batches = args.total_iter_per_epoch

saved_models_filepath, logs_filepath = build_experiment_folder(args.experiment_title)


save_statistics(logs_filepath, ["epoch", "total_train_c_loss_mean", "total_train_c_loss_std",
                                  "total_train_accuracy_mean", "total_train_accuracy_std", "total_val_c_loss_mean",
                                  "total_val_c_loss_std", "total_val_accuracy_mean", "total_val_accuracy_std",
                                  "total_test_c_loss_mean", "total_test_c_loss_std", "total_test_accuracy_mean",
                                  "total_test_accuracy_std"], create=True)

# Experiment initialization and running
with tf.Session() as sess:
    sess.run(init)
    train_saver = tf.train.Saver()
    val_saver = tf.train.Saver()
    if args.continue_from_epoch != -1: #load checkpoint if needed
        checkpoint = "saved_models/{}_{}.ckpt".format(args.experiment_title, args.continue_from_epoch)
        variables_to_restore = []
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(var)
            variables_to_restore.append(var)

        tf.logging.info('Fine-tuning from %s' % checkpoint)

        fine_tune = slim.assign_from_checkpoint_fn(
            checkpoint,
            variables_to_restore,
            ignore_missing_vars=True)
        fine_tune(sess)
    start_epoch = args.continue_from_epoch if args.continue_from_epoch != -1 else 0
    best_val_acc_mean = 0.
    best_val_epoch = 6
    with tqdm.tqdm(total=args.total_epochs) as pbar_e:
        for e in range(start_epoch, args.total_epochs):
            total_train_c_loss_mean, total_train_c_loss_std, total_train_accuracy_mean, total_train_accuracy_std =\
                experiment.run_training_epoch(total_train_batches=total_train_batches,
                                                                                sess=sess)
            print("Epoch {}: train_loss_mean: {}, train_loss_std: {}, train_accuracy_mean: {}, train_accuracy_std: {}"
                  .format(e, total_train_c_loss_mean, total_train_c_loss_std, total_train_accuracy_mean, total_train_accuracy_std))

            total_val_c_loss_mean, total_val_c_loss_std, total_val_accuracy_mean, total_val_accuracy_std = \
                experiment.run_validation_epoch(total_val_batches=total_val_batches,
                                              sess=sess)
            print("Epoch {}: val_loss_mean: {}, val_loss_std: {}, val_accuracy_mean: {}, val_accuracy_std: {}"
                  .format(e, total_val_c_loss_mean, total_val_c_loss_std, total_val_accuracy_mean,
                          total_val_accuracy_std))

            if total_val_accuracy_mean >= best_val_acc_mean: #if new best val accuracy -> produce test statistics
                best_val_acc_mean = total_val_accuracy_mean
                best_val_epoch = e

                val_save_path = val_saver.save(sess, "{}/best_val_{}_{}.ckpt".format(saved_models_filepath, args.experiment_title, e))

                total_test_c_loss_mean, total_test_c_loss_std, total_test_accuracy_mean, total_test_accuracy_std \
                    = -1, -1, -1, -1

            save_statistics(logs_filepath,
                            [e, total_train_c_loss_mean, total_train_c_loss_std, total_train_accuracy_mean,
                             total_train_accuracy_std, total_val_c_loss_mean, total_val_c_loss_std,
                             total_val_accuracy_mean, total_val_accuracy_std,
                             total_test_c_loss_mean, total_test_c_loss_std, total_test_accuracy_mean,
                             total_test_accuracy_std])

            save_path = train_saver.save(sess, "{}/{}_{}.ckpt".format(saved_models_filepath, args.experiment_title, e))
            pbar_e.update(1)

        val_saver.restore(sess,
                          "{}/best_val_{}_{}.ckpt".format(saved_models_filepath, args.experiment_title, best_val_epoch))
        total_test_c_loss_mean, total_test_c_loss_std, total_test_accuracy_mean, total_test_accuracy_std = \
            experiment.run_testing_epoch(total_test_batches=total_test_batches,
                                         sess=sess)
        print(
            "Epoch {}: test_loss_mean: {}, test_loss_std: {}, test_accuracy_mean: {}, test_accuracy_std: {}"
                .format(best_val_epoch, total_test_c_loss_mean, total_test_c_loss_std, total_test_accuracy_mean,
                        total_test_accuracy_std))
        save_statistics(logs_filepath,
                        ["Test error on best validation model", -1, -1, -1,
                         -1, -1, -1,
                         -1, -1,
                         total_test_c_loss_mean, total_test_c_loss_std, total_test_accuracy_mean,
                         total_test_accuracy_std])

