from data import FolderDatasetLoader
import tqdm
from one_shot_learning_network import MatchingNetworkHandler
from utils.parser_utils import get_args
import os
import numpy as np

from utils.storage import build_experiment_folder, create_json_experiment_log, save_statistics, \
    update_json_experiment_log_epoch_stats

args = get_args()

data = FolderDatasetLoader(num_of_gpus=1, batch_size=args.batch_size, image_height=args.image_height,
                           image_width=args.image_width,
                           image_channels=args.image_channels,
                           train_val_test_split=(1200/1622, 211/1622, 211/1622),
                           samples_per_iter=1, num_workers=args.num_dataprovider_workers,
                           data_path="datasets/omniglot_dataset", name="omniglot_dataset",
                           indexes_of_folders_indicating_class=[-2, -3], reset_stored_filepaths=False,
                           num_samples_per_class=args.num_samples_per_class,
                           num_classes_per_set=args.num_classes_per_set, label_as_int=False)



matching_net = MatchingNetworkHandler(args=args, im_shape=(args.batch_size, args.image_channels, args.image_height,
                                                           args.image_width))
saved_models_filepath, logs_filepath, samples_filepath = build_experiment_folder(experiment_name=args.experiment_name)

total_losses = dict()
best_val_acc = 0.
best_val_iter = 0
max_models_to_save = args.max_models_to_save
create_summary_csv = False

if args.continue_from_iter == -1:
    create_json_experiment_log(experiment_log_dir=logs_filepath, args=args)
    create_summary_csv = True



if args.continue_from_iter != -1:
    model_idx = int(args.continue_from_iter / 500) % max_models_to_save
    best_val_iter, best_val_acc, loss = \
        matching_net.load_model(model_save_dir=saved_models_filepath, model_name="train_model", model_idx=model_idx)

def build_summary_dict(total_losses, phase, summary_losses=None):
    if summary_losses is None:
        summary_losses = dict()

    for key in total_losses:
        summary_losses["{}_{}_mean".format(phase, key)] = np.mean(total_losses[key])  #:.5f
        summary_losses["{}_{}_std".format(phase, key)] = np.std(total_losses[key])  #:.5f

    return summary_losses

def build_loss_summary_string(summary_losses):

    output_update = ""
    for key, value in zip(list(summary_losses.keys()), list(summary_losses.values())):
        output_update += "{}: {:.5f}, ".format(key, value)

    return output_update


def merge_two_dicts(first_dict, second_dict):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = first_dict.copy()
    z.update(second_dict)
    return z

current_iter = args.continue_from_iter if args.continue_from_iter != -1 else 0
with tqdm.tqdm(total=int(args.total_epochs)) as pbar_epoch:
    with tqdm.tqdm(total=int(500 * args.total_epochs)) as pbar_train:
        for sample_idx, train_sample in enumerate(data.get_train_batches(total_batches=int(500 * args.total_epochs),
                                                                                     augment_images=True)):
            x_support_set, x_target_set, y_support_set, y_target_set = train_sample
            data_batch = (x_support_set[0, 0], x_target_set[0, 0], y_support_set[0, 0], y_target_set[0, 0])
            losses = matching_net.run_train_iter(data_batch=data_batch, epoch=int(current_iter / 500))
            for key, value in zip(list(losses.keys()), list(losses.values())):
                if key not in total_losses:
                    total_losses[key] = [float(value)]
                else:
                    total_losses[key].append(float(value))


            train_losses = build_summary_dict(total_losses=total_losses, phase="train")
            train_output_update = build_loss_summary_string(train_losses)

            pbar_train.set_description("training phase {} -> {}".format(int(current_iter / 500), train_output_update))

            current_iter += 1
            if current_iter % 500 == 0:# and sample_id != 0:
                epoch = int(current_iter / 500)
                print("saved train model to", os.path.join(saved_models_filepath,
                                                           "train_model_{}".format(
                                                               int(epoch % max_models_to_save))), "\n")
                matching_net.save_model(model_save_dir=os.path.join(saved_models_filepath,
                                                                    "train_model_{}".format(
                                                                        int(epoch))),
                                        loss=train_losses['train_loss_mean'],
                                        accuracy=train_losses['train_accuracy_mean'], iter=current_iter)
                total_losses = dict()
                with tqdm.tqdm(total=150) as pbar_val:
                    for _, val_sample in enumerate(data.get_val_batches(total_batches=int(150),
                                                                                    augment_images=False)):
                        x_support_set, x_target_set, y_support_set, y_target_set = val_sample
                        data_batch = (x_support_set[0, 0], x_target_set[0, 0], y_support_set[0, 0], y_target_set[0, 0])
                        losses = matching_net.run_validation_iter(data_batch=data_batch)
                        for key, value in zip(list(losses.keys()), list(losses.values())):
                            if key not in total_losses:
                                total_losses[key] = [float(value)]
                            else:
                                total_losses[key].append(float(value))

                        val_losses = build_summary_dict(total_losses=total_losses, phase="val")
                        val_output_update = build_loss_summary_string(val_losses)

                        pbar_val.update(1)
                        pbar_val.set_description("val_phase {} -> {}".format(int(current_iter / 500), val_output_update))

                    if val_losses["val_accuracy_mean"] > best_val_acc:
                        print("Best validation accuracy", val_losses["val_accuracy_mean"])
                        best_val_acc = val_losses["val_accuracy_mean"]
                        best_val_iter = current_iter

                        matching_net.save_model(model_save_dir=os.path.join(saved_models_filepath,
                                                                            "val_model_{}".format(
                                                                                int(epoch))),
                                                loss=val_losses['val_loss_mean'],
                                                accuracy=val_losses['val_accuracy_mean'], iter=current_iter)

                pbar_epoch.update(1)

                epoch_summary_losses = merge_two_dicts(first_dict=train_losses, second_dict=val_losses)
                epoch_summary_string = build_loss_summary_string(epoch_summary_losses)
                epoch_summary_losses["epoch"] = int(current_iter / 500)

                if create_summary_csv:
                    summary_statistics_filepath = save_statistics(logs_filepath, list(epoch_summary_losses.keys()),
                                                                  create=True)
                    create_summary_csv = False

                print("epoch {} -> {}".format(epoch_summary_losses["epoch"], epoch_summary_string))

                summary_filename = update_json_experiment_log_epoch_stats(epoch_stats=epoch_summary_losses,
                                                                          experiment_log_dir=logs_filepath)

                summary_statistics_filepath = save_statistics(logs_filepath,
                                                              list(epoch_summary_losses.values()))
                # if args.use_gdrive:
                #     gdrive.save_in_logs(summary_statistics_filepath)
                #     gdrive.save_in_logs(summary_filename)

                total_losses = dict()
                # losses = matching_net.run_validation_iter(data_batch=data_batch)
            pbar_train.update(1)

print("Best validation accuracy", best_val_acc, best_val_iter)
