import torch
import torch.nn as nn
from tractoracle_irt.oracles.transformer_oracle import LightningLikeModule
from tractoracle_irt.utils.comet_monitor import OracleMonitor
from tractoracle_irt.utils.torch_utils import get_device
from tractoracle_irt.algorithms.shared.utils import \
    (add_item_to_means, mean_losses, add_losses, get_mean_item)
from tractoracle_irt.utils.hooks import HooksManager, OracleHookEvent
from collections import defaultdict
from tqdm import tqdm
import numpy as np


def to_device(obj, device):
    if isinstance(obj, (list, tuple)):
        return [to_device(o, device) for o in obj]
    return obj.to(device)


class MicroBatchInfo(object):
    def __init__(self):
        self.info = defaultdict(list)
        self.actual_micro_batch = 0
        self.scaled_loss_accum = 0

    def _add_single(self, key, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.info[key].append(value)

    def add_scaled_loss(self, loss):
        if isinstance(loss, torch.Tensor):
            self.scaled_loss_accum += loss.item()
        else:
            self.scaled_loss_accum += loss

    def add_info(self, info):
        self.actual_micro_batch += 1
        for k, v in info.items():
            self._add_single(k, v)

    def avg_and_get(self, with_loss=True):
        # We want to average the values of the micro batches
        # for each key.
        for k, v in self.info.items():
            self.info[k] = sum(v) / self.actual_micro_batch

        if with_loss:
            return self.scaled_loss_accum, self.info
        return self.info

    def sum_and_get(self, with_loss=True):
        # We want to sum the values of the micro batches
        # for each key.
        for k, v in self.info.items():
            self.info[k] = sum(v)

        if with_loss:
            return self.scaled_loss_accum, self.info
        return self.info

    def reset(self):
        self.info = defaultdict(list)
        self.actual_micro_batch = 0
        self.scaled_loss_accum = 0


class OracleTrainer(object):
    def __init__(self,
                 experiment,
                 experiment_path,
                 saving_path,
                 max_epochs,
                 use_comet=True,
                 enable_auto_checkpointing=True,
                 checkpoint_prefix='',
                 val_interval=1,
                 log_interval=1,
                 grad_accumulation_steps=1,
                 device=get_device(),
                 metrics_prefix=None,
                 first_oracle_train_steps=None,
                 disable=False,
                 offline=False
                 ):
        self.experiment = experiment
        self.saving_path = saving_path

        self.auto_checkpointing_enabled = enable_auto_checkpointing
        self.checkpoint_prefix = checkpoint_prefix
        self.device = device
        self.max_epochs = max_epochs
        self.first_oracle_train_steps = first_oracle_train_steps
        self.disabled = disable

        if self.first_oracle_train_steps is not None:
            self.is_first_training_loop = True
        else:
            self.is_first_training_loop = False # Proceed as usual, there's nothing different about the first fit_iter called.
        
        self.val_interval = val_interval

        self._global_epoch = 0
        self._last_valid_metrics = {}
        self.hooks_manager = HooksManager(OracleHookEvent)
        self.oracle_monitor = OracleMonitor(
            experiment=self.experiment,
            experiment_path=experiment_path,
            use_comet=use_comet,
            metrics_prefix=metrics_prefix,
            offline=offline
        )
        
        self.log_interval = log_interval  # Log every n update steps
        self.grad_accumulation_steps = grad_accumulation_steps
        assert self.grad_accumulation_steps > 0, \
            "grad_accumulation_steps must be greater than 0"

    def save_hyperparameters(self):
        self._verify_model_was_setup()

        hyperparameters = self.oracle_model.hyperparameters
        hyperparameters.update({
            'saving_path': self.saving_path,
            'max_epochs': self.max_epochs,
            'first_oracle_train_steps': self.first_oracle_train_steps,
            'val_interval': self.val_interval,
            'log_interval': self.log_interval,
            'grad_accumulation_steps': self.grad_accumulation_steps
        })

        self.oracle_monitor.log_parameters(hyperparameters)

    def setup_model_training(self, oracle_model: LightningLikeModule):
        """
        This method must be called before calling fit_iter().

        It is used to configure the optimizer, the scheduler and the scaler.
        Contrary to the standard fit() method from Lightning AI that takes
        the model as an argument, we want to be able to call fit() multiple
        times with a coherent configuration of the optimizer, the scheduler
        and the scaler to train the same model.
        """
        if self.disabled:
            return
        
        self.oracle_model = oracle_model
        self._reset_optimizers()
        self._global_epoch = 0
        self._global_plotting_step = 0
        self.save_hyperparameters()

    def _verify_model_was_setup(self):
        if not hasattr(self, 'oracle_model') or self.oracle_model is None:
            raise ValueError(
                "You must call setup_model_training before calling "
                "fit_iter or save_hyperparameters.\n"
                "This makes sure the model is properly setup for training,\n"
                "by configuring the optimizer, the scheduler and the scaler.")

    def _reset_optimizers(self):
        optim_info = self.oracle_model.configure_optimizers(self)
        self.optimizer = optim_info['optimizer']
        self.scheduler = optim_info['lr_scheduler']['scheduler']
        self.scaler = optim_info['scaler']

    def save_model_checkpoint(self, is_best=False):
        """
        This method is called automatically if self.auto_checkpointing_enabled
        is enabled.
        """
        if self.disabled:
            return
        
        checkpoint_dict = self.oracle_model.pack_for_checkpoint(
            self._global_epoch, self._last_valid_metrics, self.optimizer,
            self.scheduler, self.scaler)

        # Always have a copy of the latest model
        if is_best:
            filename = '{}/{}best_epoch.ckpt'.format(
                    self.saving_path, self.checkpoint_prefix + '_')
        else:
            filename = '{}/{}latest_epoch.ckpt'.format(
                self.saving_path, self.checkpoint_prefix + '_')
            
        torch.save(checkpoint_dict, filename)

    def fit_iter(
        self,
        train_dataloader,
        val_dataloader,
        reset_optimizers=False
    ):
        """
        This method trains the model for a given number of epochs.
        Contrary to the standard fit() method, this method is made
        to be called multiple times with the same model. This is
        especially useful when training a model iteratively.

        Args:
            train_dataloader: The training dataloader
            val_dataloader: The validation dataloader
            reset_optimizers: If True, the optimizer, the scheduler and the
                scaler are reset before training the model instead of reusing
                the same optimizer, scheduler and scaler as well as their last
                respective states.
        """
        if self.disabled:
            return
        
        self._verify_model_was_setup()

        self.oracle_model.train()  # Set model to training mode
        self.oracle_model = self.oracle_model.to(self.device)

        if reset_optimizers:
            self._reset_optimizers()

        has_non_complete_batch = int(len(
            train_dataloader) % self.grad_accumulation_steps != 0)
        nb_of_full_batches = len(
            train_dataloader) // self.grad_accumulation_steps

        nb_epochs = self.first_oracle_train_steps if self.is_first_training_loop else self.max_epochs
        self.is_first_training_loop = False # Make sure we only use the first steps for the first training loop

        best_loss = float('inf')
        with tqdm(range(nb_of_full_batches + has_non_complete_batch)) as pbar:
            for epoch in range(nb_epochs):
                pbar.set_description(f"Training oracle epoch {epoch}")
                pbar.update()

                self.hooks_manager.trigger_hooks(
                    OracleHookEvent.ON_TRAIN_EPOCH_START)

                ep_train_metrics = defaultdict(list)
                ep_train_matrix = defaultdict(list)
                mb_accumulator = MicroBatchInfo()
                mb_accum_matrix = MicroBatchInfo()

                for i, batch in enumerate(train_dataloader):
                    self.hooks_manager.trigger_hooks(
                        OracleHookEvent.ON_TRAIN_BATCH_START)

                    batch = to_device(batch, self.device)
                    # Train step
                    loss, train_info, matrix = self.oracle_model.training_step(
                        batch, i)

                    # Adjust the loss for gradient accumulation
                    loss = loss / self.grad_accumulation_steps

                    # Since we're working with gradient accumulation, we need
                    # to accumulate the stats of the micro batches and then
                    # average them (except the loss that's already averaged
                    # or scaled correctly on the previous line.
                    mb_accumulator.add_info(train_info)
                    mb_accumulator.add_scaled_loss(loss.detach())
                    mb_accum_matrix.add_info(matrix)

                    # Clear gradients
                    self.optimizer.zero_grad()

                    # Backward pass
                    self.scaler.scale(loss).backward()

                    # Accumulate the gradients and update the parameters once
                    # we accumulated self.grad_accumulation_steps gradients.
                    # This number of steps should be calculated based on the
                    # batch size and the GPU memory available.
                    is_last_batch = i == len(train_dataloader) - 1
                    if (i + 1) % self.grad_accumulation_steps == 0\
                            or is_last_batch:
                        # End of the batch
                        batch_loss, batch_info = mb_accumulator.avg_and_get()
                        mb_accumulator.reset()
                        add_item_to_means(ep_train_metrics, batch_info)

                        matrix_info = mb_accum_matrix.sum_and_get(
                            with_loss=False)
                        mb_accum_matrix.reset()
                        add_item_to_means(ep_train_matrix, matrix_info)

                        # Log the loss
                        pbar.set_postfix(
                            loss=f"{batch_loss:.4f}",
                            lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                            avg_ep_mse=f"{get_mean_item(ep_train_metrics, 'train_mse'):.2e}",
                            train_mse=f"{batch_info['train_mse']:.2e}")
                        pbar.update()

                        ep_train_metrics['lr'].append(torch.tensor(
                            self.optimizer.param_groups[0]['lr']))

                        # Unscaling the gradients is required before
                        # clipping them.
                        # Ref: https://pytorch.org/docs/main/notes/amp_examples.html#gradient-clipping
                        self.scaler.unscale_(self.optimizer)
                        _ = nn.utils.clip_grad_norm_(
                            self.oracle_model.parameters(), 1.0)

                        # Update parameters
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()

                        self.hooks_manager.trigger_hooks(
                            OracleHookEvent.ON_TRAIN_BATCH_END)

                        # Log metrics to monitor after every log_interval
                        # steps. Each epoch is so long that we would like
                        # to see the progression of the metrics during each
                        # epoch as well.
                        if (i % self.log_interval == 0) \
                                or (i == len(train_dataloader) - 1):

                            train_metrics_avg = mean_losses(ep_train_metrics)
                            self.oracle_monitor.log_metrics(
                                train_metrics_avg,
                                step=self._global_plotting_step,
                                epoch=self._global_epoch)

                            train_matrix_avg = add_losses(ep_train_matrix)
                            self.oracle_monitor.log_metrics(
                                train_matrix_avg,
                                step=self._global_plotting_step,
                                epoch=self._global_epoch)

                            self._global_plotting_step += 1

                    self.hooks_manager.trigger_hooks(
                        OracleHookEvent.ON_TRAIN_MICRO_BATCH_END)

                if epoch % self.val_interval == 0:
                    self._validate(val_dataloader, epoch, best_loss)

                pbar.reset()
                self._global_epoch += 1
                self.hooks_manager.trigger_hooks(
                    OracleHookEvent.ON_TRAIN_EPOCH_END)

        self.oracle_model = self.oracle_model.to('cpu')

    def _validate(self, val_dataloader, epoch, best_loss):
        self.oracle_model.eval()

        val_metrics = defaultdict(list)
        val_matrix = defaultdict(list)
        for i, batch in enumerate(val_dataloader):
            batch = to_device(batch, self.device)
            self.hooks_manager.trigger_hooks(
                OracleHookEvent.ON_VAL_BATCH_START)

            # TODO: Implement validation step
            _, val_info, matrix = self.oracle_model.validation_step(batch, i)
            add_item_to_means(val_metrics, val_info)
            add_item_to_means(val_matrix, matrix)

            self.hooks_manager.trigger_hooks(OracleHookEvent.ON_VAL_BATCH_END)

        val_metrics = mean_losses(val_metrics)
        self._last_valid_metrics = val_metrics
        self.oracle_monitor.log_metrics(val_metrics,
                                        step=self._global_plotting_step,
                                        epoch=self._global_epoch)

        val_matrix = add_losses(val_matrix)
        self.oracle_monitor.log_metrics(val_matrix,
                                        step=self._global_plotting_step,
                                        epoch=self._global_epoch)

        # Checkpointing
        if self.auto_checkpointing_enabled:
            self.save_model_checkpoint()

            is_best_epoch = val_metrics['val_loss'] < best_loss
            if is_best_epoch:
                self.save_model_checkpoint(is_best=True)
                best_loss = val_metrics['val_loss']

        self.oracle_model.train()

    def test(self, test_dataloader, compute_histogram_metrics=False, step=None, epoch=None):
        if self.disabled:
            return
        
        self.hooks_manager.trigger_hooks(OracleHookEvent.ON_TEST_START)

        self.oracle_model.eval()  # Set model to evaluation mode
        self.oracle_model.to(self.device)

        test_metrics = defaultdict(list)
        histogram_metrics = {} if compute_histogram_metrics else None
        for i, batch in enumerate(tqdm(test_dataloader, desc="testing oracle")):
            batch = to_device(batch, self.device)
            _, test_info = self.oracle_model.test_step(
                batch, i, histogram_metrics)
            add_item_to_means(test_metrics, test_info)

        test_metrics = mean_losses(test_metrics)
        
        _step = step if step is not None else self._global_plotting_step
        _epoch = epoch if epoch is not None else self._global_epoch
        self.oracle_monitor.log_metrics(test_metrics, step=_step, epoch=_epoch)
        self.hooks_manager.trigger_hooks(OracleHookEvent.ON_TEST_END)

        if compute_histogram_metrics:
            for bin_name, metrics in histogram_metrics.items():
                accuracy = metrics['nb_correct'] / \
                    metrics['nb_streamlines'] if metrics['nb_streamlines'] > 0 else -1
                pos_accuracy = metrics['nb_correct_positives'] / \
                    metrics['nb_positive'] if metrics['nb_positive'] > 0 else -1
                neg_accuracy = metrics['nb_correct_negatives'] / \
                    metrics['nb_negative'] if metrics['nb_negative'] > 0 else -1
                histogram_metrics[bin_name]['accuracy'] = accuracy
                histogram_metrics[bin_name]['pos_accuracy'] = pos_accuracy
                histogram_metrics[bin_name]['neg_accuracy'] = neg_accuracy

            print("Histogram metrics:")
            for bin_name, metrics in histogram_metrics.items():
                print(
                    f"Bin {bin_name}: acc {metrics['accuracy']} | nb_streamlines {metrics['nb_streamlines']}")

            # Show the histogram with matplotlib
            import matplotlib.pyplot as plt

            def histogram_all():
                fig, ax = plt.subplots(figsize=(18, 5))

                accuracies = np.asarray([m['accuracy']
                                         for m in histogram_metrics.values()])
                nb_streamlines = np.asarray([m['nb_streamlines']
                                             for m in histogram_metrics.values()])
                # Replace -1 values by zero
                accuracies[accuracies == -1] = 0

                ax.bar(histogram_metrics.keys(),
                       accuracies, width=0.8, align='center')

                for i, v in enumerate(accuracies):
                    ax.text(i, v + 0.01,
                            str(nb_streamlines[i]), color='black', ha='center').set_fontsize(6)

                plt.xlabel('Streamline length <')
                plt.ylabel('Accuracy')
                ax.set_title('All streamlines')
                plt.savefig('histogram.png')

            def histogram_pos():
                fig, ax = plt.subplots(figsize=(18, 5))

                accuracies = np.asarray([m['pos_accuracy']
                                         for m in histogram_metrics.values()])
                nb_streamlines = np.asarray([m['nb_positive']
                                             for m in histogram_metrics.values()])
                # Replace -1 values by zero
                accuracies[accuracies == -1] = 0

                ax.bar(histogram_metrics.keys(),
                       accuracies, width=0.8, align='center')

                for i, v in enumerate(accuracies):
                    ax.text(i, v + 0.01,
                            str(nb_streamlines[i]), color='black', ha='center').set_fontsize(6)

                plt.xlabel('Streamline length <')
                plt.ylabel('Accuracy')
                ax.set_title('Positive streamlines')
                plt.savefig('histogram_pos.png')

            def histogram_neg():
                fig, ax = plt.subplots(figsize=(18, 5))

                accuracies = np.asarray([m['neg_accuracy']
                                         for m in histogram_metrics.values()])
                nb_streamlines = np.asarray([m['nb_negative']
                                             for m in histogram_metrics.values()])
                # Replace -1 values by zero
                accuracies[accuracies == -1] = 0

                ax.bar(histogram_metrics.keys(),
                       accuracies, width=0.8, align='center')

                for i, v in enumerate(accuracies):
                    ax.text(i, v + 0.01,
                            str(nb_streamlines[i]), color='black', ha='center').set_fontsize(6)

                plt.xlabel('Streamline length <')
                plt.ylabel('Accuracy')
                ax.set_title('Negative streamlines')
                plt.savefig('histogram_neg.png')

            histogram_all()
            histogram_pos()
            histogram_neg()

        return test_metrics
