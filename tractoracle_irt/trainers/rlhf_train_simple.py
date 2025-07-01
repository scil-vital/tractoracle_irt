import os
import argparse
import tempfile
import comet_ml
import json
import shutil
from pathlib import Path

from comet_ml import Experiment as CometExperiment
from comet_ml import OfflineExperiment as CometOfflineExperiment

from dipy.io.streamline import load_tractogram
from dipy.io.streamline import save_tractogram

from tractoracle_irt.utils.logging import get_logger
from tractoracle_irt.trainers.cross_q_train import CrossQTraining
from tractoracle_irt.trainers.sac_auto_train import Training, add_sac_auto_args, SACAutoTraining
from tractoracle_irt.trainers.tractoraclenet_train import add_oracle_train_args
from tractoracle_irt.trainers.train import add_training_args
from tractoracle_irt.utils.logging import setup_logging, add_logging_args
from tractoracle_irt.algorithms.rl import RLAlgorithm
from tractoracle_irt.algorithms.sac_auto import SACAuto, SACAutoHParams
from tractoracle_irt.algorithms.cross_q import CrossQ, CrossQHParams
from tractoracle_irt.environments.env import BaseEnv
from tractoracle_irt.tracking.tracker import Tracker
from tractoracle_irt.filterers.tractometer.tractometer_filterer import TractometerFilterer
from tractoracle_irt.filterers.extractor.extractor_filterer import ExtractorFilterer
from tractoracle_irt.filterers.verifyber.verifyber_filterer import VerifyberFilterer
from tractoracle_irt.filterers.rbx.rbx_filterer import RbxFilterer
from tractoracle_irt.oracles.oracle import OracleSingleton
from tractoracle_irt.trainers.oracle.oracle_trainer import OracleTrainer
from tractoracle_irt.trainers.oracle.data_module import StreamlineDataModule
from tractoracle_irt.trainers.oracle.streamline_dataset_manager import StreamlineDatasetManager
from tractoracle_irt.utils.torch_utils import assert_accelerator
from tractoracle_irt.utils.utils import prettier_metrics, prettier_dict
from tractoracle_irt.filterers.streamlines_sampler import StreamlinesSampler
from tractoracle_irt.utils.hooks import RlHookEvent
from tqdm import tqdm
from dataclasses import dataclass
assert_accelerator()

LOGGER = get_logger(__name__)

# TODO:
# Inheriting directly from the CrossQHParams isn't the best way to do it.
# Ideally, the config file would be split between the general experiment
# parameters, the agent parameters, the oracle parameters and the rlhf parameters.
@dataclass
class RlhfHParams(CrossQHParams):
    agent_checkpoint: str

    oracle_lr: float
    oracle_train_steps: int
    first_oracle_train_steps: int
    agent_train_steps: int
    num_workers: int
    rlhf_inter_npv: int
    disable_oracle_training: bool
    batch_size: int
    oracle_batch_size: int
    grad_accumulation_steps: int
    nb_new_streamlines_per_iter: int
    max_dataset_size: int
    warmup_agent_steps: int

    dataset_to_augment: str = None

    def __post_init__(self):
        if self.agent_checkpoint:
            assert os.path.isfile(
                self.agent_checkpoint), "Agent checkpoint must be a checkpoint file."

class RlhfTraining(Training):

    def __init__(
        self,
        config: dict,
        trainer_cls: Training,
        comet_experiment: CometExperiment = None
    ):
        # Only load the parameters from the parent instead of calling
        # the full constructor twice. (As we call it for the agent_trainer
        # below).
        self.init_hyperparameters(config)

        # General RLHF parameters.
        self.ref_model_dir = os.path.join(self.hp.experiment_path, "ref_model")
        self.model_saving_dirs.append(self.ref_model_dir)
        if not os.path.exists(self.ref_model_dir):
            os.makedirs(self.ref_model_dir)

        self.oracle_training_dir = os.path.join(self.hp.experiment_path, "oracle")
        if not os.path.exists(self.oracle_training_dir):
            os.makedirs(self.oracle_training_dir)

        if self.hp.disable_oracle_training:
            LOGGER.warning("Oracle training is disabled. The dataset will "
                           "be augmented to evaluate the oracles during the "
                           "agent's training.")

        ################################################
        # Start by initializing the agent trainer.     #
        if comet_experiment is None:
            if not self.hp.offline:
                comet_experiment = CometExperiment(project_name=self.hp.experiment,
                                            workspace=self.hp.workspace, parse_args=False,
                                            auto_metric_logging=False,
                                            disabled=not self.hp.use_comet)
            else:
                print(f">>> Running in offline mode, no comet logging (in {self.hp.experiment_path}). <<<")
                os.makedirs(self.hp.experiment_path, exist_ok=True)
                
                comet_experiment = CometOfflineExperiment(
                    project_name=self.hp.experiment,
                    workspace=self.hp.workspace,
                    parse_args=False,
                    auto_metric_logging=False,
                    disabled=not self.hp.use_comet,
                    offline_directory=self.hp.experiment_path
                )

        comet_experiment.set_name(self.hp.experiment_id)

        self.agent_trainer: Training = trainer_cls(config, comet_experiment)
        _ = self.agent_trainer.setup_environment_and_info() # TODO: Remove this?
        
        # Replace the get_alg method by the one from the agent trainer.
        # This way, if we have a CrossQ trainer, we have the CrossQ alg.
        self.get_alg = self.agent_trainer.get_alg

        # Since backuping is implemented in Training, we disable
        # it to avoid backuping the same files twice to control the backuping
        # process from this class.
        self.agent_trainer.backuper.disable()

        ################################################
        # Setup oracle training
        ################################################

        # Load reward oracle
        self.oracle_reward = OracleSingleton(self.hp.oracle_reward_checkpoint,
                                      device=self.device,
                                      batch_size=self.hp.oracle_batch_size,
                                      lr=self.hp.oracle_lr)
        
        self.oracle_reward_trainer = OracleTrainer(
            comet_experiment,
            self.hp.experiment_path,
            self.oracle_training_dir,
            self.hp.oracle_train_steps,
            enable_auto_checkpointing=False,
            checkpoint_prefix='reward',
            val_interval=1,
            device=self.device,
            grad_accumulation_steps=self.hp.grad_accumulation_steps,
            metrics_prefix='reward',
            first_oracle_train_steps=self.hp.first_oracle_train_steps,
            offline=self.hp.offline
        )
        self.oracle_reward_trainer.setup_model_training(self.oracle_reward.model)


        # If the two oracles are the same, that means that the checkpoint
        # is the same. The oracle instance is the same, so we don't want to
        # train it twice.
        # Especially during the stopping criterion training, we train it
        # to be able to predict partial streamlines.
        self.oracle_crit = OracleSingleton(self.hp.oracle_crit_checkpoint,
                                           device=self.device,
                                           batch_size=self.hp.oracle_batch_size,
                                           lr=self.hp.oracle_lr)
        self.oracle_crit_trainer = OracleTrainer(
            comet_experiment,
            self.hp.experiment_path,
            self.oracle_training_dir,
            self.hp.oracle_train_steps,
            enable_auto_checkpointing=False,
            checkpoint_prefix='crit',
            val_interval=1,
            device=self.device,
            grad_accumulation_steps=self.hp.grad_accumulation_steps,
            metrics_prefix='crit',
            first_oracle_train_steps=self.hp.first_oracle_train_steps,
            disable=self.oracle_crit == self.oracle_reward,
            offline=self.hp.offline
        )
        self.oracle_crit_trainer.setup_model_training(self.oracle_crit.model)
        self.oracle_crit_disabled = self.oracle_crit_trainer.disabled

        # Register hooks on best VC reached to save the oracles that
        # contributed to reach that level of VC.
        def _save_oracles_on_best_vc():
            self.oracle_crit_trainer.save_model_checkpoint(is_best=True)
            self.oracle_reward_trainer.save_model_checkpoint(is_best=True)

        self.agent_trainer._hooks_manager.register_hook(
            RlHookEvent.ON_RL_BEST_VC,
            _save_oracles_on_best_vc
        )

        ###########################################################
        # Continue by initializing the streamline dataset manager #
        self.dataset_manager = StreamlineDatasetManager(saving_path=self.oracle_training_dir,
                                                        dataset_to_augment_path=self.hp.dataset_to_augment,
                                                        max_dataset_size=self.hp.max_dataset_size,
                                                        nb_points=self.oracle_reward.nb_points)
        self.streamline_sampler = StreamlinesSampler()

    @property
    def hparams_class(self):
        return RlhfHParams

    def setup_logging(self):
        """ Override the setup_logging method to avoid creating a new experiment. """
        self.save_hyperparameters()

    def rl_train(
        self,
        alg: RLAlgorithm,
        env: BaseEnv,
        valid_env: BaseEnv,
        max_ep: int = 10,
        **kwargs
    ):
        """ Train the RL algorithm for N epochs. An epoch here corresponds to
        running tracking on the training set until all streamlines are done.
        This loop should be algorithm-agnostic. Between epochs, report stats
        so they can be monitored during training

        Parameters:
        -----------
            alg: RLAlgorithm
                The RL algorithm, either TD3, PPO or any others
            env: BaseEnv
                The tracking environment
            valid_env: BaseEnv
                The validation tracking environment (forward).
            """
        current_ep = 0

        ################################################
        # Setup agent trainer
        # (needed since we don't call the run method)
        ################################################
        self.agent_trainer.setup_logging()

        if self.hp.agent_checkpoint is not None:
            # The agent is already pretrained, just need to fine-tune it.
            LOGGER.info(
                "Skipping pretraining procedure: loading agent from checkpoint...")
            alg.load_checkpoint(self.hp.agent_checkpoint)
            
            # Instead of having to pack and serialize the model again,
            # as this takes time, just copy the file.
            # This aims to do the following:
            # self.save_model(alg, save_model_dir=self.ref_model_dir)
            #
            # We keep a copy of the initial model state just as a reference.
            # This has no real use in the training process.
            ckpt_path = Path(self.ref_model_dir) / "init_model_state.ckpt"
            shutil.copyfile(self.hp.agent_checkpoint, ckpt_path)

            LOGGER.info("Done.")

        self.agent_trainer.comet_monitor.e.add_tag(
            "RLHF-start-ep-{}".format(current_ep))

        ################################################
        # Setup environment
        ################################################
        self.tracker_env = self.get_rlhf_env(npv=self.hp.rlhf_inter_npv)
        self.tracker = Tracker(
            alg, self.hp.n_actor, prob=1.0, compress=0.0)

        # Setup filterers which will be used to filter tractograms
        # for the RLHF pipeline.
        self.filterers = []

        if self.hp.tractometer_validator:
            self.filterers.append(
                TractometerFilterer(self.hp.scoring_data, self.hp.tractometer_reference,
                                dilate_endpoints=self.hp.tractometer_dilate))
            
        if self.hp.extractor_validator:
            self.filterers.append(
                ExtractorFilterer(sif_img_path=self.hp.extractor_sif_img_path))
            
            self.extractor_filterer = self.filterers[-1]

        if self.hp.verifyber_validator:
            self.filterers.append(
                VerifyberFilterer(self.hp.verifyber_sif_img_path))
        
        if self.hp.rbx_validator:
            self.filterers.append(
                RbxFilterer(self.hp.atlas_directory, self.hp.rbx_sif_img_path))

        if len(self.filterers) < 1:
            raise ValueError("At least one of the filterers must be enabled.")

        do_warmup = self.hp.warmup_agent_steps and current_ep < self.hp.warmup_agent_steps - 1

        ################################################
        # RLHF loop to fine-tune the oracle to the RL
        # agent and vice-versa.
        ################################################
        i = 0
        while i < max_ep: 
            self.start_finetuning_epoch(i, do_warmup)

            if not do_warmup:
                ################################################
                # Add new streamlines to the dataset
                ################################################
                self._add_streamlines_to_dataset(i)

                ################################################
                # Train the Oracles
                ################################################
                if not self.hp.disable_oracle_training:
                    self.train_reward()
                    self.train_stopping_criterion()
                else:
                    self.test_reward(step=i, epoch=0)
                    self.test_stopping_criterion(step=i, epoch=0)

            ################################################
            # Train the RL agent
            ################################################
            agent_nb_steps = self.hp.agent_train_steps if not do_warmup else self.hp.warmup_agent_steps
            if do_warmup:
                LOGGER.info(
                    "Warming up agent for {} steps.".format(agent_nb_steps))

            self.agent_trainer.rl_train(
                alg,
                env,
                valid_env,
                max_ep=agent_nb_steps,
                starting_ep=current_ep,
                save_model_dir=self.model_dir,
                test_before_training=do_warmup or i == 0)

            self.end_finetuning_epoch(i, do_warmup)

            if do_warmup:
                current_ep += self.hp.warmup_agent_steps
            else:
                # Backup the model after each loop of the RLHF loop.
                # This is very time consuming.
                self.backuper.backup(step=i) 

                current_ep += self.hp.agent_train_steps
                i += 1
            do_warmup = False

    def _add_streamlines_to_dataset(self, iter_num: int):
        """
        Add new streamlines to the dataset from generated tractograms.
        """
        total_added = 0
        with tqdm(total=self.hp.nb_new_streamlines_per_iter,
                        desc="Adding new streamlines to the dataset",
                        mininterval=5.0) as sub_pbar:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Those will hold the streamlines we are collecting
                # to add to the dataset once we have enough.
                # sft_valid = None
                # sft_invalid = None

                print("Creating tractograms in tmp dir: {}".format(tmpdir))

                sfts_to_add = []

                max_nb_of_tries = 6
                nb_tries = 0
                while total_added < self.hp.nb_new_streamlines_per_iter and nb_tries < max_nb_of_tries:
                    with tempfile.TemporaryDirectory(dir=tmpdir) as sub_tmpdir:
                        # Generate a tractogram
                        tractograms_path = os.path.join(sub_tmpdir, "tractograms")
                        if not os.path.exists(tractograms_path):
                            os.makedirs(tractograms_path)
                        LOGGER.info(
                            "Generating tractograms for RLHF training...")
                        root_dir, tractograms, transform_map = \
                            self.generate_and_save_tractograms(
                                self.tracker, self.tracker_env, tractograms_path)

                        # Filter the tractogram
                        filtered_path = os.path.join(sub_tmpdir, "filtered")
                        if not os.path.exists(filtered_path):
                            os.makedirs(filtered_path)

                        LOGGER.info(
                            "Filtering tractograms for RLHF training...")
                        # Need to filter for each filterer and keep the same order.
                        f_valids, f_invalids, subject_ids, requires_transform = self.filter_tractograms(
                            root_dir, tractograms, filtered_path)
                        
                        LOGGER.info(
                            "Combining and transforming the tractograms to the reference...")
                        # Merge the valid and invalid tractograms
                        nb_new_streamlines = 0
                        for valid, invalid, subject_id, requires_transform in zip(f_valids, f_invalids, subject_ids, requires_transform):
                            _valid = valid
                            if isinstance(_valid, str):
                                # It's a path, load the tractogram here.
                                _valid = load_tractogram(
                                    valid, "same", bbox_valid_check=False)

                            _invalid = invalid
                            if isinstance(_invalid, str):
                                # It's a path, load the tractogram here.
                                _invalid = load_tractogram(
                                    invalid, "same", bbox_valid_check=False)
                                
                            # Transform the tractograms to the reference space
                            # This might take some time.
                            if requires_transform:
                                _valid = self._transform_tractogram_to_ref(
                                    self.tracker_env, _valid, subject_id, transform_map)
                                _invalid = self._transform_tractogram_to_ref(
                                    self.tracker_env, _invalid, subject_id, transform_map)
                                
                            # Resample the streamlines so that they are in equal numbers
                            # of valid vs invalid.
                            _valid, _invalid = self.streamline_sampler.sample_streamlines(
                                _valid, _invalid)

                            if len(_valid) > 0 or len(_invalid) > 0:
                                # if sft_valid is None:
                                #     sft_valid = _valid
                                #     sft_invalid = _invalid
                                # else:
                                #     sft_valid += _valid
                                #     sft_invalid += _invalid

                                sfts_to_add.append((_valid, _invalid))
                                nb_new_streamlines += len(_valid) + len(_invalid)

                        total_added += nb_new_streamlines
                        sub_pbar.update(nb_new_streamlines)
                        nb_tries += 1

                if nb_new_streamlines > 0:
                    LOGGER.info(
                        "Adding filtered tractograms to the dataset...")
                    self.dataset_manager.add_tractograms_to_dataset(
                        sfts_to_add)
                else:
                    LOGGER.warning(
                        "No streamlines were added to the dataset.")
            
        # Print dataset stats
        data_stats = self.dataset_manager.fetch_dataset_stats()
        LOGGER.info(
            prettier_dict(data_stats, title="Dataset stats (iter {})".format(
                iter_num)))

    def _transform_tractogram_to_ref(self, env, sft, subject_id, transform_map):

        # Unpack the transform map
        transforms_subject = transform_map[subject_id]
        reference = transforms_subject['reference']
        transformation = transforms_subject['transformation']
        deformation = transforms_subject['deformation']

        new_sft = env.transform_tractogram_to_reference(
            sft,
            reference=reference,
            transformation=transformation,
            deformation=deformation)
        
        return new_sft

    def train_reward(self):
        """
        Train the reward model using the dataset file.
        This reward model should have been trained on full streamlines, which
        means that dense=False and partial=False.
        """
        print(">>> Training reward model <<<")
        dm = StreamlineDataModule(self.dataset_manager.dataset_file_path,
                                  batch_size=self.hp.oracle_batch_size,
                                  num_workers=self.hp.num_workers,
                                  nb_points=self.oracle_reward.nb_points)
        

        dm.setup('test', dense=False, partial=False)
        metrics_before = self.oracle_reward_trainer.test(test_dataloader=dm.test_dataloader())
        print(prettier_metrics(metrics_before, title="Test metrics before fine-tuning"))

        dm.setup('fit', dense=False, partial=False)
        self.oracle_reward_trainer.fit_iter(train_dataloader=dm.train_dataloader(),
                                     val_dataloader=dm.val_dataloader())
        
        # Auto-checkpointing is disabled, we need to save them manually
        self.oracle_reward_trainer.save_model_checkpoint()

        metrics_after = self.oracle_reward_trainer.test(test_dataloader=dm.test_dataloader())
        print(prettier_metrics(metrics_after, title="Test metrics after fine-tuning"))
        print(">>> Finished training reward model step <<<")

    def train_stopping_criterion(self):
        """
        Train the stopping criterion oracle model using the dataset file.
        This stopping criterion model should have been trained on cut
        streamlines, which means that dense=True and partial=False.
        """
        if self.oracle_crit_disabled:
            print(">>> Skipping stopping criterion model training <<<")
            return

        print(">>> Training stopping criterion model <<<")
        dm = StreamlineDataModule(self.dataset_manager.dataset_file_path,
                                  batch_size=self.hp.oracle_batch_size,
                                  num_workers=self.hp.num_workers,
                                  nb_points=self.oracle_crit.nb_points)
        
        # Test the performance of the actual model BEFORE fine-tuning.
        # TO REVISE:
        # To get an accuracy plot, we test the stopping criterion on fully
        # tracked streamlines even though it's supposed to predict on partial
        # streamlines.
        dm.setup('test', dense=False, partial=False)
        metrics_before = self.oracle_crit_trainer.test(test_dataloader=dm.test_dataloader())
        print(prettier_metrics(metrics_before, title="Test metrics before fine-tuning"))

        dm.setup('fit', dense=True, partial=True)
        self.oracle_crit_trainer.fit_iter(train_dataloader=dm.train_dataloader(),
                                     val_dataloader=dm.val_dataloader())
        
        # Auto-checkpointing is disabled, we need to save manually
        self.oracle_crit_trainer.save_model_checkpoint()
        
        # Test the performance of the actual model AFTER fine-tuning.
        metrics_after = self.oracle_crit_trainer.test(test_dataloader=dm.test_dataloader())
        print(prettier_metrics(metrics_after, title="Test metrics after fine-tuning"))
        print(">>> Finished stopping criterion model training <<<")

    def test_reward(self, step, epoch):
        print(">>> Testing reward model <<<")
        dm = StreamlineDataModule(self.dataset_manager.dataset_file_path,
                                  batch_size=self.hp.oracle_batch_size,
                                  num_workers=self.hp.num_workers,
                                  nb_points=self.oracle_reward.nb_points)
        

        dm.setup('test', dense=False, partial=False)
        metrics_after = self.oracle_reward_trainer.test(test_dataloader=dm.test_dataloader(), step=step, epoch=epoch)
        print(prettier_metrics(metrics_after, title="Test metrics after fine-tuning"))
        print(">>> Finished testing reward model step <<<")

    def test_stopping_criterion(self, step, epoch):
        print(">>> Testing stopping criterion model <<<")
        dm = StreamlineDataModule(self.dataset_manager.dataset_file_path,
                                  batch_size=self.hp.oracle_batch_size,
                                  num_workers=self.hp.num_workers,
                                  nb_points=self.oracle_crit.nb_points)
        dm.setup('test', dense=False, partial=False)
        metrics_before = self.oracle_crit_trainer.test(test_dataloader=dm.test_dataloader(), step=step, epoch=epoch)
        print(prettier_metrics(metrics_before, title="Test metrics before fine-tuning (step: {}, epoch: {})".format(step, epoch)))
        print(">>> Finished testing stopping criterion model <<<")

    def generate_and_save_tractograms(self, tracker: Tracker, env: BaseEnv,
                                      save_dir: str,
                                      max_nb_subjects: int = 5):
        """
        Most of the flows requires a single directory containing all the
        tractograms to filter in the following structure of files:
        
        root
        ├── <subject_id_1>
        │   └── <tractogram_1>.trk
        ├── <subject_id_2>
        │   └── <tractogram_2>.trk
        └── ...

        This function tracks and organize the tractograms in this structure.
        """

        nb_subjects = min(len(env.dataset), max_nb_subjects)

        root_path = save_dir
        tractograms_path = []

        # Since the environment only loads one subject at a time, we need to
        # keep track of the following information to be able to register
        # the filtered tractograms back to the reference space, if needed.
        transform_map = {}

        # Track on several subjects if there are.
        for _ in range(nb_subjects):
            # Build the path to save the tractogram as described above.
            subject_save_path = os.path.join(save_dir, env.subject_id)
            os.makedirs(subject_save_path, exist_ok=True)

            # Track on the current subject.
            LOGGER.info("Tracking on subject: {}".format(env.subject_id))
            tractogram, _ = tracker.track_and_validate(env, enable_pbar=True) # TODO: Change to only track(), no need to validate.
            sft = self.convert_to_rasmm_sft(tractogram, env.affine_vox2rasmm, env.reference, discard_dps=True)

            # If we're using extractor_flow, we need to transform the tractogram
            # to MNI space.
            if self.hp.extractor_validator and env.can_transform_to_mni:
                LOGGER.info("Transforming tractogram to MNI space.")
                sft, transform_map_subj = env.transform_tractogram_to_mni(sft)
                transform_map.update(transform_map_subj)
            elif self.hp.extractor_validator or self.hp.verifyber_validator:
                # Add the T1w file to the in_directory
                LOGGER.info("Copying T1w file to the subject's directory.")
                t1_filename = f"{env.subject_id}_t1.nii.gz"
                env.save_anat_to(subject_save_path, t1_filename)
            
            # If we're using RBX, we need to save the FA.
            if self.hp.rbx_validator:
                LOGGER.info("Saving FA map to the subject's directory.")
                fa_filename = f"{env.subject_id}__fa.nii.gz"
                env.save_fa_to(subject_save_path, fa_filename)

            filename = self.save_sft(sft, env.subject_id,
                                     subject_save_path, extension='trk')
            
            tractograms_path.append(os.path.join(subject_save_path, filename))
            
            if len(env.dataset) > 1:
                self.tracker_env.load_subject() # Load the next subject to track on.

        return root_path, tractograms_path, transform_map

    def filter_tractograms(self, in_directory: str, tractograms: str,
                           out_dir: str):
        """
        Filter tractograms using the first filterer in the list.
        """

        valid_tractograms = []
        invalid_tractograms = []
        subject_ids = []
        requires_transform = []
        for filterer in self.filterers:
            valids, invalids, subject_ids = filterer(in_directory, tractograms, out_dir)
            valid_tractograms.extend(valids)
            invalid_tractograms.extend(invalids)
            subject_ids.extend(subject_ids)
            requires_transform.extend([not filterer.ends_up_in_orig_space] * len(tractograms))
    
        return valid_tractograms, invalid_tractograms, subject_ids, requires_transform

    def save_hyperparameters(self):
        super().save_hyperparameters(filename='rlhf_hyperparameters.json')

    def start_finetuning_epoch(self, epoch: int, warmup: bool = False):
        if warmup:
            print("==================================================")    
            print("=========== Starting WARMUP of {} steps =========".format(self.hp.warmup_agent_steps))
        else:
            print("==================================================")
            print("======= Starting RLHF finetuning epoch {}/{} =======".format(epoch+1, self.hp.max_ep))

    def end_finetuning_epoch(self, epoch: int, warmup: bool = False):
        if warmup:
            print("=========== Finished WARMUP of {} steps =========".format(self.hp.warmup_agent_steps))
            print("==================================================")
        else:
            print("======= Finished RLHF finetuning epoch {}/{} =======".format(epoch+1, self.hp.max_ep))
            print("==================================================")


def add_rlhf_training_args(parser: argparse.ArgumentParser):
    rlhf_group = parser.add_argument_group("RLHF Training Arguments")
    rlhf_group.add_argument('--alg', type=str, required=True,
                            help='The algorithm to use for training the agent.\n'
                            'Possible values are: SACAuto, PPO.')
    rlhf_group.add_argument('--num_workers', type=int, default=10,
                            help='Number of workers to use for data loading.')
    rlhf_group.add_argument("--rlhf_inter_npv", type=int, default=None,
                            help="Number of seeds to use when generating intermediate tractograms\n"
                            "for the RLHF training pipeline. If None, the general npv will be used.")
    rlhf_group.add_argument("--nb_new_streamlines_per_iter", type=int, default=500000,
                            help="Number of new streamlines to add to the dataset at each iteration.")
    rlhf_group.add_argument("--max_dataset_size", type=int, default=5000000,
                            help="Maximum number of streamlines to keep in the dataset.")
    rlhf_group.add_argument("--warmup_agent_steps", type=int,
                            help="Minimum number of steps to warm up the agent before starting the training of the oracle")

    # Agent training RLHF arguments
    agent_group = parser.add_argument_group("Agent Training Arguments")
    agent_group.add_argument('--agent_train_steps', type=int, required=True,
                             help='Number of steps to fine-tune the agent during RLHF training.')
    agent_group.add_argument('--agent_checkpoint', type=str,
                                  help='Path to the agent checkpoint FILE to load.')

    # Oracle training RLHF arguments
    oracle_group = parser.add_argument_group("Oracle Training Arguments")
    oracle_group.add_argument('--oracle_lr', type=float,
                              help='Learning rate to use for training the oracle.\n'
                              'If not set, the lr stored in the checkpoint will be used.')
    oracle_group.add_argument('--first_oracle_train_steps', type=int, 
                              help='Number of steps to train the oracle on the first training sequence, this is kinda of a warm-up to be able to quickly align the oracles to the dataset.')
    oracle_group.add_argument('--oracle_train_steps', type=int, required=True,
                              help='Number of steps to fine-tune the oracle during RLHF training.')
    oracle_group.add_argument('--oracle_batch_size', type=int, default=1408,
                              help='Batch size to use for training the oracle.')
    oracle_group.add_argument("--dataset_to_augment", type=str, help="Path to the dataset to augment.\n"
                              "If this is not set, the dataset will be created from scratch entirely by the\n"
                              "current learning agent.")
    oracle_group.add_argument("--disable_oracle_training", action="store_true",
                              help="Disable oracle training during RLHF training.\n")
    return parser


def get_trainer_cls_and_args(alg_name: str):
    trainer_map = {
        'SACAuto': SACAutoTraining,
        'CrossQ': CrossQTraining,
    }

    if alg_name not in trainer_map:
        raise ValueError(f'Invalid algorithm name: {alg_name}')

    return trainer_map[alg_name]


def get_algorithm_cls(alg_name: str):
    algorithm_map = {
        'SACAuto': SACAuto,
        'CrossQ': CrossQ,
    }

    if alg_name not in algorithm_map:
        raise ValueError(f'Invalid algorithm name: {alg_name}')

    return algorithm_map[alg_name]


def parse_args():
    """ Train an agent whilst training oracles in the loop. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_training_args(parser)
    add_sac_auto_args(parser)
    add_rlhf_training_args(parser)
    add_oracle_train_args(parser)
    add_logging_args(parser)

    arguments = parser.parse_args()
    return arguments


def main():
    args = parse_args()
    setup_logging(args)

    trainer_cls = get_trainer_cls_and_args(args.alg)

    # Create and run the experiment
    rlhf_experiment = RlhfTraining(
        vars(args),
        trainer_cls
    )
    rlhf_experiment.run()


if __name__ == "__main__":
    main()
